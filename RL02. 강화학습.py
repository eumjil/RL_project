import numpy as np
import pandas as pd
import random
from collections import deque
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import shap   # pip install shap

plt.rc('font', family='Malgun Gothic')   # 윈도우 기본 폰트
plt.rc('axes', unicode_minus=False)      # 마이너스 깨짐 방지

# =========================
# 0. 데이터 로드 & 상태 피처 설정
# =========================

DATA_PATH = r"D:\EJ\RL\auction_analysis_ready.csv"

df = pd.read_csv(DATA_PATH)

STATE_FEATURES = [
    # Log 변환 변수
    "감정가_log",
    "건물면적_log",
    "실거래가_log",

    # 비율/지표 변수
    "최저가율",
    "유찰횟수_adj",

    # 시장/지역 변수
    "월실거래건수",
    "실거래상대지수_local",
    "실거래상대지수_global",

    # 건물 특성
    "건물나이",

    # 범주형(one-hot)
    "면적_소형", "면적_중형", "면적_대형",
    "층_low", "층_mid", "층_high",
    "구클_0", "구클_1", "구클_2", "구클_3", "구클_4"
]

# 강화학습 환경에서 꼭 필요한 컬럼들만 일단 결측치 제거
required_cols = STATE_FEATURES + ["감정가", "실거래가"]
df = df.dropna(subset=required_cols).reset_index(drop=True)

print("사용 가능한 샘플 수:", len(df))


# =========================
# 1. 예산·대출 제약을 고려한 경매 RL 환경
# =========================

class PortfolioAuctionEnv:
    """
    예산·대출 제약을 고려한 경매 입찰 RL 환경

    - 한 에피소드에서 여러 개의 경매 물건을 순차적으로 본다고 가정
    - 상태(state) = [현금비율, 에피소드 진행도, (기존 물건 특성 feature 벡터)]
    - 행동(action) = 0~4  (0: 입찰 안 함, 1~4: 입찰 강도)
    - 보상(reward) = 이 물건 거래에서의 이익 (실거래가 - 입찰가)
                     (단순화를 위해 거래세/이자 등은 일단 제외)
    - LTV 60% 기준 자기자본이 부족하면 큰 패널티
    """

    def __init__(
        self,
        df,
        feature_cols,
        appraisal_col: str = "감정가",
        market_price_col: str = "실거래가",
        initial_cash: float = 100_000.0,   # 10억
        ltv: float = 0.6,
        max_items_per_episode: int = 50,
    ):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.appraisal_col = appraisal_col
        self.market_price_col = market_price_col

        self.initial_cash = float(initial_cash)
        self.ltv = float(ltv)
        self.max_items_per_episode = min(max_items_per_episode, len(self.df))

        # action → 입찰 비율 매핑 (감정가 대비)
        # 0: 입찰 안 함, 1~4: 점점 공격적
        self.action_bid_ratios = np.array([0.0, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)

        # 내부 상태
        self.cash = None
        self.t = None
        self.indices = None
        self.current_idx = None

    def reset(self):
        """
        에피소드 시작:
        - 현금 초기화
        - 사용할 물건 인덱스 샘플링
        - 첫 물건으로 상태 구성
        """
        self.cash = self.initial_cash
        self.t = 0

        # 한 에피소드에서 사용할 물건들 셔플
        self.indices = np.random.choice(
            len(self.df),
            size=self.max_items_per_episode,
            replace=False
        )
        self.current_idx = int(self.indices[self.t])

        state = self._make_state()
        return state

    def _make_state(self):
        """
        상태 벡터 구성:
        [현금비율, 진행도, STATE_FEATURES ...]
        """
        row = self.df.iloc[self.current_idx]
        item_features = row[self.feature_cols].values.astype(np.float32)

        # 현금 비율 (0~1 근처로 정규화)
        cash_ratio = self.cash / self.initial_cash
        # 너무 커지면 클리핑
        cash_ratio = float(np.clip(cash_ratio, 0.0, 2.0))

        # 에피소드 진행도 (0~1)
        progress = self.t / self.max_items_per_episode

        state = np.concatenate([
            np.array([cash_ratio, progress], dtype=np.float32),
            item_features
        ])
        return state

    def step(self, action: int):
        """
        action: 0~4
          - 0: 입찰 안 함 → reward = 0, 현금 변화 없음
          - 1~4: 감정가 * 비율로 입찰가 결정

        보상: 이 물건에서의 이익 (실거래가 - 입찰가)
             자기자본 부족이면 -5천만원 패널티
        """
        row = self.df.iloc[self.current_idx]
        base_price = float(row[self.appraisal_col])      # 감정가
        market_price = float(row[self.market_price_col]) # 실거래가 (또는 낙찰가)

        bid_ratio = float(self.action_bid_ratios[action])

        profit = 0.0

        if action == 0:
            # 입찰 안 함
            profit = 0.0
        else:
            bid_price = base_price * bid_ratio

            # LTV 60% 기준 자기자본 필요액
            equity_needed = bid_price * (1.0 - self.ltv)

            if equity_needed > self.cash:
                # 현금이 부족한데 무리 입찰 → 큰 패널티
                profit = -5_000.0   # -5천만 
            else:
                # 단순화: 바로 실거래가에 되판다고 가정
                profit = market_price - bid_price
                # 현금 업데이트: 이익을 바로 반영
                self.cash += profit

        # 현금이 음수가 되지 않도록
        self.cash = float(max(self.cash, 0.0))

        # 보상 스케일이 너무 크면 학습이 불안정할 수 있으니
        # 1e7(천만)으로 나눠서 스케일 다운 
        reward = profit / 1_000.0

        # 다음 step으로 진행
        self.t += 1
        done = False
        if self.t >= self.max_items_per_episode:
            done = True
        elif self.cash < 1_000:  # 현금이 거의 없으면 조기 종료 (1천만 미만)
            done = True

        if not done:
            self.current_idx = int(self.indices[self.t])
            next_state = self._make_state()
        else:
            # done이어도 DQN에서 shape 맞추려면 dummy state 필요
            next_state = np.zeros_like(self._make_state(), dtype=np.float32)

        info = {
            "cash": self.cash,
            "t": self.t,
            "raw_profit": profit
        }
        return next_state, reward, done, info


# =========================
# 2. DQN 정의 
# =========================

class DQN(nn.Module):
    """간단한 2층 MLP 기반 Q 네트워크"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """경험 재생 버퍼"""
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def compute_dqn_loss(batch, q_net, target_net, optimizer, gamma=0.99, device="cpu"):
    states, actions, rewards, next_states, dones = batch

    states = torch.tensor(states, device=device)
    actions = torch.tensor(actions, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, device=device).unsqueeze(1)
    next_states = torch.tensor(next_states, device=device)
    dones = torch.tensor(dones, device=device).unsqueeze(1)

    # Q(s,a)
    q_values = q_net(states).gather(1, actions)

    # target = r + gamma * max_a' Q_target(s',a') (단, done이면 r만)
    with torch.no_grad():
        next_q = target_net(next_states).max(1, keepdim=True)[0]
        target = rewards + gamma * next_q * (1 - dones)

    loss = nn.MSELoss()(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def moving_average(x, window=50):
    x = np.array(x, dtype=np.float32)
    if len(x) < window:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / float(window)
    pad = np.full(window - 1, ma[0], dtype=np.float32)
    return np.concatenate([pad, ma])


def train_dqn(
    env,
    num_episodes=500,        
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=3000,
    target_update_freq=500,
    buffer_capacity=50_000,
    device="cpu"
):
    global state_dim, n_actions

    q_net = DQN(state_dim, n_actions).to(device)
    target_net = DQN(state_dim, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity)

    epsilon = epsilon_start
    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # epsilon-greedy
            if random.random() < epsilon:
                action = random.randint(0, n_actions - 1)
            else:
                with torch.no_grad():
                    s = torch.tensor(state, device=device).unsqueeze(0)
                    q_vals = q_net(s)
                    action = int(torch.argmax(q_vals, dim=1).item())

            next_state, reward, done, info = env.step(action)
            buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                compute_dqn_loss(batch, q_net, target_net, optimizer, gamma, device)

        # epsilon decay
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-episode / epsilon_decay)

        # target 네트워크 동기화
        if episode % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        episode_rewards.append(total_reward)

        if episode % 50 == 0:
            print(f"[DQN] Episode {episode} | avg_reward(last50) = {np.mean(episode_rewards[-50:]):.4f} | epsilon = {epsilon:.3f}")

    return q_net, episode_rewards





# =========================
# A. 공통: epsilon-greedy, 정책 평가 함수
# =========================

def epsilon_greedy_from_q(q_values, epsilon: float):
    """
    q_values: shape (n_actions,)
    """
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        return int(np.argmax(q_values))


def evaluate_policy(
    env,
    policy_fn,
    num_episodes=50,
    seed_list=(42, 11, 99),
    device="cpu",
    desc="",
):
    """
    주어진 policy_fn(state) -> action 으로 여러 에피소드/시드 돌려서
    평균 Return, 표준편차, 95% CI 계산
    """
    all_returns = []

    for seed in seed_list:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        for ep in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = policy_fn(state)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                state = next_state

            all_returns.append(total_reward)

    all_returns = np.array(all_returns, dtype=np.float32)
    mean = float(all_returns.mean())
    std = float(all_returns.std(ddof=1))
    n = len(all_returns)
    ci = 1.96 * std / math.sqrt(n)

    print(f"=== {desc} ===")
    print(f"n = {n}, mean = {mean:.4f}, std = {std:.4f}, 95% CI = ±{ci:.4f}")
    print()

    return mean, std, ci
  


# =========================
# B. Baseline 정책 2개
# =========================

def random_policy(state):
    """완전 랜덤: 0~4 중 균등한 확률로 선택"""
    return random.randint(0, n_actions - 1)


def conservative70_policy(state):
    """
    항상 Action=1 (감정가의 70% 입찰)
    - LTV 60% 기준 자기자본 부담이 가장 적은 보수적 정책
    """
    return 1




# =========================
# C. SARSA (NN 기반) 구현
# =========================

class SarsaNet(nn.Module):
    """
    DQN과 거의 같은 구조의 Q 네트워크지만,
    업데이트 방식은 SARSA(on-policy)로 수행
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


def train_sarsa(
    env,
    num_episodes=500,
    gamma=0.99,
    lr=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=3000,
    device="cpu"
):
    """
    Online SARSA(0) 학습
    - ReplayBuffer / TargetNet 없이, on-policy 업데이트
    """
    sarsa_net = SarsaNet(state_dim, n_actions).to(device)
    optimizer = optim.Adam(sarsa_net.parameters(), lr=lr)
    mse = nn.MSELoss()

    epsilon = epsilon_start
    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        state_t = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            q_vals = sarsa_net(state_t).cpu().numpy()[0]
        action = epsilon_greedy_from_q(q_vals, epsilon)

        done = False
        total_reward = 0.0

        while not done:
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            next_state_t = torch.tensor(next_state, device=device, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                next_q_vals = sarsa_net(next_state_t).cpu().numpy()[0]

            if not done:
                next_action = epsilon_greedy_from_q(next_q_vals, epsilon)
            else:
                next_action = None

            # Q(s,a)
            q_values = sarsa_net(state_t)
            q_sa = q_values[0, action]

            # target = r + gamma * Q(s',a')  (done이면 r만)
            target = torch.tensor(reward, device=device, dtype=torch.float32)  # ← [] 스칼라 텐서
            if not done:
                next_q_values = sarsa_net(next_state_t)
                q_s_next_a_next = next_q_values[0, next_action]
                target = target + gamma * q_s_next_a_next.detach()

            loss = mse(q_sa, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 다음 스텝 준비
            state_t = next_state_t
            action = next_action if next_action is not None else action

        # epsilon decay (episode 단위)
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-episode / epsilon_decay)

        episode_rewards.append(total_reward)

        if episode % 50 == 0:
            print(f"[SARSA] Episode {episode} | avg_reward(last50) = {np.mean(episode_rewards[-50:]):.4f} | epsilon = {epsilon:.3f}")

    return sarsa_net, episode_rewards




# =========================
# D. 학습된 DQN, SARSA 정책 (greedy)
# =========================

def dqn_greedy_policy_factory(q_net, device="cpu"):
    def policy(state):
        with torch.no_grad():
            s = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
            q_vals = q_net(s)
            action = int(torch.argmax(q_vals, dim=1).item())
        return action
    return policy


def sarsa_greedy_policy_factory(sarsa_net, device="cpu"):
    def policy(state):
        with torch.no_grad():
            s = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
            q_vals = sarsa_net(s)
            action = int(torch.argmax(q_vals, dim=1).item())
        return action
    return policy






# =========================
# 3. 실행
# =========================

if __name__ == "__main__":
    # 재현성을 위해 seed 고정 (선택)
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    # 환경 생성
    env = PortfolioAuctionEnv(
        df=df,
        feature_cols=STATE_FEATURES,
        appraisal_col="감정가",
        market_price_col="실거래가",      
        initial_cash=100_000.0,     # 10억
        ltv=0.6,
        max_items_per_episode=50,         # 에피소드당 물건 개수
    )

    # 상태/행동 차원 설정
    sample_state = env.reset()
    state_dim = sample_state.shape[0]
    n_actions = 5   # 0~4

    print("state_dim:", state_dim, "| n_actions:", n_actions)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # DQN 학습
    q_net, episode_rewards = train_dqn(
        env=env,
        num_episodes=500,    
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=3000,
        target_update_freq=200,
        buffer_capacity=50_000,
        device=device,
    )
  
    # 학습 곡선 시각화
    ma = moving_average(episode_rewards, window=20)
    plt.figure(figsize=(10, 5))
    plt.plot(ma)
    plt.xlabel("Episode")
    plt.ylabel("Return (moving avg, window=20)")
    plt.title("DQN on Budget/LTV-Constrained Auction Environment")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

  

    # ... (위에서 env, state_dim, n_actions, device 설정)
    # ... (위에서 DQN 학습 완료: q_net, episode_rewards_dqn = train_dqn(...))

    # 1) SARSA 학습
    print("\n=== SARSA 학습 시작 ===")
    sarsa_net, episode_rewards_sarsa = train_sarsa(
        env=env,
        num_episodes=500,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=3000,
        device=device,
    )

    # 2) 정책 함수 생성
    dqn_policy = dqn_greedy_policy_factory(q_net, device=device)
    sarsa_policy = sarsa_greedy_policy_factory(sarsa_net, device=device)

    # 3) Baseline + RL 비교 평가
    print("\n=== 정책별 성능 비교 (여러 시드, 95% CI) ===")
    seed_list = (42, 11, 99)
    num_eval_episodes = 30   # 시드당 에피소드 수 (총 90 에피소드)

    evaluate_policy(
        env, random_policy,
        num_episodes=num_eval_episodes,
        seed_list=seed_list,
        device=device,
        desc="Random Policy",
    )

    evaluate_policy(
        env, conservative70_policy,
        num_episodes=num_eval_episodes,
        seed_list=seed_list,
        device=device,
        desc="Conservative 70% Policy",
    )

    evaluate_policy(
        env, dqn_policy,
        num_episodes=num_eval_episodes,
        seed_list=seed_list,
        device=device,
        desc="DQN (greedy)",
    )

    evaluate_policy(
        env, sarsa_policy,
        num_episodes=num_eval_episodes,
        seed_list=seed_list,
        device=device,
        desc="SARSA (greedy)",
    )





# ============================================================
# 4. 해석용 분석: 정책 민감도(Policy Sensitivity) + SHAP 중요도
#    - 최종 선택 모델인 SARSA 기준으로만 수행
# ============================================================

# 4-0. 상태 이름 정의 (env.state 구성 순서와 동일)
STATE_NAMES = ["cash_ratio", "episode_progress"] + STATE_FEATURES


# 4-1. SARSA 네트워크에서 greedy action 뽑는 함수
def greedy_action(model, state_vec, device="cpu"):
    """
    state_vec: shape (state_dim,) numpy array
    """
    model.eval()
    with torch.no_grad():
        s = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
        q_vals = model(s)  # shape: (1, n_actions)
        action = int(torch.argmax(q_vals, dim=1).item())
    return action


# 4-2. 기준 상태(baseline state) 만들기
#  - 물건 특성은 df의 median, cash_ratio=1.0, progress=0.5 로 설정
median_features = df[STATE_FEATURES].median().values.astype(np.float32)
baseline_state = np.concatenate([
    np.array([1.0, 0.5], dtype=np.float32),  # cash_ratio, episode_progress
    median_features
])


# 4-3. 상태 변수 변화에 따른 행동 변화 시각화 (Policy Sensitivity Plot)
import matplotlib.pyplot as plt

# 민감도 분석에 사용할 연속형 변수들
SENSITIVITY_FEATURES = [
    "cash_ratio",
    "최저가율",
    "유찰횟수_adj",
    "실거래상대지수_local",
]

def plot_policy_sensitivity_sarsa(sarsa_net, device="cpu"):
    fig, axes = plt.subplots(1, len(SENSITIVITY_FEATURES),
                             figsize=(4*len(SENSITIVITY_FEATURES), 4),
                             sharey=True)

    for ax, feat in zip(axes, SENSITIVITY_FEATURES):
        if feat not in STATE_NAMES:
            ax.set_title(f"{feat}\n(STATE_NAMES에 없음)")
            continue

        idx = STATE_NAMES.index(feat)

        # 각 변수별 값 범위 설정
        if feat == "cash_ratio":
            values = np.linspace(0.2, 1.8, 30)  # 현금비율 0.2~1.8
        elif feat == "episode_progress":
            values = np.linspace(0.0, 1.0, 30)
        else:
            # df에서 5~95 분위수 범위 사용
            v_min = df[feat].quantile(0.05)
            v_max = df[feat].quantile(0.95)
            values = np.linspace(v_min, v_max, 30)

        actions = []
        for v in values:
            s = baseline_state.copy()
            s[idx] = v
            a = greedy_action(sarsa_net, s, device)
            actions.append(a)

        ax.plot(values, actions, marker="o", linestyle="-")
        ax.set_title(feat)
        ax.set_xlabel("value")
        ax.set_ylabel("greedy action index")
        ax.grid(alpha=0.3)

    plt.suptitle("Policy Sensitivity (SARSA, greedy)")
    plt.tight_layout()
    plt.show()


# 4-4. SHAP 분석 위해 상태 샘플링 함수 정의
def sample_states(env, policy_fn, num_episodes=10):
    """
    환경에서 여러 에피소드를 돌리며 state들을 수집
    """
    states = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            states.append(state.copy())
            action = policy_fn(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
    return np.array(states, dtype=np.float32)


# 4-5. SARSA Q-value 예측 함수 (SHAP용)
def sarsa_predict(states_np):
    """
    SHAP KernelExplainer에서 사용할 예측 함수
    - 입력: (N, state_dim)
    - 출력: (N,)  각 state에서 "최적 action의 Q값"
    """
    sarsa_net.eval()
    with torch.no_grad():
        s = torch.tensor(states_np, dtype=torch.float32, device=device)
        q_vals = sarsa_net(s)              # (N, n_actions)
        best_q, _ = torch.max(q_vals, dim=1)
    return best_q.cpu().numpy()


# 4-6. SHAP 기반 Feature Importance 계산 & 시각화
def shap_feature_importance_sarsa(env, sarsa_net, sarsa_policy, device="cpu"):
    # 1) background: 랜덤 정책으로 생성한 state 일부
    bg_states_all = sample_states(env, random_policy, num_episodes=8)
    if bg_states_all.shape[0] > 200:
        background = bg_states_all[:200]
    else:
        background = bg_states_all

    # 2) 평가용 state: SARSA 정책으로 생성한 state들
    eval_states_all = sample_states(env, sarsa_policy, num_episodes=12)
    if eval_states_all.shape[0] > 300:
        eval_states = eval_states_all[:300]
    else:
        eval_states = eval_states_all

    print("background states:", background.shape)
    print("eval states:", eval_states.shape)

    # 3) SHAP 계산 
    explainer = shap.KernelExplainer(sarsa_predict, background)
    shap_values = explainer.shap_values(eval_states, nsamples=100)

    shap_values = np.array(shap_values)  # (N_eval, state_dim)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # 중요도 순서대로 정렬
    idx_sorted = np.argsort(mean_abs_shap)[::-1]
    sorted_names = [STATE_NAMES[i] for i in idx_sorted]
    sorted_importance = mean_abs_shap[idx_sorted]

    # 상위 15개 변수만 시각화
    top_k = min(15, len(sorted_names))
    top_names = sorted_names[:top_k][::-1]
    top_imps  = sorted_importance[:top_k][::-1]

    plt.figure(figsize=(7, 6))
    plt.barh(top_names, top_imps)
    plt.xlabel("Mean |SHAP value| (SARSA best-Q)")
    plt.title("Feature Importance (SARSA Q-value)")
    plt.tight_layout()
    plt.show()

    # 텍스트로도 상위 몇 개 출력
    print("\n=== SHAP 상위 중요 변수 (SARSA) ===")
    for name, imp in zip(sorted_names[:10], sorted_importance[:10]):
        print(f"{name:20s} : {imp:.6f}")


# 4-7. 실제 실행 (SARSA 기준)
if __name__ == "__main__":
    # 위에서 이미 env, sarsa_net, sarsa_policy, device 정의되어 있음

    print("\n=== [분석] SARSA Policy 민감도 시각화 ===")
    plot_policy_sensitivity_sarsa(sarsa_net, device=device)

    print("\n=== [분석] SARSA SHAP Feature Importance ===")
    shap_feature_importance_sarsa(env, sarsa_net, sarsa_policy, device=device)



def plot_q_sensitivity_sarsa(sarsa_net, device="cpu"):
    fig, axes = plt.subplots(1, len(SENSITIVITY_FEATURES),
                             figsize=(5*len(SENSITIVITY_FEATURES), 4),
                             sharey=False)

    for ax, feat in zip(axes, SENSITIVITY_FEATURES):
        if feat not in STATE_NAMES:
            ax.set_title(f"{feat}\n(STATE_NAMES에 없음)")
            continue

        idx = STATE_NAMES.index(feat)

        if feat == "cash_ratio":
            values = np.linspace(0.2, 1.8, 30)
        elif feat == "episode_progress":
            values = np.linspace(0.0, 1.0, 30)
        else:
            v_min = df[feat].quantile(0.05)
            v_max = df[feat].quantile(0.95)
            values = np.linspace(v_min, v_max, 30)

        q_all = {a: [] for a in range(n_actions)}

        for v in values:
            s = baseline_state.copy()
            s[idx] = v
            with torch.no_grad():
                t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                q_vals = sarsa_net(t)[0].cpu().numpy()  # (n_actions,)

            for a in range(n_actions):
                q_all[a].append(q_vals[a])

        for a in range(n_actions):
            ax.plot(values, q_all[a], label=f"action {a}")

        ax.set_title(feat)
        ax.set_xlabel("value")
        ax.set_ylabel("Q(s,a)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle("Q-value Sensitivity (SARSA)")
    plt.tight_layout()
    plt.show()





## Q 민감도

def plot_q_sensitivity_sarsa(sarsa_net, device="cpu"):
    fig, axes = plt.subplots(1, len(SENSITIVITY_FEATURES),
                             figsize=(5*len(SENSITIVITY_FEATURES), 4),
                             sharey=False)

    for ax, feat in zip(axes, SENSITIVITY_FEATURES):
        if feat not in STATE_NAMES:
            ax.set_title(f"{feat}\n(STATE_NAMES에 없음)")
            continue

        idx = STATE_NAMES.index(feat)

        if feat == "cash_ratio":
            values = np.linspace(0.2, 1.8, 30)
        elif feat == "episode_progress":
            values = np.linspace(0.0, 1.0, 30)
        else:
            v_min = df[feat].quantile(0.05)
            v_max = df[feat].quantile(0.95)
            values = np.linspace(v_min, v_max, 30)

        q_all = {a: [] for a in range(n_actions)}

        for v in values:
            s = baseline_state.copy()
            s[idx] = v
            with torch.no_grad():
                t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                q_vals = sarsa_net(t)[0].cpu().numpy()  # (n_actions,)

            for a in range(n_actions):
                q_all[a].append(q_vals[a])

        for a in range(n_actions):
            ax.plot(values, q_all[a], label=f"action {a}")

        ax.set_title(feat)
        ax.set_xlabel("value")
        ax.set_ylabel("Q(s,a)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle("Q-value Sensitivity (SARSA)")
    plt.tight_layout()
    plt.show()




print("\n=== [분석] SARSA Q-value 민감도 시각화 ===")
plot_q_sensitivity_sarsa(sarsa_net, device=device)

