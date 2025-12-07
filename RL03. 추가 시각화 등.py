


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.rc('font', family='Malgun Gothic')   # 윈도우 기본 폰트
plt.rc('axes', unicode_minus=False)      # 마이너스 깨짐 방지

# =========================================
# 1) 사용자 로그 기반 결과 리스트 정의
# =========================================

# DQN avg_reward(last50) 기록 (Episode 50~500, 50 단위)
dqn_rewards = [
    552.8080, 596.0743, 626.0727, 596.8394, 627.1769,
    630.9944, 620.0765, 668.2549, 653.2164, 616.0998
]
dqn_eps = [
    0.984, 0.969, 0.954, 0.939, 0.924,
    0.910, 0.895, 0.881, 0.868, 0.854
]

# SARSA avg_reward(last50) 기록
sarsa_rewards = [
    545.2129, 524.5981, 537.3134, 564.4770, 576.5645,
    576.0280, 614.1795, 593.6518, 621.0074, 657.5526
]
sarsa_eps = [
    0.984, 0.969, 0.954, 0.939, 0.924,
    0.910, 0.895, 0.881, 0.868, 0.854
]

# 평가 정책 평균 Reward 및 95% CI 설정
policy_names = ['Random', 'Conservative 70%', 'DQN', 'SARSA']
policy_means = [527.3419, 1435.0566, 1428.3290, 1435.0566]
policy_ci = [33.8400, 35.1438, 35.4365, 35.1438]

# =========================================
# 2) DQN vs SARSA 학습 곡선
# =========================================
plt.figure(figsize=(8,5))
episodes = np.arange(50, 501, 50)
plt.plot(episodes, dqn_rewards, marker='o', label='DQN')
plt.plot(episodes, sarsa_rewards, marker='o', label='SARSA')
plt.xlabel('Episode')
plt.ylabel('Avg Reward (Recent 50)')
plt.title('Training Progress: DQN vs SARSA')
plt.legend()
plt.grid(True)
plt.show()

# =========================================
# 3) 정책별 평균 Reward + 95% CI 그래프
# =========================================
plt.figure(figsize=(7,5))
plt.bar(policy_names, policy_means, yerr=policy_ci, capsize=7)
plt.ylabel('Reward (Mean ± 95% CI)')
plt.title('Policy Performance Comparison')
plt.grid(axis='y')
plt.show()

# =========================================
# 4) 정책별 Reward 분포 Boxplot (정규분포 가정 샘플링)
# ※ 실제 샘플 90개를 가진 효과를 시각적으로 표현
# =========================================
np.random.seed(42)
random_rewards = np.random.normal(527.3419, 163.7932, 90)
cons_rewards = np.random.normal(1435.0566, 170.1038, 90)
dqn_samples   = np.random.normal(1428.3290, 171.5204, 90)
sarsa_samples = np.random.normal(1435.0566, 170.1038, 90)

plt.figure(figsize=(8,5))
sns.boxplot(data=[random_rewards, cons_rewards, dqn_samples, sarsa_samples])
plt.xticks([0,1,2,3], policy_names)
plt.ylabel('Reward Distribution')
plt.title('Reward Distribution by Policy')
plt.grid(axis='y')
plt.show()
