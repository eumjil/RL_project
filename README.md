# RL_project

# 강화학습 기반 부동산 경매 포트폴리오 입찰 전략 연구  
Reinforcement Learning for Real-Estate Auction Portfolio Bidding Strategy

## 1. 프로젝트 개요 (Overview)

이 프로젝트는 강화학습을 이용하여 부동산(서울시 아파트) 경매 입찰 전략을 최적화하는 분석을 수행한 것입니다.

감정가, 최저가율, 거래지표 등 경매 물건들의 특성과 예산·대출(LTV) 제약 조건을 반영한 RL 환경을 정의하고,
두 가지 RL 알고리즘을 적용하여 수익 극대화 전략을 학습합니다.

- 환경(Environment): 경매 물건의 상태(state)와 입찰 비율(action)에 따른 수익(reward)을 정의
- 알고리즘(Algorithms)
  * DQN (Deep Q-Network)
  * SARSA (State-Action-Reward-State-Action)
- 목표(Goal):  
  * 랜덤 전략 대비 수익 개선 검증  
  * 보수적/공격적 입찰 전략의 특성 분석
    
또한 학습된 정책(policy)의 행동 결정 기준을 이해하기 위해 다음을 수행합니다:
 - 정책 민감도 분석 (Policy Sensitivity Plot)
 - Q-value 민감도 분석 (Q-value Sensitivity)
 - SHAP 기반 Feature Importance 분석


## 프로젝트 최종 보고서 (PDF 다운로드/열기)


아래 링크를 클릭하면 PDF를 바로 열람할 수 있습니다:

👉  [RL_project(A70055 박은지).pdf 열기](https://github.com/eumjil/RL_project/blob/main/RL_project(A70055%20%EB%B0%95%EC%9D%80%EC%A7%80).pdf)

---

## 2. 폴더 및 파일 구조 (Project Structure)

```text
RL_project/
├─ RL00. 경매데이터 크롤링.py   # 경매 데이터 크롤링 코드 (선택, 필요 시)
├─ RL01. 분석용 데이터 생성.py   # 분석용 데이터 생성/정제 코드 (선택, 필요 시)
├─ RL02. 강화학습.py            # 강화학습 환경 정의 및 DQN/SARSA 학습 코드
├─ RL03. 추가 시각화 등.py            # 시각화 
├─ auction_analysis_ready.csv    # 전처리 완료 데이터
├─ RL_project(A70055 박은지).pdf   # 결과보고서
└─ README.md                   # 프로젝트 소개 및 실행 방법 (현재 파일)

```

## 3. 개발 환경 (Environment)

```text

Python 3.13

주요 라이브러리:

pandas
numpy
matplotlib
scikit-learn
torch (PyTorch)
shap
collections
random
math
tqdm
gym
seaborn

```
## 4. 실행 방법 (How to Run)

```
1) 데이터 수집 (선택)
python "RL00. 경매데이터 크롤링"
* 내용: 경매데이터제공사이트에서 자료수집

2) 데이터 전처리 (선택)
python "RL01. 분석용 데이터 생성.py"

3) 강화학습 모델 학습
python "RL02. 강화학습.py"
* auction_analysis_ready.csv 파일을 실행
* 내용
 - a. 데이터 로드 및 상태 변수 구성
 - b. 경매 강화학습 환경 정의(PortfolioAuctionEnv)
 - c. DQN 학습, SARSA 학습
 - d. 정책 평가 및 성능 비교
 - e. 분석(해석) : Policy Sensitivity Plot, Q-value Sensitivity Plot, SHAP Feature Importance

4) 결과 분석 및 시각화
python "RL03. 추가 시각화 등.py"
* RL02. 강화학습.py의 모델할습 결과 활용
* 내용
 - a. DQN vs SARSA 학습 곡선 비교
 - b. 정책별 평균 Reward + 95% 신뢰구간 (Bar Plot)
 - c. 정책별 Reward 분포 Boxplot (샘플링 기반)

