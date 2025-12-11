# RL_project

# 강화학습 기반 부동산 경매 포트폴리오 입찰 전략 연구  
Reinforcement Learning for Real-Estate Auction Portfolio Bidding Strategy

## 1. 프로젝트 개요 (Overview)

이 프로젝트는 서울시 아파트 경매 데이터를 활용하여  
강화학습(DQN, SARSA 등)을 이용한 입찰 전략 수립 및 성능 비교를 수행한 코드와 실험 결과를 정리한 것입니다.

- 환경(Environment): 경매 물건의 상태(state)와 입찰 비율(action)에 따른 수익(reward)을 정의
- 알고리즘(Algorithms): DQN, SARSA 기반 에이전트 학습
- 목표(Goal):  
  - 랜덤 전략 대비 수익 개선 검증  
  - 보수적/공격적 입찰 전략의 특성 분석  

---

## 2. 폴더 및 파일 구조 (Project Structure)

```text
RL_project/
├─ RL00. 경매데이터 크롤링.py   # 경매 데이터 크롤링 코드 (선택, 필요 시)
├─ RL01. 분석용 데이터 생성.py   # 분석용 데이터 생성/정제 코드 (선택, 필요 시)
├─ RL02. 강화학습.py            # 강화학습 환경 정의 및 DQN/SARSA 학습 코드
├─ RL03. 추가 시각화 등.py            # 시각화 
├─ auction_analysis_ready.csv    # 전처리 완료 데이터
├─ RL_project(A70055 박은지).pptx
└─ README.md                   # 프로젝트 소개 및 실행 방법 (현재 파일)

```

## 3. 개발 환경 (Environment)

```text

Python 3.x

주요 라이브러리:

pandas, numpy, matplotlib

scikit-learn

torch (PyTorch)

tqdm, gym

```
## 4. 실행 방법 (How to Run)

```
1) 데이터 수집 (선택)
python "RL00. 경매데이터 크롤링"

2) 데이터 전처리 (선택)
python "RL01. 분석용 데이터 생성.py"

3) 강화학습 모델 학습
python "RL02. 강화학습.py"
* auction_analysis_ready.csv 이용

4) 결과 분석 및 시각화
python "RL03. 추가 시각화 등.py"
* aRL02. 강화학습.py 결과값 이용

```
## 5. 결과보고서

```
RL_project(A70055 박은지).pdf
