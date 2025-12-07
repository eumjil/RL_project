import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ============================================================
# 0. 원본 데이터 불러오기
# ============================================================
# 원본 파일 경로/이름은 상황에 맞게 바꿔줘
raw_path = "C:\Users\user\Desktop\HW\분석용_매각완료건_정제데이터_251130.xlsx"
df = pd.read_excel(raw_path)

print("원본 shape:", df.shape)
print("컬럼:", list(df.columns))

# ============================================================
# 1. 숫자형 컬럼 숫자로 정리 (쉼표 제거 등)
# ============================================================

numeric_cols = [
    "건물면적_㎡",
    "토지면적_㎡",
    "감정가",
    "최저가",
    "낙찰가",
    "실거래가_adj",
    "실거래비율",
    "낙찰가율",
    "실거래가율",
    "저평가율",
    "최저가율",
    "낙찰률",
    "유찰횟수_adj",
    "건축년도",
    "월실거래건수",
    "실거래상대지수_local",
    "실거래상대지수_global",
    "기준금리",
    "floor",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
            .replace("", np.nan)
            .astype(float)
        )

# ============================================================
# 2. 파생변수 생성 (로그, 비율, 건물나이 등)
# ============================================================

# (1) 감정가 로그
if "감정가" in df.columns:
    df["감정가_log"] = np.log1p(df["감정가"])

# (1) 실거래가 로그
if "실거래가" in df.columns:
    df["실거래가_log"] = np.log1p(df["실거래가_adj"])

# (2) 건물면적 로그
if "건물면적_㎡" in df.columns:
    df["건물면적_log"] = np.log1p(df["건물면적_㎡"])


# (5) 건물나이 = 기준연도 - 건축년도
if "건축년도" in df.columns:
    기준연도 = 2025
    df["건물나이"] = 기준연도 - df["건축년도"]

# ============================================================
# 3. 시군구 기반 군집화 (구_cluster) 생성
#    - 이미 mapping 파일을 만들어둔 상태라면 이 블록은 생략하고
#      바로 4번에서 'gu_cluster_mapping.csv'를 읽어와도 됨
# ============================================================

if "시군구" in df.columns:
    cluster_features = [
        "감정가",
        "실거래가_adj",
        "낙찰가율",
        "최저가율",
        "낙찰률",
        "유찰횟수_adj",
        "실거래가율",
        "실거래상대지수_local",
        "실거래상대지수_global",
        "월실거래건수",
    ]
    cluster_features = [c for c in cluster_features if c in df.columns]

    gu_summary = df.groupby("시군구")[cluster_features].mean()

    scaler = StandardScaler()
    X = scaler.fit_transform(gu_summary)

    inertia_list = []
    k_range = range(2, 10)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        inertia_list.append(km.inertia_)

    plt.plot(k_range, inertia_list, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Within-cluster sum of squares)")
    plt.title("Elbow Method")
    plt.show()

    k = 5  # 군집 개수 (4~6 중에 골라서 바꿔도 됨)
    kmeans = KMeans(n_clusters=k, random_state=42)
    gu_summary["cluster"] = kmeans.fit_predict(X)

    # 시군구 → cluster 매핑
    gu_cluster_map = gu_summary["cluster"].to_dict()
    df["구_cluster"] = df["시군구"].map(gu_cluster_map)

    # 군집 결과 저장 (재현성을 위해)
    gu_summary[["cluster"]].to_csv("gu_cluster_mapping.csv", encoding="utf-8-sig")
    print("구 클러스터 mapping 저장 완료: gu_cluster_mapping.csv")
else:
    print("⚠ '시군구' 컬럼이 없어 군집화를 생략했습니다.")

# ============================================================
# 4. 범주형 변수 더미화 (층구분, 면적구분, 구_cluster)
# ============================================================

dummy_base_cols = []
if "층구분" in df.columns:
    dummy_base_cols.append("층구분")
if "면적구분" in df.columns:
    dummy_base_cols.append("면적구분")
if "구_cluster" in df.columns:
    dummy_base_cols.append("구_cluster")

if dummy_base_cols:
    df = pd.get_dummies(
        df,
        columns=dummy_base_cols,
        dtype=int,
        prefix={
            "층구분": "층",
            "면적구분": "면적",
            "구_cluster": "구클",
        },
    )

# ============================================================
# 5. 최종 분석용 feature 목록 정의
#   
# ============================================================

base_features = [
    "감정가_log",
    "건물면적_log",
    "최저가율",
    "낙찰가율",
    "실거래가율",
    "저평가율",
    "낙찰률",
    "유찰횟수_adj",
    "월실거래건수",
    "실거래상대지수_local",
    "실거래상대지수_global",
    "기준금리",
    "건물나이",
]

base_features = [c for c in base_features if c in df.columns]

# 더미 컬럼 자동 수집
dummy_feature_cols = [
    c
    for c in df.columns
    if c.startswith("면적_") or c.startswith("층_") or c.startswith("구클_")
]

final_feature_cols = base_features + dummy_feature_cols

# 타깃(예: 낙찰가율)을 같이 포함
target_col = "낙찰가율" if "낙찰가율" in df.columns else None

use_cols = final_feature_cols.copy()
if target_col and target_col not in use_cols:
    use_cols.append(target_col)

analysis_df = df[use_cols].copy()

# NaN / inf 처리
analysis_df = analysis_df.replace([np.inf, -np.inf], np.nan)
analysis_df = analysis_df.dropna(axis=0, how="any")

print("\n=== 최종 분석용 테이블 정보 ===")
print("shape:", analysis_df.shape)
print("features:")
for c in final_feature_cols:
    print("  -", c)

# ============================================================
# 6. 저장
# ============================================================

analysis_df.to_csv("auction_analysis_ready.csv", index=False, encoding="utf-8-sig")
print("\n→ 'auction_analysis_ready.csv' 로 저장 완료!")
"""
