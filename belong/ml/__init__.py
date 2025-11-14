"""
pybo.ml 패키지 공통 설정 모듈

- DATA_PATH : 학습/예측에 사용할 CSV 위치
- MODEL_PATH : 학습된 모델(pkl) 저장 위치
- FEATURE 목록 : 숫자형, 지역(구) 원-핫 컬럼 이름
"""

from pathlib import Path

# 현재 파일(pybo/ml/__init__.py)의 디렉터리
PACKAGE_ROOT = Path(__file__).resolve().parent

# 프로젝트 루트: flask_basic/ (pybo/ml → pybo → flask_basic)
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

# CSV 데이터 경로 (flask_basic/Dataset_ML.csv 기준)
DATA_PATH = PROJECT_ROOT / "Dataset_ML.csv"

# 학습된 모델(pipeline)을 저장할 위치 (pybo/ml/lonely_death_model.pkl)
MODEL_PATH = PACKAGE_ROOT / "lonely_death_model.pkl"

# 타깃 컬럼: 고독사 발생 인원수
TARGET_COL = "값"

# v0.4 노트북에서 사용한 숫자형 피처 목록
NUMERIC_FEATURES = [
    "연도",
    "노령화지수",
    "1인가구_비율",
    "65세 이상",
    "소비자물가",
    "저소득노인_65~79비율",
    "저소득노인_80이상비율",
    "기초생활수급자비율",
    "lag_1",  # 전년도 고독사 수 (나중에 생성)
]

# 지역(구) 원-핫 컬럼 (중구 제외)
REGION_FEATURES = [
    "강남구","강동구","강북구","강서구","관악구","광진구","구로구","금천구",
    "노원구","도봉구","동대문구","동작구","마포구","서대문구","서초구","성동구",
    "성북구","송파구","양천구","영등포구","용산구","은평구","종로구","중랑구",
]

# v0.4에서 추가로 만든 파생 피처들 (참고용)
ADDITIONAL_FEATURES = ["lag_1", "lag_2", "roll_3", "인구x노령화", "노인비x저소득"]

# 최종 학습/예측에 투입할 컬럼 (X)
FINAL_FEATURES = NUMERIC_FEATURES + REGION_FEATURES
