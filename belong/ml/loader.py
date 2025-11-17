"""
pybo.ml.loader

- 학습된 lonely_death_model.pkl 을 로딩
- Dataset_ML 기반 피처 DataFrame 로딩
- Flask 뷰에서 바로 쓸 수 있는 헬퍼 함수 제공:
    - available_regions()
    - available_years()
    - predict_for(gu, year)
"""

from __future__ import annotations

from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

from . import (
    DATA_PATH,
    MODEL_PATH,
    TARGET_COL,
    FINAL_FEATURES,
    FUTURE_PRED_PATH
)
from .preprocess import build_feature_dataframe

# 모듈 import 시점에 한 번만 로드해서 캐시처럼 사용
_df_features = build_feature_dataframe(DATA_PATH)

try:
    _model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    _model = None  # 아직 학습 안했거나 pkl 없음

# ==========================================
# v1.x: 2026~2075년 장기 예측 CSV 로드
#  - 노트북에서 미리 생성한 future_pred_*.csv 를 읽어서
#    Flask 뷰에서 바로 사용할 수 있게 준비한다.
# ==========================================
try:
    _future_df = pd.read_csv(FUTURE_PRED_PATH)

    # 최소한의 검증: 필요한 컬럼이 있는지 확인
    required_cols = {"구", "연도", "예측값"}
    if not required_cols.issubset(_future_df.columns):
        raise ValueError(
            f"미래 예측 CSV에 {required_cols} 컬럼이 필요합니다. "
            f"현재 컬럼: {list(_future_df.columns)}"
        )

    # 연도 타입 정리
    _future_df["연도"] = (
        pd.to_numeric(_future_df["연도"], errors="coerce")
        .astype("Int64")
    )

except FileNotFoundError:
    _future_df = None  # 파일이 아직 없으면, 뷰에서 에러 처리


def available_regions():
    """
    예측 가능한 자치구 목록을 반환.
    """
    gu_list = sorted(_df_features["구"].unique().tolist())
    return gu_list


def available_years():
    """
    예측 가능한 연도 목록을 반환.
    """
    year_list = sorted(_df_features["연도"].dropna().astype(int).unique().tolist())
    return year_list


def predict_for(gu: str, year: int) -> Dict[str, Any]:
    """
    단일 (구, 연도)에 대한 예측 수행.

    반환 예:
    {
        "구": "강남구",
        "연도": 2023,
        "y_pred": 12.34,
        "y_true": 10.0  # 실제값이 있을 경우
    }
    """
    if _model is None:
        raise RuntimeError(
            "lonely_death_model.pkl 을 찾을 수 없습니다. "
            "먼저 'python -m pybo.ml.train_lonely_death' 를 실행해 주세요."
        )

    # 해당 (구, 연도) 행 선택
    row = _df_features[(_df_features["구"] == gu) & (_df_features["연도"] == year)]

    if row.empty:
        raise ValueError(f"데이터에 존재하지 않는 (구, 연도) 조합입니다: ({gu}, {year})")

    X = row[FINAL_FEATURES]
    y_true = None
    if TARGET_COL in row.columns:
        # 하나의 행만 있다고 가정
        y_true = float(row[TARGET_COL].iloc[0])

    # 모델 예측
    y_pred_arr = _model.predict(X)
    # XGBoost 출력은 numpy 배열이므로 float로 형변환
    y_pred = float(np.asarray(y_pred_arr)[0])

    return {
        "구": gu,
        "연도": int(year),
        "y_pred": y_pred,
        "y_true": y_true,
    }
    # ==========================================
# v1.x: 미래(2026~2075) 장기 예측 조회용 함수들
#  - Flask 뷰에서 이 함수만 사용해서
#    구 하나의 전체 예측 곡선을 가져올 수 있다.
# ==========================================

def get_future_curve_for_gu(gu: str):
    """
    특정 구(예: '중랑구')에 대한
    2026~2075년 예측 결과를 리스트로 반환.

    반환 예:
    [
        {"구": "중랑구", "연도": 2026, "예측값": 26.5, "예측값_명": 27},
        ...
    ]
    """
    if _future_df is None:
        raise RuntimeError(
            "미래 예측 CSV(_future_df)가 로드되지 않았습니다. "
            "Jupyter에서 미래 예측 CSV를 생성하고 "
            "FUTURE_PRED_PATH 위치에 파일을 배치하세요."
        )

    df_gu = _future_df[_future_df["구"] == gu].copy()
    if df_gu.empty:
        return []

    # 연도 순 정렬 + 명 단위로 반올림한 컬럼 추가
    df_gu = df_gu.sort_values("연도")
    df_gu["예측값_명"] = df_gu["예측값"].round().astype(int)

    return df_gu.to_dict(orient="records")


def future_available_years():
    """
    (필요하면) 미래 CSV에 들어 있는 연도 목록을 반환.
    UI에서 축 범위 확인용으로만 쓰고,
    사용자가 직접 연도를 선택하게 만들 필요는 없음.
    """
    if _future_df is None:
        return []
    years = (
        _future_df["연도"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    return sorted(years)
