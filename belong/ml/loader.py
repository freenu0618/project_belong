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

from . import (
    DATA_PATH,
    MODEL_PATH,
    TARGET_COL,
    FINAL_FEATURES,
)
from .preprocess import build_feature_dataframe

# 모듈 import 시점에 한 번만 로드해서 캐시처럼 사용
_df_features = build_feature_dataframe(DATA_PATH)

try:
    _model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    _model = None  # 아직 학습 안했거나 pkl 없음


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
