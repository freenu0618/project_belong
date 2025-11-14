"""
pybo.ml.train_lonely_death

- Dataset_ML.csv 를 읽고
- 피처 엔지니어링 → 학습 데이터 구성
- XGBoost Regressor + StandardScaler + ColumnTransformer 파이프라인 학습
- lonely_death_model.pkl 로 저장
"""

from __future__ import annotations

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import (
    DATA_PATH,
    MODEL_PATH,
    TARGET_COL,
    FINAL_FEATURES,
    NUMERIC_FEATURES,
    REGION_FEATURES,
)
from .preprocess import build_feature_dataframe


def train_and_save_model() -> None:
    # 1) 데이터 로드 + 피처 엔지니어링
    df = build_feature_dataframe(DATA_PATH)

    # 2) X, y 분리
    X = df[FINAL_FEATURES]
    y = df[TARGET_COL]

    # 3) 학습/검증 나누기
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) 전처리 파이프라인 (수치형 스케일링 + 구 원-핫 패스스루)
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", "passthrough", REGION_FEATURES),
        ]
    )

    # 5) 모델 정의 (v0.4 노트북 설정 반영)
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
        tree_method="hist",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    # 6) 학습
    pipeline.fit(X_train, y_train)

    # 7) 학습된 파이프라인 저장
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)


if __name__ == "__main__":
    train_and_save_model()