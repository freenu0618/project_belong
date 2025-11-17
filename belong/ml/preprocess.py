"""
pybo.ml.preprocess

- Dataset_ML.csv 를 로드하고
- 고독사 타깃(값)을 기준으로 lag/roll 및 파생 피처를 생성하는 모듈
"""
"""
...
- v1.x 서비스에서는 Flask에서 직접 사용하지 않고,
  오프라인 학습/연구용으로만 사용.
"""

from __future__ import annotations

from typing import Union
from pathlib import Path

import pandas as pd

from . import DATA_PATH, TARGET_COL

PathLike = Union[str, Path]


def load_raw_data(path: PathLike = DATA_PATH) -> pd.DataFrame:
    """
    원본 CSV(Dataset_ML.csv)를 읽어서 DataFrame으로 반환.
    """
    path = Path(path)
    df = pd.read_csv(path)

    # 연도는 int(또는 float)로 맞춰두는 편이 안전함
    df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype("Int64")

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    v0.4 노트북에서 했던 피처 엔지니어링 적용:

    - 구, 연도 기준 정렬
    - lag_1, lag_2 : 전년도 / 재작년 고독사 수
    - roll_3       : 3년 이동평균
    - 인구x노령화   : 총인구 × 노령화지수
    - 노인비x저소득 : 65세 이상 × 저소득노인_80이상비율
    """
    df = df.copy()

    # 정렬
    df = df.sort_values(["구", "연도"])

    # 그룹별로 타깃 기준 lag 생성
    df["lag_1"] = df.groupby("구")[TARGET_COL].shift(1)
    df["lag_2"] = df.groupby("구")[TARGET_COL].shift(2)

    # 3년 이동평균
    roll = df.groupby("구")[TARGET_COL].apply(lambda x: x.rolling(3).mean())
    # groupby 인덱스 정리
    roll = roll.reset_index(level=0, drop=True)
    df["roll_3"] = roll

    # 파생 피처
    # 인구x노령화
    if "총인구" in df.columns and "노령화지수" in df.columns:
        df["인구x노령화"] = df["총인구"] * df["노령화지수"]

    # 노인비x저소득
    if "65세 이상" in df.columns and "저소득노인_80이상비율" in df.columns:
        df["노인비x저소득"] = df["65세 이상"] * df["저소득노인_80이상비율"]

    # lag/roll 계산 때문에 앞부분 연도에 NaN이 생김 → 해당 행 제거
    df = df.dropna(subset=["lag_1", "lag_2", "roll_3"]).reset_index(drop=True)

    return df


def build_feature_dataframe(path: PathLike = DATA_PATH) -> pd.DataFrame:
    """
    CSV 로드 + 피처 엔지니어링까지 한 번에 수행하여
    학습/예측에 바로 쓸 수 있는 DataFrame을 반환.
    """
    df = load_raw_data(path)
    df = add_engineered_features(df)
    return df
