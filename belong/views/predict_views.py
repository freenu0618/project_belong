# belong/views/predict_views.py

from flask import Blueprint, render_template, request, current_app
from pathlib import Path
import pandas as pd

bp = Blueprint("predict", __name__, url_prefix="/predict")

# ==========================================
# 1) 미래 예측 CSV 로드 (2026~2075)
#    - Jupyter/노트북에서 미리 만들어둔
#      future_pred_2026_2075_v1_1_linear.csv 사용
# ==========================================

# 전역 캐시용 (매 요청마다 파일 다시 안 읽도록)
_FUTURE_DF = None

# CSV 안에서 "예측값" 컬럼 이름
#   - 네 CSV에서 컬럼명이 다르면 여기만 바꾸면 됨 (예: "y_pred")
PRED_COL = "예측값"


def _load_future_df() -> pd.DataFrame:
    """미래 예측 CSV를 한 번만 로드해서 캐시에 담아두는 헬퍼."""
    global _FUTURE_DF

    if _FUTURE_DF is not None:
        return _FUTURE_DF

    # Flask 앱(root_path) 기준: belong/ml/future_pred_*.csv
    csv_path = Path(current_app.root_path) / "ml" / "future_pred_2026_2075_v1_1_linear.csv"

    df = pd.read_csv(csv_path)

    # 최소 컬럼 체크
    required = {"구", "연도", PRED_COL}
    if not required.issubset(df.columns):
        raise RuntimeError(
            f"미래 예측 CSV에 필요한 컬럼이 없습니다. "
            f"필요: {required}, 현재: {set(df.columns)}"
        )

    # 연도 타입 정리
    df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype(int)

    _FUTURE_DF = df
    return _FUTURE_DF


def _future_regions():
    """미래 예측이 가능한 자치구 목록 반환."""
    df = _load_future_df()
    return sorted(df["구"].dropna().unique().tolist())


def _future_series_for_gu(gu: str):
    """
    특정 구에 대해 2026~2075년 예측 시계열 반환.

    return:
        years       : [2026, 2027, ..., 2075]
        values      : [명 단위 정수 리스트]
        table_rows  : 템플릿에서 표로 쓰기 좋은 dict 리스트
    """
    df = _load_future_df()

    df_gu = df[df["구"] == gu].copy()
    if df_gu.empty:
        return [], [], []

    df_gu = df_gu.sort_values("연도")

    # 명 단위 반올림 컬럼 추가
    df_gu["예측값_명"] = df_gu[PRED_COL].round().astype(int)

    years = df_gu["연도"].tolist()
    values = df_gu["예측값_명"].tolist()

    # 표에 필요한 컬럼만 dict로 변환
    table_rows = df_gu[["연도", PRED_COL, "예측값_명"]].rename(
        columns={PRED_COL: "예측값"}
    ).to_dict(orient="records")

    return years, values, table_rows


# ==========================================
# 2) 뷰 함수: /predict/future
#    - 구만 선택 → 2026~2075 예측 곡선 + 표 렌더링
# ==========================================

@bp.route("/future", methods=["GET", "POST"])
def future():
    """
    1인 고령가구 고독사 장기 예측 서비스 뷰 (2026~2075년)
    """
    gu_list = _future_regions()

    # 기본 선택 구: 첫 번째 구
    selected_gu = gu_list[0] if gu_list else None

    if request.method == "POST":
        # 폼에서 넘어온 구 값으로 갱신
        selected_gu = request.form.get("gu") or selected_gu

    years, values, table_rows = ([], [], [])
    if selected_gu is not None:
        years, values, table_rows = _future_series_for_gu(selected_gu)

    return render_template(
        "predict/future_predict.html",
        gu_list=gu_list,
        selected_gu=selected_gu,
        years=years,
        values=values,
        table_records=table_rows,
    )
