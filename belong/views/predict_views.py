from datetime import datetime
from flask import Blueprint, render_template, request, flash, current_app
from werkzeug.utils import redirect
from pathlib import Path
import pandas as pd

from .. import db
from ..models import LonelyPrediction
from ..ml.loader import available_regions, available_years, predict_for, get_future_curve_for_gu

bp = Blueprint("predict", __name__, url_prefix="/predict")

# ==========================================
# 1) 현재년도 예측 (ML 모델)
# ==========================================

@bp.route("/", methods=("GET", "POST"))
def index():
    """
    /predict/ - ML 기반 고독사 예측
    """
    regions = available_regions()
    years = available_years()

    prediction = None
    from_cache = False

    if request.method == "POST":
        gu = request.form.get("gu")
        year_raw = request.form.get("year")

        if not gu or not year_raw:
            flash("구와 연도를 모두 선택해 주세요.")
        else:
            try:
                year = int(year_raw)
            except ValueError:
                flash("연도 값이 올바르지 않습니다.")
                return render_template(
                    "predict/form.html",
                    regions=regions,
                    years=years,
                    prediction=None,
                    from_cache=False,
                )

            # DB 캐시 조회
            pred_row = LonelyPrediction.query.filter_by(gu=gu, year=year).first()

            if pred_row:
                prediction = pred_row
                from_cache = True
            else:
                # ML 예측 호출
                result = predict_for(gu, year)

                prediction = LonelyPrediction(
                    gu=gu,
                    year=year,
                    predicted_value=result["y_pred"],
                    actual_value=result.get("y_true"),
                    created_at=datetime.now(),
                )
                db.session.add(prediction)
                db.session.commit()
                from_cache = False

    return render_template(
        "predict/form.html",
        regions=regions,
        years=years,
        prediction=prediction,
        from_cache=from_cache,
    )

# ==========================================
# 2) 미래 예측 (2026~2075 CSV 기반)
# ==========================================

_FUTURE_DF = None
PRED_COL = "예측값"


def _load_future_df() -> pd.DataFrame:
    """미래 예측 CSV를 캐싱하여 로드"""
    global _FUTURE_DF

    if _FUTURE_DF is not None:
        return _FUTURE_DF

    csv_path = Path(current_app.root_path) / "ml" / "future_pred_2026_2075_v1_1_linear.csv"

    df = pd.read_csv(csv_path)

    required = {"구", "연도", PRED_COL}
    if not required.issubset(df.columns):
        raise RuntimeError(
            f"미래 예측 CSV에 필요한 컬럼이 없습니다. 필요: {required}, 현재: {set(df.columns)}"
        )

    df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype(int)

    _FUTURE_DF = df
    return _FUTURE_DF


def _future_regions():
    df = _load_future_df()
    return sorted(df["구"].dropna().unique().tolist())


def _future_series_for_gu(gu: str):
    df = _load_future_df()

    df_gu = df[df["구"] == gu].copy()
    if df_gu.empty:
        return [], [], []

    df_gu = df_gu.sort_values("연도")
    df_gu["예측값_명"] = df_gu[PRED_COL].round().astype(int)

    years = df_gu["연도"].tolist()
    values = df_gu["예측값_명"].tolist()

    table_rows = df_gu[["연도", PRED_COL, "예측값_명"]].rename(
        columns={PRED_COL: "예측값"}
    ).to_dict(orient="records")

    return years, values, table_rows



@bp.route("/future", methods=["GET", "POST"])
def future():
    """
    2026~2075년 장기 예측 서비스 전용 뷰
    - 연도 선택 없음
    - 구 선택만 존재
    """
    gu_list = available_regions()   # 25개 구
    selected_gu = gu_list[0] if gu_list else None

    if request.method == "POST":
        selected_gu = request.form.get("gu") or selected_gu

    # 특정 구의 미래 예측 50년 데이터
    records = []
    if selected_gu:
        records = get_future_curve_for_gu(selected_gu)  # [{연도, 예측값, 예측값_명}, ...]

    # 템플릿 렌더링
    return render_template(
        "predict/future_predict.html",
        gu_list=gu_list,
        selected_gu=selected_gu,
        records=records
    )