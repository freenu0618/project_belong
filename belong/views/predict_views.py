from datetime import datetime

from flask import Blueprint, render_template, request, flash
from werkzeug.utils import redirect

from .. import db
from ..models import LonelyPrediction
from ..ml.loader import available_regions, available_years, predict_for

bp = Blueprint(
    "predict",              # ★ blueprint 이름
    __name__,
    url_prefix="/predict"   # ★ URL prefix → /predict/...
)


@bp.route("/", methods=("GET", "POST"))
def index():
    """
    /predict/ 에서 고독사 예측을 처리하는 뷰.

    - GET : 구/연도 선택 폼만 렌더링
    - POST:
        1) LonelyPrediction 테이블에서 (구, 연도) 캐시 조회
        2) 없으면 ML 모델을 통해 예측 후 DB에 저장
        3) 결과를 템플릿으로 전달
    """
    regions = available_regions()
    years = available_years()

    prediction = None
    from_cache = False

    if request.method == "POST":
        gu = request.form.get("gu")         # ★ HTML name="gu"
        year_raw = request.form.get("year") # ★ HTML name="year"

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

            # 1) DB 캐시 조회
            pred_row = LonelyPrediction.query.filter_by(gu=gu, year=year).first()

            if pred_row:
                prediction = pred_row
                from_cache = True
            else:
                # 2) ML 예측 호출
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
        "predict/form.html",   # ★ 템플릿 경로
        regions=regions,
        years=years,
        prediction=prediction,
        from_cache=from_cache,
    )
