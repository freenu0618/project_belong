from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

import config

db = SQLAlchemy()
migrate = Migrate()


def create_app():
    app = Flask(__name__)
    app.config.from_object(config)

    # ORM initialization
    db.init_app(app)
    migrate.init_app(app, db)

    # 모델 등록 (마이그레이션 / 테이블 인식을 위해)
    from . import models  # noqa: F401

    # Blueprint 등록
    from .views import (
        main_views,
        question_views,
        answer_views,
        auth_views,      # ✅ 기존에 있던 auth 그대로 유지
        predict_views,   # ✅ 우리가 새로 만든 고독사 예측 뷰
    )

    app.register_blueprint(main_views.bp)
    app.register_blueprint(question_views.bp)
    app.register_blueprint(answer_views.bp)
    app.register_blueprint(auth_views.bp)      # ✅ 계정/로그인 관련
    app.register_blueprint(predict_views.bp)   # ✅ /predict URL 담당

    return app


