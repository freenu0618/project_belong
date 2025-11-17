from belong import db
from datetime import datetime

class Question(db.Model):
    __tablename__ = 'question'
    id = db.Column(db.Integer,db.Sequence('question_seq', start=1, increment=1), primary_key=True)
    subject = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text(), nullable=False)
    create_date = db.Column(db.DateTime(), nullable=False)

class Answer(db.Model):
    __tablename__ = 'answer'
    id = db.Column(db.Integer, db.Sequence('answer_seq', start=1, increment=1), primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('question.id', ondelete='CASCADE'))
    question = db.relationship('Question', backref=db.backref('answer_set'))
    content = db.Column(db.Text(), nullable=False)
    create_date = db.Column(db.DateTime(), nullable=False)

class Users(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer,db.Sequence('users_seq', start=1, increment=1), primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password=db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120), unique=True,  nullable=False)

class LonelyPrediction(db.Model):
    """
    고독사 예측 결과를 저장하는 테이블.

    - gu   : 자치구 이름 (예: '강남구')
    - year : 연도 (예: 2023)
    - predicted_value : ML 모델이 예측한 고독사 인원수
    - actual_value    : 실제 관측값 (있다면 입력, 없으면 NULL)
    """
    __tablename__ = "lonely_prediction"

    id = db.Column(
        db.Integer,
        db.Sequence('lonely_prediction_seq', start=1, increment=1),
        primary_key=True,
    )
    gu = db.Column(db.String(20), nullable=False)
    year = db.Column(db.Integer, nullable=False)

    predicted_value = db.Column(db.Float, nullable=False)
    actual_value = db.Column(db.Float, nullable=True)

    created_at = db.Column(db.DateTime(), default=datetime.now)

    __table_args__ = (
        db.UniqueConstraint("gu", "year", name="uq_lonely_prediction_gu_year"),
    )