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