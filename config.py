#pip install cx_oracle
import os
from flask_sqlalchemy import SQLAlchemy

BASE_DIR = os.path.dirname(__file__)
print("BASE_DIR:",BASE_DIR)

#Oracle 11g 설정
SQLALCHEMY_DATABASE_URI = "oracle+cx_oracle://scott:tiger@localhost:1521/xe"
print("SQLALCHEMY_DATABASE_URI:",SQLALCHEMY_DATABASE_URI)

SQLALCHEMY_TRACK_MODIFICATIONS = False
SECRET_KEY = "dev"