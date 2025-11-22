from sqlalchemy import Column, Integer, String, LargeBinary
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)
    password_hash = Column(String)

    front_embedding = Column(LargeBinary)
    left_embedding = Column(LargeBinary)
    right_embedding = Column(LargeBinary)
