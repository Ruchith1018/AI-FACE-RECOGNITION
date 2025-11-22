from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./users.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String, nullable=True)
    login_method = Column(String, default="both")   # "password", "face", "both"
    embedding = Column(LargeBinary, nullable=True)  # FACE embedding stored as BLOB


# Create tables
def init_db():
    Base.metadata.create_all(bind=engine)
