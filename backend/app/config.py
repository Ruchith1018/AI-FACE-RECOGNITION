import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    DB_URL = os.getenv("DB_URL", "sqlite:///./faceauth.db")
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
    YAW_THRESHOLD = float(os.getenv("YAW_THRESHOLD", "12"))
    PITCH_THRESHOLD = float(os.getenv("PITCH_THRESHOLD", "12"))

settings = Settings()
