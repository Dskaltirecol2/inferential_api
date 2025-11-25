import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

class Settings:
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    
    # Remove accidental spaces — VERY IMPORTANT
    S3_BUCKET = os.getenv("S3_BUCKET", "kttmgcolmlprojectsbucket").strip()

    DB_HOST = os.getenv("DB_HOST")
    
    # Convert DB_PORT to int safely
    DB_PORT = int(os.getenv("DB_PORT", "3306"))

    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_NAME = os.getenv("DB_NAME")

    BASE_PREFIX = "machinelearningprojects"

settings = Settings()
