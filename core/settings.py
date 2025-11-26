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

    #Dron
    DB_USER_DRON= os.getenv("DB_USER_DRON")
    DB_PASSWORD_DRON= os.getenv("DB_PS_DRON")

    FTP_HOST= os.getenv("FTP_HOST")
    FTP_USER= os.getenv("FTP_USER")
    FTP_PASSWORD= os.getenv("FTP_PASSWORD")
    FTP_PORT= os.getenv("FTP_PORT")





settings = Settings()
