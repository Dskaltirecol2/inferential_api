import boto3
from core.settings import settings

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=settings.AWS_REGION
    )
