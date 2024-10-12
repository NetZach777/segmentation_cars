import boto3
import os

BUCKET_NAME = 'awsmodelseg'
MODEL_KEY = 'unet_light_model_weighted_data_normal.h5'
LOCAL_MODEL_PATH = '/home/ubuntu/unet_api/models/unet_light_model_weighted_data_normal.h5'

# Vérifie si le répertoire existe, sinon crée-le
os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)

# Télécharge le modèle depuis S3
s3 = boto3.client('s3')
s3.download_file(BUCKET_NAME, MODEL_KEY, LOCAL_MODEL_PATH)
print(f"Model downloaded from S3: {MODEL_KEY}")
