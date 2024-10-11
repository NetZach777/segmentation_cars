import boto3

BUCKET_NAME = 'awsmodelseg'
MODEL_KEY = 'unet_light_model_weighted_data_normal.h5'
LOCAL_MODEL_PATH = '/tmp/unet_light_model_weighted_data_normal.h5'

s3 = boto3.client('s3')
s3.download_file(BUCKET_NAME, MODEL_KEY, LOCAL_MODEL_PATH)
print(f"Model downloaded from S3: {MODEL_KEY}")