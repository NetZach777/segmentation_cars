import os
import boto3
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import numpy as np
import io

# Désactiver l'utilisation des GPU pour forcer TensorFlow à utiliser le CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Utiliser les variables d'environnement pour configurer l'accès à AWS
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'eu-north-1')

# Créer une session AWS avec boto3
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# Configuration du bucket et du modèle
BUCKET_NAME = 'private-modelseg-637423565561'
OBJECT_KEY = 'unet_light_model_weighted_data_normal.h5'
LOCAL_MODEL_PATH = '/home/ubuntu/unet_api/models/unet_light_model_weighted_data_normal.h5'

# Fonction améliorée pour vérifier et créer le chemin du modèle
def check_model_path():
    if not os.path.exists(LOCAL_MODEL_PATH):
        model_dir = os.path.dirname(LOCAL_MODEL_PATH)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        download_model_from_s3(BUCKET_NAME, OBJECT_KEY, LOCAL_MODEL_PATH)
    return LOCAL_MODEL_PATH

# Fonction pour télécharger le modèle depuis S3
def download_model_from_s3(bucket_name, object_key, local_path):
    try:
        s3.download_file(bucket_name, object_key, local_path)
        print(f"✅ Modèle téléchargé depuis S3: s3://{bucket_name}/{object_key}")
    except Exception as e:
        print(f"❌ Échec du téléchargement : {e}")
        raise

# Créer l'application FastAPI
app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Event de démarrage
@app.on_event("startup")
async def startup_event():
    try:
        model_path = check_model_path()
        global model
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Application démarrée et modèle chargé")
    except Exception as e:
        print(f"❌ Erreur au démarrage : {e}")
        raise

# Endpoint de santé
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": os.path.exists(LOCAL_MODEL_PATH)
    }

# Palette de couleurs pour chaque classe du dataset Cityscapes
CITYSCAPES_PALETTE = [
    (128, 64, 128),  # void
    (244, 35, 232),  # flat
    (70, 70, 70),    # construction
    (102, 102, 156), # object
    (107, 142, 35),  # nature
    (70, 130, 180),  # sky
    (220, 20, 60),   # human
    (0, 0, 142)      # vehicle
]
CITYSCAPES_LABELS = ['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle']

# Fonction pour appliquer la palette de couleurs à un masque
def apply_color_palette(mask, palette):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        color_mask[mask == class_id] = color
    return color_mask

# Fonction pour prétraiter l'image
def preprocess_image(image, target_size):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Point de terminaison pour la segmentation
@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
            print(f"✅ Image chargée : {file.filename}")
        except UnidentifiedImageError:
            return {"error": "Format d'image non reconnu"}
        
        target_size = (256, 256)
        preprocessed_image = preprocess_image(image, target_size)

        prediction = model.predict(preprocessed_image)
        predicted_mask = np.argmax(prediction, axis=-1)[0]

        colored_mask = apply_color_palette(predicted_mask, CITYSCAPES_PALETTE)
        color_image = Image.fromarray(colored_mask)

        img_byte_arr = io.BytesIO()
        color_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except UnidentifiedImageError:
        return {"error": "Format d'image non reconnu"}
    except Exception as e:
        print(f"❌ Erreur durant le traitement : {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
