import os
import boto3
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
import numpy as np
import io

# Désactiver le GPU pour éviter les erreurs si non disponible
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Paramètres S3
BUCKET_NAME = 'model-unet'  
MODEL_KEY = 'unet_light_model_weighted_data_normal.h5'
LOCAL_MODEL_PATH = '/tmp/unet_model.h5'

# Vérifier et créer le dossier temporaire si nécessaire
os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)

# Connexion S3 et téléchargement du modèle
s3 = boto3.client('s3')
try:
    s3.head_bucket(Bucket=BUCKET_NAME)
    print("✅ Connexion à S3 réussie.")
except Exception as e:
    print(f"❌ Erreur de connexion à S3 : {e}")
    raise

if not os.path.exists(LOCAL_MODEL_PATH):
    try:
        print(f"📥 Téléchargement du modèle depuis S3 ({MODEL_KEY})...")
        s3.download_file(BUCKET_NAME, MODEL_KEY, LOCAL_MODEL_PATH)
        print("✅ Modèle téléchargé avec succès.")
    except Exception as e:
        print(f"❌ Échec du téléchargement du modèle : {e}")
        raise

# Charger le modèle U-Net
try:
    model = tf.keras.models.load_model(LOCAL_MODEL_PATH, compile=False)
    print("✅ Modèle chargé avec succès.")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    raise

# Démarrer FastAPI
app = FastAPI()

# Endpoint santé
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running"}

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

CITYSCAPES_LABELS = ["void", "flat", "construction", "object", "nature", "sky", "human", "vehicle"]

# Appliquer la palette de couleurs
def apply_color_palette(mask, palette):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        color_mask[mask == class_id] = color
    return color_mask

# Prétraiter l'image
def preprocess_image(image, target_size=(256, 256)):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Endpoint de segmentation (renvoie une image)
@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Prétraitement
        preprocessed_image = preprocess_image(image)

        # Prédiction
        prediction = model.predict(preprocessed_image)
        predicted_mask = np.argmax(prediction, axis=-1)[0]

        # Appliquer la palette
        colored_mask = apply_color_palette(predicted_mask, CITYSCAPES_PALETTE)
        color_image = Image.fromarray(colored_mask)

        # Retourner l'image segmentée
        img_byte_arr = io.BytesIO()
        color_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except UnidentifiedImageError:
        return {"error": "Format d'image non valide."}
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return {"error": str(e)}

# Endpoint pour voir les résultats en JSON (DEBUG)
@app.post("/predict-json")
async def predict_json(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Prétraitement
        preprocessed_image = preprocess_image(image)

        # Prédiction
        prediction = model.predict(preprocessed_image)
        predicted_mask = np.argmax(prediction, axis=-1)[0]

        # Compter les classes présentes
        unique_classes, counts = np.unique(predicted_mask, return_counts=True)
        class_counts = {CITYSCAPES_LABELS[i]: int(counts[idx]) for idx, i in enumerate(unique_classes)}

        return {"prediction_summary": class_counts}

    except UnidentifiedImageError:
        return {"error": "Format d'image non valide."}
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return {"error": str(e)}

# Ajouter cette condition pour démarrer le serveur
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
