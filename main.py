import os
import requests
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
import numpy as np
import io

# Désactiver l'utilisation des GPU pour forcer TensorFlow à utiliser le CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# URL publique du modèle
MODEL_URL = 'https://awsmodelia.s3.eu-north-1.amazonaws.com/unet_light_model_weighted_data_normal.h5'

# Fonction pour télécharger le modèle depuis l'URL publique
def download_model_from_url(model_url, local_path):
    try:
        response = requests.get(model_url)
        response.raise_for_status()  # Vérifie s'il y a une erreur dans la réponse HTTP
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"Model downloaded from URL: {model_url}")
    except Exception as e:
        print(f"Failed to download model from URL: {e}")
        raise

# Téléchargement du modèle depuis l'URL publique vers un fichier local
LOCAL_MODEL_PATH = '/tmp/unet_light_model_weighted_data_normal.h5'
if not os.path.exists(LOCAL_MODEL_PATH):
    download_model_from_url(MODEL_URL, LOCAL_MODEL_PATH)

# Charger le modèle U-Net (utilise le chemin local après le téléchargement)
model = tf.keras.models.load_model(LOCAL_MODEL_PATH, compile=False)

# Créer l'application FastAPI
app = FastAPI()

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

# Fonction pour prétraiter l'image (ajuster la taille à l'entrée du modèle)
def preprocess_image(image, target_size):
    # Vérifier si l'image a un canal alpha (RGBA) et le convertir en RGB
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Redimensionner l'image à la taille cible
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalisation de l'image
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension batch
    return image

# Point de terminaison pour la segmentation avec visualisation des couleurs
@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    try:
        # Lire et ouvrir l'image
        contents = await file.read()

        try:
            image = Image.open(io.BytesIO(contents))
            print(f"Image successfully loaded: {file.filename}")
        except UnidentifiedImageError:
            return {"error": "Cannot identify image. Make sure the image is in the correct format."}

        # Prétraiter l'image
        target_size = (256, 256)
        preprocessed_image = preprocess_image(image, target_size)

        # Effectuer la prédiction
        prediction = model.predict(preprocessed_image)
        predicted_mask = np.argmax(prediction, axis=-1)[0]

        # Appliquer la palette de couleurs au masque
        colored_mask = apply_color_palette(predicted_mask, CITYSCAPES_PALETTE)

        # Convertir le masque coloré en image
        color_image = Image.fromarray(colored_mask)

        # Créer un flux binaire pour l'image
        img_byte_arr = io.BytesIO()
        color_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except UnidentifiedImageError:
        return {"error": "Cannot identify image. Make sure the image is in the correct format."}
    except Exception as e:
        # Log the full exception et return an error response
        print(f"Error during processing: {e}")
        return {"error": str(e)}

