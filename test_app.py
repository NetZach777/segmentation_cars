import io
import numpy as np  # Importer numpy
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from main import app, apply_color_palette, preprocess_image, CITYSCAPES_PALETTE

# Créer un client de test pour FastAPI
client = TestClient(app)

# 1. Test de l'endpoint /segment avec une image valide
def test_segment_valid_image():
    # Créer une image de test
    img = Image.new('RGB', (256, 256), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Envoyer la requête POST à l'API avec l'image
    response = client.post("/segment", files={"file": ("test_image.png", img_byte_arr, "image/png")})
    
    # Vérifier si la requête est réussie
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

# 2. Test de l'endpoint /segment avec une image invalide (fichier texte par exemple)
def test_segment_invalid_file():
    # Créer un fichier texte à la place d'une image
    file_content = io.BytesIO(b"this is not an image")
    
    # Envoyer la requête POST à l'API avec le fichier non image
    response = client.post("/segment", files={"file": ("test.txt", file_content, "text/plain")})

    # Vérifier si l'erreur est bien renvoyée
    assert response.status_code == 200
    assert response.json() == {"error": "Cannot identify image. Make sure the image is in the correct format."}

# 3. Test de la fonction apply_color_palette
def test_apply_color_palette():
    # Créer un masque factice avec deux classes
    mask = np.array([[0, 1], [2, 3]])
    expected_output = np.array([[[128, 64, 128], [244, 35, 232]], 
                                [[70, 70, 70], [102, 102, 156]]], dtype=np.uint8)
    
    # Appliquer la palette
    result = apply_color_palette(mask, CITYSCAPES_PALETTE)
    
    # Vérifier si le résultat est correct
    np.testing.assert_array_equal(result, expected_output)

# 4. Test de la fonction preprocess_image
def test_preprocess_image():
    # Créer une image de test de taille différente
    img = Image.new('RGB', (512, 512), color='blue')

    # Prétraiter l'image
    result = preprocess_image(img, target_size=(256, 256))

    # Vérifier que l'image est bien redimensionnée
    assert result.shape == (1, 256, 256, 3)  # Vérifie la forme de l'image prétraitée
    assert np.allclose(result[0, 0, 0], [0, 0, 1], atol=0.1)  # Vérifie que la couleur bleue est conservée
