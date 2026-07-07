import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

MODEL_PATH = os.path.join(BASE_DIR, "model", "plant_model.keras")

CLASS_NAMES = os.path.join(BASE_DIR, "model", "class_names.json")

DATABASE = os.path.join(BASE_DIR, "database", "plant.db")

IMAGE_SIZE = (224, 224)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

SECRET_KEY = "agriai_secret_key_2026"
