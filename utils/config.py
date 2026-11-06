"""Configuración compartida entre el notebook y la app Streamlit (ACA N° 2)."""
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DIR_DATA = BASE / "data"
DIR_RAW = DIR_DATA / "raw" / "Dataset_BUSI_with_GT"
DIR_EJEMPLOS = DIR_DATA / "ejemplos"
DIR_MODELOS = BASE / "modelos"
DIR_FIGURAS = BASE / "figuras"
RUTA_DB = DIR_DATA / "busi.db"

SEED = 42
IMG_CACHE = 256          # tamaño al que se cachean las imágenes en memoria
IMG_SIZE = 224           # tamaño de entrada a las redes
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
STD_IMAGENET = (0.229, 0.224, 0.225)

# Orden de las clases = índice de salida de la red
CLASES = ["normal", "benigno", "maligno"]
CARPETA_A_CLASE = {"normal": "normal", "benign": "benigno", "malignant": "maligno"}
COLORES = {"normal": "#6C757D", "benigno": "#2E86AB", "maligno": "#D1495B"}
