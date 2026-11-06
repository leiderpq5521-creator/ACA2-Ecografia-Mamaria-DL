"""Inventario, depuración, partición y pipeline de imágenes (PASOS 1 y 2 de la ruta)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from .config import (CARPETA_A_CLASE, CLASES, DIR_RAW, IMG_CACHE, IMG_SIZE, MEDIA_IMAGENET, SEED,
                     STD_IMAGENET)


# ─────────────────────────── Inventario y depuración ───────────────────────────
def escanear_busi(raiz: Path = DIR_RAW) -> pd.DataFrame:
    """Recorre las carpetas de BUSI y devuelve una fila por imagen (sin máscaras)."""
    filas = []
    for carpeta, clase in CARPETA_A_CLASE.items():
        for ruta in sorted((raiz / carpeta).glob("*.png")):
            if "_mask" in ruta.stem:
                continue
            img = Image.open(ruta)
            arr = np.asarray(img.convert("L"))
            mascaras = sorted(ruta.parent.glob(f"{ruta.stem}_mask*.png"))
            union = np.zeros(arr.shape, bool)
            for m in mascaras:
                mk = np.asarray(Image.open(m).convert("L").resize(arr.shape[::-1], Image.NEAREST)) > 127
                union |= mk
            filas.append({
                "archivo": f"{carpeta}/{ruta.name}",
                "clase": clase,
                "ancho": img.width, "alto": img.height, "canales": len(img.getbands()),
                "intensidad_media": float(arr.mean()),
                "n_mascaras": len(mascaras),
                "area_lesion": float(union.mean()),
                "md5": hashlib.md5(ruta.read_bytes()).hexdigest(),
                "phash": str(imagehash.phash(Image.fromarray(arr))),
            })
    return pd.DataFrame(filas)


def agrupar_casi_duplicados(df: pd.DataFrame, umbral: int = 4) -> np.ndarray:
    """Union-find sobre la distancia de Hamming entre hashes perceptuales.
    Imágenes a distancia <= umbral quedan en el mismo grupo para que nunca se repartan
    entre entrenamiento y prueba (evita fuga de información)."""
    hashes = [imagehash.hex_to_hash(h) for h in df["phash"]]
    padre = list(range(len(df)))

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if hashes[i] - hashes[j] <= umbral:
                padre[raiz(j)] = raiz(i)
    raices = [raiz(i) for i in range(len(df))]
    _, grupo = np.unique(raices, return_inverse=True)
    return grupo


def depurar(df: pd.DataFrame, umbral_grupo: int = 10, umbral_conflicto: int = 4) -> pd.DataFrame:
    """Marca exclusiones y agrupa casi-duplicados.
    - Conflicto de etiqueta: la misma imagen (distancia <= umbral_conflicto) aparece en dos clases → se excluyen todas.
    - Duplicado visual (pHash idéntico dentro de la misma clase) → se conserva sólo la primera.
    - El resto de imágenes muy parecidas (mismo paciente / cortes contiguos) comparten `grupo_dup`."""
    df = df.copy()
    df["excluida"], df["motivo"] = 0, None
    g_conf = agrupar_casi_duplicados(df, umbral_conflicto)
    multiclase = pd.Series(g_conf).map(df.groupby(g_conf)["clase"].nunique()) > 1
    df.loc[multiclase.values, ["excluida", "motivo"]] = [1, "conflicto de etiqueta"]
    dup = df.duplicated(subset=["clase", "phash"], keep="first") & (df["excluida"] == 0)
    df.loc[dup, ["excluida", "motivo"]] = [1, "duplicado visual"]
    df["grupo_dup"] = agrupar_casi_duplicados(df, umbral_grupo)
    return df


def particionar(df: pd.DataFrame, n_splits: int = 20, n_test: int = 3, n_val: int = 3) -> pd.Series:
    """Partición estratificada por clase y agrupada por casi-duplicados (~70/15/15)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    part = pd.Series("train", index=df.index)
    for k, (_, idx) in enumerate(sgkf.split(df, df["clase"], groups=df["grupo_dup"])):
        if k < n_test:
            part.iloc[idx] = "test"
        elif k < n_test + n_val:
            part.iloc[idx] = "val"
    return part


# ─────────────────────────── Pipeline para la red ──────────────────────────────
def cargar_imagen(ruta: Path | str, tam: int = IMG_CACHE) -> torch.Tensor:
    """Lee una ecografía, la pasa a 3 canales y la redimensiona a tam×tam (uint8, C×H×W)."""
    img = Image.open(ruta).convert("RGB").resize((tam, tam), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)


def transformaciones(entrenamiento: bool) -> T.Compose:
    norm = [T.ToDtype(torch.float32, scale=True), T.Normalize(MEDIA_IMAGENET, STD_IMAGENET)]
    if entrenamiento:  # aumento de datos: sólo en entrenamiento
        return T.Compose([
            T.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            *norm,
        ])
    return T.Compose([T.Resize(IMG_SIZE, antialias=True), *norm])


class BusiDataset(Dataset):
    """Dataset en memoria: las 780 imágenes caben en RAM (~150 MB a 256×256×3)."""

    def __init__(self, df: pd.DataFrame, raiz: Path = DIR_RAW, transform=None):
        self.etiquetas = torch.tensor([CLASES.index(c) for c in df["clase"]])
        self.imagenes = [cargar_imagen(raiz / a) for a in df["archivo"]]
        self.archivos = list(df["archivo"])
        self.transform = transform

    def __len__(self):
        return len(self.imagenes)

    def __getitem__(self, i):
        x = self.imagenes[i]
        return (self.transform(x) if self.transform else x), self.etiquetas[i]

