"""Preprocesamiento para inferencia (app). Ligero: no depende de scikit-learn ni imagehash."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2 as T

from .config import IMG_CACHE, IMG_SIZE, MEDIA_IMAGENET, STD_IMAGENET

_TRANSFORM = T.Compose([T.Resize(IMG_SIZE, antialias=True), T.ToDtype(torch.float32, scale=True),
                        T.Normalize(MEDIA_IMAGENET, STD_IMAGENET)])


def preparar(img: Image.Image) -> torch.Tensor:
    """Imagen PIL → tensor 1×3×224×224, idéntico al pipeline de validación del notebook."""
    arr = np.asarray(img.convert("RGB").resize((IMG_CACHE, IMG_CACHE), Image.BILINEAR)).copy()
    return _TRANSFORM(torch.from_numpy(arr).permute(2, 0, 1)).unsqueeze(0)
