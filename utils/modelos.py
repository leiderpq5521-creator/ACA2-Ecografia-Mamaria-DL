"""Arquitecturas neuronales (PASO 3 de la ruta)."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .config import CLASES

N_CLASES = len(CLASES)
NOMBRE_EFFNET = "efficientnet_b0.ra_in1k"
NOMBRE_VIT = "facebook/deit-small-patch16-224"


class CNNSimple(nn.Module):
    """CNN entrenada desde cero (línea base): 4 bloques Conv-BN-ReLU-MaxPool."""

    def __init__(self, n_clases: int = N_CLASES, canales=(32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        capas, c_in = [], 3
        for c in canales:
            capas += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                      nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                      nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*capas)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, n_clases))

    def forward(self, x):
        return self.head(self.features(x))


def crear_efficientnet(preentrenado: bool = True, dropout: float = 0.3) -> nn.Module:
    """EfficientNet-B0 (ImageNet) con una nueva cabeza de 3 clases (transfer learning)."""
    import timm
    modelo = timm.create_model(NOMBRE_EFFNET, pretrained=preentrenado, num_classes=N_CLASES, drop_rate=dropout)
    # timm inicializa la cabeza con pesos U(±1/√3) (std ≈ 0,33): con 1280 entradas los logits iniciales son enormes
    # (pérdida ≈ 3,5) y la fase 1 no converge. Se reinicia la cabeza con pesos pequeños y sesgo 0.
    nn.init.normal_(modelo.classifier.weight, std=0.01)
    nn.init.zeros_(modelo.classifier.bias)
    return modelo


def congelar_efficientnet(modelo: nn.Module, bloques_entrenables: int = 0) -> None:
    """Fase 1: sólo la cabeza (bloques_entrenables=0). Fase 2: también los últimos k bloques + conv_head."""
    for p in modelo.parameters():
        p.requires_grad = False
    entrenables = [modelo.classifier]
    if bloques_entrenables > 0:
        entrenables += [modelo.conv_head, modelo.bn2, *modelo.blocks[-bloques_entrenables:]]
    for m in entrenables:
        for p in m.parameters():
            p.requires_grad = True


class ViTClasificador(nn.Module):
    """DeiT-Small (Vision Transformer de Hugging Face) adaptado a 3 clases; devuelve logits."""

    def __init__(self, preentrenado: bool = True, config_local=None):
        """config_local: carpeta con config.json guardada junto a los pesos (la app no descarga nada)."""
        super().__init__()
        from transformers import AutoConfig, AutoModelForImageClassification
        kw = dict(id2label=dict(enumerate(CLASES)), label2id={c: i for i, c in enumerate(CLASES)})
        if preentrenado:
            self.vit = AutoModelForImageClassification.from_pretrained(NOMBRE_VIT, ignore_mismatched_sizes=True, **kw)
        else:
            config = AutoConfig.from_pretrained(config_local or NOMBRE_VIT, **kw)
            self.vit = AutoModelForImageClassification.from_config(config)

    def forward(self, x):
        return self.vit(pixel_values=x).logits


def contar_parametros(modelo: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in modelo.parameters())
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return total, entrenables


def cargar_para_inferencia(ruta_pesos, arquitectura: str = "efficientnet_b0") -> nn.Module:
    """Reconstruye la red (sin descargar pesos de ImageNet) y carga los pesos ajustados."""
    if arquitectura == "efficientnet_b0":
        modelo = crear_efficientnet(preentrenado=False)
    elif arquitectura == "cnn_simple":
        modelo = CNNSimple()
    elif arquitectura == "deit_small":
        modelo = ViTClasificador(preentrenado=False, config_local=Path(ruta_pesos).parent / "deit_small_config")
    else:
        raise ValueError(f"Arquitectura no soportada en la app: {arquitectura}")
    modelo.load_state_dict(torch.load(ruta_pesos, map_location="cpu", weights_only=True))
    return modelo.eval()
