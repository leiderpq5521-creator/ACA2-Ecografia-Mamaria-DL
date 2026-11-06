"""Grad-CAM (Selvaraju et al., 2017): mapa de calor de las regiones que más influyen en la predicción."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image


class GradCAM:
    def __init__(self, modelo: torch.nn.Module, capa: torch.nn.Module):
        self.modelo = modelo.eval()
        self._act = self._grad = None
        capa.register_forward_hook(lambda m, i, o: setattr(self, "_act", o))
        capa.register_full_backward_hook(lambda m, gi, go: setattr(self, "_grad", go[0]))

    def __call__(self, x: torch.Tensor, clase: int | None = None):
        """x: 1×3×H×W normalizado. Devuelve (cam H×W en [0,1], probabilidades)."""
        x = x.clone().requires_grad_(True)
        logits = self.modelo(x)
        probs = torch.softmax(logits, 1)[0].detach()
        clase = int(probs.argmax()) if clase is None else clase
        self.modelo.zero_grad()
        logits[0, clase].backward()
        act, grad = self._act, self._grad
        if act.dim() == 3:  # Transformer: tokens (B, 1+N, C) → mapa (B, C, √N, √N), sin el token [CLS]
            act, grad = (_tokens_a_mapa(t) for t in (act, grad))
        pesos = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((pesos * act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().numpy(), probs.numpy()


def _tokens_a_mapa(t: torch.Tensor) -> torch.Tensor:
    parches = t[:, 1:, :]
    lado = int(parches.shape[1] ** 0.5)
    return parches.reshape(t.shape[0], lado, lado, t.shape[2]).permute(0, 3, 1, 2)


def capa_objetivo(modelo: torch.nn.Module) -> torch.nn.Module:
    """Capa sobre la que se calcula Grad-CAM según la arquitectura."""
    if hasattr(modelo, "vit"):          # DeiT/ViT (Hugging Face): normalización de entrada del último bloque
        vm = modelo.vit.vit
        capas = vm.layers if hasattr(vm, "layers") else vm.encoder.layer  # según versión de transformers
        return capas[-1].layernorm_before
    if hasattr(modelo, "conv_head"):   # EfficientNet (timm)
        return modelo.conv_head
    if hasattr(modelo, "features"):    # CNNSimple
        return [m for m in modelo.features if isinstance(m, torch.nn.Conv2d)][-1]
    raise ValueError("No se encontró una capa convolucional para Grad-CAM")


def superponer(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = img.convert("RGB")
    calor = colormaps["jet"](np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize(base.size, Image.BILINEAR)) / 255.0)
    calor = Image.fromarray((calor[..., :3] * 255).astype(np.uint8))
    return Image.blend(base, calor, alpha)


def energia_en_mascara(cam: np.ndarray, mascara: np.ndarray) -> float:
    """Fracción de la 'energía' del mapa Grad-CAM que cae dentro de la lesión segmentada."""
    m = np.asarray(Image.fromarray(mascara.astype(np.uint8) * 255).resize(cam.shape[::-1], Image.NEAREST)) > 127
    return float(cam[m].sum() / (cam.sum() + 1e-8))
