"""Bucle de entrenamiento, predicción y métricas (PASOS 3 y 4 de la ruta)."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from torch import nn

from .config import CLASES, SEED


def fijar_semillas(seed: int = SEED) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _epoca(modelo, loader, criterio, optimizador=None):
    entrenando = optimizador is not None
    modelo.train(entrenando)
    perdida, aciertos, n, y_all, p_all = 0.0, 0, 0, [], []
    with torch.set_grad_enabled(entrenando):
        for x, y in loader:
            logits = modelo(x)
            loss = criterio(logits, y)
            if entrenando:
                optimizador.zero_grad()
                loss.backward()
                optimizador.step()
            perdida += loss.item() * len(y)
            pred = logits.argmax(1)
            aciertos += (pred == y).sum().item()
            n += len(y)
            y_all.append(y); p_all.append(pred)
    y_all, p_all = torch.cat(y_all).numpy(), torch.cat(p_all).numpy()
    return perdida / n, aciertos / n, f1_score(y_all, p_all, average="macro")


def entrenar(modelo, dl_train, dl_val, epocas: int, lr: float, pesos_clase=None, paciencia: int = 6,
             weight_decay: float = 1e-4, historial=None, fase: str = "", verbose: bool = True):
    """Entrena con AdamW + ReduceLROnPlateau y early stopping sobre el F1 macro de validación.
    Devuelve el historial (lista de dicts) y deja en el modelo los mejores pesos."""
    historial = [] if historial is None else historial
    criterio = nn.CrossEntropyLoss(weight=pesos_clase)
    criterio_val = nn.CrossEntropyLoss()  # pérdida de validación sin ponderar (comparable entre modelos)
    params = [p for p in modelo.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    mejor_f1, mejor_estado, sin_mejora = -1.0, copy.deepcopy(modelo.state_dict()), 0
    for ep in range(1, epocas + 1):
        t0 = time.time()
        lt, at, ft = _epoca(modelo, dl_train, criterio, opt)
        lv, av, fv = _epoca(modelo, dl_val, criterio_val)
        sched.step(fv)
        historial.append(dict(epoca=len(historial) + 1, fase=fase, loss_train=lt, acc_train=at, f1_train=ft,
                              loss_val=lv, acc_val=av, f1_val=fv, lr=opt.param_groups[0]["lr"], seg=time.time() - t0))
        if verbose:
            print(f"[{fase}] ép {ep:02d} | loss {lt:.3f}/{lv:.3f} | acc {at:.3f}/{av:.3f} | F1 val {fv:.3f} "
                  f"| {time.time() - t0:.0f}s")
        if fv > mejor_f1 + 1e-4:
            mejor_f1, mejor_estado, sin_mejora = fv, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                if verbose:
                    print(f"  early stopping (mejor F1 val = {mejor_f1:.3f})")
                break
    modelo.load_state_dict(mejor_estado)
    return historial


@torch.no_grad()
def predecir(modelo, loader):
    modelo.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.softmax(modelo(x), 1)); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def metricas(y: np.ndarray, prob: np.ndarray) -> dict:
    pred = prob.argmax(1)
    pr, rc, f1, sop = precision_recall_fscore_support(y, pred, labels=range(len(CLASES)), zero_division=0)
    return {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "roc_auc_ovr": roc_auc_score(y, prob, multi_class="ovr", average="macro"),
        "recall_maligno": rc[CLASES.index("maligno")],
        "por_clase": {c: dict(precision=pr[i], recall=rc[i], f1=f1[i], soporte=int(sop[i])) for i, c in enumerate(CLASES)},
        "matriz_confusion": confusion_matrix(y, pred, labels=range(len(CLASES))).tolist(),
    }


def bootstrap_ic(y: np.ndarray, prob: np.ndarray, n: int = 1000, seed: int = SEED) -> dict:
    """Intervalos de confianza al 95 % por remuestreo bootstrap del conjunto de prueba."""
    rng = np.random.default_rng(seed)
    res = {k: [] for k in ("accuracy", "f1_macro", "roc_auc_ovr", "recall_maligno")}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < len(CLASES):
            continue
        m = metricas(y[idx], prob[idx])
        for k in res:
            res[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in res.items()}
