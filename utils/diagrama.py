"""Diagrama de la arquitectura EfficientNet-B0 adaptada (figura para notebook, póster y video)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .config import DIR_FIGURAS

CONGELADO, AJUSTADO, NUEVO = "#B8C4D6", "#2E86AB", "#D1495B"


def dibujar_arquitectura(ruta=DIR_FIGURAS / "11_arquitectura_efficientnet.png"):
    capas = [  # (texto, salida, color)
        ("Ecografía\n224×224\n×3", "", "#FFFFFF"),
        ("Conv 3×3\nstem", "112²×32", CONGELADO),
        ("MBConv1\n×1", "112²×16", CONGELADO),
        ("MBConv6\n×2", "56²×24", CONGELADO),
        ("MBConv6\n×2", "28²×40", CONGELADO),
        ("MBConv6\n×3", "14²×80", AJUSTADO),
        ("MBConv6\n×3", "14²×112", AJUSTADO),
        ("MBConv6\n×4", "7²×192", AJUSTADO),
        ("MBConv6\n×1", "7²×320", AJUSTADO),
        ("Conv 1×1\nhead", "7²×1280", AJUSTADO),
        ("Global\nAvgPool", "1280", NUEVO),
        ("Dropout\n+ Densa", "3", NUEVO),
        ("Softmax", "normal\nbenigno\nmaligno", NUEVO),
    ]
    fig, ax = plt.subplots(figsize=(17, 3.3))
    ax.set_xlim(0, len(capas) * 1.3); ax.set_ylim(-1.05, 2.7); ax.axis("off")
    for i, (txt, out, col) in enumerate(capas):
        x = i * 1.3 + .1
        h = 1.9 if i < 10 else 1.3
        ax.add_patch(FancyBboxPatch((x, 1 - h / 2), 1.05, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                    fc=col, ec="#1d2f4f", lw=1.2))
        ax.text(x + .525, 1, txt, ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="white" if col in (AJUSTADO, NUEVO) else "#1d2f4f")
        if out:
            ax.text(x + .525, -.05, out, ha="center", va="top", fontsize=8.5, color="#333")
        if i:
            ax.add_patch(FancyArrowPatch((x - .2, 1), (x - .02, 1), arrowstyle="-|>", mutation_scale=12, color="#1d2f4f"))
    # Grad-CAM
    xh = 9 * 1.3 + .625
    ax.add_patch(FancyArrowPatch((xh, 2.0), (xh, 2.45), arrowstyle="-|>", mutation_scale=12, color="#E07A00", lw=1.5))
    ax.text(xh, 2.5, "Grad-CAM: mapas 7×7 ponderados por el gradiente → mapa de calor", ha="center", fontsize=9,
            color="#E07A00", fontweight="bold")
    for k, (col, lab) in enumerate([(CONGELADO, "Congelado (pesos ImageNet)"),
                                    (AJUSTADO, "Fine-tuning fase 2 (LR 2e-4)"),
                                    (NUEVO, "Cabeza nueva, entrenada desde fase 1 (LR 1e-3)")]):
        ax.add_patch(FancyBboxPatch((.2 + k * 5.4, -.95), .35, .25, boxstyle="round,pad=0.02", fc=col, ec="#1d2f4f"))
        ax.text(.65 + k * 5.4, -.83, lab, va="center", fontsize=9.5)
    ax.set_title("EfficientNet-B0 adaptada a ecografía mamaria · 4,0 M parámetros · MBConv = convolución invertida "
                 "con squeeze-and-excitation", fontsize=11.5, color="#1d2f4f", loc="left")
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return ruta


def dibujar_vit(ruta=DIR_FIGURAS / "11b_arquitectura_vit.png"):
    """DeiT-Small (Vision Transformer) de Hugging Face con fine-tuning completo."""
    fig, ax = plt.subplots(figsize=(17, 3.3))
    ax.set_xlim(0, 17); ax.set_ylim(-1.05, 2.7); ax.axis("off")

    def caja(x, w, txt, col, sub="", h=1.9):
        ax.add_patch(FancyBboxPatch((x, 1 - h / 2), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                    fc=col, ec="#1d2f4f", lw=1.2))
        ax.text(x + w / 2, 1, txt, ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="white" if col in (AJUSTADO, NUEVO) else "#1d2f4f")
        if sub:
            ax.text(x + w / 2, -.05, sub, ha="center", va="top", fontsize=8.5, color="#333")

    def flecha(x0, x1):
        ax.add_patch(FancyArrowPatch((x0, 1), (x1, 1), arrowstyle="-|>", mutation_scale=12, color="#1d2f4f"))

    caja(.1, 1.05, "Ecografía\n224×224\n×3", "#FFFFFF")
    flecha(1.2, 1.4)
    # parches
    for k in range(4):
        for j in range(4):
            ax.add_patch(FancyBboxPatch((1.45 + j * .27, .5 + k * .27), .22, .22, boxstyle="square,pad=0", fc="#DDE6F0",
                                        ec="#1d2f4f", lw=.6))
    ax.text(1.97, -.05, "196 parches\n16×16", ha="center", va="top", fontsize=8.5, color="#333")
    flecha(2.55, 2.75)
    caja(2.8, 1.5, "Proyección\nlineal + [CLS]\n+ posición", AJUSTADO, "197 tokens × 384")
    flecha(4.35, 4.55)
    ax.add_patch(FancyBboxPatch((4.6, -.1), 7.3, 2.2, boxstyle="round,pad=0.05,rounding_size=0.1", fc="#EEF4FA",
                                ec="#2E86AB", lw=1.5, ls="--"))
    for k, (t, w) in enumerate([("LayerNorm", 1.1), ("Atención\nmulti-cabeza\n(6 cabezas)", 1.5), ("+ residual", 1.0),
                                ("LayerNorm", 1.1), ("MLP\n384→1536→384", 1.5)]):
        x = 4.75 + sum(ww + .2 for _, ww in [("", 1.1), ("", 1.5), ("", 1.0), ("", 1.1), ("", 1.5)][:k])
        caja(x, w, t, AJUSTADO, h=1.5)
    ax.text(8.25, 2.25, "Bloque Transformer × 12", ha="center", fontsize=10.5, fontweight="bold", color="#2E86AB")
    flecha(11.95, 12.15)
    caja(12.2, 1.2, "Token\n[CLS]\n+ LayerNorm", AJUSTADO, "384", h=1.3)
    flecha(13.45, 13.65)
    caja(13.7, 1.2, "Densa\n384→3", NUEVO, "3", h=1.3)
    flecha(14.95, 15.15)
    caja(15.2, 1.2, "Softmax", NUEVO, "normal\nbenigno\nmaligno", h=1.3)
    xa = 4.75 + 1.1 / 2
    ax.add_patch(FancyArrowPatch((xa, 1.95), (xa, 2.45), arrowstyle="-|>", mutation_scale=12, color="#E07A00", lw=1.5))
    ax.text(xa - .3, 2.5, "Grad-CAM: tokens 14×14 del último bloque", ha="left", fontsize=9, color="#E07A00",
            fontweight="bold")
    for k, (col, lab) in enumerate([(AJUSTADO, "Fine-tuning completo desde ImageNet (LR 5e-5, weight decay 0,05)"),
                                    (NUEVO, "Cabeza nueva de 3 clases")]):
        ax.add_patch(FancyBboxPatch((.2 + k * 8.0, -.95), .35, .25, boxstyle="round,pad=0.02", fc=col, ec="#1d2f4f"))
        ax.text(.65 + k * 8.0, -.83, lab, va="center", fontsize=9.5)
    ax.set_title("DeiT-Small (Vision Transformer, Hugging Face facebook/deit-small-patch16-224) · 21,7 M parámetros · "
                 "atención global entre parches", fontsize=11.5, color="#1d2f4f", loc="left")
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return ruta


if __name__ == "__main__":
    print(dibujar_arquitectura())
    print(dibujar_vit())
