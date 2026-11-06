"""Genera el póster (A3 vertical) a partir de modelos/metricas.json y lo exporta a PDF con Edge/Chrome headless.

Uso:  python entregables/generar_poster.py [--app URL] [--video URL] [--repo URL]
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "entregables"

ap = argparse.ArgumentParser()
ap.add_argument("--app", default="(enlace de Streamlit Cloud – por publicar)")
ap.add_argument("--video", default="(enlace del video – por publicar)")
ap.add_argument("--repo", default="(enlace del repositorio GitHub – por publicar)")
args = ap.parse_args()


def enlace(valor: str) -> str:
    """Hipervínculo clicable en el HTML y en el PDF; texto plano mientras no haya URL."""
    if valor.startswith(("http://", "https://")):
        return f'<a href="{valor}">{valor}</a>'
    return f"<i>{valor}</i>"


m = json.loads((BASE / "modelos" / "metricas.json").read_text(encoding="utf-8"))
R = m["resultados_test"]
app = R[m["modelo_app"]]
n = m["n"]
dep = m["depuracion"]
gc = m["gradcam"]
pu = m["puente_wdbc"]
pct = lambda x: f"{100 * x:.1f}".replace(".", ",") + " %"
num = lambda x, d=3: f"{x:.{d}f}".replace(".", ",")
ic = app["ic95"]
pc = app["por_clase"]
cm = app["matriz_confusion"]

filas = "".join(
    f"<tr{' class=sel' if k == m['modelo_app'] else ''}><td>{k}</td><td>{v['parametros'] / 1e6:.1f}".replace(".", ",")
    + f"</td><td>{num(v['accuracy'])}</td><td>{num(v['f1_macro'])}</td><td>{num(v['roc_auc_ovr'])}</td>"
    f"<td>{num(v['recall_maligno'])}</td></tr>" for k, v in R.items())

ruta_fig = lambda f: (BASE / "figuras" / f).as_uri()


def figura_poster():
    """Curvas y matriz de confusión del modelo de la app, con letra grande para el póster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    h = m["historial"][m["modelo_app"]]
    ep = [x["epoca"] for x in h]
    f2 = [x["epoca"] for x in h if x["fase"].endswith("f2")]
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9), gridspec_kw={"width_ratios": [1, 1, .8]})
    ax[0].plot(ep, [x["loss_train"] for x in h], "o-", ms=3, label="entrenamiento")
    ax[0].plot(ep, [x["loss_val"] for x in h], "o-", ms=3, label="validación")
    ax[0].set(title="Pérdida (entropía cruzada)", xlabel="Época")
    ax[1].plot(ep, [x["acc_train"] for x in h], "o-", ms=3, label="accuracy entren.")
    ax[1].plot(ep, [x["acc_val"] for x in h], "o-", ms=3, label="accuracy val.")
    ax[1].plot(ep, [x["f1_val"] for x in h], "s--", ms=3, label="F1 macro val.")
    ax[1].set(title="Accuracy / F1", xlabel="Época")
    for a in ax[:2]:
        if f2:
            a.axvspan(min(f2) - .5, max(f2) + .5, color="orange", alpha=.1)
        a.legend(fontsize=9)
    ax[0].text(min(f2) if f2 else 1, ax[0].get_ylim()[1] * .92, " fine-tuning", color="darkorange", fontsize=10)
    c = np.array(cm)
    ax[2].imshow(c, cmap="Blues")
    for (i, j), v in np.ndenumerate(c):
        ax[2].text(j, i, v, ha="center", va="center", fontsize=15, color="white" if v > c.max() / 2 else "black")
    et = ["normal", "benigno", "maligno"]
    ax[2].set_xticks(range(3), et); ax[2].set_yticks(range(3), et)
    ax[2].set(title=f"Matriz de confusión (prueba)", xlabel="Predicho", ylabel="Real")
    ax[2].grid(False)
    ruta = BASE / "figuras" / "13_poster_modelo_app.png"
    fig.savefig(ruta, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return ruta


figura_poster()
preview = BASE / "figuras" / "12_app_preview.png"

ARQ_TXT = {
    "efficientnet_b0": ("EfficientNet-B0 con <i>transfer learning</i>", "11_arquitectura_efficientnet.png",
        "Fase 1: sólo la cabeza nueva (backbone congelado). Fase 2: <i>fine-tuning</i> de los 4 últimos bloques MBConv + "
        "conv_head con LR 5× menor. Reutilizar filtros de ImageNet evita el sobreajuste con ~500 imágenes; es ligera "
        "(16 MB) y permite Grad-CAM."),
    "deit_small": ("DeiT-Small, Vision Transformer de Hugging Face", "11b_arquitectura_vit.png",
        "La imagen se divide en 196 parches de 16×16 que se tratan como «palabras»; 12 bloques de autoatención relacionan "
        "todos los parches entre sí (contexto global de la lesión y su entorno). Preentrenado en ImageNet y ajustado por "
        "completo (LR 5e-5, weight decay 0,05). Superó a EfficientNet-B0 y a la CNN desde cero en validación."),
}
t_arq, img_arq, nota_arq = ARQ_TXT[m["arquitectura_app"]]
ARQ_HTML = (f'<div class="card mb"><h2>Arquitectura seleccionada para la app (mejor F1 en validación): {t_arq}</h2>'
            f'<img src="{ruta_fig(img_arq)}" style="max-height:44mm"><p class=note style="margin:0">{nota_arq}</p></div>')

CONCLUSIONES = json.loads((OUT / "conclusiones.json").read_text(encoding="utf-8"))

html = f"""<!doctype html><html lang=es><head><meta charset=utf-8><title>Póster ACA 2 – Ecografía mamaria con Deep Learning</title><style>
@page{{size:297mm 420mm;margin:0}}
*{{box-sizing:border-box}}body{{margin:0;width:297mm;height:420mm;font-family:'Segoe UI','DejaVu Sans',sans-serif;color:#1d2433;background:#f4f6fa;font-size:9.1pt;overflow:hidden}}
header{{background:linear-gradient(135deg,#1d2f4f,#2E86AB);color:#fff;padding:6mm 10mm 4.5mm;display:flex;gap:6mm;align-items:center}}header .tit{{flex:1}}header .lk a{{color:#fff;text-decoration:underline}}header .lk{{width:78mm;background:#D1495B;border-radius:2.5mm;padding:2.5mm 3.5mm;font-size:8pt;line-height:1.35;word-break:break-all}}
h1{{margin:0;font-size:20pt;line-height:1.12}}header p{{margin:2mm 0 0;font-size:9pt;opacity:.93}}
.wrap{{padding:4mm 10mm}}.row{{display:flex;gap:4mm;margin-bottom:3mm}}.mb{{margin-bottom:3mm}}
.card{{background:#fff;border-radius:3mm;padding:2.8mm 3.8mm;box-shadow:0 1px 3px #0002;flex:1;min-width:0}}
.card h2{{margin:0 0 2mm;font-size:11.5pt;color:#1d2f4f;border-left:2mm solid #D1495B;padding-left:2.5mm}}
.card p,.card li{{margin:0 0 1.2mm;line-height:1.33}}ul{{margin:0;padding-left:4.5mm}}
.steps{{display:flex;gap:2.5mm}}.step{{flex:1;background:#f7f9fc;border-top:2mm solid #2E86AB;border-radius:2mm;padding:2.2mm;font-size:8.3pt;line-height:1.3}}
.step>b:first-child{{display:block;color:#D1495B;font-size:9pt;margin-bottom:.8mm}}
.kpis{{display:flex;gap:2.5mm;margin-bottom:2.5mm}}.kpi{{flex:1;background:#1d2f4f;color:#fff;border-radius:2mm;padding:2mm;text-align:center}}
.kpi span{{display:block;font-size:17pt;font-weight:bold;color:#ffd166}}.kpi small{{font-size:7.6pt}}
img{{max-width:100%;display:block;margin:0 auto}}
table{{border-collapse:collapse;width:100%;font-size:8.3pt}}th,td{{padding:1.1mm 1.4mm;border-bottom:1px solid #dde3ec;text-align:center}}
th{{background:#eef2f8;color:#1d2f4f}}td:first-child,th:first-child{{text-align:left}}tr.sel td{{font-weight:bold;background:#fff4d6}}
.links{{background:#D1495B;color:#fff}}.links h2{{color:#fff;border-color:#fff}}.links a,.links p{{color:#fff;word-break:break-all}}
.ref{{font-size:6.9pt;line-height:1.22;padding-left:5mm;text-indent:-5mm;margin:0 0 1mm}}
.note{{font-size:7.8pt;color:#555}}
</style></head><body>
<header><div class=tit><h1>Del ML clásico al Deep Learning: clasificación de ecografías mamarias con redes neuronales y Streamlit</h1>
<p>ACA N° 2 · Aprendizaje Automático · Especialización en Inteligencia Artificial · CUN – Escuela de Ingeniería · Docente: Hamilton Rivera Flor<br>
<b>Integrantes:</b> Kevin David Guzman Payares · Leider Antonio Pacheco Quinto · Joaquin Pablo Redondo Tapias · Jose Luis Ruiz Ospino</p></div>
<div class=lk><b>App:</b> {enlace(args.app)}<br><b>Video:</b> {enlace(args.video)}<br><b>Código:</b> {enlace(args.repo)}</div></header>
<div class=wrap>
<div class=row>
<div class=card><h2>Objetivo general</h2><p>Entrenar y ajustar redes neuronales profundas (CNN y Transformer) que clasifiquen ecografías mamarias en <b>normal, benigno o maligno</b>, priorizando la sensibilidad para la clase maligna, y desplegarlas en una app Streamlit con explicación visual (Grad-CAM).</p></div>
<div class=card><h2>Situación problema</h2><p><i>¿Cómo aprovechar redes neuronales o modelos de Hugging Face para extraer patrones complejos en datos de salud y llevarlos a una interfaz interactiva?</i></p><p>En el ACA 1 la regresión logística clasificó tumores con 30 medidas <i>ya extraídas</i> (WDBC). Ahora la red aprende <b>directamente de la imagen</b> de ultrasonido, el primer examen clínico.</p></div>
<div class=card><h2>Dataset: BUSI (Al-Dhabyani et al., 2020)</h2><ul>
<li>{n['total_original']} ecografías de 600 mujeres (Hospital Baheya, El Cairo) con máscara de la lesión: 56 % benignas · 27 % malignas · 17 % normales.</li>
<li><b>Depuración:</b> {dep.get('conflicto de etiqueta', 0)} imágenes con la misma ecografía en dos clases y {dep.get('duplicado visual', 0)} duplicados excluidos → {n['train'] + n['val'] + n['test']} útiles: train {n['train']} · val {n['val']} · test {n['test']} (70/15/15, sin fuga).</li></ul></div>
</div>
<div class="card mb"><h2>Ruta metodológica</h2><div class=steps>
<div class=step><b>1 · Exploración y SQL</b>Inventario en <b>SQLite</b> (tamaño, intensidad, área de lesión, MD5, pHash). Consultas de balance y duplicados.</div>
<div class=step><b>2 · Pipeline</b>pHash para conflictos y casi-duplicados; StratifiedGroupKFold; 224×224, normalización ImageNet; <b>aumento de datos</b> (recorte, volteo, rotación ±15°, brillo/contraste); pérdida ponderada.</div>
<div class=step><b>3 · Modelado</b>CNN desde cero · <b>EfficientNet-B0</b> (<i>transfer learning</i> en 2 fases) · <b>DeiT-Small</b> (ViT, Hugging Face). AdamW, early stopping por F1 macro.</div>
<div class=step><b>4 · Evaluación</b>Curvas loss/accuracy, F1 macro, recall maligno, matriz 3×3, ROC OvR, IC 95 % bootstrap, Grad-CAM vs. máscara.</div>
<div class=step><b>5 · Despliegue</b>App Streamlit: imagen + Grad-CAM, lotes, historial y consultas SQL, desempeño.</div></div></div>
{ARQ_HTML}
<div class=row>
<div class=card style="flex:1"><h2>Resultados en prueba (n = {n['test']}, imágenes nunca vistas)</h2>
<div class=kpis><div class=kpi><span>{pct(app['accuracy'])}</span><small>Accuracy</small></div><div class=kpi><span>{num(app['f1_macro'])}</span><small>F1 macro</small></div>
<div class=kpi><span>{pct(app['recall_maligno'])}</span><small>Recall maligno</small></div><div class=kpi><span>{num(app['roc_auc_ovr'])}</span><small>ROC-AUC (OvR)</small></div></div>
<table><tr><th>Modelo</th><th>Parám. (M)</th><th>Accuracy</th><th>F1 macro</th><th>ROC-AUC</th><th>Recall mal.</th></tr>{filas}</table>
<p class=note style="margin-top:1.5mm">IC 95 % (bootstrap) de {m['modelo_app']}: accuracy {num(ic['accuracy'][0])}–{num(ic['accuracy'][1])} · F1 macro {num(ic['f1_macro'][0])}–{num(ic['f1_macro'][1])} · recall maligno {num(ic['recall_maligno'][0])}–{num(ic['recall_maligno'][1])}. Por clase (F1): normal {num(pc['normal']['f1'])} · benigno {num(pc['benigno']['f1'])} · maligno {num(pc['maligno']['f1'])}.</p></div>
<div class=card><h2>Vista previa de la app Streamlit</h2>{f'<img src="{preview.as_uri()}" style="max-height:56mm">' if preview.exists() else '<p>(captura pendiente)</p>'}</div>
</div>
<div class="card mb"><h2>{m['modelo_app']}: curvas de entrenamiento/validación y matriz de confusión</h2>
<img src="{ruta_fig('13_poster_modelo_app.png')}" style="max-height:41mm"></div>
<div class=row>
<div class=card style="flex:1.3"><h2>Interpretabilidad: Grad-CAM</h2><img src="{ruta_fig('09_gradcam.png')}" style="max-height:40mm">
<p class=note>En las {gc['n']} lesiones de prueba, {pct(gc['energia_media_en_lesion'])} del mapa de calor cae dentro de la lesión segmentada, que ocupa en promedio {pct(gc['area_media_lesion'])} de la imagen: <b>{num(gc['razon'], 1)}×</b> lo esperado si la red mirara al azar.</p></div>
<div class=card style="flex:1"><h2>Conclusiones</h2><ul>{''.join(f'<li>{c}</li>' for c in CONCLUSIONES)}</ul></div>
</div>
<div class="card mb" style="columns:2;column-gap:6mm"><h2 style="column-span:all">Referencias (APA 7)</h2>
<p class=ref>Al-Dhabyani, W., Gomaa, M., Khaled, H., &amp; Fahmy, A. (2020). Dataset of breast ultrasound images. <i>Data in Brief, 28</i>, 104863. https://doi.org/10.1016/j.dib.2019.104863</p>
<p class=ref>Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., &amp; Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. <i>Proceedings of the IEEE ICCV</i>, 618–626.</p>
<p class=ref>Tan, M., &amp; Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. <i>Proceedings of ICML</i>, 6105–6114.</p>
<p class=ref>Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., &amp; Jégou, H. (2021). Training data-efficient image transformers &amp; distillation through attention. <i>Proceedings of ICML</i>, 10347–10357.</p>
<p class=ref>Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. <i>Proceedings of EMNLP: System Demonstrations</i>, 38–45.</p>
</div>

</div></body></html>"""

ruta_html = OUT / "poster.html"
navegador = next((p for p in (shutil.which("msedge"), shutil.which("chrome"),
                              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                              r"C:\Program Files\Google\Chrome\Application\chrome.exe") if p and Path(p).exists()), None)


def escribir(zoom: float) -> None:
    ruta_html.write_text(html.replace("<style>", f"<style>.wrap{{zoom:{zoom:.2f}}}", 1), encoding="utf-8")


def paginas(pdf: Path) -> int:
    import pypdfium2
    doc = pypdfium2.PdfDocument(str(pdf))
    n = len(doc)
    doc.close()
    return n


escribir(1.0)
if navegador:
    pdf = OUT / "Poster_ACA2_Cancer_Mama_DL.pdf"
    zoom = 1.0
    while True:  # reduce la escala hasta que el póster quepa en una sola hoja A3
        escribir(zoom)
        subprocess.run([navegador, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        "--allow-file-access-from-files", f"--print-to-pdf={pdf}", ruta_html.as_uri()],
                       check=True, capture_output=True, timeout=120)
        if paginas(pdf) <= 1 or zoom <= 0.8:
            break
        zoom -= 0.02
    print(f"HTML: {ruta_html}\nPDF: {pdf} (escala {zoom:.2f}, {paginas(pdf)} página/s)")
else:
    print("No se encontró Edge/Chrome: abre poster.html e imprímelo a PDF (A3 vertical, sin márgenes).")
