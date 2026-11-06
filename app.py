"""
App Streamlit – ACA N° 2 Aprendizaje Automático (CUN)
Clasificación de ecografías mamarias (normal / benigno / maligno) con la red de mejor F1 macro en
validación (EfficientNet-B0 o DeiT-Small/ViT) ajustada sobre BUSI, explicación visual con Grad-CAM
e historial de predicciones en SQLite.

Ejecutar:  streamlit run app.py
"""
import hashlib
import io
import json
import sqlite3

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import db
from utils.config import CLASES, COLORES, DIR_EJEMPLOS, DIR_FIGURAS, DIR_MODELOS, RUTA_DB
from utils.gradcam import GradCAM, capa_objetivo, superponer
from utils.inferencia import preparar
from utils.modelos import cargar_para_inferencia

st.set_page_config(page_title="Ecografía mamaria con Deep Learning", page_icon="🩺", layout="wide")
NOMBRES = {"normal": "Normal (sin lesión)", "benigno": "Lesión benigna", "maligno": "Lesión maligna"}
DESCRIPCION_ARQ = {
    "efficientnet_b0": "CNN EfficientNet-B0, *transfer learning* desde ImageNet",
    "deit_small": "Vision Transformer DeiT-Small (Hugging Face), *fine-tuning* desde ImageNet",
    "cnn_simple": "CNN de 4 bloques entrenada desde cero",
}


# ───────────────────────────── Carga de artefactos ──────────────────────────────
@st.cache_data
def cargar_metricas():
    return json.loads((DIR_MODELOS / "metricas.json").read_text(encoding="utf-8"))


met = cargar_metricas()


@st.cache_resource(show_spinner="Cargando la red neuronal…")
def cargar_modelo():
    torch.set_num_threads(max(1, torch.get_num_threads()))
    modelo = cargar_para_inferencia(DIR_MODELOS / met["pesos_app"], met["arquitectura_app"])
    return modelo, GradCAM(modelo, capa_objetivo(modelo))


@st.cache_resource
def conexion():
    return db.conectar(RUTA_DB)


modelo, gradcam = cargar_modelo()
con = conexion()
res = met["resultados_test"][met["modelo_app"]]


# ───────────────────────────── Funciones auxiliares ─────────────────────────────
def analizar(img: Image.Image):
    """Devuelve (probabilidades, mapa Grad-CAM de la clase predicha)."""
    cam, probs = gradcam(preparar(img))
    return probs, cam


def decision(probs: np.ndarray, umbral: float) -> tuple[str, bool]:
    """Clase con mayor probabilidad + alerta si P(maligno) supera el umbral de revisión."""
    clase = CLASES[int(np.argmax(probs))]
    return clase, (clase != "maligno" and probs[CLASES.index("maligno")] >= umbral)


def registrar(origen: str, contenido: bytes, clase: str, probs) -> None:
    """Guarda la predicción en SQLite una sola vez por imagen y sesión."""
    clave = hashlib.md5(contenido).hexdigest()
    vistos = st.session_state.setdefault("registradas", set())
    if clave not in vistos:
        db.registrar_prediccion(con, origen, met["modelo_app"], clase, probs)
        vistos.add(clave)


def grafico_probs(probs: np.ndarray, umbral: float):
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.barh(CLASES[::-1], probs[::-1], color=[COLORES[c] for c in CLASES[::-1]])
    for i, v in enumerate(probs[::-1]):
        dentro = v > .85
        ax.text(v - .01 if dentro else v + .01, i, f"{v:.1%}", va="center", ha="right" if dentro else "left",
                fontweight="bold", color="white" if dentro else "black")
    ax.axvline(umbral, color="black", ls="--", lw=1)
    ax.text(umbral, 2.45, f" umbral revisión maligno {umbral:.2f}", fontsize=8)
    ax.set_xlim(0, 1); ax.set_xlabel("Probabilidad")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig


# ───────────────────────────── Encabezado y barra lateral ───────────────────────
st.title("🩺 Clasificación de ecografías mamarias con Deep Learning")
st.caption("ACA N° 2 · Aprendizaje Automático · CUN – Especialización en Inteligencia Artificial · "
           f"Red neuronal **{met['modelo_app']}** ajustada con *Breast Ultrasound Images (BUSI)*")
st.warning("**Herramienta educativa.** No es un dispositivo médico y no sustituye el criterio ni el diagnóstico de un "
           "profesional de la salud.", icon="⚠️")

with st.sidebar:
    st.header("Modelo")
    st.markdown(f"**Arquitectura:** {DESCRIPCION_ARQ.get(met['arquitectura_app'], met['modelo_app'])}  \n"
                f"*Elegida por el mejor F1 macro en validación*  \n"
                f"**Clases:** normal · benigno · maligno  \n"
                f"**Entrenamiento:** {met['n']['train']} imágenes  \n**Validación:** {met['n']['val']}  \n"
                f"**Prueba:** {met['n']['test']}")
    st.divider()
    st.subheader("Umbral de revisión")
    umbral = st.slider("Alertar si P(maligno) supera", 0.05, 0.95, 0.30, 0.05,
                       help="Además de la clase más probable, la app avisa cuando la probabilidad de malignidad supera "
                            "este valor, para no pasar por alto casos dudosos (prioriza la sensibilidad).")
    st.divider()
    st.subheader("Desempeño en prueba")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{res['accuracy']:.1%}")
    c2.metric("F1 macro", f"{res['f1_macro']:.3f}")
    c1.metric("Recall maligno", f"{res['recall_maligno']:.1%}")
    c2.metric("ROC-AUC", f"{res['roc_auc_ovr']:.3f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔬 Diagnóstico por imagen", "📁 Análisis por lotes", "🗄️ Historial (SQL)",
                                        "📈 Desempeño del modelo", "ℹ️ Acerca de"])

# ───────────────────────────── 1. Diagnóstico por imagen ────────────────────────
with tab1:
    ejemplos = sorted(DIR_EJEMPLOS.glob("*.png"))
    c_in1, c_in2 = st.columns([1, 1])
    # Se analiza la ÚLTIMA fuente que el usuario tocó (archivo subido o ejemplo)
    def _fuente(nombre):
        st.session_state["fuente"] = nombre

    with c_in1:
        subida = st.file_uploader("Sube una ecografía mamaria (PNG/JPG)", type=["png", "jpg", "jpeg"],
                                  on_change=_fuente, args=("subida",))
    with c_in2:
        opcion = st.selectbox("…o elige una imagen de ejemplo (conjunto de prueba, no vista en el entrenamiento)",
                              ["—"] + [e.name for e in ejemplos], on_change=_fuente, args=("ejemplo",),
                              format_func=lambda n: n if n == "—" else f"{n}  (etiqueta real: {n.split('_')[0]})")
    hay_subida, hay_ejemplo = subida is not None, opcion != "—"
    fuente = st.session_state.get("fuente")
    if hay_subida and hay_ejemplo:
        usar = fuente if fuente in ("subida", "ejemplo") else "subida"
    else:
        usar = "subida" if hay_subida else ("ejemplo" if hay_ejemplo else None)
    contenido, origen = None, None
    if usar == "subida":
        contenido, origen = subida.getvalue(), subida.name
    elif usar == "ejemplo":
        contenido, origen = (DIR_EJEMPLOS / opcion).read_bytes(), f"ejemplo:{opcion}"
    if hay_subida and hay_ejemplo:
        st.caption(f"Analizando **{origen.replace('ejemplo:', 'el ejemplo ')}** (la última opción que elegiste).")

    if contenido is None:
        st.info("Sube una imagen o elige un ejemplo para obtener la predicción.")
    else:
        try:
            img = Image.open(io.BytesIO(contenido))
            img.load()
        except Exception:  # noqa: BLE001
            st.error("No se pudo leer el archivo como imagen.")
            st.stop()
        with st.spinner("Analizando…"):
            probs, cam = analizar(img)
        clase, alerta = decision(probs, umbral)
        registrar(origen, contenido, clase, probs)

        st.divider()
        r1, r2 = st.columns([1, 1])
        with r1:
            st.markdown("**Ecografía**")
            ver_cam = st.toggle("Mostrar mapa Grad-CAM", value=True,
                                help="Zonas que más influyeron en la decisión (rojo = mayor influencia).")
            st.image(superponer(img, cam) if ver_cam else img.convert("RGB"), width="stretch")
            st.caption(f"{origen} · {img.width}×{img.height} px")
        with r2:
            st.markdown("**Resultado**")
            conf = float(probs.max())
            if clase == "maligno":
                st.error(f"**{NOMBRES[clase]}** · confianza {conf:.1%}  \nSe recomienda valoración médica prioritaria "
                         "y estudio complementario (p. ej. biopsia).", icon="🔴")
            elif clase == "benigno":
                st.info(f"**{NOMBRES[clase]}** · confianza {conf:.1%}  \nSeguimiento según criterio médico.", icon="🔵")
            else:
                st.success(f"**{NOMBRES[clase]}** · confianza {conf:.1%}", icon="🟢")
            if alerta:
                st.warning(f"P(maligno) = {probs[2]:.1%} supera el umbral de revisión ({umbral:.2f}): "
                           "**caso a revisar** aunque la clase más probable no sea maligna.", icon="⚠️")
            if conf < 0.6:
                st.info("La confianza es baja (< 60 %): el modelo duda entre clases. Interpreta con cautela.")
            st.pyplot(grafico_probs(probs, umbral), clear_figure=True)
            with st.expander("¿Cómo leer el mapa Grad-CAM?"):
                st.markdown("Grad-CAM resalta las regiones de la imagen cuyo cambio más alteraría la puntuación de la "
                            "clase predicha. Si el calor está **sobre la lesión**, la red se basa en la anatomía "
                            "relevante; si está en bordes, textos o marcas del ecógrafo, desconfía del resultado. "
                            f"En prueba, {met['gradcam']['energia_media_en_lesion']:.0%} del mapa cayó dentro de la "
                            f"lesión, que ocupa en promedio {met['gradcam']['area_media_lesion']:.0%} de la imagen "
                            f"({met['gradcam']['razon']:.1f}× lo esperado al azar).")

# ───────────────────────────── 2. Lotes ─────────────────────────────────────────
with tab2:
    st.subheader("Clasifica varias ecografías a la vez")
    archivos = st.file_uploader("Selecciona varias imágenes", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    usar_ej = st.checkbox(f"Usar las {len(ejemplos)} imágenes de ejemplo")
    lote = [(a.name, a.getvalue()) for a in archivos] if archivos else []
    if usar_ej and not lote:
        lote = [(e.name, e.read_bytes()) for e in ejemplos]
    if lote:
        filas, prog = [], st.progress(0.0, "Procesando…")
        for k, (nombre, contenido) in enumerate(lote):
            try:
                probs, _ = analizar(Image.open(io.BytesIO(contenido)))
                clase, alerta = decision(probs, umbral)
                registrar(nombre, contenido, clase, probs)
                filas.append({"archivo": nombre, "clase_predicha": clase, "p_normal": probs[0], "p_benigno": probs[1],
                              "p_maligno": probs[2], "revisar": "⚠️" if alerta or clase == "maligno" else ""})
            except Exception as e:  # noqa: BLE001
                filas.append({"archivo": nombre, "clase_predicha": f"error: {e}"})
            prog.progress((k + 1) / len(lote), f"{k + 1}/{len(lote)}")
        prog.empty()
        out = pd.DataFrame(filas)
        k1, k2, k3 = st.columns(3)
        for col, c in zip((k1, k2, k3), CLASES):
            col.metric(NOMBRES[c], int((out.clase_predicha == c).sum()))
        st.dataframe(out.style.format({c: "{:.1%}" for c in ("p_normal", "p_benigno", "p_maligno")}, na_rep="")
                     .background_gradient(subset=["p_maligno"], cmap="Reds", vmin=0, vmax=1), width="stretch")
        st.download_button("⬇️ Descargar resultados (CSV)", out.to_csv(index=False).encode("utf-8"),
                           "resultados_ecografias.csv", "text/csv")

# ───────────────────────────── 3. Historial SQL ─────────────────────────────────
with tab3:
    st.subheader("Base de datos SQLite (`data/busi.db`)")
    hist = db.consultar(con, "SELECT * FROM predicciones ORDER BY id DESC")
    a1, a2 = st.columns([1, 2])
    with a1:
        st.metric("Predicciones registradas", len(hist))
        if len(hist):
            fig, ax = plt.subplots(figsize=(4, 2.6))
            cnt = hist.clase_predicha.value_counts().reindex(CLASES, fill_value=0)
            ax.bar(cnt.index, cnt.values, color=[COLORES[c] for c in CLASES])
            ax.set_title("Predicciones por clase", fontsize=10)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            st.pyplot(fig, clear_figure=True)
    with a2:
        st.markdown("**Últimas predicciones** (tabla `predicciones`)")
        st.dataframe(hist.head(200), width="stretch", height=280, hide_index=True)

    st.markdown("**Consultas sobre el dataset** (tabla `imagenes`)")
    CONSULTAS = {
        "Imágenes por clase y partición": """SELECT clase, SUM(particion='train') AS train, SUM(particion='val') AS val,
       SUM(particion='test') AS test, SUM(excluida) AS excluidas
FROM imagenes GROUP BY clase ORDER BY clase""",
        "Motivos de exclusión (depuración)": """SELECT clase, motivo, COUNT(*) AS n
FROM imagenes WHERE excluida = 1 GROUP BY clase, motivo""",
        "Tamaño medio de lesión por clase": """SELECT clase, ROUND(100*AVG(area_lesion),1) AS pct_area_lesion,
       ROUND(AVG(ancho)) AS ancho, ROUND(AVG(alto)) AS alto
FROM imagenes WHERE excluida = 0 GROUP BY clase""",
        "Predicciones de la app con alta probabilidad de malignidad": """SELECT fecha, origen, clase_predicha,
       ROUND(p_maligno, 3) AS p_maligno
FROM predicciones WHERE p_maligno >= 0.5 ORDER BY p_maligno DESC""",
    }
    sel = st.selectbox("Consulta predefinida", list(CONSULTAS))
    sql = st.text_area("SQL (sólo lectura: se permite únicamente SELECT)", CONSULTAS[sel], height=110)
    if st.button("Ejecutar consulta"):
        if not sql.strip().lower().startswith(("select", "with")):
            st.error("Sólo se permiten consultas SELECT.")
        else:
            try:
                ro = sqlite3.connect(f"file:{RUTA_DB.as_posix()}?mode=ro", uri=True)
                st.dataframe(pd.read_sql_query(sql, ro), width="stretch", hide_index=True)
                ro.close()
            except Exception as e:  # noqa: BLE001
                st.error(f"Error en la consulta: {e}")

# ───────────────────────────── 4. Desempeño ─────────────────────────────────────
with tab4:
    st.subheader(f"Evaluación sobre {met['n']['test']} ecografías de prueba (nunca vistas)")
    cols = st.columns(4)
    ic = res["ic95"]
    for col, (k, nombre) in zip(cols, [("accuracy", "Accuracy"), ("f1_macro", "F1 macro"),
                                       ("recall_maligno", "Recall maligno"), ("roc_auc_ovr", "ROC-AUC (OvR)")]):
        col.metric(nombre, f"{res[k]:.3f}", help=f"IC 95 % (bootstrap): {ic[k][0]:.3f} – {ic[k][1]:.3f}")

    st.markdown("**Comparación de arquitecturas**")
    comp = pd.DataFrame({n: {"Parámetros (M)": r["parametros"] / 1e6, "Épocas": r["epocas"],
                             "F1 macro (val)": r["f1_macro_val"], "Accuracy": r["accuracy"], "F1 macro": r["f1_macro"],
                             "ROC-AUC": r["roc_auc_ovr"], "Recall maligno": r["recall_maligno"]}
                         for n, r in met["resultados_test"].items()}).T
    st.dataframe(comp.style.format("{:.3f}").format("{:.1f}", subset=["Parámetros (M)"]).format("{:.0f}", subset=["Épocas"])
                 .background_gradient(subset=["F1 macro", "Recall maligno"], cmap="Greens"), width="stretch")

    st.markdown(f"**Métricas por clase – {met['modelo_app']}**")
    st.dataframe(pd.DataFrame(res["por_clase"]).T.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}),
                 width="stretch")
    for archivo, titulo in [("05_curvas_entrenamiento.png", "Curvas de entrenamiento y validación"),
                            ("06_matrices_confusion.png", "Matrices de confusión"), ("07_roc.png", "Curvas ROC"),
                            ("09_gradcam.png", "Grad-CAM en prueba"), ("10_puente_wdbc.png", "Puente con el ACA 1 (WDBC)")]:
        ruta = DIR_FIGURAS / archivo
        if ruta.exists():
            st.markdown(f"**{titulo}**")
            st.image(str(ruta), width="stretch")

# ───────────────────────────── 5. Acerca de ─────────────────────────────────────
with tab5:
    dep = met["depuracion"]
    st.markdown(f"""
### Flujo metodológico
1. **Exploración y gestión SQL:** inventario de las {met['n']['total_original']} ecografías de BUSI en SQLite (tamaño, intensidad, área de la lesión, MD5 y hash perceptual).
2. **Preparación y pipeline:** se excluyeron {met['n']['excluidas']} imágenes ({dep.get('conflicto de etiqueta', 0)} con la misma ecografía etiquetada en dos clases y {dep.get('duplicado visual', 0)} duplicados); partición 70/15/15 estratificada y agrupada por casi-duplicados (sin fuga); redimensionado a 224×224, normalización ImageNet y aumento de datos (recortes, volteo horizontal, rotación ±15°, brillo/contraste).
3. **Modelado neuronal:** CNN desde cero (línea base), **EfficientNet-B0** con *transfer learning* en dos fases y **DeiT-Small** (Vision Transformer de Hugging Face). La app usa **{met['modelo_app']}**, el de mejor F1 macro en validación.
4. **Evaluación:** loss/accuracy por época, F1 macro, recall de la clase maligna, matriz de confusión, ROC uno-contra-resto, IC 95 % bootstrap y Grad-CAM.
5. **Despliegue:** esta aplicación Streamlit (inferencia en CPU, historial en SQLite).

### Limitaciones
- Un único hospital y equipo de ultrasonido; {met['n']['test']} imágenes de prueba → intervalos de confianza amplios.
- Sin validación externa; las etiquetas de BUSI presentan inconsistencias conocidas.
- Uso educativo; no apto para decisiones clínicas. En Streamlit Cloud el historial SQLite se reinicia al redesplegar.

### Referencias
- Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020). Dataset of breast ultrasound images. *Data in Brief, 28*, 104863. https://doi.org/10.1016/j.dib.2019.104863
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of ICML 2019*, 6105–6114.
- Touvron, H., et al. (2021). Training data-efficient image transformers & distillation through attention. *Proceedings of ICML 2021*, 10347–10357.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of ICCV 2017*, 618–626.
- Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. *Proceedings of EMNLP 2020: System Demonstrations*, 38–45.
""")
