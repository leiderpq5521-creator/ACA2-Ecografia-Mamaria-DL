# ACA N° 2 – Aprendizaje Automático (CUN)
**Del ML clásico al Deep Learning:** clasificación de ecografías mamarias (normal / benigno / maligno) con redes neuronales (CNN y Vision Transformer de Hugging Face) y despliegue en Streamlit.

**Integrantes:**
- Kevin David Guzman Payares
- Leider Antonio Pacheco Quinto
- Joaquin Pablo Redondo Tapias
- Jose Luis Ruiz Ospino

**Docente:** Hamilton Rivera Flor · **Pilar:** Salud · **Dataset:** Breast Ultrasound Images – BUSI (Al-Dhabyani et al., 2020)

**App en línea:** https://aca2-ecografia-mamaria-dl-26xgtzbp9fp9bjsii42gxm.streamlit.app/

## Resultados (conjunto de prueba, 109 imágenes nunca vistas)
| Modelo | Parámetros | Accuracy | F1 macro | ROC-AUC (OvR) | Recall maligno |
|---|---|---|---|---|---|
| CNN desde cero | 1,2 M | 0,569 | 0,554 | 0,746 | 0,581 |
| EfficientNet-B0 (*transfer learning*) | 4,0 M | 0,872 | 0,844 | 0,976 | 0,806 |
| **DeiT-Small – ViT de Hugging Face** (modelo de la app) | 21,7 M | **0,927** | **0,924** | **0,979** | **0,903** |

- El modelo de la app se eligió por el mejor **F1 macro en validación** (ViT 0,840 vs. EfficientNet 0,826). IC 95 % (bootstrap) del ViT en prueba: F1 macro 0,87–0,97; recall maligno 0,79–1,00.
- El ViT detecta 28 de 31 tumores malignos y no clasifica ninguna imagen normal como maligna. Acierta los 9 ejemplos de la app.
- Grad-CAM: la energía del mapa dentro de la lesión es 2,8× (EfficientNet-B0) y 1,8× (ViT) lo esperado al azar. La app permite alternar entre ambas redes.
- Puente con el ACA 1 (WDBC tabular): MLP F1 0,988 vs. regresión logística 0,976 (empate práctico).

## Contenido
| Ruta | Qué es |
|---|---|
| `ACA2_Deep_Learning_Cancer_Mama.ipynb` | Notebook completo: SQL → EDA → depuración → pipeline y aumento de datos → 3 redes → evaluación → Grad-CAM → puente con ACA 1 → exportación |
| `app.py` | App Streamlit (diagnóstico por imagen con Grad-CAM, lotes, historial SQL, desempeño) |
| `utils/` | Módulos: `db.py` (SQLite), `datos.py` (inventario, depuración, partición, aumento), `modelos.py`, `entrenamiento.py`, `gradcam.py`, `inferencia.py`, `diagrama.py` |
| `data/busi.db` | Base SQLite: tabla `imagenes` (inventario, exclusiones, partición) y `predicciones` (historial de la app) |
| `data/ejemplos/` | 9 ecografías del conjunto de **prueba** para la demo (3 por clase) |
| `data/wdbc_limpio.csv` | Dataset del ACA 1 (puente MLP vs. regresión logística) |
| `modelos/` | Pesos (`*.pt`), configuración del ViT y `metricas.json` |
| `figuras/` | Gráficos generados por el notebook |
| `entregables/` | Póster (`poster.html`, `Poster_ACA2_Cancer_Mama_DL.pdf`, `generar_poster.py`) y guion del video |
| `requirements.txt` / `requirements-notebook.txt` | Dependencias de la app / del notebook |

## Ejecutar la app (Windows, PowerShell)
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/streamlit run app.py
```
Se abre en http://localhost:8501. La app no necesita el dataset completo ni internet: usa los pesos de `modelos/`.

## Reentrenar (opcional, ~80 min en CPU)
```bash
.venv/Scripts/python -m pip install -r requirements-notebook.txt
.venv/Scripts/jupyter nbconvert --to notebook --execute ACA2_Deep_Learning_Cancer_Mama.ipynb --inplace --ExecutePreprocessor.timeout=10800
```
Si `data/raw/` no existe, el notebook descarga BUSI (204 MB, CC BY 4.0) desde la copia pública de Hugging Face. En Google Colab con GPU tarda mucho menos.

## Regenerar el póster con los enlaces
```bash
.venv/Scripts/python entregables/generar_poster.py --app https://TU-APP.streamlit.app --video https://TU-VIDEO --repo https://github.com/TU-USUARIO/TU-REPO
```

## Desplegar en Streamlit Community Cloud
1. Subir esta carpeta a GitHub (`.gitignore` excluye `.venv/` y `data/raw/`).
2. En https://share.streamlit.io → **Create app** → repositorio, rama y `app.py`; en *Advanced settings* elegir Python 3.12.
3. El historial SQLite se reinicia en cada redespliegue (limitación de la nube gratuita).

## Decisiones técnicas clave
- **Depuración:** BUSI contiene la misma ecografía guardada en clases distintas y duplicados. Se detectan con hash perceptual (pHash): se excluyen las de etiqueta contradictoria y los duplicados, y las casi idénticas se agrupan para que nunca queden repartidas entre entrenamiento y prueba.
- **Selección del modelo de la app:** por F1 macro en **validación**; la prueba sólo se usa para reportar.
- **Corrección en EfficientNet:** `timm` inicializa la cabeza con pesos U(±1/√3); con 1 280 entradas la pérdida inicial era ≈ 3,5 y la fase 1 no convergía. La cabeza se reinicia con N(0; 0,01).

> Herramienta educativa: no es un dispositivo médico ni sustituye el diagnóstico de un profesional de la salud.
