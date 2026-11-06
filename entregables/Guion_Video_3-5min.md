# GUION TÉCNICO Y LOCUCIÓN DE VIDEO (3 A 5 MINUTOS)
## PROYECTO ACA N° 2 – APRENDIZAJE AUTOMÁTICO
**Tema:** Del ML clásico al Deep Learning: clasificación de ecografías mamarias con redes neuronales y despliegue en Streamlit
**Institución:** Corporación Unificada Nacional de Educación Superior (CUN) · Especialización en Inteligencia Artificial
**Docente:** Hamilton Rivera Flor
**Integrantes:** Kevin David Guzman Payares · Leider Antonio Pacheco Quinto · Joaquin Pablo Redondo Tapias · Jose Luis Ruiz Ospino

**Duración estimada:** ~4 min 20 s (≈ 600 palabras a 135–140 palabras/min).
**Enlaces para la demo:**
- App: https://aca2-ecografia-mamaria-dl-26xgtzbp9fp9bjsii42gxm.streamlit.app/
- Código: https://github.com/leiderpq5521-creator/ACA2-Ecografia-Mamaria-DL
- Póster: `entregables/Poster_ACA2_Cancer_Mama_DL.pdf`

> **Antes de grabar:** abre la app unos minutos antes (en Streamlit Cloud tarda ~30 s en "despertar" si estuvo inactiva) y deja listo el póster en PDF.

---

## TABLA ESCENA POR ESCENA

| Escena y tiempo | Recurso visual en pantalla | Locutor sugerido | Texto de locución |
|---|---|---|---|
| **1** · 0:00–0:30 (30 s) | Encabezado del póster (título, integrantes, logos CUN). | Kevin David Guzman | "Cordial saludo. Presentamos el ACA número 2 de Aprendizaje Automático de la Especialización en Inteligencia Artificial de la CUN, con el docente Hamilton Rivera Flor. Somos Leider Pacheco, Joaquín Redondo, José Luis Ruiz y Kevin Guzmán. En el ACA 1 clasificamos tumores de mama a partir de medidas ya extraídas de la imagen. Ahora damos el paso al Deep Learning: que una red neuronal aprenda directamente de la ecografía." |
| **2** · 0:30–1:10 (40 s) | `figuras/02_mosaico_clases.png` → `figuras/03_depuracion_conflictos.png` → en la app, pestaña **Historial (SQL)**, consulta «Motivos de exclusión». | Leider Antonio Pacheco | "Usamos el dataset BUSI: 780 ecografías de 600 mujeres, en tres clases: normal, benigno y maligno. Todo el inventario se gestiona en una base de datos SQLite. Al depurarlo con hash perceptual encontramos un problema serio: 18 imágenes eran la misma ecografía guardada como benigna y como maligna, y 38 eran duplicados. Las excluimos y agrupamos las imágenes casi idénticas para que nunca quedaran a la vez en entrenamiento y en prueba. Así evitamos la fuga de información." |
| **3** · 1:10–2:00 (50 s) | `figuras/04_aumento_datos.png` → `figuras/11b_arquitectura_vit.png` (diagrama del póster). | Joaquin Pablo Redondo | "Partimos los datos en setenta, quince y quince por ciento. En entrenamiento aplicamos aumento de datos: recortes, volteos, rotaciones y cambios de brillo, que simulan cómo varía una ecografía real. Comparamos tres arquitecturas: una CNN entrenada desde cero, EfficientNet-B0 con transfer learning, y DeiT-Small, un Vision Transformer de Hugging Face. El Transformer divide la imagen en 196 parches y, con autoatención, relaciona cada parche con todos los demás: ve la lesión y su entorno al mismo tiempo." |
| **4** · 2:00–2:55 (55 s) | Bloque de resultados del póster (KPIs y tabla) → `figuras/13_poster_modelo_app.png` (curvas y matriz de confusión). | Jose Luis Ruiz | "Elegimos el modelo con el mejor F1 en validación, y fue el Vision Transformer. En 109 imágenes que nunca vio, logra una exactitud del 92,7 por ciento, un F1 macro de 0,92 y un área bajo la curva ROC de 0,98. Lo más importante clínicamente: detecta 28 de 31 tumores malignos, un 90 por ciento de sensibilidad. La CNN desde cero se quedó en 0,55 de F1: con apenas 500 imágenes, el transfer learning es decisivo. También corregimos un error de inicialización en EfficientNet que le impedía aprender en la primera fase; al arreglarlo pasó del 82 al 87 por ciento." |
| **5** · 2:55–3:55 (60 s) | **Demo en vivo** de la app: 1) elegir `maligno_2.png` (resultado, barras, Grad-CAM); 2) elegir `benigno_1.png`; 3) cambiar a **EfficientNet-B0** en la barra lateral y mostrar el mapa más localizado; 4) pestaña **Análisis por lotes** con las 9 imágenes de ejemplo; 5) pestaña **Historial (SQL)**. | Leider (o rotativo) | "Esta es la aplicación desplegada en Streamlit Cloud. Elegimos una ecografía del conjunto de prueba: el modelo la clasifica como lesión maligna con 98 por ciento de confianza y muestra la probabilidad de cada clase. El mapa Grad-CAM indica en qué zonas se apoyó la red. Si una imagen supera el treinta por ciento de probabilidad de malignidad, la app la marca para revisión aunque la clase más probable sea otra. Podemos cambiar a EfficientNet, cuyo mapa se concentra más sobre la lesión. También analizamos varias imágenes por lotes, y cada predicción queda registrada en la base de datos SQLite, que se puede consultar desde la propia app." |
| **6** · 3:55–4:20 (25 s) | Conclusiones del póster → enlaces de App y Código. | Voz conjunta / cierre | "En conclusión, el Deep Learning con transfer learning permite clasificar ecografías con alta sensibilidad, siempre que los datos estén bien depurados. Es una herramienta educativa de apoyo: no reemplaza el criterio médico y necesita validación externa. El código, el notebook y la app están en los enlaces del póster. Muchas gracias." |

---

## TEXTO CONTINUO PARA VOZ CON IA (TTS)
*(Para ElevenLabs, Clipchamp, Fliki, HeyGen, etc. Copia este bloque tal cual.)*

> Cordial saludo. Presentamos el ACA número 2 de Aprendizaje Automático de la Especialización en Inteligencia Artificial de la CUN, con el docente Hamilton Rivera Flor. Nuestro equipo está integrado por Kevin David Guzmán, Leider Antonio Pacheco, Joaquín Pablo Redondo y José Luis Ruiz. En el ACA 1 clasificamos tumores de mama a partir de medidas ya extraídas de la imagen. Ahora damos el paso al Deep Learning: que una red neuronal aprenda directamente de la ecografía.
>
> Usamos el dataset BUSI: 780 ecografías de 600 mujeres, en tres clases: normal, benigno y maligno. Todo el inventario se gestiona en una base de datos SQLite. Al depurarlo con hash perceptual encontramos un problema serio: 18 imágenes eran la misma ecografía guardada como benigna y como maligna, y 38 eran duplicados. Las excluimos y agrupamos las imágenes casi idénticas para que nunca quedaran a la vez en entrenamiento y en prueba. Así evitamos la fuga de información.
>
> Partimos los datos en setenta, quince y quince por ciento. En entrenamiento aplicamos aumento de datos: recortes, volteos, rotaciones y cambios de brillo, que simulan cómo varía una ecografía real. Comparamos tres arquitecturas: una CNN entrenada desde cero, EfficientNet-B0 con transfer learning, y DeiT-Small, un Vision Transformer de Hugging Face. El Transformer divide la imagen en 196 parches y, con autoatención, relaciona cada parche con todos los demás: ve la lesión y su entorno al mismo tiempo.
>
> Elegimos el modelo con el mejor F1 en validación, y fue el Vision Transformer. En 109 imágenes que nunca vio, logra una exactitud del 92,7 por ciento, un F1 macro de 0,92 y un área bajo la curva ROC de 0,98. Lo más importante clínicamente: detecta 28 de 31 tumores malignos, un 90 por ciento de sensibilidad. La CNN desde cero se quedó en 0,55 de F1: con apenas 500 imágenes, el transfer learning es decisivo. También corregimos un error de inicialización en EfficientNet que le impedía aprender en la primera fase; al arreglarlo pasó del 82 al 87 por ciento.
>
> Esta es la aplicación desplegada en Streamlit Cloud. Elegimos una ecografía del conjunto de prueba: el modelo la clasifica como lesión maligna con 98 por ciento de confianza y muestra la probabilidad de cada clase. El mapa Grad-CAM indica en qué zonas se apoyó la red. Si una imagen supera el treinta por ciento de probabilidad de malignidad, la app la marca para revisión aunque la clase más probable sea otra. Podemos cambiar a EfficientNet, cuyo mapa se concentra más sobre la lesión. También analizamos varias imágenes por lotes, y cada predicción queda registrada en la base de datos SQLite, que se puede consultar desde la propia app.
>
> En conclusión, el Deep Learning con transfer learning permite clasificar ecografías con alta sensibilidad, siempre que los datos estén bien depurados. Es una herramienta educativa de apoyo: no reemplaza el criterio médico y necesita validación externa. El código, el notebook y la app están en los enlaces del póster. Muchas gracias.

---

## LISTA DE CONTROL ANTES DE ENTREGAR
- [ ] Duración entre 3:00 y 5:00 (este guion: ~4:20).
- [ ] Se ve la app en vivo y el póster.
- [ ] Subir el video (YouTube «no listado» o Google Drive «cualquiera con el enlace») y regenerar el póster:
  `.venv/Scripts/python entregables/generar_poster.py --video https://TU-VIDEO`
