"""
Demo web del detector de fugas de gas en video OGI.

Despliegue: Render (Web Service, entorno Python).
    Build command:  pip install -r requirements.txt
    Start command:  python app.py

Archivos necesarios en el repositorio:
    app.py
    requirements.txt      -> ultralytics, gradio, opencv-python-headless
    best.pt               -> pesos entrenados
"""

import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO


RUTA_MODELO = Path(__file__).parent / "best.pt"
modelo = YOLO(str(RUTA_MODELO))

COLOR_CONTORNO = (0, 230, 90)
COLOR_RELLENO = (0, 180, 70)


# ---------------------------------------------------------------------------
# Clasificación de severidad
# ---------------------------------------------------------------------------

def clasificar_severidad(area_relativa, confianza):
    """
    Traduce el tamaño de la pluma y la confianza del modelo a una categoría
    operativa, alineada con el criterio de priorización de inspecciones.
    """
    if area_relativa > 0.085 and confianza > 0.55:
        return "ALTA", (40, 40, 220)
    if area_relativa > 0.030:
        return "MEDIA", (40, 150, 230)
    return "BAJA", (60, 190, 90)


# ---------------------------------------------------------------------------
# Anotación de un frame
# ---------------------------------------------------------------------------

def anotar(frame, resultado, umbral):
    """Dibuja máscaras, contornos y etiqueta de severidad sobre el frame."""
    salida = frame.copy()
    h, w = frame.shape[:2]
    detecciones = []

    if resultado.masks is None or len(resultado.boxes) == 0:
        cv2.putText(salida, "Sin fuga detectada", (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        return salida, detecciones

    capa = salida.copy()

    for mascara, caja in zip(resultado.masks.xy, resultado.boxes):
        conf = float(caja.conf[0])
        if conf < umbral:
            continue

        pts = np.asarray(mascara, dtype=np.int32)
        if len(pts) < 3:
            continue

        area_rel = cv2.contourArea(pts) / float(h * w)
        severidad, color = clasificar_severidad(area_rel, conf)

        cv2.fillPoly(capa, [pts], COLOR_RELLENO)
        cv2.polylines(salida, [pts], True, COLOR_CONTORNO, 2)

        x, y = pts[:, 0].min(), pts[:, 1].min()
        etiqueta = f"Fuga {conf:.0%}  ·  {severidad}"
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(salida, (x, max(0, y - th - 10)), (x + tw + 10, y), color, -1)
        cv2.putText(salida, etiqueta, (x + 5, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        detecciones.append({
            "confianza": conf,
            "area_relativa": area_rel,
            "severidad": severidad,
        })

    salida = cv2.addWeighted(capa, 0.28, salida, 0.72, 0)

    if not detecciones:
        cv2.putText(salida, "Sin fuga detectada", (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)

    return salida, detecciones


# ---------------------------------------------------------------------------
# Procesamiento de imagen
# ---------------------------------------------------------------------------

def procesar_imagen(imagen, umbral):
    if imagen is None:
        return None, "Sube una imagen para analizar."

    frame = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    resultado = modelo.predict(frame, conf=umbral, verbose=False)[0]
    anotada, detecciones = anotar(frame, resultado, umbral)

    if not detecciones:
        reporte = "### Resultado\n\nNo se detectaron fugas sobre el umbral configurado."
    else:
        filas = "\n".join(
            f"| {i+1} | {d['confianza']:.1%} | {d['area_relativa']:.2%} | **{d['severidad']}** |"
            for i, d in enumerate(detecciones)
        )
        peor = max(detecciones, key=lambda d: d["area_relativa"])["severidad"]
        reporte = (
            f"### Resultado\n\n"
            f"**{len(detecciones)} fuga(s) detectada(s)** · severidad máxima: **{peor}**\n\n"
            f"| # | Confianza | Área del encuadre | Severidad |\n"
            f"|---|-----------|-------------------|-----------|\n{filas}\n"
        )

    return cv2.cvtColor(anotada, cv2.COLOR_BGR2RGB), reporte


# ---------------------------------------------------------------------------
# Procesamiento de video
# ---------------------------------------------------------------------------

def procesar_video(ruta_video, umbral, progreso=gr.Progress()):
    if ruta_video is None:
        return None, "Sube un video para analizar."

    cap = cv2.VideoCapture(ruta_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    salida_path = "salida_anotada.mp4"
    escritor = cv2.VideoWriter(salida_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frames_con_fuga = 0
    severidades = []
    confianzas = []
    n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        resultado = modelo.predict(frame, conf=umbral, verbose=False)[0]
        anotada, detecciones = anotar(frame, resultado, umbral)
        escritor.write(anotada)

        if detecciones:
            frames_con_fuga += 1
            severidades.append(max(d["area_relativa"] for d in detecciones))
            confianzas.extend(d["confianza"] for d in detecciones)

        n += 1
        if n % 5 == 0:
            progreso(n / total, desc=f"Analizando frame {n}/{total}")

    cap.release()
    escritor.release()

    if n == 0:
        return None, "No se pudo leer el video."

    cobertura = frames_con_fuga / n
    if severidades:
        area_pico = max(severidades)
        sev_global, _ = clasificar_severidad(area_pico, float(np.mean(confianzas)))
        conf_media = float(np.mean(confianzas))
    else:
        area_pico, sev_global, conf_media = 0.0, "SIN DETECCIÓN", 0.0

    reporte = (
        f"### Reporte de inspección\n\n"
        f"| Métrica | Valor |\n|---|---|\n"
        f"| Frames analizados | {n} |\n"
        f"| Frames con fuga | {frames_con_fuga} |\n"
        f"| Persistencia de la fuga | {cobertura:.1%} del video |\n"
        f"| Área máxima de pluma | {area_pico:.2%} del encuadre |\n"
        f"| Confianza media | {conf_media:.1%} |\n"
        f"| **Severidad global** | **{sev_global}** |\n\n"
        f"La persistencia distingue una fuga sostenida de un falso positivo puntual: "
        f"una detección real aparece en una fracción alta de los frames."
    )

    return salida_path, reporte


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------

DESCRIPCION = """
# Detector de fugas de gas en video OGI

Segmentación de plumas de gas en imágenes de cámara termográfica OGI
(*Optical Gas Imaging*), entrenada con un dataset sintético generado
proceduralmente.

El modelo devuelve la máscara de la pluma, su confianza y una clasificación
de severidad basada en el área que ocupa en el encuadre.

**Nota:** entrenado sobre datos sintéticos como prueba de concepto. El
desempeño sobre video OGI real requiere reentrenamiento con imágenes de campo.
"""

with gr.Blocks(title="Detector de fugas OGI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPCION)

    umbral = gr.Slider(0.05, 0.95, value=0.25, step=0.05,
                       label="Umbral de confianza",
                       info="Más alto = menos detecciones, pero más seguras")

    with gr.Tab("Imagen"):
        with gr.Row():
            img_in = gr.Image(label="Imagen OGI", type="numpy")
            img_out = gr.Image(label="Detección")
        rep_img = gr.Markdown()
        btn_img = gr.Button("Analizar imagen", variant="primary")
        btn_img.click(procesar_imagen, [img_in, umbral], [img_out, rep_img])

    with gr.Tab("Video"):
        with gr.Row():
            vid_in = gr.Video(label="Video OGI")
            vid_out = gr.Video(label="Video anotado")
        rep_vid = gr.Markdown()
        btn_vid = gr.Button("Analizar video", variant="primary")
        btn_vid.click(procesar_video, [vid_in, umbral], [vid_out, rep_vid])

    gr.Markdown(
        "---\n"
        "Construido por Franyer López · "
        "[GitHub](https://github.com/franyer98) · "
        "[Portafolio](https://franyer98.github.io/Franyer-lopez)"
    )


if __name__ == "__main__":
    # Render asigna el puerto por variable de entorno
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_api=False,
    )
