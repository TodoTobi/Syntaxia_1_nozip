# api_client/yolo_client.py
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Any, List

import cv2
from ultralytics import YOLO

from modelado_3d.generar_modelo import generar_modelo_3d_desde_imagen


# --- rutas y modelo ----------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent  # raíz del proyecto
MODEL_PATH = ROOT / "yolov5su.pt"              # ajustá si tu .pt está en otro lado

MODELOS3D_DIR = ROOT / "data" / "modelos3d"
MODELOS3D_DIR.mkdir(parents=True, exist_ok=True)

model = None
_modelo_error = None


def _cargar_modelo() -> YOLO:
    """Carga el modelo YOLO una sola vez, con validaciones claras."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo YOLO en: {MODEL_PATH}\n"
            "Asegurate de que 'yolov5su.pt' esté en la raíz del proyecto o actualiza la ruta."
        )
    return YOLO(str(MODEL_PATH))


try:
    model = _cargar_modelo()
    print(f"✔ Modelo YOLO cargado desde: {MODEL_PATH}")
except Exception as e:
    model = None
    _modelo_error = e
    print(f"❌ Error cargando modelo YOLO: {e}")


# --- helpers -----------------------------------------------------------------

def _clase_principal(objetos: List[Dict[str, Any]]) -> str:
    """
    Devuelve la clase principal de la imagen.
    Estrategia: la detección con mayor confianza.
    """
    if not objetos:
        return ""
    top = max(objetos, key=lambda o: o.get("confianza", 0.0))
    return (top.get("clase") or "").strip()


# --- lógica principal --------------------------------------------------------

def analizar_imagen_yolo(path_imagen: str) -> Dict[str, Any]:
    """
    Analiza una imagen con YOLO y (opcionalmente) genera un .obj básico.
    Devuelve siempre un dict JSON-friendly (sin lanzar excepciones).
    """
    try:
        img_path = Path(path_imagen).resolve()
        if not img_path.exists():
            return {
                "descripcion": "No se detectaron objetos.",
                "respuesta": f"No se pudo leer la imagen: {img_path}",
                "objetos": []
            }

        if model is None:
            return {
                "descripcion": "No se detectaron objetos.",
                "respuesta": f"Error cargando modelo YOLO: {_modelo_error}",
                "objetos": []
            }

        # OpenCV
        img = cv2.imread(str(img_path))
        if img is None:
            return {
                "descripcion": "No se detectaron objetos.",
                "respuesta": "La imagen no pudo ser decodificada (formato no soportado o archivo corrupto).",
                "objetos": []
            }

        # Predicción
        results = model.predict(img, verbose=False)
        if not results:
            return {
                "descripcion": "No se detectaron objetos.",
                "respuesta": "El modelo no devolvió resultados.",
                "objetos": []
            }

        r = results[0]
        objetos_detectados: List[Dict[str, Any]] = []
        names = getattr(r, "names", getattr(model, "names", {}))

        if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
            for box in r.boxes:
                # cls y conf son tensores; .item() cuando aplica
                cls_idx = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
                conf = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])
                clase = names.get(cls_idx, str(cls_idx))
                objetos_detectados.append({
                    "clase": clase,
                    "confianza": round(conf * 100, 2)
                })

        if not objetos_detectados:
            return {
                "descripcion": "No se detectaron objetos.",
                "respuesta": "No se encontró ningún objeto relevante en la imagen.",
                "objetos": []
            }

        # Resumen (todas las clases) + principal (para el modelo 3D)
        clases_unicas = sorted({obj["clase"] for obj in objetos_detectados})
        descripcion = ", ".join(clases_unicas)
        respuesta = f"Se detectaron los siguientes objetos: {descripcion}."
        clase_top = _clase_principal(objetos_detectados)  # p.ej. 'laptop'

        # Intentar generar modelo 3D usando la clase principal
        modelo_url = None
        try:
            base_nombre = (clase_top or "objeto").lower().replace(" ", "_")
            nombre_archivo = f"{base_nombre}_{random.randint(1000, 9999)}.obj"
            ruta_modelo = MODELOS3D_DIR / nombre_archivo

            # Preferimos pasar la clase al generador (MVP con placeholders).
            try:
                generar_modelo_3d_desde_imagen(
                    str(img_path),
                    salida_obj=str(ruta_modelo),
                    clase_objeto=clase_top,  # <-- clave para elegir placeholder
                )
            except TypeError:
                # Compatibilidad si tu función aún no acepta clase_objeto
                generar_modelo_3d_desde_imagen(str(img_path), salida_obj=str(ruta_modelo))

            modelo_url = f"/modelos/{nombre_archivo}"
        except Exception as gen_err:
            print(f"⚠ Error en generación 3D: {gen_err}")
            modelo_url = None

        return {
            "descripcion": descripcion,
            "respuesta": respuesta,
            "objetos": objetos_detectados,
            "modelo_url": modelo_url
        }

    except Exception as e:
        print(f"❌ Error inesperado en YOLO: {e}")
        return {
            "descripcion": "No se detectaron objetos.",
            "respuesta": f"Error interno en YOLO: {e}",
            "objetos": []
        }
    