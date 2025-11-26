# projects/dronvision/predict_dronvision.py

import base64
import numpy as np
import cv2
from ultralytics import YOLO
from datetime import datetime
import threading
import os

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from ftplib import FTP_TLS


# ======================================================================
# 1. SQLAlchemy Engine (una vez por proyecto)
# ======================================================================

def get_db_engine(db_settings: dict):
    """
    Creates a SQLAlchemy Engine with connection pooling.
    """
    url = (
        f"mysql+pymysql://{db_settings['user']}:{db_settings['password']}@"
        f"{db_settings['host']}:{db_settings['port']}/{db_settings['database']}"
    )

    return create_engine(
        url,
        pool_pre_ping=True,     # detect dead connections
        pool_recycle=3600,      # avoid timeout disconnections
        pool_size=5,
        max_overflow=10
    )


# ======================================================================
# 2. Insertar alerta en DB (thread-safe)
# ======================================================================

def insert_alert_db(fecha_evento, tipo_evento, filename, cliente, db_settings):
    """
    Inserts a new alert into MySQL using SQLAlchemy.
    Safe to run inside a thread.
    """
    try:
        engine = get_db_engine(db_settings)

        with engine.begin() as conn:
            query = text("""
                INSERT INTO alertas_sis (fecha_evento, tipo_evento, foto, cliente)
                VALUES (:fecha_evento, :tipo_evento, :foto, :cliente)
            """)

            conn.execute(query, {
                "fecha_evento": fecha_evento,
                "tipo_evento": tipo_evento,
                "foto": f"foto_alertas_sis/{filename}",
                "cliente": cliente
            })

        print(f"✅ Insertada alerta en DB: {filename}")

    except SQLAlchemyError as e:
        print(f"❌ Error DB: {str(e)}")


# ======================================================================
# 3. Subida a FTP (thread-safe)
# ======================================================================

def upload_ftp(local_path, filename, ftp_config):
    """
    Uploads file to FTP securely and deletes local file after.
    """
    try:
        ftp = FTP_TLS()
        ftp.connect(ftp_config["host"], int(ftp_config["port"]))
        ftp.login(ftp_config["user"], ftp_config["password"])
        ftp.prot_p()

        with open(local_path, "rb") as f:
            ftp.storbinary(f"STOR {filename}", f)

        ftp.quit()
        print(f"📤 Subido a FTP: {filename}")

    except Exception as e:
        print(f"❌ Error FTP: {e}")

    finally:
        try:
            os.remove(local_path)
            print(f"🗑️ Archivo local eliminado: {local_path}")
        except:
            pass


# ======================================================================
# 4. Clasificación del evento según las clases detectadas
# ======================================================================

def determinar_evento(clases: list):
    s = set(clases)

    if "bultos" in s:
        return "Botadero"

    if "pala" in s:
        if "equipo_soporte" in s:
            return "Zona de carga con equipo de soporte"
        else:
            return "Zona de carga sin equipo de soporte"

    return None


# ======================================================================
# 5. FUNCIÓN PRINCIPAL DE PREDICCIÓN (USADA POR FASTAPI)
# ======================================================================

def predict_dronvision(
    payload: dict,
    loader,
    db_settings: dict,
    ftp_config: dict
):
    """
    Ejecuta:
      - Decodificación base64
      - Inferencia YOLO
      - Determinación de evento
      - Guardado local de imagen
      - Subida a FTP (thread)
      - Registro en DB (thread)
    """

    # ---------------------------------------------------------
    # Validación mínima
    # ---------------------------------------------------------
    if "image_base64" not in payload:
        return {"success": False, "error": "Debe enviar image_base64"}

    cliente = payload.get("cliente", "First Quantum")

    # ---------------------------------------------------------
    # Decodificar imagen
    # ---------------------------------------------------------
    try:
        img_bytes = base64.b64decode(payload["image_base64"])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return {"success": False, "error": "Error decodificando imagen"}

    # ---------------------------------------------------------
    # Cargar modelo YOLO (rápido si ya está en RAM)
    # ---------------------------------------------------------
    model = loader.model

    # ---------------------------------------------------------
    # Inferencia
    # ---------------------------------------------------------
    results = model.predict(image, conf=0.25, save=False)
    result = results[0]

    clases_detectadas = [model.names[int(b.cls[0])] for b in result.boxes]
    tipo_evento = determinar_evento(clases_detectadas)

    # ---------------------------------------------------------
    # No hay evento → no hacemos nada más
    # ---------------------------------------------------------
    if tipo_evento is None:
        return {
            "success": True,
            "alerta": False,
            "evento": None,
            "clases_detectadas": clases_detectadas
        }

    # ---------------------------------------------------------
    # Dibujar bounding boxes YOLO
    # ---------------------------------------------------------
    img_boxes = result.plot()

    # ---------------------------------------------------------
    # Guardar local en /tmp/
    # ---------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"alerta_{timestamp}.jpg"
    local_path = f"/tmp/{filename}"
    fecha_evento = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cv2.imwrite(local_path, img_boxes)

    # ---------------------------------------------------------
    # Procesos asíncronos: DB + FTP
    # ---------------------------------------------------------
    threading.Thread(
        target=insert_alert_db,
        args=(fecha_evento, tipo_evento, filename, cliente, db_settings),
        daemon=True
    ).start()

    threading.Thread(
        target=upload_ftp,
        args=(local_path, filename, ftp_config),
        daemon=True
    ).start()

    # ---------------------------------------------------------
    # Respuesta a API
    # ---------------------------------------------------------
    return {
        "success": True,
        "alerta": True,
        "evento": tipo_evento,
        "clases_detectadas": clases_detectadas,
        "archivo": filename,
        "timestamp": fecha_evento
    }
