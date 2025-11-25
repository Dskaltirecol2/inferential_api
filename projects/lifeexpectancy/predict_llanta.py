import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime

from custom_classes.lifeexpectancy_classes import LlantaPreprocessor
from projects.lifeexpectancy.sql_query import QUERY_PREDICCION


# ---------------------------------------------------------
# Normalize Colombian numeric formats
# ---------------------------------------------------------
def normalize_kms_value(x):
    import re

    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    if s == "":
        return np.nan

    # decimal comma
    if "," in s and "." not in s:
        try:
            return float(s.replace(",", "."))
        except:
            return np.nan

    # thousands: 1.234.567 → 1234567
    if s.count(".") > 1:
        try:
            return float(s.replace(".", ""))
        except:
            return np.nan

    # ambiguous decimal/thousands
    if s.count(".") == 1:
        left, right = s.split(".", 1)
        if re.fullmatch(r"\d{3}", right):
            # thousands
            try:
                return float(left + right)
            except:
                return np.nan

        try:
            return float(s)
        except:
            return np.nan

    # fallback: integer-like
    try:
        return float(s)
    except:
        return np.nan


# ---------------------------------------------------------
# Estado de la llanta basado en % vida consumida
# ---------------------------------------------------------
def get_estado_llanta(porcentaje_vida):
    if porcentaje_vida >= 90:
        return {
            "codigo": "CRITICO",
            "color": "red",
            "recomendacion": "Retirar inmediatamente. Riesgo alto de falla."
        }
    if porcentaje_vida >= 75:
        return {
            "codigo": "ADVERTENCIA",
            "color": "yellow",
            "recomendacion": "Monitorear y planificar reemplazo."
        }
    if porcentaje_vida >= 50:
        return {
            "codigo": "NORMAL",
            "color": "green",
            "recomendacion": "Estado aceptable. Inspecciones regulares."
        }
    return {
        "codigo": "EXCELENTE",
        "color": "green",
        "recomendacion": "Llanta en óptimas condiciones."
    }


# ---------------------------------------------------------
# Conexión a BD (puede venir de settings.py más adelante)
# ---------------------------------------------------------
def get_db_engine(db_settings):
    """
    db_settings = {
        "host": "...",
        "port": 3306,
        "user": "...",
        "password": "...",
        "database": "..."
    }
    """
    return create_engine(
        f"mysql+pymysql://{db_settings['user']}:{db_settings['password']}@"
        f"{db_settings['host']}:{db_settings['port']}/{db_settings['database']}",
        pool_pre_ping=True,
        pool_recycle=3600
    )


# ---------------------------------------------------------
# FUNCIÓN PRINCIPAL DE PREDICCIÓN
# ---------------------------------------------------------
def predict_llanta(nrointerno: str, loader, db_settings: dict):
    """
    Ejecuta:
        - Consulta SQL
        - Preprocesamiento
        - Predicción del modelo
        - Cálculos derivados
        - Construcción del resultado
    
    Params:
        nrointerno: str → ID de llanta
        loader: LifeExpectancyLoader instance (ya con modelo y preprocessor cargados)
        db_settings: dict → parámetros de conexión MySQL

    Returns:
        dict → respuesta formateada lista para FastAPI
    """

    try:
        # ----------------------------
        # 1. validar input
        # ----------------------------
        if not nrointerno or str(nrointerno).strip() == "":
            return {
                "success": False,
                "error": "nrointerno vacío o inválido",
                "error_code": "INVALID_INPUT"
            }

        nrointerno = str(nrointerno).strip()

        model = loader.model
        preprocessor = loader.preprocessor

        # ----------------------------
        # 2. Ejecutar query SQL
        # ----------------------------
        engine = get_db_engine(db_settings)

        with engine.connect() as conn:
            result = conn.execute(
                text(QUERY_PREDICCION),
                {"id_llanta": nrointerno}
            )
            df_llanta = pd.DataFrame(result.fetchall(), columns=result.keys())

        # ----------------------------
        # 3. Validar existencia
        # ----------------------------
        if df_llanta.empty:
            return {
                "success": False,
                "error": f"llanta {nrointerno} no encontrada o no cumple condiciones",
                "error_code": "NOT_FOUND"
            }

        # ----------------------------
        # 4. Extraer información básica
        # ----------------------------
        info_llanta = {
            "id_llanta": nrointerno,
            "serial_toms": str(df_llanta["serialtoms"].iloc[0]),
            "modelo_flota": str(df_llanta["modelo_flota"].iloc[0]),
            "id_equipo": str(df_llanta["id_equipo"].iloc[0]),
            "posicion": str(df_llanta["posicion"].iloc[0]),
            "componente": str(df_llanta["componente"].iloc[0]),
            "fecha_montaje": str(df_llanta["fecha_montaje"].iloc[0]),
            "fecha_primera_inspeccion_hallazgo": str(df_llanta["fecha_penultima_insp"].iloc[0]),
        }

        # ----------------------------
        # 5. Normalizar kms actuales
        # ----------------------------
        raw_kms = df_llanta["total_kms_llanta"].iloc[0]
        kms_actuales = normalize_kms_value(raw_kms)

        if pd.isna(kms_actuales):
            tmp = pd.to_numeric(raw_kms, errors="coerce")
            kms_actuales = float(tmp) if not pd.isna(tmp) else 0.0

        # ----------------------------
        # 6. Preprocessing
        # ----------------------------
        df_processed = preprocessor.preprocess(df_llanta)

        # ----------------------------
        # 7. Predicción
        # ----------------------------
        pred_kms = float(model.predict(df_processed)[0])

        kms_restantes = pred_kms - kms_actuales
        porcentaje_vida = (kms_actuales / pred_kms) * 100 if pred_kms != 0 else 0.0

        estado = get_estado_llanta(porcentaje_vida)

        # ----------------------------
        # 8. Respuesta final
        # ----------------------------
        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "llanta": info_llanta,
            "prediccion": {
                "kms_totales_estimados": round(pred_kms, 2),
                "kms_actuales": round(kms_actuales, 2),
                "kms_restantes": round(kms_restantes, 2),
                "porcentaje_vida_consumida": round(porcentaje_vida, 2),
                "dias_desde_montaje": int(df_llanta["diasprimerfalla"].iloc[0])
            },
            "estado": estado,
            "metadata": {
                "modelo_version": loader.metadata["model_info"]["version"]
                if loader.metadata else "v1.0",
                "modelo_tipo": loader.metadata["model_info"]["model_type"]
                if loader.metadata else "RandomForest",
                "modelo_origen": "s3"
            }
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "error_code": "INTERNAL_ERROR",
            "traceback": traceback.format_exc()
        }
