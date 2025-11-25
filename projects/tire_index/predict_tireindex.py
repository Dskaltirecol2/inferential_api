# projects/tire_index/predict_tireindex.py

from __future__ import annotations

from typing import Dict, Any, Tuple, List

from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# -------------------------------------------------------------------
# 1. Helpers: DB, safe_div, etc.
# -------------------------------------------------------------------

def get_db_engine(db_settings: Dict[str, Any]) -> Engine:
    """
    Create a SQLAlchemy engine for MySQL using pymysql.
    db_settings must have: host, port, user, password, database
    """
    url = (
        f"mysql+pymysql://{db_settings['user']}:{db_settings['password']}@"
        f"{db_settings['host']}:{db_settings['port']}/{db_settings['database']}"
    )
    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=3600,
    )
    return engine


def safe_div(num: pd.Series, den: pd.Series, factor: float = 1.0) -> pd.Series:
    """Safe division with inf → NaN and optional scaling factor."""
    s = num / den
    s = s.replace([np.inf, -np.inf], np.nan)
    if factor != 1.0:
        s = s * factor
    return s


# -------------------------------------------------------------------
# 2. SQL queries (filtradas por marcaje y datetime de revisión)
# -------------------------------------------------------------------

SQL_INSPECCIONES = """
SELECT
    i.nrointerno,
    i.ajuste,
    i.codigop,
    i.rtdsfext,
    i.rtdsfint,
    i.kmtoms,
    i.horastoms,
    t.fecha_fin,
    t.horafin
FROM inspecciones i
LEFT JOIN trabajos t ON t.refe = i.refe
WHERE t.cliente = 'Glencore'
  AND i.nrointerno = :marcaje
  AND i.nrointerno IS NOT NULL
  AND i.nrointerno != ''
  AND i.rtdsfext > 0
  AND i.rtdsfint > 0
  AND CONCAT(t.fecha_fin, ' ', t.horafin) <= :dt_revision
"""

SQL_DESMONTAJES = """
SELECT 
    pa.fechadesmontaje, 
    pa.nro_seriei,
    pa.horafinal
FROM p_atendidas pa
LEFT JOIN trabajos t ON t.refe = pa.refe
WHERE t.cliente = 'Glencore'
  AND pa.tipotrabajo NOT IN ('Ajuste Presion','Retorque')
  AND pa.nro_seriei = :marcaje
  AND pa.nro_seriei IS NOT NULL
  AND pa.nro_seriei != ''
  AND pa.fechadesmontaje IS NOT NULL
  AND CONCAT(pa.fechadesmontaje, ' ', COALESCE(pa.horafinal, '00:00:00')) <= :dt_revision
"""

SQL_REPARACIONES = """
SELECT 
    r.fecha,
    r.horasalida as hora,
    r.serie as marcaje,
    pr.tipo,
    COALESCE(pr.longitud, 0) as longitud,
    COALESCE(pr.profundidad, 0) as profundidad,
    pr.refe 
FROM produc_repa pr
LEFT JOIN reparaciones r ON pr.refe = r.refe
WHERE r.cliente = 'Glencore'
  AND r.serie = :marcaje
  AND r.serie IS NOT NULL
  AND r.serie != ''
  AND r.fecha IS NOT NULL
  AND CONCAT(r.fecha, ' ', COALESCE(r.horasalida, '00:00:00')) <= :dt_revision
ORDER BY r.fecha, r.horasalida, r.serie
"""


# -------------------------------------------------------------------
# 3. Stats por llanta (equivalente a calcular_estadisticas)
# -------------------------------------------------------------------

def compute_revision_stats(
    marcaje: str,
    dt_revision: datetime,
    distancia_actual: float,
    horas_actual: float,
    engine: Engine
) -> Dict[str, Any]:
    """
    Reproduce la lógica de calcular_estadisticas pero solo para UNA revisión.
    Devuelve un dict con todas las columnas:
      - total_inspecciones, min_rtdsfext, ...
      - total_desmontajes, km_por_desmontaje, ...
      - total_reparaciones, numero_preventiva, ...
    """

    params = {
        "marcaje": marcaje,
        "dt_revision": dt_revision.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ----- INSPECCIONES -----
    with engine.connect() as conn:
        df_ins = pd.read_sql(text(SQL_INSPECCIONES), conn, params=params)

    if df_ins.empty:
        stats_ins = {
            "total_inspecciones": 0,
            "inspecciones_con_ajuste": 0,
            "inspecciones_codigop_2_3": 0,
            "min_rtdsfext": np.nan,
            "max_rtdsfext": np.nan,
            "min_rtdsfint": np.nan,
            "max_rtdsfint": np.nan,
            "max_kmtoms": np.nan,
            "max_horastoms": np.nan,
        }
    else:
        df_ins["ajuste"] = pd.to_numeric(df_ins["ajuste"], errors="coerce").fillna(0)
        df_ins["codigop"] = df_ins["codigop"].astype(str)

        for col in ["rtdsfext", "rtdsfint", "kmtoms", "horastoms"]:
            df_ins[col] = pd.to_numeric(df_ins[col], errors="coerce")

        stats_ins = {
            "total_inspecciones": int(len(df_ins)),
            "inspecciones_con_ajuste": int((df_ins["ajuste"] == 1).sum()),
            "inspecciones_codigop_2_3": int(df_ins["codigop"].isin(["2", "3"]).sum()),
            "min_rtdsfext": df_ins["rtdsfext"].min(),
            "max_rtdsfext": df_ins["rtdsfext"].max(),
            "min_rtdsfint": df_ins["rtdsfint"].min(),
            "max_rtdsfint": df_ins["rtdsfint"].max(),
            "max_kmtoms": df_ins["kmtoms"].max(),
            "max_horastoms": df_ins["horastoms"].max(),
        }

    # ----- DESMONTAJES -----
    with engine.connect() as conn:
        df_des = pd.read_sql(text(SQL_DESMONTAJES), conn, params=params)

    if not df_des.empty:
        df_des["datetime_desmontaje"] = pd.to_datetime(
            df_des["fechadesmontaje"].astype(str)
            + " "
            + df_des["horafinal"].fillna("00:00:00").astype(str),
            errors="coerce",
        )
        df_des = df_des[df_des["datetime_desmontaje"] <= dt_revision]

    num_desmontajes = len(df_des) if not df_des.empty else 0

    if (
        num_desmontajes > 0
        and distancia_actual is not None
        and not pd.isna(distancia_actual)
        and distancia_actual > 0
    ):
        km_por_desmontaje = distancia_actual / num_desmontajes
    else:
        km_por_desmontaje = np.nan

    if (
        num_desmontajes > 0
        and horas_actual is not None
        and not pd.isna(horas_actual)
        and horas_actual > 0
    ):
        horas_por_desmontaje = horas_actual / num_desmontajes
    else:
        horas_por_desmontaje = np.nan

    if horas_actual is not None and not pd.isna(horas_actual) and horas_actual > 0:
        km_por_hora = distancia_actual / horas_actual if distancia_actual is not None else np.nan
    else:
        km_por_hora = np.nan

    stats_des = {
        "total_desmontajes": int(num_desmontajes),
        "km_por_desmontaje": km_por_desmontaje,
        "horas_por_desmontaje": horas_por_desmontaje,
        "km_por_hora": km_por_hora,
    }

    # ----- REPARACIONES -----
    with engine.connect() as conn:
        df_rep = pd.read_sql(text(SQL_REPARACIONES), conn, params=params)

    if df_rep.empty:
        stats_rep = {
            "total_reparaciones": 0,
            "numero_preventiva": 0,
            "numero_correctiva": 0,
            "total_longitud": np.nan,
            "promedio_longitud": np.nan,
            "max_longitud": np.nan,
            "min_longitud": np.nan,
            "total_profundidad": np.nan,
            "promedio_profundidad": np.nan,
            "max_profundidad": np.nan,
            "min_profundidad": np.nan,
            "ratio_preventiva_correctiva": np.nan,
        }
    else:
        df_rep["longitud"] = pd.to_numeric(df_rep["longitud"], errors="coerce")
        df_rep["profundidad"] = pd.to_numeric(df_rep["profundidad"], errors="coerce")
        df_rep["tipo"] = df_rep["tipo"].astype(str).str.strip().str.lower()

        numero_preventiva = int(df_rep["tipo"].str.contains("preventiv", na=False).sum())
        numero_correctiva = int(df_rep["tipo"].str.contains("correctiv", na=False).sum())

        longitudes_validas = df_rep["longitud"].dropna()
        if not longitudes_validas.empty:
            total_longitud = float(longitudes_validas.sum())
            promedio_longitud = float(longitudes_validas.mean())
            max_longitud = float(longitudes_validas.max())
            min_longitud = float(longitudes_validas.min())
        else:
            total_longitud = promedio_longitud = max_longitud = min_longitud = np.nan

        profundidades_validas = df_rep["profundidad"].dropna()
        if not profundidades_validas.empty:
            total_profundidad = float(profundidades_validas.sum())
            promedio_profundidad = float(profundidades_validas.mean())
            max_profundidad = float(profundidades_validas.max())
            min_profundidad = float(profundidades_validas.min())
        else:
            total_profundidad = promedio_profundidad = max_profundidad = min_profundidad = np.nan

        if numero_correctiva > 0:
            ratio_pc = numero_preventiva / numero_correctiva
        else:
            ratio_pc = np.nan

        stats_rep = {
            "total_reparaciones": int(len(df_rep)),
            "numero_preventiva": numero_preventiva,
            "numero_correctiva": numero_correctiva,
            "total_longitud": total_longitud,
            "promedio_longitud": promedio_longitud,
            "max_longitud": max_longitud,
            "min_longitud": min_longitud,
            "total_profundidad": total_profundidad,
            "promedio_profundidad": promedio_profundidad,
            "max_profundidad": max_profundidad,
            "min_profundidad": min_profundidad,
            "ratio_preventiva_correctiva": ratio_pc,
        }

    # Merge all
    out = {}
    out.update(stats_ins)
    out.update(stats_des)
    out.update(stats_rep)
    return out


# -------------------------------------------------------------------
# 4. Feature engineering (versión para UNA fila)
# -------------------------------------------------------------------

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replica la lógica de FE clave del script de entrenamiento,
    pero pensada para un solo registro.
    Asume que df tiene las columnas crudas + stats ya calculadas.
    """
    # 3. Variables derivadas
    # --- desgaste ---
    df["desgaste"] = df["otd"] - df["rtd"]
    df["desgaste_por_km"] = safe_div(df["desgaste"], df["distancia"], factor=1000.0)
    df["desgaste_por_hora"] = safe_div(df["desgaste"], df["horas"])
    df["porcentaje_desgaste"] = safe_div(df["desgaste"], df["otd"], factor=100.0)

    # --- RTD ---
    df["rtd_avg_ext"] = (df["min_rtdsfext"].fillna(0) + df["max_rtdsfext"].fillna(0)) / 2
    df["rtd_avg_int"] = (df["min_rtdsfint"].fillna(0) + df["max_rtdsfint"].fillna(0)) / 2
    df["rtd_avg_total"] = (df["rtd_avg_ext"] + df["rtd_avg_int"]) / 2
    df["diff_rtd_ext_int"] = df["max_rtdsfext"].fillna(0) - df["max_rtdsfint"].fillna(0)
    df["desgaste_irregular"] = df["diff_rtd_ext_int"].abs()

    # --- Inspecciones ---
    df["inspecciones_por_km"] = safe_div(
        df["total_inspecciones"], df["max_kmtoms"], factor=1000.0
    )
    df["inspecciones_por_hora"] = safe_div(
        df["total_inspecciones"], df["max_horastoms"]
    )
    df["ratio_ajuste"] = safe_div(
        df["inspecciones_con_ajuste"], df["total_inspecciones"]
    )
    df["ratio_prioridad_alta"] = safe_div(
        df["inspecciones_codigop_2_3"], df["total_inspecciones"]
    )

    # --- Desmontajes ---
    df["ratio_desmont_km"] = safe_div(
        df["total_desmontajes"], df["max_kmtoms"], factor=1000.0
    )
    df["ratio_desmont_hora"] = safe_div(
        df["total_desmontajes"], df["max_horastoms"]
    )

    ef_uso = safe_div(df["km_por_desmontaje"], df["horas_por_desmontaje"])
    df["eficiencia_uso"] = ef_uso.fillna(df["km_por_hora"])

    # --- Reparaciones ---
    df["ratio_reparaciones_km"] = safe_div(
        df["total_reparaciones"], df["max_kmtoms"], factor=1000.0
    )
    df["porcentaje_preventiva"] = safe_div(
        df["numero_preventiva"], df["total_reparaciones"], factor=100.0
    )
    df["porcentaje_correctiva"] = safe_div(
        df["numero_correctiva"], df["total_reparaciones"], factor=100.0
    )

    # --- Mediciones ---
    df["variabilidad_longitud"] = df["max_longitud"] - df["min_longitud"]
    df["variabilidad_profundidad"] = df["max_profundidad"] - df["min_profundidad"]
    df["ratio_long_prof"] = safe_div(
        df["promedio_longitud"], df["promedio_profundidad"]
    )

    # Indicadores binarios: en entrenamiento se basan en cuantiles globales.
    # En inferencia online, si no guardaste esos umbrales en config,
    # los dejamos en 0 para no romper el pipeline.
    df["llanta_critica"] = 0
    df["alto_desgaste"] = 0
    df["requiere_atencion"] = 0
    df["alto_mantenimiento"] = 0
    df["desgaste_irregular_flag"] = 0

    # NaN → 0 en columnas semánticamente contables / tasas
    cols_zero_fill = [
        "total_inspecciones",
        "inspecciones_con_ajuste",
        "inspecciones_codigop_2_3",
        "inspecciones_por_km",
        "inspecciones_por_hora",
        "ratio_ajuste",
        "ratio_prioridad_alta",
        "total_desmontajes",
        "km_por_desmontaje",
        "horas_por_desmontaje",
        "ratio_desmont_km",
        "ratio_desmont_hora",
        "total_reparaciones",
        "numero_preventiva",
        "numero_correctiva",
        "ratio_reparaciones_km",
        "porcentaje_preventiva",
        "porcentaje_correctiva",
        "ratio_preventiva_correctiva",
        "eficiencia_uso",
        "llanta_critica",
        "alto_desgaste",
        "requiere_atencion",
        "alto_mantenimiento",
        "desgaste_irregular_flag",
    ]

    cols_zero_fill = [c for c in cols_zero_fill if c in df.columns]
    df[cols_zero_fill] = df[cols_zero_fill].fillna(0)

    # Limpiar infinitos si quedara algo
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


# -------------------------------------------------------------------
# 5. Función principal de predicción
# -------------------------------------------------------------------

def predict_tire_index(
    payload: Dict[str, Any],
    model,                     # sklearn Pipeline ya cargado
    model_config: Dict[str, Any],  # dict con feature_names, threshold, classes, etc.
    db_settings: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Predicción de CRITICO / DISPONIBLE para una revisión.
    
    payload esperado:
    {
      "marcaje": "1600023",
      "fecha": "2025-01-12",
      "hora": "14:35:00",
      "rtd": 63.4,
      "distancia": 24500,
      "horas": 320,
      "pos": 3,
      "otd": 105
    }
    """

    required = ["marcaje", "fecha", "hora", "rtd", "distancia", "horas", "pos", "otd"]
    missing_fields = [f for f in required if f not in payload]
    if missing_fields:
        return {
            "success": False,
            "error": f"Missing required fields: {missing_fields}",
            "error_code": "INVALID_INPUT",
        }

    marcaje = str(payload["marcaje"]).strip()

    try:
        dt_revision = datetime.strptime(
            f"{payload['fecha']} {payload['hora']}", "%Y-%m-%d %H:%M:%S"
        )
    except ValueError as e:
        return {
            "success": False,
            "error": f"Invalid fecha/hora format: {e}",
            "error_code": "INVALID_DATETIME",
        }

    # Coerción numérica de campos básicos
    def to_float(x, field):
        try:
            return float(x)
        except Exception:
            raise ValueError(f"Field '{field}' must be numeric")

    try:
        rtd = to_float(payload["rtd"], "rtd")
        distancia = to_float(payload["distancia"], "distancia")
        horas = to_float(payload["horas"], "horas")
        pos = int(payload["pos"])
        otd = to_float(payload["otd"], "otd")
    except ValueError as ve:
        return {
            "success": False,
            "error": str(ve),
            "error_code": "INVALID_INPUT_TYPE",
        }

    # --- DB + stats históricos ---
    engine = get_db_engine(
        {
            "host": db_settings["host"],
            "port": db_settings["port"],
            "user": db_settings["user"],
            "password": db_settings["password"],
            "database": db_settings["database"],
        }
    )

    try:
        stats = compute_revision_stats(
            marcaje=marcaje,
            dt_revision=dt_revision,
            distancia_actual=distancia,
            horas_actual=horas,
            engine=engine,
        )
    finally:
        engine.dispose()

    # --- Construir DataFrame de UNA fila con todo ---
    row = {
        "fecha": payload["fecha"],
        "hora": payload["hora"],
        "marcaje": marcaje,
        "pos": pos,
        "otd": otd,
        "rtd": rtd,
        "distancia": distancia,
        "horas": horas,
    }
    row.update(stats)

    df = pd.DataFrame([row])

    # FE
    df = apply_feature_engineering(df)

    # --- Seleccionar features en el orden correcto ---
    # En el config que generes desde entrenamiento deberías guardar esto:
    # model_config["feature_names"] o model_config["features"]
    feature_names: List[str] = (
        model_config.get("feature_names")
        or model_config.get("features")
        or model_config.get("data", {}).get("feature_names")
    )

    if not feature_names:
        return {
            "success": False,
            "error": "Model config does not define 'feature_names'.",
            "error_code": "INVALID_CONFIG",
        }

    # Asegurar que todas las columnas existen; si falta alguna, crearla con 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_names].astype(float)

    # --- Predicción ---
    # Si el pipeline es un clasificador con predict_proba y tienes threshold:
    threshold = model_config.get("decision_threshold", None)
    classes = model_config.get("classes", None)
    minority_idx = model_config.get("minority_idx", None)

    if hasattr(model, "predict_proba") and threshold is not None and classes is not None and minority_idx is not None:
        proba = model.predict_proba(X)[:, minority_idx]
        y_pred_int = (proba >= threshold).astype(int)
        # Mapear entero a etiqueta:
        # Si en entrenamiento minority_idx era el índice de 'CRITICO', ajusta aquí:
        label_minority = classes[minority_idx]
        label_majority = classes[1 - minority_idx]
        predicted_label = label_minority if y_pred_int[0] == 1 else label_majority
        score = float(proba[0])
    else:
        # Fallback genérico
        y_pred = model.predict(X)
        predicted_label = (
            y_pred[0]
            if classes is None
            else classes[int(y_pred[0])] if isinstance(y_pred[0], (int, np.integer)) else y_pred[0]
        )
        # Probabilidad opcional si existe
        if hasattr(model, "predict_proba"):
            proba_all = model.predict_proba(X)[0]
            score = float(np.max(proba_all))
        else:
            score = None

    return {
        "success": True,
        "error": None,
        "prediction": {
            "label": str(predicted_label),
            "score": score,
        },
        "input_meta": {
            "marcaje": marcaje,
            "fecha": payload["fecha"],
            "hora": payload["hora"],
            "pos": pos,
            "rtd": rtd,
            "distancia": distancia,
            "horas": horas,
            "otd": otd,
        },
        "model_info": {
            "name": model_config.get("name"),
            "version": model_config.get("version"),
            "best_experiment": model_config.get("best_experiment"),
        },
    }
