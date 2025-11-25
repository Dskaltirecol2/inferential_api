import os
import json
import joblib
from typing import Dict

from core.s3_client import get_s3_client
from core.utils import ensure_local_dir


class BaseLoader:
    """
    Loader base para cualquier proyecto.
    - Descarga config.json desde S3
    - Detecta los modelos definidos allí
    - Descarga los modelos
    - Los carga en RAM
    """

    def __init__(self, bucket: str, project_prefix: str):
        self.bucket = bucket
        self.project_prefix = project_prefix   # ej: machinelearningprojects/monthlyperformance
        self.s3 = get_s3_client()

        self.models: Dict[str, object] = {}    # { model_name: {model, config} }
        self.config = None                     # contenido de config.json

    # -----------------------------------------------------
    # DESCARGA ARCHIVOS DESDE S3
    # -----------------------------------------------------
    def _download(self, s3_path: str, local_path: str):
        """Descarga un archivo desde S3 a local (asegura carpeta target)."""
        ensure_local_dir(local_path)

        print(f"↓ Descargando desde S3: s3://{self.bucket}/{s3_path}")
        self.s3.download_file(self.bucket, s3_path, local_path)

        return local_path

    # -----------------------------------------------------
    # CARGA DEL CONFIG JSON
    # -----------------------------------------------------
    def load_config(self):
        """Descarga y carga el config.json del proyecto."""
        config_s3_path = f"{self.project_prefix}/config/config.json"
        local_config_path = f"/tmp/{self.project_prefix.replace('/', '_')}_config.json"

        self._download(config_s3_path, local_config_path)

        with open(local_config_path, "r") as f:
            self.config = json.load(f)

        print(f"✓ Config cargado para proyecto: {self.project_prefix}")

    # -----------------------------------------------------
    # CARGA DE TODOS LOS MODELOS DEFINIDOS EN CONFIG.JSON
    # -----------------------------------------------------
    def load_models(self):
        """Carga todos los modelos definidos en config.json."""
        if self.config is None:
            self.load_config()

        for entry in self.config:

            model_name = entry["name"]
            load_method = entry["load_method"]

            filename = os.path.basename(entry["path"])

            # from 'kms_prediction_model_190TON' extract last part -> '190TON'
            folder_name = model_name.split("_")[-1]

            # Ruta en S3
            s3_model_path = f"{self.project_prefix}/{folder_name}/{filename}"

            # Ruta local temporal
            local_model_path = f"/tmp/{self.project_prefix.replace('/', '_')}_{filename}"

            # Descargar
            self._download(s3_model_path, local_model_path)

            # -----------------------------------------------------
            # REGISTRO DE CLASES CUSTOM (solo si existen)
            # Esto es crítico para Windows (multiproceso) y joblib.
            # -----------------------------------------------------
            try:
                import __main__
                from custom_classes.ml_pipeline_classes import (
                    CompleteMLPipeline,
                    PredictionCalibrator
                )
                __main__.CompleteMLPipeline = CompleteMLPipeline
                __main__.PredictionCalibrator = PredictionCalibrator
                print("✓ Clases custom registradas correctamente.")
            except ImportError:
                # Proyectos sin clases custom entran aquí
                pass

            # -----------------------------------------------------
            # Cargar modelo
            # -----------------------------------------------------
            if load_method == "joblib":
                model = joblib.load(local_model_path)
            elif load_method == "pickle":
                import pickle
                with open(local_model_path, "rb") as f:
                    model = pickle.load(f)
            else:
                raise ValueError(f"Load method '{load_method}' no está soportado.")

            # Guardar modelo ya cargado
            self.models[model_name] = {
                "model": model,
                "config": entry
            }

            print(f"✓ Modelo cargado: {model_name}")

        print(f"🔥 Todos los modelos cargados para el proyecto: {self.project_prefix}")
