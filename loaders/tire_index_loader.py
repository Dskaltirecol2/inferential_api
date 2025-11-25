import os
import json
import joblib
import __main__

from loaders.base_loader import BaseLoader


class TireIndexLoader(BaseLoader):
    """
    Loader para el proyecto TIRE INDEX.
    Estructura en S3:
        machinelearningprojects/tireindex/
            models/
                best_model.joblib   (o .pkl)
            config/
                config.json
    """

    PROJECT = "tireindex"

    def __init__(self, bucket: str):
        super().__init__(
            bucket=bucket,
            project_prefix=f"machinelearningprojects/{self.PROJECT}"
        )

        self.model = None          # sklearn Pipeline
        self.config = None         # dict del config.json

        # Registrar clases custom si en el futuro las usas
        # (Aquí normalmente no se necesitan, pero mantenemos el patrón)
        __main__.TireIndexLoader = TireIndexLoader

    # ---------------------------------------------------------
    # 🧠 Load MODEL
    # ---------------------------------------------------------
    def load_model(self):
        """
        Busca y carga el modelo del proyecto TireIndex.
        Prioriza joblib > pkl.
        """
        model_candidates = [
            f"{self.project_prefix}/models/best_balanced_model_top20.joblib",
            f"{self.project_prefix}/models/best_balanced_model_top20.pkl",
            f"{self.project_prefix}/models/tire_index_model.joblib",
            f"{self.project_prefix}/models/tire_index_model.pkl",
        ]

        for s3_path in model_candidates:
            try:
                local_path = f"/tmp/{os.path.basename(s3_path)}"
                self._download(s3_path, local_path)

                try:
                    self.model = joblib.load(local_path)
                    print(f"✓ Modelo TireIndex cargado con joblib: {local_path}")
                    return
                except Exception as e:
                    print(f"⚠ joblib.load falló ({e}). Intentando pickle directo...")

                    import pickle
                    with open(local_path, "rb") as f:
                        self.model = pickle.load(f)

                    print(f"✓ Modelo TireIndex cargado con pickle compat: {local_path}")
                    return

            except Exception:
                continue

        raise FileNotFoundError("❌ No se encontró ningún modelo válido en S3 para TireIndex.")

    # ---------------------------------------------------------
    # 📘 Load CONFIG
    # ---------------------------------------------------------
    def load_config(self):
        """
        Carga el config.json con:
        - feature_names
        - classes
        - minority_idx
        - decision_threshold
        - version, best_experiment, etc.
        """
        config_paths = [
            f"{self.project_prefix}/config/config.json",
            f"{self.project_prefix}/config/metadata.json",
        ]

        for s3_path in config_paths:
            try:
                local_path = f"/tmp/{os.path.basename(s3_path)}"
                self._download(s3_path, local_path)

                with open(local_path, "r") as f:
                    self.config = json.load(f)

                print(f"✓ Config TireIndex cargado desde {s3_path}")
                return

            except Exception:
                continue

        raise FileNotFoundError("❌ No existe ningún config.json / metadata.json en S3 para TireIndex.")

    # ---------------------------------------------------------
    # 🚀 Load BOTH
    # ---------------------------------------------------------
    def load_models(self):
        print("🚛 Cargando modelo + config de TireIndex...")
        self.load_model()
        self.load_config()
        print("🔥 TireIndex cargado completamente.")
