import os
import pickle
import joblib
import __main__

from loaders.base_loader import BaseLoader
from custom_classes.lifeexpectancy_classes import (
    LlantaPreprocessor,
    _CompatUnpickler
)


class LifeExpectancyLoader(BaseLoader):
    PROJECT = "lifeexpectancy"

    def __init__(self, bucket: str):
        super().__init__(
            bucket=bucket,
            project_prefix=f"machinelearningprojects/{self.PROJECT}"
        )

        # Register custom classes for pickle compatibility
        __main__.LlantaPreprocessor = LlantaPreprocessor
        __main__._CompatUnpickler = _CompatUnpickler

        self.model = None
        self.preprocessor = None
        self.metadata = None  # Optional, for future support

    # ---------------------------
    # Load model
    # ---------------------------
    def load_model(self):
        model_s3_paths = [
            f"{self.project_prefix}/models/rf_best_final.joblib",
            f"{self.project_prefix}/models/rf_best_final.pkl"
        ]

        for s3_path in model_s3_paths:
            try:
                local_path = f"/tmp/{os.path.basename(s3_path)}"
                self._download(s3_path, local_path)

                try:
                    self.model = joblib.load(local_path)
                    print(f"✓ Modelo cargado: {local_path} (joblib)")
                    return
                except Exception as e1:
                    print(f"⚠ joblib.load falló: {e1}")

                # Intentar pickle fallback
                try:
                    with open(local_path, "rb") as f:
                        self.model = _CompatUnpickler(f).load()
                    print(f"✓ Modelo cargado: {local_path} (pickle compat)")
                    return
                except Exception as e2:
                    print(f"❌ pickle fallback falló: {e2}")

            except Exception as dl_err:
                print(f"⚠ No se pudo descargar {s3_path}: {dl_err}")

        raise FileNotFoundError("No se encontró ningún modelo válido en S3.")

    # ---------------------------
    # Load preprocessor
    # ---------------------------
    def load_preprocessor(self):
        pre_s3_paths = [
            f"{self.project_prefix}/models/preprocessor.joblib",
            f"{self.project_prefix}/models/preprocessor.pkl"
        ]

        for s3_path in pre_s3_paths:
            try:
                local_path = f"/tmp/{os.path.basename(s3_path)}"
                self._download(s3_path, local_path)

                try:
                    self.preprocessor = joblib.load(local_path)
                    print(f"✓ Preprocessor cargado: {local_path} (joblib)")
                    return
                except Exception as e1:
                    print(f"⚠ joblib.load preprocessor falló: {e1}")

                # pickle fallback
                try:
                    with open(local_path, "rb") as f:
                        self.preprocessor = _CompatUnpickler(f).load()
                    print(f"✓ Preprocessor cargado: {local_path} (pickle compat)")
                    return
                except Exception as e2:
                    print(f"❌ pickle fallback falló: {e2}")

            except Exception as dl_err:
                print(f"⚠ No se pudo descargar {s3_path}: {dl_err}")

        raise FileNotFoundError("No se encontró preprocessor válido en S3.")

    # ---------------------------
    # Optional: Load metadata.json (future)
    # ---------------------------
    def load_metadata(self):
        metadata_s3 = f"{self.project_prefix}/config/metadata.json"
        local_meta = "/tmp/metadata.json"

        try:
            self._download(metadata_s3, local_meta)
            import json
            self.metadata = json.load(open(local_meta, "r"))
            print("✓ metadata.json cargado.")
        except Exception:
            print("ℹ metadata.json no encontrado (opcional).")

    # ---------------------------
    # Load all (model + preprocessor + metadata)
    # ---------------------------
    def load_models(self):
        print("Cargando modelo y preprocessor de lifeexpectancy...")
        self.load_model()
        self.load_preprocessor()
        self.load_metadata()  # optional
        print("🔥 modelos lifeexpectancy cargados.")
