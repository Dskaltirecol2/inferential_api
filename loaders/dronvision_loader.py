import os
from ultralytics import YOLO
from loaders.base_loader import BaseLoader


class DronVisionLoader(BaseLoader):
    PROJECT = "dronvision"

    def __init__(self, bucket: str):
        super().__init__(
            bucket=bucket,
            project_prefix=f"machinelearningprojects/{self.PROJECT}"
        )
        self.model_path=None
        self.model=None

    # -----------------------------------------------------
    # Override: load_models (solo YOLO)
    # -----------------------------------------------------
    def load_models(self):
        """
        Sobrescribe el load_models estándar:
        - carga config.json desde S3
        - descarga best.pt
        - carga YOLO
        """

        # 1. Leer config.json
        if self.config is None:
            self.load_config()

        # 2. Parsear entrada única del config
        entry = self.config[0]
        model_name = entry["name"]
        filename = os.path.basename(entry["path"])

        # Ruta en S3
        s3_path = f"{self.project_prefix}/models/{filename}"

        # Ruta local
        local_path = f"/tmp/{self.project_prefix.replace('/', '_')}_{filename}"

        self.model_path = local_path
        self.model = YOLO(self.model_path)
        # 3. Descargar desde S3
        self._download(s3_path, local_path)

        # 4. Cargar YOLO
        print(f"⚙️ Cargando modelo YOLO: {local_path}")
        model = YOLO(local_path)
        print("🚀 YOLO cargado, clases:", list(model.names.values()))

        # 5. Registrar como los otros loaders
        self.models = {
            model_name: {
                "model": model,
                "config": entry
            }
        }

        print(f"🔥 Modelo DronVision cargado correctamente: {model_name}")
