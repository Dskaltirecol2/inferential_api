from loaders.base_loader import BaseLoader

# IMPORTA LAS CLASES CUSTOM DEL PROYECTO
from custom_classes.ml_pipeline_classes import (
    CompleteMLPipeline,
    PredictionCalibrator
)

import __main__


class MonthlyPerformanceLoader(BaseLoader):

    PROJECT = "monthlyperformance"

    def __init__(self, bucket: str):

        # REGISTRO DE CLASES – MOMENTO EXACTO
        __main__.CompleteMLPipeline = CompleteMLPipeline
        __main__.PredictionCalibrator = PredictionCalibrator

        # AHORA sí inicializamos el Loader
        super().__init__(
            bucket=bucket,
            project_prefix=f"machinelearningprojects/{self.PROJECT}"
        )
