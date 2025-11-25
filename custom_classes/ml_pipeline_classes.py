# ml_pipeline_classes.py - Clases necesarias para cargar tus modelos

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

class PredictionCalibrator(BaseEstimator, TransformerMixin):
    """
    Calibrador personalizado que puede usar diferentes métodos de calibración.
    """
    def __init__(self, method='linear', degree=2):
        self.method = method
        self.degree = degree
        self.calibrator = None
        self.use_calibration = False
        
    def fit(self, y_pred_train, y_true_train):
        """
        Entrena el calibrador basándose en predicciones vs valores reales del conjunto de entrenamiento.
        """
        residuals = y_true_train - y_pred_train
        
        # Solo calibrar si hay suficiente error sistemático
        residual_std = np.std(residuals)
        residual_mean_abs = np.abs(np.mean(residuals))
        
        if residual_mean_abs > 0.1 * residual_std:  # Umbral de calibración
            self.use_calibration = True
            
            if self.method == 'linear':
                self.calibrator = LinearRegression()
                self.calibrator.fit(y_pred_train.reshape(-1, 1), residuals)
                
            elif self.method == 'polynomial':
                self.calibrator = Pipeline([
                    ("poly", PolynomialFeatures(degree=self.degree)),
                    ("linreg", LinearRegression())
                ])
                self.calibrator.fit(y_pred_train.reshape(-1, 1), residuals)
                
            elif self.method == 'isotonic':
                self.calibrator = IsotonicRegression(out_of_bounds="clip")
                self.calibrator.fit(y_pred_train, y_true_train)
                
            print(f"✅ Calibración {self.method} activada (error sistemático detectado)")
        else:
            print("ℹ️ Sin calibración necesaria (error sistemático mínimo)")
            
        return self
    
    def transform(self, y_pred):
        """
        Aplica la calibración a las predicciones.
        """
        if not self.use_calibration or self.calibrator is None:
            return y_pred
            
        if self.method in ['linear', 'polynomial']:
            correction = self.calibrator.predict(y_pred.reshape(-1, 1))
            return y_pred + correction
        elif self.method == 'isotonic':
            return self.calibrator.predict(y_pred)
        else:
            return y_pred

class CompleteMLPipeline(BaseEstimator):
    """
    Pipeline completo que incluye preprocesamiento, modelo y calibración.
    """
    def __init__(self, model, feature_names, calibration_method='linear', scale_features=True):
        self.model = model
        self.feature_names = feature_names
        self.calibration_method = calibration_method
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.calibrator = PredictionCalibrator(method=calibration_method)
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Entrena el pipeline completo.
        """
        # Validar columnas
        X_processed = self._validate_and_process_input(X)
        
        # Escalar si es necesario
        if self.scaler:
            X_scaled = self.scaler.fit_transform(X_processed)
        else:
            X_scaled = X_processed
            
        # Entrenar modelo principal
        self.model.fit(X_scaled, y)
        
        # Obtener predicciones de entrenamiento para calibración
        y_pred_train = self.model.predict(X_scaled)
        
        # Entrenar calibrador
        self.calibrator.fit(y_pred_train, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Realiza predicciones con el pipeline completo.
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
            
        # Procesar entrada
        X_processed = self._validate_and_process_input(X)
        
        # Escalar si es necesario
        if self.scaler:
            X_scaled = self.scaler.transform(X_processed)
        else:
            X_scaled = X_processed
            
        # Predicción del modelo base
        y_pred = self.model.predict(X_scaled)
        
        # Aplicar calibración
        y_pred_calibrated = self.calibrator.transform(y_pred)
        
        return y_pred_calibrated
    
    def _validate_and_process_input(self, X):
        """
        Valida y procesa la entrada para asegurar consistencia.
        """
        if isinstance(X, dict):
            # Si se pasa un diccionario, convertir a DataFrame
            X = pd.DataFrame([X])
        elif not isinstance(X, pd.DataFrame):
            # Si es numpy array, convertir a DataFrame
            X = pd.DataFrame(X, columns=self.feature_names)
            
        # Verificar que todas las columnas necesarias estén presentes
        missing_cols = set(self.feature_names) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Faltan columnas requeridas: {missing_cols}")
            
        # Seleccionar y ordenar columnas en el orden correcto
        X_processed = X[self.feature_names].copy()
        
        # Verificar valores faltantes
        if X_processed.isnull().any().any():
            print("⚠️ Advertencia: Se encontraron valores faltantes, rellenando con la mediana")
            X_processed = X_processed.fillna(X_processed.median())
            
        return X_processed
    
    def get_model_info(self):
        """
        Retorna información sobre el modelo.
        """
        info = {
            "model_type": type(self.model).__name__,
            "feature_names": self.feature_names,
            "uses_scaling": self.scaler is not None,
            "calibration_method": self.calibration_method,
            "uses_calibration": self.calibrator.use_calibration if hasattr(self.calibrator, 'use_calibration') else False,
            "is_fitted": self.is_fitted
        }
        
        # Agregar información específica del modelo si está disponible
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            info["feature_importance"] = importance_dict
        elif hasattr(self.model, 'coef_'):
            coef_dict = dict(zip(self.feature_names, self.model.coef_))
            info["coefficients"] = coef_dict
            if hasattr(self.model, 'intercept_'):
                info["intercept"] = self.model.intercept_
                
        return info