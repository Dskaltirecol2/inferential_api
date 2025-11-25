import pandas as pd
import numpy as np
import pickle
import re


# ============================================================================
# COMPAT UNPICKLER (para cargar pickle con clases custom)
# ============================================================================
class _CompatUnpickler(pickle.Unpickler):
    """
    Custom Unpickler para mapear clases antiguas al tipo actual.
    Evita errores del tipo:
    
        can't get attribute 'LlantaPreprocessor'
    
    """
    def find_class(self, module, name):
        if name == 'LlantaPreprocessor':
            return LlantaPreprocessor
        return super().find_class(module, name)


# Utilidad pública para cargar pickle con compatibilidad.
def load_with_compat(path):
    with open(path, 'rb') as f:
        return _CompatUnpickler(f).load()


# ============================================================================
# LLANTA PREPROCESSOR
# ============================================================================
class LlantaPreprocessor:
    """
    Preprocesador de llantas tal como fue usado en el entrenamiento.
    
    Atributos esperados:
        - scaler
        - feature_names
        - numeric_features
        - median_values
    """

    def __init__(self, scaler, feature_names, numeric_features):
        self.scaler = scaler
        self.feature_names = feature_names
        self.numeric_features = numeric_features
        self.median_values = {}  # se rellena al cargar pickle

    # ---------------------------------------------------------
    # UTILIDADES INTERNAS
    # ---------------------------------------------------------
    def agrupar_fabricante(self, modelo):
        if 'CAT' in str(modelo): return 'Caterpillar'
        if 'HIT' in str(modelo): return 'Hitachi'
        if 'KOM' in str(modelo): return 'Komatsu'
        if 'LET' in str(modelo): return 'Letourneau'
        return 'Otros'

    def extract_tamano(self, componente):
        """Extrae patrón tipo '29.5R25'."""
        if pd.isna(componente): return None
        match = re.search(r'^(\d+(?:\.\d+)?/\d+R\d+|\d+(?:\.\d+)?R\d+)', str(componente))
        return match.group(1) if match else None

    def clean_numeric_column(self, series, col_name):
        """
        Normaliza formatos colombianos.
        Basado en tu lógica de Lambda.
        """
        thousands_cols = {'total_kms_llanta', 'total_horas_llanta'}

        def to_str(x):
            if pd.isna(x): return ''
            if isinstance(x, (int, float, np.number)):
                s = str(x).replace(',', '.')
                return s
            return str(x).strip()

        s = series.map(to_str)
        s = s.replace(r'^(nan|none|null|\s*)$', '', regex=True)

        def parse_number(val):
            if val == '':
                return np.nan
            # comma decimal
            if ',' in val and '.' not in val:
                try: return float(val.replace(',', '.'))
                except: return np.nan

            # thousands
            if val.count('.') > 1:
                try: return float(val.replace('.', ''))
                except: return np.nan

            # decimal or thousands ambiguous
            if val.count('.') == 1:
                left, right = val.split('.', 1)
                if re.fullmatch(r'\d{3}', right):
                    # thousands, e.g. "39.142" -> 39142
                    try: return float(left + right)
                    except: return np.nan
                try: return float(val)
                except: return np.nan

            # pure number
            try:
                return float(val)
            except:
                return np.nan

        return s.apply(parse_number)

    # ---------------------------------------------------------
    # PREPROCESS PRINCIPAL
    # ---------------------------------------------------------
    def preprocess(self, df_raw):
        df = df_raw.copy()

        # 1. Agrupar fabricante
        df['modelo_agrupado'] = df['modelo_flota'].apply(self.agrupar_fabricante)

        # 2. Extraer tamaño
        if 'componente' in df.columns:
            df['tamano'] = df['componente'].apply(self.extract_tamano)
        else:
            df['tamano'] = None

        # 3. Columns to clean
        numeric_cols_to_clean = [
            'rtdext_montaje', 'rtdsfext', 'rtdint_montaje', 'rtdsfint',
            'total_horas_llanta', 'total_kms_llanta', 'diasprimerfalla',
            'no_correctivos', 'no_preventivos', 'no_parches', 'no_malla',
            'max_longi', 'max_prof', 'min_longi', 'min_prof',
            'prof_prom', 'longi_prom', 'sum_prof', 'sum_longi',
            'no_baja_presion', 'no_dano_corte', 'dano_separacion_corte',
            'dano_corte_costado', 'dano_corte_banda', 'desgaste_irregular',
            'dano_corte_hombro', 'falla_reparacion_reencauchaje',
            'desgaste_total', 'dano_corte_arrancabanda',
            'carcasa_danada_banda', 'dano_separacioncorte_banda',
            'burbuja_llanta', 'dano_talon', 'fuga_pequena',
            'carcasa_danada_costado', 'llanta_alta_temp',
            'carcasada_danada_hombro', 'dano_separacion_calor',
            'dano_impacto', 'dano_accidente', 'dano_rayo_ele_fue',
            'dano_mecanico', 'dano_sepa_meca', 'impacto',
            'penetracion_roca_metal', 'desinflado_rodado',
            'cortes_circunferenciales', 'dano_sepacor_costado',
            'dano_sepacor_hombro', 'baja_presion_pinchazo',
            'prio_3', 'prio_2', 'prio_1',
            'no_pitcrew', 'no_pitcrew_2'
        ]

        for col in numeric_cols_to_clean:
            if col in df.columns:
                df[col] = self.clean_numeric_column(df[col], col)

        # 4. Calcular desgastes
        df['desgaste_ext'] = df['rtdext_montaje'] - df['rtdsfext']
        df['desgaste_int'] = df['rtdint_montaje'] - df['rtdsfint']

        # 5. Reparaciones
        repair_cols = [
            'no_correctivos', 'no_preventivos', 'no_parches', 'no_malla',
            'max_longi', 'min_longi', 'max_prof', 'min_prof',
            'sum_longi', 'sum_prof', 'longi_prom', 'prof_prom'
        ]
        for col in repair_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # 6. Imputar medianas
        for col in self.numeric_features:
            if col in df.columns and col in self.median_values:
                df[col] = df[col].fillna(self.median_values[col])

        # 7. Selección de features
        categorical_features = ['modelo_agrupado', 'tamano', 'posicion']

        numeric_present = [
            f for f in self.numeric_features 
            if f in df.columns and f not in categorical_features
        ]

        all_features = numeric_present + categorical_features
        df_proc = df[all_features].copy()

        # 8. One-hot encoding
        cat_cols = [c for c in categorical_features if c in df_proc.columns]

        df_encoded = pd.get_dummies(
            df_proc,
            columns=cat_cols,
            prefix=['fabricante', 'tamano', 'posicion'][:len(cat_cols)],
            drop_first=False,
            dtype=int
        )

        # 9. Asegurar columnas de entrenamiento
        for col in self.feature_names:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        df_encoded = df_encoded[self.feature_names]

        # 10. Escalar numéricas
        num_cols = [c for c in self.numeric_features if c in df_encoded.columns]
        df_encoded[num_cols] = self.scaler.transform(df_encoded[num_cols])

        return df_encoded
