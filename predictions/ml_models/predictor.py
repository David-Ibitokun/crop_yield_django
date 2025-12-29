
"""ML model loading and prediction logic."""
import numpy as np
import pandas as pd
from pathlib import Path
from django.conf import settings
import tensorflow as tf
from tensorflow import keras
import joblib

class YieldPredictor:
    """Singleton class for loading and using ML models.

    Notes:
    - `fnn` accepts a single static feature vector (as shown below).
    - `lstm` requires a sequence input (e.g., key 'sequence' in `input_data`).
    - `hybrid` requires both `temporal` (sequence) and `static` inputs.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_models()
        return cls._instance

    def _load_models(self):
        """Load all ML models and preprocessors"""
        models_dir = Path(settings.ML_MODELS_DIR)

        # Load models (may raise if files are missing)
        self.fnn_model = keras.models.load_model(models_dir / 'fnn_model.keras')
        self.lstm_model = keras.models.load_model(models_dir / 'lstm_model.keras')
        self.hybrid_model = keras.models.load_model(models_dir / 'hybrid_model.keras')

        # Load scalers and encoders
        self.fnn_scaler = joblib.load(models_dir / 'fnn_scaler.pkl')
        self.crop_encoder = joblib.load(models_dir / 'crop_encoder.pkl')
        self.zone_encoder = joblib.load(models_dir / 'zone_encoder.pkl')
        self.state_encoder = joblib.load(models_dir / 'state_encoder.pkl')

        print("✓ ML models loaded successfully")

    def prepare_features(self, input_data):
        """Prepare a single static-feature vector for FNN/hybrid static branch.

        This is a minimal example — production code should validate keys and handle missing values.
        """
        feature_cols = [
            'Avg_Temp_C', 'Min_Temp_C', 'Max_Temp_C', 'Temp_Range_C',
            'Rainfall_mm', 'Rainy_Days', 'Max_Daily_Rainfall_mm', 'Rainfall_Intensity',
            'Avg_Humidity_Percent', 'Min_Humidity_Percent', 'Max_Humidity_Percent',
            'CO2_ppm', 'CO2_Growth_Rate_ppm_per_year',
            'Heat_Stress_Days', 'Cold_Stress_Days', 'Drought_Index', 'Flood_Risk_Index',
            'Soil_pH', 'Organic_Matter_Percent', 'Nitrogen_ppm', 'Phosphorus_ppm',
            'Potassium_ppm', 'Cation_Exchange_Capacity', 'Bulk_Density',
            'Water_Holding_Capacity_Percent',
            'Crop_encoded', 'Zone_encoded', 'State_encoded'
        ]

        features = np.zeros((1, len(feature_cols)))

        # Map inputs (use .get with defaults to avoid KeyError)
        features[0, feature_cols.index('Avg_Temp_C')] = input_data.get('avg_temp_c', 0.0)
        features[0, feature_cols.index('Rainfall_mm')] = input_data.get('rainfall_mm', 0.0)
        features[0, feature_cols.index('Avg_Humidity_Percent')] = input_data.get('avg_humidity', 0.0)
        features[0, feature_cols.index('CO2_ppm')] = input_data.get('co2_ppm', 0.0)
        features[0, feature_cols.index('Soil_pH')] = input_data.get('soil_ph', 6.5)
        features[0, feature_cols.index('Nitrogen_ppm')] = input_data.get('nitrogen_ppm', 0.0)
        features[0, feature_cols.index('Phosphorus_ppm')] = input_data.get('phosphorus_ppm', 0.0)
        features[0, feature_cols.index('Potassium_ppm')] = input_data.get('potassium_ppm', 0.0)

        # Encode categorical variables (assumes encoders were fit on training labels)
        if 'crop' in input_data:
            features[0, feature_cols.index('Crop_encoded')] = self.crop_encoder.transform([input_data['crop']])[0]
        if 'geopolitical_zone' in input_data:
            features[0, feature_cols.index('Zone_encoded')] = self.zone_encoder.transform([input_data['geopolitical_zone']])[0]
        if 'state' in input_data:
            features[0, feature_cols.index('State_encoded')] = self.state_encoder.transform([input_data['state']])[0]

        # Simple derived defaults
        features[0, feature_cols.index('Min_Temp_C')] = features[0, feature_cols.index('Avg_Temp_C')] - 5
        features[0, feature_cols.index('Max_Temp_C')] = features[0, feature_cols.index('Avg_Temp_C')] + 5
        features[0, feature_cols.index('Temp_Range_C')] = 10

        return features

    def predict(self, input_data, model='fnn'):
        """Make yield prediction.

        Expected `input_data` shapes:
        - fnn: dict of scalar features (as used by `prepare_features`)
        - lstm: dict containing key 'sequence' with shape (1, seq_len, n_features)
        - hybrid: dict containing 'temporal' (1, seq_len, n_temp_features) and 'static' (1, n_static_features)
        """

        # FNN/static path (single vector)
        features = self.prepare_features(input_data)
        features_scaled = self.fnn_scaler.transform(features)

        if model == 'fnn':
            pred = self.fnn_model.predict(features_scaled, verbose=0)
            yield_prediction = float(np.asarray(pred).flatten()[0])

        elif model == 'lstm':
            # Require explicit sequence input for LSTM
            seq = input_data.get('sequence')
            if seq is None:
                raise ValueError("LSTM model requires 'sequence' key with shape (1, seq_len, n_features).")
            seq_arr = np.asarray(seq)
            pred = self.lstm_model.predict(seq_arr, verbose=0)
            yield_prediction = float(np.asarray(pred).flatten()[0])

        elif model == 'hybrid':
            # Require both temporal and static inputs
            temporal = input_data.get('temporal')
            static = input_data.get('static')
            if temporal is None or static is None:
                raise ValueError("Hybrid model requires 'temporal' and 'static' inputs.")
            temporal_arr = np.asarray(temporal)
            static_arr = np.asarray(static)
            pred = self.hybrid_model.predict([temporal_arr, static_arr], verbose=0)
            yield_prediction = float(np.asarray(pred).flatten()[0])

        else:
            # Fallback to FNN
            pred = self.fnn_model.predict(features_scaled, verbose=0)
            yield_prediction = float(np.asarray(pred).flatten()[0])

        # Simple confidence interval (±10%) — replace with calibrated intervals in production
        ci = yield_prediction * 0.1

        return {
            'predicted_yield': yield_prediction,
            'confidence_lower': yield_prediction - ci,
            'confidence_upper': yield_prediction + ci,
            'model_used': model
        }
