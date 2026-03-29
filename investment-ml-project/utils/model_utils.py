"""
Model utilities for saving and loading trained models.
"""

import pickle
import json
import os
import joblib
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np


def save_model(model: Any, filepath: str, model_type: str = 'auto'):
    """
    Save a trained model.
    
    Parameters:
    -----------
    model : Any
        Trained model object
    filepath : str
        Path to save model
    model_type : str
        Model type ('sklearn', 'xgboost', 'tensorflow', 'auto')
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if model_type == 'auto':
        # Auto-detect model type
        model_class = type(model).__name__.lower()
        if 'randomforest' in model_class or 'sklearn' in str(type(model)):
            model_type = 'sklearn'
        elif 'xgb' in model_class or 'xgboost' in str(type(model)):
            model_type = 'xgboost'
        elif 'tensorflow' in str(type(model)) or 'keras' in str(type(model)):
            model_type = 'tensorflow'
        else:
            model_type = 'sklearn'  # Default
    
    if model_type == 'sklearn':
        joblib.dump(model, filepath)
        print(f"Saved sklearn model to {filepath}")
    elif model_type == 'xgboost':
        model.save_model(filepath)
        print(f"Saved XGBoost model to {filepath}")
    elif model_type == 'tensorflow':
        model.save(filepath)
        print(f"Saved TensorFlow model to {filepath}")
    else:
        # Fallback to pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved model (pickle) to {filepath}")


def load_model(filepath: str, model_type: str = 'auto') -> Any:
    """
    Load a saved model.
    
    Parameters:
    -----------
    filepath : str
        Path to model file
    model_type : str
        Model type ('sklearn', 'xgboost', 'tensorflow', 'auto')
    
    Returns:
    --------
    Any : Loaded model
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    if model_type == 'auto':
        # Auto-detect from file extension
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.pkl' or ext == '.joblib':
            model_type = 'sklearn'
        elif ext == '.json' or ext == '.model':
            model_type = 'xgboost'
        elif ext == '' or os.path.isdir(filepath):
            model_type = 'tensorflow'
        else:
            model_type = 'sklearn'
    
    if model_type == 'sklearn':
        model = joblib.load(filepath)
    elif model_type == 'xgboost':
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(filepath)
    elif model_type == 'tensorflow':
        import tensorflow as tf
        model = tf.keras.models.load_model(filepath)
    else:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
    
    print(f"Loaded model from {filepath}")
    return model


def save_metrics(metrics: Dict[str, Any], filepath: str):
    """Save model metrics to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy types to native Python types
    metrics_serializable = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer, np.floating)):
            metrics_serializable[k] = float(v)
        elif isinstance(v, np.ndarray):
            metrics_serializable[k] = v.tolist()
        elif isinstance(v, pd.DataFrame):
            metrics_serializable[k] = v.to_dict()
        else:
            metrics_serializable[k] = v
    
    with open(filepath, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"Saved metrics to {filepath}")


def load_metrics(filepath: str) -> Dict[str, Any]:
    """Load model metrics from JSON."""
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    return metrics


def save_predictions(
    predictions: pd.Series,
    filepath: str,
    index_name: str = 'Date'
):
    """Save predictions to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if isinstance(predictions, pd.Series):
        df = pd.DataFrame({index_name: predictions.index, 'prediction': predictions.values})
        df = df.set_index(index_name)
    else:
        df = predictions
    
    df.to_csv(filepath)
    print(f"Saved predictions to {filepath}")


def load_predictions(filepath: str) -> pd.Series:
    """Load predictions from CSV."""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    if 'prediction' in df.columns:
        return df['prediction']
    return df.iloc[:, 0]

