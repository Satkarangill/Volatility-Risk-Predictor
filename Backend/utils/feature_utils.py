"""
Feature engineering utilities for ML models.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def compute_technical_indicators(
    prices: pd.DataFrame,
    windows: List[int] = [7, 14, 21, 30, 63, 90]
) -> pd.DataFrame:
    """
    Compute technical indicators from price data.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data (Date index, Ticker columns)
    windows : List[int]
        Rolling windows for indicators
    
    Returns:
    --------
    pd.DataFrame : Features with MultiIndex (Date, Feature)
    """
    features = {}
    
    returns = prices.pct_change()
    
    # Rolling returns
    for window in windows:
        features[f'return_{window}d'] = returns.rolling(window).sum()
        features[f'return_{window}d_mean'] = returns.rolling(window).mean()
        features[f'return_{window}d_std'] = returns.rolling(window).std()
    
    # Rolling volatility
    for window in windows:
        vol = returns.rolling(window).std() * np.sqrt(252)
        features[f'volatility_{window}d'] = vol
    
    # Price momentum
    for window in windows:
        features[f'momentum_{window}d'] = prices.pct_change(window)
    
    # Moving averages
    for window in [20, 50, 200]:
        if window <= len(prices):
            ma = prices.rolling(window).mean()
            features[f'ma_{window}d'] = ma
            features[f'price_to_ma_{window}d'] = prices / ma - 1
    
    # RSI (Relative Strength Index)
    for window in [14, 21]:
        delta = returns
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features[f'rsi_{window}d'] = rsi
    
    # Bollinger Bands
    for window in [20, 30]:
        ma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        features[f'bb_upper_{window}d'] = ma + 2 * std
        features[f'bb_lower_{window}d'] = ma - 2 * std
        features[f'bb_width_{window}d'] = (ma + 2 * std - (ma - 2 * std)) / ma
    
    # Combine all features
    feature_df = pd.DataFrame(features)
    
    return feature_df


def create_lag_features(
    data: pd.DataFrame,
    lags: List[int] = [1, 2, 3, 5, 10]
) -> pd.DataFrame:
    """
    Create lagged features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Feature data
    lags : List[int]
        Lag periods
    
    Returns:
    --------
    pd.DataFrame : Features with lags
    """
    lagged_features = {}
    
    for col in data.columns:
        for lag in lags:
            lagged_features[f'{col}_lag{lag}'] = data[col].shift(lag)
    
    return pd.DataFrame(lagged_features)


def create_target_variable(
    returns: pd.DataFrame,
    horizon: int = 1,
    method: str = 'volatility'
) -> pd.Series:
    """
    Create target variable for prediction.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns
    horizon : int
        Prediction horizon (days ahead)
    method : str
        'volatility' or 'return'
    
    Returns:
    --------
    pd.Series : Target variable
    """
    if method == 'volatility':
        # Future volatility (rolling std of future returns)
        future_returns = returns.shift(-horizon)
        target = future_returns.rolling(window=horizon).std() * np.sqrt(252)
    elif method == 'return':
        # Future return
        target = returns.shift(-horizon)
    else:
        raise ValueError("method must be 'volatility' or 'return'")
    
    return target.iloc[:, 0] if target.shape[1] == 1 else target


def prepare_ml_features(
    prices: pd.DataFrame,
    target_col: Optional[str] = None,
    include_technical: bool = True,
    include_lags: bool = True,
    lag_periods: List[int] = [1, 2, 3, 5, 10]
) -> pd.DataFrame:
    """
    Prepare feature matrix for ML models.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data
    target_col : str, optional
        Target column name
    include_technical : bool
        Include technical indicators
    include_lags : bool
        Include lagged features
    lag_periods : List[int]
        Lag periods to include
    
    Returns:
    --------
    pd.DataFrame : Feature matrix
    """
    features_list = []
    
    # Basic returns
    returns = prices.pct_change()
    features_list.append(returns)
    
    # Technical indicators
    if include_technical:
        tech_features = compute_technical_indicators(prices)
        features_list.append(tech_features)
    
    # Combine features
    all_features = pd.concat(features_list, axis=1)
    
    # Lag features
    if include_lags:
        lagged = create_lag_features(all_features, lags=lag_periods)
        all_features = pd.concat([all_features, lagged], axis=1)
    
    # Drop NaN
    all_features = all_features.dropna()
    
    return all_features


def normalize_features(
    features: pd.DataFrame,
    method: str = 'standardize'
) -> tuple:
    """
    Normalize features.
    
    Parameters:
    -----------
    features : pd.DataFrame
        Feature matrix
    method : str
        'standardize' or 'minmax'
    
    Returns:
    --------
    tuple : (normalized_features, scaler_params)
    """
    if method == 'standardize':
        mean = features.mean()
        std = features.std()
        normalized = (features - mean) / std
        scaler_params = {'mean': mean, 'std': std}
    elif method == 'minmax':
        min_val = features.min()
        max_val = features.max()
        normalized = (features - min_val) / (max_val - min_val)
        scaler_params = {'min': min_val, 'max': max_val}
    else:
        raise ValueError("method must be 'standardize' or 'minmax'")
    
    return normalized, scaler_params

