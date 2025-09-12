"""
Preprocessing utilities for time series analysis
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import warnings


def make_stationary(df: pd.DataFrame, 
                   value_col: str = 'value',
                   method: str = 'diff',
                   **kwargs) -> pd.DataFrame:
    """
    Make time series stationary
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series dataframe
    value_col : str
        Column to make stationary
    method : str
        Method for making stationary ('diff', 'log', 'log_diff')
    
    Returns:
    --------
    pd.DataFrame
        Transformed dataframe
    """
    df_stationary = df.copy()
    
    if method == 'diff':
        periods = kwargs.get('periods', 1)
        df_stationary[f'{value_col}_diff'] = df[value_col].diff(periods=periods)
        
    elif method == 'log':
        if (df[value_col] <= 0).any():
            warnings.warn("Log transformation not suitable for non-positive values")
            return df_stationary
        df_stationary[f'{value_col}_log'] = np.log(df[value_col])
        
    elif method == 'log_diff':
        if (df[value_col] <= 0).any():
            warnings.warn("Log transformation not suitable for non-positive values")
            return df_stationary
        periods = kwargs.get('periods', 1)
        df_stationary[f'{value_col}_log_diff'] = np.log(df[value_col]).diff(periods=periods)
        
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    return df_stationary


def check_stationarity(series: pd.Series, alpha: float = 0.05) -> dict:
    """
    Test for stationarity using Augmented Dickey-Fuller test
    
    Parameters:
    -----------
    series : pd.Series
        Time series to test
    alpha : float
        Significance level
    
    Returns:
    --------
    dict
        Test results
    """
    result = adfuller(series.dropna())
    
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < alpha
    }


def scale_data(df: pd.DataFrame, 
              columns: List[str],
              method: str = 'standard',
              fit_on_train: bool = True,
              train_size: float = 0.8) -> Tuple[pd.DataFrame, object]:
    """
    Scale time series data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series dataframe
    columns : List[str]
        Columns to scale
    method : str
        Scaling method ('standard', 'minmax')
    fit_on_train : bool
        Whether to fit scaler only on training data
    train_size : float
        Proportion of data to use for training (if fit_on_train=True)
    
    Returns:
    --------
    Tuple[pd.DataFrame, object]
        Scaled dataframe and fitted scaler
    """
    df_scaled = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaling method: {method}")
    
    if fit_on_train:
        split_point = int(len(df) * train_size)
        train_data = df[columns].iloc[:split_point]
        scaler.fit(train_data)
    else:
        scaler.fit(df[columns])
    
    df_scaled[columns] = scaler.transform(df[columns])
    
    return df_scaled, scaler


def create_lag_features(df: pd.DataFrame, 
                       target_col: str,
                       lags: List[int] = [1, 2, 3, 24, 48]) -> pd.DataFrame:
    """
    Create lag features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series dataframe
    target_col : str
        Column to create lags for
    lags : List[int]
        List of lag periods
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with lag features
    """
    df_lagged = df.copy()
    
    for lag in lags:
        df_lagged[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    return df_lagged


def create_rolling_features(df: pd.DataFrame, 
                           target_col: str,
                           windows: List[int] = [3, 7, 24],
                           stats: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
    """
    Create rolling window features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series dataframe
    target_col : str
        Column to create rolling features for
    windows : List[int]
        List of window sizes
    stats : List[str]
        List of statistics to compute
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with rolling features
    """
    df_rolling = df.copy()
    
    for window in windows:
        rolling_obj = df[target_col].rolling(window=window)
        
        for stat in stats:
            if hasattr(rolling_obj, stat):
                df_rolling[f'{target_col}_rolling_{stat}_{window}'] = getattr(rolling_obj, stat)()
    
    return df_rolling


def create_expanding_features(df: pd.DataFrame, 
                             target_col: str,
                             stats: List[str] = ['mean', 'std']) -> pd.DataFrame:
    """
    Create expanding window features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series dataframe
    target_col : str
        Column to create expanding features for
    stats : List[str]
        List of statistics to compute
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with expanding features
    """
    df_expanding = df.copy()
    
    expanding_obj = df[target_col].expanding()
    
    for stat in stats:
        if hasattr(expanding_obj, stat):
            df_expanding[f'{target_col}_expanding_{stat}'] = getattr(expanding_obj, stat)()
    
    return df_expanding


def remove_trend(df: pd.DataFrame, 
                value_col: str = 'value',
                method: str = 'linear') -> pd.DataFrame:
    """
    Remove trend from time series
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series dataframe
    value_col : str
        Column to detrend
    method : str
        Detrending method ('linear', 'polynomial')
    
    Returns:
    --------
    pd.DataFrame
        Detrended dataframe
    """
    from scipy import signal
    
    df_detrended = df.copy()
    
    if method == 'linear':
        detrended = signal.detrend(df[value_col], type='linear')
    elif method == 'constant':
        detrended = signal.detrend(df[value_col], type='constant')
    else:
        raise ValueError(f"Unsupported detrending method: {method}")
    
    df_detrended[f'{value_col}_detrended'] = detrended
    
    return df_detrended


def create_fourier_features(df: pd.DataFrame, 
                           period: int,
                           n_harmonics: int = 3) -> pd.DataFrame:
    """
    Create Fourier features for seasonality
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series dataframe with datetime index
    period : int
        Period of seasonality (e.g., 24 for daily, 7 for weekly)
    n_harmonics : int
        Number of Fourier harmonics to include
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with Fourier features
    """
    df_fourier = df.copy()
    
    # Create time index
    t = np.arange(len(df))
    
    for i in range(1, n_harmonics + 1):
        df_fourier[f'fourier_sin_{period}_{i}'] = np.sin(2 * np.pi * i * t / period)
        df_fourier[f'fourier_cos_{period}_{i}'] = np.cos(2 * np.pi * i * t / period)
    
    return df_fourier


def handle_seasonality(df: pd.DataFrame, 
                      value_col: str = 'value',
                      method: str = 'decompose',
                      period: int = 24) -> pd.DataFrame:
    """
    Handle seasonality in time series
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series dataframe
    value_col : str
        Column to deseasonalize
    method : str
        Method for handling seasonality ('decompose', 'diff')
    period : int
        Seasonal period
    
    Returns:
    --------
    pd.DataFrame
        Processed dataframe
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    df_seasonal = df.copy()
    
    if method == 'decompose':
        decomposition = seasonal_decompose(df[value_col], 
                                         model='additive', 
                                         period=period)
        df_seasonal[f'{value_col}_trend'] = decomposition.trend
        df_seasonal[f'{value_col}_seasonal'] = decomposition.seasonal
        df_seasonal[f'{value_col}_residual'] = decomposition.resid
        df_seasonal[f'{value_col}_deseasonalized'] = df[value_col] - decomposition.seasonal
        
    elif method == 'diff':
        df_seasonal[f'{value_col}_seasonal_diff'] = df[value_col].diff(periods=period)
        
    else:
        raise ValueError(f"Unsupported seasonality method: {method}")
    
    return df_seasonal


def prepare_sequences(data: np.ndarray, 
                     sequence_length: int,
                     prediction_length: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for deep learning models
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data
    sequence_length : int
        Length of input sequences
    prediction_length : int
        Length of prediction sequences
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Input sequences and target sequences
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length - prediction_length + 1):
        X.append(data[i:(i + sequence_length)])
        y.append(data[(i + sequence_length):(i + sequence_length + prediction_length)])
    
    return np.array(X), np.array(y)