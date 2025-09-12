"""
Data loading utilities for time series analysis
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List
import warnings


def load_time_series(file_path: str, 
                    timestamp_col: str = 'timestamp', 
                    value_col: str = 'value',
                    parse_dates: bool = True,
                    set_index: bool = True) -> pd.DataFrame:
    """
    Load time series data from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    timestamp_col : str
        Name of the timestamp column
    value_col : str
        Name of the value column
    parse_dates : bool
        Whether to parse dates automatically
    set_index : bool
        Whether to set timestamp as index
    
    Returns:
    --------
    pd.DataFrame
        Loaded time series data
    """
    try:
        if parse_dates:
            df = pd.read_csv(file_path, parse_dates=[timestamp_col])
        else:
            df = pd.read_csv(file_path)
            
        if set_index:
            df = df.set_index(timestamp_col)
            
        return df
    
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")


def validate_time_series(df: pd.DataFrame, 
                        value_col: str = 'value',
                        check_duplicates: bool = True,
                        check_missing: bool = True) -> dict:
    """
    Validate time series data quality
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series dataframe
    value_col : str
        Name of the value column to validate
    check_duplicates : bool
        Whether to check for duplicate timestamps
    check_missing : bool
        Whether to check for missing values
    
    Returns:
    --------
    dict
        Validation results
    """
    results = {}
    
    # Basic info
    results['total_records'] = len(df)
    results['date_range'] = (df.index.min(), df.index.max())
    
    # Check for duplicates
    if check_duplicates:
        duplicates = df.index.duplicated().sum()
        results['duplicate_timestamps'] = duplicates
        
    # Check for missing values
    if check_missing:
        missing_values = df[value_col].isnull().sum()
        results['missing_values'] = missing_values
        results['missing_percentage'] = (missing_values / len(df)) * 100
    
    # Check for negative values
    negative_values = (df[value_col] < 0).sum()
    results['negative_values'] = negative_values
    
    # Basic statistics
    results['mean'] = df[value_col].mean()
    results['std'] = df[value_col].std()
    results['min'] = df[value_col].min()
    results['max'] = df[value_col].max()
    
    return results


def resample_time_series(df: pd.DataFrame, 
                        freq: str,
                        agg_func: str = 'mean',
                        value_col: str = 'value') -> pd.DataFrame:
    """
    Resample time series to different frequency
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series dataframe with datetime index
    freq : str
        Target frequency ('H', 'D', '30T', etc.)
    agg_func : str
        Aggregation function ('mean', 'sum', 'max', 'min')
    value_col : str
        Column to resample
    
    Returns:
    --------
    pd.DataFrame
        Resampled dataframe
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be datetime for resampling")
    
    agg_functions = {
        'mean': lambda x: x.mean(),
        'sum': lambda x: x.sum(),
        'max': lambda x: x.max(),
        'min': lambda x: x.min(),
        'median': lambda x: x.median(),
        'std': lambda x: x.std()
    }
    
    if agg_func not in agg_functions:
        raise ValueError(f"Unsupported aggregation function: {agg_func}")
    
    resampled = df.resample(freq)[value_col].agg(agg_functions[agg_func])
    return pd.DataFrame({value_col: resampled})


def fill_missing_values(df: pd.DataFrame, 
                       method: str = 'interpolate',
                       value_col: str = 'value',
                       **kwargs) -> pd.DataFrame:
    """
    Fill missing values in time series
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series dataframe
    method : str
        Method to fill missing values ('interpolate', 'forward_fill', 'backward_fill', 'mean')
    value_col : str
        Column to fill missing values
    **kwargs : dict
        Additional arguments for filling method
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with filled missing values
    """
    df_filled = df.copy()
    
    if method == 'interpolate':
        df_filled[value_col] = df_filled[value_col].interpolate(**kwargs)
    elif method == 'forward_fill':
        df_filled[value_col] = df_filled[value_col].fillna(method='ffill')
    elif method == 'backward_fill':
        df_filled[value_col] = df_filled[value_col].fillna(method='bfill')
    elif method == 'mean':
        mean_val = df[value_col].mean()
        df_filled[value_col] = df_filled[value_col].fillna(mean_val)
    else:
        raise ValueError(f"Unsupported filling method: {method}")
    
    return df_filled


def detect_outliers(df: pd.DataFrame, 
                   value_col: str = 'value',
                   method: str = 'iqr',
                   threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in time series
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series dataframe
    value_col : str
        Column to check for outliers
    method : str
        Method for outlier detection ('iqr', 'zscore')
    threshold : float
        Threshold for outlier detection
    
    Returns:
    --------
    pd.Series
        Boolean series indicating outliers
    """
    values = df[value_col]
    
    if method == 'iqr':
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (values < lower_bound) | (values > upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs((values - values.mean()) / values.std())
        outliers = z_scores > threshold
        
    else:
        raise ValueError(f"Unsupported outlier detection method: {method}")
    
    return outliers


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from datetime index
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series dataframe with datetime index
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with additional time features
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be datetime to create time features")
    
    df_features = df.copy()
    
    # Basic time features
    df_features['hour'] = df.index.hour
    df_features['day'] = df.index.day
    df_features['day_of_week'] = df.index.dayofweek
    df_features['day_of_year'] = df.index.dayofyear
    df_features['week'] = df.index.isocalendar().week
    df_features['month'] = df.index.month
    df_features['quarter'] = df.index.quarter
    df_features['year'] = df.index.year
    
    # Cyclical features
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    # Binary features
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
    df_features['is_business_hour'] = ((df_features['hour'] >= 9) & (df_features['hour'] <= 17)).astype(int)
    
    return df_features