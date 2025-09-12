"""
Visualization utilities for time series analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Optional, List, Tuple
import warnings


def plot_time_series(df: pd.DataFrame, 
                    value_col: str = 'value',
                    title: str = 'Time Series Plot',
                    figsize: Tuple[int, int] = (12, 6),
                    interactive: bool = False) -> None:
    """
    Plot time series data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series dataframe
    value_col : str
        Column to plot
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size for matplotlib
    interactive : bool
        Whether to create interactive plotly plot
    """
    if interactive:
        fig = px.line(df.reset_index(), x=df.index.name or 'timestamp', y=value_col,
                     title=title)
        fig.show()
    else:
        plt.figure(figsize=figsize)
        plt.plot(df.index, df[value_col])
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel(value_col.replace('_', ' ').title())
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_decomposition(df: pd.DataFrame, 
                      value_col: str = 'value',
                      period: int = 24,
                      model: str = 'additive',
                      figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Plot time series decomposition
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series dataframe
    value_col : str
        Column to decompose
    period : int
        Seasonal period
    model : str
        Decomposition model ('additive' or 'multiplicative')
    figsize : Tuple[int, int]
        Figure size
    """
    decomposition = seasonal_decompose(df[value_col], model=model, period=period)
    
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    
    decomposition.observed.plot(ax=axes[0], title='Original')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    
    plt.tight_layout()
    plt.show()


def plot_acf_pacf(df: pd.DataFrame, 
                 value_col: str = 'value',
                 lags: int = 40,
                 figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot ACF and PACF
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series dataframe
    value_col : str
        Column to analyze
    lags : int
        Number of lags to show
    figsize : Tuple[int, int]
        Figure size
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    plot_acf(df[value_col].dropna(), lags=lags, ax=ax1, title='Autocorrelation Function')
    plot_pacf(df[value_col].dropna(), lags=lags, ax=ax2, title='Partial Autocorrelation Function')
    
    plt.tight_layout()
    plt.show()


def plot_seasonal_patterns(df: pd.DataFrame, 
                          value_col: str = 'value',
                          freq: str = 'hour',
                          figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot seasonal patterns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Time series dataframe with datetime index
    value_col : str
        Column to analyze
    freq : str
        Frequency for grouping ('hour', 'day', 'month', 'dayofweek')
    figsize : Tuple[int, int]
        Figure size
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be datetime")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Hourly pattern
    if freq in ['hour', 'all']:
        hourly_avg = df.groupby(df.index.hour)[value_col].mean()
        axes[0, 0].plot(hourly_avg.index, hourly_avg.values, marker='o')
        axes[0, 0].set_title('Average by Hour')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Average Value')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Daily pattern
    if freq in ['day', 'dayofweek', 'all']:
        daily_avg = df.groupby(df.index.dayofweek)[value_col].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[0, 1].bar(days, daily_avg.values)
        axes[0, 1].set_title('Average by Day of Week')
        axes[0, 1].set_xlabel('Day')
        axes[0, 1].set_ylabel('Average Value')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Monthly pattern
    if freq in ['month', 'all']:
        monthly_avg = df.groupby(df.index.month)[value_col].mean()
        axes[1, 0].plot(monthly_avg.index, monthly_avg.values, marker='o')
        axes[1, 0].set_title('Average by Month')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Average Value')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot by day of week
    df_with_dow = df.copy()
    df_with_dow['day_of_week'] = df.index.dayofweek
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    df_with_dow['day_name'] = df_with_dow['day_of_week'].map(lambda x: day_names[x])
    
    sns.boxplot(data=df_with_dow, x='day_name', y=value_col, ax=axes[1, 1])
    axes[1, 1].set_title('Distribution by Day of Week')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def plot_forecast_comparison(train_data: pd.Series,
                           test_data: pd.Series,
                           forecasts: dict,
                           figsize: Tuple[int, int] = (15, 8),
                           interactive: bool = False) -> None:
    """
    Plot multiple forecasts comparison
    
    Parameters:
    -----------
    train_data : pd.Series
        Training data
    test_data : pd.Series
        Test data (actual values)
    forecasts : dict
        Dictionary of {model_name: forecast_values}
    figsize : Tuple[int, int]
        Figure size for matplotlib
    interactive : bool
        Whether to create interactive plotly plot
    """
    if interactive:
        fig = go.Figure()
        
        # Add training data
        fig.add_trace(go.Scatter(x=train_data.index, y=train_data.values,
                                mode='lines', name='Training Data',
                                line=dict(color='blue')))
        
        # Add test data
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data.values,
                                mode='lines', name='Actual',
                                line=dict(color='black', width=3)))
        
        # Add forecasts
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            fig.add_trace(go.Scatter(x=test_data.index, y=forecast,
                                    mode='lines', name=f'{model_name} Forecast',
                                    line=dict(color=colors[i % len(colors)], dash='dash')))
        
        fig.update_layout(title='Forecast Comparison',
                         xaxis_title='Time',
                         yaxis_title='Value',
                         width=1000, height=600)
        fig.show()
    
    else:
        plt.figure(figsize=figsize)
        
        # Plot training data
        plt.plot(train_data.index, train_data.values, label='Training Data', color='blue')
        
        # Plot test data
        plt.plot(test_data.index, test_data.values, label='Actual', color='black', linewidth=2)
        
        # Plot forecasts
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            plt.plot(test_data.index, forecast, label=f'{model_name} Forecast',
                    color=colors[i % len(colors)], linestyle='--')
        
        plt.legend()
        plt.title('Forecast Comparison')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_residuals(actual: np.ndarray, 
                  predicted: np.ndarray,
                  model_name: str = 'Model',
                  figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot residual analysis
    
    Parameters:
    -----------
    actual : np.ndarray
        Actual values
    predicted : np.ndarray
        Predicted values
    model_name : str
        Name of the model
    figsize : Tuple[int, int]
        Figure size
    """
    residuals = actual - predicted
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Residuals vs Fitted
    axes[0, 0].scatter(predicted, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title(f'{model_name} - Residuals vs Fitted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[0, 1].hist(residuals, bins=30, alpha=0.7, density=True)
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title(f'{model_name} - Residual Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f'{model_name} - Q-Q Plot')
    
    # Residuals over time
    axes[1, 1].plot(residuals)
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title(f'{model_name} - Residuals Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importance_df: pd.DataFrame,
                           top_n: int = 15,
                           figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot feature importance
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        Dataframe with 'feature' and 'importance' columns
    top_n : int
        Number of top features to show
    figsize : Tuple[int, int]
        Figure size
    """
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_prediction_intervals(actual: pd.Series,
                            forecast: np.ndarray,
                            lower_bound: np.ndarray,
                            upper_bound: np.ndarray,
                            train_data: Optional[pd.Series] = None,
                            figsize: Tuple[int, int] = (15, 8)) -> None:
    """
    Plot forecast with prediction intervals
    
    Parameters:
    -----------
    actual : pd.Series
        Actual values
    forecast : np.ndarray
        Forecast values
    lower_bound : np.ndarray
        Lower bound of prediction interval
    upper_bound : np.ndarray
        Upper bound of prediction interval
    train_data : pd.Series, optional
        Training data to plot
    figsize : Tuple[int, int]
        Figure size
    """
    plt.figure(figsize=figsize)
    
    if train_data is not None:
        plt.plot(train_data.index, train_data.values, label='Training Data', color='blue')
    
    plt.plot(actual.index, actual.values, label='Actual', color='black', linewidth=2)
    plt.plot(actual.index, forecast, label='Forecast', color='red', linestyle='--')
    plt.fill_between(actual.index, lower_bound, upper_bound, alpha=0.3, label='95% CI')
    
    plt.legend()
    plt.title('Forecast with Prediction Intervals')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()