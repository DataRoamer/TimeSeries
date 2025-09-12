"""
Example time series analysis script
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import load_time_series, validate_time_series
from utils.preprocessing import create_lag_features, create_rolling_features
from models.forecasting import ARIMAForecaster, RandomForestForecaster, ModelSelector
from visualization.plots import plot_time_series, plot_forecast_comparison
import warnings
warnings.filterwarnings('ignore')


def main():
    """
    Example time series analysis workflow
    """
    print("Time Series Analysis Example")
    print("=" * 40)
    
    # 1. Load data (example with synthetic data)
    print("1. Loading data...")
    
    # Generate example data
    dates = pd.date_range('2020-01-01', periods=1000, freq='H')
    trend = np.linspace(100, 200, 1000)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(1000) / 24)  # Daily seasonality
    noise = np.random.normal(0, 5, 1000)
    values = trend + seasonal + noise
    
    df = pd.DataFrame({
        'timestamp': dates,
        'value': values
    })
    df.set_index('timestamp', inplace=True)
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # 2. Data validation
    print("\n2. Validating data...")
    validation_results = validate_time_series(df)
    for key, value in validation_results.items():
        print(f"  {key}: {value}")
    
    # 3. Visualization
    print("\n3. Creating visualizations...")
    plot_time_series(df, title="Example Time Series Data")
    
    # 4. Feature engineering
    print("\n4. Feature engineering...")
    df_features = create_lag_features(df, 'value', lags=[1, 2, 24])
    df_features = create_rolling_features(df_features, 'value', windows=[3, 24])
    print(f"Features created. New shape: {df_features.shape}")
    
    # 5. Train/test split
    print("\n5. Splitting data...")
    split_point = int(len(df) * 0.8)
    train_data = df['value'][:split_point]
    test_data = df['value'][split_point:]
    
    print(f"Training data: {len(train_data)} points")
    print(f"Test data: {len(test_data)} points")
    
    # 6. Model training and comparison
    print("\n6. Training models...")
    
    # Initialize models
    arima_model = ARIMAForecaster(order=(1, 1, 1))
    rf_model = RandomForestForecaster(lags=[1, 2, 24])
    
    # Fit models
    arima_model.fit(train_data)
    rf_model.fit(train_data)
    
    # Make forecasts
    steps = len(test_data)
    arima_forecast = arima_model.predict(steps)
    rf_forecast = rf_model.predict(steps)
    
    # 7. Model evaluation
    print("\n7. Evaluating models...")
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    arima_mae = mean_absolute_error(test_data, arima_forecast)
    rf_mae = mean_absolute_error(test_data, rf_forecast)
    
    arima_rmse = np.sqrt(mean_squared_error(test_data, arima_forecast))
    rf_rmse = np.sqrt(mean_squared_error(test_data, rf_forecast))
    
    print(f"ARIMA - MAE: {arima_mae:.2f}, RMSE: {arima_rmse:.2f}")
    print(f"Random Forest - MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}")
    
    # 8. Visualization of results
    print("\n8. Plotting results...")
    forecasts = {
        'ARIMA': arima_forecast,
        'Random Forest': rf_forecast
    }
    
    plot_forecast_comparison(train_data, test_data, forecasts)
    
    # 9. Automatic model selection
    print("\n9. Automatic model selection...")
    selector = ModelSelector()
    best_name, best_model, all_scores = selector.select_best_model(
        train_data, test_data, metric='mae'
    )
    
    print(f"Best model: {best_name}")
    print("All model scores (MAE):")
    for name, score in all_scores.items():
        print(f"  {name}: {score:.2f}")
    
    print("\n" + "=" * 40)
    print("Analysis completed successfully!")


if __name__ == "__main__":
    main()