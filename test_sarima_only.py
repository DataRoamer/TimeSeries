"""
Test script for SARIMA model only (without LSTM/TensorFlow dependency)
"""

import sys
import os
sys.path.append('TimeSeries/src')

import pandas as pd
import numpy as np
from models.forecasting import SARIMAForecaster, ARIMAForecaster

def test_sarima_model():
    """Test SARIMA model"""
    
    print("Loading NYC taxi data...")
    df = pd.read_csv('TimeSeries/data/raw/nyc_taxi.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Use a smaller subset for testing
    data = df['value'][:500]  # First 500 points for faster testing
    
    # Split data
    split_point = int(len(data) * 0.8)
    train_data = data[:split_point]
    test_data = data[split_point:]
    
    print(f"Training data: {len(train_data)} points")
    print(f"Test data: {len(test_data)} points")
    
    results = {}
    
    # Test SARIMA
    print("\n1. Testing SARIMA model...")
    try:
        sarima_model = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
        print("   Fitting SARIMA model...")
        sarima_model.fit(train_data)
        print("   Making predictions...")
        sarima_forecast = sarima_model.predict(len(test_data))
        sarima_mae = np.mean(np.abs(test_data - sarima_forecast))
        results['SARIMA'] = {'mae': sarima_mae, 'status': 'Success'}
        print(f"   SARIMA MAE: {sarima_mae:.2f}")
        
        # Test prediction intervals
        print("   Testing prediction intervals...")
        forecast, lower, upper = sarima_model.forecast_with_intervals(len(test_data))
        print(f"   Forecast range: {np.min(forecast):.2f} to {np.max(forecast):.2f}")
        print(f"   Confidence interval width: {np.mean(upper - lower):.2f}")
        
    except Exception as e:
        results['SARIMA'] = {'mae': float('inf'), 'status': f'Failed: {str(e)}'}
        print(f"   SARIMA failed: {e}")
    
    # Test ARIMA for comparison
    print("\n2. Testing ARIMA model (for comparison)...")
    try:
        arima_model = ARIMAForecaster(order=(1, 1, 1))
        print("   Fitting ARIMA model...")
        arima_model.fit(train_data)
        print("   Making predictions...")
        arima_forecast = arima_model.predict(len(test_data))
        arima_mae = np.mean(np.abs(test_data - arima_forecast))
        results['ARIMA'] = {'mae': arima_mae, 'status': 'Success'}
        print(f"   ARIMA MAE: {arima_mae:.2f}")
    except Exception as e:
        results['ARIMA'] = {'mae': float('inf'), 'status': f'Failed: {str(e)}'}
        print(f"   ARIMA failed: {e}")
    
    # Summary
    print("\n" + "="*50)
    print("TESTING SUMMARY")
    print("="*50)
    for model, result in results.items():
        status = result['status']
        mae = result['mae']
        if mae != float('inf'):
            print(f"{model:<15}: {status:<10} MAE: {mae:.2f}")
        else:
            print(f"{model:<15}: {status}")
    
    # Compare performance
    if results['SARIMA']['mae'] != float('inf') and results['ARIMA']['mae'] != float('inf'):
        sarima_mae = results['SARIMA']['mae']
        arima_mae = results['ARIMA']['mae']
        improvement = ((arima_mae - sarima_mae) / arima_mae) * 100
        print(f"\nSARIMA vs ARIMA improvement: {improvement:.1f}%")
        if improvement > 0:
            print("✅ SARIMA performs better than ARIMA")
        else:
            print("⚠️  SARIMA does not improve over ARIMA")
    
    print("\n✅ SARIMA testing completed!")
    return results

if __name__ == "__main__":
    results = test_sarima_model()