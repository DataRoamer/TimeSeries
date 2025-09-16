"""
Test script for the updated forecasting models with SARIMA and LSTM
"""

import sys
import os
sys.path.append('TimeSeries/src')

import pandas as pd
import numpy as np
from models.forecasting import (
    SARIMAForecaster, 
    LSTMForecaster, 
    ARIMAForecaster,
    RandomForestForecaster,
    ModelSelector
)

def test_new_models():
    """Test SARIMA and LSTM models"""
    
    print("Loading NYC taxi data...")
    df = pd.read_csv('TimeSeries/data/raw/nyc_taxi.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Use a smaller subset for testing
    data = df['value'][:1000]  # First 1000 points for faster testing
    
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
        sarima_model.fit(train_data)
        sarima_forecast = sarima_model.predict(len(test_data))
        sarima_mae = np.mean(np.abs(test_data - sarima_forecast))
        results['SARIMA'] = {'mae': sarima_mae, 'status': 'Success'}
        print(f"   SARIMA MAE: {sarima_mae:.2f}")
    except Exception as e:
        results['SARIMA'] = {'mae': float('inf'), 'status': f'Failed: {str(e)}'}
        print(f"   SARIMA failed: {e}")
    
    # Test LSTM
    print("\n2. Testing LSTM model...")
    try:
        lstm_model = LSTMForecaster(sequence_length=24, hidden_units=25, epochs=10)  # Reduced for testing
        lstm_model.fit(train_data)
        lstm_forecast = lstm_model.predict(len(test_data))
        lstm_mae = np.mean(np.abs(test_data - lstm_forecast))
        results['LSTM'] = {'mae': lstm_mae, 'status': 'Success'}
        print(f"   LSTM MAE: {lstm_mae:.2f}")
    except Exception as e:
        results['LSTM'] = {'mae': float('inf'), 'status': f'Failed: {str(e)}'}
        print(f"   LSTM failed: {e}")
    
    # Test existing ARIMA for comparison
    print("\n3. Testing ARIMA model (for comparison)...")
    try:
        arima_model = ARIMAForecaster(order=(1, 1, 1))
        arima_model.fit(train_data)
        arima_forecast = arima_model.predict(len(test_data))
        arima_mae = np.mean(np.abs(test_data - arima_forecast))
        results['ARIMA'] = {'mae': arima_mae, 'status': 'Success'}
        print(f"   ARIMA MAE: {arima_mae:.2f}")
    except Exception as e:
        results['ARIMA'] = {'mae': float('inf'), 'status': f'Failed: {str(e)}'}
        print(f"   ARIMA failed: {e}")
    
    # Test ModelSelector with new models
    print("\n4. Testing ModelSelector with all models...")
    try:
        # Create a minimal set for testing
        models = {
            'arima': ARIMAForecaster(order=(1, 1, 1)),
            'sarima': SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)),
            'lstm': LSTMForecaster(sequence_length=24, hidden_units=25, epochs=5)
        }
        
        selector = ModelSelector(models=models)
        
        # Use even smaller test set for model selection
        mini_train = train_data[:400]
        mini_test = train_data[400:450]  # Small test set
        
        best_name, best_model, scores = selector.select_best_model(mini_train, mini_test)
        print(f"   Best model: {best_name}")
        print(f"   Model scores: {scores}")
        results['ModelSelector'] = {'best_model': best_name, 'status': 'Success'}
    except Exception as e:
        results['ModelSelector'] = {'best_model': 'None', 'status': f'Failed: {str(e)}'}
        print(f"   ModelSelector failed: {e}")
    
    # Summary
    print("\n" + "="*50)
    print("TESTING SUMMARY")
    print("="*50)
    for model, result in results.items():
        status = result['status']
        if model != 'ModelSelector':
            mae = result['mae']
            if mae != float('inf'):
                print(f"{model:<15}: {status:<10} MAE: {mae:.2f}")
            else:
                print(f"{model:<15}: {status}")
        else:
            best = result['best_model']
            print(f"{model:<15}: {status:<10} Best: {best}")
    
    print("\n✅ Testing completed!")
    return results

if __name__ == "__main__":
    results = test_new_models()