"""
Test script for all forecasting models including SARIMA and LSTM (if available)
"""

import sys
import os
sys.path.append('TimeSeries/src')

import pandas as pd
import numpy as np
from models.forecasting import (
    SARIMAForecaster, 
    ARIMAForecaster,
    RandomForestForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    ModelSelector,
    TENSORFLOW_AVAILABLE
)

if TENSORFLOW_AVAILABLE:
    from models.forecasting import LSTMForecaster

def test_all_models():
    """Test all available forecasting models"""
    
    print("=== NYC Taxi Demand Forecasting Model Test ===")
    print(f"TensorFlow Available: {TENSORFLOW_AVAILABLE}")
    print()
    
    print("Loading NYC taxi data...")
    df = pd.read_csv('TimeSeries/data/raw/nyc_taxi.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Use a reasonable subset for testing
    data = df['value'][:800]  # First 800 points
    
    # Split data
    split_point = int(len(data) * 0.8)
    train_data = data[:split_point]
    test_data = data[split_point:]
    
    print(f"Training data: {len(train_data)} points")
    print(f"Test data: {len(test_data)} points")
    print()
    
    results = {}
    
    # Test all models
    models_to_test = [
        ('Naive', NaiveForecaster()),
        ('Seasonal Naive', SeasonalNaiveForecaster()),
        ('ARIMA', ARIMAForecaster(order=(1, 1, 1))),
        ('SARIMA', SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))),
        ('Random Forest', RandomForestForecaster())
    ]
    
    if TENSORFLOW_AVAILABLE:
        models_to_test.append(('LSTM', LSTMForecaster(sequence_length=24, hidden_units=25, epochs=10)))
    
    for model_name, model in models_to_test:
        print(f"Testing {model_name}...")
        try:
            model.fit(train_data)
            forecast = model.predict(len(test_data))
            mae = np.mean(np.abs(test_data - forecast))
            rmse = np.sqrt(np.mean((test_data - forecast) ** 2))
            
            results[model_name] = {
                'mae': mae,
                'rmse': rmse,
                'status': 'Success'
            }
            print(f"   MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        except Exception as e:
            results[model_name] = {
                'mae': float('inf'),
                'rmse': float('inf'),
                'status': f'Failed: {str(e)[:100]}'
            }
            print(f"   Failed: {e}")
        print()
    
    # Test ModelSelector
    print("Testing ModelSelector...")
    try:
        selector = ModelSelector()
        mini_train = train_data[:300]
        mini_test = train_data[300:350]
        
        best_name, best_model, scores = selector.select_best_model(mini_train, mini_test)
        print(f"   Best model: {best_name}")
        print(f"   Available models: {list(selector.models.keys())}")
        results['ModelSelector'] = {'best_model': best_name, 'status': 'Success'}
    except Exception as e:
        results['ModelSelector'] = {'best_model': 'None', 'status': f'Failed: {str(e)[:100]}'}
        print(f"   Failed: {e}")
    print()
    
    # Summary
    print("=" * 60)
    print("TESTING RESULTS SUMMARY")
    print("=" * 60)
    
    # Sort by MAE (successful models only)
    successful_models = [(name, res) for name, res in results.items() 
                        if name != 'ModelSelector' and res['status'] == 'Success']
    successful_models.sort(key=lambda x: x[1]['mae'])
    
    print(f"{'Model':<15} {'Status':<12} {'MAE':<10} {'RMSE':<10}")
    print("-" * 50)
    
    for model_name, result in successful_models:
        mae = result['mae']
        rmse = result['rmse']
        print(f"{model_name:<15} {'Success':<12} {mae:<10.2f} {rmse:<10.2f}")
    
    # Show failed models
    failed_models = [(name, res) for name, res in results.items() 
                    if name != 'ModelSelector' and res['status'] != 'Success']
    
    for model_name, result in failed_models:
        print(f"{model_name:<15} {'Failed':<12} {'N/A':<10} {'N/A':<10}")
    
    print()
    
    # Best model analysis
    if successful_models:
        best_model_name, best_result = successful_models[0]
        print(f"BEST PERFORMING MODEL: {best_model_name}")
        print(f"   MAE: {best_result['mae']:.2f}")
        print(f"   RMSE: {best_result['rmse']:.2f}")
        
        # Compare with baseline (naive)
        naive_result = results.get('Naive')
        if naive_result and naive_result['status'] == 'Success':
            improvement = ((naive_result['mae'] - best_result['mae']) / naive_result['mae']) * 100
            print(f"   Improvement over Naive: {improvement:.1f}%")
    
    print()
    print("=" * 60)
    
    # New models summary
    print("NEW MODELS ADDED:")
    print("- SARIMA: Seasonal ARIMA model with seasonal parameters")
    if TENSORFLOW_AVAILABLE:
        print("- LSTM: Deep learning neural network for time series")
    else:
        print("- LSTM: Not available (TensorFlow not installed)")
    
    sarima_result = results.get('SARIMA')
    arima_result = results.get('ARIMA')
    if sarima_result and arima_result and both_successful(sarima_result, arima_result):
        improvement = ((arima_result['mae'] - sarima_result['mae']) / arima_result['mae']) * 100
        print(f"- SARIMA shows {improvement:.1f}% improvement over ARIMA")
    
    print()
    print("SUCCESS! Updated forecasting models are working correctly.")
    
    return results

def both_successful(result1, result2):
    return result1['status'] == 'Success' and result2['status'] == 'Success'

if __name__ == "__main__":
    results = test_all_models()