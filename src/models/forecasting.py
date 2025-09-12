"""
Forecasting models for time series analysis
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings


class BaseForecaster:
    """Base class for forecasting models"""
    
    def __init__(self):
        self.model = None
        self.fitted = False
    
    def fit(self, data: pd.Series) -> None:
        """Fit the model to the data"""
        raise NotImplementedError
    
    def predict(self, steps: int) -> np.ndarray:
        """Make forecast for specified steps"""
        raise NotImplementedError
    
    def forecast_with_intervals(self, steps: int, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make forecast with prediction intervals"""
        forecast = self.predict(steps)
        return forecast, forecast, forecast  # Base implementation returns point forecast


class NaiveForecaster(BaseForecaster):
    """Naive forecasting (last value)"""
    
    def fit(self, data: pd.Series) -> None:
        self.last_value = data.iloc[-1]
        self.fitted = True
    
    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return np.full(steps, self.last_value)


class SeasonalNaiveForecaster(BaseForecaster):
    """Seasonal naive forecasting"""
    
    def __init__(self, season_length: int = 24):
        super().__init__()
        self.season_length = season_length
    
    def fit(self, data: pd.Series) -> None:
        self.data = data
        self.fitted = True
    
    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        seasonal_values = self.data.iloc[-self.season_length:]
        forecast = []
        
        for i in range(steps):
            forecast.append(seasonal_values.iloc[i % self.season_length])
        
        return np.array(forecast)


class ARIMAForecaster(BaseForecaster):
    """ARIMA forecasting model"""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        super().__init__()
        self.order = order
    
    def fit(self, data: pd.Series) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.model = ARIMA(data, order=self.order)
            self.fitted_model = self.model.fit()
        self.fitted = True
    
    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.fitted_model.forecast(steps=steps)
    
    def forecast_with_intervals(self, steps: int, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecast_result = self.fitted_model.get_forecast(steps=steps, alpha=alpha)
        forecast = forecast_result.predicted_mean.values
        confidence_int = forecast_result.conf_int()
        
        return forecast, confidence_int.iloc[:, 0].values, confidence_int.iloc[:, 1].values


class ExponentialSmoothingForecaster(BaseForecaster):
    """Exponential Smoothing forecasting model"""
    
    def __init__(self, trend: Optional[str] = 'add', seasonal: Optional[str] = 'add', 
                 seasonal_periods: int = 24):
        super().__init__()
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
    
    def fit(self, data: pd.Series) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.model = ExponentialSmoothing(data, 
                                            trend=self.trend,
                                            seasonal=self.seasonal,
                                            seasonal_periods=self.seasonal_periods)
            self.fitted_model = self.model.fit()
        self.fitted = True
    
    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.fitted_model.forecast(steps=steps)


class RandomForestForecaster(BaseForecaster):
    """Random Forest forecasting model"""
    
    def __init__(self, lags: list = [1, 2, 3, 24, 48], n_estimators: int = 100):
        super().__init__()
        self.lags = lags
        self.n_estimators = n_estimators
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    
    def _create_features(self, data: pd.Series) -> pd.DataFrame:
        """Create lag features for ML model"""
        df = pd.DataFrame({'value': data})
        
        # Add lag features
        for lag in self.lags:
            df[f'lag_{lag}'] = df['value'].shift(lag)
        
        # Add time features if data has datetime index
        if isinstance(data.index, pd.DatetimeIndex):
            df['hour'] = data.index.hour
            df['day_of_week'] = data.index.dayofweek
            df['month'] = data.index.month
        
        return df.dropna()
    
    def fit(self, data: pd.Series) -> None:
        df_features = self._create_features(data)
        
        X = df_features.drop('value', axis=1)
        y = df_features['value']
        
        self.model.fit(X, y)
        self.feature_columns = X.columns.tolist()
        self.last_values = data.iloc[-max(self.lags):].values
        self.fitted = True
    
    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecast = []
        current_values = self.last_values.copy()
        
        for _ in range(steps):
            # Create features for next prediction
            features = []
            for lag in self.lags:
                if lag <= len(current_values):
                    features.append(current_values[-lag])
                else:
                    features.append(0)  # Default value for missing lags
            
            # Add dummy time features (could be improved with actual future dates)
            if len(self.feature_columns) > len(self.lags):
                features.extend([0] * (len(self.feature_columns) - len(self.lags)))
            
            # Make prediction
            X_pred = np.array(features).reshape(1, -1)
            pred = self.model.predict(X_pred)[0]
            
            # Update current values for next prediction
            current_values = np.append(current_values, pred)
            forecast.append(pred)
        
        return np.array(forecast)


class EnsembleForecaster(BaseForecaster):
    """Ensemble of multiple forecasting models"""
    
    def __init__(self, models: list, weights: Optional[list] = None):
        super().__init__()
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
    
    def fit(self, data: pd.Series) -> None:
        for model in self.models:
            model.fit(data)
        self.fitted = True
    
    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecasts = []
        for model in self.models:
            forecasts.append(model.predict(steps))
        
        # Weighted average of forecasts
        ensemble_forecast = np.average(forecasts, axis=0, weights=self.weights)
        return ensemble_forecast


class ModelSelector:
    """Automatically select best forecasting model"""
    
    def __init__(self, models: Optional[Dict[str, BaseForecaster]] = None):
        if models is None:
            self.models = {
                'naive': NaiveForecaster(),
                'seasonal_naive': SeasonalNaiveForecaster(),
                'arima': ARIMAForecaster(),
                'exp_smoothing': ExponentialSmoothingForecaster(),
                'random_forest': RandomForestForecaster()
            }
        else:
            self.models = models
    
    def select_best_model(self, train_data: pd.Series, test_data: pd.Series, 
                         metric: str = 'mae') -> Tuple[str, BaseForecaster, Dict[str, float]]:
        """
        Select best model based on test performance
        
        Parameters:
        -----------
        train_data : pd.Series
            Training data
        test_data : pd.Series
            Test data for model selection
        metric : str
            Evaluation metric ('mae', 'mse', 'rmse')
        
        Returns:
        --------
        Tuple[str, BaseForecaster, Dict[str, float]]
            Best model name, fitted model, and all model scores
        """
        scores = {}
        fitted_models = {}
        
        for name, model in self.models.items():
            try:
                # Fit model on training data
                model_copy = type(model)(**model.__dict__)
                model_copy.fit(train_data)
                
                # Make predictions on test data
                predictions = model_copy.predict(len(test_data))
                
                # Calculate metric
                if metric == 'mae':
                    score = mean_absolute_error(test_data, predictions)
                elif metric == 'mse':
                    score = mean_squared_error(test_data, predictions)
                elif metric == 'rmse':
                    score = np.sqrt(mean_squared_error(test_data, predictions))
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                
                scores[name] = score
                fitted_models[name] = model_copy
                
            except Exception as e:
                print(f"Failed to fit {name}: {str(e)}")
                scores[name] = float('inf')
        
        # Select best model (lowest score)
        best_model_name = min(scores, key=scores.get)
        best_model = fitted_models[best_model_name]
        
        return best_model_name, best_model, scores
    
    def cross_validate(self, data: pd.Series, n_splits: int = 5, 
                      test_size: int = 48, metric: str = 'mae') -> pd.DataFrame:
        """
        Perform time series cross-validation
        
        Parameters:
        -----------
        data : pd.Series
            Time series data
        n_splits : int
            Number of cross-validation splits
        test_size : int
            Size of test set for each split
        metric : str
            Evaluation metric
        
        Returns:
        --------
        pd.DataFrame
            Cross-validation results for each model
        """
        results = []
        total_len = len(data)
        min_train_size = total_len - n_splits * test_size
        
        for fold in range(n_splits):
            train_end = min_train_size + fold * test_size
            test_start = train_end
            test_end = test_start + test_size
            
            if test_end > total_len:
                break
            
            train_data = data[:train_end]
            test_data = data[test_start:test_end]
            
            best_name, best_model, scores = self.select_best_model(train_data, test_data, metric)
            
            for model_name, score in scores.items():
                results.append({
                    'fold': fold + 1,
                    'model': model_name,
                    'score': score
                })
        
        return pd.DataFrame(results)