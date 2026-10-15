# TimeSeries Analysis Project

A comprehensive Python toolkit for time series analysis, forecasting, and visualization.

## 🚀 Project Structure

```
TimeSeries/
├── data/
│   ├── raw/          # Raw, unmodified data
│   ├── processed/    # Cleaned and preprocessed data
│   └── external/     # Data from external sources
├── notebooks/        # Jupyter notebooks for analysis
├── src/
│   ├── utils/        # Utility functions
│   ├── models/       # Time series models
│   └── visualization/ # Plotting functions
├── tests/           # Unit tests
├── docs/            # Documentation
├── scripts/         # Standalone scripts
└── requirements.txt # Python dependencies
```

## 📊 Features

- **Data Processing**: Clean, transform, and prepare time series data
- **Exploratory Analysis**: Statistical summaries and visualizations
- **Forecasting Models**: 
  - ARIMA/SARIMA
  - Prophet
  - Exponential Smoothing
  - Machine Learning approaches (RF, XGBoost, LSTM)
- **Model Evaluation**: Cross-validation, metrics, backtesting
- **Visualization**: Interactive plots with Plotly and static with Matplotlib

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/DataRoamer/TimeSeries.git
cd TimeSeries
```

2. Create a virtual environment:
```bash
python -m venv ts_env
source ts_env/bin/activate  # On Windows: ts_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📈 Quick Start

```python
import pandas as pd
from src.utils.data_loader import load_time_series
from src.models.forecasting import ARIMAForecaster

# Load your data
df = load_time_series('data/raw/your_data.csv')

# Create and fit model
model = ARIMAForecaster()
model.fit(df['value'])

# Generate forecast
forecast = model.predict(steps=30)
```

## 📁 Usage Examples

- `notebooks/01_data_exploration.ipynb` - Data loading and EDA
- `notebooks/02_forecasting_basics.ipynb` - Basic forecasting models
- `notebooks/03_advanced_models.ipynb` - ML and deep learning approaches
- `notebooks/04_model_evaluation.ipynb` - Performance assessment
- `notebooks/nyc_taxi_eda.ipynb` - NYC taxi demand exploratory analysis
- `notebooks/nyc_taxi_forecasting.ipynb` - NYC taxi demand forecasting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details