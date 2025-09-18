# 🏭 Industrial Pollution Detection Pro

Advanced pollution source detection using machine learning and comprehensive data analysis.

## Features

- **🏠 Home Dashboard**: Project overview with interactive maps and key statistics
- **📊 Data Explorer**: Comprehensive dataset analysis with interactive visualizations
- **🎯 Quick Prediction**: Real-time pollution source attribution using ML models
- **📈 Batch Analysis**: Multi-scenario testing and batch predictions
- **🗺️ Scenario Testing**: Pre-built pollution scenarios
- **ℹ️ Model Information**: Model performance metrics and feature importance

## Machine Learning Models

- **Random Forest Ensemble**: 50-tree ensemble for high-accuracy predictions
- **Bayesian Ridge Regression**: Probabilistic model with uncertainty quantification
- **Real-time Training**: Models train automatically in ~10 seconds
- **Performance**: R² > 0.85 for ensemble model

## Dataset

- 25 industrial facilities across different types
- 12 monitoring stations (urban, industrial, residential, background)
- 90 days of weather and pollution measurements
- Multiple pollutants: PM2.5, PM10, SO2, NO2, CO, O3

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

3. Open your browser to `http://localhost:8501`

## Deployment

This app is ready for deployment on:
- Streamlit Cloud
- Heroku
- AWS
- Google Cloud Platform
- Any Python hosting platform

## Usage

1. **Home**: View project overview and facility locations
2. **Data Explorer**: Analyze datasets with interactive charts
3. **Quick Prediction**: Adjust parameters to predict pollution attribution
4. **Batch Analysis**: Test multiple scenarios simultaneously
5. **Scenario Test**: Run predefined pollution scenarios
6. **Model Info**: View model performance and feature importance

## Model Interpretation

- **> 0.7**: High confidence pollution source - immediate action recommended
- **0.4 - 0.7**: Moderate confidence - further investigation needed
- **< 0.4**: Low confidence - unlikely significant source

## Technology Stack

- **Frontend**: Streamlit
- **ML Models**: scikit-learn (Random Forest, Bayesian Ridge)
- **Visualization**: Plotly
- **Data Processing**: pandas, numpy
- **Maps**: Plotly Mapbox 
