# Portfolio Price Prediction - Streamlit App

Interactive Streamlit application for real-time portfolio price direction prediction.

## Overview

This Streamlit application provides an interactive interface for:
- Model selection from Unity Catalog
- Ticker and date selection  
- Real-time feature engineering
- Price direction prediction with confidence scores
- Historical performance visualization

## Features

- **Model Management**: Load trained models from Unity Catalog
- **Interactive Predictions**: Real-time price direction forecasting
- **Feature Engineering**: Automatic calculation of technical indicators
- **Visualization**: Performance charts and model confidence metrics
- **Multi-Ticker Support**: Predictions for multiple stock symbols

## Usage

### Local Development
```bash
# Install dependencies
pip install streamlit pandas numpy matplotlib seaborn

# Run the app
cd streamlit_app/
streamlit run app.py
```

### Databricks Deployment
1. Upload `app.py` to Databricks workspace
2. Run as Databricks App or in notebook environment
3. Ensure Unity Catalog access and trained models available

## Prerequisites

- Trained models registered in Unity Catalog (from 05_predictive_modeling.ipynb)
- Feature tables in Unity Catalog silver schema
- Databricks environment with MLflow integration

## Architecture

The app integrates with:
- **Unity Catalog**: Model and data access
- **MLflow**: Model loading and tracking
- **PySpark**: Feature engineering and data processing
- **Streamlit**: Interactive web interface

## Files

- `app.py`: Main Streamlit application
- `README.md`: This documentation