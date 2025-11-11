"""
Portfolio Price Prediction Inference App

A Databricks Streamlit application for real-time portfolio price direction prediction.
This app loads trained models from Unity Catalog and provides an interactive interface
for making predictions on stock price movements.

Features:
- Model selection from Unity Catalog
- Ticker and date selection
- Real-time feature engineering
- Price direction prediction with confidence scores
- Historical performance visualization

Author: Portfolio Management Team
Created: November 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings('ignore')

# Databricks specific imports
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Custom imports
import sys
import os
sys.path.append('../src')

try:
    from agents.predictive_model_agent import PredictiveModelAgent
    from agents.feature_engineering_agent import FeatureEngineeringAgent
except ImportError:
    st.error("Unable to import custom agents. Please ensure the src directory is accessible.")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Portfolio Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App configuration
CATALOG_NAME = "portfolio_catalog"
SCHEMA_NAME = "portfolio_schema"
DEFAULT_MODEL_NAME = "portfolio_price_predictor_aapl_msft"

# Available tickers for prediction
AVAILABLE_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']

@st.cache_resource
def initialize_spark():
    """Initialize Spark session with caching for performance."""
    try:
        spark = SparkSession.builder \
            .appName("PortfolioPredictionApp") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        return spark
    except Exception as e:
        st.error(f"Failed to initialize Spark session: {str(e)}")
        return None

@st.cache_resource
def initialize_agents(_spark):
    """Initialize prediction and feature engineering agents."""
    try:
        pred_agent = PredictiveModelAgent(
            catalog=CATALOG_NAME,
            schema=SCHEMA_NAME
        )
        
        feature_agent = FeatureEngineeringAgent(
            catalog=CATALOG_NAME,
            schema=SCHEMA_NAME
        )
        
        return pred_agent, feature_agent
    except Exception as e:
        st.error(f"Failed to initialize agents: {str(e)}")
        return None, None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_models():
    """Get list of available models from Unity Catalog."""
    try:
        client = mlflow.MlflowClient()
        models = client.search_registered_models(filter_string="name LIKE 'portfolio_%'")
        return [(model.name, model.latest_versions[0].version if model.latest_versions else "1") 
                for model in models]
    except Exception as e:
        st.warning(f"Could not fetch models from Unity Catalog: {str(e)}")
        return [(DEFAULT_MODEL_NAME, "latest")]

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_latest_features(_spark, ticker, as_of_date):
    """Get the latest features for a ticker as of a specific date."""
    try:
        table_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.features_{ticker}"
        
        # Check if table exists
        if not _spark.catalog.tableExists(table_name):
            return None, f"Feature table for {ticker} does not exist"
        
        # Get features as of the specified date
        df = _spark.table(table_name) \
            .filter(F.col("date") <= as_of_date) \
            .orderBy(F.col("date").desc()) \
            .limit(1)
        
        if df.count() == 0:
            return None, f"No features found for {ticker} as of {as_of_date}"
        
        # Convert to pandas for easier handling
        features_pdf = df.toPandas()
        return features_pdf, None
        
    except Exception as e:
        return None, f"Error fetching features: {str(e)}"

def load_model(model_name, model_version="latest"):
    """Load model from Unity Catalog."""
    try:
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.spark.load_model(model_uri)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {str(e)}"

def make_prediction(model, features_df, spark):
    """Make prediction using the loaded model."""
    try:
        # Convert pandas DataFrame back to Spark DataFrame
        spark_df = spark.createDataFrame(features_df)
        
        # Make prediction
        predictions = model.transform(spark_df)
        
        # Extract prediction and probability
        result = predictions.select("prediction", "probability").collect()[0]
        prediction = int(result["prediction"])
        probability_vector = result["probability"].toArray()
        
        # Get confidence (probability of predicted class)
        confidence = float(probability_vector[prediction])
        
        return prediction, confidence, None
        
    except Exception as e:
        return None, None, f"Prediction failed: {str(e)}"

def create_confidence_gauge(confidence, prediction):
    """Create a confidence gauge visualization."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    radius = 1
    
    # Background arc
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), 'lightgray', linewidth=10)
    
    # Confidence arc
    confidence_theta = np.linspace(0, confidence * np.pi, int(confidence * 100))
    color = 'green' if prediction == 1 else 'red'
    ax.plot(radius * np.cos(confidence_theta), radius * np.sin(confidence_theta), 
            color, linewidth=10)
    
    # Add text
    ax.text(0, -0.3, f'{confidence:.1%}', ha='center', va='center', 
            fontsize=20, fontweight='bold')
    ax.text(0, -0.5, 'Confidence', ha='center', va='center', fontsize=12)
    
    # Direction indicator
    direction = "UP ↗" if prediction == 1 else "DOWN ↘"
    ax.text(0, 0.3, direction, ha='center', va='center', 
            fontsize=16, fontweight='bold', color=color)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.7, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Price Direction Prediction', fontsize=14, fontweight='bold')
    
    return fig

def display_feature_importance(features_df):
    """Display the input features used for prediction."""
    st.subheader("📊 Input Features")
    
    # Define feature descriptions
    feature_descriptions = {
        'daily_return': 'Daily price return (%)',
        'moving_avg_7': '7-day moving average',
        'moving_avg_30': '30-day moving average', 
        'volatility_7': '7-day volatility',
        'momentum': 'Price momentum indicator'
    }
    
    # Create feature display
    cols = st.columns(len(feature_descriptions))
    
    for i, (feature, description) in enumerate(feature_descriptions.items()):
        if feature in features_df.columns:
            value = features_df[feature].iloc[0]
            with cols[i]:
                st.metric(
                    label=description,
                    value=f"{value:.4f}",
                    help=f"Feature: {feature}"
                )

def main():
    """Main Streamlit application."""
    
    # App header
    st.title("📈 Portfolio Price Prediction App")
    st.markdown("---")
    
    # Initialize Spark and agents
    spark = initialize_spark()
    if spark is None:
        st.error("Cannot proceed without Spark session. Please check Databricks configuration.")
        st.stop()
    
    pred_agent, feature_agent = initialize_agents(spark)
    if pred_agent is None or feature_agent is None:
        st.error("Cannot proceed without agents. Please check agent initialization.")
        st.stop()
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Model selection
    available_models = get_available_models()
    model_options = [f"{name} (v{version})" for name, version in available_models]
    
    selected_model_display = st.sidebar.selectbox(
        "Select Model",
        model_options,
        help="Choose a trained model from Unity Catalog"
    )
    
    # Extract model name and version
    model_name = available_models[model_options.index(selected_model_display)][0]
    model_version = available_models[model_options.index(selected_model_display)][1]
    
    # Ticker selection
    selected_ticker = st.sidebar.selectbox(
        "Select Ticker",
        AVAILABLE_TICKERS,
        index=0,  # Default to AAPL
        help="Choose a stock ticker for prediction"
    )
    
    # Date selection
    max_date = date.today() - timedelta(days=1)  # Yesterday
    min_date = max_date - timedelta(days=365)    # One year ago
    
    selected_date = st.sidebar.date_input(
        "Prediction Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        help="Select the date for which to make the prediction"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"🎯 Prediction for {selected_ticker}")
        
        # Load model
        with st.spinner("Loading model..."):
            model, model_error = load_model(model_name, model_version)
        
        if model_error:
            st.error(model_error)
            return
        
        st.success(f"✅ Model loaded: {model_name} v{model_version}")
        
        # Get features
        with st.spinner("Fetching latest features..."):
            features_df, features_error = get_latest_features(spark, selected_ticker, selected_date)
        
        if features_error:
            st.error(features_error)
            st.info("💡 Try running the feature engineering notebook first, or select a different ticker/date.")
            return
        
        st.success(f"✅ Features loaded for {selected_ticker} as of {selected_date}")
        
        # Display input features
        display_feature_importance(features_df)
        
        # Make prediction button
        if st.button("🔮 Make Prediction", type="primary"):
            with st.spinner("Making prediction..."):
                prediction, confidence, pred_error = make_prediction(model, features_df, spark)
            
            if pred_error:
                st.error(pred_error)
                return
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            # Create columns for results
            result_col1, result_col2 = st.columns([1, 1])
            
            with result_col1:
                # Direction and confidence
                direction = "UP ↗" if prediction == 1 else "DOWN ↘"
                color = "green" if prediction == 1 else "red"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; border-radius: 10px; 
                           background-color: {'#e8f5e8' if prediction == 1 else '#f5e8e8'};">
                    <h2 style="color: {color}; margin: 0;">
                        Next Day Direction: {direction}
                    </h2>
                    <h3 style="color: {color}; margin: 10px 0;">
                        Confidence: {confidence:.1%}
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional metrics
                st.markdown("### 📊 Prediction Metrics")
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    st.metric("Prediction Class", prediction, help="0 = Down, 1 = Up")
                
                with metric_col2:
                    st.metric("Confidence Score", f"{confidence:.3f}", help="Model confidence (0-1)")
            
            with result_col2:
                # Confidence gauge
                fig = create_confidence_gauge(confidence, prediction)
                st.pyplot(fig)
    
    with col2:
        st.header("ℹ️ Model Information")
        
        # Model details
        st.markdown(f"""
        **Model Details:**
        - **Name:** {model_name}
        - **Version:** {model_version}
        - **Algorithm:** Gradient Boosted Trees
        - **Catalog:** {CATALOG_NAME}
        - **Schema:** {SCHEMA_NAME}
        
        **Prediction Target:**
        - Next-day price direction
        - Binary classification (Up/Down)
        - Based on technical indicators
        
        **Model Features:**
        - Daily returns
        - Moving averages (7d, 30d)
        - Volatility measures
        - Momentum indicators
        """)
        
        # Usage instructions
        st.markdown("---")
        st.markdown("""
        **How to use:**
        1. Select a trained model
        2. Choose ticker symbol
        3. Pick prediction date
        4. Click 'Make Prediction'
        5. Review results and confidence
        
        **Interpretation:**
        - **UP ↗**: Price likely to increase
        - **DOWN ↘**: Price likely to decrease
        - **Confidence**: Model certainty (higher is better)
        """)
        
        # Performance note
        st.info("💡 **Note:** Predictions are based on historical patterns and should be used as one factor in investment decisions.")

if __name__ == "__main__":
    main()