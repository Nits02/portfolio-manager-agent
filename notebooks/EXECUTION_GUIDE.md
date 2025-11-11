# Portfolio Manager Agent - Notebook Execution Sequence

This document outlines the proper sequence for running the portfolio management notebooks in Databricks.

## 📋 Notebook Execution Order

### Phase 1: Setup and Data Ingestion
```
00_setup_workspace.ipynb          → Initial workspace configuration
01_ingest_financial_data.ipynb    → Raw data ingestion from APIs
02_validate_ingest.ipynb          → Data quality validation
```

### Phase 2: Feature Engineering
```
03_feature_engineering.ipynb      → Technical indicator calculation
04_validate_features.ipynb        → Feature quality validation
```

### Phase 3: Model Training and Deployment
```
05-predictive-model-agent-demo.ipynb  → Demo of modeling capabilities (optional)
```
05_predictive_modeling.ipynb          → Production model training
06_inference_app.py                   → Streamlit inference application
```
```

## 🎯 Detailed Execution Guide

### 1. Initial Setup (00_setup_workspace.ipynb)
**Purpose:** Configure Unity Catalog, schemas, and permissions
**Prerequisites:** None
**Outputs:** 
- Unity Catalog: `portfolio_catalog`
- Schema: `portfolio_schema`
- Proper permissions setup

**Run when:**
- First time setup
- Switching environments (dev/staging/prod)

---

### 2. Data Ingestion (01_ingest_financial_data.ipynb)
**Purpose:** Fetch and store raw financial data
**Prerequisites:** Workspace setup completed
**Outputs:**
- Raw price data tables
- Volume and market data
- Historical data spanning configured period

**Run when:**
- Initial data load
- Daily/weekly data refresh
- Adding new tickers

---

### 3. Ingestion Validation (02_validate_ingest.ipynb)
**Purpose:** Verify data quality and completeness
**Prerequisites:** Data ingestion completed
**Outputs:**
- Data quality reports
- Missing data identification
- Schema validation results

**Run when:**
- After each data ingestion
- Troubleshooting data issues
- Regular data health checks

---

### 4. Feature Engineering (03_feature_engineering.ipynb)
**Purpose:** Calculate technical indicators and features
**Prerequisites:** Validated raw data
**Outputs:**
- Feature tables (one per ticker)
- Technical indicators (SMA, RSI, volatility, etc.)
- Engineered features ready for ML

**Run when:**
- After data ingestion/validation
- Adding new features
- Regular feature refresh

---

### 5. Feature Validation (04_validate_features.ipynb)
**Purpose:** Verify feature quality and distribution
**Prerequisites:** Feature engineering completed
**Outputs:**
- Feature quality metrics
- Distribution analysis
- Feature correlation reports

**Run when:**
- After feature engineering
- Before model training
- Feature quality monitoring

---

### 6. Model Training Demo (05-predictive-model-agent-demo.ipynb) [OPTIONAL]
**Purpose:** Educational overview of modeling capabilities
**Prerequisites:** Feature engineering completed
**Outputs:**
- Conceptual understanding
- Architecture overview
- Capability demonstration

**Run when:**
- Learning about the system
- Demonstrating capabilities
- Understanding model architecture

---

### 5. Production Model Training (05_predictive_modeling.ipynb)
**Purpose:** Train production-ready ML models
**Prerequisites:** Validated features available
**Outputs:**
- Trained GBT classifier
- Performance metrics and visualizations
- Registered model in Unity Catalog

**Run when:**
- Ready for production model training
- Model retraining (weekly/monthly)
- Performance evaluation needed

---

### 6. Inference Application (06_inference_app.py)
**Purpose:** Interactive prediction interface
**Prerequisites:** Trained model registered in Unity Catalog
**Outputs:**
- Real-time predictions
- Interactive UI
- Model confidence scores

**Run when:**
- Model is trained and registered
- Need interactive predictions
- Demonstrating model capabilities

## 🚀 Databricks Deployment Instructions

### Step 1: Upload Notebooks to Databricks

1. **Import notebooks to Databricks workspace:**
   ```bash
   # Upload each notebook to /Workspace/portfolio-manager-agent/
   - 00_setup_workspace.ipynb
   - 01_ingest_financial_data.ipynb
   - 02_validate_ingest.ipynb
   - 03_feature_engineering.ipynb
   - 04_validate_features.ipynb
   - 05-predictive-model-agent-demo.ipynb
   - 05_predictive_modeling.ipynb
   ```

2. **Deploy Streamlit App:**
   ```bash
   # Upload 06_inference_app.py to Databricks Apps or run as notebook
   # Databricks Apps location: /Workspace/portfolio-manager-agent/apps/
   ```

### Step 2: Cluster Configuration

**Recommended Cluster Settings:**
```yaml
Databricks Runtime: 13.3 LTS ML or higher
Worker Type: i3.xlarge (or similar)
Driver Type: i3.xlarge (or similar)
Min Workers: 1
Max Workers: 4
Auto Termination: 120 minutes

Python Libraries:
- yfinance
- pandas
- numpy
- matplotlib
- seaborn
- streamlit (for inference app)

Spark Config:
spark.sql.adaptive.enabled true
spark.sql.adaptive.coalescePartitions.enabled true
spark.sql.execution.arrow.pyspark.enabled true
```

### Step 3: Execution Schedule

**Daily Operations:**
1. Run `01_ingest_financial_data.ipynb` (automated via jobs)
2. Run `02_validate_ingest.ipynb` (data quality check)

**Weekly Operations:**
1. Run `03_feature_engineering.ipynb` (refresh features)
2. Run `04_validate_features.ipynb` (feature quality)

**Monthly Operations:**
1. Run `05_predictive_modeling.ipynb` (retrain models)
2. Update `06_inference_app.py` with new model versions

**Continuous:**
- `06_inference_app.py` runs continuously for predictions

### Step 4: Automation Setup

**Create Databricks Jobs:**

1. **Daily Data Pipeline:**
   ```yaml
   Job Name: "portfolio-daily-data-pipeline"
   Tasks:
     - 01_ingest_financial_data.ipynb
     - 02_validate_ingest.ipynb
   Schedule: Daily at 6 PM EST (after market close)
   ```

2. **Weekly Feature Pipeline:**
   ```yaml
   Job Name: "portfolio-weekly-features"
   Tasks:
     - 03_feature_engineering.ipynb
     - 04_validate_features.ipynb
   Schedule: Weekly on Sunday at 8 PM EST
   ```

3. **Monthly Model Training:**
   ```yaml
   Job Name: "portfolio-monthly-training"
   Tasks:
     - 05_predictive_modeling.ipynb
   Schedule: First Monday of each month at 10 PM EST
   ```

## 🔧 Environment-Specific Configurations

### Development Environment
```python
CATALOG_NAME = "main"
SCHEMA_NAME = "finance_dev"
MODEL_NAME_PREFIX = "dev_"
```

### Staging Environment
```python
CATALOG_NAME = "portfolio_catalog"
SCHEMA_NAME = "staging_schema"
MODEL_NAME_PREFIX = "staging_"
```

### Production Environment
```python
CATALOG_NAME = "portfolio_catalog"
SCHEMA_NAME = "portfolio_schema"
MODEL_NAME_PREFIX = "prod_"
```

## 🎯 Success Criteria

After running the complete sequence, you should have:

✅ **Data Infrastructure:**
- Unity Catalog with organized schemas
- Raw data tables with historical prices
- Feature tables with technical indicators

✅ **ML Pipeline:**
- Trained and validated models
- Model registry in Unity Catalog
- Performance metrics and monitoring

✅ **Applications:**
- Interactive prediction interface
- Automated data pipelines
- Model retraining workflows

## 🚨 Troubleshooting

**Common Issues:**

1. **Permission Errors:**
   - Ensure Unity Catalog permissions are set
   - Verify cluster has access to external APIs

2. **Data Issues:**
   - Check API rate limits for financial data
   - Verify internet connectivity for data fetching

3. **Model Training Failures:**
   - Ensure sufficient data (minimum 6 months)
   - Check feature table completeness

4. **Streamlit App Issues:**
   - Verify model registration in Unity Catalog
   - Check cluster libraries installation

**Support Resources:**
- Check notebook logs for detailed error messages
- Review Databricks cluster event logs
- Monitor MLflow experiment tracking for model issues

## 📞 Contact Information

For technical support or questions about this pipeline:
- Portfolio Management Team
- Created: November 2025
- Repository: portfolio-manager-agent