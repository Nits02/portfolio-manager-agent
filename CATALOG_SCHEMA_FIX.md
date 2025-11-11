# Notebook Catalog/Schema Mismatch Resolution

## ✅ Issue Resolved

**Problem**: The predictive modeling notebook (`05_predictive_modeling.ipynb`) was looking for feature tables that didn't exist because of a catalog/schema naming mismatch between the feature engineering and predictive modeling notebooks.

**Error Message**:
```
🔍 Verifying feature tables...
❌ AAPL: Table portfolio_catalog.portfolio_schema.features_AAPL does not exist
❌ MSFT: Table portfolio_catalog.portfolio_schema.features_MSFT does not exist
📋 Available tickers for training: []
⚠️  No feature tables available. Please run feature engineering first.
```

## 🔧 Root Cause Analysis

### Before Fix:
- **Feature Engineering Notebook** (`03_feature_engineering.ipynb`):
  - Catalog: `"main"`
  - Schema: `"finance"`
  - Created tables: `main.finance.features_AAPL`, `main.finance.features_MSFT`

- **Predictive Modeling Notebook** (`05_predictive_modeling.ipynb`):
  - Catalog: `"portfolio_catalog"`
  - Schema: `"portfolio_schema"`
  - Expected tables: `portfolio_catalog.portfolio_schema.features_AAPL`, `portfolio_catalog.portfolio_schema.features_MSFT`

### After Fix:
- **Both notebooks now use consistent naming**:
  - Catalog: `"portfolio_catalog"`
  - Schema: `"portfolio_schema"`
  - Tables: `portfolio_catalog.portfolio_schema.features_<ticker>`

## 📝 Changes Made

### 1. Feature Engineering Notebook Updates (`03_feature_engineering.ipynb`)

**Cell 6 - Agent Initialization:**
```python
# Before
feature_agent = FeatureEngineeringAgent(catalog="main", schema="finance")

# After  
feature_agent = FeatureEngineeringAgent(catalog="portfolio_catalog", schema="portfolio_schema")
```

**Cell 8 - Unity Catalog Setup:**
```python
# Before
setup_unity_catalog("main", "finance")

# After
setup_unity_catalog("portfolio_catalog", "portfolio_schema")
```

**Cell 10 - Schema Search Order:**
```python
# Updated possible_schemas to include portfolio_catalog.portfolio_schema first
possible_schemas = [
    "portfolio_catalog.portfolio_schema",  # Current target schema
    "finance_catalog.bronze",              # Data ingestion agent schema  
    "main.finance",                        # Legacy schema
    "main.default"                         # Default schema fallback
]
```

**Cell 12 - Feature Table Creation:**
```python
# Before
feature_table_name = f"main.finance.features_{ticker}"

# After
feature_table_name = f"portfolio_catalog.portfolio_schema.features_{ticker}"
```

**Cell 14 - Table Validation:**
```python
# Before
table_name = f"main.finance.features_{ticker}"

# After
table_name = f"portfolio_catalog.portfolio_schema.features_{ticker}"
```

**Cell 22 - Summary Output:**
```python
# Before
print(f"   - main.finance.features_{ticker}")

# After
print(f"   - portfolio_catalog.portfolio_schema.features_{ticker}")
```

## 🎯 Expected Outcome

After running the updated feature engineering notebook:
1. ✅ Unity Catalog will create `portfolio_catalog.portfolio_schema` if it doesn't exist
2. ✅ Feature tables will be created as `portfolio_catalog.portfolio_schema.features_AAPL` and `portfolio_catalog.portfolio_schema.features_MSFT`
3. ✅ Predictive modeling notebook will find the expected tables
4. ✅ Model training will proceed successfully

## 🚀 Next Steps

1. **Run Feature Engineering**: Execute the updated `03_feature_engineering.ipynb` notebook
2. **Verify Tables**: Confirm that the feature tables are created in the correct location
3. **Run Predictive Modeling**: Execute `05_predictive_modeling.ipynb` to train the model
4. **Monitor Pipeline**: Ensure the job pipeline uses consistent naming

## 📊 Verification Query

To confirm the tables exist after running feature engineering:

```sql
-- Check if catalog and schema exist
SHOW CATALOGS LIKE 'portfolio_catalog';
SHOW SCHEMAS IN portfolio_catalog LIKE 'portfolio_schema';

-- Check feature tables
SHOW TABLES IN portfolio_catalog.portfolio_schema LIKE 'features_*';

-- Verify table contents
SELECT COUNT(*) FROM portfolio_catalog.portfolio_schema.features_AAPL;
SELECT COUNT(*) FROM portfolio_catalog.portfolio_schema.features_MSFT;
```

## ✅ Status: **RESOLVED**

The catalog/schema mismatch has been fixed and both notebooks now use consistent Unity Catalog naming conventions. The issue should be resolved after re-running the feature engineering notebook.