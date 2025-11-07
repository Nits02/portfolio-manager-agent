# Portfolio Manager Pipeline Deployment

This directory contains all the configuration and scripts needed to deploy and manage the Portfolio Manager Agent pipeline on Databricks and GitHub Actions.

## 🏗️ Architecture Overview

The pipeline consists of three main stages:

1. **Data Ingestion** - Fetches financial data from Yahoo Finance
2. **Feature Engineering** - Creates ML-ready features from raw data  
3. **Data Validation** - Validates data quality and feature completeness

## 📁 File Structure

```
infra/
├── databricks_config.yml          # Environment-specific configurations
├── job_ingest.json                # Data ingestion job configuration
├── job_feature_engineering.json   # Feature engineering job configuration
└── job_validate_features.json     # Data validation job configuration

.github/
├── workflows/
│   ├── ci.yml                     # Continuous integration
│   ├── feature-engineering.yml    # Feature engineering pipeline
│   ├── data-validation.yml        # Data validation pipeline
│   └── portfolio-pipeline.yml     # Complete pipeline orchestration
└── SECRETS_TEMPLATE.md           # GitHub secrets configuration guide

scripts/
└── deploy_pipeline.sh            # Manual deployment script

docker-compose.yml                # Local development environment
Dockerfile                       # Container for testing
```

## 🚀 Quick Start

### 1. Prerequisites

- Databricks workspace with Unity Catalog enabled
- GitHub repository with Actions enabled
- Databricks CLI installed (for manual deployment)

### 2. Configure Secrets

Follow the instructions in [`.github/SECRETS_TEMPLATE.md`](../.github/SECRETS_TEMPLATE.md) to set up required GitHub secrets.

### 3. Deploy via GitHub Actions

The easiest way to deploy is using the GitHub Actions workflow:

1. Go to your repository's Actions tab
2. Run the "Portfolio Manager Pipeline" workflow
3. Select your environment (dev/staging/prod)
4. Specify tickers to process
5. Monitor the execution

### 4. Manual Deployment

For manual deployment, use the provided script:

```bash
# Deploy to development environment
./scripts/deploy_pipeline.sh --environment dev --tickers "AAPL,MSFT"

# Deploy to production
./scripts/deploy_pipeline.sh --environment prod --tickers "AAPL,MSFT,GOOGL,AMZN,META"

# Check deployment status
./scripts/deploy_pipeline.sh --action status --environment prod

# Dry run (show what would be done)
./scripts/deploy_pipeline.sh --environment dev --dry-run
```

## 🔧 Configuration

### Environment Configuration

Edit `databricks_config.yml` to customize:

- Databricks workspace URLs
- Catalog and schema names
- Cluster policies
- Validation thresholds
- Default tickers

### Job Scheduling

Jobs are scheduled via cron expressions:

- **Data Ingestion**: 6:00 AM UTC daily
- **Feature Engineering**: 7:00 AM UTC daily  
- **Data Validation**: 7:30 AM UTC daily

### Data Quality Thresholds

Default validation thresholds:
- Max null percentage: 5%
- Max outlier percentage: 1%
- Min data completeness: 95%

## 🏃‍♂️ Running the Pipeline

### Automated Execution

The pipeline runs automatically on schedule or can be triggered:

1. **Scheduled**: Daily at 6:30 AM UTC
2. **Manual**: Via GitHub Actions "workflow_dispatch"
3. **Event-driven**: After successful data ingestion

### Manual Execution

```bash
# Run complete pipeline
./scripts/deploy_pipeline.sh --action run --environment prod

# Run individual components
databricks jobs run-now --job-id <job-id>
```

## 📊 Monitoring and Validation

### Pipeline Status

Check pipeline status through:

1. **GitHub Actions**: View workflow runs and artifacts
2. **Databricks**: Monitor job runs and logs
3. **CLI**: Use deployment script status command

### Data Validation Reports

Each validation run generates:

- Null value analysis
- Outlier detection results
- Schema validation status
- Overall data quality score

### Artifacts

GitHub Actions preserve:
- Test results
- Validation reports
- Pipeline execution logs

## 🐳 Local Development

### Using Docker

```bash
# Run tests locally
docker-compose run portfolio-manager

# Test feature engineering specifically
docker-compose run feature-engineering-test

# Deploy to Databricks (requires credentials)
docker-compose --profile deploy run databricks-deploy
```

### Environment Variables

Set these for local development:

```bash
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="your-token"
export CLUSTER_POLICY_ID="your-policy-id"
export ENVIRONMENT="dev"
export TICKERS="AAPL,MSFT"
```

## 🛠️ Troubleshooting

### Common Issues

1. **Job Creation Failed**
   - Check cluster policy ID
   - Verify workspace permissions
   - Validate JSON configuration

2. **Authentication Errors**
   - Verify DATABRICKS_TOKEN is valid
   - Check token permissions
   - Ensure workspace URL is correct

3. **Pipeline Failures**
   - Check Databricks job logs
   - Verify data table existence
   - Review Unity Catalog permissions

### Debug Commands

```bash
# Check Databricks connectivity
databricks auth profiles

# List existing jobs
databricks jobs list

# Get job details
databricks jobs get --job-id <job-id>

# View recent runs
databricks runs list --limit 10
```