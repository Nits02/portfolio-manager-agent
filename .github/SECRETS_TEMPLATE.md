# GitHub Repository Secrets Configuration
# Add these secrets to your GitHub repository for the workflows to work

## Required Secrets for All Environments

### Databricks Configuration
# DATABRICKS_HOST: Your Databricks workspace URL (e.g., https://your-workspace.cloud.databricks.com)
# DATABRICKS_TOKEN: Personal access token or service principal token for Databricks

### Databricks Cluster Configuration  
# CLUSTER_POLICY_ID: ID of the cluster policy to use for jobs

## Environment-Specific Configuration

### Development Environment
# Set these in GitHub repository settings > Environments > dev
DATABRICKS_HOST_DEV="https://your-dev-workspace.cloud.databricks.com"
CLUSTER_POLICY_ID_DEV="your-dev-cluster-policy-id"

### Staging Environment  
# Set these in GitHub repository settings > Environments > staging
DATABRICKS_HOST_STAGING="https://your-staging-workspace.cloud.databricks.com"
CLUSTER_POLICY_ID_STAGING="your-staging-cluster-policy-id"

### Production Environment
# Set these in GitHub repository settings > Environments > prod
DATABRICKS_HOST_PROD="https://your-prod-workspace.cloud.databricks.com"
CLUSTER_POLICY_ID_PROD="your-prod-cluster-policy-id"

## Optional Secrets

### Notification Configuration (Optional)
# SLACK_WEBHOOK_URL: Slack webhook for notifications
# TEAMS_WEBHOOK_URL: Microsoft Teams webhook for notifications
# EMAIL_NOTIFICATION: Email address for failure notifications

### Advanced Configuration (Optional)
# DATABRICKS_WORKSPACE_ID: Workspace ID for advanced configurations
# DATABRICKS_ORG_ID: Organization ID for Unity Catalog operations

## Environment Variables in Workflows

The workflows use the following environment-specific variables:
- TICKERS: Comma-separated list of stock symbols (default: AAPL,MSFT)
- CATALOG: Unity Catalog name (default: main)
- SCHEMA: Schema name (default: finance)

## Setup Instructions

1. Go to your GitHub repository
2. Navigate to Settings > Secrets and variables > Actions
3. Add the required repository secrets listed above
4. Create environments (dev, staging, prod) in Settings > Environments
5. Add environment-specific secrets to each environment
6. Configure environment protection rules as needed

## Databricks Token Setup

To create a Databricks personal access token:
1. Log in to your Databricks workspace
2. Go to User Settings > Developer > Access tokens
3. Click "Generate new token"
4. Set an appropriate lifetime and description
5. Copy the token and add it as DATABRICKS_TOKEN secret

## Cluster Policy Setup

To get your cluster policy ID:
1. Go to your Databricks workspace
2. Navigate to Compute > Policies
3. Find your desired policy and copy the ID
4. Add it as CLUSTER_POLICY_ID secret