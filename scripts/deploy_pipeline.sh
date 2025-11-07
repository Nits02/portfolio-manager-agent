#!/bin/bash

# Portfolio Manager Pipeline Deployment Script
# This script helps deploy and manage Databricks jobs for the portfolio management pipeline

set -e

# Default values
ENVIRONMENT="dev"
TICKERS="AAPL,MSFT"
ACTION="deploy"
DRY_RUN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy and manage Portfolio Manager Pipeline on Databricks

OPTIONS:
    -e, --environment ENV    Environment (dev|staging|prod) [default: dev]
    -t, --tickers TICKERS    Comma-separated list of tickers [default: AAPL,MSFT]
    -a, --action ACTION      Action (deploy|update|delete|run|status) [default: deploy]
    -d, --dry-run           Show what would be done without executing
    -h, --help              Show this help message

ACTIONS:
    deploy    Create or update all pipeline jobs
    update    Update existing jobs with new configurations
    delete    Remove all pipeline jobs
    run       Trigger a complete pipeline run
    status    Check status of pipeline jobs

EXAMPLES:
    $0 --environment prod --tickers "AAPL,MSFT,GOOGL"
    $0 --action run --environment staging
    $0 --action status --environment prod
    $0 --action delete --environment dev --dry-run

REQUIREMENTS:
    - Databricks CLI installed and configured
    - DATABRICKS_HOST and DATABRICKS_TOKEN environment variables set
    - Or Databricks CLI profile configured

EOF
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if databricks CLI is installed
    if ! command -v databricks &> /dev/null; then
        log_error "Databricks CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if databricks is configured
    if ! databricks auth profiles &> /dev/null; then
        if [[ -z "$DATABRICKS_HOST" || -z "$DATABRICKS_TOKEN" ]]; then
            log_error "Databricks CLI is not configured. Please set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables or configure a profile."
            exit 1
        fi
    fi
    
    # Check if required files exist
    local required_files=(
        "infra/job_ingest.json"
        "infra/job_feature_engineering.json"
        "infra/job_validate_features.json"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    
    log_success "Prerequisites check passed"
}

deploy_job() {
    local job_config="$1"
    local job_name="$2"
    
    log_info "Deploying job: $job_name"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would deploy job configuration: $job_config"
        return 0
    fi
    
    # Create job configuration with environment variables
    envsubst < "$job_config" > "temp_${job_name}.json"
    
    # Check if job already exists
    if databricks jobs list --output json | jq -e ".jobs[] | select(.settings.name == \"$job_name\")" > /dev/null; then
        local job_id=$(databricks jobs list --output json | jq -r ".jobs[] | select(.settings.name == \"$job_name\") | .job_id")
        log_info "Updating existing job with ID: $job_id"
        databricks jobs reset --job-id "$job_id" --json-file "temp_${job_name}.json"
        log_success "Updated job: $job_name (ID: $job_id)"
    else
        local job_id=$(databricks jobs create --json-file "temp_${job_name}.json" | jq -r '.job_id')
        log_success "Created new job: $job_name (ID: $job_id)"
    fi
    
    # Clean up temporary file
    rm -f "temp_${job_name}.json"
}

deploy_all_jobs() {
    log_info "Deploying all pipeline jobs for environment: $ENVIRONMENT"
    
    # Export environment variables for envsubst
    export TICKERS="$TICKERS"
    export ENVIRONMENT="$ENVIRONMENT"
    export CLUSTER_POLICY_ID="${CLUSTER_POLICY_ID:-your-cluster-policy-id}"
    
    # Deploy jobs
    deploy_job "infra/job_ingest.json" "Data Ingestion Agent"
    deploy_job "infra/job_feature_engineering.json" "Feature Engineering Pipeline"
    
    log_success "All jobs deployed successfully"
}

delete_job() {
    local job_name="$1"
    
    log_info "Deleting job: $job_name"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would delete job: $job_name"
        return 0
    fi
    
    if databricks jobs list --output json | jq -e ".jobs[] | select(.settings.name == \"$job_name\")" > /dev/null; then
        local job_id=$(databricks jobs list --output json | jq -r ".jobs[] | select(.settings.name == \"$job_name\") | .job_id")
        databricks jobs delete --job-id "$job_id"
        log_success "Deleted job: $job_name (ID: $job_id)"
    else
        log_warning "Job not found: $job_name"
    fi
}

delete_all_jobs() {
    log_info "Deleting all pipeline jobs"
    
    delete_job "Data Ingestion Agent"
    delete_job "Feature Engineering Pipeline"
    
    log_success "All jobs deleted"
}

run_pipeline() {
    log_info "Running complete pipeline for environment: $ENVIRONMENT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run complete pipeline"
        return 0
    fi
    
    # Run data ingestion
    local ingest_job_id=$(databricks jobs list --output json | jq -r '.jobs[] | select(.settings.name == "Data Ingestion Agent") | .job_id')
    if [[ "$ingest_job_id" != "null" && -n "$ingest_job_id" ]]; then
        log_info "Starting data ingestion..."
        local ingest_run_id=$(databricks jobs run-now --job-id "$ingest_job_id" | jq -r '.run_id')
        log_info "Data ingestion started with run ID: $ingest_run_id"
    else
        log_error "Data ingestion job not found"
        exit 1
    fi
    
    # Wait for ingestion to complete
    log_info "Waiting for data ingestion to complete..."
    while true; do
        local status=$(databricks runs get --run-id "$ingest_run_id" | jq -r '.state.life_cycle_state')
        if [[ "$status" == "TERMINATED" ]]; then
            local result=$(databricks runs get --run-id "$ingest_run_id" | jq -r '.state.result_state')
            if [[ "$result" == "SUCCESS" ]]; then
                log_success "Data ingestion completed successfully"
                break
            else
                log_error "Data ingestion failed"
                exit 1
            fi
        fi
        sleep 30
    done
    
    # Run feature engineering
    local fe_job_id=$(databricks jobs list --output json | jq -r '.jobs[] | select(.settings.name == "Feature Engineering Pipeline") | .job_id')
    if [[ "$fe_job_id" != "null" && -n "$fe_job_id" ]]; then
        log_info "Starting feature engineering..."
        local fe_run_id=$(databricks jobs run-now --job-id "$fe_job_id" | jq -r '.run_id')
        log_success "Feature engineering started with run ID: $fe_run_id"
    else
        log_error "Feature engineering job not found"
        exit 1
    fi
    
    log_success "Pipeline execution initiated successfully"
}

show_status() {
    log_info "Checking pipeline status for environment: $ENVIRONMENT"
    
    local jobs=("Data Ingestion Agent" "Feature Engineering Pipeline")
    
    for job_name in "${jobs[@]}"; do
        if databricks jobs list --output json | jq -e ".jobs[] | select(.settings.name == \"$job_name\")" > /dev/null; then
            local job_id=$(databricks jobs list --output json | jq -r ".jobs[] | select(.settings.name == \"$job_name\") | .job_id")
            local job_status=$(databricks jobs get --job-id "$job_id" | jq -r '.settings.schedule.pause_status // "UNSCHEDULED"')
            log_success "$job_name (ID: $job_id) - Status: $job_status"
        else
            log_warning "$job_name - Not found"
        fi
    done
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--tickers)
            TICKERS="$2"
            shift 2
            ;;
        -a|--action)
            ACTION="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT. Must be dev, staging, or prod."
    exit 1
fi

# Main execution
log_info "Portfolio Manager Pipeline Deployment"
log_info "Environment: $ENVIRONMENT"
log_info "Tickers: $TICKERS"
log_info "Action: $ACTION"
if [[ "$DRY_RUN" == "true" ]]; then
    log_info "Mode: DRY RUN"
fi

# Check prerequisites
check_prerequisites

# Execute action
case $ACTION in
    deploy)
        deploy_all_jobs
        ;;
    update)
        deploy_all_jobs
        ;;
    delete)
        delete_all_jobs
        ;;
    run)
        run_pipeline
        ;;
    status)
        show_status
        ;;
    *)
        log_error "Invalid action: $ACTION"
        usage
        exit 1
        ;;
esac

log_success "Operation completed successfully!"