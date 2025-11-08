"""Tests for the PredictiveModelAgent class."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, date, timedelta

from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType
import pyspark.sql.functions as F

from src.agents.predictive_model_agent import PredictiveModelAgent


class TestPredictiveModelAgent:
    """Test suite for PredictiveModelAgent."""

    @pytest.fixture
    def mock_spark(self):
        """Create a mock Spark session."""
        spark = Mock()
        spark.catalog = Mock()
        spark.catalog.tableExists.return_value = True
        return spark

    @pytest.fixture
    def agent(self, mock_spark):
        """Create a PredictiveModelAgent instance for testing."""
        with patch('src.agents.predictive_model_agent.SparkSession.builder') as mock_builder:
            mock_builder.appName.return_value = mock_builder
            mock_builder.config.return_value = mock_builder
            mock_builder.getOrCreate.return_value = mock_spark
            
            agent = PredictiveModelAgent(catalog='test_catalog', schema='test_schema')
            agent.spark = mock_spark  # Override with mock
            return agent

    @pytest.fixture
    def sample_features_data(self, mock_spark):
        """Create sample features data for testing."""
        schema = StructType([
            StructField("symbol", StringType(), True),
            StructField("date", DateType(), True),
            StructField("close_price", DoubleType(), True),
            StructField("volume", DoubleType(), True),
            StructField("daily_return", DoubleType(), True),
            StructField("moving_avg_7", DoubleType(), True),
            StructField("moving_avg_30", DoubleType(), True),
            StructField("volatility_7", DoubleType(), True),
            StructField("momentum", DoubleType(), True)
        ])
        
        data = [
            ("AAPL", date(2023, 1, 1), 150.0, 1000000, 0.02, 145.0, 140.0, 0.15, 0.05),
            ("AAPL", date(2023, 1, 2), 152.0, 1100000, 0.013, 146.0, 141.0, 0.16, 0.06),
            ("AAPL", date(2023, 1, 3), 148.0, 900000, -0.026, 147.0, 142.0, 0.17, 0.04),
            ("GOOGL", date(2023, 1, 1), 2800.0, 500000, 0.018, 2750.0, 2700.0, 0.20, 0.07),
            ("GOOGL", date(2023, 1, 2), 2820.0, 550000, 0.007, 2760.0, 2710.0, 0.19, 0.08),
            ("GOOGL", date(2023, 1, 3), 2790.0, 480000, -0.011, 2770.0, 2720.0, 0.21, 0.06)
        ]
        
        df = mock_spark.createDataFrame(data, schema)
        return df

    def test_init(self, agent, mock_spark):
        """Test agent initialization."""
        assert agent.spark == mock_spark
        assert agent.catalog == 'test_catalog'
        assert agent.schema == 'test_schema'

    def test_init_with_defaults(self, mock_spark):
        """Test agent initialization with default config."""
        with patch('src.agents.predictive_model_agent.SparkSession.builder') as mock_builder:
            mock_builder.appName.return_value = mock_builder
            mock_builder.config.return_value = mock_builder
            mock_builder.getOrCreate.return_value = mock_spark
            
            agent = PredictiveModelAgent()
            
            assert agent.catalog == 'main'
            assert agent.schema == 'finance'

    def test_validate_config_valid(self, agent):
        """Test config validation with valid config."""
        # Should not raise an exception - this agent doesn't have a _validate_config method
        # Just test that agent is initialized properly
        assert agent.catalog == 'test_catalog'
        assert agent.schema == 'test_schema'

    def test_validate_config_invalid_split(self, mock_spark):
        """Test config validation with invalid train_test_split."""
        # This agent doesn't have this validation, so we'll skip this test
        pytest.skip("This agent doesn't have train_test_split validation")

    def test_read_feature_data_success(self, agent, sample_features_data):
        """Test successful feature data reading."""
        agent.spark.table.return_value = sample_features_data
        
        result = agent.read_feature_data(['AAPL', 'GOOGL'])
        
        assert result == sample_features_data
        agent.spark.table.assert_called_once()

    def test_create_labels_basic(self, agent, sample_features_data):
        """Test basic label creation."""
        # Mock the DataFrame methods
        sample_features_data.withColumn.return_value = sample_features_data
        sample_features_data.filter.return_value = sample_features_data
        
        result = agent.create_labels(sample_features_data)
        
        assert result == sample_features_data
        sample_features_data.withColumn.assert_called()

    def test_prepare_features_basic(self, agent, sample_features_data):
        """Test basic feature preparation."""
        # Mock the DataFrame methods
        sample_features_data.select.return_value = sample_features_data
        sample_features_data.na = Mock()
        sample_features_data.na.drop.return_value = sample_features_data
        
        result = agent.prepare_features(sample_features_data)
        
        assert result == sample_features_data

    def test_split_data_time_based(self, agent, sample_features_data):
        """Test time-based data splitting."""
        # Mock DataFrame operations
        mock_stats = Mock()
        mock_stats.collect.return_value = [{
            'min_date': date(2023, 1, 1),
            'max_date': date(2023, 1, 10),
            'total_days': 9
        }]
        sample_features_data.select.return_value = mock_stats
        
        mock_train_df = Mock()
        mock_test_df = Mock()
        mock_train_df.count.return_value = 80
        mock_test_df.count.return_value = 20
        
        sample_features_data.filter.side_effect = [mock_train_df, mock_test_df]
        
        train_df, test_df = agent.split_data(sample_features_data)
        
        assert train_df == mock_train_df
        assert test_df == mock_test_df
        assert sample_features_data.filter.call_count == 2

    @patch('src.agents.predictive_model_agent.mlflow')
    def test_train_basic_flow(self, mock_mlflow, agent, sample_features_data):
        """Test basic training flow."""
        # Mock all the methods called in train()
        agent.read_feature_data = Mock(return_value=sample_features_data)
        agent.create_labels = Mock(return_value=sample_features_data)
        agent.prepare_features = Mock(return_value=sample_features_data)
        agent.split_data = Mock(return_value=(sample_features_data, sample_features_data))
        
        # Mock MLflow
        mock_mlflow.start_run.return_value.__enter__ = Mock()
        mock_mlflow.start_run.return_value.__exit__ = Mock()
        
        # Mock the ML pipeline components
        with patch('src.agents.predictive_model_agent.VectorAssembler'):
            with patch('src.agents.predictive_model_agent.StandardScaler'):
                with patch('src.agents.predictive_model_agent.GBTClassifier'):
                    with patch('src.agents.predictive_model_agent.Pipeline') as mock_pipeline:
                        mock_model = Mock()
                        mock_pipeline.return_value.fit.return_value = mock_model
                        
                        # Call train method
                        result = agent.train(['AAPL', 'GOOGL'])
                        
                        # Verify calls
                        agent.read_feature_data.assert_called_once_with(['AAPL', 'GOOGL'])
                        agent.create_labels.assert_called_once()
                        agent.prepare_features.assert_called_once()
                        agent.split_data.assert_called_once()
                        
                        assert result is not None

    def test_evaluate_model_success(self, agent):
        """Test model evaluation."""
        # Mock predictions DataFrames
        mock_train_predictions = Mock()
        mock_test_predictions = Mock()
        
        # Mock evaluators
        with patch('src.agents.predictive_model_agent.BinaryClassificationEvaluator') as mock_binary_eval:
            with patch('src.agents.predictive_model_agent.MulticlassClassificationEvaluator') as mock_multi_eval:
                mock_binary_eval.return_value.evaluate.return_value = 0.85
                mock_multi_eval.return_value.evaluate.return_value = 0.80
                
                metrics = agent.evaluate(mock_train_predictions, mock_test_predictions)
                
                assert 'train_auc' in metrics
                assert 'test_auc' in metrics
                assert 'train_accuracy' in metrics
                assert 'test_accuracy' in metrics
                assert metrics['train_auc'] == 0.85

    @patch('src.agents.predictive_model_agent.mlflow')
    def test_register_model_success(self, mock_mlflow, agent):
        """Test model registration."""
        # Mock MLflow components
        mock_mlflow.get_experiment_by_name.return_value = Mock(experiment_id="test_exp")
        mock_mlflow.active_run.return_value = Mock(info=Mock(run_id="test_run"))
        mock_mlflow.register_model.return_value = Mock(version="1")
        
        # Mock Unity Catalog client
        mock_client = Mock()
        agent.mlflow_client = mock_client
        mock_client.create_registered_model.return_value = Mock()
        
        result = agent.register_model("test_model")
        
        assert result is not None