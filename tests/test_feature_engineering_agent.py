"""Unit tests for the FeatureEngineeringAgent class."""

import pytest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType

from src.agents.feature_engineering_agent import (FeatureEngineeringAgent,
                                                  FeatureEngineeringError)


@pytest.fixture
def mock_spark_session():
    """Create a mock SparkSession."""
    spark = Mock(spec=SparkSession)
    spark.table = Mock(return_value=Mock(spec=DataFrame))
    spark.catalog.tableExists = Mock(return_value=True)
    return spark


@pytest.fixture
def feature_agent(mock_spark_session):
    """Create a FeatureEngineeringAgent instance with mocked SparkSession."""
    with patch('src.agents.feature_engineering_agent.SparkSession') as mock_session:
        mock_builder = mock_session.builder.appName.return_value.config.return_value
        mock_builder.config.return_value.getOrCreate.return_value = mock_spark_session
        agent = FeatureEngineeringAgent()
        agent.spark = mock_spark_session
        return agent


@pytest.fixture
def sample_market_data():
    """Create sample market data DataFrame structure."""
    return Mock(spec=DataFrame)


class TestFeatureEngineeringAgent:
    """Test suite for FeatureEngineeringAgent class."""

    def test_init(self, feature_agent):
        """Test agent initialization."""
        assert feature_agent.catalog == "main"
        assert feature_agent.schema == "finance"
        assert feature_agent.spark is not None
        assert isinstance(feature_agent.processing_stats, dict)
        assert "processed_tickers" in feature_agent.processing_stats
        assert "failed_tickers" in feature_agent.processing_stats

    def test_define_feature_schema(self):
        """Test feature schema definition."""
        schema = FeatureEngineeringAgent.define_feature_schema()
        assert isinstance(schema, StructType)

        expected_fields = {
            "ticker", "date", "open", "high", "low", "close", "volume",
            "daily_return", "moving_avg_7", "moving_avg_30",
            "volatility_7", "momentum", "feature_timestamp"
        }

        actual_fields = {field.name for field in schema.fields}
        assert actual_fields == expected_fields

    def test_read_raw_data_success(self, feature_agent):
        """Test successful raw data reading."""
        ticker = "AAPL"
        mock_df = Mock(spec=DataFrame)
        mock_df.count.return_value = 100
        feature_agent.spark.table.return_value = mock_df
        feature_agent.spark.catalog.tableExists.return_value = True

        result = feature_agent.read_raw_data(ticker)

        feature_agent.spark.catalog.tableExists.assert_called_with(
            "main.finance.raw_market_AAPL"
        )
        feature_agent.spark.table.assert_called_with("main.finance.raw_market_AAPL")
        assert result == mock_df

    def test_read_raw_data_table_not_exists(self, feature_agent):
        """Test raw data reading when table doesn't exist."""
        ticker = "AAPL"
        feature_agent.spark.catalog.tableExists.return_value = False

        with pytest.raises(FeatureEngineeringError):
            feature_agent.read_raw_data(ticker)

    def test_clean_data(self, feature_agent, sample_market_data):
        """Test data cleaning functionality."""
        mock_df = sample_market_data
        mock_clean_df = Mock(spec=DataFrame)
        mock_clean_df.count.return_value = 90

        # Setup mock chain
        mock_df.count.return_value = 100
        mock_df.dropna.return_value.dropDuplicates.return_value.orderBy.return_value = mock_clean_df

        result = feature_agent.clean_data(mock_df)

        mock_df.dropna.assert_called_once()
        assert result == mock_clean_df

    def test_write_features(self, feature_agent):
        """Test writing features using CREATE OR REPLACE TABLE with spark.sql()."""
        ticker = "AAPL"
        mock_df = Mock(spec=DataFrame)
        mock_typed_df = Mock(spec=DataFrame)

        # Mock the select and type casting chain
        mock_df.select.return_value = mock_typed_df
        mock_typed_df.count.return_value = 100
        mock_typed_df.createOrReplaceTempView = Mock()

        # Mock spark operations
        feature_agent.spark.catalog.dropTempView = Mock()
        feature_agent.spark.sql = Mock()

        # Use patch to mock the entire write_features method logic
        with patch('src.agents.feature_engineering_agent.F') as mock_F:
            # Mock F.col().cast() chain to return mock objects
            mock_col = Mock()
            mock_col.cast.return_value = mock_col
            mock_F.col.return_value = mock_col
            mock_F.lit.return_value = mock_col

            # Call the method
            feature_agent.write_features(mock_df, ticker)

        # Verify DataFrame operations
        mock_df.select.assert_called_once()
        mock_typed_df.createOrReplaceTempView.assert_called_once()

        # Verify spark.sql was called with CREATE OR REPLACE TABLE
        feature_agent.spark.sql.assert_called_once()
        sql_call_args = feature_agent.spark.sql.call_args[0][0]
        assert "CREATE OR REPLACE TABLE" in sql_call_args
        assert f"main.finance.features_{ticker}" in sql_call_args
        assert "AS SELECT * FROM" in sql_call_args
        assert "temp_features_" in sql_call_args

        # Verify cleanup
        feature_agent.spark.catalog.dropTempView.assert_called_once()

        # Verify stats update
        assert feature_agent.processing_stats["total_features_created"] == 100

    def test_validate_success(self, feature_agent):
        """Test successful validation."""
        ticker = "AAPL"
        mock_df = Mock(spec=DataFrame)
        mock_df.columns = [
            "ticker", "date", "open", "high", "low", "close", "volume",
            "daily_return", "moving_avg_7", "moving_avg_30",
            "volatility_7", "momentum", "feature_timestamp"
        ]

        feature_agent.spark.catalog.tableExists.return_value = True
        feature_agent.spark.table.return_value = mock_df

        result = feature_agent.validate(ticker)

        assert result is True

    def test_validate_missing_columns(self, feature_agent):
        """Test validation with missing columns."""
        ticker = "AAPL"
        mock_df = Mock(spec=DataFrame)
        mock_df.columns = ["ticker", "date", "close"]  # Missing several columns

        feature_agent.spark.catalog.tableExists.return_value = True
        feature_agent.spark.table.return_value = mock_df

        result = feature_agent.validate(ticker)

        assert result is False

    def test_validate_table_not_exists(self, feature_agent):
        """Test validation when table doesn't exist."""
        ticker = "AAPL"
        feature_agent.spark.catalog.tableExists.return_value = False

        result = feature_agent.validate(ticker)

        assert result is False

    def test_process_ticker_success(self, feature_agent):
        """Test successful ticker processing."""
        ticker = "AAPL"

        # Mock all the intermediate steps
        mock_raw_df = Mock(spec=DataFrame)
        mock_clean_df = Mock(spec=DataFrame)
        mock_features_df = Mock(spec=DataFrame)

        feature_agent.read_raw_data = Mock(return_value=mock_raw_df)
        feature_agent.clean_data = Mock(return_value=mock_clean_df)
        feature_agent.create_features = Mock(return_value=mock_features_df)
        feature_agent.write_features = Mock()
        feature_agent.validate = Mock(return_value=True)

        result = feature_agent.process_ticker(ticker)

        assert result is True
        feature_agent.read_raw_data.assert_called_once_with(ticker)
        feature_agent.clean_data.assert_called_once_with(mock_raw_df)
        feature_agent.create_features.assert_called_once_with(mock_clean_df)
        feature_agent.write_features.assert_called_once_with(mock_features_df, ticker)
        feature_agent.validate.assert_called_once_with(ticker)

    def test_process_ticker_failure(self, feature_agent):
        """Test ticker processing failure."""
        ticker = "AAPL"
        feature_agent.read_raw_data = Mock(side_effect=Exception("Read error"))

        result = feature_agent.process_ticker(ticker)

        assert result is False
        assert ticker in feature_agent.processing_stats["failed_tickers"]

    def test_process_tickers(self, feature_agent):
        """Test processing multiple tickers."""
        tickers = ["AAPL", "MSFT"]

        def mock_process_ticker(ticker):
            """Mock process_ticker with side effects."""
            if ticker == "AAPL":
                feature_agent.processing_stats["processed_tickers"].append(ticker)
                return True
            else:  # MSFT
                feature_agent.processing_stats["failed_tickers"].append(ticker)
                return False

        # Mock the process_ticker method with side effects
        with patch.object(feature_agent, 'process_ticker', side_effect=mock_process_ticker):
            result = feature_agent.process_tickers(tickers)

        assert len(result["processed_tickers"]) == 1
        assert "AAPL" in result["processed_tickers"]
        assert len(result["failed_tickers"]) == 1
        assert "MSFT" in result["failed_tickers"]
        assert "start_time" in result
        assert "end_time" in result

    def test_error_handling(self, feature_agent):
        """Test error handling for feature engineering."""
        # Mock tableExists to return False to trigger the error
        feature_agent.spark.catalog.tableExists.return_value = False

        with pytest.raises(FeatureEngineeringError):
            feature_agent.read_raw_data("INVALID")


if __name__ == '__main__':
    pytest.main([__file__])
