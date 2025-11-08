"""Unit tests for the DataIngestionAgent class."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, DateType, DoubleType, LongType

from src.agents.ingest_agent import DataIngestionAgent, DataIngestionError


@pytest.fixture
def sample_price_data():
    """Create a sample price DataFrame that mimics yfinance output."""
    dates = pd.date_range(start='2025-01-01', end='2025-01-05')
    data = {
        'Open': np.random.uniform(100, 200, len(dates)),
        'High': np.random.uniform(100, 200, len(dates)),
        'Low': np.random.uniform(100, 200, len(dates)),
        'Close': np.random.uniform(100, 200, len(dates)),
        'Adj Close': np.random.uniform(100, 200, len(dates)),
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def mock_spark_session():
    """Create a mock SparkSession."""
    spark = Mock(spec=SparkSession)
    spark.createDataFrame = Mock(return_value=Mock())
    spark.sql = Mock()
    return spark


@pytest.fixture
def ingest_agent(mock_spark_session):
    """Create a DataIngestionAgent instance with mocked SparkSession."""
    with patch('src.agents.ingest_agent.SparkSession') as mock_session:
        (mock_session.builder.appName.return_value.config.return_value
         .config.return_value.getOrCreate.return_value) = mock_spark_session
        agent = DataIngestionAgent()
        agent.spark = mock_spark_session
        return agent


class TestDataIngestionAgent:
    """Test suite for DataIngestionAgent class."""

    def test_init(self, ingest_agent):
        """Test agent initialization."""
        assert ingest_agent.catalog == "finance_catalog"
        assert ingest_agent.database == "bronze"
        assert ingest_agent.spark is not None
        assert isinstance(ingest_agent.ingestion_stats, dict)

    def test_define_schema(self, ingest_agent):
        """Test schema definition."""
        schema = ingest_agent.define_schema()
        assert isinstance(schema, StructType)
        expected_fields = {
            "ticker": StringType(),
            "date": DateType(),
            "open": DoubleType(),
            "high": DoubleType(),
            "low": DoubleType(),
            "close": DoubleType(),
            "adj_close": DoubleType(),
            "volume": LongType(),
            "ingestion_timestamp": DateType()
        }

        for field in schema.fields:
            assert field.name in expected_fields
            assert isinstance(field.dataType, type(expected_fields[field.name]))

    @patch('yfinance.Ticker')
    def test_download_price_data(self, mock_yf_ticker, ingest_agent, sample_price_data):
        """Test price data download for a single ticker."""
        # Configure mock
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_price_data
        mock_yf_ticker.return_value = mock_ticker_instance

        # Test data download
        tickers = ['AAPL']
        result = ingest_agent.download_price_data(tickers)

        # Verify calls and results
        mock_yf_ticker.assert_called_once_with('AAPL')
        mock_ticker_instance.history.assert_called_once()

        assert isinstance(result, pd.DataFrame)
        assert 'ticker' in result.columns
        assert all(result['ticker'] == 'AAPL')
        assert len(result) == len(sample_price_data)

    @patch('yfinance.Ticker')
    def test_download_price_data_multiple_tickers(
            self,
            mock_yf_ticker,
            ingest_agent,
            sample_price_data
    ):
        """Test price data download for multiple tickers."""
        # Create separate mock instances for each ticker
        ticker_instances = {}
        tickers = ['AAPL', 'MSFT', 'GOOGL']

        for ticker in tickers:
            mock_instance = Mock()
            # Create a copy of sample data with proper date index reset
            ticker_data = sample_price_data.copy()
            # Don't add ticker column here - download_price_data will add it
            mock_instance.history.return_value = ticker_data
            ticker_instances[ticker] = mock_instance

        # Configure mock_yf_ticker to return different instances for different tickers
        mock_yf_ticker.side_effect = lambda ticker: ticker_instances[ticker]

        result = ingest_agent.download_price_data(tickers)

        assert mock_yf_ticker.call_count == len(tickers)
        assert len(result['ticker'].unique()) == len(tickers)
        assert all(ticker in result['ticker'].unique() for ticker in tickers)

    def test_ingest_to_delta(self, ingest_agent, sample_price_data):
        """Test Delta table ingestion."""
        # Prepare test data with correct column names (matching what ingest_to_delta expects)
        test_df = sample_price_data.reset_index()
        test_df['ticker'] = 'AAPL'
        # Set ingestion_timestamp as datetime (not date) so .dt accessor works
        test_df['ingestion_timestamp'] = pd.to_datetime(datetime.now().date())
        
        # Rename columns to match the expected format (lowercase, matching yfinance processing)
        test_df.rename(columns={
            'index': 'date',
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }, inplace=True)
        
        # Ensure date column is also datetime for .dt accessor
        test_df['date'] = pd.to_datetime(test_df['date'])

        # Create mock for spark DataFrame operations
        mock_spark_df = Mock()
        mock_write_chain = (mock_spark_df.write.format.return_value.mode.return_value.
                            option.return_value)
        mock_write_chain.saveAsTable = Mock()
        ingest_agent.spark.createDataFrame.return_value = mock_spark_df

        # Test the method
        ingest_agent.ingest_to_delta(test_df)

        # Verify correct schema was used
        ingest_agent.spark.createDataFrame.assert_called_once()
        args, kwargs = ingest_agent.spark.createDataFrame.call_args
        assert 'schema' in kwargs
        assert isinstance(kwargs['schema'], StructType)

        # Verify Delta table write
        mock_spark_df.write.format.assert_called_with('delta')
        mock_spark_df.write.format.return_value.mode.assert_called_with('append')

    def test_error_handling(self, ingest_agent):
        """Test error handling for data ingestion."""
        with pytest.raises(DataIngestionError):
            ingest_agent.download_price_data([])

    @patch('yfinance.Ticker')
    def test_failed_download_handling(self, mock_yf_ticker, ingest_agent):
        """Test handling of failed downloads."""
        mock_yf_ticker.return_value.history.side_effect = Exception("API Error")

        with pytest.raises(DataIngestionError):
            ingest_agent.download_price_data(['AAPL'])

        assert len(ingest_agent.ingestion_stats['failed_tickers']) == 1
        assert 'AAPL' in ingest_agent.ingestion_stats['failed_tickers']

    def test_date_range_validation(self, ingest_agent, sample_price_data):
        """Test date range validation in download_price_data."""
        start_date = datetime.now() - timedelta(days=5)
        end_date = datetime.now()

        with patch('yfinance.Ticker') as mock_yf_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = sample_price_data
            mock_yf_ticker.return_value = mock_ticker_instance

            ingest_agent.download_price_data(['AAPL'], start_date=start_date, end_date=end_date)

            # Verify date range was passed correctly
            mock_ticker_instance.history.assert_called_once()
            call_kwargs = mock_ticker_instance.history.call_args[1]
            assert 'start' in call_kwargs
            assert 'end' in call_kwargs
            assert call_kwargs['start'] == start_date
            assert call_kwargs['end'] == end_date


if __name__ == '__main__':
    pytest.main([__file__])
