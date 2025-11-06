"""
Data Ingestion Agent for downloading and storing financial data.

This module provides a DataIngestionAgent class that:
1. Downloads OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance
2. Transforms the data into a Spark DataFrame
3. Writes the data to a Delta table in the bronze layer
"""

import logging
import sys
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import yfinance as yf
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, LongType

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_ingestion.log')
    ]
)
logger = logging.getLogger(__name__)

class DataIngestionError(Exception):
    """Custom exception for data ingestion errors."""
    pass



class DataIngestionAgent:
    """Agent responsible for ingesting financial data into the data lake."""

    @staticmethod
    def define_schema() -> StructType:
        """
        Define the schema for financial data that matches yfinance output.

        Returns:
            StructType: Spark schema for financial data
        """
        return StructType([
            StructField("ticker", StringType(), False),
            StructField("date", DateType(), False),
            StructField("open", DoubleType(), True),
            StructField("high", DoubleType(), True),
            StructField("low", DoubleType(), True),
            StructField("close", DoubleType(), True),
            StructField("adj_close", DoubleType(), True),
            StructField("volume", LongType(), True),
            StructField("ingestion_timestamp", DateType(), False)
        ])

    def __init__(self, catalog: str = "finance_catalog", database: str = "bronze"):
        """
        Initialize the Data Ingestion Agent.

        Args:
            catalog: Unity Catalog name
            database: Schema/database name within the catalog
        """
        self.catalog = catalog
        self.database = database
        self.spark = None
        self.ingestion_stats: Dict[str, Any] = {
            "successful_tickers": [],
            "failed_tickers": [],
            "total_rows": 0,
            "start_time": None,
            "end_time": None
        }
        try:
            self.spark = self._initialize_spark()
            logger.info(f"Initialized DataIngestionAgent with catalog={catalog}, database={database}")
        except Exception as exc:
            logger.critical(f"Failed to initialize DataIngestionAgent: {str(exc)}", exc_info=True)
            raise DataIngestionError(f"Agent initialization failed: {str(exc)}")
        
    def _initialize_spark(self) -> SparkSession:
        """Initialize Spark session with Delta Lake support."""
        try:
            spark = (
                SparkSession.builder
                .appName("FinanceDataIngestion")
                .config(
                    "spark.sql.extensions",
                    "io.delta.sql.DeltaSparkSessionExtension"
                )
                .config(
                    "spark.sql.catalog.spark_catalog",
                    "org.apache.spark.sql.delta.catalog.DeltaCatalog"
                )
                .getOrCreate()
            )
            
            logger.info("SparkSession initialized successfully")
            return spark
        except Exception as e:
            logger.error("SparkSession initialization failed", exc_info=True)
            raise DataIngestionError(f"SparkSession initialization failed: {str(e)}")

    def _log_ingestion_summary(self) -> None:
        """Log summary of the ingestion process."""
        if self.ingestion_stats["start_time"]:
            duration = self.ingestion_stats["end_time"] - self.ingestion_stats["start_time"]
            logger.info("\n=== Ingestion Summary ===")
            logger.info(f"Duration: {duration.total_seconds():.2f} seconds")
            logger.info(f"Total rows ingested: {self.ingestion_stats['total_rows']}")
            logger.info(f"Successful tickers: {', '.join(self.ingestion_stats['successful_tickers'])}")
            if self.ingestion_stats['failed_tickers']:
                logger.warning(f"Failed tickers: {', '.join(self.ingestion_stats['failed_tickers'])}")
            logger.info("=====================")

    def download_price_data(self, tickers: List[str], start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Download OHLCV data for given tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for data download (default: 1 year ago)
            end_date: End date for data download (default: today)

        Returns:
            DataFrame with price data

        Raises:
            DataIngestionError: If no data could be downloaded for any ticker
        """
        self.ingestion_stats["start_time"] = datetime.now()
        try:
            # Set default dates if not provided
            end_date = end_date or datetime.now()
            start_date = start_date or (end_date - timedelta(days=365))

            logger.info(f"Starting data download for {len(tickers)} tickers")
            logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
            
            # Download data for each ticker
            all_data = []
            for ticker in tickers:
                try:
                    logger.info(f"Downloading data for {ticker}...")
                    yf_ticker = yf.Ticker(ticker)
                    hist = yf_ticker.history(start=start_date, end=end_date)
                    
                    if hist.empty:
                        raise DataIngestionError(f"No data available for {ticker}")
                    
                    # Handle timezone and date conversion properly
                    if hasattr(hist.index, 'tz_localize'):
                        hist.index = hist.index.tz_localize(None)  # Remove timezone if present
                    hist.reset_index(inplace=True)
                    # Ensure date column has the correct type
                    if 'Date' in hist.columns:
                        hist['Date'] = pd.to_datetime(hist['Date']).dt.date
                    hist['ticker'] = ticker
                    all_data.append(hist)
                    
                    logger.info(f"Successfully downloaded {len(hist)} rows for {ticker}")
                    self.ingestion_stats["successful_tickers"].append(ticker)
                    
                except Exception as exc:
                    logger.error(f"Failed to download data for {ticker}: {str(exc)}", exc_info=True)
                    self.ingestion_stats["failed_tickers"].append(ticker)
                    continue

            if not all_data:
                raise DataIngestionError("No data downloaded for any ticker")

            # Combine all data and add ingestion timestamp
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data['ingestion_timestamp'] = datetime.now().date()
            
            # Rename columns to match schema
            combined_data.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            }, inplace=True)

            self.ingestion_stats["total_rows"] = len(combined_data)
            logger.info(f"Successfully combined data: {len(combined_data)} total rows")
            
            return combined_data

        except Exception as exc:
            logger.error("Data download failed", exc_info=True)
            raise DataIngestionError(f"Data download failed: {str(exc)}")
        finally:
            self.ingestion_stats["end_time"] = datetime.now()
            self._log_ingestion_summary()

    def ingest_to_delta(self, data: pd.DataFrame, table_name: str = "prices") -> None:
        """
        Write data to Delta table.

        Args:
            data: DataFrame containing the price data
            table_name: Name of the target Delta table

        Raises:
            DataIngestionError: If writing to Delta table fails
        """
        start_time = datetime.now()
        try:
            logger.info(f"Starting Delta table ingestion for {table_name}")
            
            # Convert pandas DataFrame to Spark DataFrame with defined schema
            logger.debug("Converting to Spark DataFrame with schema validation...")
            spark_df = self.spark.createDataFrame(data, schema=self.define_schema())
            
            # Set up the full table path
            full_table_name = f"{self.catalog}.{self.database}.{table_name}"
            logger.info(f"Writing to table: {full_table_name}")
            
            # Write to Delta table
            (spark_df.write
             .format("delta")
             .mode("append")
             .option("mergeSchema", "true")
             .saveAsTable(full_table_name))
            
            row_count = spark_df.count()
            logger.info(f"Successfully wrote {row_count} rows to {full_table_name}")

        except Exception as e:
            logger.error("Failed to write to Delta table", exc_info=True)
            raise DataIngestionError(f"Delta table write failed: {str(e)}")
        finally:
            duration = datetime.now() - start_time
            logger.info(f"Delta table ingestion completed in {duration.total_seconds():.2f} seconds")

def main():
    """Main function to run the data ingestion workflow."""
    try:
        logger.info("Starting data ingestion workflow")
        
        # Initialize agent
        agent = DataIngestionAgent()

        # Define sample tickers (can be extended or moved to config)
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

        # Download and ingest data
        price_data = agent.download_price_data(tickers)
        agent.ingest_to_delta(price_data)

        logger.info("Data ingestion workflow completed successfully")

    except Exception as exc:
        logger.critical(f"Data ingestion workflow failed: {str(exc)}", exc_info=True)
        sys.exit(1)



if __name__ == "__main__":
    main()
