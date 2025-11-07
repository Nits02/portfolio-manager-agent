"""
Feature Engineering Agent for financial data transformation and feature creation.

This module provides a FeatureEngineeringAgent class that:
1. Reads raw market data from Unity Catalog tables
2. Cleans and preprocesses the data
3. Creates derived financial features (returns, moving averages, volatility, momentum)
4. Writes enriched feature datasets to Unity Catalog for ML workflows
"""

import logging
import sys
from typing import List, Dict, Any
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('feature_engineering.log')
    ]
)
logger = logging.getLogger(__name__)


class FeatureEngineeringError(Exception):
    """Custom exception for feature engineering errors."""
    pass


class FeatureEngineeringAgent:
    """Agent responsible for creating financial features from raw market data."""

    def __init__(self, catalog: str = "main", schema: str = "finance"):
        """
        Initialize the Feature Engineering Agent.

        Args:
            catalog: Unity Catalog name
            schema: Schema/database name within the catalog
        """
        self.catalog = catalog
        self.schema = schema
        self.spark = None
        self.processing_stats: Dict[str, Any] = {
            "processed_tickers": [],
            "failed_tickers": [],
            "total_features_created": 0,
            "start_time": None,
            "end_time": None
        }

        try:
            self.spark = self._initialize_spark()
            logger.info(f"Initialized FeatureEngineeringAgent with "
                        f"catalog={catalog}, schema={schema}")
        except Exception as exc:
            logger.critical(f"Feature engineering workflow failed: {str(exc)}",
                            exc_info=True)
            raise FeatureEngineeringError(f"Agent initialization failed: {str(exc)}")

    def _initialize_spark(self) -> SparkSession:
        """Initialize Spark session with Delta Lake support."""
        try:
            spark = (
                SparkSession.builder
                .appName("FeatureEngineering")
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
            raise FeatureEngineeringError(f"SparkSession initialization failed: "
                                          f"{str(e)}")

    @staticmethod
    def define_feature_schema() -> StructType:
        """
        Define the schema for feature engineered data.

        Returns:
            StructType: Spark schema for feature data
        """
        return StructType([
            StructField("ticker", StringType(), False),
            StructField("date", DateType(), False),
            StructField("open", DoubleType(), True),
            StructField("high", DoubleType(), True),
            StructField("low", DoubleType(), True),
            StructField("close", DoubleType(), True),
            StructField("volume", DoubleType(), True),
            StructField("daily_return", DoubleType(), True),
            StructField("moving_avg_7", DoubleType(), True),
            StructField("moving_avg_30", DoubleType(), True),
            StructField("volatility_7", DoubleType(), True),
            StructField("momentum", DoubleType(), True),
            StructField("feature_timestamp", DateType(), False)
        ])

    def read_raw_data(self, ticker: str) -> DataFrame:
        """
        Read raw market data for a specific ticker from Unity Catalog.

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with raw market data

        Raises:
            FeatureEngineeringError: If table doesn't exist or read fails
        """
        try:
            table_name = f"{self.catalog}.{self.schema}.raw_market_{ticker}"
            logger.info(f"Reading raw data from {table_name}")

            # Check if table exists
            if not self.spark.catalog.tableExists(table_name):
                raise FeatureEngineeringError(f"Table {table_name} does not exist")

            df = self.spark.table(table_name)
            logger.info(f"Successfully read {df.count()} rows from {table_name}")
            return df

        except Exception as e:
            error_msg = f"Failed to read raw data for {ticker}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise FeatureEngineeringError(error_msg)

    def clean_data(self, df: DataFrame) -> DataFrame:
        """
        Clean the raw market data.

        Args:
            df: Raw market data DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning process")

        # Remove rows with null values in critical columns
        critical_columns = ["date", "open", "high", "low", "close", "volume"]
        df_clean = df.dropna(subset=critical_columns)

        # Remove duplicates based on ticker and date
        df_clean = df_clean.dropDuplicates(["ticker", "date"])

        # Sort by ticker and date
        df_clean = df_clean.orderBy("ticker", "date")

        rows_before = df.count()
        rows_after = df_clean.count()
        logger.info(f"Data cleaning completed: {rows_before} -> {rows_after} rows "
                    f"({rows_before - rows_after} rows removed)")

        return df_clean

    def create_features(self, df: DataFrame) -> DataFrame:
        """
        Create derived financial features.

        Args:
            df: Cleaned market data DataFrame

        Returns:
            DataFrame with additional feature columns
        """
        logger.info("Creating derived financial features")

        # Define window specifications
        window_7 = Window.partitionBy("ticker").orderBy("date").rowsBetween(-6, 0)
        window_30 = Window.partitionBy("ticker").orderBy("date").rowsBetween(-29, 0)
        window_lag = Window.partitionBy("ticker").orderBy("date")

        # Create features
        df_features = df.withColumn(
            # Daily return: (close - open) / open
            "daily_return",
            (F.col("close") - F.col("open")) / F.col("open")
        ).withColumn(
            # 7-day moving average of close price
            "moving_avg_7",
            F.avg("close").over(window_7)
        ).withColumn(
            # 30-day moving average of close price
            "moving_avg_30",
            F.avg("close").over(window_30)
        ).withColumn(
            # 7-day volatility (standard deviation of daily returns)
            "volatility_7",
            F.stddev("daily_return").over(window_7)
        ).withColumn(
            # Momentum: current close / close 7 days ago
            "close_lag_7",
            F.lag("close", 7).over(window_lag)
        ).withColumn(
            "momentum",
            F.col("close") / F.col("close_lag_7")
        ).withColumn(
            # Feature creation timestamp
            "feature_timestamp",
            F.lit(datetime.now().date())
        ).drop("close_lag_7")  # Remove intermediate column

        feature_count = len([col for col in df_features.columns
                            if col not in df.columns])
        logger.info(f"Created {feature_count} new features")

        return df_features

    def write_features(self, df: DataFrame, ticker: str) -> None:
        """
        Write feature data to Unity Catalog managed Delta table using CREATE OR REPLACE TABLE.

        Args:
            df: DataFrame with features
            ticker: Stock ticker symbol

        Raises:
            FeatureEngineeringError: If writing fails
        """
        try:
            table_name = f"{self.catalog}.{self.schema}.features_{ticker}"
            temp_view_name = f"temp_features_{ticker}_{int(datetime.now().timestamp())}"

            logger.info(f"Creating managed table {table_name} using spark.sql()")

            # Ensure schema compliance and type casting
            df_typed = df.select([
                F.col("ticker").cast("string"),
                F.col("date").cast("date"),
                F.col("open").cast("double"),
                F.col("high").cast("double"),
                F.col("low").cast("double"),
                F.col("close").cast("double"),
                F.col("volume").cast("double"),
                F.col("daily_return").cast("double"),
                F.col("moving_avg_7").cast("double"),
                F.col("moving_avg_30").cast("double"),
                F.col("volatility_7").cast("double"),
                F.col("momentum").cast("double"),
                F.col("feature_timestamp").cast("date")
            ])

            # Create temporary view for the data
            df_typed.createOrReplaceTempView(temp_view_name)

            # Create or replace managed table using spark.sql()
            create_table_sql = f"""
                CREATE OR REPLACE TABLE {table_name}
                USING DELTA
                TBLPROPERTIES (
                    'delta.autoOptimize.optimizeWrite' = 'true',
                    'delta.autoOptimize.autoCompact' = 'true'
                )
                AS SELECT * FROM {temp_view_name}
            """

            logger.info(f"Executing SQL: {create_table_sql}")
            self.spark.sql(create_table_sql)

            # Clean up temporary view
            self.spark.catalog.dropTempView(temp_view_name)

            row_count = df_typed.count()
            logger.info(f"Successfully created managed table {table_name} with {row_count} rows")
            self.processing_stats["total_features_created"] += row_count

        except Exception as e:
            # Clean up temporary view if it exists
            try:
                self.spark.catalog.dropTempView(temp_view_name)
            except Exception:
                pass  # View might not exist

            error_msg = f"Failed to write features for {ticker}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise FeatureEngineeringError(error_msg)

    def validate(self, ticker: str) -> bool:
        """
        Validate that the feature table has all required columns.

        Args:
            ticker: Stock ticker symbol

        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            table_name = f"{self.catalog}.{self.schema}.features_{ticker}"

            if not self.spark.catalog.tableExists(table_name):
                logger.error(f"Feature table {table_name} does not exist")
                return False

            df = self.spark.table(table_name)
            actual_columns = set(df.columns)

            required_columns = {
                "ticker", "date", "open", "high", "low", "close", "volume",
                "daily_return", "moving_avg_7", "moving_avg_30",
                "volatility_7", "momentum", "feature_timestamp"
            }

            missing_columns = required_columns - actual_columns

            if missing_columns:
                logger.error(f"Missing required columns in {table_name}: "
                             f"{missing_columns}")
                return False

            logger.info(f"Validation passed for {table_name}")
            return True

        except Exception as e:
            logger.error(f"Validation failed for {ticker}: {str(e)}", exc_info=True)
            return False

    def process_ticker(self, ticker: str) -> bool:
        """
        Process a single ticker through the complete feature engineering pipeline.

        Args:
            ticker: Stock ticker symbol

        Returns:
            bool: True if processing succeeded, False otherwise
        """
        try:
            logger.info(f"Starting feature engineering for {ticker}")

            # Read raw data
            raw_df = self.read_raw_data(ticker)

            # Clean data
            clean_df = self.clean_data(raw_df)

            # Create features
            features_df = self.create_features(clean_df)

            # Write features
            self.write_features(features_df, ticker)

            # Validate results
            if self.validate(ticker):
                logger.info(f"Successfully processed {ticker}")
                self.processing_stats["processed_tickers"].append(ticker)
                return True
            else:
                logger.error(f"Validation failed for {ticker}")
                self.processing_stats["failed_tickers"].append(ticker)
                return False

        except Exception as e:
            logger.error(f"Failed to process {ticker}: {str(e)}", exc_info=True)
            self.processing_stats["failed_tickers"].append(ticker)
            return False

    def process_tickers(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Process multiple tickers through feature engineering pipeline.

        Args:
            tickers: List of stock ticker symbols

        Returns:
            Dict with processing statistics
        """
        self.processing_stats["start_time"] = datetime.now()

        logger.info(f"Starting feature engineering for {len(tickers)} tickers: "
                    f"{', '.join(tickers)}")

        for ticker in tickers:
            self.process_ticker(ticker)

        self.processing_stats["end_time"] = datetime.now()

        # Log summary
        duration = (self.processing_stats["end_time"] -
                    self.processing_stats["start_time"])
        logger.info("\n=== Feature Engineering Summary ===")
        logger.info(f"Duration: {duration.total_seconds():.2f} seconds")
        logger.info(f"Total features created: "
                    f"{self.processing_stats['total_features_created']}")
        logger.info(f"Successfully processed: "
                    f"{', '.join(self.processing_stats['processed_tickers'])}")

        if self.processing_stats['failed_tickers']:
            failed = ', '.join(self.processing_stats['failed_tickers'])
            logger.warning(f"Failed tickers: {failed}")

        logger.info("===================================")

        return self.processing_stats


def main():
    """Main function to run feature engineering workflow."""
    try:
        logger.info("Starting feature engineering workflow")

        # Initialize agent
        agent = FeatureEngineeringAgent()

        # Define sample tickers (should match what was ingested)
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

        # Process tickers
        results = agent.process_tickers(tickers)

        if results['failed_tickers']:
            logger.warning("Feature engineering completed with some failures")
            sys.exit(1)
        else:
            logger.info("Feature engineering completed successfully")

    except Exception as exc:
        logger.critical(f"Feature engineering workflow failed: {str(exc)}",
                        exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
