"""
Predictive Model Agent for financial data modeling and prediction.

This module provides a PredictiveModelAgent class that:
1. Reads features from Unity Catalog tables
2. Combines multiple tickers into training datasets
3. Creates binary classification labels for price direction prediction
4. Trains Gradient Boosted Tree Classifiers using PySpark ML
5. Logs experiments and models using MLflow
6. Registers models in Unity Catalog for production use
"""

import logging
import sys
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta

import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('predictive_model.log')
    ]
)
logger = logging.getLogger(__name__)


class PredictiveModelError(Exception):
    """Custom exception for predictive modeling errors."""
    pass


class PredictiveModelAgent:
    """Agent responsible for training and managing predictive models for financial data."""

    def __init__(self, catalog: str = "main", schema: str = "finance"):
        """
        Initialize the Predictive Model Agent.

        Args:
            catalog: Unity Catalog name
            schema: Schema/database name within the catalog
        """
        self.catalog = catalog
        self.schema = schema
        self.spark = None
        self.mlflow_client = None

        # Model configuration
        self.feature_cols = [
            'daily_return', 'moving_avg_7', 'moving_avg_30',
            'volatility_7', 'momentum'
        ]
        self.label_col = 'next_day_up'
        self.features_col = 'features'
        self.scaled_features_col = 'scaled_features'

        # Training parameters
        self.train_test_split = 0.8
        self.random_seed = 42

        try:
            self.spark = self._initialize_spark()
            self.mlflow_client = self._initialize_mlflow()
            logger.info(
                f"Initialized PredictiveModelAgent with catalog={catalog}, schema={schema}")
        except Exception as exc:
            logger.critical("Failed to initialize PredictiveModelAgent", exc_info=True)
            raise PredictiveModelError(f"Agent initialization failed: {str(exc)}")

    def _initialize_spark(self) -> SparkSession:
        """Initialize or get existing Spark session."""
        try:
            # Try to get existing Spark session (common in Databricks)
            spark = SparkSession.getActiveSession()
            if spark is None:
                spark = (
                    SparkSession.builder
                    .appName("PredictiveModelAgent")
                    .config("spark.sql.extensions",
                             "io.delta.sql.DeltaSparkSessionExtension")
                    .config("spark.sql.catalog.spark_catalog",
                             "org.apache.spark.sql.delta.catalog.DeltaCatalog")
                    .getOrCreate()
                )

            logger.info("SparkSession initialized successfully")
            return spark
        except Exception as e:
            logger.error("SparkSession initialization failed", exc_info=True)
            raise PredictiveModelError(f"SparkSession initialization failed: {str(e)}")

    def _initialize_mlflow(self) -> MlflowClient:
        """Initialize MLflow client for experiment tracking."""
        try:
            # Set MLflow tracking URI (in Databricks, this is automatically configured)
            mlflow.set_tracking_uri("databricks")

            # Create or get experiment
            experiment_name = f"/Shared/portfolio_manager_experiments"
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                    logger.info(f"Created new MLflow experiment: {experiment_name}")
                else:
                    experiment_id = experiment.experiment_id
                    logger.info(f"Using existing MLflow experiment: {experiment_name}")
            except Exception as exp_error:
                logger.warning(f"Could not set experiment: {str(exp_error)}")
                experiment_id = None

            mlflow.set_experiment(experiment_name if experiment_id else None)

            client = MlflowClient()
            logger.info("MLflow client initialized successfully")
            return client
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {str(e)}")
            return None

    def read_feature_data(self, tickers: List[str]) -> DataFrame:
        """
        Read and combine feature data from Unity Catalog tables.

        Args:
            tickers: List of ticker symbols to process

        Returns:
            Combined DataFrame with features from all tickers

        Raises:
            PredictiveModelError: If reading feature data fails
        """
        try:
            logger.info(f"Reading feature data for tickers: {', '.join(tickers)}")

            combined_data = []

            for ticker in tickers:
                table_name = f"{self.catalog}.{self.schema}.features_{ticker}"
                logger.info(f"Reading features from {table_name}")

                # Check if table exists
                if not self.spark.catalog.tableExists(table_name):
                    logger.warning(f"Table {table_name} does not exist, skipping {ticker}")
                    continue

                # Read feature data
                df = self.spark.table(table_name)

                # Validate required columns exist
                missing_cols = [col for col in self.feature_cols if col not in df.columns]
                if missing_cols:
                    logger.warning(f"Missing columns in {table_name}: {missing_cols}")
                    continue

                # Select required columns and add ticker info
                feature_df = df.select(
                    F.col("ticker"),
                    F.col("date"),
                    F.col("close"),
                    *[F.col(col) for col in self.feature_cols]
                ).filter(
                    # Remove rows with null values in critical columns
                    F.col("close").isNotNull() &
                    F.col("daily_return").isNotNull()
                )

                row_count = feature_df.count()
                logger.info(f"Loaded {row_count:,} rows for {ticker}")

                if row_count > 0:
                    combined_data.append(feature_df)

            if not combined_data:
                raise PredictiveModelError("No valid feature data found for any ticker")

            # Combine all ticker data
            combined_df = combined_data[0]
            for df in combined_data[1:]:
                combined_df = combined_df.union(df)

            total_rows = combined_df.count()
            logger.info(f"Combined dataset contains {total_rows:,} total rows")

            return combined_df

        except Exception as e:
            error_msg = f"Failed to read feature data: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise PredictiveModelError(error_msg)

    def create_labels(self, df: DataFrame) -> DataFrame:
        """
        Create binary classification labels for next-day price direction.

        Args:
            df: DataFrame with feature data

        Returns:
            DataFrame with added label column
        """
        try:
            logger.info("Creating binary classification labels")

            # Create window for getting next day's price
            window_spec = Window.partitionBy("ticker").orderBy("date")

            # Add next day's close price and create label
            labeled_df = df.withColumn(
                "next_close",
                F.lead("close", 1).over(window_spec)
            ).withColumn(
                self.label_col,
                F.when(F.col("next_close") > F.col("close"), 1).otherwise(0)
            ).filter(
                # Remove last row for each ticker (no next day data)
                F.col("next_close").isNotNull()
            ).drop("next_close")

            # Show label distribution
            label_dist = labeled_df.groupBy(self.label_col).count().collect()
            for row in label_dist:
                label = "UP" if row[self.label_col] == 1 else "DOWN"
                count = row['count']
                logger.info(f"Label distribution - {label}: {count:,} samples")

            return labeled_df

        except Exception as e:
            error_msg = f"Failed to create labels: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise PredictiveModelError(error_msg)

    def prepare_features(self, df: DataFrame) -> DataFrame:
        """
        Prepare features for ML training by assembling and scaling.

        Args:
            df: DataFrame with feature columns

        Returns:
            DataFrame with assembled and scaled features
        """
        try:
            logger.info("Preparing features for ML training")

            # Assemble features into vector
            assembler = VectorAssembler(
                inputCols=self.feature_cols,
                outputCol=self.features_col,
                handleInvalid="skip"  # Skip rows with invalid values
            )

            # Scale features
            scaler = StandardScaler(
                inputCol=self.features_col,
                outputCol=self.scaled_features_col,
                withStd=True,
                withMean=True
            )

            # Create feature pipeline
            feature_pipeline = Pipeline(stages=[assembler, scaler])

            # Fit and transform
            feature_model = feature_pipeline.fit(df)
            prepared_df = feature_model.transform(df)

            logger.info("Feature preparation completed")
            return prepared_df, feature_model

        except Exception as e:
            error_msg = f"Failed to prepare features: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise PredictiveModelError(error_msg)

    def split_data(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """
        Split data into training and testing sets using time-based split.

        Args:
            df: DataFrame to split

        Returns:
            Tuple of (train_df, test_df)
        """
        try:
            logger.info(f"Splitting data with {self.train_test_split:.0%} train split")

            # Get date range for time-based split
            date_stats = df.select(
                F.min("date").alias("min_date"),
                F.max("date").alias("max_date")
            ).collect()[0]

            min_date = date_stats['min_date']
            max_date = date_stats['max_date']

            # Calculate split date
            total_days = (max_date - min_date).days
            train_days = int(total_days * self.train_test_split)
            split_date = min_date + timedelta(days=train_days)

            # Split data
            train_df = df.filter(F.col("date") <= split_date)
            test_df = df.filter(F.col("date") > split_date)

            train_count = train_df.count()
            test_count = test_df.count()

            logger.info(f"Data split completed:")
            logger.info(f"  Training set: {train_count:,} rows (up to {split_date})")
            logger.info(f"  Test set: {test_count:,} rows (after {split_date})")

            return train_df, test_df

        except Exception as e:
            error_msg = f"Failed to split data: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise PredictiveModelError(error_msg)

    def train(self, tickers: List[str], hyperparameter_tuning: bool = False, model_type: str = "gbt") -> Dict[str, Any]:
        """
        Train a binary classifier for price direction prediction.

        Args:
            tickers: List of ticker symbols to train on
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            model_type: Type of model to train ("gbt" or "rf")

        Returns:
            Dictionary with training results and model info
        """
        try:
            logger.info(f"Starting model training for tickers: {', '.join(tickers)}")

            # Start MLflow run
            model_name = "RF" if model_type == "rf" else "GBT"
            with mlflow.start_run(run_name=f"{model_name}_{'_'.join(tickers)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

                # Log parameters
                mlflow.log_params({
                    "tickers": ",".join(tickers),
                    "feature_columns": ",".join(self.feature_cols),
                    "train_test_split": self.train_test_split,
                    "random_seed": self.random_seed,
                    "hyperparameter_tuning": hyperparameter_tuning,
                    "model_type": model_type
                })

                # Read and prepare data
                raw_df = self.read_feature_data(tickers)
                labeled_df = self.create_labels(raw_df)
                prepared_df, feature_model = self.prepare_features(labeled_df)
                train_df, test_df = self.split_data(prepared_df)

                # Configure classifier based on model type
                if model_type == "rf":
                    # RandomForest - more memory efficient for large datasets
                    classifier = RandomForestClassifier(
                        featuresCol=self.scaled_features_col,
                        labelCol=self.label_col,
                        predictionCol="prediction",
                        probabilityCol="probability",
                        seed=self.random_seed,
                        numTrees=20,  # Limited number of trees for memory efficiency
                        maxDepth=5,   # Reasonable depth for good performance
                        subsamplingRate=0.8,
                        maxMemoryInMB=256
                    )
                    logger.info("Using RandomForest classifier (memory-optimized)")
                else:
                    # GBT Classifier with memory-optimized parameters for Databricks
                    # Note: GBTClassifier doesn't support probabilityCol parameter
                    classifier = GBTClassifier(
                        featuresCol=self.scaled_features_col,
                        labelCol=self.label_col,
                        predictionCol="prediction",
                        seed=self.random_seed,
                        maxIter=10,  # Reduced from 20 to limit model size
                        maxDepth=4,  # Reduced from 5 to limit model size
                        stepSize=0.1,
                        subsamplingRate=0.8,  # Add subsampling to reduce model complexity
                        maxMemoryInMB=256  # Explicit memory limit
                    )
                    logger.info("Using GBT classifier (memory-optimized)")

                if hyperparameter_tuning:
                    logger.info("Performing hyperparameter tuning")

                    # Parameter grid for tuning - optimized for Databricks memory limits
                    if model_type == "rf":
                        param_grid = ParamGridBuilder() \
                            .addGrid(classifier.numTrees, [10, 20, 30]) \
                            .addGrid(classifier.maxDepth, [3, 5, 7]) \
                            .addGrid(classifier.subsamplingRate, [0.7, 0.8, 0.9]) \
                            .build()
                    else:
                        param_grid = ParamGridBuilder() \
                            .addGrid(classifier.maxIter, [5, 10, 15]) \
                            .addGrid(classifier.maxDepth, [2, 3, 4]) \
                            .addGrid(classifier.stepSize, [0.1, 0.15, 0.2]) \
                            .addGrid(classifier.subsamplingRate, [0.7, 0.8, 0.9]) \
                            .build()

                    # Cross-validator
                    evaluator = BinaryClassificationEvaluator(
                        labelCol=self.label_col,
                        metricName="areaUnderROC"
                    )

                    cv = CrossValidator(
                        estimator=classifier,
                        estimatorParamMaps=param_grid,
                        evaluator=evaluator,
                        numFolds=3,
                        seed=self.random_seed
                    )

                    # Train with cross-validation
                    cv_model = cv.fit(train_df)
                    model = cv_model.bestModel

                    # Log best parameters
                    if model_type == "rf":
                        best_params = {
                            "best_numTrees": model.getNumTrees(),
                            "best_maxDepth": model.getMaxDepth(),
                            "best_subsamplingRate": model.getSubsamplingRate()
                        }
                    else:
                        best_params = {
                            "best_maxIter": model.getMaxIter(),
                            "best_maxDepth": model.getMaxDepth(),
                            "best_stepSize": model.getStepSize(),
                            "best_subsamplingRate": model.getSubsamplingRate()
                        }
                    mlflow.log_params(best_params)
                    logger.info(f"Best parameters: {best_params}")

                else:
                    # Train with default parameters
                    model = classifier.fit(train_df)

                # Make predictions
                train_predictions = model.transform(train_df)
                test_predictions = model.transform(test_df)

                # Evaluate model
                metrics = self.evaluate(train_predictions, test_predictions)

                # Log metrics
                mlflow.log_metrics(metrics)

                # Log model
                mlflow.spark.log_model(
                    model,
                    "model",
                    registered_model_name=f"portfolio_gbt_{'_'.join(tickers)}"
                )

                # Log feature model
                mlflow.spark.log_model(feature_model, "feature_pipeline")

                training_results = {
                    "model": model,
                    "feature_model": feature_model,
                    "metrics": metrics,
                    "train_predictions": train_predictions,
                    "test_predictions": test_predictions,
                    "mlflow_run_id": mlflow.active_run().info.run_id
                }

                logger.info("Model training completed successfully")
                return training_results

        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise PredictiveModelError(error_msg)

    def evaluate(self, train_predictions: DataFrame, test_predictions: DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on training and test sets.

        Args:
            train_predictions: Training predictions DataFrame
            test_predictions: Test predictions DataFrame

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            logger.info("Evaluating model performance")

            # Binary classification evaluator (ROC-AUC)
            # For GBT models, this will use rawPredictionCol automatically
            # For RandomForest models, this will use probabilityCol automatically
            binary_evaluator = BinaryClassificationEvaluator(
                labelCol=self.label_col,
                metricName="areaUnderROC"
            )

            # Multiclass evaluator (Accuracy, F1)
            multiclass_evaluator = MulticlassClassificationEvaluator(
                labelCol=self.label_col,
                predictionCol="prediction"
            )

            # Calculate metrics
            metrics = {}

            # Training metrics
            metrics["train_auc"] = binary_evaluator.evaluate(train_predictions)

            multiclass_evaluator.setMetricName("accuracy")
            metrics["train_accuracy"] = multiclass_evaluator.evaluate(train_predictions)

            multiclass_evaluator.setMetricName("f1")
            metrics["train_f1"] = multiclass_evaluator.evaluate(train_predictions)

            # Test metrics
            metrics["test_auc"] = binary_evaluator.evaluate(test_predictions)

            multiclass_evaluator.setMetricName("accuracy")
            metrics["test_accuracy"] = multiclass_evaluator.evaluate(test_predictions)

            multiclass_evaluator.setMetricName("f1")
            metrics["test_f1"] = multiclass_evaluator.evaluate(test_predictions)

            # Log metrics
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")

            return metrics

        except Exception as e:
            error_msg = f"Model evaluation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise PredictiveModelError(error_msg)

    def register_model(self, model_name: str, model_version: str = None,
                       description: str = None) -> Dict[str, Any]:
        """
        Register trained model in Unity Catalog.

        Args:
            model_name: Name for the registered model
            model_version: Specific version to register (optional)
            description: Model description

        Returns:
            Model registration information
        """
        try:
            if not self.mlflow_client:
                raise PredictiveModelError("MLflow client not initialized")

            logger.info(f"Registering model: {model_name}")

            # Create registered model if it doesn't exist
            try:
                registered_model = self.mlflow_client.create_registered_model(
                    name=model_name,
                    description=description or f"GBT Classifier for financial prediction - {datetime.now()}"
                )
                logger.info(f"Created new registered model: {model_name}")
            except Exception:
                # Model already exists
                registered_model = self.mlflow_client.get_registered_model(model_name)
                logger.info(f"Using existing registered model: {model_name}")

            # Get latest model version if not specified
            if model_version is None:
                latest_versions = self.mlflow_client.get_latest_versions(
                    model_name,
                    stages=["None", "Staging", "Production"]
                )
                if latest_versions:
                    model_version = max([int(v.version) for v in latest_versions]) + 1
                else:
                    model_version = 1

            registration_info = {
                "model_name": model_name,
                "model_version": str(model_version),
                "registration_time": datetime.now(),
                "model_uri": f"models:/{model_name}/{model_version}"
            }

            logger.info(f"Model registered successfully: {registration_info}")
            return registration_info

        except Exception as e:
            error_msg = f"Model registration failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise PredictiveModelError(error_msg)


def main():
    """Main function for testing the PredictiveModelAgent."""
    try:
        logger.info("Starting PredictiveModelAgent test")

        # Initialize agent
        agent = PredictiveModelAgent()

        # Test with sample tickers
        tickers = ['AAPL', 'MSFT']

        # Train model
        results = agent.train(tickers, hyperparameter_tuning=False)

        # Register model
        model_info = agent.register_model(
            model_name=f"portfolio_gbt_{'_'.join(tickers)}",
            description=f"GBT Classifier for {', '.join(tickers)} price direction prediction"
        )

        logger.info("PredictiveModelAgent test completed successfully")

    except Exception as exc:
        logger.critical(f"PredictiveModelAgent test failed: {str(exc)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
