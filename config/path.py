import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# File paths configuration
EMBEDDINGS_PATH = os.getenv('EMBEDDINGS_PATH', 'data/embeddings')
CLEAN_CSV_PATH = os.getenv('CLEAN_CSV_PATH', 'data/clean/sample.csv')
RAW_CSV_PATH = os.getenv('CLEAN_CSV_PATH', 'data/raw/sample.csv')
BENCHMARK_QUERY_CSV = os.getenv('BENCHMARK_QUERY_CSV', 'data/csv/benchmark/benchmark_query_v2.csv')

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')