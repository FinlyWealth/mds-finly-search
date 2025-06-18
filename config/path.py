import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# File paths configuration
CLEAN_CSV_PATH = os.getenv('CLEAN_CSV_PATH', 'data/csv/clean/data.csv')
RAW_CSV_PATH = os.getenv('RAW_CSV_PATH', 'data/csv/raw/data.csv')
BENCHMARK_QUERY_CSV = os.getenv('BENCHMARK_QUERY_CSV', 'data/csv/benchmark/benchmark_query_v2.csv')
EMBEDDINGS_PATH = os.getenv('EMBEDDINGS_PATH', 'data/embeddings')

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')