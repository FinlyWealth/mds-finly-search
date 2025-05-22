import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# File paths configuration
EMBEDDINGS_PATH = os.getenv('EMBEDDINGS_PATH', 'data/embeddings')
METADATA_PATH = os.getenv('METADATA_PATH', 'data/sample.csv')
BENCHMARK_QUERY_CSV = os.getenv('BENCHMARK_QUERY_CSV', 'data/csv/benchmark_query.csv')

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')