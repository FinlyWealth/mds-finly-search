import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database connection parameters
DB_CONFIG = {
    'dbname': os.getenv('PGDATABASE', 'finly'),
    'user': os.getenv('PGUSER', 'postgres'),
    'password': os.getenv('PGPASSWORD', 'postgres'),
    'host': os.getenv('PGHOST', 'localhost'),
    'port': os.getenv('PGPORT', '5432')
}

# Table name configuration
TABLE_NAME = os.getenv('PGTABLE', 'products_100k')


SEARCH_WEIGHTS = {
    "text_only": [0.5, 0, 0.5],
    "image_only": [0, 1, 0],
    "hybrid": [0.5, 0, 0.5],

}