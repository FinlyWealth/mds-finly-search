import os
import json
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values, execute_batch, Json
from typing import List, Tuple
import sys
from tqdm import tqdm
import re
import glob
from dotenv import load_dotenv
import time
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import config.db
from config.path import METADATA_PATH, EMBEDDINGS_PATH

# Load environment variables
load_dotenv()

# Add checkpoint file path
CHECKPOINT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'insert_checkpoint.pkl')

def get_base_embedding_type(filename: str) -> str:
    """Extract the base embedding type from a filename
    
    Args:
        filename (str): The embedding filename (e.g., 'fusion_embeddings_chunk_0.npz')
        
    Returns:
        str: The base embedding type (e.g., 'fusion')
    """
    # Remove the file extension and chunk information
    base_name = os.path.splitext(filename)[0]
    # Remove '_chunk_X' pattern
    base_name = re.sub(r'_chunk_\d+$', '', base_name)
    # Remove '_embeddings' if present
    base_name = base_name.replace('_embeddings', '')
    return base_name

def get_embedding_paths():
    """Get paths for all embedding files in the embeddings directory
    
    Returns:
        dict: Dictionary mapping embedding type names to their file paths.
        For chunked embeddings, returns the first chunk file found.
    """
    paths = {}
    # Get all .npz files in the embeddings directory
    all_files = glob.glob(os.path.join(EMBEDDINGS_PATH, "*.npz"))
    
    # Group files by their base embedding type
    for file_path in all_files:
        filename = os.path.basename(file_path)
        base_name = get_base_embedding_type(filename)
        
        # If this is a new embedding type or the first chunk we've seen
        if base_name not in paths:
            paths[base_name] = file_path
            
    return paths

def get_enabled_embedding_types():
    """Get list of all embedding types found in the embeddings directory"""
    return list(get_embedding_paths().keys())

def get_chunked_files(embedding_type: str) -> list:
    """Get all chunk files for a given embedding type
    
    Args:
        embedding_type (str): The name of the embedding type
        
    Returns:
        list: List of chunk file paths, sorted by chunk number
    """
    # Get all files that match the embedding type pattern
    pattern = os.path.join(EMBEDDINGS_PATH, f"{embedding_type}_embeddings*.npz")
    chunk_files = glob.glob(pattern)
    
    if not chunk_files:
        return []
        
    # If there's only one file and it doesn't have 'chunk' in the name,
    # it's a non-chunked file
    if len(chunk_files) == 1 and 'chunk' not in chunk_files[0]:
        return chunk_files
        
    # Extract chunk numbers and validate sequence
    chunk_info = []
    for file_path in chunk_files:
        filename = os.path.basename(file_path)
        if 'chunk' in filename:
            try:
                # Extract chunk number using regex
                chunk_num = int(re.search(r'_chunk_(\d+)', filename).group(1))
                chunk_info.append((chunk_num, file_path))
            except (AttributeError, ValueError) as e:
                print(f"Warning: Could not parse chunk number from {filename}: {e}")
                continue
    
    # Sort by chunk number
    chunk_info.sort(key=lambda x: x[0])
    
    # Validate chunk sequence
    expected_chunks = set(range(len(chunk_info)))
    actual_chunks = set(num for num, _ in chunk_info)
    missing_chunks = expected_chunks - actual_chunks
    
    if missing_chunks:
        print(f"Warning: Missing chunks for {embedding_type}: {sorted(missing_chunks)}")
    
    return [path for _, path in chunk_info]

def init_db(embedding_dims: dict, drop: bool = False):
    """Initialize the database and create necessary tables
    
    Args:
        embedding_dims (dict): Dictionary mapping embedding types to their dimensions
        drop (bool): Whether to drop the existing table before creating a new one. Defaults to False.
    """
    conn = psycopg2.connect(**config.db.DB_CONFIG)
    cur = conn.cursor()
    
    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Drop the table if drop is True
    if drop:
        cur.execute(f"DROP TABLE IF EXISTS {config.db.TABLE_NAME};")
    
    # Create embedding columns based on enabled types and their dimensions
    embedding_columns = []
    for embedding_type in get_enabled_embedding_types():
        dim = embedding_dims[embedding_type]
        embedding_columns.append(f"{embedding_type}_embedding vector({dim})")
    
    # Create single products table with dynamic vector dimensions and metadata columns
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {config.db.TABLE_NAME} (
            id SERIAL PRIMARY KEY,
            Pid TEXT UNIQUE,
            {', '.join(embedding_columns)},
            document tsvector,
            Name TEXT,
            Description TEXT,
            Category TEXT,
            Price DECIMAL,
            PriceCurrency TEXT,
            FinalPrice DECIMAL,
            Discount DECIMAL,
            isOnSale BOOLEAN,
            IsInStock BOOLEAN,
            Brand TEXT,
            Color TEXT,
            Gender TEXT,
            Size TEXT,
            Condition TEXT
        );
    """)
    
    # Create GIST index on the document tsvector column
    print("Creating GIST index on document column...")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{config.db.TABLE_NAME}_document ON {config.db.TABLE_NAME} USING GIST (document);")
    
    conn.commit()
    cur.close()
    conn.close()

def validate_numeric(value, field_name):
    """Validate and convert numeric values
    
    Args:
        value: The value to validate
        field_name: Name of the field for error reporting
        
    Returns:
        float or None: Validated numeric value
    """
    if pd.isna(value) or value == '':
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        print(f"Warning: Invalid numeric value for {field_name}: {value}")
        return None

def validate_boolean(value, field_name):
    """Validate and convert boolean values
    
    Args:
        value: The value to validate
        field_name: Name of the field for error reporting
        
    Returns:
        bool or None: Validated boolean value
    """
    if pd.isna(value) or value == '':
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        value = value.lower()
        if value in ('true', 't', 'yes', 'y', '1'):
            return True
        if value in ('false', 'f', 'no', 'n', '0'):
            return False
    print(f"Warning: Invalid boolean value for {field_name}: {value}")
    return None

def validate_text(value, field_name):
    """Validate and convert text values
    
    Args:
        value: The value to validate
        field_name: Name of the field for error reporting
        
    Returns:
        str or None: Validated text value
    """
    if not isinstance(value, (str, int, float, bool)) and not pd.api.types.is_scalar(value):
        print(f"Warning: Invalid text value type for {field_name}: {value}")
        return None

    if pd.isna(value) or value == '':
        return None
    try:
        return str(value).strip()
    except (ValueError, TypeError):
        print(f"Warning: Invalid text value for {field_name}: {value}")
        return None

def validate_and_clean_dataframe(df):
    """Validate and clean the DataFrame, dropping invalid rows
    
    Args:
        df (pd.DataFrame): Input DataFrame to validate and clean
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with valid rows only
    """
    print("\nValidating and cleaning data...")
    original_rows = len(df)
    
    # Create a mask for valid rows
    valid_mask = pd.Series(True, index=df.index)
    
    # Validate Pid (required field)
    valid_mask &= df['Pid'].notna() & (df['Pid'] != '')
    
    # Validate numeric fields
    numeric_fields = ['Price', 'FinalPrice', 'Discount']
    for field in numeric_fields:
        if field in df.columns:
            # Convert to numeric, invalid values become NaN
            df[field] = pd.to_numeric(df[field], errors='coerce')
            # Keep rows where the field is either valid numeric or NaN
            valid_mask &= (df[field].notna() | df[field].isna())
    
    # Validate boolean fields
    boolean_fields = ['isOnSale', 'IsInStock']
    for field in boolean_fields:
        if field in df.columns:
            # Convert to boolean, invalid values become NaN
            df[field] = df[field].map({
                True: True, 'true': True, 't': True, 'yes': True, 'y': True, '1': True, 1: True,
                False: False, 'false': False, 'f': False, 'no': False, 'n': False, '0': False, 0: False
            })
            # Keep rows where the field is either valid boolean or NaN
            valid_mask &= (df[field].notna() | df[field].isna())
    
    # Apply the mask to keep only valid rows
    df_cleaned = df[valid_mask].copy()
    
    # Report statistics
    dropped_rows = original_rows - len(df_cleaned)
    print(f"Original rows: {original_rows}")
    print(f"Valid rows: {len(df_cleaned)}")
    print(f"Dropped rows: {dropped_rows} ({dropped_rows/original_rows*100:.2f}%)")
    
    return df_cleaned

def save_checkpoint(batch_number, total_batches):
    """Save the current batch number to checkpoint file
    
    Args:
        batch_number (int): Current batch number
        total_batches (int): Total number of batches
    """
    checkpoint_data = {
        'batch_number': batch_number,
        'total_batches': total_batches,
        'timestamp': time.time()
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"Checkpoint saved: Batch {batch_number}/{total_batches}")

def load_checkpoint():
    """Load the last checkpoint if it exists
    
    Returns:
        tuple: (batch_number, total_batches) or (0, None) if no checkpoint exists
    """
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                checkpoint_data = pickle.load(f)
            print(f"Loaded checkpoint: Batch {checkpoint_data['batch_number']}/{checkpoint_data['total_batches']}")
            return checkpoint_data['batch_number'], checkpoint_data['total_batches']
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    return 0, None

def insert_data(embeddings_dict, pids, df):
    """Store embeddings and metadata in the database
    
    Args:
        embeddings_dict (dict): Dictionary containing different types of embeddings and their product IDs
        pids (list): List of product IDs that have all required embeddings
        df (pd.DataFrame): DataFrame containing product metadata
    """
    # Validate and clean the DataFrame first
    df = validate_and_clean_dataframe(df)
    
    # Create a set to track seen Pids
    seen_ids = set()
    
    # Create a mapping of product IDs to their indices for each embedding type
    embedding_indices = {}
    for embedding_type in get_enabled_embedding_types():
        embedding_pids = embeddings_dict[f"{embedding_type}_pids"]
        embedding_indices[embedding_type] = {pid: idx for idx, pid in enumerate(embedding_pids)}
    
    # Prepare column names for SQL query
    embedding_columns = [f"{embedding_type}_embedding" for embedding_type in get_enabled_embedding_types()]
    columns = ['Pid'] + embedding_columns + [
        'document', 'Name', 'Description', 'Category',
        'Price', 'PriceCurrency', 'FinalPrice', 'Discount',
        'isOnSale', 'IsInStock', 'Brand',
        'Color', 'Gender', 'Size', 'Condition'
    ]
    
    # Text fields to combine for TF-IDF search
    text_fields = ['Name', 'Category', 'MergedBrand', 
                   'Color', 'Gender', 'Size', 'Condition']
    
    batch_size = 3000
    total_products = len(pids)
    max_retries = 3
    retry_delay = 5  # seconds
    
    # Load checkpoint if exists
    start_batch, checkpoint_total = load_checkpoint()
    if checkpoint_total is not None and checkpoint_total != (total_products + batch_size - 1) // batch_size:
        print("Warning: Checkpoint total batches doesn't match current total. Starting from beginning.")
        start_batch = 0
    
    print("\nInserting data into database...")
    for i in range(start_batch * batch_size, total_products, batch_size):
        batch_pids = pids[i:i + batch_size]
        batch_data = []
        
        # Pre-filter DataFrame for this batch to reduce memory usage
        batch_df = df[df['Pid'].isin(batch_pids)]
        
        for pid in batch_pids:
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            
            # Get product data from DataFrame
            product_data = batch_df[batch_df['Pid'] == pid].iloc[0]
            
            # Get embeddings for this product
            embedding_values = []
            for embedding_type in get_enabled_embedding_types():
                if pid in embedding_indices[embedding_type]:
                    idx = embedding_indices[embedding_type][pid]
                    # Convert NumPy array to list for database insertion
                    embedding_values.append(embeddings_dict[embedding_type][idx].tolist())
                else:
                    embedding_values.append(None)
            
            # Prepare text for ts_vector
            combined_text = ' '.join(str(product_data[field]) for field in text_fields if pd.notna(product_data[field])).lower()
            
            # Prepare metadata values (no need for validation since DataFrame is already cleaned)
            metadata_values = [
                str(pid),
                *embedding_values,  # Unpack embedding values
                combined_text,  # Document text for ts_vector
                str(product_data['Name']) if pd.notna(product_data['Name']) else None,
                str(product_data['Description']) if pd.notna(product_data['Description']) else None,
                str(product_data['Category']) if pd.notna(product_data['Category']) else None,
                float(product_data['Price']) if pd.notna(product_data['Price']) else None,
                str(product_data['PriceCurrency']) if pd.notna(product_data['PriceCurrency']) else None,
                float(product_data['FinalPrice']) if pd.notna(product_data['FinalPrice']) else None,
                float(product_data['Discount']) if pd.notna(product_data['Discount']) else None,
                bool(product_data['isOnSale']) if pd.notna(product_data['isOnSale']) else None,
                bool(product_data['IsInStock']) if pd.notna(product_data['IsInStock']) else None,
                str(product_data['MergedBrand']) if pd.notna(product_data['MergedBrand']) else None,
                str(product_data['Color']) if pd.notna(product_data['Color']) else None,
                str(product_data['Gender']) if pd.notna(product_data['Gender']) else None,
                str(product_data['Size']) if pd.notna(product_data['Size']) else None,
                str(product_data['Condition']) if pd.notna(product_data['Condition']) else None
            ]
            
            batch_data.append(tuple(metadata_values))
        
        if batch_data:
            # Retry logic for database operations
            for retry in range(max_retries):
                try:
                    conn = psycopg2.connect(**config.db.DB_CONFIG)
                    cur = conn.cursor()
                    
                    # Use execute_values for bulk insert with VALUES clause
                    execute_values(
                        cur,
                        f"""
                        INSERT INTO {config.db.TABLE_NAME} (
                            {', '.join(columns)}
                        ) 
                        SELECT 
                            x.Pid,
                            {', '.join(f'x.{col}_embedding' for col in get_enabled_embedding_types())},
                            to_tsvector('english', x.document),
                            x.Name,
                            x.Description,
                            x.Category,
                            x.Price,
                            x.PriceCurrency,
                            x.FinalPrice,
                            x.Discount,
                            x.isOnSale,
                            x.IsInStock,
                            x.Brand,
                            x.Color,
                            x.Gender,
                            x.Size,
                            x.Condition
                        FROM (VALUES %s) AS x (
                            Pid,
                            {', '.join(f'{col}_embedding' for col in get_enabled_embedding_types())},
                            document,
                            Name,
                            Description,
                            Category,
                            Price,
                            PriceCurrency,
                            FinalPrice,
                            Discount,
                            isOnSale,
                            IsInStock,
                            Brand,
                            Color,
                            Gender,
                            Size,
                            Condition
                        )
                        ON CONFLICT (Pid) DO NOTHING
                        """,
                        batch_data
                    )
                    conn.commit()
                    cur.close()
                    conn.close()
                    
                    # Save checkpoint after successful batch
                    current_batch = i // batch_size + 1
                    total_batches = (total_products + batch_size - 1) // batch_size
                    save_checkpoint(current_batch, total_batches)
                    
                    print(f"Processed batch {current_batch}/{total_batches}")
                    break  # Success, exit retry loop
                    
                except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                    print(f"Database error on batch {i//batch_size + 1} (retry {retry + 1}/{max_retries}): {str(e)}")
                    if retry < max_retries - 1:
                        print(f"Waiting {retry_delay} seconds before retrying...")
                        time.sleep(retry_delay)
                    else:
                        print("Max retries reached. Moving to next batch.")
                    # Ensure connection is closed before retrying
                    try:
                        if 'cur' in locals():
                            cur.close()
                        if 'conn' in locals():
                            conn.close()
                    except:
                        pass
                except Exception as e:
                    print(f"Unexpected error on batch {i//batch_size + 1}: {str(e)}")
                    raise  # Re-raise unexpected errors
    
    # Remove checkpoint file after successful completion
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint file removed after successful completion")

# This script is used to load the embeddings and update documents in the database
def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    
    # Print database connection info
    print("\nDatabase Connection Info:")
    print(f"Host: {config.db.DB_CONFIG['host']}")
    print(f"Database: {config.db.DB_CONFIG['dbname']}")
    print(f"User: {config.db.DB_CONFIG['user']}")
    print(f"Port: {config.db.DB_CONFIG['port']}")
    print("\n" + "="*50 + "\n")
    
    print("Loading metadata...")
    df = pd.read_csv(os.path.join(project_root, METADATA_PATH))
    print(f"Number of metadata entries: {len(df)}")
    
    # Load all embedding files
    embedding_files = get_embedding_paths()
    embeddings_dict = {}
    embedding_dims = {}
    
    print("\nLoading embeddings...")
    for name, _ in embedding_files.items():
        print(f"\nLoading {name} embeddings...")
        base_name = get_base_embedding_type(name)
        
        # Get all chunk files for this embedding type
        chunk_files = get_chunked_files(name)
        if not chunk_files:
            print(f"No embedding files found for {name}")
            continue
            
        # Load and concatenate all chunks
        all_embeddings = []
        all_pids = []
        total_chunks = len(chunk_files)
        
        for chunk_idx, chunk_file in enumerate(chunk_files, 1):
            print(f"Loading chunk {chunk_idx}/{total_chunks}: {os.path.basename(chunk_file)}")
            try:
                data = np.load(os.path.join(project_root, chunk_file))
                
                # Validate required keys exist
                if 'embeddings' not in data or 'product_ids' not in data:
                    print(f"Warning: Missing required keys in {chunk_file}")
                    continue
                    
                # Validate shapes
                if len(data['embeddings']) != len(data['product_ids']):
                    print(f"Warning: Mismatched lengths in {chunk_file}")
                    continue
                    
                all_embeddings.append(data['embeddings'])
                all_pids.append(data['product_ids'].astype(str))
                
            except Exception as e:
                print(f"Error loading chunk {chunk_file}: {e}")
                continue
        
        if not all_embeddings:
            print(f"Warning: No valid chunks found for {name}")
            continue
            
        # Concatenate embeddings and product IDs
        try:
            embeddings_dict[base_name] = np.concatenate(all_embeddings, axis=0)
            embeddings_dict[f"{base_name}_pids"] = np.concatenate(all_pids)
            
            # Store dimension from first chunk (should be same for all chunks)
            embedding_dims[base_name] = all_embeddings[0].shape[1]
            print(f"{base_name} embedding dimension: {embedding_dims[base_name]}")
            print(f"Total number of {base_name} embeddings: {len(embeddings_dict[base_name])}")
            
        except Exception as e:
            print(f"Error concatenating chunks for {name}: {e}")
            continue
    
    # Find intersection of product IDs that have all required embeddings
    common_pids = set(df['Pid'].astype(str))
    for name in get_enabled_embedding_types():
        base_name = get_base_embedding_type(name)
        common_pids = common_pids.intersection(set(embeddings_dict[f"{base_name}_pids"]))
    
    common_pids = list(common_pids)
    print(f"\nNumber of products with all required embeddings: {len(common_pids)}")
    
    print("\nInitializing database...")
    init_db(embedding_dims, drop=True)
    
    print("\nInserting embeddings and ts_vector in database...")
    insert_data(embeddings_dict, common_pids, df)
    
    print("\nDone!")

if __name__ == '__main__':
    main()