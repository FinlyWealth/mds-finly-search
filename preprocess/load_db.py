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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.db import DB_CONFIG, TABLE_NAME
from config.path import METADATA_PATH, EMBEDDINGS_PATH

# Load environment variables
load_dotenv()

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
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Drop the table if drop is True
    if drop:
        cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME};")
    
    # Create embedding columns based on enabled types and their dimensions
    embedding_columns = []
    for embedding_type in get_enabled_embedding_types():
        dim = embedding_dims[embedding_type]
        embedding_columns.append(f"{embedding_type}_embedding vector({dim})")
    
    # Create single products table with dynamic vector dimensions and metadata columns
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
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
    
    conn.commit()
    cur.close()
    conn.close()

def store_embeddings(embeddings_dict, pids, df):
    """Store embeddings and metadata in the database
    
    Args:
        embeddings_dict (dict): Dictionary containing different types of embeddings and their product IDs
        pids (list): List of product IDs that have all required embeddings
        df (pd.DataFrame): DataFrame containing product metadata
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
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
    
    # Process in batches
    batch_size = 1000
    total_products = len(pids)
    
    print("\nInserting data into database...")
    for i in range(0, total_products, batch_size):
        batch_pids = pids[i:i + batch_size]
        batch_data = []
        
        for pid in batch_pids:
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            
            # Get product data from DataFrame
            product_data = df[df['Pid'] == pid].iloc[0]
            
            # Get embeddings for this product
            embedding_values = []
            for embedding_type in get_enabled_embedding_types():
                if pid in embedding_indices[embedding_type]:
                    idx = embedding_indices[embedding_type][pid]
                    # Convert NumPy array to list for database insertion
                    embedding_values.append(embeddings_dict[embedding_type][idx].tolist())
                else:
                    embedding_values.append(None)
            
            # Prepare metadata values
            metadata_values = [
                pid,
                *embedding_values,  # Unpack embedding values
                None,  # Document will be updated separately
                product_data['Name'],
                product_data['Description'],
                product_data['Category'],
                float(product_data['Price']) if pd.notna(product_data['Price']) else None,
                product_data['PriceCurrency'],
                float(product_data['FinalPrice']) if pd.notna(product_data['FinalPrice']) else None,
                float(product_data['Discount']) if pd.notna(product_data['Discount']) else None,
                bool(product_data['isOnSale']), 
                bool(product_data['IsInStock']),
                product_data['MergedBrand'],
                product_data['Color'],
                product_data['Gender'],
                product_data['Size'],
                product_data['Condition']
            ]
            
            batch_data.append(tuple(metadata_values))
        
        if batch_data:
            execute_values(
                cur,
                f"""
                INSERT INTO {TABLE_NAME} (
                    {', '.join(columns)}
                ) 
                VALUES %s 
                ON CONFLICT (Pid) DO NOTHING
                """,
                batch_data
            )
            conn.commit()
        
        print(f"Processed batch {i//batch_size + 1}/{(total_products + batch_size - 1)//batch_size}")
    
    cur.close()
    conn.close()

def update_documents(product_texts, overwrite: bool = False):
    """Update the document field with the actual text content for full-text search
    
    Args:
        product_texts (list): List of tuples containing (pid, text) pairs
        overwrite (bool): If True, updates all documents. If False, only updates documents that are None.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Update documents in batches using execute_batch
    batch_size = 1000
    total_batches = (len(product_texts) + batch_size - 1) // batch_size
    
    print("\nUpdating ts_vector in database...")
    for i in tqdm(range(0, len(product_texts), batch_size), total=total_batches, desc="Processing batches"):
        batch = product_texts[i:i + batch_size]
        data = [(text, pid) for pid, text in batch]
        
        if overwrite:
            # Update all documents in the batch
            execute_batch(
                cur,
                f"""
                UPDATE {TABLE_NAME} 
                SET document = to_tsvector('english', %s)
                WHERE Pid = %s
                """,
                data,
                page_size=batch_size
            )
        else:
            # Only update documents that are None
            execute_batch(
                cur,
                f"""
                UPDATE {TABLE_NAME} 
                SET document = to_tsvector('english', %s)
                WHERE Pid = %s AND document IS NULL
                """,
                data,
                page_size=batch_size
            )
        conn.commit()
    
    cur.close()
    conn.close()

# This script is used to load the embeddings and update documents in the database
def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    
    # Print database connection info
    print("\nDatabase Connection Info:")
    print(f"Host: {DB_CONFIG['host']}")
    print(f"Database: {DB_CONFIG['dbname']}")
    print(f"User: {DB_CONFIG['user']}")
    print(f"Port: {DB_CONFIG['port']}")
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
    
    # Text fields to combine for TF-IDF search
    text_fields = ['Name', 'Category', 'MergedBrand', 
                   'Color', 'Gender', 'Size', 'Condition']
    
    print("\nInitializing database...")
    init_db(embedding_dims, drop=False)  # Default to not dropping the table
    
    print("\nStoring embeddings in database...")
    store_embeddings(embeddings_dict, common_pids, df)
    
    # Prepare product texts for ts_vector
    df['combined_text'] = df[text_fields].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
    product_texts = list(zip(df['Pid'].tolist(), df['combined_text'].tolist()))
    
    # Update ts_vector in database
    print("\nUpdating ts_vector in database...")
    update_documents(product_texts, overwrite=False)  # Default to not overwriting existing documents
    
    print("\nDone!")

if __name__ == '__main__':
    main()