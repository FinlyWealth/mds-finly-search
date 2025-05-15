import os
import json
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values, Json
from typing import List, Tuple
import sys
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.db import DB_CONFIG
from config.path import METADATA_PATH
from config.embeddings import get_embedding_paths, get_enabled_embedding_types


def init_db(embedding_dims: dict):
    """Initialize the database and create necessary tables
    
    Args:
        embedding_dims (dict): Dictionary mapping embedding types to their dimensions
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Drop the table if it exists
    cur.execute("DROP TABLE IF EXISTS products;")
    
    # Create embedding columns based on enabled types and their dimensions
    embedding_columns = []
    for embedding_type in get_enabled_embedding_types():
        dim = embedding_dims[embedding_type]
        embedding_columns.append(f"{embedding_type}_embedding vector({dim})")
    
    # Create single products table with dynamic vector dimensions and metadata columns
    cur.execute(f"""
        CREATE TABLE products (
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
            if pid not in seen_ids:
                seen_ids.add(pid)
                # Get metadata for this product
                product_data = df[df['Pid'] == pid].iloc[0]
                
                # Prepare embedding values
                embedding_values = []
                for embedding_type in get_enabled_embedding_types():
                    try:
                        # Get the index for this product ID in the current embedding type
                        idx = embedding_indices[embedding_type].get(pid)
                        if idx is not None:
                            embedding_values.append([float(x) for x in embeddings_dict[embedding_type][idx]])
                        else:
                            # If no embedding exists for this product, use zeros
                            dim = embeddings_dict[embedding_type].shape[1]
                            embedding_values.append([0.0] * dim)
                    except IndexError as e:
                        raise IndexError(f"Failed to access embedding at index {idx} for type {embedding_type}. "
                                       f"Embedding length: {len(embeddings_dict[embedding_type])}, "
                                       f"PIDs length: {len(pids)}") from e
                
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
                INSERT INTO products (
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

def update_documents(product_texts):
    """Update the document field with the actual text content for full-text search"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Update documents in batches using execute_values
    batch_size = 1000
    total_batches = (len(product_texts) + batch_size - 1) // batch_size
    
    print("\nUpdating ts_vector in database...")
    for i in tqdm(range(0, len(product_texts), batch_size), total=total_batches, desc="Processing batches"):
        batch = product_texts[i:i + batch_size]
        data = [(text, pid) for pid, text in batch]
        
        execute_values(
            cur,
            """
            UPDATE products 
            SET document = to_tsvector('english', %s)
            WHERE Pid = %s
            """,
            data
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
    for name, path in embedding_files.items():
        print(f"\nLoading {name} embeddings from {path}...")
        data = np.load(os.path.join(project_root, path))
        embeddings_dict[name] = data['embeddings']
        embedding_dims[name] = data['embeddings'].shape[1]
        print(f"{name} embedding dimension: {embedding_dims[name]}")
        
        # Create mapping of product IDs to indices for this embedding type
        embedding_pids = data['product_ids'].astype(str)
        embeddings_dict[f"{name}_pids"] = embedding_pids
        print(f"Number of {name} embeddings: {len(embedding_pids)}")
    
    # Find intersection of product IDs that have all required embeddings
    common_pids = set(df['Pid'].astype(str))
    for name in get_enabled_embedding_types():
        common_pids = common_pids.intersection(set(embeddings_dict[f"{name}_pids"]))
    
    common_pids = list(common_pids)
    print(f"\nNumber of products with all required embeddings: {len(common_pids)}")
    
    # Text fields to combine for TF-IDF search
    text_fields = ['Name', 'Description', 'Category', 'MergedBrand', 
                   'Color', 'Gender', 'Size', 'Condition']
    
    print("\nInitializing database...")
    init_db(embedding_dims)
    
    print("\nStoring embeddings in database...")
    store_embeddings(embeddings_dict, common_pids, df)
    
    # Prepare product texts for ts_vector
    df['combined_text'] = df[text_fields].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
    product_texts = list(zip(df['Pid'].tolist(), df['combined_text'].tolist()))
    
    # Update ts_vector in database
    print("\nUpdating ts_vector in database...")
    update_documents(product_texts)
    
    print("\nDone!")

if __name__ == '__main__':
    main()