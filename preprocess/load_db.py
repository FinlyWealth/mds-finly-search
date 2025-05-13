import os
import json
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values, Json
from typing import List, Tuple
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.db import DB_CONFIG
from config.path import METADATA_PATH
from config.embeddings import get_embedding_paths, get_enabled_embedding_types


def init_db(embedding_dim: int):
    """Initialize the database and create necessary tables"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Enable pgvector extension
    # cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Drop the table if it exists
    cur.execute("DROP TABLE IF EXISTS products;")
    
    # Create embedding columns based on enabled types
    embedding_columns = []
    for embedding_type in get_enabled_embedding_types():
        embedding_columns.append(f"{embedding_type}_embedding vector({embedding_dim})")
    
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
        embeddings_dict (dict): Dictionary containing different types of embeddings
        pids (list): List of product IDs
        df (pd.DataFrame): DataFrame containing product metadata
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Create a set to track seen Pids
    seen_ids = set()
    data = []
    
    # Only add the first occurrence of each Pid
    for i, pid in enumerate(pids):
        if pid not in seen_ids:
            seen_ids.add(pid)
            # Get metadata for this product
            product_data = df[df['Pid'] == pid].iloc[0]
            
            # Prepare embedding values
            embedding_values = []
            for embedding_type in get_enabled_embedding_types():
                embedding_values.append([float(x) for x in embeddings_dict[embedding_type][i]])
            
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
            
            data.append(tuple(metadata_values))
    
    # Prepare column names for SQL query
    embedding_columns = [f"{embedding_type}_embedding" for embedding_type in get_enabled_embedding_types()]
    columns = ['Pid'] + embedding_columns + [
        'document', 'Name', 'Description', 'Category',
        'Price', 'PriceCurrency', 'FinalPrice', 'Discount',
        'isOnSale', 'IsInStock', 'Brand',
        'Color', 'Gender', 'Size', 'Condition'
    ]
    
    # Store all data in a single table
    execute_values(
        cur,
        f"""
        INSERT INTO products (
            {', '.join(columns)}
        ) 
        VALUES %s 
        ON CONFLICT (Pid) DO NOTHING
        """,
        data
    )
    
    conn.commit()
    cur.close()
    conn.close()

def update_documents(product_texts):
    """Update the document field with the actual text content for full-text search"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Update documents in batches to avoid memory issues
    batch_size = 1000
    for i in range(0, len(product_texts), batch_size):
        batch = product_texts[i:i + batch_size]
        for pid, text in batch:
            cur.execute("""
                UPDATE products 
                SET document = to_tsvector('english', %s)
                WHERE Pid = %s
            """, (text, pid))
        conn.commit()
    
    cur.close()
    conn.close()

# This script is used to load the embeddings and update documents in the database
def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    
    # Load all embedding files
    embedding_files = get_embedding_paths()
    embeddings_dict = {}
    embedding_dim = None
    
    print("Loading embeddings...")
    for name, path in embedding_files.items():
        print(f"Loading {name} embeddings from {path}...")
        data = np.load(os.path.join(project_root, path))
        embeddings_dict[name] = data['embeddings']
        
        # Verify embedding dimensions are consistent
        if embedding_dim is None:
            embedding_dim = data['embeddings'].shape[1]
        elif data['embeddings'].shape[1] != embedding_dim:
            raise ValueError(f"Inconsistent embedding dimensions: {name} has {data['embeddings'].shape[1]} dimensions")
    
    print(f"Embedding dimension: {embedding_dim}")
    
    print("Loading metadata...")
    df = pd.read_csv(os.path.join(project_root, METADATA_PATH))
    
    # Text fields to combine for TF-IDF search
    text_fields = ['Name', 'Description', 'Category', 'MergedBrand', 
                   'Color', 'Gender', 'Size', 'Condition']
    
    # Create combined text
    print("Creating combined text...")
    df['combined_text'] = df[text_fields].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
    
    print("Initializing database...")
    init_db(embedding_dim)
    
    print("Storing embeddings in database...")
    store_embeddings(embeddings_dict, data['product_ids'].astype(str), df)
    
    # Prepare product texts for update
    print("Preparing product texts...")
    product_texts = list(zip(df['Pid'].tolist(), df['combined_text'].tolist()))
    
    # Update documents in database
    print("Updating documents in database...")
    update_documents(product_texts)
    
    print("Done!")

if __name__ == '__main__':
    main()