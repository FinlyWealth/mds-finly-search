import os
import json
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values, Json
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database configuration
DB_CONFIG = {
    'dbname': os.getenv('PGDATABASE', 'finly'),
    'user': os.getenv('PGUSER', 'postgres'),
    'password': os.getenv('PGPASSWORD', 'postgres'),
    'host': os.getenv('PGHOST', 'localhost'),
    'port': os.getenv('PGPORT', '5432')
}

def drop_and_recreate_table(embedding_dim: int):
    """Drop and recreate the products table"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Drop the table if it exists
    cur.execute("DROP TABLE IF EXISTS products;")
    
    # Create the table with dynamic vector dimensions and metadata columns
    cur.execute(f"""
        CREATE TABLE products (
            id SERIAL PRIMARY KEY,
            Pid TEXT UNIQUE,
            text_embedding vector({embedding_dim}),
            image_embedding vector({embedding_dim}),
            document tsvector,
            Name TEXT,
            ShortDescription TEXT,
            Description TEXT,
            CategoryId TEXT,
            Category TEXT,
            Price DECIMAL,
            PriceCurrency TEXT,
            SalePrice DECIMAL,
            FinalPrice DECIMAL,
            Discount DECIMAL,
            isOnSale BOOLEAN,
            IsInStock BOOLEAN,
            Brand TEXT,
            Manufacturer TEXT,
            MPN TEXT,
            UPCorEAN TEXT,
            SKU TEXT,
            Color TEXT,
            Gender TEXT,
            Size TEXT,
            Condition TEXT
        );
    """)
    
    conn.commit()
    cur.close()
    conn.close()

def init_db(embedding_dim: int):
    """Initialize the database and create necessary tables"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Create single products table with dynamic vector dimensions and metadata columns
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS products (
            id SERIAL PRIMARY KEY,
            Pid TEXT UNIQUE,
            text_embedding vector({embedding_dim}),
            image_embedding vector({embedding_dim}),
            document tsvector,
            Name TEXT,
            ShortDescription TEXT,
            Description TEXT,
            CategoryId TEXT,
            Category TEXT,
            Price DECIMAL,
            PriceCurrency TEXT,
            SalePrice DECIMAL,
            FinalPrice DECIMAL,
            Discount DECIMAL,
            isOnSale BOOLEAN,
            IsInStock BOOLEAN,
            Brand TEXT,
            Manufacturer TEXT,
            MPN TEXT,
            UPCorEAN TEXT,
            SKU TEXT,
            Color TEXT,
            Gender TEXT,
            Size TEXT,
            Condition TEXT
        );
    """)
    
    conn.commit()
    cur.close()
    conn.close()

def store_embeddings(text_embeddings, image_embeddings, pids, df):
    """Store embeddings and metadata in the database"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Create a set to track seen Pids
    seen_ids = set()
    data = []
    
    # Only add the first occurrence of each Pid
    for pid, text_emb, img_emb in zip(pids, text_embeddings, image_embeddings):
        if pid not in seen_ids:
            seen_ids.add(pid)
            # Get metadata for this product
            product_data = df[df['Pid'] == pid].iloc[0]
            data.append((
                pid,
                [float(x) for x in text_emb],
                [float(x) for x in img_emb],
                None,  # Document will be updated separately
                product_data['Name'],
                product_data['ShortDescription'],
                product_data['Description'],
                product_data['CategoryId'],
                product_data['Category'],
                float(product_data['Price']) if pd.notna(product_data['Price']) else None,
                product_data['PriceCurrency'],
                float(product_data['SalePrice']) if pd.notna(product_data['SalePrice']) else None,
                float(product_data['FinalPrice']) if pd.notna(product_data['FinalPrice']) else None,
                float(product_data['Discount']) if pd.notna(product_data['Discount']) else None,
                bool(product_data['isOnSale']), 
                bool(product_data['IsInStock']),
                product_data['Brand'],
                product_data['Manufacturer'],
                product_data['MPN'],
                product_data['UPCorEAN'],
                product_data['SKU'],
                product_data['Color'],
                product_data['Gender'],
                product_data['Size'],
                product_data['Condition']
            ))
    
    # Store all data in a single table
    execute_values(
        cur,
        """
        INSERT INTO products (
            Pid, text_embedding, image_embedding, document,
            Name, ShortDescription, Description, CategoryId, Category,
            Price, PriceCurrency, SalePrice, FinalPrice, Discount,
            isOnSale, IsInStock, Brand, Manufacturer, MPN, UPCorEAN,
            SKU, Color, Gender, Size, Condition
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
    
    print(f"Loading embeddings from {data_dir}...")
    load_path = os.path.join(data_dir, 'embeddings.npz')
    
    data = np.load(load_path)
    
    # Get embedding dimensions from the file
    embedding_dim = data['text_embeddings'].shape[1]
    print(f"Embedding dimension: {embedding_dim}")
    
    print("Loading metadata...")
    df = pd.read_parquet(os.path.join(data_dir, 'merged_output_sample_100k.parquet'))
    
    # Text fields to combine for TF-IDF search
    text_fields = ['Name', 'ShortDescription', 'Category', 'Brand', 
                   'Manufacturer', 'Color', 'Gender', 'Size', 'Condition']
    
    # Create combined text
    print("Creating combined text...")
    df['combined_text'] = df[text_fields].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
    
    print("Initializing database...")
    init_db(embedding_dim)
    
    print("Dropping and recreating table...")
    drop_and_recreate_table(embedding_dim)
    
    print("Storing embeddings in database...")
    store_embeddings(data['text_embeddings'], data['image_embeddings'], data['product_ids'].astype(str), df)
    
    # Prepare product texts for update
    print("Preparing product texts...")
    product_texts = list(zip(df['Pid'].tolist(), df['combined_text'].tolist()))
    
    # Update documents in database
    print("Updating documents in database...")
    update_documents(product_texts)
    
    print("Done!")

if __name__ == '__main__':
    main()