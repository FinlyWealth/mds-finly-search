import os
import psycopg2
from psycopg2.extras import execute_values, Json
import numpy as np
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
    
    # Create the table with dynamic vector dimensions
    cur.execute(f"""
        CREATE TABLE products (
            id SERIAL PRIMARY KEY,
            Pid TEXT UNIQUE,
            text_embedding vector({embedding_dim}),
            image_embedding vector({embedding_dim}),
            document tsvector
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
    
    # Create single products table with dynamic vector dimensions
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS products (
            id SERIAL PRIMARY KEY,
            Pid TEXT UNIQUE,
            text_embedding vector({embedding_dim}),
            image_embedding vector({embedding_dim}),
            document tsvector
        );
    """)
    
    conn.commit()
    cur.close()
    conn.close()

def store_embeddings(text_embeddings, image_embeddings, pids):
    """Store embeddings in the database"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Create a set to track seen Pids
    seen_ids = set()
    data = []
    
    # Only add the first occurrence of each Pid
    for pid, text_emb, img_emb in zip(pids, text_embeddings, image_embeddings):
        if pid not in seen_ids:
            seen_ids.add(pid)
            data.append((
                pid,
                [float(x) for x in text_emb],
                [float(x) for x in img_emb],
                None  # Document will be updated separately
            ))
    
    # Store all data in a single table
    execute_values(
        cur,
        """
        INSERT INTO products (Pid, text_embedding, image_embedding, document) 
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

def text_search(query, top_k=5):
    """Find top-k most relevant documents using full-text search"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    query = f"""
        SELECT Pid, ts_rank(document, plainto_tsquery('english', %s)) as rank
        FROM products
        WHERE document @@ plainto_tsquery('english', %s)
        ORDER BY rank DESC
        LIMIT %s
    """
    
    cur.execute(query, (query, query, top_k))
    results = cur.fetchall()
    
    cur.close()
    conn.close()
    
    if not results:
        return [], []
    
    pids, ranks = zip(*results)
    return list(pids), list(ranks)

def hybrid_search(
    query,
    query_embedding,
    top_k=10,
    text_weight=0.5,
    image_weight=0.3,
    ts_weight=0.2
):
    """Find top-k most relevant products using hybrid search"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Convert numpy array to Python list of floats
    query_vector = [float(x) for x in query_embedding]
    
    # Build the query based on whether we have a text query or image query
    """
    final_score =
        text_weight  * (1 - cosine_distance(text_embedding, query_embedding) / 2) +
        image_weight * (1 - cosine_distance(image_embedding, query_embedding) / 2) +
        textmatch_weight * sigmoid(ts_rank_cd(document, plainto_tsquery(query)))

    - pgvector's <=> operator computes cosine distance. Therefore it needs to be converted to cosine similarities and scaled to [0, 1] range.
    - Sigmoid is applied to ts_rank_cd() to scale its output to the [0, 1] range.
    """

    sql = """
        WITH scores AS (
            SELECT 
                Pid,
                %s * ((1 - (text_embedding <=> %s::vector)) / 2) AS text_score,
                %s * ((1 - (image_embedding <=> %s::vector)) / 2) AS image_score,
                %s * COALESCE(
                    (1 / (1 + EXP(-ts_rank_cd(document, plainto_tsquery('english', %s))))),
                    0
                ) AS ts_score
            FROM products
        )
        SELECT Pid, text_score, image_score, ts_score
        FROM scores
        ORDER BY (text_score + image_score + ts_score) DESC
        LIMIT %s;
    """
    params = (
        text_weight, query_vector,
        image_weight, query_vector,
        ts_weight, query,
        top_k
    )
    
    cur.execute(sql, params)
    results = cur.fetchall()

    cur.close()
    conn.close()

    if not results:
        return [], []

    pids = []
    final_scores = []
    components = []

    for row in results:
        pid, text_score, image_score, ts_score = row
        total_score = text_score + image_score + ts_score
        components.append({
            "text": round(text_score, 4),
            "image": round(image_score, 4),
            "ts": round(ts_score, 4),
            "total": round(total_score, 4)
        })
        
        pids.append(pid)
        final_scores.append(total_score)

    # Print breakdowns
    for pid, comp in zip(pids, components):
        print(f"Product ID: {pid} | Text: {comp['text']} | Image: {comp['image']} | TS: {comp['ts']} | Total: {comp['total']}")

    return pids, final_scores