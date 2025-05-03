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

    # # Print breakdowns
    # for pid, comp in zip(pids, components):
    #     print(f"Product ID: {pid} | Text: {comp['text']} | Image: {comp['image']} | TS: {comp['ts']} | Total: {comp['total']}")

    return pids, final_scores

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
