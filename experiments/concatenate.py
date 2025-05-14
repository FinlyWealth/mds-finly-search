import os
import sys
import psycopg2
import pgvector.psycopg2
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.db import DB_CONFIG
from src.backend.embedding import get_text_embedding
from sentence_transformers import SentenceTransformer

def generate_combined_embeddings(text):
    """
    Generate combined embeddings by concatenating CLIP and MiniLM embeddings for text
    
    Args:
        text (str): Text to generate embeddings for
    
    Returns:
        numpy.ndarray: The combined embedding vector
    """
    # Get CLIP embedding
    clip_embedding = get_text_embedding(text)
    
    # Get MiniLM embedding
    minilm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    minilm_embedding = minilm_model.encode(text)
    
    # Concatenate the embeddings
    combined_embedding = np.concatenate([clip_embedding, minilm_embedding])
    
    return combined_embedding

def concatenate_embeddings():
    """Concatenate image_clip_embedding and minilm_embedding columns into a new column"""
    conn = psycopg2.connect(**DB_CONFIG)
    pgvector.psycopg2.register_vector(conn)
    cur = conn.cursor()
    
    # Add new column for concatenated embeddings
    cur.execute("""
        ALTER TABLE products 
        ADD COLUMN IF NOT EXISTS combined_embedding vector;
    """)
    
    # Get the dimensions of both embedding types
    cur.execute("SELECT image_clip_embedding, minilm_embedding FROM products LIMIT 1")
    sample = cur.fetchone()
    if sample and sample[0] is not None and sample[1] is not None:
        image_dim = len(sample[0])
        minilm_dim = len(sample[1])
        total_dim = image_dim + minilm_dim
        
        # Update the column type with the correct dimension
        cur.execute(f"ALTER TABLE products ALTER COLUMN combined_embedding TYPE vector({total_dim})")
        
        # Update the combined_embedding column by concatenating the two embeddings
        cur.execute("""
            UPDATE products 
            SET combined_embedding = image_clip_embedding || minilm_embedding
            WHERE image_clip_embedding IS NOT NULL 
            AND minilm_embedding IS NOT NULL;
        """)
        
        conn.commit()
        print(f"Successfully concatenated embeddings. New dimension: {total_dim}")
    else:
        print("Error: Could not determine embedding dimensions")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    concatenate_embeddings()
