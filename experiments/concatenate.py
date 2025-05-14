import os
import sys
import psycopg2
import pgvector.psycopg2
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.db import DB_CONFIG
from src.backend.embedding import get_text_embedding
from transformers import AutoModel, AutoTokenizer

# Initialize device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

def generate_fusion_embeddings(text):
    """
    Generate fusion embeddings by concatenating CLIP and MiniLM embeddings for text
    
    Args:
        text (str): Text to generate embeddings for
    
    Returns:
        numpy.ndarray: The fusion embedding vector
    """
    # Get CLIP embedding
    clip_embedding = get_text_embedding(text)
    
    # Get MiniLM embedding
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    # Process text with MiniLM
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the [CLS] token embedding (first token) as the sentence embedding
        minilm_embedding = outputs.last_hidden_state[:, 0, :]
        minilm_embedding /= minilm_embedding.norm(dim=-1, keepdim=True)
        minilm_embedding = minilm_embedding.cpu().numpy()[0]
    
    # Concatenate the embeddings
    fusion_embedding = np.concatenate([clip_embedding, minilm_embedding])
    
    # Normalize the concatenated embedding
    fusion_embedding = fusion_embedding / np.linalg.norm(fusion_embedding)
    
    return fusion_embedding

def concatenate_embeddings():
    """Concatenate image_clip_embedding and minilm_embedding columns into a new column"""
    conn = psycopg2.connect(**DB_CONFIG)
    pgvector.psycopg2.register_vector(conn)
    cur = conn.cursor()
    
    # Add new column for fusion embeddings
    cur.execute("""
        ALTER TABLE products 
        ADD COLUMN IF NOT EXISTS fusion_embedding vector;
    """)
    
    # Get the dimensions of both embedding types
    cur.execute("SELECT image_clip_embedding, minilm_embedding FROM products LIMIT 1")
    sample = cur.fetchone()
    if sample and sample[0] is not None and sample[1] is not None:
        image_dim = len(sample[0])
        minilm_dim = len(sample[1])
        total_dim = image_dim + minilm_dim
        
        # Update the column type with the correct dimension
        cur.execute(f"ALTER TABLE products ALTER COLUMN fusion_embedding TYPE vector({total_dim})")
        
        # Update the fusion_embedding column by concatenating the two embeddings and normalizing
        cur.execute("""
            UPDATE products 
            SET fusion_embedding = l2_normalize(image_clip_embedding || minilm_embedding)
            WHERE image_clip_embedding IS NOT NULL 
            AND minilm_embedding IS NOT NULL;
        """)
        
        conn.commit()
        print(f"Successfully concatenated and normalized embeddings. New dimension: {total_dim}")
    else:
        print("Error: Could not determine embedding dimensions")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    concatenate_embeddings()
