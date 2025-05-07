import os
import numpy as np
import faiss
import psycopg2
import pgvector.psycopg2
import json
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'dbname': os.getenv('PGDATABASE', 'finly'),
    'user': os.getenv('PGUSER', 'postgres'),
    'password': os.getenv('PGPASSWORD', 'postgres'),
    'host': os.getenv('PGHOST', 'localhost'),
    'port': os.getenv('PGPORT', '5432')
}

def load_embeddings_from_db():
    """Load text and image embeddings from the database"""
    conn = psycopg2.connect(**DB_CONFIG)
    pgvector.psycopg2.register_vector(conn)
    cur = conn.cursor()
    
    # Load text embeddings
    cur.execute("SELECT Pid, text_embedding FROM products WHERE text_embedding IS NOT NULL")
    text_data = cur.fetchall()
    text_pids = [row[0] for row in text_data]
    text_embeddings = np.array([row[1] for row in text_data], dtype=np.float32)
    
    # Get embedding dimension from the first embedding
    if len(text_embeddings) > 0:
        embedding_dim = len(text_embeddings[0])
    else:
        raise ValueError("No text embeddings found in the database")
    
    # Load image embeddings
    cur.execute("SELECT Pid, image_embedding FROM products WHERE image_embedding IS NOT NULL")
    image_data = cur.fetchall()
    image_pids = [row[0] for row in image_data]
    image_embeddings = np.array([row[1] for row in image_data], dtype=np.float32)
    
    cur.close()
    conn.close()
    
    return text_pids, text_embeddings, image_pids, image_embeddings, embedding_dim

def create_faiss_index(embeddings, embedding_dim, nlist=100):
    """
    Create a FAISS index with cosine similarity
    nlist: number of clusters for the IVF index
    """
    # Normalize the embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create the index
    quantizer = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
    index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # Train the index
    if not index.is_trained and len(embeddings) > nlist:
        index.train(embeddings)
    
    # Add the embeddings to the index
    index.add(embeddings)
    
    return index

def save_index(index, filename):
    """Save the FAISS index to disk"""
    faiss.write_index(index, filename)

def create_index_mapping(pids):
    """Create a mapping between index positions and PIDs"""
    return {str(i): pid for i, pid in enumerate(pids)}

def save_mapping(mapping, filename):
    """Save the index-to-PID mapping to disk"""
    with open(filename, 'w') as f:
        json.dump(mapping, f)

def main():
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'faiss_indexes')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading embeddings from database...")
    text_pids, text_embeddings, image_pids, image_embeddings, embedding_dim = load_embeddings_from_db()
    
    print(f"Creating text embedding index (dimension: {embedding_dim})...")
    text_index = create_faiss_index(text_embeddings, embedding_dim)
    
    print(f"Creating image embedding index (dimension: {embedding_dim})...")
    image_index = create_faiss_index(image_embeddings, embedding_dim)
    
    # Create index-to-PID mappings
    print("Creating index mappings...")
    text_mapping = create_index_mapping(text_pids)
    image_mapping = create_index_mapping(image_pids)
    
    # Save the indexes
    print("Saving indexes...")
    save_index(text_index, os.path.join(output_dir, 'text_index.faiss'))
    save_index(image_index, os.path.join(output_dir, 'image_index.faiss'))
    
    # Save the mappings
    print("Saving index mappings...")
    save_mapping(text_mapping, os.path.join(output_dir, 'text_index_mapping.json'))
    save_mapping(image_mapping, os.path.join(output_dir, 'image_index_mapping.json'))
    
    # Save the product IDs (keeping this for backward compatibility)
    np.save(os.path.join(output_dir, 'text_pids.npy'), text_pids)
    np.save(os.path.join(output_dir, 'image_pids.npy'), image_pids)
    
    print("Done!")

if __name__ == '__main__':
    main()
