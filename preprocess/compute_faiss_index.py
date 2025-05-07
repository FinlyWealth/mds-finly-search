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

# FAISS configuration
N_LIST = int(os.getenv('FAISS_NLIST', '100'))  # Default to 100 if not set

def load_embeddings_from_db():
    """Load text and image embeddings from the database"""
    conn = psycopg2.connect(**DB_CONFIG)
    pgvector.psycopg2.register_vector(conn)
    cur = conn.cursor()
    
    # Load text embeddings
    cur.execute("SELECT Pid, text_embedding FROM products WHERE text_embedding IS NOT NULL ORDER BY Pid")
    text_data = cur.fetchall()
    text_pids = [row[0] for row in text_data]
    text_embeddings = np.array([row[1] for row in text_data], dtype=np.float32)
    
    # Get embedding dimension from the first embedding
    if len(text_embeddings) > 0:
        embedding_dim = len(text_embeddings[0])
    else:
        raise ValueError("No text embeddings found in the database")
    
    # Load image embeddings
    cur.execute("SELECT Pid, image_embedding FROM products WHERE image_embedding IS NOT NULL ORDER BY Pid")
    image_data = cur.fetchall()
    image_pids = [row[0] for row in image_data]
    image_embeddings = np.array([row[1] for row in image_data], dtype=np.float32)
    
    cur.close()
    conn.close()
    
    return text_pids, text_embeddings, image_pids, image_embeddings, embedding_dim

def create_faiss_index(embeddings, pids, embedding_dim, nlist=N_LIST):
    """
    Create a FAISS index with cosine similarity and explicit PID mapping
    nlist: number of clusters for the IVF index (configurable via FAISS_NLIST env var)
    """
    # Create PID to index mapping
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    idx_to_pid = {i: pid for pid, i in pid_to_idx.items()}
    
    # Convert to FAISS-compatible IDs
    ids = np.array([pid_to_idx[pid] for pid in pids], dtype=np.int64)
    
    # Create the base index
    quantizer = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
    index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # Train the index
    if not index.is_trained:
        if len(embeddings) < nlist * 10:
            raise ValueError(f"Not enough vectors to train the index. Got {len(embeddings)}, need at least {nlist * 10}.")
        index.train(embeddings)
    
    # Create ID-mapped index and add vectors with IDs
    id_index = faiss.IndexIDMap(index)
    id_index.add_with_ids(embeddings, ids)
    
    return id_index, idx_to_pid

def verify_index(index, embeddings, pids):
    """Verify that all embeddings are present in the index and mapped to correct PIDs."""
    print(f"\nVerifying index with {len(embeddings)} embeddings...")
    mismatches = 0
    
    for i, (embedding, pid) in enumerate(zip(embeddings, pids)):
        # Normalize embedding for cosine similarity
        normalized_emb = embedding.reshape(1, -1).copy()
        faiss.normalize_L2(normalized_emb)
        
        # Search for the embedding
        D, I = index.search(normalized_emb, 1)
        
        # Check if the embedding is found with high similarity
        if D[0][0] < 0.99:  # Cosine similarity threshold
            print(f"❌ Embedding for PID {pid} not found in index")
            print(f"Best match similarity: {D[0][0]}")
            mismatches += 1
            
        if i > 0 and i % 1000 == 0:
            print(f"Verified {i} embeddings...")
    
    if mismatches == 0:
        print("✅ All embeddings verified successfully!")
    else:
        print(f"❌ Found {mismatches} mismatches out of {len(embeddings)} embeddings")
    
    return mismatches == 0

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
    text_index, text_pid_map = create_faiss_index(text_embeddings, text_pids, embedding_dim)
    
    print("Verifying text index...")
    if not verify_index(text_index, text_embeddings, text_pids):
        raise ValueError("Text index verification failed!")
    
    print(f"Creating image embedding index (dimension: {embedding_dim})...")
    image_index, image_pid_map = create_faiss_index(image_embeddings, image_pids, embedding_dim)
    
    print("Verifying image index...")
    if not verify_index(image_index, image_embeddings, image_pids):
        raise ValueError("Image index verification failed!")
    
    # Save the indexes
    print("Saving indexes...")
    save_index(text_index, os.path.join(output_dir, 'text_index.faiss'))
    save_index(image_index, os.path.join(output_dir, 'image_index.faiss'))
    
    # Save the mappings
    print("Saving index mappings...")
    save_mapping(text_pid_map, os.path.join(output_dir, 'text_index_mapping.json'))
    save_mapping(image_pid_map, os.path.join(output_dir, 'image_index_mapping.json'))
    
    # Save the product IDs (keeping this for backward compatibility)
    np.save(os.path.join(output_dir, 'text_pids.npy'), text_pids)
    np.save(os.path.join(output_dir, 'image_pids.npy'), image_pids)
    
    print("Done!")

if __name__ == '__main__':
    main()
