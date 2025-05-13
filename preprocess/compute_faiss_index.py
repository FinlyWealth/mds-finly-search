import os
import numpy as np
import faiss
import psycopg2
import pgvector.psycopg2
import json
from psycopg2.extras import execute_values
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.db import DB_CONFIG
from config.embeddings import get_enabled_embedding_types

# FAISS configuration
N_LIST = int(os.getenv('FAISS_NLIST', '100'))  # Default to 100 if not set

def load_embeddings_from_db():
    """Load embeddings from the database for all enabled embedding types"""
    conn = psycopg2.connect(**DB_CONFIG)
    pgvector.psycopg2.register_vector(conn)
    cur = conn.cursor()
    
    embeddings_data = {}
    embedding_dim = None
    
    # Load embeddings for each enabled type
    for embedding_type in get_enabled_embedding_types():
        column_name = f"{embedding_type}_embedding"
        cur.execute(f"SELECT Pid, {column_name} FROM products WHERE {column_name} IS NOT NULL ORDER BY Pid")
        data = cur.fetchall()
        pids = [row[0] for row in data]
        embeddings = np.array([row[1] for row in data], dtype=np.float32)
        
        # Verify embedding dimension
        if len(embeddings) > 0:
            if embedding_dim is None:
                embedding_dim = len(embeddings[0])
            elif len(embeddings[0]) != embedding_dim:
                raise ValueError(f"Inconsistent embedding dimensions: {embedding_type} has {len(embeddings[0])} dimensions")
        else:
            raise ValueError(f"No {embedding_type} embeddings found in the database")
        
        embeddings_data[embedding_type] = {
            'pids': pids,
            'embeddings': embeddings
        }
    
    cur.close()
    conn.close()
    
    return embeddings_data, embedding_dim

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
    embeddings_data, embedding_dim = load_embeddings_from_db()
    
    # Create and save indexes for each embedding type
    for embedding_type, data in embeddings_data.items():
        print(f"\nProcessing {embedding_type} embeddings...")
        print(f"Creating {embedding_type} index (dimension: {embedding_dim})...")
        
        index, pid_map = create_faiss_index(
            data['embeddings'], 
            data['pids'], 
            embedding_dim
        )
        
        # Save the index
        index_path = os.path.join(output_dir, f'{embedding_type}_index.faiss')
        print(f"Saving index to {index_path}...")
        save_index(index, index_path)
        
        # Save the mapping
        mapping_path = os.path.join(output_dir, f'{embedding_type}_index_mapping.json')
        print(f"Saving index mapping to {mapping_path}...")
        save_mapping(pid_map, mapping_path)
        
        # Save the product IDs (keeping this for backward compatibility)
        pids_path = os.path.join(output_dir, f'{embedding_type}_pids.npy')
        print(f"Saving product IDs to {pids_path}...")
        np.save(pids_path, data['pids'])
        
        # Verify the index
        print(f"Verifying {embedding_type} index...")
        verify_index(index, data['embeddings'], data['pids'])
    
    print("\nDone!")

if __name__ == '__main__':
    main()
