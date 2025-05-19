import os
import numpy as np
import faiss
import json
import sys
import psutil
import gc
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# FAISS configuration
N_LIST_VALUES = [int(x) for x in os.getenv('FAISS_NLIST', '100').split(',')]  # Accept comma-separated list of nlist values
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10000'))  # Default batch size for database loading

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def load_embeddings_from_files():
    """Load embeddings from NPZ files in the data/embeddings folder"""
    embeddings_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'embeddings')
    
    embeddings_data = {}
    embedding_type = 'fusion'
    
    # Find all NPZ files for fusion embeddings
    npz_files = sorted([f for f in os.listdir(embeddings_dir) if f.startswith('fusion_embeddings_chunk_') and f.endswith('.npz')])
    
    if not npz_files:
        raise ValueError("No fusion embedding NPZ files found in the data/embeddings folder")
    
    print(f"Found {len(npz_files)} NPZ files to process...")
    
    all_pids = []
    all_embeddings = []
    
    # Load each NPZ file
    for npz_file in tqdm(npz_files, desc="Loading NPZ files"):
        file_path = os.path.join(embeddings_dir, npz_file)
        data = np.load(file_path)
        
        # Load data using correct key names
        pids = data['product_ids']
        embeddings = data['embeddings']
        
        all_pids.extend(pids)
        all_embeddings.append(embeddings)
        
        # Print memory usage after each file
        print(f"\nCurrent memory usage after loading {npz_file}: {get_memory_usage():.2f} GB")
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    if len(all_embeddings) == 0:
        raise ValueError("No fusion embeddings found in the NPZ files")
    
    embeddings_data[embedding_type] = {
        'pids': all_pids,
        'embeddings': all_embeddings,
        'dim': len(all_embeddings[0])
    }
    
    return embeddings_data

def create_faiss_index(embeddings, pids, embedding_dim, nlist):
    """
    Create a FAISS index with cosine similarity and explicit PID mapping
    nlist: number of clusters for the IVF index
    """
    print(f"Creating index with {len(embeddings)} vectors and nlist={nlist}...")
    
    # Create PID to index mapping
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    idx_to_pid = {i: pid for pid, i in pid_to_idx.items()}
    
    # Convert to FAISS-compatible IDs
    ids = np.array([pid_to_idx[pid] for pid in pids], dtype=np.int64)
    
    # Create the base index
    quantizer = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
    index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # Train the index with memory-efficient approach
    if not index.is_trained:
        if len(embeddings) < nlist * 10:
            raise ValueError(f"Not enough vectors to train the index. Got {len(embeddings)}, need at least {nlist * 10}.")
        
        print(f"Training index with nlist={nlist}...")
        # Use 30*nlist vectors for training
        train_size = min(40 * nlist, len(embeddings))
        print(f"Using {train_size} vectors for training (40*{nlist})")
        train_vectors = embeddings[:train_size]
        index.train(train_vectors)
        del train_vectors
        gc.collect()
    
    # Add vectors in batches to save memory
    print("Adding vectors to index...")
    batch_size = 100000
    for i in tqdm(range(0, len(embeddings), batch_size)):
        end_idx = min(i + batch_size, len(embeddings))
        batch_embeddings = embeddings[i:end_idx]
        batch_ids = ids[i:end_idx]
        index.add_with_ids(batch_embeddings, batch_ids)
        
        if i % 500000 == 0:
            print(f"\nCurrent memory usage: {get_memory_usage():.2f} GB")
    
    return index, idx_to_pid

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
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'faiss_indices')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    print("Loading embeddings from files...")
    embeddings_data = load_embeddings_from_files()
    
    # Create and save indexes for each embedding type and nlist value
    for embedding_type, data in embeddings_data.items():
        print(f"\nProcessing {embedding_type} embeddings...")
        
        for nlist in N_LIST_VALUES:
            print(f"\nCreating {embedding_type} index with nlist={nlist} (dimension: {data['dim']})...")
            
            index, pid_map = create_faiss_index(
                data['embeddings'], 
                data['pids'], 
                data['dim'],
                nlist
            )
            
            # Save the index
            index_path = os.path.join(output_dir, f'{embedding_type}_index_nlist{nlist}.faiss')
            print(f"Saving index to {index_path}...")
            save_index(index, index_path)
            
            # Save the mapping
            mapping_path = os.path.join(output_dir, f'{embedding_type}_index_mapping_nlist{nlist}.json')
            print(f"Saving index mapping to {mapping_path}...")
            save_mapping(pid_map, mapping_path)
            
            # Save the product IDs (keeping this for backward compatibility)
            pids_path = os.path.join(output_dir, f'{embedding_type}_pids_nlist{nlist}.npy')
            print(f"Saving product IDs to {pids_path}...")
            np.save(pids_path, data['pids'])
            
            # Verify the index with a subset of vectors
            print(f"Verifying {embedding_type} index with nlist={nlist}...")
            verify_size = min(1000, len(data['embeddings']))
            verify_index(index, data['embeddings'][:verify_size], data['pids'][:verify_size])
            
            # Clear memory
            del index
            gc.collect()
            print(f"Memory usage after processing {embedding_type} with nlist={nlist}: {get_memory_usage():.2f} GB")
        
        # Clear embeddings data after processing all nlist values
        del data['embeddings']
        gc.collect()
    
    print("\nDone!")

if __name__ == '__main__':
    main()
