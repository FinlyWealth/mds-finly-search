import os
import numpy as np
import faiss
import json
import sys
import psutil
import gc
from tqdm import tqdm
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config.path import EMBEDDINGS_PATH

# FAISS configuration
N_LIST_VALUES = [int(x) for x in os.getenv('FAISS_NLIST', '100').split(',')]  # Accept comma-separated list of nlist values
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10000'))  # Default batch size for database loading
COMPRESSED = os.getenv('COMPRESSED', 'false').lower() == 'true'  # Whether to use scalar quantization

def get_memory_usage():
    """Get current memory usage in GB.
    
    Returns
    -------
    float
        Current memory usage in gigabytes.
    """
    return psutil.Process().memory_info().rss / 1024 / 1024 / 1024

def load_embeddings_from_files():
    """Load embeddings from NPZ files in the embeddings folder.
    
    Returns
    -------
    dict
        Dictionary containing embedding data for each type, with keys:
        - 'pids': List of product IDs
        - 'embeddings': numpy.ndarray of embeddings
        - 'dim': int, dimension of the embeddings
    
    Raises
    ------
    ValueError
        If no embedding NPZ files are found or no valid embeddings are found.
    """
    embeddings_dir = EMBEDDINGS_PATH
    
    embeddings_data = {}
    
    # Find all NPZ files and group them by their prefix
    npz_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.npz')]
    embedding_groups = {}
    
    for npz_file in npz_files:
        # Extract the prefix (everything before _chunk)
        if '_chunk_' in npz_file:
            prefix = npz_file.split('_chunk_')[0]
            if prefix not in embedding_groups:
                embedding_groups[prefix] = []
            embedding_groups[prefix].append(npz_file)
    
    if not embedding_groups:
        raise ValueError("No embedding NPZ files found in the embeddings folder")
    
    print(f"Found {len(embedding_groups)} embedding types to process...")
    
    # Process each embedding type
    for embedding_type, npz_files in embedding_groups.items():
        print(f"\nProcessing {embedding_type} embeddings...")
        print(f"Found {len(npz_files)} NPZ files for {embedding_type}")
        
        all_pids = []
        all_embeddings = []
        
        # Load each NPZ file for this embedding type
        for npz_file in tqdm(sorted(npz_files), desc=f"Loading {embedding_type} NPZ files"):
            file_path = os.path.join(embeddings_dir, npz_file)
            data = np.load(file_path)
            
            # Load data using correct key names
            pids = data['product_ids']
            embeddings = data['embeddings']
            
            all_pids.extend(pids)
            all_embeddings.append(embeddings)
            
            # Print memory usage after each file
            print(f"\nCurrent memory usage after loading {npz_file}: {get_memory_usage():.2f} GB")
        
        # Concatenate all embeddings for this type
        all_embeddings = np.vstack(all_embeddings)
        
        if len(all_embeddings) == 0:
            print(f"Warning: No embeddings found for {embedding_type}")
            continue
        
        embeddings_data[embedding_type] = {
            'pids': all_pids,
            'embeddings': all_embeddings,
            'dim': len(all_embeddings[0])
        }
    
    if not embeddings_data:
        raise ValueError("No valid embeddings found in any of the NPZ files")
    
    return embeddings_data

def create_faiss_index(embeddings, pids, embedding_dim, nlist, compressed=False):
    """Create a FAISS index with cosine similarity and explicit PID mapping.
    
    Parameters
    ----------
    embeddings : numpy.ndarray
        Array of embeddings to index.
    pids : list
        List of product IDs corresponding to the embeddings.
    embedding_dim : int
        Dimension of the embeddings.
    nlist : int
        Number of clusters for the IVF index.
    compressed : bool, optional
        Whether to use scalar quantization for compression, by default False.
    
    Returns
    -------
    tuple
        A tuple containing:
        - faiss.Index: The trained FAISS index
        - dict: Mapping from index positions to PIDs
    
    Raises
    ------
    ValueError
        If there are not enough vectors to train the index.
    """
    print(f"Creating {'compressed' if compressed else 'uncompressed'} index with {len(embeddings)} vectors and nlist={nlist}...")
    
    # Create PID to index mapping
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    idx_to_pid = {i: pid for pid, i in pid_to_idx.items()}
    
    # Convert to FAISS-compatible IDs
    ids = np.array([pid_to_idx[pid] for pid in pids], dtype=np.int64)
    
    # Create the base index
    quantizer = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
    
    if compressed:
        # Create a scalar quantized IVF index
        # Using 8 bits per dimension for quantization
        index = faiss.IndexIVFScalarQuantizer(
            quantizer, 
            embedding_dim, 
            nlist, 
            faiss.ScalarQuantizer.QT_8bit,
            faiss.METRIC_INNER_PRODUCT
        )
    else:
        # Create the standard IVF index
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
    """Verify that all embeddings are present in the index and mapped to correct PIDs.
    
    Parameters
    ----------
    index : faiss.Index
        The FAISS index to verify.
    embeddings : numpy.ndarray
        Array of embeddings to verify.
    pids : list
        List of product IDs corresponding to the embeddings.
    
    Returns
    -------
    bool
        True if all embeddings are verified successfully, False otherwise.
    """
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
    """Save the FAISS index to disk.
    
    Parameters
    ----------
    index : faiss.Index
        The FAISS index to save.
    filename : str
        Path where the index will be saved.
    """
    faiss.write_index(index, filename)

def create_index_mapping(pids):
    """Create a mapping between index positions and PIDs.
    
    Parameters
    ----------
    pids : list
        List of product IDs.
    
    Returns
    -------
    dict
        Dictionary mapping index positions to PIDs.
    """
    return {str(i): pid for i, pid in enumerate(pids)}

def save_mapping(mapping, filename):
    """Save the index-to-PID mapping to disk.
    
    Parameters
    ----------
    mapping : dict
        Dictionary containing the index-to-PID mapping.
    filename : str
        Path where the mapping will be saved.
    """
    with open(filename, 'w') as f:
        json.dump(mapping, f)

def main():
    """Main function to create and save FAISS indices for different embedding types."""
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'faiss_indices')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    print("Loading embeddings from files...")
    embeddings_data = load_embeddings_from_files()
    
    # Create and save indexes for each embedding type and nlist value
    for embedding_type, data in embeddings_data.items():
        # Remove 'embeddings' from the type name
        base_type = embedding_type.replace('_embeddings', '')
        print(f"\nProcessing {embedding_type} embeddings...")
        
        for nlist in N_LIST_VALUES:
            print(f"\nCreating {embedding_type} index with nlist={nlist} (dimension: {data['dim']})...")
            
            index_path = os.path.join(output_dir, f'{base_type}_index.faiss')
            mapping_path = os.path.join(output_dir, f'{base_type}_index_mapping.json')
            pids_path = os.path.join(output_dir, f'{base_type}_pids.npy')
            
            if os.path.exists(index_path) and os.path.exists(mapping_path):
                print(f"Index exists at {index_path}, loading and appending new embeddings...")
                index = faiss.read_index(index_path)
                with open(mapping_path, 'r') as f:
                    existing_mapping = json.load(f)
                # Convert mapping to pid->idx
                existing_pid_to_idx = {pid: int(idx) for idx, pid in existing_mapping.items()}
                existing_idxs = set(existing_pid_to_idx.values())
                loaded_pids = data['pids']
                loaded_embeddings = data['embeddings']
                # Identify new entries
                new_entries = [(i, pid) for i, pid in enumerate(loaded_pids) if pid not in existing_pid_to_idx]
                if new_entries:
                    start_idx = max(existing_idxs) + 1 if existing_idxs else 0
                    indices, new_pids = zip(*new_entries)
                    new_embeddings = loaded_embeddings[list(indices)]
                    new_ids = np.arange(start_idx, start_idx + len(new_pids), dtype=np.int64)
                    index.add_with_ids(new_embeddings, new_ids)
                    for pid, id_val in zip(new_pids, new_ids):
                        existing_mapping[str(id_val)] = pid
                    print(f"Added {len(new_pids)} new embeddings to index")
                else:
                    print("No new embeddings to add; index is up to date")
                print(f"Saving updated index to {index_path}...")
                save_index(index, index_path)
                print(f"Saving updated index mapping to {mapping_path}...")
                save_mapping(existing_mapping, mapping_path)
                print(f"Saving updated product IDs to {pids_path}...")
                np.save(pids_path, data['pids'])
            else:
                index, pid_map = create_faiss_index(
                    data['embeddings'],
                    data['pids'],
                    data['dim'],
                    nlist,
                    compressed=COMPRESSED
                )
                print(f"Saving new index to {index_path}...")
                save_index(index, index_path)
                print(f"Saving new index mapping to {mapping_path}...")
                save_mapping(pid_map, mapping_path)
                print(f"Saving new product IDs to {pids_path}...")
                np.save(pids_path, data['pids'])
            # print(f"Verifying {embedding_type} index with nlist={nlist}...")
            # verify_size = min(1000, len(data['embeddings']))
            # verify_index(index, data['embeddings'][:verify_size], data['pids'][:verify_size])
            del index
            gc.collect()
            print(f"Memory usage after processing {embedding_type} with nlist={nlist}: {get_memory_usage():.2f} GB")
        
        # Clear embeddings data after processing all nlist values
        del data['embeddings']
        gc.collect()
    
    print("\nDone!")

if __name__ == '__main__':
    main()
