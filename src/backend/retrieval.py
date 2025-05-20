import os
import psycopg2
from psycopg2.extras import execute_values, Json
import numpy as np
from typing import Dict, Union, Optional
import faiss
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config.db import DB_CONFIG, TABLE_NAME

# FAISS index configuration
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAISS_INDEX_DIR = os.path.join(REPO_ROOT, 'data', 'faiss_indices')

# Cache for FAISS indices and mappings
_faiss_index_cache = {}
_faiss_mapping_cache = {}

class SimilarityRetrieval:
    """Base class for similarity retrieval"""
    def score(self, query: Union[str, np.ndarray], k: int = 10) -> Dict[int, float]:
        """
        Return a {pid: score} dict for the given query.
        
        Args:
            query: Query (text or vector)
            k: Number of top results to return
        """
        raise NotImplementedError

class PostgresVectorRetrieval(SimilarityRetrieval):
    """Postgres vector search using pgvector"""
    def __init__(self, column_name: str, db_config: Dict[str, str]):
        self.column_name = column_name
        self.db_config = db_config

    def score(self, query: np.ndarray, k: int = 10) -> Dict[int, float]:
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        # Convert numpy array to Python list of floats
        query_vector = query.tolist()
        
        sql = f"""
            WITH scores AS (
                SELECT 
                    Pid,
                    1 - ({self.column_name} <=> %s::vector) AS raw_score
                FROM {TABLE_NAME}
                ORDER BY {self.column_name} <=> %s::vector
                LIMIT %s
            )
            SELECT 
                Pid,
                CASE 
                    WHEN MAX(raw_score) OVER () = 0 THEN 0
                    ELSE raw_score / MAX(raw_score) OVER ()
                END AS normalized_score
            FROM scores
        """
        
        cur.execute(sql, [query_vector, query_vector, k])
        results = {pid: score for pid, score in cur.fetchall()}
        
        cur.close()
        conn.close()
        return results

class FaissVectorRetrieval(SimilarityRetrieval):
    """FAISS vector search using saved indexes"""
    def __init__(self, column_name: str, nprobe: int = 1, db_config: Dict[str, str] = None):
        """
        Initialize FAISS retrieval with saved index.
        
        Args:
            column_name: Name of the embedding column (e.g., 'text_embedding', 'image_embedding', 'fusion_embedding')
            nprobe: Number of clusters to visit during search (default: 1)
            db_config: Database configuration dictionary
        """
        # Extract the base type from the column name (e.g., 'fusion' from 'fusion_embedding')
        index_type = column_name.replace('_embedding', '')
        if index_type not in ['text', 'image', 'fusion']:
            raise ValueError("column_name must end with '_embedding' and the base type must be 'text', 'image', or 'fusion'")
            
        # Load the index from cache or file
        index_path = os.path.join(FAISS_INDEX_DIR, f'{index_type}_index.faiss')
        if index_type not in _faiss_index_cache:
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index not found at {index_path}. Please ensure the index has been created and FAISS_INDEX_DIR is set correctly in .env")
            _faiss_index_cache[index_type] = faiss.read_index(index_path)
        self.index = _faiss_index_cache[index_type]
        
        # Set nprobe parameter
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
        
        # Load the PID mapping from cache or file
        mapping_path = os.path.join(FAISS_INDEX_DIR, f'{index_type}_index_mapping.json')
        if index_type not in _faiss_mapping_cache:
            if not os.path.exists(mapping_path):
                raise FileNotFoundError(f"Index mapping not found at {mapping_path}. Please ensure the mapping has been created and FAISS_INDEX_DIR is set correctly in .env")
            with open(mapping_path, 'r') as f:
                _faiss_mapping_cache[index_type] = json.load(f)
        self.idx_to_pid = _faiss_mapping_cache[index_type]
            
        self.column_name = column_name
        self.db_config = db_config or DB_CONFIG

    def score(self, query: np.ndarray, k: int = 10) -> Dict[str, float]:
        # First get the top k indices from FAISS
        distances, indices = self.index.search(query.reshape(1, -1), k)
        raw_scores = distances[0]
        top_indices = indices[0]

        # Get the corresponding PIDs
        pids = [self.idx_to_pid[str(idx)] for idx in top_indices if str(idx) in self.idx_to_pid]
        
        if not pids:
            return {}

        # Query the database to get the actual embeddings for these PIDs
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        # Convert list of PIDs to a comma-separated string for SQL
        pid_list = ','.join([f"'{pid}'" for pid in pids])
        
        sql = f"""
            SELECT Pid, {self.column_name} as embedding
            FROM {TABLE_NAME}
            WHERE Pid IN ({pid_list})
        """
        
        cur.execute(sql)
        results = cur.fetchall()
        
        # Calculate actual cosine similarity scores
        normalized_scores = {}
        for pid, embedding in results:
            if embedding is not None:
                # Parse the vector string into a numpy array
                # The format is like '[1,2,3]' or '{1,2,3}'
                vector_str = embedding.strip('[]{}')
                db_embedding = np.array([float(x) for x in vector_str.split(',')], dtype=np.float32)
                db_embedding = db_embedding / np.linalg.norm(db_embedding)
                
                # Calculate cosine similarity
                similarity = float(np.dot(query, db_embedding))
                normalized_scores[pid] = similarity
        
        cur.close()
        conn.close()
        
        # Normalize scores to [0, 1] based on highest score
        max_score = max(normalized_scores.values()) if normalized_scores else 1.0
        if max_score > 0:
            normalized_scores = {pid: score / max_score for pid, score in normalized_scores.items()}
        
        return normalized_scores

class TextSearchRetrieval(SimilarityRetrieval):
    """Text search using PostgreSQL full-text search"""
    def __init__(self, method: str, db_config: Dict[str, str]):
        self.method = method  # e.g., 'ts_rank', 'ts_rank_cd'
        self.db_config = db_config

    def score(self, query: str, k: int = 10) -> Dict[int, float]:
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        sql = f"""
            WITH scores AS (
                SELECT 
                    Pid,
                    {self.method}(document, plainto_tsquery('english', %s)) AS raw_score
                FROM {TABLE_NAME}
                WHERE document @@ plainto_tsquery('english', %s)
                ORDER BY {self.method}(document, plainto_tsquery('english', %s)) DESC
                LIMIT %s
            )
            SELECT 
                Pid,
                CASE 
                    WHEN MAX(raw_score) OVER () = 0 THEN 0
                    ELSE raw_score / MAX(raw_score) OVER ()
                END AS normalized_score
            FROM scores
        """
        
        cur.execute(sql, [query, query, query, k])
        results = {pid: score for pid, score in cur.fetchall()}
        
        cur.close()
        conn.close()
        return results


def create_retrieval_component(component_config, db_config=None):
    """Create a retrieval component from config."""
    component_type = component_config['type']
    params = component_config['params']
    
    if component_type == 'PostgresVectorRetrieval':
        return PostgresVectorRetrieval(params['column_name'], db_config)
    elif component_type == 'TextSearchRetrieval':
        return TextSearchRetrieval(params['rank_method'], db_config)
    elif component_type == 'FaissVectorRetrieval':
        return FaissVectorRetrieval(params['column_name'], params.get('nprobe', 1), db_config)
    else:
        raise ValueError(f"Unknown component type: {component_type}")

        
def hybrid_retrieval(
    query: str,
    query_embedding: np.ndarray,
    components: list[SimilarityRetrieval],
    weights: list[float],
    top_k: int = 10
) -> tuple[list[str], list[float]]:
    """
    Find top-k most relevant products using hybrid search.
    
    Args:
        query: Text query for text search components
        query_embedding: Vector query for vector search components
        components: List of SimilarityRetrieval instances
        weights: List of weights for each component (must sum to 1)
        top_k: Number of results to return
    
    Returns:
        Tuple of (pids, scores) where pids are strings
    """
    # Filter out components with zero weights
    active_components = [(comp, weight) for comp, weight in zip(components, weights) if weight > 0]
    
    # Get scores from each active component
    all_scores = []
    for comp, _ in active_components:
        if isinstance(comp, (PostgresVectorRetrieval, FaissVectorRetrieval)):
            scores = comp.score(query_embedding, k=top_k)
        else:
            scores = comp.score(query, k=top_k)
        all_scores.append(scores)

    # Combine scores
    combined_scores = {}
    for scores, (_, weight) in zip(all_scores, active_components):
        for pid, score in scores.items():
            combined_scores[pid] = combined_scores.get(pid, 0) + score * weight

    # Sort and get top_k
    top = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    pids, scores = zip(*top) if top else ([], [])
    return list(pids), list(scores)