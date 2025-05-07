import os
import psycopg2
from psycopg2.extras import execute_values, Json
import numpy as np
from typing import Dict, Union, Optional
from dotenv import load_dotenv
import faiss
import json

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

# FAISS index configuration
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_INDEX_DIR = os.path.join(REPO_ROOT, os.getenv('FAISS_INDEX_DIR', 'data/faiss_indexes'))

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
                FROM products
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
    def __init__(self, index_type: str = 'text'):
        """
        Initialize FAISS retrieval with saved index.
        
        Args:
            index_type: Either 'text' or 'image' to specify which index to use
        """
        if index_type not in ['text', 'image']:
            raise ValueError("index_type must be either 'text' or 'image'")
            
        # Load the index
        index_path = os.path.join(FAISS_INDEX_DIR, f'{index_type}_index.faiss')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Please ensure the index has been created and FAISS_INDEX_DIR is set correctly in .env")
        self.index = faiss.read_index(index_path)
        
        # Load the PID mapping
        mapping_path = os.path.join(FAISS_INDEX_DIR, f'{index_type}_index_mapping.json')
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Index mapping not found at {mapping_path}. Please ensure the mapping has been created and FAISS_INDEX_DIR is set correctly in .env")
        with open(mapping_path, 'r') as f:
            self.idx_to_pid = json.load(f)
            
        self.column_name = f'{index_type}_embedding'

    def score(self, query: np.ndarray, k: int = 10) -> Dict[str, float]:
        # Search using FAISS (dot product = cosine similarity since vectors are normalized)
        distances, indices = self.index.search(query.reshape(1, -1), k)

        # Raw cosine similarity scores
        raw_scores = distances[0]
        top_indices = indices[0]

        # Normalize scores to [0, 1] based on highest score (assumes higher = more similar)
        max_score = max(raw_scores) if len(raw_scores) > 0 else 1.0
        normalized_scores = {
            self.idx_to_pid[str(idx)]: float(score / max_score) if max_score > 0 else 0.0
            for idx, score in zip(top_indices, raw_scores)
            if str(idx) in self.idx_to_pid  # Only include valid indices
        }

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
                FROM products
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