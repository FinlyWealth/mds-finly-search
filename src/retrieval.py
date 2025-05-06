import os
import psycopg2
from psycopg2.extras import execute_values, Json
import numpy as np
from typing import Dict, Union, Optional
from dotenv import load_dotenv
import faiss

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
            SELECT 
                Pid,
                1 - ({self.column_name} <=> %s::vector) AS score
            FROM products
            ORDER BY {self.column_name} <=> %s::vector
            LIMIT %s
        """
        
        cur.execute(sql, [query_vector, query_vector, k])
        results = {pid: score for pid, score in cur.fetchall()}
        
        cur.close()
        conn.close()
        return results

class FaissVectorRetrieval(SimilarityRetrieval):
    """FAISS vector search"""
    def __init__(self, index: faiss.Index, pids: list[int], column_name: str):
        self.index = index
        self.pids = pids  # Array of PIDs corresponding to FAISS indices
        self.column_name = column_name

    def score(self, query: np.ndarray, k: int = 10) -> Dict[int, float]:
        # Run FAISS search
        distances, indices = self.index.search(query.reshape(1, -1), k)
        
        # Convert results to score dict
        pids = [self.pids[idx] for idx in indices[0]]
        scores = {pid: 1 - (dist / 2) for pid, dist in zip(pids, distances[0])}
        return scores

class TextSearchRetrieval(SimilarityRetrieval):
    """Text search using PostgreSQL full-text search"""
    def __init__(self, method: str, db_config: Dict[str, str]):
        self.method = method  # e.g., 'ts_rank', 'ts_rank_cd'
        self.db_config = db_config

    def score(self, query: str, k: int = 10) -> Dict[int, float]:
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        sql = f"""
            SELECT 
                Pid,
                {self.method}(document, plainto_tsquery('english', %s)) AS score
            FROM products
            WHERE document @@ plainto_tsquery('english', %s)
            ORDER BY {self.method}(document, plainto_tsquery('english', %s)) DESC
            LIMIT %s
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
) -> tuple[list[int], list[float]]:
    """
    Find top-k most relevant products using hybrid search.
    
    Args:
        query: Text query for text search components
        query_embedding: Vector query for vector search components
        components: List of SimilarityRetrieval instances
        weights: List of weights for each component (must sum to 1)
        top_k: Number of results to return
    
    Returns:
        Tuple of (pids, scores)
    """
    # Get scores from each component
    all_scores = []
    for comp in components:
        if isinstance(comp, (PostgresVectorRetrieval, FaissVectorRetrieval)):
            scores = comp.score(query_embedding, k=top_k)
        else:
            scores = comp.score(query, k=top_k)
        all_scores.append(scores)

    # Combine scores
    combined_scores = {}
    for scores, weight in zip(all_scores, weights):
        for pid, score in scores.items():
            combined_scores[pid] = combined_scores.get(pid, 0) + score * weight

    # Sort and get top_k
    top = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    pids, scores = zip(*top) if top else ([], [])
    return list(pids), list(scores)