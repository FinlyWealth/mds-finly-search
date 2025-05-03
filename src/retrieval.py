import os
import psycopg2
from psycopg2.extras import execute_values, Json
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
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
            queries: List of queries (text or vector)
            k: Number of top results to return per query
        """
        raise NotImplementedError

class PostgresVectorRetrieval(SimilarityRetrieval):
    """Postgres vector search using pgvector"""
    def __init__(self, embedding_type: str, db_config: Dict[str, str]):
        self.embedding_type = embedding_type
        self.db_config = db_config

    def score(self, query: Union[str, np.ndarray], k: int = 10) -> Dict[int, float]:
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        # Convert numpy arrays to Python lists of floats
        query_vector = [float(x) for x in query]
        
        # Build the query using LATERAL join for batch processing
        sql = f"""
            WITH query_vectors AS (
                SELECT unnest(%s::vector[]) as query_vector
            )
            SELECT 
                qv.query_vector,
                p.Pid,
                (1 - ({self.embedding_type} <=> qv.query_vector)) / 2 AS score
            FROM query_vectors qv
            CROSS JOIN LATERAL (
                SELECT Pid
                FROM products
                ORDER BY {self.embedding_type} <=> qv.query_vector
                LIMIT %s
            ) p
            ORDER BY qv.query_vector, score DESC
        """
        
        cur.execute(sql, [query_vector, k])
        results = cur.fetchall()
        
        cur.close()
        conn.close()
        
        # Group results by query
        query_results = []
        current_query = None
        current_scores = {}
        
        for query_vector, pid, score in results:
            if current_query is not None and query_vector != current_query:
                query_results.append(current_scores)
                current_scores = {}
            current_query = query_vector
            current_scores[pid] = score
        
        if current_scores:
            query_results.append(current_scores)
            
        return query_results

class FaissVectorRetrieval(SimilarityRetrieval):
    """FAISS vector search"""
    def __init__(self, index: faiss.Index, pids: List[int], embedding_type: str):
        self.index = index
        self.pids = pids  # Array of PIDs corresponding to FAISS indices
        self.embedding_type = embedding_type

    def score(self, query: Union[str, np.ndarray], k: int = 10) -> Dict[int, float]:
        # Stack embeddings for batch search
        query_vector = [float(x) for x in query]
        
        # Run FAISS batch search
        distances, indices = self.index.search(query_vector, k)
        
        # Convert results to list of score dicts
        results = []
        for i in range(len(query_vector)):
            pids = [self.pids[idx] for idx in indices[i]]
            scores = {pid: 1 - (dist / 2) for pid, dist in zip(pids, distances[i])}
            results.append(scores)
        
        return results

class TextSearchRetrieval(SimilarityRetrieval):
    """Text search using PostgreSQL full-text search"""
    def __init__(self, method: str, db_config: Dict[str, str]):
        self.method = method  # e.g., 'ts_rank', 'ts_rank_cd'
        self.db_config = db_config

    def score(self, query: str, k: int = 10) -> Dict[int, float]:
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        # Build the query using LATERAL join for batch processing
        sql = f"""
            WITH query_texts AS (
                SELECT unnest(%s::text[]) as query_text
            )
            SELECT 
                qt.query_text,
                p.Pid,
                {self.method}(p.document, plainto_tsquery('english', qt.query_text)) AS score
            FROM query_texts qt
            CROSS JOIN LATERAL (
                SELECT Pid, document
                FROM products
                WHERE document @@ plainto_tsquery('english', qt.query_text)
                ORDER BY {self.method}(document, plainto_tsquery('english', qt.query_text)) DESC
                LIMIT %s
            ) p
            ORDER BY qt.query_text, score DESC
        """
        
        cur.execute(sql, [query, k])
        results = cur.fetchall()
        
        cur.close()
        conn.close()
        
        # Group results by query
        query_results = []
        current_query = None
        current_scores = {}
        
        for query_text, pid, score in results:
            if current_query is not None and query_text != current_query:
                query_results.append(current_scores)
                current_scores = {}
            current_query = query_text
            current_scores[pid] = score
        
        if current_scores:
            query_results.append(current_scores)
            
        return query_results

def hybrid_retrieval(
    queries: List[str],
    query_embeddings: List[np.ndarray],
    components: List[SimilarityRetrieval],
    weights: List[float],
    top_k: int = 10
) -> List[Tuple[List[int], List[float]]]:
    """
    Find top-k most relevant products for multiple queries using hybrid search.
    
    Args:
        queries: List of text queries for text search components
        query_embeddings: List of vector queries for vector search components
        components: List of SimilarityRetrieval instances
        weights: List of weights for each component (must sum to 1)
        top_k: Number of results to return per query
    
    Returns:
        List of (pids, scores) tuples, one per query
    """
    if len(components) != len(weights):
        raise ValueError("Number of components must match number of weights")
    if len(queries) != len(query_embeddings):
        raise ValueError("Number of queries must match number of query embeddings")
    
    # Get scores from each component
    all_scores = []
    for comp in components:
        if isinstance(comp, (PostgresVectorRetrieval, FaissVectorRetrieval)):
            scores = comp.score(query_embeddings, k=top_k)
        else:
            scores = comp.score(queries, k=top_k)
        all_scores.append(scores)

    # Combine scores for each query
    results = []
    for i in range(len(queries)):
        combined_scores = {}
        for scores in all_scores:
            for pid, score in scores[i].items():
                combined_scores[pid] = combined_scores.get(pid, 0) + score

        # Sort and get top_k
        top = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        pids, scores = zip(*top) if top else ([], [])
        results.append((list(pids), list(scores)))

    return results

# Example usage:
def example_hybrid_retrieval(queries: List[str], query_embeddings: List[np.ndarray], top_k: int = 10):
    # Load FAISS index (if using FAISS)
    # index = faiss.read_index("path/to/your/index.faiss")
    # pids = [...]  # List of PIDs corresponding to FAISS indices
    
    # Create components
    components = [
        PostgresVectorRetrieval('text_embedding', DB_CONFIG),
        # FaissVectorRetrieval(index, pids, 'image_embedding'),  # Uncomment if using FAISS
        PostgresVectorRetrieval('image_embedding', DB_CONFIG),
        TextSearchRetrieval('ts_rank_cd', DB_CONFIG)
    ]
    
    weights = [0.5, 0.3, 0.2]  # Must sum to 1
    
    # Run hybrid search
    results = hybrid_retrieval(
        queries=queries,
        query_embeddings=query_embeddings,
        components=components,
        weights=weights,
        top_k=top_k
    )
    
    return results
