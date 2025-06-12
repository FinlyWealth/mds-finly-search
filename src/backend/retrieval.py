import os
import logging
import psycopg2
from psycopg2.extras import execute_values, Json
import numpy as np
from typing import Dict, Union, Optional, List
import faiss
import json
import sys
import requests
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config.db import DB_CONFIG, TABLE_NAME

# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FAISS index configuration
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAISS_INDEX_DIR = os.path.join(REPO_ROOT, "data", "faiss_indices")
GCS_BUCKET_URL = os.getenv("GCS_BUCKET_URL", "https://storage.googleapis.com/mds-finly")
GCS_INDEX_PREFIX = "faiss_indices"

# Cache for FAISS indices and mappings
_faiss_index_cache = {}
_faiss_mapping_cache = {}


# Pydantic
class ReorderOutput(BaseModel):
    reordered_indices: List[int] = Field(
        ..., description="Indices in new relevance order"
    )
    reasoning: str = Field(..., description="Explanation of reordering")


def download_from_gcs(source_path: str, destination_file_name: str):
    """Downloads a file from public GCS bucket."""
    url = f"{GCS_BUCKET_URL}/{source_path}"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)

    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes

    with open(destination_file_name, "wb") as f:
        f.write(response.content)
    print(f"Downloaded {source_path} to {destination_file_name}")


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

    def __init__(
        self, column_name: str, nprobe: int = 1, db_config: Dict[str, str] = None
    ):
        """
        Initialize FAISS retrieval with saved index.

        Args:
            column_name: Name of the embedding column (e.g., 'text_embedding', 'image_embedding', 'fusion_embedding')
            nprobe: Number of clusters to visit during search (default: 1)
            db_config: Database configuration dictionary
        """
        # Extract the base type from the column name (e.g., 'fusion' from 'fusion_embedding')
        index_type = column_name.replace('_embedding', '')
            
        # Load the index from cache or file
        index_path = os.path.join(FAISS_INDEX_DIR, f"{index_type}_index.faiss")
        if index_type not in _faiss_index_cache:
            if not os.path.exists(index_path):
                print(
                    f"FAISS index not found locally at {index_path}, downloading from GCS..."
                )
                try:
                    download_from_gcs(
                        f"{GCS_INDEX_PREFIX}/{index_type}_index.faiss", index_path
                    )
                except Exception as e:
                    raise FileNotFoundError(
                        f"Failed to download FAISS index from GCS: {str(e)}"
                    )
            _faiss_index_cache[index_type] = faiss.read_index(index_path)
        self.index = _faiss_index_cache[index_type]

        # Set nprobe parameter
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe

        # Load the PID mapping from cache or file
        mapping_path = os.path.join(FAISS_INDEX_DIR, f"{index_type}_index_mapping.json")
        if index_type not in _faiss_mapping_cache:
            if not os.path.exists(mapping_path):
                print(
                    f"Index mapping not found locally at {mapping_path}, downloading from GCS..."
                )
                try:
                    download_from_gcs(
                        f"{GCS_INDEX_PREFIX}/{index_type}_index_mapping.json",
                        mapping_path,
                    )
                except Exception as e:
                    raise FileNotFoundError(
                        f"Failed to download index mapping from GCS: {str(e)}"
                    )
            with open(mapping_path, "r") as f:
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
        pids = [
            self.idx_to_pid[str(idx)]
            for idx in top_indices
            if str(idx) in self.idx_to_pid
        ]

        if not pids:
            return {}

        # Query the database to get the actual embeddings for these PIDs
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        # Convert list of PIDs to a comma-separated string for SQL
        pid_list = ",".join([f"'{pid}'" for pid in pids])

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
                vector_str = embedding.strip("[]{}")
                db_embedding = np.array(
                    [float(x) for x in vector_str.split(",")], dtype=np.float32
                )
                db_embedding = db_embedding / np.linalg.norm(db_embedding)

                # Calculate cosine similarity
                similarity = float(np.dot(query, db_embedding))
                normalized_scores[pid] = similarity

        cur.close()
        conn.close()

        # Normalize scores to [0, 1] based on highest score
        max_score = max(normalized_scores.values()) if normalized_scores else 1.0
        if max_score > 0:
            normalized_scores = {
                pid: score / max_score for pid, score in normalized_scores.items()
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
    component_type = component_config["type"]
    params = component_config["params"]

    if component_type == "PostgresVectorRetrieval":
        return PostgresVectorRetrieval(params["column_name"], db_config)
    elif component_type == "TextSearchRetrieval":
        return TextSearchRetrieval(params["rank_method"], db_config)
    elif component_type == "FaissVectorRetrieval":
        return FaissVectorRetrieval(
            params["column_name"], params.get("nprobe", 1), db_config
        )
    else:
        raise ValueError(f"Unknown component type: {component_type}")


def hybrid_retrieval(
    query: str,
    query_embedding: np.ndarray,
    components: list[SimilarityRetrieval],
    weights: list[float],
    top_k: int = 10,
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
    active_components = [
        (comp, weight) for comp, weight in zip(components, weights) if weight > 0
    ]

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


def reorder_search_results_by_relevancy(
    query: str,
    search_results: List[Dict],
    api_key: Optional[str] = None,
    model: str = "gpt-3.5-turbo",  # or a gemini model
    provider: str = "openai",  # can be 'openai' or 'gemini'
    max_results: int = 30,
) -> tuple[List[Dict], str]:
    """
    Reorders search results based on relevancy to the query using an LLM via LangChain.
    Parameters:
    - query (str): The search query.
    - search_results (list): List of dicts with at least a 'Pid' and metadata fields.
    - model (str): Model name (e.g., "gpt-4", "gemini-pro").
    - provider (str): LLM provider, either "openai" or "gemini".
    - api_key (str): API key for the selected provider.
    - max_results (int): Maximum number of search results to consider.
    Returns:
    - Tuple containing:
        - List of results reordered by relevance
        - String containing the LLM's reasoning for the reordering
    """

    results_to_process = search_results[:max_results]
    keys_to_keep = {"Name", "Brand", "Category", "Color", "Gender", "Size", "Price"}
    results_summary = []
    pid_tracker = {}

    for i, result in enumerate(results_to_process):
        relevant_data = {k: v for k, v in result.items() if k in keys_to_keep}
        results_summary.append({"index": i, **relevant_data})
        pid_tracker[i] = result["Pid"]

    prompt = f"""You are a search relevancy expert. Given a search query and a list of search results, please reorder the results based on their relevancy to the query.
    Search Query: "{query}"
    Search Results:
    {json.dumps(results_summary, indent=2)}
    Please analyze each result's relevancy to the search query and return a JSON array containing the indices of the results in order from MOST relevant to LEAST relevant.
    Consider factors like:
    1. Semantic similarity to the query intent
    2. Direct keyword matches
    3. Brand Name mentions
    4. Price comparison i.e if the query contains "under 100" products below that price should be weighted more etc
    Return your response in this exact JSON format:
    {{
      "reordered_indices": [2, 0, 1, ...],
      "reasoning": "Brief explanation of the reordering logic"
    }}
    Only return the JSON, nothing else."""

    try:
        # Select the appropriate LLM provider
        if provider.lower() == "gemini":
            api_key = api_key or os.getenv("GOOGLE_API_KEY")
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=0,
                convert_system_message_to_human=True,
            )
        else:
            # Default to OpenAI
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            llm = ChatOpenAI(
                model_name=model,
                openai_api_key=api_key,
                temperature=0,
            )

        # First try to get structured output
        try:
            parser = PydanticOutputParser(pydantic_object=ReorderOutput)
            messages = [
                SystemMessage(content="You are a search relevancy expert that outputs JSON."),
                HumanMessage(content=prompt),
            ]
            prompt = ChatPromptTemplate.from_messages(messages)
            chain = prompt | llm | parser
            result: ReorderOutput = chain.invoke({})
            reordered_indices = result.reordered_indices
            reasoning = result.reasoning
        except Exception as e:
            # If structured parsing fails, get raw output
            logger.warning(f"Failed to parse structured output: {str(e)}")
            raw_response = llm.invoke(prompt)
            # Try to extract indices and reasoning from raw response
            try:
                # Look for JSON-like structure in the response
                import re
                json_match = re.search(r'\{.*\}', raw_response.content, re.DOTALL)
                if json_match:
                    raw_json = json.loads(json_match.group())
                    reordered_indices = raw_json.get('reordered_indices', list(range(len(results_to_process))))
                    reasoning = raw_json.get('reasoning', 'No reasoning provided')
                else:
                    # If no JSON found, use default ordering
                    reordered_indices = list(range(len(results_to_process)))
                    reasoning = raw_response.content
            except Exception as json_e:
                logger.warning(f"Failed to extract JSON from raw response: {str(json_e)}")
                reordered_indices = list(range(len(results_to_process)))
                reasoning = raw_response.content

        reordered_indices = set(reordered_indices)
        logger.info(f"LLM Reasoning: {reasoning}")

        expected_indices = set(range(len(results_to_process)))
        provided_indices = set(reordered_indices)

        if expected_indices != provided_indices:
            logger.warning("Provided indices by LLM less than expected.")
            logger.warning(f"Expected indices: {expected_indices}")
            logger.warning(f"Provided indices: {provided_indices}")

        reordered_results = []
        # Track which indices we've processed
        processed_indices = set()
        
        # First add results for the indices provided by the LLM
        for index in reordered_indices:
            pid = pid_tracker[index]
            reordered_results += [
                result for result in search_results if result["Pid"] == pid
            ]
            processed_indices.add(index)
        
        # Add any missing indices at the end
        for i in range(len(results_to_process)):
            if i not in processed_indices:
                pid = pid_tracker[i]
                reordered_results += [
                    result for result in search_results if result["Pid"] == pid
                ]

        if len(search_results) > max_results:
            reordered_results.extend(search_results[max_results:])

        logger.info(f"Successfully reordered {len(results_to_process)} search results")
        return reordered_results, reasoning

    except Exception as e:
        logger.error(f"Failed to process search results: {str(e)}")
        return search_results, f"Error during reordering: {str(e)}"
