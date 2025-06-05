import torch
import pandas as pd
import sys
import os
import json
import mlflow
import mlflow.pytorch
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# Add src directory to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src" / "backend"))
sys.path.insert(0, str(root_dir / "preprocess"))

from embedding import generate_embedding
from retrieval import hybrid_retrieval, PostgresVectorRetrieval, TextSearchRetrieval, FaissVectorRetrieval, reorder_search_results_by_relevancy
from config.db import DB_CONFIG, TABLE_NAME
from config.path import MLFLOW_TRACKING_URI, BENCHMARK_QUERY_CSV
from api import format_results

def load_configs(config_path):
    """Load experiment configurations from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

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

def run_experiment(config, db_config=None):
    """Run a single experiment and log results to MLflow."""
    print(f"\nRunning experiment: {config['name']}")
    
    with mlflow.start_run(run_name=f"{config['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Create components from config
        components = [create_retrieval_component(c, db_config) for c in config['components']]
        
        # Log parameters
        component_weights = {}
        component_params = {}
        for i, (c, w) in enumerate(zip(components, config['weights'])):
            if isinstance(c, PostgresVectorRetrieval):
                component_name = f"postgres_vector_{c.column_name}"
                component_params[f"{i}_type"] = "PostgresVectorRetrieval"
                component_params[f"{i}_column_name"] = c.column_name
            elif isinstance(c, FaissVectorRetrieval):
                component_name = f"faiss_vector_{c.column_name}"
                component_params[f"{i}_type"] = "FaissVectorRetrieval"
                component_params[f"{i}_column_name"] = c.column_name
                component_params[f"{i}_nprobe"] = c.index.nprobe if hasattr(c.index, 'nprobe') else 1
            else:
                component_name = type(c).__name__.lower()
                component_params[f"{i}_type"] = "TextSearchRetrieval"
                component_params[f"{i}_rank_method"] = c.method
            component_weights[component_name] = w
        
        # Get benchmark file name without extension
        benchmark_file = Path(BENCHMARK_QUERY_CSV).stem
        
        mlflow.log_params({
            "benchmark": benchmark_file,
            "table": TABLE_NAME,
            "top_k": TOP_K,
            "faiss_nlist": next((c.index.nlist for c in components if isinstance(c, FaissVectorRetrieval)), None),
            **component_weights,
            **component_params
        })
        
        # Initialize counters
        query_types = ['basic_query', 'attribute_query', 'natural_query']
        results = {
            'overall': {'hits': 0, 'total': 0},
            'basic_query': {'hits': 0, 'total': 0},
            'attribute_query': {'hits': 0, 'total': 0},
            'natural_query': {'hits': 0, 'total': 0}
        }
        
        category_results = defaultdict(lambda: {
            'basic_query': {'hits': 0, 'total': 0},
            'attribute_query': {'hits': 0, 'total': 0},
            'natural_query': {'hits': 0, 'total': 0}
        })

        # Create a single progress bar for all queries
        total_queries = len(df) * len(query_types)
        with tqdm(total=total_queries, desc="Processing queries", ncols=80) as pbar:
            for query_type in query_types:
                for _, row in df.iterrows():
                    query = row[query_type]
                    target_pid = row['Pid']
                    category = row['Category']
                    
                    # Generate embedding and run hybrid search
                    query_embedding = generate_embedding(query_text=query)
                    pids, scores = hybrid_retrieval(
                        query=query,
                        query_embedding=query_embedding,
                        components=components,
                        weights=config['weights'],
                        top_k=40
                    )
                    
                    # Format results using the existing function
                    formatted_results = format_results(pids, scores)
                    
                    # Reorder results if we have text queries and API keys
                    if query and (os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")):
                        reordered_results, _ = reorder_search_results_by_relevancy(
                            query=query,
                            search_results=formatted_results
                        )
                        # Extract PIDs from reordered results
                        pids = [result['Pid'] for result in reordered_results]
                    
                    # Check if the ground truth Pid is in the results
                    hit = target_pid in pids
                    
                    # Update counters
                    results[query_type]['total'] += 1
                    results['overall']['total'] += 1
                    category_results[category][query_type]['total'] += 1
                    
                    if hit:
                        results[query_type]['hits'] += 1
                        results['overall']['hits'] += 1
                        category_results[category][query_type]['hits'] += 1
                    
                    pbar.update(1)
        
        # Calculate and log metrics
        for category in results:
            if results[category]['total'] > 0:
                recall = results[category]['hits'] / results[category]['total']
                mlflow.log_metric(f"{category}_recall_at_k", round(recall, 2))
                
                # Print overall recall for this configuration
                if category == 'overall':
                    print(f"Overall Recall@{TOP_K}: {recall:.2%}\n")

        for category in category_results:
            for query_type in query_types:
                if category_results[category][query_type]['total'] > 0:
                    recall = category_results[category][query_type]['hits'] / category_results[category][query_type]['total']
                    mlflow.log_metric(f"{category}_{query_type}_recall_at_k", round(recall, 2))

def main():
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Initialize MLflow
    mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    # Load configurations
    config_path = Path(__file__).parent / "experiment_configs.json"
    configs = load_configs(config_path)
    print(f"Loaded {len(configs)} experiment configurations")

    # Global settings
    global TOP_K, df
    TOP_K = 20

    # Load benchmark data
    print(f"Loading benchmark dataset from: {BENCHMARK_QUERY_CSV}")
    df = pd.read_csv(BENCHMARK_QUERY_CSV)
    print(f"Loaded benchmark dataset with {len(df)} queries")

    # Run all experiments
    for experiment_name, experiment_configs in configs.items():
        print(f"\n{'='*50}")
        print(f"Starting experiment group: {experiment_name}")
        print(f"{'='*50}")
        mlflow.set_experiment(experiment_name)
        
        for config in experiment_configs:
            run_experiment(config, db_config=DB_CONFIG)

if __name__ == "__main__":
    main() 