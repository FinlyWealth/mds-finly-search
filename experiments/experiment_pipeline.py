import torch
import pandas as pd
import sys
import os
import json
import mlflow
import mlflow.pytorch
import plotly.express as px
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# Add src directory to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src" / "backend"))
sys.path.insert(0, str(root_dir / "preprocess"))

from embedding import initialize_clip_model, generate_embedding
from retrieval import hybrid_retrieval, PostgresVectorRetrieval, TextSearchRetrieval, FaissVectorRetrieval
from concatenate import generate_fusion_embeddings
from config.db import DB_CONFIG
from config.path import MLFLOW_TRACKING_URI

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
        return FaissVectorRetrieval(params['column_name'])
    else:
        raise ValueError(f"Unknown component type: {component_type}")

def create_recall_plot(results, experiment_name):
    """Create a bar plot of recall metrics by query type."""
    # Prepare data for plotting
    plot_data = []
    for query_type in ['basic_query', 'attribute_query', 'natural_query']:
        if results[query_type]['total'] > 0:
            recall = results[query_type]['hits'] / results[query_type]['total']
            plot_data.append({
                'Query Type': query_type.replace('_', ' ').title(),
                'Recall': recall
            })
    
    # Create the bar plot
    fig = px.bar(
        pd.DataFrame(plot_data),
        x='Query Type',
        y='Recall',
        title=f'Recall@{TOP_K} by Query Type - {experiment_name}',
        labels={'Recall': f'Recall@{TOP_K}'},
        color='Query Type',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Update layout
    fig.update_layout(
        yaxis_tickformat='.1%',
        yaxis_range=[0, 1],
        showlegend=False,
        template='plotly_white'
    )
    
    # Save the plot
    plot_path = Path(__file__).parent / 'plots' / f'{experiment_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
    plot_path.parent.mkdir(exist_ok=True)
    fig.write_html(str(plot_path))
    
    # Log the plot to MLflow
    mlflow.log_artifact(str(plot_path))

def run_experiment(config, dataset_name=None, model_name=None, db_config=None):
    """Run a single experiment and log results to MLflow."""
    print(f"\nRunning experiment: {config['name']}")
    
    with mlflow.start_run(run_name=f"{config['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Create components from config
        components = [create_retrieval_component(c, db_config) for c in config['components']]
        
        # Log parameters
        component_weights = {}
        for c, w in zip(components, config['weights']):
            if isinstance(c, PostgresVectorRetrieval):
                component_name = f"postgres_vector_{c.column_name}"
            elif isinstance(c, FaissVectorRetrieval):
                component_name = f"faiss_vector_{c.column_name}"
            else:
                component_name = type(c).__name__.lower()
            component_weights[component_name] = w
        
        mlflow.log_params({
            "dataset": dataset_name,
            "model": model_name,
            "clip_model": clip_model,
            "top_k": TOP_K,
            **component_weights
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
                    #query_embedding = generate_embedding(query_text=query)
                    query_embedding = generate_fusion_embeddings(query)
                    pids, _ = hybrid_retrieval(
                        query=query,
                        query_embedding=query_embedding,
                        components=components,
                        weights=config['weights'],
                        top_k=TOP_K
                    )
                    
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
        
        # Create and save the recall plot
        # create_recall_plot(results, config['name'])

def main():
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Initialize CLIP model
    global clip_model
    clip_model = "openai/clip-vit-base-patch32"
    print(f"Initializing CLIP model: {clip_model}")
    initialize_clip_model(clip_model)

    # Initialize MLflow
    mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    # Load configurations
    config_path = Path(__file__).parent / "experiment_configs.json"
    configs = load_configs(config_path)
    print(f"Loaded {len(configs)} experiment configurations")

    # Global settings
    global TOP_K, df, dataset_name, model_name
    TOP_K = 5
    dataset_name = "benchmark_query"
    model_name = clip_model

    # Load benchmark data
    df = pd.read_csv(Path(__file__).parent.parent / "data" / "csv" / "benchmark_query.csv")
    print(f"Loaded benchmark dataset with {len(df)} queries")

    # Run all experiments
    for experiment_name, experiment_configs in configs.items():
        print(f"\n{'='*50}")
        print(f"Starting experiment group: {experiment_name}")
        print(f"{'='*50}")
        mlflow.set_experiment(experiment_name)
        
        for config in experiment_configs:
            run_experiment(config, dataset_name=dataset_name, model_name=model_name, db_config=DB_CONFIG)

if __name__ == "__main__":
    main() 