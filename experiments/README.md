## Running Experiments

`experiments/experiment_pipeline.py` is designed to run multiple experiments to evaluate the performance of different retrieval components. These components can be combined with different weights in the experiment configuration to perform hybrid search.

1. Edit `experiments/experiment_configs.json` to setup the experiment configurations. See next section on supported retrieval components that can be specified in the config.

    ``` json
     {
         "experiment_group_name": [
             {
                 "name": "experiment_name",
                 "components": [
                     {
                         "type": "PostgresVectorRetrieval",
                         "params": {
                             "column_name": "text_embedding"
                         }
                     },
                     {
                         "type": "PostgresVectorRetrieval",
                         "params": {
                             "column_name": "image_embedding"
                         }
                     },
                     {
                         "type": "TextSearchRetrieval",
                         "params": {
                             "rank_method": "ts_rank_cd"
                         }
                     }
                 ],
                 "weights": [0.4, 0.3, 0.3]
             }
         ]
     } 
    ```

2. Run experiments using the make command:

    ``` bash
    make experiments
    ```

    This will execute each experiment defined in experiment_configs.json and log results to MLflow

3. View experiment results: <http://35.209.59.178:8591>

### Supported Retrieval Components

The search engine supports the following retrieval components that can be combined in experiments:

1. **PostgresVectorRetrieval**
    - Uses pgvector for vector similarity search
    - Parameters:
        - `column_name`: Name of the vector column to search (e.g., "fusion_embedding")
2. **FaissVectorRetrieval**
    - Uses FAISS for efficient vector similarity search
    - Parameters:
        - `column_name`: Name of the vector column to search (e.g., "fusion_embedding")
        - `nprobe`: Number of clusters to search in
3. **TextSearchRetrieval**
    - Uses PostgreSQL full-text search capabilities
    - Parameters:
        - `rank_method`: Ranking method to use (e.g., "ts_rank" which ranks purely on frequency or "ts_rank_cd" which also measure proximity of words)