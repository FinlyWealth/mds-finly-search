# Experimentation Framework

This document outlines the framework for evaluating and comparing different search configurations to enhance natural language query performance. It provides a summary of key findings, instructions on how to reproduce these experiments, and guidance on interpreting the results.

## Key Results and Interpretation

Our primary goal was to improve search performance for natural language queries, where traditional keyword-based methods often fall short. We evaluated three main configurations, with the following results for natural language queries:

| Method                          | Precision@20 | Recall@20 | Search Time (s) |
| :------------------------------ | :----------- | :-------- | :-------------- |
| Text Search (Baseline)          | 7%           | 7%        | 0.3             |
| Text + Embeddings               | 70%          | 53%       | 0.6             |
| Text + Embeddings + LLM Reranking | 62%          | 58%       | 4.24            |

### Summary of Findings

-   **Embeddings are crucial for understanding natural language**: The most significant performance gain came from introducing embeddings, which boosted recall from a mere 7% to 53%. This confirms that semantic search is far more effective than keyword-based text search for complex, natural language queries.
-   **LLM Reranking enhances recall**: While embeddings are great at finding a broad set of semantically relevant candidates, the LLM-based reranker excels at identifying the most precise matches from this set. It further increased recall to 58%, demonstrating its ability to recover relevant items that even a pure vector search might miss.
-   **Trade-offs in precision and search time**: The LLM reranker introduced a slight dip in precision, which may be attributed to the inherent variability in judging relevance. It also increased search time due to the additional processing overhead. However, all configurations remained within our 5-second performance target.

For a detailed breakdown including results for basic keyword queries, please refer to the complete summary table below.

<details>
<summary>View Full Experiment Results</summary>

| Method                    | Query Type    | Precision@20 | Recall@20 | Search Time (s) |
| :------------------------ | :------------ | :----------- | :-------- | :-------------- |
| Text Search               | Basic Query   | 73           | 42        | 0.3             |
| Text + Embeddings         | Basic Query   | 81           | 33        | 0.6             |
| Text + Embeddings + LLM   | Basic Query   | 78           | 41        | 4.24            |
| Text Search               | Natural Query | 7            | 7         | 0.3             |
| Text + Embeddings         | Natural Query | 70           | 53        | 0.6             |
| Text + Embeddings + LLM   | Natural Query | 62           | 58        | 4.24            |

</details>

## Reproducing Our Experiments

We use **MLflow** to track, manage, and reproduce our experiments. This allows for a systematic comparison of different models and configurations. For those new to MLflow, you can learn more from the [official MLflow documentation](https://mlflow.org/docs/latest/index.html).

The core of the experimentation pipeline is `experiments/experiment_pipeline.py`, which is designed to run a suite of experiments defined in a configuration file.

### 1. Set Up MLflow Tracking

To run experiments and view results, you first need to set up an MLflow tracking server.

1.  **Start the local MLflow server:**
    ```bash
    mlflow server --host 0.0.0.0 --port 5000
    ```
2.  **Set the tracking URI:**
    Create or update your `.env` file with the following line to point the pipeline to your local server:
    ```bash
    MLFLOW_TRACKING_URI=http://localhost:5000
    ```
3.  **Access the MLflow UI:**
    You can now view runs and compare results by navigating to [http://localhost:5000](http://localhost:5000) in your browser.

*Note: The project is also configured to log to a remote, shared MLflow server. To switch back, simply update the `MLFLOW_TRACKING_URI` in your experiment configuration.*

### 2. Configure and Run Experiments

1.  **Define experiment parameters:**
    Edit `experiments/experiment_configs.json` to define the search configurations you want to test. You can combine different retrieval components and assign weights for hybrid search. See the "Supported Retrieval Components" section below for more details.

    ```json
    {
      "1M_faiss_fusion": [
        {
          "name": "tfidf_fusion_gpt3.5_turbo",
          "components": [
            {
              "type": "FaissVectorRetrieval",
              "params": { "column_name": "fusion_embedding", "nprobe": 32 }
            },
            {
              "type": "TextSearchRetrieval",
              "params": { "rank_method": "ts_rank_cd" }
            }
          ],
          "weights": [0.5, 0.5]
        }
      ]
    }
    ```

2.  **Execute the experiment pipeline:**
    Run the following command to execute all experiments defined in the configuration file:
    ```bash
    make experiments
    ```
    The results for each run will be logged to your configured MLflow tracking server.

3.  **View experiment results on remote server:** <http://35.209.59.178:8591>

### Supported Retrieval Components

The search engine supports the following retrieval components that can be combined in experiments:

1.  **`PostgresVectorRetrieval`**
    -   Uses `pgvector` for vector similarity search.
    -   **Parameters**:
        -   `column_name`: Name of the vector column to search (e.g., "fusion_embedding").
2.  **`FaissVectorRetrieval`**
    -   Uses FAISS for efficient vector similarity search.
    -   **Parameters**:
        -   `column_name`: Name of the vector column to search (e.g., "fusion_embedding").
        -   `nprobe`: Number of IVF clusters to search in.
3.  **`TextSearchRetrieval`**
    -   Uses PostgreSQL's full-text search.
    -   **Parameters**:
        -   `rank_method`: Ranking method (`ts_rank` or `ts_rank_cd`). `ts_rank_cd` considers word proximity and is generally recommended.