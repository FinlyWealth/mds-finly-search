# FinlyWealth Product Search Engine

A scalable, multimodal product search engine developed for [FinlyWealth](https://finlywealth.com/), an affiliate marketing platform expanding into e-commerce.

## Setup Instructions

**Download and Extract Data**

1. Download the `sample_100k_v2.csv` and `images_100k_v2.zip` from: https://drive.google.com/drive/folders/1LQzeuo9PZ_Y-Xj_QhhzYEYJP8XFZn48K
2. Unless you intend to genereate your own custom embeddings via `make embed`, it is recommended to download the pre-generated embeddings `embeddings_100k_v2.npz` from the same Google Drive 
3. Extract `images_100k_v2.zip` into the `data/images` folder. Put `sample_100k_v2.csv` under `data/csv`. Put `embeddings_100k_v2.npz` under `data/embeddings`.

**Setup Python Environment and Environment Variables**

4. Set up Python environment using `environment.yaml`: `conda env create --f environment.yaml`
5. Create environment variable `.env` file in the root folder
6. Add the following to environment variables. Change the Postgres credentials as needed. 

    ```
    # User, password and location of the Postgres database
    PGUSER=finly-admin
    PGPASSWORD=123
    PGHOST=localhost
    PGPORT=5432
    PGDATABASE=finly

    # Location of the embeddings and metadata that'll be imported in to the database
    EMBEDDINGS_PATH=data/embeddings
    METADATA_PATH=data/csv/sample_100k_v2.csv

    # Type of embeddings to generate
    ENABLE_TEXT_CLIP=true
    ENABLE_IMAGE_CLIP=true
    ENABLE_MINILM=true

    # Model configurations
    TEXT_CLIP_MODEL=openai/clip-vit-base-patch32
    IMAGE_CLIP_MODEL=openai/clip-vit-base-patch32
    MINILM_MODEL=sentence-transformers/all-MiniLM-L6-v2


    # Location to save the index
    FAISS_INDEX_DIR=data/faiss_indexes

    # Number of clusters to use
    FAISS_NLIST=100

    # URL of mlflow server
    MLFLOW_TRACKING_URI=http://35.209.59.178:8591
    ```

**Install Postgres and pgvector**

7. Setup postgres database locally. The credentials need to match the `.env` file. 

    ```{bash}
    # Install psql command line tool
    conda install -c conda-forge postgresql

    # Login to postgres, username and password will be the one you set when install the postgres
    psql -U postgres

    # Create new finly database user
    CREATE USER "finly-admin" WITH PASSWORD '123';

    # Give the user permission to create databases
    ALTER USER "finly-admin" CREATEDB;

    # Create the database finly
    CREATE DATABASE finly OWNER "finly-admin";

    # Grant all privileges on the database to finly-admin user
    GRANT ALL PRIVILEGES ON DATABASE finly TO "finly-admin";
    ```

8. Add pgvector extenstion

    ```{bash}
    # Login to finly database
    psql -U postgres -d finly

    # Create the extension
    CREATE EXTENSION IF NOT EXISTS vector;
    ```

9. Run `make db` and then `make faiss` from the root folder. Run `make preprocess all` if you want to run all 3 preprocessing scripts including embedding generation.

**Start frontend application**

```{bash}
streamlit run src/frontend/ap.py
```

**Start backend api**

```{bash}
# from the root directory, the api will be running at http://127.0.0.1:5001
python src/backend/api.py
```

To test the api through command line:

```{bash}
# test text search
curl -X POST http://127.0.0.1:5001/api/search/text -H "Content-Type: application/json" -d '{"query": "red pant"}'

# test image search
# download any product image and stored as test-img.jpeg
curl -X POST http://127.0.0.1:5001/api/search/image -H "Content-Type: application/json" -d '{"image_path": "{absolute-path-to-repo}/mds-finly-search/test-img.jpeg"}'
```

## Running Experiments

`experiments/experiment_pipeline.py` is designed to run multiple experiments to evaluate the performance of different retrieval components. These components can be combined with different weights in the experiment configuration to perform hybrid search.

1. Edit `experiments/experiment_configs.json` to setup the experiment configurations. See next section on supported retrieval components that can be specified in the config. 
   ```json
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
   ```bash
   make experiments
   ```
   This will execute each experiment defined in experiment_configs.json and log results to MLflow

3. View experiment results: http://35.209.59.178:8591

### Supported Retrieval Components

The search engine supports the following retrieval components that can be combined in experiments:

1. **PostgresVectorRetrieval**
   - Uses pgvector for vector similarity search
   - Parameters:
     - `column_name`: Name of the vector column to search (e.g., "text_embedding" or "image_embedding")

2. **FaissVectorRetrieval**
   - Uses FAISS for efficient vector similarity search
   - Parameters:
     - `index_type`: Either "text" or "image" to specify which pre-computed index to use

3. **TextSearchRetrieval**
   - Uses PostgreSQL full-text search capabilities
   - Parameters:
     - `rank_method`: Ranking method to use (e.g., "ts_rank" which ranks purely on frequency or "ts_rank_cd" which also measure proximity of words)

## Available Makefile Commands

- `make all`: Runs all preprocessing steps and generates the report

### Data Proprocessing

- `make preprocess-all`: Runs all preprocessing steps (generate embeddings, load database, compute FAISS index)

- `make embed`: Generates embeddings for the data

- `make db`: Loads data into the PostgreSQL database

- `make faiss`: Computes the FAISS index for vector search

### MLflow Experiments

- `make experiments`: Run all experiments and log results to MLflow

### Report Rendering

- `make report`: Generates the Quarto report

- `make clean`: Removes generated report files
