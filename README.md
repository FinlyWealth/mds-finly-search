# FinlyWealth Product Search Engine

A scalable, multimodal product search engine developed for [FinlyWealth](https://finlywealth.com/), an affiliate marketing platform expanding into e-commerce.

## Setup Instructions - User

Follow the instructions below to set up the database, then run the command to start the frontend and backend applications in Docker containers:

```{bash}
docker-compose up --build
```

The frontend Streamlit application is at `http://0.0.0.0:8501`.

## Setup Instructions - Developer

### Step 1. Download product data and product images

1. Download the `sample_100k_v2.csv` and `images_100k_v2.zip` from: <https://drive.google.com/drive/folders/1LQzeuo9PZ_Y-Xj_QhhzYEYJP8XFZn48K>

2. Extract `images_100k_v2.zip` into the `data/images` folder. Put `sample_100k_v2.csv` under `data/csv`.

### Step 2. Setup Python environment and environment variables

1. Set up Python environment using `environment.yaml`: `conda env create --f environment.yaml`
2. Create environment variable `.env` file in the root folder

### Step 3. Database Setup

Choose Option A or Option B based on your use case.

**Option A: Google Cloud**

This setup uses the Google Cloud SQL proxy. It connects to the cloud database via a localhost connection. Requires Google authentication.

1. Copy and paste the following in to the `.env` file.

    ```         
    # User, password and location of the Postgres database
    PGUSER=postgres
    PGPASSWORD=u&P{l{hq1r`^.u76
    PGHOST=localhost
    PGPORT=5433
    PGDATABASE=postgres

    # Location of the embeddings and metadata that'll be imported in to the database
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

2. Ensure you've been granted access to Google Cloud from the repo admin and then install Google Cloud CLI for your platform: https://cloud.google.com/sdk/docs/install-sdk

3. To setup the proxy connection:

    ```{bash}
    # If running for the first time, this will setup and run the proxy
    make proxy-setup

    # Use the following to start the proxy after a reboot
    make proxy
    ```

4. To start the app and the server:

    ```{bash}
    # Starts streamlit frontend and API backend
    make run
    ```

**Option B: Local Postgres**

This setup is for a running the app with a local Postgres database. You would use this setup if you wish to develop with different embeddings.

1. Unless you intend to genereate your own custom embeddings via `make embed`, it is recommended to download the pre-generated embeddings `text_clip.npz`, `image_clip.npz` and `minilm.npz` from the same Google Drive. Put all 3 files under `data/embeddings`.

2. Add the following to environment variables. Change the Postgres credentials as needed to the local db.

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

3. To setup the database:

    ```{bash}
    # If running for the first time, this will setup the sql table, add pgvector and load the embedding files in to the db
    make db-setup

    # Once the db is setup and you want to use other types of embeddings, use the following to load the db
    # CAUTION: This will drop the existing table and create a new one
    make db-load
    ```

4. Optional: If using FAISS indexes, run the following to build the indexes after the embeddings have been imported.

    ```{bash}
    # Builds FAISS index for each embedding column
    make faiss
    ```

5. To start the app and the server:

    ```{bash}
    # Starts streamlit frontend and API backend
    make run
    ```


## Setup Troubleshooting

**To test the api through command line**

```{bash}
# test text search
curl -X POST http://127.0.0.1:5001/api/search/text -H "Content-Type: application/json" -d '{"query": "red pant"}'

# test image search
# download any product image and stored as test-img.jpeg
curl -X POST http://127.0.0.1:5001/api/search/image -H "Content-Type: application/json" -d '{"image_path": "{absolute-path-to-repo}/mds-finly-search/test-img.jpeg"}'
```

**Postgres not installed through homebrew**

In case your postgres is installed at `/Library/PostgreSQL/16`, not through home brew, try the following method to install pgvector.

In the finly conda environment, from any directory:

```{bash}
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

If you see any error like `make: arm64-apple-darwin20.0.0-clang: No such file or directory` when run the `make` command, try to run the following, and then run `make` again:

```{bash}
export PG_CONFIG=/Library/PostgreSQL/16/bin/pg_config
```

Now we need to copy the pgvector we installed in the finly conda enviornment into place where our Postgres database is installed.

```{bash}
# create the postgres extension folder
sudo mkdir -p /Library/PostgreSQL/16/share/extension

sudo cp /Users/{your_username}/miniforge3/envs/finly/share/extension/vector.control /Library/PostgreSQL/16/share/extension/

sudo cp /Users/{your_username}/miniforge3/envs/finly/lib/vector.dylib /Library/PostgreSQL/16/lib/postgresql/
```

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
        - `column_name`: Name of the vector column to search (e.g., "text_embedding" or "image_embedding")
2. **FaissVectorRetrieval**
    - Uses FAISS for efficient vector similarity search
    - Parameters:
        - `index_type`: Either "text" or "image" to specify which pre-computed index to use
3. **TextSearchRetrieval**
    - Uses PostgreSQL full-text search capabilities
    - Parameters:
        - `rank_method`: Ranking method to use (e.g., "ts_rank" which ranks purely on frequency or "ts_rank_cd" which also measure proximity of words)
