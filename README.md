# FinlyWealth Product Search Engine

A scalable, multimodal product search engine developed for [FinlyWealth](https://finlywealth.com/), an affiliate marketing platform expanding into e-commerce.

Use **Setup Instructions - Docker Containers** to run the application.

Use **Setup Instructions - Makefile** to run preprocessing scripts or experiments. 

## Setup Instructions - Docker Containers

**Prerequisites**
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (if using docker-compose.yaml)
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install-sdk) (for cloud database access)
- Git (optional, for cloning the repo)

### Google Cloud Setup
1. Install Google Cloud CLI for your platform from [here](https://cloud.google.com/sdk/docs/install-sdk)
2. Ensure you've been granted access to Google Cloud from the repo admin
3. Run the following commands to set up and start the proxy:

```bash
# If running for the first time, this will also setup the proxy
make proxy
```

### Step 1. Clone the Repository
In a separate terminal, clone the repository.

```bash
git clone FinlyWealth/mds-finly-search
cd mds-finly-search
```

### Step 2. Configure Environment Variables
Set up the required environment variables for database connection and API access by creating a `.env` file in the root folder with the following configurations.

```bash
# Database configuration
PGUSER=postgres
PGPASSWORD=ZK3RjyBv6twoA9
PGHOST=localhost
PGPORT=5433
PGDATABASE=postgres
PGTABLE=products_1M

# LLM API key
OPENAI_API_KEY=<insert-api-key>
```

### Step 3. Build Docker Containers
Start Docker and create the Docker images for both frontend and backend services.

```bash
docker compose build
```
This step may take several minutes as it downloads and builds all required dependencies.

### Step 4. Start the Application
Make sure the proxy is running and launch the application.

```bash
docker compose up
```

The application will start two services:
- Frontend: Access the search interface at http://localhost:8501
- Backend API: Running at http://localhost:5001

### Step 5. Clean Up
To close the container and free up the port once the proxy is not needed
```bash
docker compose down
make clean
``` 


## Setup Instructions - Makefile

### Step 1. Setup Python environment and environment variables

1. Set up Python environment using `environment.yaml`: `conda env create --f environment.yaml`
2. Create environment variable `.env` file in the root folder

### Step 2. Database Setup

Choose Option A or Option B based on your use case.

**Option A: Google Cloud**

This setup uses the Google Cloud SQL proxy. It connects to the cloud database via a localhost connection. Requires Google authentication.

1. Copy and paste the following in to the `.env` file.

    ```
    # User, password and location of the Postgres database
    PGUSER=postgres
    PGPASSWORD=ZK3RjyBv6twoA9
    PGHOST=localhost
    PGPORT=5433
    PGDATABASE=postgres
    PGTABLE=products_1M

    # LLM API key (do not put this variable in the env file if you plan to use LLM)
    OPENAI_API_KEY=<your-api-key>
    ```

2. Ensure you've been granted access to Google Cloud from the repo admin and then install Google Cloud CLI for your platform: <https://cloud.google.com/sdk/docs/install-sdk>

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

**Option B: Local Postgres (use a 100k sample data)**

This setup is for a running the app with a local Postgres database. You would use this setup if you wish to develop with different embeddings.

1. Unless you intend to genereate your own custom embeddings via `make embed`, it is recommended to download the pre-generated embeddings `fusion_embedding.npz` from the same Google Drive. If the file is large, there could be many chunks. Put all chunks under `data/embeddings`.

2. Add the following to environment variables. Change the Postgres credentials as needed to the local db.

    ```
    # User, password and location of the Postgres database
    PGUSER=finly-admin
    PGPASSWORD=123
    PGHOST=localhost
    PGPORT=5432
    PGDATABASE=finly
    PGTABLE=products_100k

    # Variables for preprocessing scripts
    # Location of the embeddings and metadata that'll be imported in to the database
    EMBEDDINGS_PATH=data/embeddings
    METADATA_PATH=data/csv/sample_100k_v2.csv
    # Number of clusters to use
    FAISS_NLIST=100

    # URL of mlflow server
    MLFLOW_TRACKING_URI=http://35.209.59.178:8591
    ```

3. Get embeddings:
    - It is recommended to download the pre-generated embeddings `fusion_embeddings_chunk_0.npz` and `image_clip_embeddings_chunk_0.npz` from the Google Drive: https://drive.google.com/drive/folders/1tRf1Ps0gcMdJOCWQ_7bEMpjPBPnpc8a1
    - If you want to generate your own embeddings:
        - Download the sample_100k_v2.csv and images_100k_v2.zip from the same Google Drive
        - Extract images_100k_v2.zip into the data/images folder. Put sample_100k_v2.csv under data/csv
        - Run `make embed`
    - Put `fusion_embeddings_chunk_0.npz` and `image_clip_embeddings_chunk_0.npz` under data/embeddings
    
4. To setup the database:

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

5. To run the overall app:

    ```{bash}
    # Starts streamlit frontend and API backend
    make run
    ```

    Use Ctrl+C to stop the app. Use `make clean` afterwards to release the assigned ports. Otherwise you may encounter a message about port conflict the next time you start the app.

### Setup Troubleshooting

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
