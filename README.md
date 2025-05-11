# FinlyWealth Product Search Engine

A scalable, multimodal product search engine developed for [FinlyWealth](https://finlywealth.com/), an affiliate marketing platform expanding into e-commerce.

## Makefile

The project includes a Makefile to simplify tasks. Before running the commands, please see setup instructions below:

### Setup Instructions

**Download and Extract Data**

1. Download the `sample_100k_v2.csv` and `images_100k_v2.zip` from: https://drive.google.com/drive/folders/1LQzeuo9PZ_Y-Xj_QhhzYEYJP8XFZn48K
2. Unless you intend to genereate your own custom embeddings via `make embed`, it is recommended to download the pre-generated embeddings `embeddings_100k_v2.npz` from the same Google Drive 
3. Extract the zip file and rename it into the `data/images` folder. Put sample_100k_v2.csv under `data/csv`. Put embeddings_100k_v2.npz under `data/embeddings`

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
    EMBEDDINGS_PATH=data/embeddings/embeddings_100k_v2.npz
    METADATA_PATH=data/csv/sample_100k_v2.csv

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

### Available Makefile Commands

- `make all`: Runs all preprocessing steps and generates the report

#### Data Proprocessing

- `make preprocess-all`: Runs all preprocessing steps (generate embeddings, load database, compute FAISS index)

- `make embed`: Generates embeddings for the data

- `make db`: Loads data into the PostgreSQL database

- `make faiss`: Computes the FAISS index for vector search

#### Report Rendering

- `make report`: Generates the Quarto report

- `make clean`: Removes generated report files
