# FinlyWealth Product Search Engine

A scalable, multimodal product search engine developed for [FinlyWealth](https://finlywealth.com/), an affiliate marketing platform expanding into e-commerce.

## Makefile

The project includes a Makefile to simplify tasks. Before running the commands, please see setup instructions below:

### Setup Instructions

1. Download the `sample_100k_v2.csv` and `images_100k_v2.zip` from: https://drive.google.com/drive/folders/1LQzeuo9PZ_Y-Xj_QhhzYEYJP8XFZn48K.
2. Unless you intend to genereate your own custom embeddings via `make embed`, it is recommended to download the pre-generated embeddings `embeddings_100k_v2.npz` from the same Google Drive. 
3. Extract the zip file into the `data` folder
4. Set up Python environment using `environment.yaml`
5. Install [postgresql](https://www.postgresql.org) and [pgvector](https://github.com/pgvector/pgvector) extension
6. Create environment variable `.env` file in the root folder
7. Add the following to environment variables

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
    ```
8.  Run `make db` and then `make faiss` from the root folder. Run `make preprocess all` if you want to run all 3 preprocessing scripts including embedding generation.

### Available Makefile Commands
- `make all`: Runs all preprocessing steps and generates the report

**Data Proprocessing**

- `make preprocess-all`: Runs all preprocessing steps (generate embeddings, load database, compute FAISS index)

- `make embed`: Generates embeddings for the data

- `make db`: Loads data into the PostgreSQL database

- `make faiss`: Computes the FAISS index for vector search

**Report Rendering**

- `make report`: Generates the Quarto report

- `make clean`: Removes generated report files

To use these commands, simply run them from the project root directory. For example:
```bash
make preprocess-all  # Run all preprocessing steps
make report         # Generate the report
make clean          # Clean up generated files
```



