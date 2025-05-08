# mds-finly-search

### Setup Instructions for Experimental Pipeline and MLflow
The `experiment_pipeline.ipynb` notebook is a testing framework that evaluates different product search methods using the CLIP model and MLflow for experiment tracking. 

Retrieval can be done using a combination of:

- Vector based retrieval using [pgvector](https://github.com/pgvector/pgvector) (cosine distance)
- Vector based retrieval using [FAISS IVF](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) (approximate nearest neighbor)
- Text based retrieval using Postgres [ts_rank](https://www.postgresql.org/docs/current/textsearch-controls.html) (approximate TF-IDF)

**Setup**

1. Download the metadata, sample embeddings and images from: https://drive.google.com/drive/folders/1LQzeuo9PZ_Y-Xj_QhhzYEYJP8XFZn48K
2. Extract the zip file in the `data` folder
3. Set up Python environment using `environment.yaml`
4. Install [postgresql](https://www.postgresql.org) and [pgvector](https://github.com/pgvector/pgvector) extension
5. Create environment variable `.env` file in the root folder
6. Add the following to environment variables

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
6. Run `python src/util/load_db.py` to load the database with the embeddings and metadata
7. Run `python src/util/compute_faiss_index.py` to compute the the index used for FAISS search

