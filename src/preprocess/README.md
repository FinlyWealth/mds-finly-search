# Preprocessing Instructions

This directory contains scripts for preprocessing data and generating embeddings for the Finly Search application.

## Setup and Configuration

1. Add the following to environment variables. Change the Postgres credentials as needed for your local db:

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
    CLEAN_CSV_PATH=data/csv/sample_100k_v2.csv
    # Number of clusters to use
    FAISS_NLIST=4000
    ```

2. Get embeddings:
    - It is recommended to download the pre-generated embeddings `fusion_embeddings_chunk_0.npz` and `image_clip_embeddings_chunk_0.npz` from the Google Drive: https://drive.google.com/drive/folders/1tRf1Ps0gcMdJOCWQ_7bEMpjPBPnpc8a1
    - If you want to generate your own embeddings:
        - Download the sample_100k_v2.csv and images_100k_v2.zip from the same Google Drive
        - Extract images_100k_v2.zip into the data/images folder. Put sample_100k_v2.csv under data/csv
        - Run `make embed`
    - Put `fusion_embeddings_chunk_0.npz` and `image_clip_embeddings_chunk_0.npz` under data/embeddings

## Running Preprocessing Scripts

You can run the preprocessing scripts using the following Makefile commands:

1. Generate embeddings:
```bash
make embed
```
This runs `generate_embed.py` to create embeddings for your documents.

2. Setup and load data into the database:
```bash
# If running for the first time, this will setup the sql table, add pgvector and load the embedding files in to the db
make db-setup

# Once the db is setup and you want to use other types of embeddings, use the following to load the db
# CAUTION: This will drop the existing table and create a new one
make db-load
```

3. Compute FAISS index:
```bash
make faiss
```
This runs `compute_faiss_index.py` to create the FAISS index for efficient similarity search.

## Running the Complete Pipeline

To run all preprocessing steps in sequence, you can use:
```bash
make preprocess-all
```

## Running the Application

To run the overall app:
```bash
# Starts streamlit frontend and API backend
make run
```

Use Ctrl+C to stop the app. Use `make clean` afterwards to release the assigned ports. Otherwise you may encounter a message about port conflict the next time you start the app.

## Adding New Items

1. Generate embedding for new item
2. Upload embedding, product image and product metadata
3. Update FAISS index with new item

Note: Make sure you have all the required dependencies installed and your environment properly configured before running these scripts.
