# Preprocessing Instructions

This directory contains scripts for preprocessing data and generating embeddings for the Finly Search application.

## Running Preprocessing Scripts

You can run the preprocessing scripts using the following Makefile commands:

1. Generate embeddings:
```bash
make embed
```
This runs `generate_embed.py` to create embeddings for your documents.

2. Load data into the database:
```bash
make db-setup  # First time setup of the database
make db-load   # Load the processed data into the database
```
This sequence will:
- Set up the local database (first time only)
- Load the processed data into the database using `load_db.py`

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

Note: Make sure you have all the required dependencies installed and your environment properly configured before running these scripts.
