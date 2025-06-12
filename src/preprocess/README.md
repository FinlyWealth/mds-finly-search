# Preprocessing Instructions

This directory contains scripts for preprocessing data and generating embeddings/indices for the Finly Search application.

## Setup and Configuration

Before running the scripts, make sure you've setup the database either a local database (Postgres) or a cloud database (Google Cloud) following the README instructions in the root folder. 

## Running the Complete Preprocessing Pipeline

You can run the entire preprocessing script using:
```bash
make train
```

## Running the Individual Pipeline Components

**Process and clean raw CSV file**
```bash
make csv
```
This runs `clean_data.py` to process the raw data and save a clean csv file. 

**Generate embeddings**
```bash
make embed
```
This runs `generate_embed.py` to generate embeddings and save them as .npz files. 

**Load data into the database**
```bash
make db-load
```
This runs `load_db.py` to load the embeddings and product metadata in to the database. It also setup the TF-IDF indices in the database. 

**Compute FAISS index**
```bash
make faiss
```
This runs `compute_faiss_index.py` to create the FAISS indices. If an indices exist, the script will add to the indices instead of creating a new one. 