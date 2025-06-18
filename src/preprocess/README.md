# Data Preprocessing Pipeline

This directory contains the preprocessing pipeline for the FinlyWealth Product Search Engine. The pipeline transforms raw product data into searchable embeddings and indices that power the multimodal search functionality.

![Process Product Data](../../img/preprocess_pipeline.png)

## Overview

The preprocessing pipeline consists of four main stages:

1. **Data Cleaning** - Process and validate raw CSV data
2. **Embedding Processing** - Create multimodal embeddings (text + images)
3. **Database Loading** - Store embeddings and metadata in the database
4. **Embedding Clustering** - Build similarity search indices using FAISS

## Prerequisites

Before running the preprocessing pipeline, ensure you have:

- ✅ **Database Setup**: Either [Google Cloud SQL](#database-setup) or [Local PostgreSQL](#database-setup)
- ✅ **Python Environment**: All dependencies installed
- ✅ **Raw Data**: CSV file with product data in the correct format
- ✅ **Product Images**: JPEG images named as `{Pid}.jpeg` (optional for multimodal search)

## Quick Start

Run the complete preprocessing pipeline with a single command:

```bash
make train
```

This executes all four stages in the correct order:
1. `make csv` - Clean the raw data
2. `make embed` - Generate embeddings
3. `make db-load` - Load into database
4. `make faiss` - Create search indices

## Configuration

### Environment Variables

Add these variables to your `.env` file:

```bash
# Data file paths
RAW_CSV_PATH=data/csv/raw/data.csv          # Path to your raw CSV file
CLEAN_CSV_PATH=data/csv/clean/data.csv      # Where cleaned data will be saved

# Embedding storage
EMBEDDINGS_PATH=data/embeddings             # Directory for embedding files

# FAISS index configuration
FAISS_INDEX_DIR=data/faiss_indices          # Directory for FAISS indices
FAISS_NLIST=100                             # Number of clusters for FAISS

# Database connection (if using local setup)
DATABASE_URL=postgresql://user:password@localhost:5432/finly_search
```

### Data Structure Requirements

Your CSV file must contain these **required columns**:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Pid` | string/int | Unique product ID | "12345" |
| `Name` | string | Product name | "Apple iPhone 14 Pro" |
| `Description` | string | Product description | "Latest smartphone with..." |
| `Category` | string | Product category | "Electronics" |
| `Price` | float | Product price | 999.99 |
| `PriceCurrency` | string | Currency code | "USD" |

**Optional columns** that enhance search quality:
- `Brand`, `Manufacturer`, `Color`, `Size`, `Condition`
- `FinalPrice`, `Discount`, `isOnSale`, `IsInStock`

### Directory Structure

Ensure your data follows this structure:

```
data/
├── csv/
│   ├── raw/
│   │   └── data.csv              # Your raw CSV file
│   └── clean/                    # Cleaned data (auto-generated)
├── images/                       # Product images (optional)
│   ├── 12345.jpeg               # Named as {Pid}.jpeg
│   └── 12346.jpeg
└── embeddings/                   # Generated embeddings (auto-generated)
```

## Pipeline Stages

### 1. Data Cleaning (`make csv`)

**Script**: `clean_data.py`

**What it does**:
- Validates required columns are present
- Filters by supported currencies (USD, CAD, GBP)
- Cleans and standardizes text fields
- Removes duplicate products
- Saves cleaned data to `CLEAN_CSV_PATH`

**Usage**:
```bash
make csv
# or directly:
python src/preprocess/clean_data.py
```

**Output**: Cleaned CSV file ready for embedding generation

### 2. Embedding Generation (`make embed`)

**Script**: `generate_embed.py`

**What it does**:
- Generates **text embeddings** using MiniLM for product names
- Generates **image embeddings** using CLIP for product images (if available)
- Creates **fusion embeddings** combining text and image features
- Saves embeddings as compressed `.npz` files

**Usage**:
```bash
make embed
# or directly:
python src/preprocess/generate_embed.py
```

**Output**: 
- `text_embeddings.npz` - Text-only embeddings
- `image_embeddings.npz` - Image embeddings (if images available)
- `fusion_embeddings.npz` - Combined text+image embeddings

### 3. Database Loading (`make db-load`)

**Script**: `load_db.py`

**What it does**:
- Loads product metadata and embeddings into PostgreSQL
- Creates TF-IDF indices for text search
- Sets up database tables and indexes
- Validates data integrity

**Usage**:
```bash
make db-load
# or directly:
python src/preprocess/load_db.py
```

**Output**: Database populated with products and search indices

### 4. FAISS Index Creation (`make faiss`)

**Script**: `compute_faiss_index.py`

**What it does**:
- Creates FAISS indices for fast similarity search
- Supports incremental updates (adds to existing indices)
- Optimizes for search speed and accuracy
- Creates separate indices for text, image, and fusion embeddings

**Usage**:
```bash
make faiss
# or directly:
python src/preprocess/compute_faiss_index.py
```

**Output**: FAISS index files in `FAISS_INDEX_DIR`

## Individual Commands

Run specific stages as needed:

```bash
# Clean data only
make csv

# Generate embeddings only
make embed

# Load into database only
make db-load

# Create FAISS indices only
make faiss

# Run complete pipeline
make train
```

## Monitoring and Logs

Each script provides detailed logging:

- **Progress indicators** for long-running operations
- **Error messages** with specific troubleshooting guidance
- **Validation checks** to ensure data quality
- **Performance metrics** for embedding generation

Check the console output for:
- ✅ Success confirmations
- ⚠️ Warnings about missing data
- ❌ Error messages with specific fixes

## Troubleshooting

### Common Issues

**"Missing required columns"**
- Ensure your CSV has all required columns with exact names
- Check for typos in column headers

**"Currency not supported"**
- Only USD, CAD, and GBP are supported
- Filter your data or modify `clean_data.py` to support other currencies

**"Database connection failed"**
- Verify your database is running
- Check connection credentials in `.env`
- Ensure database tables exist

**"Memory error during embedding generation"**
- The pipeline processes data in chunks of 500K rows
- For very large datasets, consider reducing batch size in `generate_embed.py`

### Data Quality Checks

Before running the pipeline, verify:

- [ ] CSV file is UTF-8 encoded
- [ ] All required columns are present
- [ ] Product IDs are unique
- [ ] Text fields contain meaningful content
- [ ] Images are in JPEG format and named correctly
- [ ] Currency codes are supported (USD, CAD, GBP)

## Advanced Configuration

### Customizing the Pipeline

You can modify individual scripts to:

- **Add new data sources**: Modify `clean_data.py` to handle different file formats
- **Change embedding models**: Update `generate_embed.py` to use different models
- **Customize FAISS parameters**: Adjust clustering and indexing in `compute_faiss_index.py`
- **Add data validation**: Enhance quality checks in any stage

### Batch Processing

For large datasets, consider:

- Processing data in smaller chunks
- Using parallel processing where possible
- Implementing checkpointing for long-running operations
- Monitoring memory usage and system resources

## Next Steps

After successful preprocessing:

1. **Start the application**: `make run`
2. **Test search functionality**: Use the web interface
3. **Monitor performance**: Check search speed and accuracy
4. **Iterate**: Adjust parameters based on results
