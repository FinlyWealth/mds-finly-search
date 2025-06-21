# FinlyWealth Product Search Engine

Contributors: Jenson Chang, Chukwunonso Ebele-Muolokwu, Da (Catherine) Meng, Jingyuan Wang

The goal of this project is to design and implement a fast, scalable multimodal search engine that captures the semantic meaning of user queries, allowing users to search using text, images, or both to find the most relevant products for [FinlyWealth](https://finlywealth.com/), an affiliate marketing platform expanding into e-commerce.

## Table of Contents

- [FinlyWealth Product Search Engine](#finlywealth-product-search-engine)
  - [Table of Contents](#table-of-contents)
  - [Complete Setup Instructions](#complete-setup-instructions)
    - [Prerequisites](#prerequisites)
    - [Setting up the database (choose one)](#setting-up-the-database-choose-one)
    - [Run the application (choose one)](#run-the-application-choose-one)
    - [Run experiment (optional)](#run-experiment-optional)
    - [Deployment](#deployment)
  - [Data Structure Requirements](#data-structure-requirements)
    - [Required Data Format](#required-data-format)
    - [Optional Columns](#optional-columns)
    - [Data Organization](#data-organization)
    - [Data Quality Requirements](#data-quality-requirements)
    - [Processing Pipeline Steps](#processing-pipeline-steps)
    - [Environment Configuration](#environment-configuration)
    - [Troubleshooting Data Issues](#troubleshooting-data-issues)
  - [Database Setup Instructions - Google Cloud SDK](#database-setup-instructions---google-cloud-sdk)
    - [Step 1. Setup Google Cloud SDK](#step-1-setup-google-cloud-sdk)
    - [Step 2. Sign in to Google Cloud](#step-2-sign-in-to-google-cloud)
    - [Step 3. Add the Google Cloud SQL Credentials](#step-3-add-the-google-cloud-sql-credentials)
  - [Database Setup Instructions - Docker Postgres](#database-setup-instructions---docker-postgres)
    - [Step 1. Start the Docker container](#step-1-start-the-docker-container)
    - [Step 2. Create Database Credentials](#step-2-create-database-credentials)
    - [Step 3. Load Data](#step-3-load-data)
  - [Application Setup Instructions - Makefile](#application-setup-instructions---makefile)
    - [Step 1. Setup Python environment](#step-1-setup-python-environment)
    - [Step 2. Configure Environment Variables](#step-2-configure-environment-variables)
    - [Step 3. Start Application](#step-3-start-application)
  - [Application Setup Instructions - Docker](#application-setup-instructions---docker)
    - [Step 1. Clone the Repository](#step-1-clone-the-repository)
    - [Step 2. Configure Environment Variables](#step-2-configure-environment-variables-1)
    - [Step 3. Build Docker Containers](#step-3-build-docker-containers)
    - [Step 4. Start the Application](#step-4-start-the-application)
    - [Step 5. Clean Up](#step-5-clean-up)
  - [API Commands](#api-commands)

## Complete Setup Instructions

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (if using docker-compose.yaml)
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install-sdk) (optional, for cloud database access)
- Git (optional, for cloning the repo)

### Setting up the database (choose one)

- [Instructions](#database-setup-instructions---google-cloud-sdk) for setting up Google SQL proxy to connect to the database hosted on Google Cloud.

- [Instructions](#database-setup-instructions---docker-postgres) for setting up a Postgres database in Docker. This method is recommended if you want to develop using your own embeddings locally. You will need to run indexing scripts to generate indices and load data.

### Run the application (choose one)

- [Instructions](#application-setup-instructions---makefile) for running the search engine application through make file.
- [Instructions](#application-setup-instructions---docker) for running the search engine application through docker container. This is to test for deployment. 

### Run experiment (optional)

- Please refer to the [Experiment Instructions](experiments/README.md) for running experiments.

### Deployment

- Deployments are done using Docker images. Follow [instructions](#application-setup-instructions---docker) to build and test Docker images locally.
- Use GitHub Actions to build and deploy images to Google Cloud.

## Data Structure Requirements

If you want to use this pipeline with your own dataset, your data must follow a specific structure to be compatible with the indexing scripts.

### Required Data Format

Your dataset should be provided as a **CSV file** with the following required columns:

| Column Name | Data Type | Description | Example |
|-------------|-----------|-------------|---------|
| `Pid` | string/int | Unique product identifier | "12345" or 12345 |
| `Name` | string | Product name | "Apple iPhone 14 Pro" |
| `Description` | string | Detailed product description | "Latest smartphone with advanced camera..." |
| `Category` | string | Product category | "Electronics" |
| `Price` | float | Original product price | 999.99 |
| `PriceCurrency` | string | Currency code (USD, CAD, GBP) | "USD" |

### Optional Columns

The pipeline can handle additional columns that will be processed and included in the final dataset. You can modify the `src/indexing/clean_data.py` to remove the columns not available in your dataset:

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `FinalPrice` | float | Final price after discounts |
| `Discount` | float | Discount amount or percentage |
| `isOnSale` | boolean | Whether the product is on sale |
| `IsInStock` | boolean | Stock availability |
| `Brand` | string | Product brand |
| `Manufacturer` | string | Product manufacturer |
| `Color` | string | Product color |
| `Gender` | string | Target gender (if applicable) |
| `Size` | string | Product size |
| `Condition` | string | Product condition (new, used, etc.) |

### Data Organization

Your data should be organized in the following directory structure:

```
data/
├── csv
│   ├── clean            # Cleaned CSV will be saved here
│   └── raw              # Put raw CSV here
├── images/              # Product images (optional for image search)
│   ├── 12345.jpeg       # Images named by Pid
│   ├── 12346.jpeg
│   └── ...
└── embeddings/          # Generated embeddings will be saved here
```

### Data Quality Requirements

- **Required columns**: `Pid`, `Name`, `Description`, `Category`, `Price`, `PriceCurrency` must be present
- **Currency filtering**: Only USD, CAD, and GBP currencies are supported
- **Text fields** should contain meaningful, non-empty strings
- **Product IDs** (`Pid`) must be unique
- **Encoding**: CSV files should be UTF-8 encoded
- **Images** (if using multimodal search): JPEG format, named as `{Pid}.jpeg`

### Processing Pipeline Steps

Once your data is properly formatted and placed in the correct directories:

1. **Data Cleaning**:

   ```bash
   python src/indexing/clean_data.py
   ```

   - Filters by supported currencies
   - Merges Brand and Manufacturer columns
   - Saves cleaned data to `data/csv/clean/data.csv`

2. **Embedding Generation**:

   ```bash
   python src/indexing/generate_embed.py
   ```

   - Generates CLIP embeddings for images (if available)
   - Generates MiniLM text embeddings from product names
   - Creates fusion embeddings combining both modalities
   - Saves embeddings to `data/embeddings/`

### Environment Configuration

You can customize data paths by setting environment variables in your `.env` file:

```bash
# Custom data paths (optional)
RAW_CSV_PATH=data/csv/raw/my_products.csv
CLEAN_CSV_PATH=data/csv/clean/my_clean_products.csv
EMBEDDINGS_PATH=data/my_embeddings
```

### Troubleshooting Data Issues

- **Missing required columns**: Ensure all mandatory columns are present with exact names (case-sensitive)
- **Currency issues**: Only USD, CAD, and GBP are supported in `PriceCurrency`
- **Empty values**: Remove rows with missing required information before processing
- **Image format**: Ensure product images are in JPEG format and named correctly (`{Pid}.jpeg`)
- **Large datasets**: The pipeline processes data in chunks of 500,000 rows for memory efficiency
- **Encoding issues**: Save your CSV with UTF-8 encoding to handle special characters

## Database Setup Instructions - Google Cloud SDK

### Step 1. Setup Google Cloud SDK

Install Google Cloud SDK for your platform from [here](https://cloud.google.com/sdk/docs/install-sdk)

For Mac, we suggest installing via Homebrew:

```bash
brew install google-cloud-sdk
```

Once installation is complete, sign in to your Google account. Ensure you've been granted access to the Google project from the repo admin.

### Step 2. Sign in to Google Cloud

```bash
gcloud init
```

Select your Google project (repo admin should provide you with the project ID)
![google-project](./img/google_project.png)

When prompted to configure a default Compute Region and Zone, select `n`.
![region-zone](./img/region_zone.png)

### Step 3. Add the Google Cloud SQL Credentials

Add the following by creating a `.env` text file in the root folder with the following configurations.

```bash
# Database configuration
PGUSER=postgres
PGPASSWORD=ZK3RjyBv6twoA9
PGHOST=localhost
PGPORT=5433
PGDATABASE=postgres
PGTABLE=products_1M
```

## Database Setup Instructions - Docker Postgres

### Step 1. Start the Docker container

```bash
docker compose -f docker-compose.db.yml up -d
```

### Step 2. Create Database Credentials

Add the following to the `.env` file

```bash
# Database configuration
PGUSER=postgres
PGPASSWORD=postgres # this need to match the environment POSTGRES_PASSWORD in the docker-compose.db.yml file
PGHOST=localhost
PGPORT=5432
PGDATABASE=postgres
PGTABLE=products
```

### Step 3. Load Data

Please refer to the [Indexing Instructions](src/indexing/README.md) on how to load the database with product data. You must do this before run the application.

## Application Setup Instructions - Makefile

### Step 1. Setup Python environment

Set up Python environment:

```{bash}
# Create a new Python environment
conda env create --f environment.yaml
```

### Step 2. Configure Environment Variables

Add the following to the `.env` file.

```bash
# LLM API key
OPENAI_API_KEY=<insert-api-key>

# Location of product images and FAISS indices on Google Cloud Storage
GCS_BUCKET_URL=https://storage.googleapis.com/mds-finly
```

### Step 3. Start Application

To start the app and the server:

```{bash}
# Starts streamlit frontend and API backend
make run
```

## Application Setup Instructions - Docker

### Step 1. Clone the Repository

In a separate terminal, clone the repository.

```bash
git clone FinlyWealth/mds-finly-search
cd mds-finly-search
```

### Step 2. Configure Environment Variables

Add the following to the `.env` file.

```bash
# LLM API key
OPENAI_API_KEY=<insert-api-key>

# Location of product images and FAISS indices on Google Cloud Storage
GCS_BUCKET_URL=https://storage.googleapis.com/mds-finly
```

### Step 3. Build Docker Containers

Start Docker and create the Docker images for both frontend and backend services.

```bash
docker compose build
```

This step may take several minutes as it downloads and builds all required dependencies.

### Step 4. Start the Application

Optinal: If using Google Cloud database, run `make proxy` to connect to the database.

```bash
docker compose up
```

The application will start two services:

- Frontend: Access the search interface at <http://localhost:8501>
- Backend API: Running at <http://localhost:5001>

### Step 5. Clean Up

To close the container and free up the port once the proxy is not needed

```bash
docker compose down
make clean
```

## API Commands

Please refer to [API documentation](src/backend/README.md) for more details. 
