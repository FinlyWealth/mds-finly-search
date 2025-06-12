# FinlyWealth Product Search Engine

A scalable, multimodal product search engine developed for [FinlyWealth](https://finlywealth.com/), an affiliate marketing platform expanding into e-commerce.

**Prerequisites**
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (if using docker-compose.yaml)
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install-sdk) (optional, for cloud database access)
- Git (optional, for cloning the repo)

**Setting up the database (choose one):**
- [Instructions](#setup-instructions---google-cloud-sdk) for setting up Google SQL proxy to connect to the database hosted on Google Cloud. 

- [Instructions](#setup-instructions---local-postgres) for setting up a local Postgres database. This method is recommended if you plan to develop with your own embeddings. 

**Running the application or related scripts**
- [Instructions](#setup-instructions---makefile) for running the search engine application 
- [Instructions](#using-makefile-to-run-mlflow-experiments) for running experiments.
- [Instructions](#using-makefile-for-preprocessing-and-generating-indices) for preprocessing scripts to generate indices and load data.

**Deployment**
- Deployments are done using Docker images. Follow [instructions](#setup-instructions---docker) to build and test Docker images locally.
- Use GitHub Actions to build and deploy images to Google Cloud.

## Setup Instructions - Google Cloud SDK

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
![google-project](./img/google-project.png)

When prompted to configure a default Compute Region and Zone, select `n`. 
![region-zone](./img/region-zone.png)

## Setup Instructions - Makefile

### Step 1. Setup Python environment

Set up Python environment:

```{bash}
# Create a new Python environment
conda env create --f environment.yaml
```

### Step 2. Configure Environment Variables
Set up the required environment variables for database connection and API access by creating a `.env` text file in the root folder with the following configurations.

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

### Step 3. Start Application
To start the app and the server:

```{bash}
# Starts streamlit frontend and API backend
make run
```

## Using Makefile to Run mlflow Experiments
Please refer to the [Experiment Instructions](experiments/README.md).

## Using Makefile for Preprocessing and Generating Indices
Please refer to the [Preprocessing Instructions](src/preprocess/README.md).

## Setup Instructions - Docker

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

## Setup Instructions - Local Postgres

### Step 1. Install Postgres
Install Postgres for your platform from [here](https://www.postgresql.org)

For Mac, we suggest installing via Homebrew:

```bash
brew install postgresql@17
```
### Step 2. Initialize Postgres
Initialize the database and sets the username for Postgres to be the same as the current bash user name. 
```bash
initdb -U $(whoami) -D /usr/local/var/postgresql@17
brew services start postgresql@17
```

### Step 3. Create Database Credentials
Add the following to the `.env` file

```bash
# Database configuration
PGUSER=<bash-usename> # From Step 2. 
PGPASSWORD=ZK3RjyBv6twoA9 # Or any other password you want to use
PGDATABASE=postgres
PGTABLE=products
```

### Step 4. Setup Database
This will create the database using information from Step 3. It will also add the pgvector extension to the database. 
```bash
make db-setup
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