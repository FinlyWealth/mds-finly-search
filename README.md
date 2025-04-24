# mds-finly-search

## Setup Instructions
1. Download the sample embeddings and images from: https://drive.google.com/file/d/1NW-ZzWFXdKlNZPJ64fkmVotWvr1zQmQx/view?usp=share_link
2. Extract the zip file in the `data` folder
3. Set up Python environment using `environment.yaml`
4. Install postgresql and pgvector extension
5. Create envrionment `.env` file
    ```bash
    PGUSER=jchang
    PGPASSWORD=123
    PGHOST=localhost
    PGPORT=5432
    PGDATABASE=finly
    ```
6. `python src/util/load_db.py` to load the database with the embeddings and metadata
7. `python app.py` to run the app

## Search workflow
![](./workflow.jpeg)
