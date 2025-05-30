import psycopg2
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
import sys
import os
import time
from src.preprocess import load_db
from src.preprocess.load_db import (
    get_base_embedding_type,
    get_embedding_paths,
    get_enabled_embedding_types,
    get_chunked_files,
    validate_numeric,
    validate_boolean,
    validate_text,
    validate_and_clean_dataframe,
    save_checkpoint,
    load_checkpoint,
    insert_data,
    init_db,
    main
)
from config.db import DB_CONFIG, TABLE_NAME


TABLE_NAME = "products_test"
EMBEDDING_TYPE = "fusion"
EMBEDDING_DIM= 1536
EMBEDDING_DIMS = {EMBEDDING_TYPE: EMBEDDING_DIM}
PIDS = [
    "127.2.AFF9DD86A06W8144.3F8EC3F904474916.194345954245",
    "127.2.BGF8DD86P0648244.39BD7AB4FBF9CB0C.193093842019"
]


class MockNpzFile:
    def __init__(self, embeddings, product_ids):
        self.data = {
            "embeddings": embeddings,
            "product_ids": product_ids
        }

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

@pytest.fixture
def sample_metadata_df():
    """Return a sample metadata DataFrame with PIDs."""
    return pd.DataFrame([
        {
            'Pid': PIDS[0],
            'Name': 'Alfani Polo Shirt',
            'Description': 'Alfani Men Mercerized Polo Shirt, Neo Navy',
            'Category': 'Apparel & Accessories ',
            'Price': 98,
            'PriceCurrency': 'USD',
            'FinalPrice': 98,
            'Discount': 0,
            'isOnSale': True,
            'IsInStock': True,
            'MergedBrand': 'Alfani',
            'Color': 'Neo Navy',
            'Gender': 'male',
            'Size': 'S',
            'Condition': 'new'
        },
        {
            'Pid': PIDS[1],
            'Name': 'Wood Traditional Stool',
            'Description': 'Wood Traditional Stool - Black',
            'Category': 'Furniture >Chairs >Table & Bar Stools ',
            'Price': 120,
            'PriceCurrency': 'USD',
            'FinalPrice': 99.95,
            'Discount': 17,
            'isOnSale': True,
            'IsInStock': True,
            'MergedBrand': 'Rosemary',
            'Color': 'Black',
            'Gender': 'unisex',
            'Condition': 'new'
        },
    ])

def get_enabled_embedding_types():
    return ['fusion']

def wait_for_db(container_config, max_retries=5, delay=2):
    """Wait for database to be ready with retries"""
    for i in range(max_retries):
        try:
            conn = psycopg2.connect(**container_config)
            conn.close()
            return True
        except psycopg2.OperationalError:
            if i < max_retries - 1:
                time.sleep(delay)
            continue
    return False

def test_init_db(monkeypatch, pg_container, pg_conn):
    """
    Test the `init_db` function to ensure the database and required tables are created correctly.

    This test uses a real PostgreSQL database running in a Docker container with the pgvector extension enabled.
    It monkeypatches the database configuration to connect to the test container and verifies that:
      - The `init_db` function runs without error.
      - The target table is created with the expected columns.
      - The GIST index on the tsvector `document` column is created.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest fixture to patch the DB configuration dynamically.
        pg_container (dict): Dictionary containing dynamic connection parameters for the test Postgres container.
        pg_conn (psycopg2.connection): A live connection to the test Postgres instance for verification queries.
    """
    # Wait for database to be ready
    assert wait_for_db(pg_container), "Database failed to become ready"
    
    # Print connection details for debugging
    print(f"Container config: {pg_container}")
    
    # Patch config.db.DB_CONFIG and TABLE_NAME to test container values
    monkeypatch.setattr('config.db.DB_CONFIG', pg_container)
    monkeypatch.setattr('config.db.TABLE_NAME', TABLE_NAME)
    monkeypatch.setattr('src.preprocess.load_db.get_enabled_embedding_types', get_enabled_embedding_types)

    # Now call init_db, it uses the patched config.db values
    init_db(embedding_dims=EMBEDDING_DIMS, drop=True)

    cur = pg_conn.cursor()

    # Check table created
    cur.execute(f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = '{TABLE_NAME}'
        );
    """)
    assert cur.fetchone()[0]

    # Check embedding columns
    cur.execute(f"""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = '{TABLE_NAME}';
    """)
    columns = [row[0] for row in cur.fetchall()]
    for emb_type in get_enabled_embedding_types():
        assert f"{emb_type}_embedding" in columns

    # Check GIST (Generalized Search Tree) index on document column exists
    cur.execute(f"""
        SELECT indexname FROM pg_indexes
        WHERE tablename = '{TABLE_NAME}'
          AND indexdef LIKE '%USING gist%document%';
    """)
    assert cur.fetchone() is not None

    cur.close()


def test_insert_data(pg_container, pg_conn, monkeypatch, sample_metadata_df):
    """
    Test inserting two product rows with embeddings into the test PostgreSQL table.

    Parameters
    ----------
    pg_container : dict
        Dictionary with PostgreSQL connection details provided by the test container.
    pg_conn : psycopg2.extensions.connection
        A live PostgreSQL connection to run verification queries.
    monkeypatch : _pytest.monkeypatch.MonkeyPatch
        Pytest utility to patch module-level constants and functions.

    Asserts
    -------
    The inserted rows match the product metadata and are stored in the correct table.
    """
    # Patch DB configuration and table name
    monkeypatch.setattr('config.db.DB_CONFIG', pg_container)
    monkeypatch.setattr('config.db.TABLE_NAME', TABLE_NAME)
    monkeypatch.setattr('src.preprocess.load_db.get_enabled_embedding_types', get_enabled_embedding_types)

    # Initialize the table
    init_db(embedding_dims=EMBEDDING_DIMS, drop=True)

    cur = pg_conn.cursor()

    # Check table created
    cur.execute(f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = '{TABLE_NAME}'
        );
    """)
    assert cur.fetchone()[0]

    # Sample PIDs and embeddings
    embeddings = np.random.rand(2, EMBEDDING_DIM)
    embeddings_dict = {
        f"{EMBEDDING_TYPE}_pids": PIDS,
        EMBEDDING_TYPE: embeddings,
    }

    # Run insert
    insert_data(embeddings_dict, PIDS, sample_metadata_df)

    # Verify inserted rows
    cur.execute(f"SELECT Pid, Name FROM {TABLE_NAME} ORDER BY Pid")
    results = cur.fetchall()
    assert results == [(PIDS[0], 'Alfani Polo Shirt'), (PIDS[1], 'Wood Traditional Stool')]

    cur.close()


def test_load_db_main(sample_metadata_df, fake_embeddings_dir, monkeypatch, capsys):
    """
    Test the `main` function's embedding loading and filtering logic using mocked embedding files.

    This test verifies:
    - Embedding chunk files are correctly discovered and loaded.
    - Only valid chunks with required keys and matching shapes are used.
    - The resulting embeddings and product IDs are concatenated correctly.
    - The number of products with all required embeddings is printed correctly.

    Parameters
    ----------
    sample_metadata_df : pandas.DataFrame
        A fixture providing a sample product metadata DataFrame, simulating the CSV content used
        for joining with product IDs from the embeddings.
    fake_embeddings_dir : tuple
        A fixture that returns fake embedding filenames and the path to the temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture for patching modules or attributes.
    capsys : _pytest.capture.CaptureFixture
        Pytest fixture to capture stdout/stderr for assertion.
    """
    filenames, tmp_path = fake_embeddings_dir

    # Mock get_embedding_paths to simulate expected embedding types
    monkeypatch.setattr('src.preprocess.load_db.get_embedding_paths', lambda: {
        "fusion": "fusion_embeddings_chunk_0.npz",
        "image": "image_embeddings_chunk_0.npz",
        "text": "text_embeddings.npz"
    })

    # Mock get_base_embedding_type to just return the name (since it's identity in test)
    monkeypatch.setattr('src.preprocess.load_db.get_base_embedding_type', lambda name: name)

    # Mock get_chunked_files to return the list of .npz files
    def mock_get_chunked_files(name):
        if name == "fusion":
            return ["fusion_embeddings_chunk_0.npz", "fusion_embeddings_chunk_1.npz"]
        if name == "image":
            return [f"{name}_embeddings_chunk_0.npz"]
        return f"{name}_embeddings.npz"
    monkeypatch.setattr('src.preprocess.load_db.get_chunked_files', mock_get_chunked_files)

    # Mock get_enabled_embedding_types to return the test types
    monkeypatch.setattr('src.preprocess.load_db.get_enabled_embedding_types', lambda: ["fusion", "image", "text"])

    # Mock pandas.read_csv to return our test DataFrame
    monkeypatch.setattr(pd, "read_csv", lambda _: sample_metadata_df)

    # Mock np.load to load our dummy embedding data with expected structure
    additional_pid = "127.2.CFF8DD86A0648144.580609EDA08971F5.195027714054"
    def mock_np_load(path, *args, **kwargs):
        if "chunk_0" in path:
            return MockNpzFile(
                embeddings=np.array([[0]*768, [1]*768]),
                product_ids=np.array(PIDS)
            )
        if "chunk_1" in path:
            return MockNpzFile(
                embeddings=np.array([[2]*768]),
                product_ids=np.array([additional_pid])
            )
        return MockNpzFile(
            embeddings=np.array([[0]*768, [1]*768]),
            product_ids=np.array(PIDS)
        ) 
    monkeypatch.setattr(np, "load", mock_np_load)

    # Mock init_db and insert_data to verify they're called
    mock_init_db = MagicMock()
    mock_insert_data = MagicMock()
    monkeypatch.setattr('src.preprocess.load_db.init_db', mock_init_db)
    monkeypatch.setattr('src.preprocess.load_db.insert_data', mock_insert_data)

    # Run the main function
    main()

    # Assert init_db and insert_data were called
    assert mock_init_db.called, "init_db should be called"
    assert mock_insert_data.called, "insert_data should be called"

    args, _ = mock_insert_data.call_args
    embeddings_dict, common_pids, df_arg = args

    # Check fusion embedding shape (2 from chunk_0 + 1 from chunk_1 = 3 rows in total, each row has 768)
    assert "fusion" in embeddings_dict
    assert embeddings_dict["fusion"].shape == (3, 768)

    assert embeddings_dict["fusion_pids"].tolist() == PIDS + [additional_pid]

    # PID intersection should be only PIDS
    assert sorted(common_pids) == PIDS

    # DataFrame should match original
    assert df_arg.equals(sample_metadata_df)

