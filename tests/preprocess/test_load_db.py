import psycopg2
import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from preprocess import load_db


TABLE_NAME = "products_test"
EMBEDDING_TYPE = "fusion"
EMBEDDING_DIM= 1536
EMBEDDING_DIMS = {EMBEDDING_TYPE: EMBEDDING_DIM}

def get_enabled_embedding_types():
    return ['fusion']

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
    # Patch config.db.DB_CONFIG and TABLE_NAME to test container values
    monkeypatch.setattr(load_db, "DB_CONFIG", pg_container)
    monkeypatch.setattr(load_db, "TABLE_NAME", TABLE_NAME)

    # Also patch get_enabled_embedding_types if needed inside your module
    monkeypatch.setattr(load_db, "get_enabled_embedding_types", get_enabled_embedding_types)

    # Now call init_db, it uses the patched config.db values
    load_db.init_db(embedding_dims=EMBEDDING_DIMS, drop=True)

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


def test_insert_data(pg_container, pg_conn, monkeypatch):
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
    monkeypatch.setattr(load_db, "DB_CONFIG", pg_container)
    monkeypatch.setattr(load_db, "TABLE_NAME", TABLE_NAME)
    monkeypatch.setattr(load_db, "get_enabled_embedding_types", get_enabled_embedding_types)

    # Initialize the table
    load_db.init_db(embedding_dims=EMBEDDING_DIMS, drop=True)

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
    pids = [
        "127.2.AFF9DD86A06W8144.3F8EC3F904474916.194345954245", 
        "127.2.BGF8DD86P0648244.39BD7AB4FBF9CB0C.193093842019"
    ]
    embeddings = np.random.rand(2, EMBEDDING_DIM)
    embeddings_dict = {
        f"{EMBEDDING_TYPE}_pids": pids,
        EMBEDDING_TYPE: embeddings,
    }

    # Sample metadata DataFrame
    df = pd.DataFrame([
        {
            'Pid': pids[0],
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
            'Pid': pids[1],
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

    # Run insert
    load_db.insert_data(embeddings_dict, pids, df)

    # Verify inserted rows
    cur.execute(f"SELECT Pid, Name FROM {TABLE_NAME} ORDER BY Pid")
    results = cur.fetchall()
    assert results == [(pids[0], 'Alfani Polo Shirt'), (pids[1], 'Wood Traditional Stool')]

    cur.close()
