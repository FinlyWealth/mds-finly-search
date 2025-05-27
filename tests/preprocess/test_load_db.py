import psycopg2
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from preprocess import load_db


TABLE_NAME = "products_test"

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

    embedding_dims = {'fusion': 1536}

    # Also patch get_enabled_embedding_types if needed inside your module
    monkeypatch.setattr(load_db, "get_enabled_embedding_types", get_enabled_embedding_types)

    # Now call init_db, it uses the patched config.db values
    load_db.init_db(embedding_dims=embedding_dims, drop=True)

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