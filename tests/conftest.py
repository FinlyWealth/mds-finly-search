import pytest
import psycopg2
from testcontainers.postgres import PostgresContainer
import urllib.parse
import numpy as np
import sys
import os
from src.preprocess import load_db
import numpy as np

@pytest.fixture(scope="module")
def pg_container():
    """Start postgres container with pgvector once per test module"""
    with PostgresContainer("ankane/pgvector:latest") as postgres:
        # Parse connection URL
        url = urllib.parse.urlparse(postgres.get_connection_url())
        db_config = {
            'dbname': url.path[1:],  # skip leading slash
            'user': url.username,
            'password': url.password,
            'host': url.hostname,
            'port': url.port,
        }
        yield db_config
        # container and connection will be cleaned up automatically here

@pytest.fixture
def pg_conn(pg_container):
    """Create a new connection for each test"""
    conn = psycopg2.connect(**pg_container)
    yield conn
    conn.close()

@pytest.fixture(scope="session")
def sample_embeddings():
    """Create sample embeddings for testing"""
    return np.random.rand(100, 128).astype(np.float32)

@pytest.fixture(scope="session")
def sample_pids():
    """Create sample product IDs for testing"""
    return [f"pid_{i}" for i in range(100)]

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data"""
    return tmp_path_factory.mktemp("test_data")

@pytest.fixture(scope="session")
def mock_npz_file(test_data_dir, sample_embeddings, sample_pids):
    """Create a mock NPZ file for testing"""
    npz_path = test_data_dir / "test_embeddings_chunk_0.npz"
    np.savez(
        npz_path,
        product_ids=sample_pids,
        embeddings=sample_embeddings
    )
    return str(npz_path) 


@pytest.fixture
def fake_embeddings_dir(tmp_path, monkeypatch):
    """
    Create a temporary directory with fake embedding .npz files.
    Monkeypatch EMBEDDINGS_PATH to use this temporary directory.
    """
    filenames = [
        "fusion_embeddings_chunk_0.npz",
        "fusion_embeddings_chunk_1.npz",
        "image_embeddings_chunk_0.npz",
        "text_embeddings.npz",
    ]
    for name in filenames:
        # tmp_path is a built-in pytest fixture that provides a temporary directory 
        # it is automatically cleaned up after the test finishes
        np.savez(tmp_path / name, dummy=np.array([1, 2, 3]))
    
    # monkeypatch is a built-in fixture that allows
    # override the EMBEDDINGS_PATH variable inside load_db to point to the temporary test folder
    monkeypatch.setattr(load_db, "EMBEDDINGS_PATH", str(tmp_path))

    return filenames, tmp_path