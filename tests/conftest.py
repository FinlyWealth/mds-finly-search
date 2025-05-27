import pytest
import psycopg2
from testcontainers.postgres import PostgresContainer
import urllib.parse


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
