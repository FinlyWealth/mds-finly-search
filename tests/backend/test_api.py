import pytest
import json
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.backend import api
from src.backend.api import app
from src.backend.api import load_image
from src.backend.api import format_results

@pytest.fixture
def client():
    """Fixture to create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    """Test the root endpoint.

    Parameters
    ----------
    client : flask.testing.FlaskClient

    Returns
    -------
    None
    """
    response = client.get("/")
    assert response.status_code == 200
    assert b"Backend API is running!" in response.data

def test_ready_endpoint(client):
    """Test the readiness endpoint for API initialization.

    Parameters
    ----------
    client : flask.testing.FlaskClient

    Returns
    -------
    None
    """
    api.initialize_app()
    response = client.get("/api/ready")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "state" in data
    assert "components" in data

def test_feedback_missing_fields(client):
    """Test the /api/feedback endpoint with missing required fields.

    Parameters
    ----------
    client : flask.testing.FlaskClient

    Returns
    -------
    None
    """
    response = client.post("/api/feedback", json={})
    assert response.status_code in [400, 500]
    data = json.loads(response.data)
    assert "error" in data

def test_initialize_app_success():
    """Test that initialize_app() runs successfully with mocked components.

    Returns
    -------
    None
    """
    with patch("src.backend.api.create_retrieval_component", return_value=MagicMock()), \
         patch("src.backend.api.spacy.load", return_value=MagicMock()), \
         patch("src.backend.api.initialize_minilm_model"), \
         patch("src.backend.api.initialize_clip_model"), \
         patch("src.backend.api.psycopg2.connect") as mock_connect:

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.execute.return_value = None
        mock_connect.return_value = mock_conn

        result = api.initialize_app()

        assert result is True
        assert api.initialization_state == "ready"
        assert all(api.initialization_status.values())

def test_initialize_app_failure():
    """Test that initialize_app() handles initialization failure gracefully.

    This test mocks a failure during one of the initialization steps and checks that:
    - The function returns False.
    - The initialization_state is set to 'failed'.
    - All entries in initialization_status are reset to False.

    Returns
    -------
    None
    """
    with patch("src.backend.api.create_retrieval_component", side_effect=Exception("mock failure")), \
         patch("src.backend.api.spacy.load"), \
         patch("src.backend.api.initialize_minilm_model"), \
         patch("src.backend.api.initialize_clip_model"), \
         patch("src.backend.api.psycopg2.connect"):

        result = api.initialize_app()

        assert result is False

        assert api.initialization_state == "failed"
        assert all(value is False for value in api.initialization_status.values())

def test_load_image_from_url():
    """Test load_image with a valid URL.

    Returns
    -------
    None
    """
    fake_image = MagicMock()
    with patch("src.backend.api.requests.get") as mock_get, \
         patch("src.backend.api.Image.open", return_value=fake_image) as mock_open:

        mock_get.return_value.content = b"fake_image_bytes"
        image = load_image("http://example.com/image.jpg")

        mock_get.assert_called_once()
        mock_open.assert_called_once()
        assert image == fake_image

def test_load_image_from_local_path():
    """Test load_image with a local file path.

    Returns
    -------
    None
    """
    fake_image = MagicMock()
    with patch("src.backend.api.Image.open", return_value=fake_image) as mock_open:
        image = load_image("path/to/local/image.png")
        mock_open.assert_called_once_with("path/to/local/image.png")
        assert image == fake_image

def test_load_image_failure():
    """Test load_image raises exception for invalid input.

    Returns
    -------
    None
    """
    with patch("src.backend.api.Image.open", side_effect=IOError("file not found")):
        with pytest.raises(Exception) as exc_info:
            load_image("invalid_path.jpg")
        assert "Error loading image" in str(exc_info.value)

def test_format_results_success():
    """Test format_results formats matching products and scores correctly.

    This test mocks `fetch_products_by_pids` to return product info for the given PIDs,
    and checks whether `format_results` builds the result list as expected.

    Returns
    -------
    None
    """
    indices = [101, 102]
    scores = [0.95, 0.87]

    mock_products = {
        101: {
            "Name": "Shoe A",
            "Description": "Good shoe",
            "Brand": "Nike",
            "Category": "Footwear",
            "Color": "Black",
            "Gender": "Unisex",
            "Size": "42",
            "Price": 129.99,
        },
        102: {
            "Name": "Shoe B",
            "Description": "Running shoe",
            "Brand": "Adidas",
            "Category": "Footwear",
            "Color": "White",
            "Gender": "Men",
            "Size": "43",
            "Price": None,
        },
    }

    with patch("src.backend.api.fetch_products_by_pids", return_value=mock_products):
        results = format_results(indices, scores)

        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]["Pid"] == "101"
        assert results[0]["Price"] == "129.99"
        assert results[1]["Pid"] == "102"
        assert results[1]["Price"] is None

def test_format_results_missing_pid():
    """Test format_results skips indices not found in the fetched product dictionary.

    Returns
    -------
    None
    """
    indices = [200]
    scores = [0.75]
    mock_products = {}  # Simulate no data returned

    with patch("src.backend.api.fetch_products_by_pids", return_value=mock_products):
        results = format_results(indices, scores)
        assert results == []
