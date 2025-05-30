import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os
import json
from PIL import Image
import io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.backend.api import app, search

@pytest.fixture
def client():
    """Fixture to create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_search_with_text_query_updated(client):
    """Test /api/search with a valid text query using mocked dependencies."""
    fake_embedding = np.random.rand(512).astype(np.float32)
    fake_pids = [1, 2, 3]
    fake_scores = [0.9, 0.85, 0.8]
    fake_results = [
        {
            "Pid": "1",
            "Name": "Item 1",
            "Description": "A product",
            "Brand": "BrandX",
            "Category": "Shoes",
            "Color": "Red",
            "Gender": "Unisex",
            "Size": "M",
            "Price": "99.99",
            "similarity": 0.9,
        },
        {
            "Pid": "2",
            "Name": "Item 2",
            "Description": "Another product",
            "Brand": "BrandY",
            "Category": "Shoes",
            "Color": "Black",
            "Gender": "Men",
            "Size": "L",
            "Price": "89.99",
            "similarity": 0.85,
        },
    ]


    with (
    patch("src.backend.api.initialization_status", {
        "minilm_model": True,
        "clip_model": True,
        "faiss_indices": True,
        "database": True,
    }),
    patch.dict(search.__globals__, {
        "components": ["mock_component_1", "mock_component_2", "mock_component_3"],
        "top_k": 10
    }),
    patch("src.backend.api.generate_embedding", return_value=fake_embedding),
    patch("src.backend.api.hybrid_retrieval", return_value=(fake_pids, fake_scores)),
    patch("src.backend.api.format_results", return_value=fake_results),
    patch("src.backend.api.reorder_search_results_by_relevancy", return_value=(fake_results, "mocked reasoning")),
    ):
        response = client.post("/api/search", data={"query": "sneakers"})
        assert response.status_code == 200
        data = response.get_json()

        assert "results" in data
        assert isinstance(data["results"], list)
        assert "elapsed_time_sec" in data
        assert "category_distribution" in data
        assert "brand_distribution" in data
        assert "price_range" in data
        assert "average_price" in data
        assert "session_id" in data
        assert "reasoning" in data

def test_search_with_image_only(client):
    """Test /api/search with image query only."""
    fake_embedding = np.random.rand(512).astype(np.float32)
    fake_pids = [1, 2, 3]
    fake_scores = [0.9, 0.85, 0.8]
    fake_results = [
        {"Pid": "1", "Name": "Item 1", "Brand": "BrandX", "Category": "Shoes", "Price": "99.99", "similarity": 0.9},
        {"Pid": "2", "Name": "Item 2", "Brand": "BrandY", "Category": "Shoes", "Price": "89.99", "similarity": 0.85},
    ]

    # Create fake image
    img = Image.new("RGB", (64, 64), color="black")
    img_io = io.BytesIO()
    img.save(img_io, format="PNG")
    img_io.seek(0)

    with (
        patch("src.backend.api.initialization_status", {
            "minilm_model": True,
            "clip_model": True,
            "faiss_indices": True,
            "database": True,
        }),
        patch.dict(search.__globals__, {
            "components": ["mock_component_1", "mock_component_2", "mock_component_3"],
            "top_k": 10
        }),
        patch("src.backend.api.generate_embedding", return_value=fake_embedding),
        patch("src.backend.api.hybrid_retrieval", return_value=(fake_pids, fake_scores)),
        patch("src.backend.api.format_results", return_value=fake_results),
        patch("src.backend.api.reorder_search_results_by_relevancy", return_value=(fake_results, "image search - no LLM")),
    ):
        response = client.post("/api/search", data={"file": (img_io, "test.png")}, content_type='multipart/form-data')
        assert response.status_code == 200
        data = response.get_json()
        assert "results" in data
        assert isinstance(data["results"], list)
        assert "reasoning" in data

def test_search_with_text_and_image(client):
    """Test /api/search with both text and image input."""
    fake_embedding = np.random.rand(512).astype(np.float32)
    fake_pids = [1, 2, 3]
    fake_scores = [0.9, 0.85, 0.8]
    fake_results = [
        {"Pid": "1", "Name": "Item 1", "Brand": "BrandX", "Category": "Shoes", "Price": "99.99", "similarity": 0.9},
        {"Pid": "2", "Name": "Item 2", "Brand": "BrandY", "Category": "Shoes", "Price": "89.99", "similarity": 0.85},
    ]

    from PIL import Image
    import io

    # Create a dummy black image
    img = Image.new("RGB", (64, 64), color="black")
    img_io = io.BytesIO()
    img.save(img_io, format="PNG")
    img_io.seek(0)

    with (
        patch("src.backend.api.initialization_status", {
            "minilm_model": True,
            "clip_model": True,
            "faiss_indices": True,
            "database": True,
        }),
        patch.dict(search.__globals__, {
            "components": ["mock_component_1", "mock_component_2", "mock_component_3"],
            "top_k": 10
        }),
        patch("src.backend.api.generate_embedding", return_value=fake_embedding),
        patch("src.backend.api.hybrid_retrieval", return_value=(fake_pids, fake_scores)),
        patch("src.backend.api.format_results", return_value=fake_results),
        patch("src.backend.api.reorder_search_results_by_relevancy", return_value=(fake_results, "mocked reasoning for multimodal")),
    ):
        data = {
            "query": "sneakers",
            "file": (img_io, "test.png")
        }

        response = client.post("/api/search", data=data, content_type='multipart/form-data')
        assert response.status_code == 200
        result = response.get_json()
        assert "results" in result
        assert isinstance(result["results"], list)
        assert result["reasoning"].lower() in [
            "mocked reasoning for multimodal",
            "no api key available, no llm reordering performed"
        ]

def test_search_missing_input(client):
    """Test the /api/search endpoint with missing query parameters.

    Parameters
    ----------
    client : flask.testing.FlaskClient

    Returns
    -------
    None
    """
    with patch("src.backend.api.initialization_status", {
        "minilm_model": True,
        "clip_model": True,
        "faiss_indices": True,
        "database": True,
    }):
        response = client.post("/api/search", data={})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
