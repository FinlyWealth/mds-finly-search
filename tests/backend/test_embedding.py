import pytest
import numpy as np
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.backend.embedding import (
    get_text_embedding,
    get_image_embedding,
    get_minilm_embeddings,
    generate_embedding,
    generate_image_caption
)

sample_text = "A test sentence for embedding."

@pytest.fixture
def sample_image():
    """Create a simple black image for testing.

    Returns
    -------
    PIL.Image.Image
        A black RGB image of size 64x64.
    """
    img = Image.new("RGB", (64, 64), color="black")
    return img

def test_text_embedding():
    """Test generating CLIP embedding for a text input.

    Ensures the returned embedding is a non-empty NumPy array
    without NaN values.
    """
    emb = get_text_embedding(sample_text)
    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] > 0
    assert not np.isnan(emb).any()

def test_minilm_embedding():
    """Test generating MiniLM embedding for a text input.

    Ensures the returned embedding is a non-empty NumPy array
    without NaN values.
    """
    emb = get_minilm_embeddings(sample_text)
    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] > 0
    assert not np.isnan(emb).any()

def test_image_embedding(sample_image):
    """Test generating CLIP embedding for a PIL image.

    Ensures the returned embedding is a non-empty NumPy array
    without NaN values.
    """
    emb = get_image_embedding(sample_image)
    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] > 0
    assert not np.isnan(emb).any()

def test_generate_text_only_embedding():
    """Test fused embedding generation using only text input.

    Checks that the result is a valid NumPy vector.
    """
    emb = generate_embedding(query_text=sample_text)
    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] > 0
    assert not np.isnan(emb).any()

def test_generate_image_only_embedding(sample_image):
    """Test fused embedding generation using only image input.

    Checks that the result is a valid NumPy vector.
    """
    emb = generate_embedding(query_image=sample_image)
    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] > 0
    assert not np.isnan(emb).any()

def test_generate_fusion_embedding(sample_image):
    """Test fused embedding generation using both text and image.

    Checks that the concatenated vector is valid and normalized.
    """
    emb = generate_embedding(query_text=sample_text, query_image=sample_image)
    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] > 0
    assert not np.isnan(emb).any()
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)

def test_image_embedding_invalid_input():
    """Test image embedding with invalid input.

    Expects None to be returned when input is not a valid image.
    """
    result = get_image_embedding("not_an_image")
    assert result is None


