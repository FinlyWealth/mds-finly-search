import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch
import os
import tempfile
from pathlib import Path

# Import the functions to test
from src.indexing.generate_embed import (
    save_embeddings,
    calculate_image_clip_embeddings,
    calculate_text_clip_embeddings,
    calculate_minilm_embeddings,
    concatenate_embeddings,
    filter_valid_products
)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'Pid': ['prod1', 'prod2', 'prod3'],
        'Name': ['Product 1', 'Product 2', 'Product 3']
    })

@pytest.fixture
def mock_embeddings():
    """Create sample embeddings for testing."""
    return np.random.rand(3, 512)  # 3 samples with 512 dimensions

@pytest.fixture
def mock_product_ids():
    """Create sample product IDs for testing."""
    return ['prod1', 'prod2', 'prod3']

def test_save_embeddings(tmp_path):
    """Test saving embeddings to a file."""
    # Create sample data
    embeddings = np.random.rand(3, 512)
    product_ids = ['prod1', 'prod2', 'prod3']
    embedding_type = "test_embedding"
    save_path = tmp_path / "test_embeddings.npz"
    
    # Save embeddings
    save_embeddings(embeddings, product_ids, embedding_type, str(save_path))
    
    # Load and verify saved data
    loaded_data = np.load(str(save_path))
    assert np.array_equal(loaded_data['embeddings'], embeddings)
    assert np.array_equal(loaded_data['product_ids'], product_ids)
    assert loaded_data['embedding_type'] == embedding_type

def test_concatenate_embeddings():
    """Test concatenation of image and text embeddings."""
    # Create sample embeddings
    image_embeddings = np.random.rand(3, 512)
    text_embeddings = np.random.rand(3, 512)
    
    # Concatenate embeddings
    result = concatenate_embeddings(image_embeddings, text_embeddings)
    
    # Verify shape and content
    assert result.shape == (3, 1024)  # 512 + 512 = 1024
    assert np.array_equal(result[:, :512], image_embeddings)
    assert np.array_equal(result[:, 512:], text_embeddings)

@pytest.mark.slow
@patch('PIL.Image.open')
def test_calculate_image_clip_embeddings(mock_image_open, sample_data):
    """Test image CLIP embedding calculation with real model and processor."""
    import torch
    from transformers import AutoModel, AutoProcessor
    device = 'cpu'
    model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    # Mock image open to avoid actual file I/O
    from PIL import Image
    dummy_image = Image.new('RGB', (224, 224))
    mock_image_open.return_value = dummy_image

    # Create data/images directory if it doesn't exist
    os.makedirs('data/images', exist_ok=True)
    for pid in sample_data['Pid']:
        Path(f"data/images/{pid}.jpeg").touch()
    try:
        embeddings, valid_indices = calculate_image_clip_embeddings(
            sample_data,
            model,
            processor,
            device=device,
            batch_size=1
        )
        assert len(embeddings) > 0
        assert len(valid_indices) > 0
    finally:
        # Only remove the test image files, not the directory
        for pid in sample_data['Pid']:
            os.remove(f"data/images/{pid}.jpeg")

@pytest.mark.slow
def test_calculate_text_clip_embeddings(sample_data):
    """Test text CLIP embedding calculation with real model and processor."""
    import torch
    from transformers import AutoModel, AutoProcessor
    device = 'cpu'
    model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    embeddings, product_ids = calculate_text_clip_embeddings(
        sample_data,
        model,
        processor,
        device=device,
        batch_size=1
    )
    assert len(embeddings) == len(sample_data)
    assert len(product_ids) == len(sample_data)

@pytest.mark.slow
def test_calculate_minilm_embeddings(sample_data):
    """Test MiniLM embedding calculation with real model and tokenizer."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = 'cpu'
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model.eval()
    embeddings, product_ids = calculate_minilm_embeddings(
        sample_data,
        model,
        tokenizer,
        device=device,
        batch_size=1
    )
    assert len(embeddings) == len(sample_data)
    assert len(product_ids) == len(sample_data)

@patch('os.path.exists')
def test_filter_valid_products(mock_exists, sample_data):
    """Test filtering of valid products."""
    # Mock file existence
    mock_exists.return_value = True
    
    # Filter products
    filtered_df = filter_valid_products(sample_data)
    
    # Verify results
    assert len(filtered_df) == len(sample_data)
    assert all(filtered_df['Name'].notna())
    assert all(filtered_df['Name'].str.strip() != '') 