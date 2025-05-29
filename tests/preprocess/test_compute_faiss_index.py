import os
import pytest
import numpy as np
import faiss
import json
from unittest.mock import patch, mock_open, MagicMock
import sys
import psutil
from preprocess.compute_faiss_index import (
    get_memory_usage,
    load_embeddings_from_files,
    create_faiss_index,
    verify_index,
    save_index,
    create_index_mapping,
    save_mapping
)

# Test data
SAMPLE_EMBEDDINGS = np.random.rand(100, 128).astype(np.float32)
SAMPLE_PIDS = [f"pid_{i}" for i in range(100)]

@pytest.fixture
def mock_npz_data():
    """Create mock NPZ data for testing"""
    return {
        'product_ids': SAMPLE_PIDS,
        'embeddings': SAMPLE_EMBEDDINGS
    }

def test_get_memory_usage():
    """Test memory usage tracking"""
    memory_usage = get_memory_usage()
    assert isinstance(memory_usage, float)
    assert memory_usage > 0

@patch('os.listdir')
@patch('numpy.load')
def test_load_embeddings_from_files(mock_np_load, mock_listdir):
    """Test loading embeddings from NPZ files"""
    # Mock directory listing
    mock_listdir.return_value = ['test_embeddings_chunk_0.npz']
    
    # Mock NPZ file loading
    mock_np_load.return_value = mock_npz_data()
    
    # Mock file path
    with patch('os.path.join', return_value='dummy/path'):
        with patch('os.path.exists', return_value=True):
            embeddings_data = load_embeddings_from_files()
    
    assert 'test' in embeddings_data
    assert 'pids' in embeddings_data['test']
    assert 'embeddings' in embeddings_data['test']
    assert 'dim' in embeddings_data['test']
    assert embeddings_data['test']['dim'] == 128

def test_create_faiss_index():
    """Test FAISS index creation"""
    nlist = 10
    index, pid_map = create_faiss_index(
        SAMPLE_EMBEDDINGS,
        SAMPLE_PIDS,
        embedding_dim=128,
        nlist=nlist,
        compressed=False
    )
    
    assert isinstance(index, faiss.Index)
    assert len(pid_map) == len(SAMPLE_PIDS)
    assert all(str(i) in pid_map for i in range(len(SAMPLE_PIDS)))

def test_verify_index():
    """Test index verification"""
    # Create a simple index for testing
    index = faiss.IndexFlatIP(128)
    index.add(SAMPLE_EMBEDDINGS)
    
    # Test verification
    result = verify_index(index, SAMPLE_EMBEDDINGS[:10], SAMPLE_PIDS[:10])
    assert result is True

@patch('faiss.write_index')
def test_save_index(mock_write_index):
    """Test saving FAISS index"""
    index = faiss.IndexFlatIP(128)
    save_index(index, 'test_index.faiss')
    mock_write_index.assert_called_once()

def test_create_index_mapping():
    """Test creation of index mapping"""
    mapping = create_index_mapping(SAMPLE_PIDS)
    assert len(mapping) == len(SAMPLE_PIDS)
    assert all(str(i) in mapping for i in range(len(SAMPLE_PIDS)))
    assert mapping['0'] == SAMPLE_PIDS[0]

@patch('builtins.open', new_callable=mock_open)
def test_save_mapping(mock_file):
    """Test saving index mapping"""
    mapping = create_index_mapping(SAMPLE_PIDS)
    save_mapping(mapping, 'test_mapping.json')
    mock_file.assert_called_once_with('test_mapping.json', 'w')
    mock_file().write.assert_called_once()

@pytest.mark.parametrize("compressed", [True, False])
def test_create_faiss_index_compression(compressed):
    """Test FAISS index creation with and without compression"""
    nlist = 10
    index, pid_map = create_faiss_index(
        SAMPLE_EMBEDDINGS,
        SAMPLE_PIDS,
        embedding_dim=128,
        nlist=nlist,
        compressed=compressed
    )
    
    assert isinstance(index, faiss.Index)
    if compressed:
        assert isinstance(index, faiss.IndexIVFScalarQuantizer)
    else:
        assert isinstance(index, faiss.IndexIVFFlat)

def test_create_faiss_index_insufficient_vectors():
    """Test FAISS index creation with insufficient vectors"""
    with pytest.raises(ValueError):
        create_faiss_index(
            SAMPLE_EMBEDDINGS[:5],  # Too few vectors
            SAMPLE_PIDS[:5],
            embedding_dim=128,
            nlist=10,
            compressed=False
        ) 