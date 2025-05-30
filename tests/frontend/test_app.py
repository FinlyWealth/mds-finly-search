import pytest
from unittest.mock import patch, MagicMock
import requests
from PIL import Image
import io
import os
from src.frontend.app import (
    get_component_description,
    check_api_ready,
    load_image,
    submit_feedback
)

def test_get_component_description():
    """Test component description retrieval"""
    # Test known components
    assert get_component_description("minilm_model") == "Text embedding model for semantic search"
    assert get_component_description("clip_model") == "Vision-language model for image and text understanding"
    assert get_component_description("faiss_indices") == "Vector search indices for fast similarity search"
    assert get_component_description("database") == "Product database connection and tables"
    
    # Test unknown component
    assert get_component_description("unknown_component") == ""

@patch('requests.get')
def test_check_api_ready_success(mock_get):
    """Test API ready check when API is available"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"ready": True}
    mock_get.return_value = mock_response
    
    assert check_api_ready() is True
    mock_get.assert_called_once()

@patch('requests.get')
def test_check_api_ready_failure(mock_get):
    """Test API ready check when API is not available"""
    mock_get.side_effect = requests.exceptions.RequestException()
    assert check_api_ready() is False

@patch('requests.get')
def test_load_image_from_url(mock_get):
    """Test loading image from URL"""
    # Create a mock image
    mock_image = Image.new('RGB', (100, 100))
    img_byte_arr = io.BytesIO()
    mock_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Mock the response
    mock_response = MagicMock()
    mock_response.content = img_byte_arr
    mock_get.return_value = mock_response
    
    # Test loading from URL
    image = load_image("https://example.com/image.jpg")
    assert isinstance(image, Image.Image)
    assert image.size == (100, 100)

def test_load_image_from_local():
    """Test loading image from local path"""
    # Create a temporary test image
    test_image = Image.new('RGB', (100, 100))
    test_path = "test_image.png"
    test_image.save(test_path)
    
    try:
        image = load_image(test_path)
        assert isinstance(image, Image.Image)
        assert image.size == (100, 100)
    finally:
        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)

@patch('requests.post')
def test_submit_feedback(mock_post):
    """Test submitting feedback"""
    # Mock the session state
    with patch('streamlit.session_state') as mock_session:
        mock_session.get.side_effect = lambda key, default: {
            'query_text': 'test query',
            'image_input': 'test_image.jpg',
            'search_results': {'session_id': 'test_session'}
        }.get(key, default)
        
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Test feedback submission
        submit_feedback('test_pid', True)
        
        # Verify the API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        assert call_args['json']['pid'] == 'test_pid'
        assert call_args['json']['feedback'] is True
        assert call_args['json']['query_text'] == 'test query'
        assert call_args['json']['image_path'] == 'test_image.jpg'
        assert call_args['json']['session_id'] == 'test_session' 