import pytest
from unittest.mock import patch, MagicMock
import requests
from PIL import Image
import io
import os
import tempfile
import streamlit as st
from src.frontend.app import (
    get_component_description,
    check_api_ready,
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

def test_file_upload_handling():
    """Test handling of file upload in the app"""
    # Create a mock image
    mock_image = Image.new('RGB', (100, 100))
    img_byte_arr = io.BytesIO()
    mock_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Create a mock uploaded file
    mock_uploaded_file = MagicMock()
    mock_uploaded_file.name = "test.png"
    mock_uploaded_file.type = "image/png"
    mock_uploaded_file.getvalue.return_value = img_byte_arr.getvalue()
    mock_uploaded_file.stream = img_byte_arr
    
    # Mock streamlit's file uploader
    with patch('streamlit.file_uploader') as mock_uploader:
        mock_uploader.return_value = mock_uploaded_file
        
        # Test the file upload handling
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        assert uploaded_file is not None
        assert uploaded_file.name == "test.png"
        assert uploaded_file.type == "image/png"
        
        # Test image opening
        image = Image.open(uploaded_file.stream)
        assert isinstance(image, Image.Image)
        assert image.size == (100, 100)

def test_search_request_with_image():
    """Test search request with image upload"""
    # Create a mock image
    mock_image = Image.new('RGB', (100, 100))
    img_byte_arr = io.BytesIO()
    mock_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Create a mock uploaded file
    mock_uploaded_file = MagicMock()
    mock_uploaded_file.name = "test.png"
    mock_uploaded_file.type = "image/png"
    mock_uploaded_file.getvalue.return_value = img_byte_arr.getvalue()
    mock_uploaded_file.stream = img_byte_arr
    
    # Mock streamlit's file uploader and requests.post
    with patch('streamlit.file_uploader') as mock_uploader, \
         patch('streamlit.session_state') as mock_session, \
         patch('requests.post') as mock_post, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.empty') as mock_empty:
        
        # Setup mocks
        mock_uploader.return_value = mock_uploaded_file
        mock_session.get.side_effect = lambda key, default: {
            'trigger_search': True,
            'query_text': None
        }.get(key, default)
        mock_button.return_value = True  # Simulate search button click
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            'results': [],
            'elapsed_time_sec': 0.1,
            'category_distribution': {},
            'brand_distribution': {},
            'price_range': [0, 0],
            'average_price': 0,
            'session_id': 'test-session',
            'reasoning': 'Test reasoning'
        }
        
        # Import the app module
        import src.frontend.app
        
        # Simulate file upload
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        assert uploaded_file is not None
        
        # Simulate the app flow
        if uploaded_file is not None:
            # Save uploaded file to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file.flush()
                
                # Prepare request data
                request_data = {
                    "search_type": "image",
                    "top_k": 100
                }
                
                # Prepare files
                files = {
                    'file': ('test.png', open(tmp_file.name, 'rb'), 'image/png')
                }
                
                # Make API request
                response = requests.post(
                    "http://localhost:8000/search",
                    files=files,
                    data=request_data
                )
                
                # Clean up temp file
                os.unlink(tmp_file.name)
        
        # Verify the request was made with correct data
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        assert 'files' in call_args
        assert 'data' in call_args
        assert call_args['data']['search_type'] == 'image'
        assert 'file' in call_args['files']