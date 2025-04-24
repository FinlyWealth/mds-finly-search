import pandas as pd
import requests
import os
import sys
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
import spacy
from .embeddings import generate_embedding, generate_image_caption
from src.db.db import hybrid_search

# Workaround for Streamlit and custom PyTorch class issue
# torch.classes.__path__ = []

app = Flask(__name__)

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

def load_data():
    try:
        print("Starting data load...")
        # Get the project root directory (three levels up from current file)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_root, 'data')
        print(f"Data directory: {data_dir}")
        
        print("Loading metadata...")
        df = pd.read_parquet(os.path.join(data_dir, 'merged_output_sample_100k.parquet'))
        print("Styles loaded successfully")
        
        return df
    except Exception as e:
        print(f"Detailed error in load_data: {str(e)}")
        raise Exception(f"Error loading data: {str(e)}")

def load_image(image_path):
    try:
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)
        return image
    except Exception as e:
        raise Exception(f"Error loading image: {e}")

def format_results(indices, scores, df):
    results = []
    for idx, score in zip(indices, scores):
        try:
            # Look up product by ID instead of using iloc
            product = df[df['Pid'] == idx].iloc[0]
            results.append({
                'Pid': str(idx),
                'Name': str(product['Name']),
                'Description': str(product['Description']),
                'Brand': str(product['Brand']),
                'Manufacturer': str(product['Manufacturer']),
                'Color': str(product['Color']),
                'Gender': str(product['Gender']),
                'Size': str(product['Size']),
                'similarity': float(score)
            })
        except IndexError:
            # Skip products that aren't found in the DataFrame
            print(f"Warning: Product ID {idx} not found in DataFrame")
            continue
    return results

# Initialize data
print("Initializing data...")
try:
    df = load_data()
    print("Data initialization complete")
except Exception as e:
    print(f"Failed to initialize data: {str(e)}")
    raise e

@app.route('/api/search/text', methods=['POST'])
def search_by_text():
    try:
        print("Received text search request")
        data = request.get_json()
        print(f"Request data: {data}")
        query_text = data.get('query')
        print(f"Query text: {query_text}")
        
        if not query_text:
            return jsonify({'error': 'Query text is required'}), 400
            
        print("Generating embedding...")
        # Generate embedding and get results from database
        query_embedding = generate_embedding(query_text=query_text)
        print(f"Generated embedding shape: {query_embedding.shape}")
        
        print("Performing hybrid search...")
        # Get hybrid search results
        pids, scores = hybrid_search(
            query=query_text,
            query_embedding=query_embedding,
            top_k=10,
            text_weight=0.5,
            image_weight=0.3,
            ts_weight=0.2
        )
        
        response = {
            'results': format_results(pids, scores, df)
        }
        print(f"Response: {response}")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in search_by_text: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search/image', methods=['POST'])
def search_by_image():
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        
        if not image_path:
            return jsonify({'error': 'Image path is required'}), 400
            
        # Load and validate image
        image = load_image(image_path)
        if not image:
            return jsonify({'error': 'Failed to load image'}), 400
            
        print("Generating image caption...")
        # Generate caption using BLIP
        caption = generate_image_caption(image_path)
        if not caption:
            return jsonify({'error': 'Failed to generate image caption'}), 500
        print(f"Generated caption: {caption}")
        
        # Extract brand names using spaCy
        doc = nlp(caption)
        brand_names = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT']]
        brand_names = ' '.join(brand_names)
        print(f"Extracted brand names: {brand_names}")
            
        print("Generating embedding...")
        # Generate embedding and get results from database
        query_embedding = generate_embedding(query_image_path=image_path)
        print(f"Generated embedding shape: {query_embedding.shape}")
        
        print("Performing hybrid search...")
        # Get hybrid search results
        pids, scores = hybrid_search(
            query=brand_names,  # Use the brand names as the text query
            query_embedding=query_embedding,
            top_k=10,
            text_weight=0.3,
            image_weight=0.5,
            ts_weight=0.2
        )
        
        response = {
            'results': format_results(pids, scores, df),
            'caption': caption
        }
        print(f"Response: {response}")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in search_by_image: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running and data is loaded"""
    try:
        # Check if data is loaded
        if df is None:
            return jsonify({'status': 'error', 'message': 'Data not loaded'}), 500
        
        # Check if we can access the data
        sample_size = min(5, len(df))
        sample_data = df.head(sample_size).to_dict('records')
        
        return jsonify({
            'status': 'healthy',
            'message': 'API is running and data is loaded',
            'data_sample': sample_data
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Health check failed: {str(e)}'
        }), 500

# Set port to 5001
if __name__ == '__main__':
    app.run(port=5001) 