import pandas as pd
import requests
import os
import sys
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
import spacy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.backend.embedding import generate_embedding, generate_image_caption
from src.backend.retrieval import hybrid_retrieval, create_retrieval_component
from src.backend.db import fetch_product_by_pid
from config.db import DB_CONFIG


app = Flask(__name__)

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

top_k = 10

components_config = [
    {
        "type": "PostgresVectorRetrieval",
        "params": {
            "column_name": "text_clip_embedding"
        }
    },
    {
        "type": "PostgresVectorRetrieval",
        "params": {
            "column_name": "image_clip_embedding"
        }
    },
    {
        "type": "TextSearchRetrieval",
        "params": {
            "rank_method": "ts_rank_cd"
        }
    }
]

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

def format_results(indices, scores):
    results = []
    for idx, score in zip(indices, scores):
        try:
            # Look up product by ID instead of using iloc
            product = fetch_product_by_pid(idx)
            if product:
                results.append({
                    'Pid': str(idx),
                    'Name': str(product['Name']),
                    'Description': str(product['Description']),
                    'Brand': str(product['Brand']),
                    'Category': str(product['Category']),
                    'Color': str(product['Color']),
                    'Gender': str(product['Gender']),
                    'Size': str(product['Size']),
                    'similarity': float(score)
                })
        except IndexError:
            # Skip products that aren't found in the Database
            print(f"Warning: Product ID {idx} not found in Database")
            continue
    return results


@app.route("/")
def index():
    return "Backend API is running!"


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
        weights = [0.5, 0.3, 0.2]
        components = [create_retrieval_component(c, DB_CONFIG) for c in components_config]
        pids, scores = hybrid_retrieval(
            query=query_text,
            query_embedding=query_embedding,
            components=components, 
            weights=weights,
            top_k=top_k
        )
        response = {
            'results': format_results(pids, scores)
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
        if 'file' in request.files:
            # uploaded image
            file = request.files['file']
            image = Image.open(file.stream)
        else:
            # image from url
            image_path = request.form.get('image_path')
            if not image_path:
                return jsonify({'error': 'Image path or file required'}), 400
            image = load_image(image_path)

        if not image:
            return jsonify({'error': 'Failed to load image'}), 400

        print("Generating image caption...")
        # Generate caption using BLIP
        caption = generate_image_caption(image)
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
        query_embedding = generate_embedding(query_image=image)
        print(f"Generated embedding shape: {query_embedding.shape}")
        
        print("Performing hybrid search...")
        # Get hybrid search results
        weights = [0.3, 0.5, 0.2]
        components = [create_retrieval_component(c, DB_CONFIG) for c in components_config]
        pids, scores = hybrid_retrieval(
            query=brand_names,
            query_embedding=query_embedding,
            components=components, 
            weights=weights,
            top_k=top_k
        )
        
        response = {
            'results': format_results(pids, scores),
            'caption': caption
        }
        print(f"Response: {response}")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in search_by_image: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# Set port to 5001
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
