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
import time


app = Flask(__name__)

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

top_k = 100

components_config = [
    {
        "type": "PostgresVectorRetrieval",
        "params": {
            "column_name": "fusion_embedding"
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
            "rank_method": "ts_rank"
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


@app.route('/api/search', methods=['POST'])
def search():
    try:
        start_time = time.time()  # start the timer
        print("Received search request")
        print(f"Form data: {request.form}")
        print(f"Files: {request.files}")
        query_text = None
        query_image = None
        query_embedding = None

        # Handle text query from form data
        query_text = request.form.get('query')
        print(f"Query text: {query_text}")

        # Handle image query
        if 'file' in request.files:
            # uploaded image
            file = request.files['file']
            query_image = Image.open(file.stream)
        elif request.form.get('image_path'):
            # image from url
            image_path = request.form.get('image_path')
            query_image = load_image(image_path)

        if not query_text and not query_image:
            return jsonify({'error': 'Either query text or image is required'}), 400

        print("Generating embedding...")
        # Generate embedding based on available inputs
        if query_text and query_image:
            query_embedding = generate_embedding(query_text=query_text, query_image=query_image)
        elif query_text:
            query_embedding = generate_embedding(query_text=query_text)
        else:  # query_image only
            query_embedding = generate_embedding(query_image=query_image)

        print(f"Generated embedding shape: {query_embedding.shape}")
        
        print("Performing retrieval...")
        # Adjust weights based on input type
        # Weights: [fusion_embedding, image_clip_embedding, text_search]
        if query_text and query_image:
            # Text+Image search: Use fusion_embedding (CLIP image + MiniLM text) and text search
            weights = [0.8, 0, 0.2]  # 80% fusion embedding, 20% text search
        elif query_text:
            # Text-only search: Use fusion_embedding (CLIP text + MiniLM text) and text search
            weights = [0.8, 0, 0.2]  # 80% fusion embedding, 20% text search
        else:
            # Image-only search: Use only image_clip_embedding
            weights = [0, 1, 0]  # 100% image CLIP embedding

        components = [create_retrieval_component(c, DB_CONFIG) for c in components_config]
        search_query = query_text if query_text else ""
        pids, scores = hybrid_retrieval(
            query=search_query,
            query_embedding=query_embedding,
            components=components, 
            weights=weights,
            top_k=top_k
        )

        elapsed_time = time.time() - start_time  # calculate the time
        response = {
            'results': format_results(pids, scores)
            'elapsed_time_sec': round(elapsed_time, 3)
        }
        print(f"Response: {response}")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in search: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# Set port to 5001
if __name__ == "__main__":
    # For Google Cloud Run compatibility
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
