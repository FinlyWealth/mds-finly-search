import pandas as pd
import requests
import os
import sys
import uuid
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
import spacy
import psycopg2
import time
from datetime import datetime, timedelta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.backend.embedding import generate_embedding, initialize_minilm_model, initialize_clip_model
from src.backend.retrieval import hybrid_retrieval, create_retrieval_component
from src.backend.db import fetch_products_by_pids
from collections import Counter
from config.db import DB_CONFIG, TABLE_NAME
from psycopg2.extras import Json

# Track initialization status
initialization_status = {
    "minilm_model": False,
    "clip_model": False,
    "faiss_indices": False,
    "database": False
}

# Track initialization state
initialization_state = "starting"  # Can be: "starting", "ready", "failed"
initialization_start_time = None
CLOUD_RUN_TIMEOUT = 600  # 10 minutes
WARNING_THRESHOLD = 540  # 9 minutes

top_k = 100

components_config = [
    {
        "type": "FaissVectorRetrieval",
        "params": {
            "column_name": "fusion_embedding",
            "nprobe": 32
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

# Create retrieval components with database config
components = [create_retrieval_component(comp, DB_CONFIG) for comp in components_config]

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

def initialize_app():
    """Initialize all required components and update status"""
    global initialization_state, initialization_start_time
    initialization_start_time = datetime.now()
    
    try:
        # Print database connection details
        print(f"Connecting to database: {DB_CONFIG['dbname']}")
        print(f"Table: {TABLE_NAME}")
        
        # Print components configuration
        print("\nInitialized retrieval components:")
        for i, comp in enumerate(components_config):
            print(f"Component {i+1}:")
            print(f"  Type: {comp['type']}")
            print(f"  Params: {comp['params']}")
        print()
        
        # Initialize MiniLM model
        print("Initializing MiniLM model...")
        initialize_minilm_model()
        initialization_status["minilm_model"] = True
        print("✓ MiniLM model initialized successfully")
        
        # Initialize CLIP model
        print("Initializing CLIP model...")
        initialize_clip_model()
        initialization_status["clip_model"] = True
        print("✓ CLIP model initialized successfully")
        
        # Initialize FAISS indices (this happens automatically when creating components)
        print("Initializing FAISS indices...")
        initialization_status["faiss_indices"] = True
        print("✓ FAISS indices initialized successfully")
        
        # Test database connection
        print("Testing database connection...")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(f"SELECT 1 FROM {TABLE_NAME} LIMIT 1")
        cur.close()
        conn.close()
        initialization_status["database"] = True
        print("✓ Database connection successful")
        
        print("\nAll components initialized successfully!")
        initialization_state = "ready"
        return True
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        print("Warning: Some components failed to initialize")
        # Reset all status flags to False on failure
        for key in initialization_status:
            initialization_status[key] = False
        initialization_state = "failed"
        return False

# Initialize the application
if not initialize_app():
    print("Fatal: Application initialization failed. Exiting...")
    sys.exit(1)

app = Flask(__name__)

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
    # Fetch all products in a single batch
    products = fetch_products_by_pids(indices)
    
    for idx, score in zip(indices, scores):
        try:
            product = products.get(idx)
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
                    'Price': str(product['Price']) if product['Price'] is not None else None,
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


@app.route('/api/ready', methods=['GET'])
def ready():
    """Check if the API is ready to accept queries"""
    global initialization_state
    
    if initialization_state == "starting":
        elapsed_time = (datetime.now() - initialization_start_time).total_seconds()
        
        # Check if we're approaching the Cloud Run timeout
        if elapsed_time > WARNING_THRESHOLD:
            print(f"WARNING: Initialization has taken {elapsed_time:.1f} seconds. "
                  f"Cloud Run will timeout in {CLOUD_RUN_TIMEOUT - elapsed_time:.1f} seconds.")
        
        # If we've exceeded the timeout, mark as failed
        if elapsed_time > CLOUD_RUN_TIMEOUT:
            print("ERROR: Initialization exceeded Cloud Run timeout limit")
            initialization_state = "failed"
            for key in initialization_status:
                initialization_status[key] = False
    
    status = {
        "state": initialization_state,
        "ready": initialization_state == "ready",
        "components": initialization_status,
        "elapsed_seconds": (datetime.now() - initialization_start_time).total_seconds() if initialization_start_time else 0
    }
    return jsonify(status)


@app.route('/api/search', methods=['POST'])
def search():
    # Check if API is ready
    if not all(initialization_status.values()):
        return jsonify({'error': 'API is not ready yet. Please wait for initialization to complete.'}), 503
        
    try:
        start_time = time.time()  # start the timer
        print("Received search request")
        print(f"Form data: {request.form}")
        print(f"Files: {request.files}")
        query_text = None
        query_image = None
        query_embedding = None

        # Generate a new session ID for this search
        session_id = str(uuid.uuid4())

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
            weights = [0.5, 0, 0.5]
        elif query_text:
            # Text-only search: Use fusion_embedding (CLIP text + MiniLM text) and text search
            weights = [0.5, 0, 0.5]
        else:
            # Image-only search: Use only image_clip_embedding
            weights = [0, 1, 0]  # 100% image CLIP embedding

        # Print active components with non-zero weights
        print("\nActive retrieval components:")
        for comp, weight in zip(components_config, weights):
            if weight > 0:
                print(f"  {comp['type']}:")
                print(f"    Params: {comp['params']}")
                print(f"    Weight: {weight}")
        print()

        search_query = query_text if query_text else ""
        pids, scores = hybrid_retrieval(
            query=search_query,
            query_embedding=query_embedding,
            components=components, 
            weights=weights,
            top_k=top_k
        )

        elapsed_time = time.time() - start_time  # calculate the time

        # transfer the results into data frame for statistics purpose
        results = format_results(pids, scores)
        df = pd.DataFrame(results)

        # Transfer NaN to None
        df['Brand'] = df['Brand'].fillna('None')
        df['Category'] = df['Category'].fillna('None')

        # Calculate the distribution
        category_dist = (df['Category'].value_counts(normalize=True) * 100).round(2).to_dict()
        brand_dist = (df['Brand'].value_counts(normalize=True) * 100).round(2).to_dict()
        
        response = {
            'results': format_results(pids, scores),
            'elapsed_time_sec': round(elapsed_time, 3),
            'category_distribution': category_dist,   
            'brand_distribution': brand_dist, 
            'session_id': session_id
        }
        print(f"Response: {response}")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in search: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Handle user feedback for search results"""
    try:
        data = request.get_json()
        query_text = data.get('query_text')
        image_path = data.get('image_path')
        pid = data.get('pid')
        feedback = data.get('feedback')  # True for thumbs up, False for thumbs down
        session_id = data.get('session_id')  # Get session ID from request
        
        if not pid or feedback is None:
            return jsonify({'error': 'Missing required fields'}), 400
            
        if not session_id:
            return jsonify({'error': 'Missing session_id'}), 400
            
        try:
            # Store feedback in database
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            
            # First check if a row exists for this session_id
            cur.execute("""
                SELECT feedback FROM user_feedback 
                WHERE session_id = %s
            """, (session_id,))
            
            existing_row = cur.fetchone()
            
            if existing_row:
                # Update existing row by appending to feedback list
                existing_feedback = existing_row[0]
                existing_feedback.append({"pid": pid, "feedback": feedback})
                
                cur.execute("""
                    UPDATE user_feedback 
                    SET feedback = %s
                    WHERE session_id = %s
                """, (Json(existing_feedback), session_id))
            else:
                # Create new row for this session
                feedback_list = [{"pid": pid, "feedback": feedback}]
                cur.execute("""
                    INSERT INTO user_feedback (query_text, query_image, feedback, session_id)
                    VALUES (%s, %s, %s, %s)
                """, (query_text, image_path, Json(feedback_list), session_id))
            
            conn.commit()
            cur.close()
            conn.close()
        except (psycopg2.OperationalError, psycopg2.Error) as db_error:
            # Log the error but don't expose it to the client
            print(f"Database error while storing feedback: {str(db_error)}")
            # Continue execution to return success response
        
        return jsonify({'success': True})
        
    except Exception as e:
        # Only return error for non-database related issues
        return jsonify({'error': str(e)}), 500

# Set port to 5001
if __name__ == "__main__":
    # For Google Cloud Run compatibility
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
