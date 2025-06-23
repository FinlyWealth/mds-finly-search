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
import logging
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.backend.embedding import (
    generate_embedding,
    initialize_minilm_model,
    initialize_clip_model,
)
from src.backend.retrieval import (
    hybrid_retrieval,
    create_retrieval_component,
    reorder_search_results_by_relevancy,
)
from src.backend.db import fetch_products_by_pids
from collections import Counter
import config.db
from psycopg2.extras import Json

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Track initialization status
initialization_status = {
    "minilm_model": False,
    "clip_model": False,
    "faiss_indices": False,
    "database": False,
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
        },
    },
    {
        "type": "FaissVectorRetrieval",
        "params": {
            "column_name": "image_clip_embedding",
            "nprobe": 32
        }
    },
    {   "type": "TextSearchRetrieval", 
        "params": {
            "rank_method": "ts_rank"
        }
    },
]

def initialize_app():
    """Initialize all required components and update status"""
    global initialization_state, initialization_start_time, components, nlp
    initialization_start_time = datetime.now()

    try:
        # Create retrieval components with database config
        components = [
            create_retrieval_component(comp, config.db.DB_CONFIG) for comp in components_config
        ]
        # Initialize spaCy
        nlp = spacy.load("en_core_web_sm")

        # Print database connection details
        logger.info(f"Connecting to database: {config.db.DB_CONFIG['dbname']}")
        logger.info(f"Table: {config.db.TABLE_NAME}")

        # Print components configuration
        logger.info("\nInitialized retrieval components:")
        for i, comp in enumerate(components_config):
            logger.info(f"Component {i+1}:")
            logger.info(f"  Type: {comp['type']}")
            logger.info(f"  Params: {comp['params']}")
        logger.info("")

        # Initialize MiniLM model
        logger.info("Initializing MiniLM model...")
        initialize_minilm_model()
        initialization_status["minilm_model"] = True
        logger.info("✓ MiniLM model initialized successfully")

        # Initialize CLIP model
        logger.info("Initializing CLIP model...")
        initialize_clip_model()
        initialization_status["clip_model"] = True
        logger.info("✓ CLIP model initialized successfully")

        # Initialize FAISS indices (this happens automatically when creating components)
        logger.info("Initializing FAISS indices...")
        initialization_status["faiss_indices"] = True
        logger.info("✓ FAISS indices initialized successfully")

        # Test database connection
        logger.info("Testing database connection...")
        conn = psycopg2.connect(**config.db.DB_CONFIG)
        cur = conn.cursor()
        cur.execute(f"SELECT 1 FROM {config.db.TABLE_NAME} LIMIT 1")
        cur.close()
        conn.close()
        initialization_status["database"] = True
        logger.info("✓ Database connection successful")

        logger.info("\nAll components initialized successfully!")
        initialization_state = "ready"
        return True
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        logger.warning("Warning: Some components failed to initialize")
        # Reset all status flags to False on failure
        for key in initialization_status:
            initialization_status[key] = False
        initialization_state = "failed"
        return False


def load_image(image_path):
    try:
        if image_path.startswith(("http://", "https://")):
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
                results.append(
                    {
                        "Pid": str(idx),
                        "Name": str(product["Name"]),
                        "Description": str(product["Description"]),
                        "Brand": str(product["Brand"]),
                        "Category": str(product["Category"]),
                        "Color": str(product["Color"]),
                        "Gender": str(product["Gender"]),
                        "Size": str(product["Size"]),
                        "Price": (
                            str(product["Price"])
                            if product["Price"] is not None
                            else None
                        ),
                        "similarity": float(score),
                    }
                )
        except IndexError:
            # Skip products that aren't found in the Database
            logger.warning(f"Warning: Product ID {idx} not found in Database")
            continue
    return results


@app.route("/")
def index():
    return "Backend API is running!"


@app.route("/api/ready", methods=["GET"])
def ready():
    """Check if the API is ready to accept queries"""
    global initialization_state

    if initialization_state == "starting":
        elapsed_time = (datetime.now() - initialization_start_time).total_seconds()

        # Check if we're approaching the Cloud Run timeout
        if elapsed_time > WARNING_THRESHOLD:
            logger.warning(
                f"WARNING: Initialization has taken {elapsed_time:.1f} seconds. "
                f"Cloud Run will timeout in {CLOUD_RUN_TIMEOUT - elapsed_time:.1f} seconds."
            )

        # If we've exceeded the timeout, mark as failed
        if elapsed_time > CLOUD_RUN_TIMEOUT:
            logger.error("ERROR: Initialization exceeded Cloud Run timeout limit")
            initialization_state = "failed"
            for key in initialization_status:
                initialization_status[key] = False

    status = {
        "state": initialization_state,
        "ready": initialization_state == "ready",
        "components": initialization_status,
        "elapsed_seconds": (
            (datetime.now() - initialization_start_time).total_seconds()
            if initialization_start_time
            else 0
        ),
    }
    return jsonify(status)


@app.route("/api/search", methods=["POST"])
def search():
    # Check if API is ready
    if not all(initialization_status.values()):
        return (
            jsonify(
                {
                    "error": "API is not ready yet. Please wait for initialization to complete."
                }
            ),
            503,
        )

    try:
        start_time = time.time()  # start the timer
        logger.info("Received search request")
        logger.debug(f"Form data: {request.form}")
        logger.debug(f"Files: {request.files}")
        query_text = None
        query_image = None
        query_embedding = None

        # Generate a new session ID for this search
        session_id = str(uuid.uuid4())

        # Handle text query from form data
        query_text = request.form.get("query")
        logger.info(f"Query text: {query_text}")
        logger.info(f"Form data: {request.form}")
        logger.info(f"Files: {request.files}")

        # Handle image query
        if "file" in request.files:
            # uploaded image
            file = request.files["file"]
            logger.info(f"Found file in request: {file.filename}")
            query_image = Image.open(file.stream)
            logger.info("Successfully opened image from file")
        elif request.form.get("image_path"):
            # image from url
            image_path = request.form.get("image_path")
            query_image = load_image(image_path)
            logger.info(f"Loaded image from path: {image_path}")

        # Get search type
        search_type = request.form.get("search_type")
        logger.info(f"Initial search type from request: {search_type}")
        if not search_type:
            # Determine search type based on available inputs
            if query_text and query_image:
                search_type = "multimodal"
            elif query_image:
                search_type = "image"
            elif query_text:
                search_type = "text"
            logger.info(f"Determined search type from inputs: {search_type}")
        logger.info(f"Final search type: {search_type}")

        if not query_text and not query_image:
            logger.error("No query text or image found in request")
            return jsonify({"error": "Either query text or image is required"}), 400

        logger.info("Generating embedding...")
        # Generate embedding based on search type
        if search_type == "multimodal":
            query_embedding = generate_embedding(
                query_text=query_text, query_image=query_image
            )
        elif search_type == "image":
            query_embedding = generate_embedding(query_image=query_image)
        else:  # text search
            query_embedding = generate_embedding(query_text=query_text)

        logger.debug(f"Generated embedding shape: {query_embedding.shape}")

        logger.info("Performing retrieval...")
        # Adjust weights based on input type
        # Weights: [fusion_embedding, image_clip_embedding, text_search]
        if query_text and query_image:
            # Text+Image search: Use fusion_embedding (CLIP image + MiniLM text) and text search
            weights = config.db.SEARCH_WEIGHTS["hybrid"]
        elif query_text:
            # Text-only search: Use fusion_embedding (CLIP text + MiniLM text) and text search
            weights = config.db.SEARCH_WEIGHTS["text_only"]
        else:
            # Image-only search: Use only image_clip_embedding
            weights = config.db.SEARCH_WEIGHTS["image_only"] # 100% image CLIP embedding

        # Print active components with non-zero weights
        logger.info("\nActive retrieval components:")
        for comp, weight in zip(components_config, weights):
            if weight > 0:
                logger.info(f"  {comp['type']}:")
                logger.info(f"    Params: {comp['params']}")
                logger.info(f"    Weight: {weight}")
        logger.info("")

        search_query = query_text if query_text else ""
        pids, scores = hybrid_retrieval(
            query=search_query,
            query_embedding=query_embedding,
            components=components,
            weights=weights,
            top_k=top_k,
        )

        elapsed_time = time.time() - start_time  # calculate the time

        formatted_result = format_results(pids, scores)

        # Only use LLM to reorder results if we have text queries
        if query_text:
            if os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY"):
                reordered_result, reasoning = reorder_search_results_by_relevancy(
                    query_text, formatted_result, max_results=int(0.3*top_k)
                )
            else:
                reordered_result = formatted_result[:]
                reasoning = "No API key available, no LLM reordering performed"
        else:
            reordered_result = formatted_result[:]
            reasoning = "Image search only, no LLM reordering performed"

        # transfer the results into data frame for statistics purpose
        df = pd.DataFrame(reordered_result)

        if df.empty:
            response = {
                'results': [],
                'elapsed_time_sec': round(elapsed_time, 3),
                'session_id': session_id,
                'reasoning': reasoning
            }
            logger.debug(f"Response: {response}")
            return jsonify(response)
            
        # Transfer NaN to None
        df["Brand"] = df["Brand"].fillna("None") 
        df["Category"] = df["Category"].fillna("None")

        # Calculate the distribution of category and brand
        category_dist = (df['Category'].value_counts(normalize=True) * 100).round(0).astype(int).to_dict()
        brand_dist = (df['Brand'].value_counts(normalize=True) * 100).round(0).astype(int).to_dict()

        # Calculate price range and average price, excluding NaN
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        min_price = df['Price'].min(skipna=True)
        max_price = df['Price'].max(skipna=True)
        avg_price = df['Price'].mean(skipna=True)
        
        response = {
            'results': reordered_result,  # Use reordered results instead of original
            'elapsed_time_sec': round(elapsed_time, 3),
            'category_distribution': category_dist,   
            'brand_distribution': brand_dist, 
            'price_range': [round(min_price, 2), round(max_price, 2)],
            'average_price': round(avg_price, 2),
            'session_id': session_id,
            'reasoning': reasoning
        }
        logger.debug(f"Response: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """Handle user feedback for search results"""
    try:
        data = request.get_json()
        query_text = data.get("query_text")
        image_path = data.get("image_path")
        pid = data.get("pid")
        feedback = data.get("feedback")  # True for thumbs up, False for thumbs down
        session_id = data.get("session_id")  # Get session ID from request

        if not pid or feedback is None:
            return jsonify({"error": "Missing required fields"}), 400

        if not session_id:
            return jsonify({"error": "Missing session_id"}), 400

        try:
            # Store feedback in database
            conn = psycopg2.connect(**config.db.DB_CONFIG)
            cur = conn.cursor()

            # First check if a row exists for this session_id
            cur.execute(
                """
                SELECT feedback FROM user_feedback 
                WHERE session_id = %s
            """,
                (session_id,),
            )

            existing_row = cur.fetchone()

            if existing_row:
                # Update existing row by appending to feedback list
                existing_feedback = existing_row[0]
                existing_feedback.append({"pid": pid, "feedback": feedback})

                cur.execute(
                    """
                    UPDATE user_feedback 
                    SET feedback = %s
                    WHERE session_id = %s
                """,
                    (Json(existing_feedback), session_id),
                )
            else:
                # Create new row for this session
                feedback_list = [{"pid": pid, "feedback": feedback}]
                cur.execute(
                    """
                    INSERT INTO user_feedback (query_text, query_image, feedback, session_id)
                    VALUES (%s, %s, %s, %s)
                """,
                    (query_text, image_path, Json(feedback_list), session_id),
                )

            conn.commit()
            cur.close()
            conn.close()
        except (psycopg2.OperationalError, psycopg2.Error) as db_error:
            # Log the error but don't expose it to the client
            logger.error(f"Database error while storing feedback: {str(db_error)}")
            # Continue execution to return success response

        return jsonify({"success": True})

    except Exception as e:
        # Only return error for non-database related issues
        return jsonify({"error": str(e)}), 500


# Set port to 5001
if __name__ == "__main__":
    # For Google Cloud Run compatibility
    port = int(os.environ.get("PORT", 5001))

    # Debug: Print all registered routes
    logger.info("\nRegistered routes:")
    for rule in app.url_map.iter_rules():
        logger.info(f"  {rule.endpoint}: {rule.rule}")

    # Initialize the application
    if not initialize_app():
        logger.error("Fatal: Application initialization failed. Exiting...")
        sys.exit(1)

    app.run(host="0.0.0.0", port=port)
