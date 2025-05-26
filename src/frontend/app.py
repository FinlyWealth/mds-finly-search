import os
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64
import pandas as pd
import time

# Set page config
st.set_page_config(
    page_title="ProductSearch",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;  height: 0rem;}
        header {visibility: hidden; height: 0rem;}
        .stMain .stMainBlockContainer {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .loading-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            background-color: #f0f2f6;
            border-radius: 10px;
            margin: 2rem 0;
        }
        .product-image-container {
            width: 100%;
            height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            margin-bottom: 1rem;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }
        .product-image-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }
        [data-testid="stSidebarHeader"] {
            padding: 4px;
        }
        .stHeading [data-testid="stMarkdownContainer"] h2 {
            padding-top: 0px;
        }
        [data-testid="stSidebarCollapsedControl"] {
            top: 0.2rem;
        }
    </style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5001")

def get_component_description(component):
    """Get description for each component"""
    descriptions = {
        "minilm_model": "Text embedding model for semantic search",
        "clip_model": "Vision-language model for image and text understanding",
        "faiss_indices": "Vector search indices for fast similarity search",
        "database": "Product database connection and tables"
    }
    return descriptions.get(component, "")

def display_loading_screen(components):
    """Display a detailed loading screen with component status"""
    st.markdown("""
        <div class="loading-container">
            <h2>🚀 Initializing Search Engine</h2>
            <p>Please wait while we load all required components...</p>
    """, unsafe_allow_html=True)
    
    for component, status in components.items():
        icon = "✅" if status else "⏳"
        status_class = "success" if status else "pending"
        description = get_component_description(component)
        
        st.markdown(f"""
            <div class="component-status">
                <span class="status-icon">{icon}</span>
                <div class="status-text">
                    <div class="status-label">{component.replace('_', ' ').title()}</div>
                    <div class="status-description">{description}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def check_api_ready():
    """Check if the API is ready to accept requests"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/ready")
        if response.status_code == 200:
            return response.json()["ready"]
        return False
    except Exception as e:
        return False

def wait_for_api_ready():
    """Wait for the API to be ready"""
    # Create an empty container for the loading message
    loading_container = st.empty()
    
    # Show loading message once
    with loading_container:
        st.markdown("""
            <div class="loading-container">
                <h2>🚀 Initializing Search Engine</h2>
                <p>Please wait while we get everything ready...</p>
            </div>
        """, unsafe_allow_html=True)
    
    while True:
        if check_api_ready():
            loading_container.empty()  # Clear the loading message
            st.success("✨ Search engine is ready!")
            time.sleep(1)  # Show success message briefly
            st.rerun()  # Refresh the page
            return True
        
        time.sleep(1)

# Function to load image from URL or local path
def load_image(image_path):
    try:
        if image_path.startswith(('http://', 'https://')):
            # Load image from URL
            headers = {
                "User-Agent": "Mozilla/5.0"
            }
            response = requests.get(image_path, headers=headers)
            image = Image.open(BytesIO(response.content))
        else:
            # Load image from local file
            image = Image.open(image_path)
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Function to display a product card
def display_product_card(product, score):
    with st.container():
        # Product image
        try:
            # Use GCS URL
            image_url = f"https://storage.googleapis.com/finly-mds/images/{product['Pid']}.jpeg"
            st.markdown(f"""
                <div class="product-image-container">
                    <img src="{image_url}" alt="{product['Name']}">
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.write(f"Image error: {str(e)}")
            st.write("No image available")
        
        st.write(f"**{product['Name']}**")
        st.write(f"Price: ${product['Price']}")

        # Product details with similarity score and feedback buttons in one line
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.write(f"Similarity: {score*100:.1f}%")
        
        # Initialize feedback state for this product if not exists
        feedback_key = f"feedback_{product['Pid']}"
        if feedback_key not in st.session_state:
            st.session_state[feedback_key] = None
        
        # Up button
        with col2:
            up_button = st.button(
                "👍", 
                key=f"up_{product['Pid']}", 
                use_container_width=True,
                type="primary" if st.session_state[feedback_key] == True else "secondary"
            )
            if up_button:
                submit_feedback(product['Pid'], True)
                st.session_state[feedback_key] = True
                st.rerun()
        
        # Down button
        with col3:
            down_button = st.button(
                "👎", 
                key=f"down_{product['Pid']}", 
                use_container_width=True,
                type="primary" if st.session_state[feedback_key] == False else "secondary"
            )
            if down_button:
                submit_feedback(product['Pid'], False)
                st.session_state[feedback_key] = False
                st.rerun()
            
        # Make description and details collapsible
        with st.expander("Show Details", expanded=False):
            st.write(f"Description: {product['Description']}")
            st.write(f"Brand: {product['Brand']}")
            st.write(f"Category: {product['Category']}")
            st.write(f"Color: {product['Color']}")
            st.write(f"Gender: {product['Gender']}")
            st.write(f"Size: {product['Size']}")
            st.write(f"ID: {product['Pid']}")

def submit_feedback(pid, feedback):
    """Submit user feedback to the API"""
    try:
        # Get current search query and image path from session state
        query_text = st.session_state.get('query_text', '')
        image_input = st.session_state.get('image_input', '')
        
        # Get session_id from search results
        search_results = st.session_state.get('search_results', {})
        session_id = search_results.get('session_id')
        
        if not session_id:
            st.error("No active search session found. Please perform a search first.")
            return
        
        # Prepare request data
        data = {
            'pid': pid,
            'feedback': feedback,
            'query_text': query_text,
            'image_path': image_input,
            'session_id': session_id
        }
        
        # Send feedback to API
        response = requests.post(
            f"{API_BASE_URL}/api/feedback",
            json=data
        )
        
        if response.status_code != 200:
            st.error(f"Error submitting feedback: {response.json().get('error', 'Unknown error')}")
            
    except Exception as e:
        st.error(f"Error submitting feedback: {str(e)}")

# Main app
def main():
    # Check if API is ready
    if not check_api_ready():
        wait_for_api_ready()
    
    # Create sidebar for search and statistics
    with st.sidebar:
        st.header("🔍 Product Search")
        st.write("Search for similar products using text or image")
        
        # Text input
        query_text = st.text_input(
            "Enter your search query", 
            placeholder="e.g., comfortable running shoes",
            key="query_text",
            on_change=lambda: st.session_state.update({"trigger_search": True}) if st.session_state.query_text else None
        )
        
        # Combined image input
        image_input = st.text_input(
            "Enter image path or URL", 
            placeholder="e.g., /path/to/image.jpg or https://example.com/image.jpg",
            key="image_input",
            on_change=lambda: st.session_state.update({"trigger_search": True}) if st.session_state.image_input else None
        )
        
        # Add search button
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        # Handle search when button is clicked or enter is pressed
        if (search_button or st.session_state.get("trigger_search", False)) and (query_text or image_input):
            # Reset the trigger
            if "trigger_search" in st.session_state:
                del st.session_state.trigger_search

            st.session_state.num_results_to_show = 20 # reset display count on new search

            st.session_state['search_start_time'] = time.time() # set the start time
            with st.spinner("Searching for similar products..."):
                try:
                    # Prepare the request data
                    request_data = {}
                    files = {}
                    
                    if query_text:
                        request_data["query"] = query_text
                    
                    if image_input:
                        image = load_image(image_input)
                        if image:
                            request_data["image_path"] = image_input
                            # Display the input image
                            st.image(image, caption="Input Image", use_container_width=True)

                    request_data["top_k"] = 100  # backend will return 100 results
                    
                    # Call API endpoint
                    response = requests.post(
                        f"{API_BASE_URL}/api/search",
                        files=files if files else None,
                        data=request_data,
                        headers={'Content-Type': 'application/x-www-form-urlencoded'}
                    )
                    
                    if response.status_code == 200:
                        try:
                            results = response.json()
                            # Store results for display in right column
                            st.session_state.search_results = results
                                
                            # Display the generated caption if available
                            if 'caption' in results:
                                st.write(f"**Generated Caption:** {results['caption']}")
                        except ValueError as e:
                            st.error(f"Invalid JSON response from API: {str(e)}")
                            st.error(f"Response content: {response.text[:200]}...")  # Show first 200 chars of response
                    elif response.status_code == 503:
                        st.error("API is not ready yet. Please wait for initialization to complete.")
                        # Wait for API to be ready
                        wait_for_api_ready()
                    else:
                        try:
                            error_msg = response.json().get('error', 'Unknown error')
                        except ValueError:
                            error_msg = f"Error {response.status_code}: {response.text[:200]}..."
                        st.error(f"Error from API: {error_msg}")
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
                    
        # Display the time and statistics 
        if 'search_results' in st.session_state:
            results = st.session_state.search_results

            if 'elapsed_time_sec' in results:
                st.markdown(f"🕒 **Search Time:** {results['elapsed_time_sec']} seconds")

            if 'search_start_time' in st.session_state:
                stats_render_end = time.time()   # get the end time
                total_elapsed = round(stats_render_end - st.session_state['search_start_time'], 3)
                st.markdown(f"🕒 **Total Time (click → display):** {total_elapsed} seconds")

            if 'results' in results and isinstance(results['results'], list):
                if 'category_distribution' in results:
                    st.subheader("📊 Category Distribution")
                    cat_df = pd.DataFrame.from_dict(results['category_distribution'], 
                                                orient='index', columns=['%'])
                    cat_df = cat_df.sort_values(by='%', ascending=False)
                    cat_df.index.name = "Category"
                    st.table(cat_df.style.format({'%': '{:.2f}'}))

                if 'brand_distribution' in results:
                    st.subheader("🏷️ Brand Distribution")
                    brand_df = pd.DataFrame.from_dict(results['brand_distribution'], 
                                                  orient='index', columns=['%'])
                    brand_df = brand_df.sort_values(by='%', ascending=False)
                    brand_df.index.name = "Brand"
                    st.table(brand_df.style.format({'%': '{:.2f}'}))
    
    # Main content area for search results
    st.header("Search Results")
    
    # Display search results if available
    if hasattr(st.session_state, 'search_results'):
        results = st.session_state.search_results
        num_total = len(results['results'])  # get total result count
        num_show = st.session_state.get('num_results_to_show', 20)  # dynamic result count

        st.caption(f"Showing {min(num_show, num_total)} of {num_total} results")  #  progress indicator
        
        # Create four rows of 5 columns each for the products
        for row in range((num_show + 4) // 5):  # dynamic rows
            cols = st.columns(5)  # Create 5 columns for each row
            for col, product in zip(cols, results['results'][row*5:(row+1)*5]):
                with col:
                    display_product_card(product, product['similarity'])

        if num_show < num_total and st.button("Show more results"):
            st.session_state.num_results_to_show += 20

if __name__ == "__main__":
    main() 