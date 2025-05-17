import os
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64

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
    </style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5001")

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
            # Try GCS first
            image_url = f"https://storage.googleapis.com/finly-mds/images/{product['Pid']}.jpeg"
            st.image(image_url, use_container_width=True)
        except:
            try:
                # Fall back to local path
                local_path = f"data/images/{product['Pid']}.jpeg"
                st.image(local_path, use_container_width=True)
            except:
                st.write("No image available")
        
        # Product details
        st.write(f"**Similarity:** {score*100:.1f}%")
        st.write(f"**{product['Name']}**")
        st.write(f"**ID:** {product['Pid']}")
        st.write(f"Description: {product['Description']}")
        st.write(f"Brand: {product['Brand']}")
        st.write(f"Category: {product['Category']}")
        st.write(f"Color: {product['Color']}")
        st.write(f"Gender: {product['Gender']}")
        st.write(f"Size: {product['Size']}")

# Main app
def main():
    # Create two columns for search inputs and results
    left_col, right_col = st.columns([1, 2])
    
    with left_col:
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
                    else:
                        try:
                            error_msg = response.json().get('error', 'Unknown error')
                        except ValueError:
                            error_msg = f"Error {response.status_code}: {response.text[:200]}..."
                        st.error(f"Error from API: {error_msg}")
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
    
    # Display results in the right column
    with right_col:
        st.header("Search Results")
        
        # Display search results if available
        if hasattr(st.session_state, 'search_results'):
            results = st.session_state.search_results
            # Create four rows of 5 columns each for the products
            for row in range(4):
                cols = st.columns(5)  # Create 5 columns for each row
                for col, product in zip(cols, results['results'][row*5:(row+1)*5]):
                    with col:
                        display_product_card(product, product['similarity'])

if __name__ == "__main__":
    main() 