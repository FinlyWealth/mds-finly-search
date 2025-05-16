import os
import streamlit as st
import requests
from PIL import Image
from io import BytesIO

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
    
    # Add radio buttons for search mode selection
    with left_col:
        st.header("🔍 Product Search")
        st.write("Search for similar products using text or image path/URL")
        search_mode = st.radio(
            "**Select Search Mode**",
            ["Text Search", "Image Search"],
            horizontal=True
        )
        
        # Clear search results when switching modes
        if 'prev_search_mode' not in st.session_state:
            st.session_state.prev_search_mode = search_mode
        elif st.session_state.prev_search_mode != search_mode:
            st.session_state.prev_search_mode = search_mode
            st.session_state.pop("text_results", None)
            st.session_state.pop("image_results", None)
            st.session_state.pop("query_text", None)
            st.session_state.pop("image_path", None)
            st.session_state.pop("uploaded_file", None)
    
    # Text Search
    with left_col:
        if search_mode == "Text Search":
            query_text = st.text_input(
                "Enter your search query", 
                placeholder="e.g., comfortable running shoes",
                key="query_text"
            )
            
            if query_text:
                with st.spinner("Searching for similar products..."):
                    try:
                        # Call API endpoint
                        response = requests.post(
                            f"{API_BASE_URL}/api/search/text",
                            json={"query": query_text}
                        )
                        
                        if response.status_code == 200:
                            results = response.json()
                            # Store results for display in right column
                            st.session_state.text_results = results
                        else:
                            st.error(f"Error from API: {response.json().get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error during search: {str(e)}")
    
    # Image Search
    with left_col:
        if search_mode == "Image Search":
            
            # Add file uploader
            uploaded_file = st.file_uploader(
                "Upload an image", 
                type=["jpg", "jpeg", "png"],
                key="uploaded_file"
            )
            
            # Keep existing text input for image paths/URLs
            image_path = st.text_input(
                "Or enter image path or URL", 
                placeholder="e.g., /path/to/image.jpg or https://example.com/image.jpg",
                key="image_path"
            )
            
            # Handle either uploaded file or image path
            if uploaded_file is not None or image_path:
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    image_path = None  # Clear image_path when using uploaded file
                else:
                    image = load_image(image_path)
                    
                if image:                   
                    with st.spinner("Searching for similar products..."):
                        try:
                            # Prepare the request data
                            request_data = {"image_path": image_path}
                            files = {}
                            if uploaded_file is not None:
                                # Convert image to bytes for upload
                                img_byte_arr = BytesIO()
                                image.save(img_byte_arr, format=image.format)
                                img_byte_arr = img_byte_arr.getvalue()
                                files = {"file": ("image.jpg", img_byte_arr, "image/jpeg")}
                                
                            # Call API endpoint
                            response = requests.post(
                                f"{API_BASE_URL}/api/search/image",
                                files=files if files else None,
                                data=request_data
                            )
                            
                            if response.status_code == 200:
                                results = response.json()
                                # Store results for display in right column
                                st.session_state.image_results = results
                                # Display the generated caption
                                if 'caption' in results:
                                    st.write(f"**Generated Caption:** {results['caption']}")
                            else:
                                st.error(f"Error from API: {response.json().get('error', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"Error during search: {str(e)}")

                    # Display input image
                    st.image(image, caption="Input Image", use_container_width=True)
    
    # Display results in the right column
    with right_col:
        st.header("Search Results")
        
        # Display text search results if available
        if hasattr(st.session_state, 'text_results'):
            results = st.session_state.text_results
            # Create fourt rows of 5 columns each for the products
            for row in range(4):
                cols = st.columns(5)  # Create 5 columns for each row
                for col, product in zip(cols, results['results'][row*5:(row+1)*5]):
                    with col:
                        display_product_card(product, product['similarity'])
        
        # Display image search results if available
        if hasattr(st.session_state, 'image_results'):
            results = st.session_state.image_results
            # Create two rows of 5 columns each for the products
            for row in range(2):
                cols = st.columns(5)  # Create 5 columns for each row
                for col, product in zip(cols, results['results'][row*5:(row+1)*5]):
                    with col:
                        display_product_card(product, product['similarity'])

if __name__ == "__main__":
    main() 