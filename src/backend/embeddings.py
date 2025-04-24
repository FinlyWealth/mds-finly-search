import numpy as np
import torch
import requests
import os
from PIL import Image
from io import BytesIO
from transformers import AutoModel, AutoProcessor, BlipProcessor, BlipForConditionalGeneration

# Initialize models and device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

# Initialize SigLIP model
siglip_model_id = "openai/clip-vit-base-patch32"
siglip_processor = AutoProcessor.from_pretrained(siglip_model_id)
siglip_model = AutoModel.from_pretrained(siglip_model_id).to(device)

# Initialize BLIP model
blip_model_id = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(blip_model_id)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_id).to(device)

def get_text_embedding(text):
    """Get embedding for text input"""
    with torch.no_grad():
        inputs = siglip_processor(text=text, return_tensors="pt").to(device)
        text_features = siglip_model.get_text_features(**inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()

def get_image_embedding(image_path):
    """Generate embedding for image from local file or URL"""
    try:
        if image_path.startswith(('http://', 'https://')):
            # Load image from URL
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            # Load image from local file
            image = Image.open(image_path)
        
        # Preprocess and encode image
        inputs = siglip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = siglip_model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def generate_image_caption(image_path):
    """Generate caption for image using BLIP"""
    try:
        if image_path.startswith(('http://', 'https://')):
            # Load image from URL
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            # Load image from local file
            image = Image.open(image_path)
        
        # Generate caption
        inputs = blip_processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_length=50)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            return caption
    except Exception as e:
        print(f"Error generating caption for image {image_path}: {e}")
        return None

def generate_embedding(query_text=None, query_image_path=None):
    """
    Generate embedding for text or image query
    Returns:
        numpy.ndarray: The embedding vector
    """
    if query_text is not None:
        query_embedding = get_text_embedding(query_text)
    else:
        query_embedding = get_image_embedding(query_image_path)
    
    # Both functions now return numpy arrays, so we just need to get the first element
    return query_embedding[0]