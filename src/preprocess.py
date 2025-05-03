import numpy as np
import torch
import requests
import os
from PIL import Image
from io import BytesIO
from transformers import AutoModel, AutoProcessor, BlipProcessor, BlipForConditionalGeneration

# Initialize device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

# Global variables for model instances
_siglip_processor = None
_siglip_model = None
_blip_processor = None
_blip_model = None

def initialize_clip_model(model_id="openai/clip-vit-base-patch32"):
    """Initialize CLIP model and processor"""
    global _siglip_processor, _siglip_model
    _siglip_processor = AutoProcessor.from_pretrained(model_id)
    _siglip_model = AutoModel.from_pretrained(model_id).to(device)
    return _siglip_processor, _siglip_model

def initialize_blip_model(model_id="Salesforce/blip-image-captioning-base"):
    """Initialize BLIP model and processor"""
    global _blip_processor, _blip_model
    _blip_processor = BlipProcessor.from_pretrained(model_id)
    _blip_model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    return _blip_processor, _blip_model

def get_clip_model():
    """Get or initialize CLIP model and processor"""
    global _siglip_processor, _siglip_model
    if _siglip_processor is None or _siglip_model is None:
        return initialize_clip_model()
    return _siglip_processor, _siglip_model

def get_blip_model():
    """Get or initialize BLIP model and processor"""
    global _blip_processor, _blip_model
    if _blip_processor is None or _blip_model is None:
        return initialize_blip_model()
    return _blip_processor, _blip_model

def get_text_embedding(text):
    """Get embedding for text input"""
    processor, model = get_clip_model()
    max_length = 77  # CLIP's max token length

    with torch.no_grad():
        inputs = processor(
            text=text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)
        text_features = model.get_text_features(**inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()

def get_image_embedding(image_path):
    """Generate embedding for image from local file or URL"""
    processor, model = get_clip_model()
    try:
        if image_path.startswith(('http://', 'https://')):
            # Load image from URL
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            # Load image from local file
            image = Image.open(image_path)
        
        # Preprocess and encode image
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def generate_image_caption(image_path):
    """Generate caption for image using BLIP"""
    processor, model = get_blip_model()
    try:
        if image_path.startswith(('http://', 'https://')):
            # Load image from URL
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            # Load image from local file
            image = Image.open(image_path)
        
        # Generate caption
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50)
            caption = processor.decode(out[0], skip_special_tokens=True)
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