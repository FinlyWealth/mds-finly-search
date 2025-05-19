import numpy as np
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoModel, AutoProcessor, BlipProcessor, BlipForConditionalGeneration, AutoTokenizer

# Initialize device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

# Global variables for model instances
_clip_processor = None
_clip_model = None
_blip_processor = None
_blip_model = None
_minilm_model = None
_minilm_tokenizer = None

def initialize_clip_model(model_id="openai/clip-vit-base-patch32"):
    """Initialize CLIP model and processor"""
    global _clip_processor, _clip_model
    _clip_processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    _clip_model = AutoModel.from_pretrained(model_id).to(device)
    return _clip_processor, _clip_model

def initialize_blip_model(model_id="Salesforce/blip-image-captioning-base"):
    """Initialize BLIP model and processor"""
    global _blip_processor, _blip_model
    _blip_processor = BlipProcessor.from_pretrained(model_id)
    _blip_model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    return _blip_processor, _blip_model

def initialize_minilm_model(model_id="sentence-transformers/all-MiniLM-L6-v2"):
    """Initialize MiniLM model and tokenizer"""
    global _minilm_model, _minilm_tokenizer
    _minilm_model = AutoModel.from_pretrained(model_id).to(device)
    _minilm_tokenizer = AutoTokenizer.from_pretrained(model_id)
    return _minilm_model, _minilm_tokenizer

def get_clip_model():
    """Get or initialize CLIP model and processor"""
    global _clip_processor, _clip_model
    if _clip_processor is None or _clip_model is None:
        return initialize_clip_model()
    return _clip_processor, _clip_model

def get_blip_model():
    """Get or initialize BLIP model and processor"""
    global _blip_processor, _blip_model
    if _blip_processor is None or _blip_model is None:
        return initialize_blip_model()
    return _blip_processor, _blip_model

def get_minilm_model():
    """Get or initialize MiniLM model and tokenizer"""
    global _minilm_model, _minilm_tokenizer
    if _minilm_model is None or _minilm_tokenizer is None:
        return initialize_minilm_model()
    return _minilm_model, _minilm_tokenizer

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
        return text_features.cpu().numpy()[0]

def get_image_embedding(image):
    """Generate embedding for image from local file or URL"""
    processor, model = get_clip_model()
    try:
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()[0]
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def get_minilm_embeddings(text):
    """Get embedding for text input using MiniLM"""
    model, tokenizer = get_minilm_model()
    
    # Process text with MiniLM
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the [CLS] token embedding (first token) as the sentence embedding
        minilm_embedding = outputs.last_hidden_state[:, 0, :]
        minilm_embedding /= minilm_embedding.norm(dim=-1, keepdim=True)
        return minilm_embedding.cpu().numpy()[0]

def generate_image_caption(image):
    """Generate caption for image using BLIP"""
    processor, model = get_blip_model()
    try:
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50)
            caption = processor.decode(out[0], skip_special_tokens=True)
            return caption
    except Exception as e:
        print(f"Error generating caption for image: {e}")
        return None

def generate_embedding(query_text=None, query_image=None):
    """
    Generate embedding for text or image query
    
    Args:
        query_text (str, optional): Text query to generate embedding for. If provided, query_image should be None.
        query_image (Image): Image file to generate embedding for. If provided, query_text should be None.
    
    Returns:
        numpy.ndarray: The embedding vector
    """

    if query_text is not None and query_image is not None:
        clip_embedding = get_image_embedding(query_image)
        minilm_embedding = get_minilm_embeddings(query_text)
    elif query_text is not None:
        clip_embedding = get_text_embedding(query_text)
        minilm_embedding = get_minilm_embeddings(query_text)
    else:
        clip_embedding = get_image_embedding(query_image)
        # Skip concatenation step and return CLIP embedding
        return clip_embedding
    
    # Concatenate the embeddings
    fusion_embedding = np.concatenate([clip_embedding, minilm_embedding])
    
    # Normalize the concatenated embedding
    fusion_embedding = fusion_embedding / np.linalg.norm(fusion_embedding)
    
    return fusion_embedding