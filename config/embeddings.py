import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base folder for embeddings
EMBEDDINGS_PATH = os.getenv('EMBEDDINGS_PATH', 'data/embeddings')

# Configure which embedding types to use
EMBEDDING_TYPES = {
    'text_clip': {
        'enabled': os.getenv('ENABLE_TEXT_CLIP', 'true').lower() == 'true',
        'filename': 'text_clip.npz',
        'model': os.getenv('TEXT_CLIP_MODEL', 'openai/clip-vit-base-patch32')
    },
    'image_clip': {
        'enabled': os.getenv('ENABLE_IMAGE_CLIP', 'true').lower() == 'true',
        'filename': 'image_clip.npz',
        'model': os.getenv('IMAGE_CLIP_MODEL', 'openai/clip-vit-base-patch32')
    },
    'minilm': {
        'enabled': os.getenv('ENABLE_MINILM', 'true').lower() == 'true',
        'filename': 'minilm.npz',
        'model': os.getenv('MINILM_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    }
}

def get_embedding_paths():
    """Get paths for all enabled embedding types"""
    return {
        name: os.path.join(EMBEDDINGS_PATH, config['filename'])
        for name, config in EMBEDDING_TYPES.items()
        if config['enabled']
    }

def get_enabled_embedding_types():
    """Get list of enabled embedding types"""
    return [name for name, config in EMBEDDING_TYPES.items() if config['enabled']] 