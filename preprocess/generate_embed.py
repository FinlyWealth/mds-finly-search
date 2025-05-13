import torch
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
import tempfile
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from tqdm.auto import tqdm
import pyarrow.parquet as pq
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.path import METADATA_PATH
from config.embeddings import EMBEDDINGS_PATH, EMBEDDING_TYPES, get_enabled_embedding_types


def save_embeddings(embeddings, product_ids, embedding_type, save_path):
    """
    Save embeddings and product IDs to a numpy compressed file.
    
    Args:
        embeddings (np.ndarray): Array of embeddings
        product_ids (np.ndarray): Array of product IDs
        embedding_type (str): Type of embeddings
        save_path (str): Path where the embeddings should be saved
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    np.savez(save_path, 
             embeddings=np.array(embeddings),
             product_ids=np.array(product_ids),
             embedding_type=embedding_type)
    print(f"Saved {embedding_type} embeddings to {save_path}")

def calculate_image_clip_embeddings(df, model, processor, device, batch_size=100):
    """
    Calculate image embeddings for the given DataFrame using CLIP model.
    
    Args:
        df (pd.DataFrame): DataFrame containing product information
        model: CLIP model for generating embeddings
        processor: CLIP processor for image preprocessing
        device: Device to run the model on
        batch_size (int): Size of batches for processing
        
    Returns:
        tuple: (image_embeddings, valid_indices)
    """
    image_embeddings = []
    valid_indices = []
    
    image_paths = [f"data/images/{pid}.jpeg" for pid in df['Pid'].tolist()]
    total_image_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    for batch_num, i in enumerate(range(0, len(image_paths), batch_size), 1):
        batch_images = []
        batch_valid_indices = []
        
        for idx, path in enumerate(image_paths[i:i+batch_size]):
            try:
                image = Image.open(path).convert("RGB")
                batch_images.append(image)
                batch_valid_indices.append(i + idx)
            except Exception as e:
                print(f"Skipping problematic image {path}: {e}")
        
        if batch_images:
            try:
                inputs = processor(
                    images=batch_images,
                    return_tensors="pt",
                    padding=True
                ).to(device)
                
                with torch.no_grad():
                    batch_features = model.get_image_features(**inputs)
                    batch_features /= batch_features.norm(dim=-1, keepdim=True)
                
                image_embeddings.extend(batch_features.cpu().numpy())
                valid_indices.extend(batch_valid_indices)
                
            except Exception as e:
                print(f"Error processing batch {batch_num}: {e}")
                continue
                
        print(f"\rImage embedding batch {batch_num}/{total_image_batches} processed", end='', flush=True)
    
    print(f"\nProcessed {len(valid_indices)} valid images out of {len(image_paths)} total images")
    return image_embeddings, valid_indices

def calculate_text_clip_embeddings(df, model, processor, device, valid_indices=None, batch_size=100):
    """
    Calculate text embeddings for the given DataFrame using CLIP model.
    
    Args:
        df (pd.DataFrame): DataFrame containing product information
        model: CLIP model for generating embeddings
        processor: CLIP processor for text preprocessing
        device: Device to run the model on
        valid_indices (list): Optional list of valid indices to process
        batch_size (int): Size of batches for processing
        
    Returns:
        tuple: (text_embeddings, product_ids)
    """
    text_embeddings = []
    product_ids = []
    
    if valid_indices is not None:
        texts = df['Name'].iloc[valid_indices].tolist()
        ids = df['Pid'].iloc[valid_indices].tolist()
    else:
        texts = df['Name'].tolist()
        ids = df['Pid'].tolist()
    
    # Filter out blank or NaN texts
    filtered_texts = []
    filtered_ids = []
    for text, pid in zip(texts, ids):
        if isinstance(text, str) and text.strip():
            filtered_texts.append(text)
            filtered_ids.append(pid)
        else:
            print(f"Skipping empty or invalid text for pid {pid}")
    
    total_text_batches = (len(filtered_texts) + batch_size - 1) // batch_size
    for batch_num, i in enumerate(range(0, len(filtered_texts), batch_size), 1):
        batch_texts = filtered_texts[i:i+batch_size]
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            batch_features = model.get_text_features(**inputs)
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
            
        text_embeddings.extend(batch_features.cpu().numpy())
        product_ids.extend(filtered_ids[i:i+batch_size])
        print(f"\rText embedding batch {batch_num}/{total_text_batches} processed", end='', flush=True)
    
    return text_embeddings, product_ids

def calculate_minilm_embeddings(df, model, tokenizer, device, valid_indices=None, batch_size=100):
    """
    Calculate sentence embeddings using MiniLM model for the given DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing product information
        model: Sentence transformer model for generating embeddings
        tokenizer: Tokenizer for text preprocessing
        device: Device to run the model on
        valid_indices (list): Optional list of valid indices to process
        batch_size (int): Size of batches for processing
        
    Returns:
        tuple: (sentence_embeddings, product_ids)
    """
    sentence_embeddings = []
    product_ids = []
    
    if valid_indices is not None:
        texts = df['Name'].iloc[valid_indices].tolist()
        ids = df['Pid'].iloc[valid_indices].tolist()
    else:
        texts = df['Name'].tolist()
        ids = df['Pid'].tolist()
    
    # Filter out blank or NaN texts
    filtered_texts = []
    filtered_ids = []
    for text, pid in zip(texts, ids):
        if isinstance(text, str) and text.strip():
            filtered_texts.append(text)
            filtered_ids.append(pid)
        else:
            print(f"Skipping empty or invalid text for pid {pid}")
    
    total_batches = (len(filtered_texts) + batch_size - 1) // batch_size
    for batch_num, i in enumerate(range(0, len(filtered_texts), batch_size), 1):
        batch_texts = filtered_texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            # MiniLM model directly outputs sentence embeddings
            batch_features = model(**inputs).sentence_embedding
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
            
        sentence_embeddings.extend(batch_features.cpu().numpy())
        product_ids.extend(filtered_ids[i:i+batch_size])
        print(f"\rSentence embedding batch {batch_num}/{total_batches} processed", end='', flush=True)
    
    return sentence_embeddings, product_ids

def filter_and_zip_product_images(df, output_zip_path="product_images.zip"):
    """
    Creates a zip file containing all product images that exist in the data/images directory.
    Returns a filtered DataFrame containing only rows where images exist.
    
    Args:
        df (pd.DataFrame): DataFrame containing the 'Pid' column
        output_zip_path (str): Path where the zip file should be saved
        
    Returns:
        pd.DataFrame: Filtered DataFrame containing only rows where images exist
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Get the list of Pids from the DataFrame
        pids = df['Pid'].tolist()
        
        # Track which Pids have images
        valid_pids = set()
        
        # Copy existing images to temp directory
        for pid in pids:
            src_path = f"data/images/{pid}.jpeg"
            if os.path.exists(src_path):
                dst_path = os.path.join(temp_dir, f"{pid}.jpeg")
                shutil.copy2(src_path, dst_path)
                valid_pids.add(pid)
        
        print(f"Found {len(valid_pids)} existing images out of {len(pids)} Pids")
        
        # Create zip file
        shutil.make_archive(
            output_zip_path.replace('.zip', ''),  # Remove .zip as make_archive adds it
            'zip',
            temp_dir
        )
        
        print(f"Created zip file: {output_zip_path}")
        
        # Return filtered DataFrame
        filtered_df = df[df['Pid'].isin(valid_pids)].copy()
        return filtered_df

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    print(f"Device is {device}")

    # Load data
    df = pd.read_csv(METADATA_PATH)

    # Identify rows with product images available and zip them
    filtered_df = filter_and_zip_product_images(df)

    # Process each enabled embedding type
    for embedding_type in get_enabled_embedding_types():
        config = EMBEDDING_TYPES[embedding_type]
        print(f"\nProcessing {embedding_type} embeddings...")
        
        if embedding_type in ['text_clip', 'image_clip']:
            # Load CLIP model
            model = AutoModel.from_pretrained(config['model']).to(device)
            processor = AutoProcessor.from_pretrained(config['model'])
            model.eval()
            
            if embedding_type == 'image_clip':
                # Calculate image embeddings first
                embeddings, valid_indices = calculate_image_clip_embeddings(
                    filtered_df,
                    model=model,
                    processor=processor,
                    device=device
                )
                product_ids = filtered_df['Pid'].iloc[valid_indices].tolist()
            else:  # text_clip
                # Calculate text embeddings for valid indices
                embeddings, product_ids = calculate_text_clip_embeddings(
                    filtered_df,
                    model=model,
                    processor=processor,
                    device=device,
                    valid_indices=valid_indices if 'valid_indices' in locals() else None
                )
                
        elif embedding_type == 'minilm':
            # Load MiniLM model
            model = AutoModel.from_pretrained(config['model']).to(device)
            tokenizer = AutoTokenizer.from_pretrained(config['model'])
            model.eval()
            
            # Calculate MiniLM embeddings
            embeddings, product_ids = calculate_minilm_embeddings(
                filtered_df,
                model=model,
                tokenizer=tokenizer,
                device=device,
                valid_indices=valid_indices if 'valid_indices' in locals() else None
            )
        
        # Save embeddings
        save_path = os.path.join(EMBEDDINGS_PATH, config['filename'])
        save_embeddings(
            embeddings=embeddings,
            product_ids=product_ids,
            embedding_type=embedding_type,
            save_path=save_path
        )

if __name__ == "__main__":
    main() 