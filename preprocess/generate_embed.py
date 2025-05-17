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

CHUNK_SIZE = 500_000  # Process 500k rows at a time

def save_embeddings(embeddings, product_ids, embedding_type, save_path, chunk_num=None):
    """
    Save embeddings and product IDs to a numpy compressed file.
    
    Args:
        embeddings (np.ndarray): Array of embeddings
        product_ids (np.ndarray): Array of product IDs
        embedding_type (str): Type of embeddings
        save_path (str): Path where the embeddings should be saved
        chunk_num (int, optional): Chunk number for large datasets
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Modify save path to include chunk number if provided
    if chunk_num is not None:
        base_path = os.path.splitext(save_path)[0]
        save_path = f"{base_path}_chunk_{chunk_num}.npz"
    
    np.savez(save_path, 
             embeddings=np.array(embeddings),
             product_ids=np.array(product_ids),
             embedding_type=embedding_type)
    print(f"\nSaved {embedding_type} embeddings to {save_path}")

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
            # Get the model output and use the last hidden state
            outputs = model(**inputs)
            # Use the [CLS] token embedding (first token) as the sentence embedding
            batch_features = outputs.last_hidden_state[:, 0, :]
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
            
        sentence_embeddings.extend(batch_features.cpu().numpy())
        product_ids.extend(filtered_ids[i:i+batch_size])
        print(f"\rSentence embedding batch {batch_num}/{total_batches} processed", end='', flush=True)
    
    return sentence_embeddings, product_ids

def concatenate_embeddings():
    """Concatenate image_clip_embedding and minilm_embedding for the given DataFrame"""


def filter_valid_products(df):
    """
    Filters the DataFrame to only include rows with valid product names and existing images.
    
    Args:
        df (pd.DataFrame): DataFrame containing the 'Pid' and 'Name' columns
        
    Returns:
        pd.DataFrame: Filtered DataFrame containing only rows with valid products
    """
    # Get the list of Pids from the DataFrame
    pids = df['Pid'].tolist()
    
    # Track which Pids have images
    valid_pids = set()
    
    # Check for existing images
    for pid in pids:
        src_path = f"data/images/{pid}.jpeg"
        if os.path.exists(src_path):
            valid_pids.add(pid)
    
    # Filter DataFrame for valid Pids and non-null names
    filtered_df = df[
        (df['Pid'].isin(valid_pids)) & 
        (df['Name'].notna()) & 
        (df['Name'].str.strip() != '')
    ].copy()
    
    print(f"Final filtered DataFrame has {len(filtered_df)} rows")
    
    return filtered_df

def zip_product_images(df, output_zip_path="product_images.zip"):
    """
    Creates a zip file containing all product images from the filtered DataFrame.
    
    Args:
        df (pd.DataFrame): Filtered DataFrame containing valid 'Pid' values
        output_zip_path (str): Path where the zip file should be saved
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy existing images to temp directory
        for pid in df['Pid']:
            src_path = f"data/images/{pid}.jpeg"
            dst_path = os.path.join(temp_dir, f"{pid}.jpeg")
            shutil.copy2(src_path, dst_path)
        
        # Create zip file
        shutil.make_archive(
            output_zip_path.replace('.zip', ''),  # Remove .zip as make_archive adds it
            'zip',
            temp_dir
        )
        
        print(f"Created zip file: {output_zip_path}")

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    print(f"Device is {device}")

    # Load data
    df = pd.read_csv(METADATA_PATH)
    total_rows = len(df)
    print(f"Total rows in dataset: {total_rows}")

    # Filter for valid products
    filtered_df = filter_valid_products(df)
    print(f"Valid products after filtering: {len(filtered_df)}")

    # Process each enabled embedding type
    for embedding_type in get_enabled_embedding_types():
        config = EMBEDDING_TYPES[embedding_type]
        if not config['enabled']:
            print(f"\nSkipping {embedding_type} embeddings as it is disabled")
            continue
            
        print(f"\nProcessing {embedding_type} embeddings...")
        
        # Calculate number of chunks needed
        num_chunks = (len(filtered_df) + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        if embedding_type in ['text_clip', 'image_clip']:
            # Load CLIP model
            model = AutoModel.from_pretrained(config['model']).to(device)
            processor = AutoProcessor.from_pretrained(config['model'])
            model.eval()
            
            for chunk_num in range(num_chunks):
                start_idx = chunk_num * CHUNK_SIZE
                end_idx = min((chunk_num + 1) * CHUNK_SIZE, len(filtered_df))
                chunk_df = filtered_df.iloc[start_idx:end_idx]
                
                print(f"\nProcessing chunk {chunk_num + 1}/{num_chunks} (rows {start_idx}-{end_idx})")
                
                if embedding_type == 'image_clip':
                    # Calculate image embeddings for chunk
                    embeddings, valid_indices = calculate_image_clip_embeddings(
                        chunk_df,
                        model=model,
                        processor=processor,
                        device=device
                    )
                    product_ids = chunk_df['Pid'].iloc[valid_indices].tolist()
                else:  # text_clip
                    # Calculate text embeddings for chunk
                    embeddings, product_ids = calculate_text_clip_embeddings(
                        chunk_df,
                        model=model,
                        processor=processor,
                        device=device
                    )
                
                # Save chunk embeddings
                save_path = os.path.join(EMBEDDINGS_PATH, config['filename'])
                save_embeddings(
                    embeddings=embeddings,
                    product_ids=product_ids,
                    embedding_type=embedding_type,
                    save_path=save_path,
                    chunk_num=chunk_num
                )
                
        elif embedding_type == 'minilm':
            # Load MiniLM model
            model = AutoModel.from_pretrained(config['model']).to(device)
            tokenizer = AutoTokenizer.from_pretrained(config['model'])
            model.eval()
            
            for chunk_num in range(num_chunks):
                start_idx = chunk_num * CHUNK_SIZE
                end_idx = min((chunk_num + 1) * CHUNK_SIZE, len(filtered_df))
                chunk_df = filtered_df.iloc[start_idx:end_idx]
                
                print(f"\nProcessing chunk {chunk_num + 1}/{num_chunks} (rows {start_idx}-{end_idx})")
                
                # Calculate MiniLM embeddings for chunk
                embeddings, product_ids = calculate_minilm_embeddings(
                    chunk_df,
                    model=model,
                    tokenizer=tokenizer,
                    device=device
                )
                
                # Save chunk embeddings
                save_path = os.path.join(EMBEDDINGS_PATH, config['filename'])
                save_embeddings(
                    embeddings=embeddings,
                    product_ids=product_ids,
                    embedding_type=embedding_type,
                    save_path=save_path,
                    chunk_num=chunk_num
                )

    # Zip product images
    zip_product_images(filtered_df)

if __name__ == "__main__":
    main() 