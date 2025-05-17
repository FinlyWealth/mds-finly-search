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
import multiprocessing
from functools import partial
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

def concatenate_embeddings(image_embeddings, text_embeddings):
    """
    Concatenate image CLIP embeddings and MiniLM embeddings.
    
    Args:
        image_embeddings (np.ndarray): Array of image CLIP embeddings
        text_embeddings (np.ndarray): Array of MiniLM embeddings
        
    Returns:
        np.ndarray: Concatenated embeddings
    """
    return np.concatenate([image_embeddings, text_embeddings], axis=1)

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

def process_batch(batch_df, batch_num, output_zip_path, total_batches, image_dir="data/images"):
    """
    Process a single batch of images and create a zip file.
    
    Args:
        batch_df (pd.DataFrame): DataFrame containing the batch of images to process
        batch_num (int): Current batch number
        output_zip_path (str): Base path where the zip files should be saved
        total_batches (int): Total number of batches being processed
        image_dir (str): Directory containing the product images
    
    Returns:
        tuple: (successful_copies, failed_copies, batch_zip_path)
    """
    try:
        batch_zip_path = output_zip_path.replace('.zip', f'_batch_{batch_num + 1}.zip')
        print(f"Starting batch {batch_num + 1}/{total_batches}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            successful_copies = 0
            failed_copies = 0
            
            # Copy existing images to temp directory
            for idx, pid in enumerate(batch_df['Pid']):
                if idx % 1000 == 0:  # Print progress every 1000 images
                    print(f"Batch {batch_num + 1}: Processing image {idx}/{len(batch_df)}")
                
                src_path = os.path.join(image_dir, f"{pid}.jpeg")
                dst_path = os.path.join(temp_dir, f"{pid}.jpeg")
                try:
                    if os.path.exists(src_path):
                        shutil.copy2(src_path, dst_path)
                        successful_copies += 1
                    else:
                        failed_copies += 1
                except Exception as e:
                    print(f"Error copying {src_path}: {str(e)}")
                    failed_copies += 1
            
            if successful_copies > 0:
                # Create zip file only if we have some successful copies
                print(f"Batch {batch_num + 1}: Creating zip file with {successful_copies} images")
                shutil.make_archive(
                    batch_zip_path.replace('.zip', ''),  # Remove .zip as make_archive adds it
                    'zip',
                    temp_dir
                )
                print(f"Completed batch {batch_num + 1}/{total_batches}")
            else:
                print(f"Batch {batch_num + 1}: No valid images found")
                
            return successful_copies, failed_copies, batch_zip_path
    except Exception as e:
        print(f"Error in batch {batch_num + 1}: {str(e)}")
        return 0, len(batch_df), None

def zip_product_images(df, output_zip_path="product_images.zip", batch_size=100000, image_dir="data/images"):
    """
    Creates multiple zip files containing product images from the filtered DataFrame,
    with each zip file containing up to batch_size images. Processes batches in parallel.
    
    Args:
        df (pd.DataFrame): Filtered DataFrame containing valid 'Pid' values
        output_zip_path (str): Base path where the zip files should be saved
        batch_size (int): Number of images per zip file (default: 100,000)
        image_dir (str): Directory containing the product images
    """
    # Verify image directory exists
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")
    
    # Calculate number of batches needed
    total_images = len(df)
    num_batches = (total_images + batch_size - 1) // batch_size
    
    print(f"Processing {total_images} images in {num_batches} batches...")
    print(f"Looking for images in: {os.path.abspath(image_dir)}")
    
    # Create batches
    batches = []
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_images)
        batch_df = df.iloc[start_idx:end_idx]
        batches.append((batch_df, batch_num))
    
    # Determine number of processes (use 75% of available CPU cores)
    num_processes = max(1, int(multiprocessing.cpu_count() * 0.75))
    print(f"Using {num_processes} processes")
    
    # Create a partial function with fixed arguments
    process_func = partial(
        process_batch,
        output_zip_path=output_zip_path,
        total_batches=num_batches,
        image_dir=image_dir
    )
    
    # Process batches in parallel with timeout
    try:
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Set a timeout of 30 minutes per batch
            results = pool.starmap(process_func, batches)
    except Exception as e:
        print(f"Error during parallel processing: {str(e)}")
        pool.terminate()
        raise
    
    # Print summary
    total_successful = sum(r[0] for r in results)
    total_failed = sum(r[1] for r in results)
    print(f"\nProcessing complete!")
    print(f"Total images processed: {total_images}")
    print(f"Successfully copied: {total_successful}")
    print(f"Missing images: {total_failed}")

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

    # Load CLIP model for images
    clip_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    # Load MiniLM model for text
    minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    minilm_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    minilm_model.eval()

    # Calculate number of chunks needed
    num_chunks = (len(filtered_df) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_num in range(num_chunks):
        start_idx = chunk_num * CHUNK_SIZE
        end_idx = min((chunk_num + 1) * CHUNK_SIZE, len(filtered_df))
        chunk_df = filtered_df.iloc[start_idx:end_idx]
        
        print(f"\nProcessing chunk {chunk_num + 1}/{num_chunks} (rows {start_idx}-{end_idx})")
        
        # Calculate image CLIP embeddings
        print("Generating image CLIP embeddings...")
        image_embeddings, valid_indices = calculate_image_clip_embeddings(
            chunk_df,
            model=clip_model,
            processor=clip_processor,
            device=device
        )
        
        # Calculate MiniLM embeddings for the same valid indices
        print("Generating MiniLM embeddings...")
        text_embeddings, product_ids = calculate_minilm_embeddings(
            chunk_df,
            model=minilm_model,
            tokenizer=minilm_tokenizer,
            device=device,
            valid_indices=valid_indices
        )
        
        # Concatenate embeddings
        print("Concatenating embeddings...")
        fusion_embeddings = concatenate_embeddings(image_embeddings, text_embeddings)
        
        # Save chunk embeddings
        save_path = os.path.join(EMBEDDINGS_PATH, "fusion_embeddings.npz")
        save_embeddings(
            embeddings=fusion_embeddings,
            product_ids=product_ids,
            embedding_type="fusion_clip_minilm",
            save_path=save_path,
            chunk_num=chunk_num
        )

    # Uncomment this if you want to zip a subset of product images from the full dataset
    # zip_product_images(filtered_df)

if __name__ == "__main__":
    main() 