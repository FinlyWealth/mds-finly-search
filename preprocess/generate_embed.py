import torch
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
import tempfile
from PIL import Image
from transformers import AutoModel, AutoProcessor
from tqdm.auto import tqdm
import pyarrow.parquet as pq
import os

def save_embeddings(text_embeddings, image_embeddings, product_ids, save_path='embeddings.npz'):
    """
    Save embeddings and product IDs to a numpy compressed file.
    
    Args:
        text_embeddings (np.ndarray): Array of text embeddings
        image_embeddings (np.ndarray): Array of image embeddings
        product_ids (np.ndarray): Array of product IDs
        save_path (str): Path where the embeddings should be saved
    """
    np.savez(save_path, 
             text_embeddings=np.array(text_embeddings),
             image_embeddings=np.array(image_embeddings),
             product_ids=np.array(product_ids))
    print(f"Saved embeddings to {save_path}")

def calculate_embeddings(df, model, processor, device, batch_size=100):
    text_embeddings = []
    image_embeddings = []
    product_ids = []
    
    # Keep track of valid indices
    valid_indices = []
    
    # Batch image embedding first to determine which samples are valid
    image_paths = [f"data/images/{pid}.jpeg" for pid in df['Pid'].tolist()]
    total_image_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    for batch_num, i in enumerate(range(0, len(image_paths), batch_size), 1):
        batch_images = []
        batch_valid_indices = []
        
        for idx, path in enumerate(image_paths[i:i+batch_size]):
            try:
                # Open and convert image to RGB
                image = Image.open(path).convert("RGB")
                batch_images.append(image)
                batch_valid_indices.append(i + idx)  # Store the global index
            except Exception as e:
                print(f"Skipping problematic image {path}: {e}")
        
        if batch_images:
            try:
                # Process images using the CLIP processor
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
                # Skip the problematic batch
                continue
                
        print(f"\rImage embedding batch {batch_num}/{total_image_batches} processed", end='', flush=True)
    
    print(f"\nProcessed {len(valid_indices)} valid images out of {len(image_paths)} total images")
    
    # Now process text only for valid indices
    texts = df['Name'].iloc[valid_indices].tolist()
    ids = df['Pid'].iloc[valid_indices].tolist()
    
    # Filter out blank or NaN texts
    filtered_texts = []
    filtered_ids = []
    for text, pid in zip(texts, ids):
        if isinstance(text, str) and text.strip():
            filtered_texts.append(text)
            filtered_ids.append(pid)
        else:
            print(f"Skipping empty or invalid text for pid {pid}")
    
    # Proceed with text embedding only for filtered inputs
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
    
    return text_embeddings, image_embeddings, product_ids

def zip_product_images(df, output_zip_path="product_images.zip"):
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
    df = pd.read_csv("data/csv/sample_100k_v2.csv")

    # Zip product images
    filtered_df = zip_product_images(df)

    # Load model
    model_id = "openai/clip-vit-base-patch32"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    # Calculate embeddings
    text_embeddings, image_embeddings, product_ids = calculate_embeddings(
        filtered_df, 
        model=model,
        processor=processor,
        device=device
    )

    # Save embeddings
    save_embeddings(
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        product_ids=product_ids,
        save_path='embeddings.npz'
    )

if __name__ == "__main__":
    main() 