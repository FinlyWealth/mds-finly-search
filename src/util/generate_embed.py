import torch
import pandas as pd
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor
from tqdm.auto import tqdm

# Global variables for model and processor
device = None
processor = None
model = None

def calculate_embeddings(df, batch_size=100):
    text_embeddings = []
    image_embeddings = []
    product_ids = []
    
    # Keep track of valid indices
    valid_indices = []
    
    # Batch image embedding first to determine which samples are valid
    image_paths = [f"../data/images/{pid}.jpeg" for pid in df['Pid'].tolist()]
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
    
    total_text_batches = (len(texts) + batch_size - 1) // batch_size
    for batch_num, i in enumerate(range(0, len(texts), batch_size), 1):
        batch_texts = texts[i:i+batch_size]
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            batch_features = model.get_text_features(**inputs)
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
            
        text_embeddings.extend(batch_features.cpu().numpy())
        product_ids.extend(ids[i:i+batch_size])
        print(f"\rText embedding batch {batch_num}/{total_text_batches} processed", end='', flush=True)
    
    print(f"\nFinal dataset size: {len(text_embeddings)} pairs")
    
    return text_embeddings, image_embeddings, product_ids

def save_embeddings(text_embeddings, image_embeddings, product_ids, save_path='embeddings.npz'):
    """
    Save the calculated embeddings to a numpy file.
    
    Args:
        text_embeddings (list): List of text embeddings
        image_embeddings (list): List of image embeddings
        product_ids (list): List of product IDs
        save_path (str): Path to save the embeddings file
    """
    np.savez(save_path, 
             text_embeddings=np.array(text_embeddings),
             image_embeddings=np.array(image_embeddings),
             product_ids=np.array(product_ids))
    print(f"Embeddings saved to {save_path}")

def main():
    global device, processor, model
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    print(f"Device is {device}")

    # Load data
    df = pd.read_parquet('../data/merged_output_sample_100k.parquet')

    # Load model
    model_id = "openai/clip-vit-base-patch32"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    # Calculate embeddings
    text_embeddings, image_embeddings, product_ids = calculate_embeddings(df)
    
    # Save embeddings
    save_embeddings(text_embeddings, image_embeddings, product_ids)

if __name__ == "__main__":
    main() 