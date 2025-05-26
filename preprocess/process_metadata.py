from keybert import KeyBERT
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
from datetime import datetime
import torch
from sentence_transformers import SentenceTransformer

# Load and sample the data
df = pd.read_csv("data/csv/sample_100k_v2.csv")
df_sample = df.sample(frac=0.0001, random_state=42)
print(f"Sample size: {df_sample.shape}")

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize the sentence transformer model with GPU support
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
kw_model = KeyBERT(model=model)

def extract_keybert_keywords(text, ngram_range, top_n=10):
    if not isinstance(text, str) or not text.strip():
        return []
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=ngram_range,
        stop_words='english',
        use_maxsum=True,
        nr_candidates=20,
        top_n=top_n
    )
    return [kw[0] for kw in keywords]

def process_batch(texts, ngram_range, batch_size=100):
    results = []
    for text in tqdm(texts, desc="Processing batch"):
        results.append(extract_keybert_keywords(text, ngram_range))
    return results

# Process both configurations
configs = [(1, 2), (1, 3)]
for ngram_range in configs:
    print(f"\nProcessing with n-gram range {ngram_range}...")
    df_working = df_sample.copy()
    
    # Process in batches
    start = time.time()
    batch_size = 100
    total_rows = len(df_working)
    df_working['tags_from_keybert'] = None

    for i in tqdm(range(0, total_rows, batch_size), desc="Processing batches"):
        end_idx = min(i + batch_size, total_rows)
        batch_texts = df_working['Description'].iloc[i:end_idx].tolist()
        batch_results = process_batch(batch_texts, ngram_range, batch_size)
        df_working.loc[df_working.index[i:end_idx], 'tags_from_keybert'] = pd.Series(batch_results, index=df_working.index[i:end_idx])

    print(f"Total process time: {(time.time() - start) / 60:.2f} minutes")

    # Create the new concatenated column
    df_working['cleaned_tags'] = df_working['tags_from_keybert'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else ''
    )
    df_working['CombinedInfo'] = 'name: ' + df_working['Name'] + ', keywords: ' + df_working['cleaned_tags']
    df_working = df_working.drop('cleaned_tags', axis=1)

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"data/csv/keybert_results_{ngram_range[0]}_{ngram_range[1]}gram_{timestamp}.csv"
    df_working.to_csv(output_filename, index=False)
    print(f"Results saved to: {output_filename}")
