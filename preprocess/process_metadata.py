from keybert import KeyBERT
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
from datetime import datetime

df = pd.read_csv("data/csv/sample_100k_v2.csv")
df_sample = df.sample(frac=0.001, random_state=42)
df_sample.shape

kw_model = KeyBERT()

def extract_keybert_keywords(text, top_n=5):
    if not isinstance(text, str) or not text.strip():
        return []
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),  # extract 1-gram and 2-gram phrases
        stop_words='english',
        use_maxsum=True,
        nr_candidates=20,  # generate the top 20 n-gram candidates, and rank them using MaxSum similarity to choose the final top_n
        top_n=top_n
    )
    return [kw[0] for kw in keywords]

def process_batch(texts, batch_size=100):
    results = []
    for text in tqdm(texts, desc="Processing batch"):
        results.append(extract_keybert_keywords(text))
    return results

# Process in batches
start = time.time()
batch_size = 100
total_rows = len(df_sample)
df_sample['tags_from_keybert'] = None

for i in tqdm(range(0, total_rows, batch_size), desc="Processing batches"):
    end_idx = min(i + batch_size, total_rows)
    batch_texts = df_sample['Description'].iloc[i:end_idx].tolist()
    batch_results = process_batch(batch_texts, batch_size)
    df_sample.loc[df_sample.index[i:end_idx], 'tags_from_keybert'] = pd.Series(batch_results, index=df_sample.index[i:end_idx])

print(f"Total process time: {(time.time() - start) / 60:.2f} minutes")

# Save results to CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"data/csv/keybert_results_{timestamp}.csv"
df_sample.to_csv(output_filename, index=False)
print(f"Results saved to: {output_filename}")
