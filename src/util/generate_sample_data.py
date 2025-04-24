import pyarrow.csv as pv
import pyarrow.parquet as pq
import pyarrow as pa
import random
import os
import zipfile
from pathlib import Path

SAMPLE_SIZE = 100_000

# Get the absolute path to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path to the project root (two levels up from script_dir)
project_root = os.path.dirname(os.path.dirname(script_dir))

# Paths
input_csv_path = os.path.join(project_root, 'data', 'merged_output.csv')
output_parquet_path = os.path.join(project_root, 'data', f'merged_output_sample_{SAMPLE_SIZE}.parquet')
zip_path = os.path.join(project_root, 'data', 'images.zip')
output_images_dir = os.path.join(project_root, 'data', 'images')

# Define column types based on the actual CSV structure
column_types = {
    'Pid': pa.string(),
    'Name': pa.string(),
    'ShortDescription': pa.string(),
    'Description': pa.string(),
    'CategoryId': pa.string(),
    'Category': pa.string(),
    'ImageURL': pa.string(),
    'Price': pa.float64(),
    'PriceCurrency': pa.string(),
    'SalePrice': pa.float64(),
    'FinalPrice': pa.float64(),
    'Discount': pa.string(),
    'isOnSale': pa.bool_(),
    'IsInStock': pa.bool_(),
    'Keywords': pa.string(),
    'Brand': pa.string(),
    'Manufacturer': pa.string(),
    'MPN': pa.string(),
    'UPCorEAN': pa.string(),
    'SKU': pa.string(),
    'Color': pa.string(),
    'Gender': pa.string(),
    'Size': pa.string(),
    'Condition': pa.string(),
    'AlternateImageUrl': pa.string(),
    'AlternateImageUrl2': pa.string(),
    'AlternateImageUrl3': pa.string(),
    'AlternateImageUrl4': pa.string(),
    'DeepLinkURL': pa.string(),
    'LinkUrl': pa.string()
}

# First pass: Count total number of rows
print("Counting total rows...")
row_count = 0
reader = pv.open_csv(input_csv_path, convert_options=pv.ConvertOptions(column_types=column_types))
for batch in reader:
    row_count += batch.num_rows

# Randomly select the indices to sample
sample_indices = set(random.sample(range(row_count), SAMPLE_SIZE))

# Second pass: Stream and collect sampled rows
print("Sampling rows...")
reader = pv.open_csv(input_csv_path, convert_options=pv.ConvertOptions(column_types=column_types))
sampled_tables = []

start_index = 0
for batch in reader:
    end_index = start_index + batch.num_rows
    local_indices = [i for i in range(batch.num_rows) if (start_index + i) in sample_indices]
    if local_indices:
        sampled_batch = batch.take(pa.array(local_indices))
        # Convert Discount column from percentage string to float
        discount_col = sampled_batch.column('Discount')
        discount_values = [float(str(x).rstrip('%')) / 100 if x else 0.0 for x in discount_col]
        sampled_batch = sampled_batch.set_column(
            sampled_batch.schema.get_field_index('Discount'),
            'Discount',
            pa.array(discount_values, type=pa.float64())
        )
        # Convert RecordBatch to Table
        sampled_table = pa.Table.from_batches([sampled_batch])
        sampled_tables.append(sampled_table)
    start_index = end_index
    if start_index > max(sample_indices):
        break

# Concatenate all sampled tables and write to Parquet
print("Writing parquet file...")
sampled_table = pa.concat_tables(sampled_tables)
pq.write_table(sampled_table, output_parquet_path)

print(f"Saved {SAMPLE_SIZE} sampled rows to '{output_parquet_path}'")

# Extract matching images
print("\nExtracting images...")
# Create output directory if it doesn't exist
os.makedirs(output_images_dir, exist_ok=True)

# Get PIDs from the sampled data
pids = set(sampled_table.column('Pid').to_pylist())

# Extract matching images from zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Get list of all files in zip
    all_files = zip_ref.namelist()
    
    # Filter for jpeg files that match our PIDs
    matching_files = [f for f in all_files if f.endswith('.jpeg') and Path(f).stem in pids]
    
    # Extract matching files
    extracted_count = 0
    for file in matching_files:
        zip_ref.extract(file, output_images_dir)
        extracted_count += 1

print(f"Extracted {extracted_count} images to: {output_images_dir}")