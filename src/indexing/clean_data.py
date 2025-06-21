import os
import pandas as pd
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config.path import CLEAN_CSV_PATH, RAW_CSV_PATH

def process_csv():
    # Columns to load
    selected_columns = [
        "Pid", "Description", "Name", "Category", 
        "Price", "PriceCurrency", "FinalPrice", "Discount", "isOnSale", "IsInStock", "Brand", 
        "Manufacturer", "Color", "Gender", "Size", "Condition"
    ]

    # Read CSV file — adjust the path and possibly set low_memory=False if needed
    df = pd.read_csv(RAW_CSV_PATH)

    # Filter allowed currencies
    df = df[df["PriceCurrency"].isin(["USD", "CAD", "GBP"])]

    # Clean and prepare both columns: Replace NaN with empty string and lowercase
    df["Brand_clean"] = df["Brand"].fillna('').str.strip().str.lower()
    df["Manufacturer_clean"] = df["Manufacturer"].fillna('').str.strip().str.lower()

    # Perform the comparison and create merged column
    df["MergedBrand"] = df["Brand"].where(df["Brand_clean"] == df["Manufacturer_clean"], df["Brand"].combine_first(df["Manufacturer"]))

    # Select columns to keep
    columns_to_keep = [
        "Pid", "Description", "Name", "Category", "Price", "PriceCurrency", "FinalPrice", 
        "Discount", "isOnSale", "IsInStock", "Color", "Gender", "Size", "Condition", "MergedBrand"
    ]

    df_filtered = df[columns_to_keep]

    # Check if directory exists and create it if not
    output_dir = os.path.dirname(CLEAN_CSV_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

    # Save to CSV
    df_filtered.to_csv(CLEAN_CSV_PATH)
    print(f"Saved cleaned data to: {CLEAN_CSV_PATH}")

def main():
    process_csv()

if __name__ == '__main__':
    main()
