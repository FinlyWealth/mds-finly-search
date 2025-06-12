import os
import importlib
from pathlib import Path

import pandas as pd
import pytest


def test_process_csv(tmp_path, monkeypatch):
    # 1) Prepare a tiny raw CSV under tmp_path/data/raw/data.csv
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    raw_csv = raw_dir / "data.csv"

    df = pd.DataFrame({
        "Pid": [1, 2, 3],
        "Description": ["d1", "d2", "d3"],
        "Name": ["n1", "n2", "n3"],
        "Category": ["c1", "c2", "c3"],
        "Price": [10.0, 20.0, 30.0],
        "PriceCurrency": ["USD", "EUR", "CAD"],      # EUR should be filtered out
        "FinalPrice": [9.0, 19.0, 29.0],
        "Discount": [1.0, 1.0, 1.0],
        "isOnSale": [True, False, True],
        "IsInStock": [True, True, False],
        "Brand": ["BrandA", "BrandB", None],
        "Manufacturer": ["branda", "BrandC", None],
        "Color": ["red", "blue", "green"],
        "Gender": ["M", "F", "M"],
        "Size": ["L", "M", "S"],
        "Condition": ["new", "used", "refurbished"]
    })
    df.to_csv(raw_csv, index=False)

    # 2) Prepare the clean output directory
    clean_dir = tmp_path / "data" / "clean"
    clean_dir.mkdir(parents=True)
    clean_csv = clean_dir / "data.csv"

    # 3) Override env vars so config.path picks them up
    monkeypatch.setenv("RAW_CSV_PATH", str(raw_csv))
    monkeypatch.setenv("CLEAN_CSV_PATH", str(clean_csv))

    # 4) Reload modules so they re-read os.getenv(...)
    import config.path
    import src.preprocess.clean_data as clean_mod
    importlib.reload(config.path)
    importlib.reload(clean_mod)
    from src.preprocess.clean_data import process_csv  

    # 5) Execute
    process_csv()

    # 6) Validate output
    out = pd.read_csv(clean_csv)
    # Only USD & CAD rows remain
    assert list(out["Pid"]) == [1, 3]

    # MergedBrand: BrandA/branda → BrandA
    assert out.loc[out["Pid"] == 1, "MergedBrand"].iloc[0] == "BrandA"
    # Row 3 had both None → pandas will produce NaN
    assert pd.isna(out.loc[out["Pid"] == 3, "MergedBrand"].iloc[0])

    # Ensure the column exists
    assert "MergedBrand" in out.columns