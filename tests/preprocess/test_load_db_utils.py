import pytest
import numpy as np
import re
import pandas as pd
import pickle
import tempfile
import time
import sys
import os
from src.preprocess.load_db import (
    get_base_embedding_type,
    get_embedding_paths,
    get_enabled_embedding_types,
    get_chunked_files,
    validate_numeric,
    validate_boolean,
    validate_text,
    validate_and_clean_dataframe,
    save_checkpoint,
    load_checkpoint,
    insert_data,
    init_db,
    main
)


@pytest.mark.parametrize("filename, expected", [
    ("fusion_embeddings_chunk_0.npz", "fusion"),
    ("text_embeddings.npz", "text"),
    ("image_embeddings_chunk_123.npz", "image"),
])
def test_get_base_embedding_type(filename, expected):
    """
    Test extracting the base embedding type from various filenames.

    Parameters
    ----------
    filename : str
        The input filename to parse.
    expected : str
        The expected base embedding type extracted from the filename.

    Asserts
    -------
    The function returns the correct base type by stripping '_embeddings',
    chunk suffixes like '_chunk_X', and file extensions.
    """
    assert get_base_embedding_type(filename) == expected


def test_get_embedding_paths(fake_embeddings_dir):
    """
    Test that get_embedding_paths returns correct mapping of base embedding types to file paths.

    Parameters
    ----------
    fake_embeddings_dir : fixture
        A pytest fixture that sets up a temporary embeddings directory with
        sample .npz files and monkeypatches the EMBEDDINGS_PATH variable.

    Returns
    -------
    None
        The function performs assertions to validate behavior, but returns nothing.
    """
    filenames, path = fake_embeddings_dir
    expected_keys = {"fusion", "image", "text"}

    result = get_embedding_paths()

    assert set(result.keys()) == expected_keys
    for key, file_path in result.items():
        assert os.path.exists(file_path)
        assert key in os.path.basename(file_path)


def test_get_enabled_embedding_types(fake_embeddings_dir):
    """
    Test get_enabled_embedding_types returns returns a list of unique
    embedding types

    Parameters
    ----------
    fake_embeddings_dir : fixture
        A pytest fixture that creates a temporary directory of fake embedding files
        and monkeypatches the EMBEDDINGS_PATH variable.

    Returns
    -------
    None
        Asserts the function output matches the expected list of base embedding types.
    """
    filenames, _ = fake_embeddings_dir

    expected = ["fusion", "image", "text"]
    result = get_enabled_embedding_types()

    assert sorted(result) == sorted(expected)


def test_get_chunked_files_returns_sorted_chunks(fake_embeddings_dir):
    """
    Test that get_chunked_files returns chunked file paths sorted by chunk number.

    Parameters
    ----------
    fake_embeddings_dir : fixture
        A pytest fixture that creates fake .npz embedding files and monkeypatches
        EMBEDDINGS_PATH to a temporary directory.

    Returns
    -------
    None
    """
    filenames, path = fake_embeddings_dir

    result = get_chunked_files("fusion")
    expected_files = sorted([
        path / "fusion_embeddings_chunk_0.npz",
        path / "fusion_embeddings_chunk_1.npz"
    ], key=lambda p: int(re.search(r'chunk_(\d+)', p.name).group(1)))
    assert result == [str(p) for p in expected_files]

    result = get_chunked_files("image")
    expected_files = sorted([
        path / "image_embeddings_chunk_0.npz",
    ], key=lambda p: int(re.search(r'chunk_(\d+)', p.name).group(1)))
    assert result == [str(p) for p in expected_files]
    

def test_get_chunked_files_returns_non_chunked(fake_embeddings_dir):
    """
    Test that get_chunked_files returns non-chunked file correctly when no chunks exist.
    """
    result = get_chunked_files("text")
    assert len(result) == 0

def test_get_chunked_files_returns_empty_for_missing_type(fake_embeddings_dir):
    """
    Test that get_chunked_files returns an empty list when no files match the type.
    """
    result = get_chunked_files("nonexistent")
    assert result == []


@pytest.mark.parametrize("value, expected",
    [
        (42, 42.0),
        (3.14, 3.14),
        ("2.718", 2.718),
        ("", None),
        (None, None),
        (pd.NA, None),
        ("not a number", None),
    ]
)
def test_validate_numeric(value, expected, capsys):
    """
    Test the validate_numeric function with various inputs.

    Parameters
    ----------
    value : Any
        The input value to validate and convert to float.
    expected : float or None
        The expected result after validation.
    capsys : fixture
        Pytest fixture to capture printed warnings for invalid inputs.

    Asserts
    -------
    The function returns the correct float or None based on input.
    Captures and checks warning message if input is invalid.
    """
    field_name = "test_field"
    result = validate_numeric(value, field_name)
    assert result == expected

    if expected is None and not pd.isna(value) and value != "":
        captured = capsys.readouterr()
        assert f"Warning: Invalid numeric value for {field_name}" in captured.out


@pytest.mark.parametrize("value, expected", [
    (True, True),
    (0, False),
    ("yes", True),
    ("no", False),
    ("", None),
    (pd.NA, None),
    ("maybe", None),  # invalid, should trigger warning
])
def test_validate_boolean_simple(value, expected, capsys):
    """
    Test `validate_boolean` with representative cases.

    Parameters
    ----------
    value : any
        Input value to be validated as boolean.
    expected : bool or None
        Expected boolean or None output.
    capsys : fixture
        Pytest fixture to capture printed warnings for invalid inputs.
    """
    result = validate_boolean(value, "test_field")
    assert result == expected

    if expected is None and not (pd.isna(value) or value == ''):
        captured = capsys.readouterr()
        assert "Warning: Invalid boolean value for test_field" in captured.out



@pytest.mark.parametrize(
    "value, expected, expect_warning",
    [
        (" hello ", "hello", False),        # strip spaces
        ("", None, False),                  # empty string
        (None, None, False),                # None value
        (pd.NA, None, False),               # pandas NA
        (123, "123", False),                # numeric value as text
        (True, "True", False),              # boolean as string
        (["a", "list"], None, True),        # unconvertible type
        ({"a": 1}, None, True),             # invalid dict
    ]
)
def test_validate_text(value, expected, expect_warning, capsys):
    """
    Test the validate_text function with various inputs.

    Parameters
    ----------
    value : Any
        The input value to validate and convert to string.
    expected : str or None
        The expected result after validation.
    expect_warning : bool
        Whether a warning is expected in output.
    capsys : fixture
        Pytest fixture to capture printed warnings for invalid inputs.

    Asserts
    -------
    The function returns the correct string or None based on input.
    Captures and checks warning message if input is invalid.
    """
    field_name = "test_field"
    result = validate_text(value, field_name)
    assert result == expected

    captured = capsys.readouterr()
    if expect_warning:
        assert f"Warning: Invalid text value" in captured.out
    else:
        assert captured.out == ""


def test_validate_and_clean_dataframe(capsys):
    """
    Test the validate_and_clean_dataframe function with a sample DataFrame.

    This test verifies that:
    - Rows with missing or empty 'Pid' values are dropped.
    - Invalid numeric fields are coerced to NaN and don't cause crashes.
    - Invalid boolean values are converted to NaN and filtered out.
    - The cleaned DataFrame has the correct number of valid rows.
    - Console output includes summary statistics of the cleaning process.

    Parameters
    ----------
    capsys : _pytest.capture.CaptureFixture
        Pytest fixture to capture standard output during the test.

    Asserts
    -------
    - The returned object is a pandas DataFrame.
    - The number of rows is correctly reduced.
    - Valid Pid values remain.
    - Numeric columns are correctly coerced to float dtype.
    - Boolean columns are correctly mapped to True/False.
    - Output contains expected logging messages.
    """
    # Create a sample DataFrame with mixed valid and invalid rows
    df = pd.DataFrame({
        "Pid": ["A001", "A002", None, "", "A005"],
        "Price": [19.99, "invalid", 25.0, 10.0, ""],
        "FinalPrice": [15.99, 18.0, "bad", None, 20.0],
        "Discount": [4.0, None, 5.0, 2.0, "oops"],
        "isOnSale": ["yes", "no", "invalid", None, True],
        "IsInStock": [1, 0, "maybe", "true", "f"]
    })

    # Run validation and cleaning
    cleaned_df = validate_and_clean_dataframe(df)

    # Capture printed output
    captured = capsys.readouterr()

    # Basic assertions
    assert isinstance(cleaned_df, pd.DataFrame)
    assert "Validating and cleaning data..." in captured.out
    assert "Original rows: 5" in captured.out
    assert "Dropped rows:" in captured.out

    # Check that the correct rows remain (by valid 'Pid' and cleanable values)
    expected_pids = ["A001", "A002", "A005"]
    assert list(cleaned_df["Pid"]) == expected_pids

    # Check dtype
    assert cleaned_df["Price"].dtype.kind in {'f', 'i'}
    assert cleaned_df["isOnSale"].isin([True, False]).all()
    assert cleaned_df["IsInStock"].isin([True, False]).all()


def test_save_checkpoint(monkeypatch, capsys):
    """
    Test the save_checkpoint function to ensure it correctly writes checkpoint data.

    This test verifies that:
    - A checkpoint file is created.
    - The file contains the correct batch number, total batches, and a recent timestamp.
    - The function prints the expected output message.

    Parameters
    ----------
    monkeypatch : _pytest.monkeypatch.MonkeyPatch
        Pytest fixture to temporarily replace attributes or dictionary values for testing.
    capsys : _pytest.capture.CaptureFixture
        Pytest fixture to capture printed output to stdout/stderr.

    Asserts
    -------
    - The printed message includes the correct batch progress.
    - The checkpoint file contains the expected dictionary values.
    - The timestamp is within a valid time range around the function call.
    """
    # Create a temporary file path for checkpoint
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    # Monkeypatch the global CHECKPOINT_FILE used in the function
    monkeypatch.setattr(load_db, "CHECKPOINT_FILE", tmp_path)

    batch_number = 5
    total_batches = 20
    before = time.time()

    # Call the function
    save_checkpoint(batch_number, total_batches)

    # Capture print output
    captured = capsys.readouterr()
    assert f"Checkpoint saved: Batch {batch_number}/{total_batches}" in captured.out

    # Check that the file was created and contains expected data
    with open(tmp_path, 'rb') as f:
        data = pickle.load(f)
        assert data['batch_number'] == batch_number
        assert data['total_batches'] == total_batches
        assert before <= data['timestamp'] <= time.time()

    # Clean up temp file
    os.remove(tmp_path)


def test_load_checkpoint(monkeypatch, capsys):
    """
    Test the load_checkpoint function to ensure it correctly loads checkpoint data.

    This test verifies that:
    - If a valid checkpoint file exists, the function returns the correct batch number and total batches.
    - If the file is missing or corrupted, the function returns default values (0, None).
    - Proper messages are printed in each case.

    Parameters
    ----------
    monkeypatch : _pytest.monkeypatch.MonkeyPatch
        Pytest fixture to temporarily override attributes such as the checkpoint file path.
    capsys : _pytest.capture.CaptureFixture
        Pytest fixture to capture standard output.

    Asserts
    -------
    - Correct tuple is returned for valid data.
    - Correct default is returned if no file or invalid file exists.
    - Output messages match expected content.
    """
    # 1. Test with valid checkpoint
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
        checkpoint_data = {
            'batch_number': 3,
            'total_batches': 10,
            'timestamp': 1234567890.0
        }
        with open(tmp_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

    monkeypatch.setattr(load_db, "CHECKPOINT_FILE", tmp_path)
    batch_number, total_batches = load_checkpoint()
    captured = capsys.readouterr()
    assert batch_number == 3
    assert total_batches == 10
    assert "Loaded checkpoint: Batch 3/10" in captured.out

    os.remove(tmp_path)

    # 2. Test with no checkpoint file
    monkeypatch.setattr(load_db, "CHECKPOINT_FILE", tmp_path)  # Same path, now deleted
    batch_number, total_batches = load_checkpoint()
    captured = capsys.readouterr()
    assert (batch_number, total_batches) == (0, None)
    assert "Loaded checkpoint" not in captured.out  # Should not print anything on missing file

    # 3. Test with corrupt file
    with open(tmp_path, 'wb') as f:
        f.write(b"not a pickle")
    monkeypatch.setattr(load_db, "CHECKPOINT_FILE", tmp_path)
    batch_number, total_batches = load_checkpoint()
    captured = capsys.readouterr()
    assert (batch_number, total_batches) == (0, None)
    assert "Error loading checkpoint:" in captured.out

    os.remove(tmp_path)