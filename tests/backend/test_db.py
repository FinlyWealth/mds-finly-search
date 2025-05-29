import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.backend.db import (
    get_db_connection,
    fetch_products_by_pids,
)


@patch("src.backend.db.psycopg2.connect")
def test_get_db_connection_success(mock_connect):
    """Test successful database connection"""
    # Arrange
    mock_connection = MagicMock()
    mock_connect.return_value = mock_connection

    # Act
    result = get_db_connection()

    # Assert
    assert result == mock_connection
    mock_connect.assert_called_once()


@patch("src.backend.db.psycopg2.connect")
def test_get_db_connection_failure(mock_connect):
    """Test database connection failure"""
    # Arrange
    mock_connect.side_effect = Exception("Connection failed")

    # Act & Assert
    with pytest.raises(Exception, match="Connection failed"):
        get_db_connection()


@patch("src.backend.db.get_db_connection")
def test_fetch_products_by_pids_success(mock_get_connection):
    """Test successful products fetch by PIDs"""
    # Arrange
    mock_connection = MagicMock()
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor
    mock_get_connection.return_value = mock_connection

    # Mock database rows data - function returns rows with Pid, Name, Description, Brand, Category, Color, Gender, Size, Price
    mock_rows = [
        (
            "pid1",
            "Product 1",
            "Description 1",
            "Brand 1",
            "Category 1",
            "Red",
            "Male",
            "L",
            29.99,
        ),
        (
            "pid2",
            "Product 2",
            "Description 2",
            "Brand 2",
            "Category 2",
            "Blue",
            "Female",
            "S",
            39.99,
        ),
        (
            "pid3",
            "Product 3",
            "Description 3",
            "Brand 3",
            "Category 3",
            "Green",
            "Unisex",
            "M",
            49.99,
        ),
    ]
    mock_cursor.fetchall.return_value = mock_rows

    test_pids = ["pid1", "pid2", "pid3"]

    # Act
    result = fetch_products_by_pids(test_pids)

    # Assert
    expected_result = {
        "pid1": {
            "Name": "Product 1",
            "Description": "Description 1",
            "Brand": "Brand 1",
            "Category": "Category 1",
            "Color": "Red",
            "Gender": "Male",
            "Size": "L",
            "Price": 29.99,
        },
        "pid2": {
            "Name": "Product 2",
            "Description": "Description 2",
            "Brand": "Brand 2",
            "Category": "Category 2",
            "Color": "Blue",
            "Gender": "Female",
            "Size": "S",
            "Price": 39.99,
        },
        "pid3": {
            "Name": "Product 3",
            "Description": "Description 3",
            "Brand": "Brand 3",
            "Category": "Category 3",
            "Color": "Green",
            "Gender": "Unisex",
            "Size": "M",
            "Price": 49.99,
        },
    }

    assert result == expected_result
    mock_cursor.execute.assert_called_once()
    mock_cursor.fetchall.assert_called_once()
    mock_cursor.close.assert_called_once()
    mock_connection.close.assert_called_once()


@patch("src.backend.db.get_db_connection")
def test_fetch_products_by_pids_empty_list(mock_get_connection):
    """Test fetch with empty PID list"""
    # Act
    result = fetch_products_by_pids([])

    # Assert
    assert result == {}
    # Connection should not be called for empty list
    mock_get_connection.assert_not_called()


@patch("src.backend.db.get_db_connection")
def test_fetch_products_by_pids_not_found(mock_get_connection):
    """Test fetch when PIDs don't exist in database"""
    # Arrange
    mock_connection = MagicMock()
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor
    mock_get_connection.return_value = mock_connection

    # Mock no data found
    mock_cursor.fetchall.return_value = []

    test_pids = ["nonexistent1", "nonexistent2"]

    # Act
    result = fetch_products_by_pids(test_pids)

    # Assert
    assert result == {}
    mock_cursor.execute.assert_called_once()
    mock_cursor.fetchall.assert_called_once()
    mock_cursor.close.assert_called_once()
    mock_connection.close.assert_called_once()


@patch("src.backend.db.get_db_connection")
def test_fetch_products_by_pids_partial_matches(mock_get_connection):
    """Test fetch when only some PIDs exist in database"""
    # Arrange
    mock_connection = MagicMock()
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor
    mock_get_connection.return_value = mock_connection

    # Only return data for pid1 and pid3, not pid2
    mock_rows = [
        (
            "pid1",
            "Product 1",
            "Description 1",
            "Brand 1",
            "Category 1",
            "Red",
            "Male",
            "L",
            29.99,
        ),
        (
            "pid3",
            "Product 3",
            "Description 3",
            "Brand 3",
            "Category 3",
            "Green",
            "Unisex",
            "M",
            49.99,
        ),
    ]
    mock_cursor.fetchall.return_value = mock_rows

    test_pids = ["pid1", "pid2", "pid3"]  # pid2 doesn't exist in DB

    # Act
    result = fetch_products_by_pids(test_pids)

    # Assert
    assert len(result) == 2  # Only pid1 and pid3 should be returned
    assert "pid1" in result
    assert "pid3" in result
    assert "pid2" not in result

    # Check product details
    assert result["pid1"]["Name"] == "Product 1"
    assert result["pid3"]["Name"] == "Product 3"

    mock_cursor.execute.assert_called_once()
    mock_cursor.fetchall.assert_called_once()
    mock_cursor.close.assert_called_once()
    mock_connection.close.assert_called_once()


@patch("src.backend.db.get_db_connection")
def test_fetch_products_by_pids_database_error(mock_get_connection):
    """Test fetch when database error occurs"""
    # Arrange
    mock_connection = MagicMock()
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor
    mock_get_connection.return_value = mock_connection

    # Mock database error
    mock_cursor.execute.side_effect = Exception("Database error")

    test_pids = ["pid1", "pid2"]

    # Act & Assert
    with pytest.raises(Exception, match="Database error"):
        fetch_products_by_pids(test_pids)


@patch("src.backend.db.get_db_connection")
def test_fetch_products_by_pids_single_product(mock_get_connection):
    """Test fetch with a single PID"""
    # Arrange
    mock_connection = MagicMock()
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor
    mock_get_connection.return_value = mock_connection

    mock_rows = [
        (
            "single_pid",
            "Single Product",
            "Single Description",
            "Single Brand",
            "Single Category",
            "Blue",
            "Unisex",
            "XL",
            99.99,
        )
    ]
    mock_cursor.fetchall.return_value = mock_rows

    test_pids = ["single_pid"]

    # Act
    result = fetch_products_by_pids(test_pids)

    # Assert
    assert len(result) == 1
    assert "single_pid" in result
    assert result["single_pid"]["Name"] == "Single Product"
    assert result["single_pid"]["Price"] == 99.99

    mock_cursor.execute.assert_called_once()
    mock_cursor.fetchall.assert_called_once()
    mock_cursor.close.assert_called_once()
    mock_connection.close.assert_called_once()


@patch("src.backend.db.get_db_connection")
def test_fetch_products_by_pids_sql_injection_protection(mock_get_connection):
    """Test that the function properly handles potential SQL injection attempts"""
    # Arrange
    mock_connection = MagicMock()
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor
    mock_get_connection.return_value = mock_connection

    mock_cursor.fetchall.return_value = []

    # Test with potentially malicious input
    test_pids = ["'; DROP TABLE products; --", "normal_pid"]

    # Act
    result = fetch_products_by_pids(test_pids)

    # Assert
    assert result == {}
    mock_cursor.execute.assert_called_once()
    
    # Verify the SQL query was constructed properly (with quotes around PIDs)
    call_args = mock_cursor.execute.call_args[0][0]
    assert "'; DROP TABLE products; --'" in call_args  # Should be quoted
    assert "'normal_pid'" in call_args

    mock_cursor.fetchall.assert_called_once()
    mock_cursor.close.assert_called_once()
    mock_connection.close.assert_called_once()
