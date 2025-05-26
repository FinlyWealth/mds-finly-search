import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.db import (
    get_db_connection,
    fetch_product_by_pid,
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
def test_fetch_product_by_pid_success(mock_get_connection):
    """Test successful product fetch by PID"""
    # Arrange
    mock_connection = MagicMock()
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor
    mock_get_connection.return_value = mock_connection

    # Mock database row data
    mock_row = (
        "Test Product",
        "Test Description",
        "Test Brand",
        "Test Category",
        "Red",
        "Unisex",
        "M",
    )
    mock_cursor.fetchone.return_value = mock_row

    test_pid = "12345"

    # Act
    result = fetch_product_by_pid(test_pid)

    # Assert
    expected_result = {
        "Name": "Test Product",
        "Description": "Test Description",
        "Brand": "Test Brand",
        "Category": "Test Category",
        "Color": "Red",
        "Gender": "Unisex",
        "Size": "M",
    }

    assert result == expected_result
    mock_cursor.execute.assert_called_once()
    mock_cursor.fetchone.assert_called_once()
    mock_cursor.close.assert_called_once()
    mock_connection.close.assert_called_once()


@patch("src.backend.db.get_db_connection")
def test_fetch_product_by_pid_not_found(mock_get_connection):
    """Test product fetch when PID doesn't exist"""
    # Arrange
    mock_connection = MagicMock()
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor
    mock_get_connection.return_value = mock_connection

    # Mock no data found
    mock_cursor.fetchone.return_value = None

    test_pid = "nonexistent"

    # Act
    result = fetch_product_by_pid(test_pid)

    # Assert
    assert result is None
    mock_cursor.execute.assert_called_once()
    mock_cursor.fetchone.assert_called_once()
    mock_cursor.close.assert_called_once()
    mock_connection.close.assert_called_once()


@patch("src.backend.db.get_db_connection")
def test_fetch_product_by_pid_database_error(mock_get_connection):
    """Test product fetch when database error occurs"""
    # Arrange
    mock_connection = MagicMock()
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor
    mock_get_connection.return_value = mock_connection

    # Mock database error
    mock_cursor.execute.side_effect = Exception("Database error")

    test_pid = "12345"

    # Act & Assert
    with pytest.raises(Exception, match="Database error"):
        fetch_product_by_pid(test_pid)


@patch("src.backend.db.get_db_connection")
def test_fetch_products_by_pids_success(mock_get_connection):
    """Test successful multiple products fetch by PIDs"""
    # Arrange
    mock_connection = MagicMock()
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor
    mock_get_connection.return_value = mock_connection

    # Mock database rows data
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
        ),
    ]
    mock_cursor.fetchall.return_value = mock_rows

    test_pids = ["pid1", "pid2", "pid3"]
    test_scores = [0.8, 0.9, 0.7]  # pid2 should be first (highest score)

    # Act
    result = fetch_products_by_pids(test_pids, test_scores)

    # Assert
    assert len(result) == 3

    # Check that results are sorted by score (descending)
    assert result[0]["Pid"] == "pid2"  # Highest score (0.9)
    assert result[0]["similarity"] == 0.9
    assert result[1]["Pid"] == "pid1"  # Second highest (0.8)
    assert result[1]["similarity"] == 0.8
    assert result[2]["Pid"] == "pid3"  # Lowest score (0.7)
    assert result[2]["similarity"] == 0.7

    # Check product details
    assert result[0]["Name"] == "Product 2"
    assert result[0]["Brand"] == "Brand 2"
    assert result[0]["Color"] == "Blue"

    mock_cursor.execute.assert_called_once()
    mock_cursor.fetchall.assert_called_once()
    mock_cursor.close.assert_called_once()
    mock_connection.close.assert_called_once()


@patch("src.backend.db.get_db_connection")
def test_fetch_products_by_pids_empty_lists(mock_get_connection):
    """Test fetch with empty PID and score lists"""
    # Arrange
    mock_connection = MagicMock()
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor
    mock_get_connection.return_value = mock_connection

    mock_cursor.fetchall.return_value = []

    test_pids = []
    test_scores = []

    # Act
    result = fetch_products_by_pids(test_pids, test_scores)

    # Assert
    assert result == []


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
        ),
    ]
    mock_cursor.fetchall.return_value = mock_rows

    test_pids = ["pid1", "pid2", "pid3"]  # pid2 doesn't exist in DB
    test_scores = [0.8, 0.9, 0.7]

    # Act
    result = fetch_products_by_pids(test_pids, test_scores)

    # Assert
    assert len(result) == 2  # Only pid1 and pid3 should be returned

    # Check that results are still sorted by score, but only include existing PIDs
    assert result[0]["Pid"] == "pid1"  # Score 0.8
    assert result[1]["Pid"] == "pid3"  # Score 0.7

    # pid2 should not be in results even though it had the highest score
    pids_in_result = [item["Pid"] for item in result]
    assert "pid2" not in pids_in_result


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
    test_scores = [0.8, 0.9]

    # Act & Assert
    with pytest.raises(Exception, match="Database error"):
        fetch_products_by_pids(test_pids, test_scores)


@patch("src.backend.db.get_db_connection")
def test_fetch_products_by_pids_mismatched_lengths(mock_get_connection):
    """Test fetch with mismatched PID and score list lengths"""
    # Arrange
    mock_connection = MagicMock()
    mock_cursor = MagicMock()
    mock_connection.cursor.return_value = mock_cursor
    mock_get_connection.return_value = mock_connection

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
        )
    ]
    mock_cursor.fetchall.return_value = mock_rows

    test_pids = ["pid1", "pid2"]
    test_scores = [0.8]  # Only one score for two PIDs

    # Act
    # The function will only process pairs that exist, so pid2 will be ignored
    result = fetch_products_by_pids(test_pids, test_scores)

    # Assert
    # Only pid1 should be in the result since it's the only one with a matching score
    assert len(result) == 1
    assert result[0]["Pid"] == "pid1"
    assert result[0]["similarity"] == 0.8


def test_fetch_products_by_pids_score_sorting():
    """Test that products are correctly sorted by similarity scores"""
    # This is an integration-style test focusing on the sorting logic
    with patch("src.backend.db.get_db_connection") as mock_get_connection:
        # Arrange
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_get_connection.return_value = mock_connection

        # Mock data with various scores
        mock_rows = [
            ("pid1", "Product 1", "Desc 1", "Brand 1", "Cat 1", "Red", "Male", "L"),
            ("pid2", "Product 2", "Desc 2", "Brand 2", "Cat 2", "Blue", "Female", "S"),
            ("pid3", "Product 3", "Desc 3", "Brand 3", "Cat 3", "Green", "Unisex", "M"),
            ("pid4", "Product 4", "Desc 4", "Brand 4", "Cat 4", "Yellow", "Male", "XL"),
        ]
        mock_cursor.fetchall.return_value = mock_rows

        test_pids = ["pid1", "pid2", "pid3", "pid4"]
        test_scores = [0.5, 0.9, 0.3, 0.7]  # Expected order: pid2, pid4, pid1, pid3

        # Act
        result = fetch_products_by_pids(test_pids, test_scores)

        # Assert
        expected_order = ["pid2", "pid4", "pid1", "pid3"]
        expected_scores = [0.9, 0.7, 0.5, 0.3]

        actual_order = [item["Pid"] for item in result]
        actual_scores = [item["similarity"] for item in result]

        assert actual_order == expected_order
        assert actual_scores == expected_scores
