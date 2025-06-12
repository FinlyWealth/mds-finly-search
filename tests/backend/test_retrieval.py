import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch, mock_open
import sys
import os
import numpy as np
import json
import tempfile
import shutil

from requests import HTTPError

# Add the src directory to the path so we can import the modules
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.backend.retrieval import (
    download_from_gcs,
    SimilarityRetrieval,
    PostgresVectorRetrieval,
    FaissVectorRetrieval,
    TextSearchRetrieval,
    create_retrieval_component,
    hybrid_retrieval,
    reorder_search_results_by_relevancy,
    ReorderOutput,
    _faiss_index_cache,
    _faiss_mapping_cache,
)

GCS_BUCKET_URL = os.getenv("GCS_BUCKET_URL", "https://storage.googleapis.com/mds-finly")

class TestDownloadFromGCS:
    """Test suite for download_from_gcs function"""

    @patch("src.backend.retrieval.requests.get")
    @patch("src.backend.retrieval.os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_download_from_gcs_success(self, mock_file, mock_makedirs, mock_get):
        """Test successful file download from GCS"""
        # Arrange
        mock_response = MagicMock()
        mock_response.content = b"test file content"
        mock_get.return_value = mock_response

        source_path = "test/file.txt"
        destination = "/tmp/test_file.txt"

        # Act
        download_from_gcs(source_path, destination)

        # Assert
        expected_url = f"{GCS_BUCKET_URL}/{source_path}"
        mock_get.assert_called_once_with(expected_url)
        mock_response.raise_for_status.assert_called_once()
        mock_makedirs.assert_called_once_with(
            os.path.dirname(destination), exist_ok=True
        )
        mock_file.assert_called_once_with(destination, "wb")
        mock_file().write.assert_called_once_with(b"test file content")

    @patch("src.backend.retrieval.requests.get")
    def test_download_from_gcs_http_error(self, mock_get):
        """Test download failure due to HTTP error"""
        # Arrange
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 404")
        mock_get.return_value = mock_response

        # Act & Assert
        with pytest.raises(Exception, match="HTTP 404"):
            download_from_gcs("test/file.txt", "/tmp/test_file.txt")


class TestSimilarityRetrieval:
    """Test suite for SimilarityRetrieval base class"""

    def test_similarity_retrieval_not_implemented(self):
        """Test that base class raises NotImplementedError"""
        retrieval = SimilarityRetrieval()

        with pytest.raises(NotImplementedError):
            retrieval.score("test query", k=10)


class TestPostgresVectorRetrieval:
    """Test suite for PostgresVectorRetrieval class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.db_config = {
            "dbname": "test_db",
            "user": "test_user",
            "password": "test_pass",
            "host": "localhost",
            "port": "5432",
        }
        self.retrieval = PostgresVectorRetrieval("text_embedding", self.db_config)

    @patch("src.backend.retrieval.psycopg2.connect")
    def test_postgres_vector_retrieval_success(self, mock_connect):
        """Test successful vector retrieval from Postgres"""
        # Arrange
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection

        # Mock database results
        mock_cursor.fetchall.return_value = [
            ("pid1", 0.9),
            ("pid2", 0.8),
            ("pid3", 0.7),
        ]

        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # Act
        result = self.retrieval.score(query_vector, k=3)

        # Assert
        expected_result = {"pid1": 0.9, "pid2": 0.8, "pid3": 0.7}
        assert result == expected_result

        mock_connect.assert_called_once_with(**self.db_config)
        mock_cursor.execute.assert_called_once()
        mock_cursor.close.assert_called_once()
        mock_connection.close.assert_called_once()

    @patch("src.backend.retrieval.psycopg2.connect")
    def test_postgres_vector_retrieval_empty_results(self, mock_connect):
        """Test vector retrieval with no results"""
        # Arrange
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection

        mock_cursor.fetchall.return_value = []

        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # Act
        result = self.retrieval.score(query_vector, k=3)

        # Assert
        assert result == {}

    @patch("src.backend.retrieval.psycopg2.connect")
    def test_postgres_vector_retrieval_database_error(self, mock_connect):
        """Test vector retrieval with database error"""
        # Arrange
        mock_connect.side_effect = Exception("Database connection failed")

        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # Act & Assert
        with pytest.raises(Exception, match="Database connection failed"):
            self.retrieval.score(query_vector, k=3)


class TestFaissVectorRetrieval:
    """Test suite for FaissVectorRetrieval class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.db_config = {
            "dbname": "test_db",
            "user": "test_user",
            "password": "test_pass",
            "host": "localhost",
            "port": "5432",
        }
        # Clear cache before each test
        _faiss_index_cache.clear()
        _faiss_mapping_cache.clear()

    @patch("src.backend.retrieval.faiss.read_index")
    @patch("src.backend.retrieval.os.path.exists")
    @patch(
        "builtins.open", new_callable=mock_open, read_data='{"0": "pid1", "1": "pid2"}'
    )
    @patch("src.backend.retrieval.psycopg2.connect")
    def test_faiss_vector_retrieval_success(
        self, mock_connect, mock_file, mock_exists, mock_read_index
    ):
        """Test successful FAISS vector retrieval"""
        # Arrange
        mock_exists.return_value = True

        # Mock FAISS index
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8]]),  # distances
            np.array([[0, 1]]),  # indices
        )
        mock_read_index.return_value = mock_index

        # Mock database connection
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection

        # Mock database results with embeddings
        mock_cursor.fetchall.return_value = [
            ("pid1", "[0.1,0.2,0.3]"),
            ("pid2", "[0.4,0.5,0.6]"),
        ]

        # Act
        retrieval = FaissVectorRetrieval(
            "text_embedding", nprobe=1, db_config=self.db_config
        )
        query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)  # Normalize

        result = retrieval.score(query_vector, k=2)

        # Assert
        assert isinstance(result, dict)
        assert len(result) <= 2
        for pid, score in result.items():
            assert isinstance(pid, str)
            assert isinstance(score, float)
            assert 0 <= score <= 1

    @patch("src.backend.retrieval.download_from_gcs")
    @patch("src.backend.retrieval.faiss.read_index")
    @patch("src.backend.retrieval.os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data='{"0": "pid1"}')
    def test_faiss_vector_retrieval_download_from_gcs(
        self, mock_file, mock_exists, mock_read_index, mock_download
    ):
        """Test FAISS retrieval when files need to be downloaded from GCS"""
        # Arrange
        mock_exists.return_value = False  # Files don't exist locally
        mock_index = MagicMock()
        mock_read_index.return_value = mock_index

        # Act
        retrieval = FaissVectorRetrieval("text_embedding", db_config=self.db_config)

        # Assert
        assert mock_download.call_count == 2  # Index and mapping files
        mock_read_index.assert_called_once()

    @patch("src.backend.retrieval.download_from_gcs")
    @patch("src.backend.retrieval.os.path.exists")
    def test_faiss_vector_retrieval_download_failure(self, mock_exists, mock_download):
        """Test FAISS retrieval when GCS download fails"""
        # Arrange
        mock_exists.return_value = False
        mock_download.side_effect = Exception("Download failed")

        # Act & Assert
        with pytest.raises(
            FileNotFoundError,
            match=r"Failed to download FAISS index from GCS:",
        ):
            FaissVectorRetrieval("text_embedding", db_config=self.db_config)


class TestTextSearchRetrieval:
    """Test suite for TextSearchRetrieval class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.db_config = {
            "dbname": "test_db",
            "user": "test_user",
            "password": "test_pass",
            "host": "localhost",
            "port": "5432",
        }
        self.retrieval = TextSearchRetrieval("ts_rank", self.db_config)

    @patch("src.backend.retrieval.psycopg2.connect")
    def test_text_search_retrieval_success(self, mock_connect):
        """Test successful text search retrieval"""
        # Arrange
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection

        # Mock database results
        mock_cursor.fetchall.return_value = [
            ("pid1", 0.9),
            ("pid2", 0.8),
            ("pid3", 0.7),
        ]

        query = "test search query"

        # Act
        result = self.retrieval.score(query, k=3)

        # Assert
        expected_result = {"pid1": 0.9, "pid2": 0.8, "pid3": 0.7}
        assert result == expected_result

        mock_connect.assert_called_once_with(**self.db_config)
        mock_cursor.execute.assert_called_once()
        mock_cursor.close.assert_called_once()
        mock_connection.close.assert_called_once()

    @patch("src.backend.retrieval.psycopg2.connect")
    def test_text_search_retrieval_no_results(self, mock_connect):
        """Test text search with no matching results"""
        # Arrange
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection

        mock_cursor.fetchall.return_value = []

        query = "nonexistent query"

        # Act
        result = self.retrieval.score(query, k=3)

        # Assert
        assert result == {}


class TestCreateRetrievalComponent:
    """Test suite for create_retrieval_component function"""

    def setup_method(self):
        """Setup test fixtures"""
        self.db_config = {"dbname": "test_db"}

    def test_create_postgres_vector_retrieval(self):
        """Test creating PostgresVectorRetrieval component"""
        config = {
            "type": "PostgresVectorRetrieval",
            "params": {"column_name": "text_embedding"},
        }

        component = create_retrieval_component(config, self.db_config)

        assert isinstance(component, PostgresVectorRetrieval)
        assert component.column_name == "text_embedding"
        assert component.db_config == self.db_config

    def test_create_text_search_retrieval(self):
        """Test creating TextSearchRetrieval component"""
        config = {"type": "TextSearchRetrieval", "params": {"rank_method": "ts_rank"}}

        component = create_retrieval_component(config, self.db_config)

        assert isinstance(component, TextSearchRetrieval)
        assert component.method == "ts_rank"
        assert component.db_config == self.db_config

    @patch("src.backend.retrieval.faiss.read_index")
    @patch("src.backend.retrieval.os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data='{"0": "pid1"}')
    def test_create_faiss_vector_retrieval(
        self, mock_file, mock_exists, mock_read_index
    ):
        """Test creating FaissVectorRetrieval component"""
        mock_exists.return_value = True
        mock_index = MagicMock()
        mock_read_index.return_value = mock_index

        config = {
            "type": "FaissVectorRetrieval",
            "params": {"column_name": "text_embedding", "nprobe": 2},
        }

        component = create_retrieval_component(config, self.db_config)

        assert isinstance(component, FaissVectorRetrieval)
        assert component.column_name == "text_embedding"

    def test_create_unknown_component_type(self):
        """Test creating component with unknown type"""
        config = {"type": "UnknownRetrieval", "params": {}}

        with pytest.raises(
            ValueError, match="Unknown component type: UnknownRetrieval"
        ):
            create_retrieval_component(config, self.db_config)


class TestHybridRetrieval:
    """Test suite for hybrid_retrieval function"""

    def test_hybrid_retrieval_success(self):
        """Test successful hybrid retrieval"""
        # Arrange
        mock_vector_component = MagicMock(spec=PostgresVectorRetrieval)
        mock_vector_component.score.return_value = {"pid1": 0.9, "pid2": 0.7}

        mock_text_component = MagicMock(spec=TextSearchRetrieval)
        mock_text_component.score.return_value = {"pid1": 0.8, "pid3": 0.6}

        components = [mock_vector_component, mock_text_component]
        weights = [0.6, 0.4]
        query = "test query"
        query_embedding = np.array([0.1, 0.2, 0.3])

        # Act
        pids, scores = hybrid_retrieval(
            query, query_embedding, components, weights, top_k=3
        )

        # Assert
        assert len(pids) == 3
        assert len(scores) == 3
        assert pids[0] == "pid1"  # Should have highest combined score

        # Verify components were called correctly
        mock_vector_component.score.assert_called_once_with(query_embedding, k=3)
        mock_text_component.score.assert_called_once_with(query, k=3)

    def test_hybrid_retrieval_zero_weights(self):
        """Test hybrid retrieval with some zero weights"""
        # Arrange
        mock_component1 = MagicMock()
        mock_component1.score.return_value = {"pid1": 0.9}

        mock_component2 = MagicMock()
        # This component should not be called due to zero weight

        components = [mock_component1, mock_component2]
        weights = [1.0, 0.0]  # Second component has zero weight
        query = "test query"
        query_embedding = np.array([0.1, 0.2, 0.3])

        # Act
        pids, scores = hybrid_retrieval(
            query, query_embedding, components, weights, top_k=1
        )

        # Assert
        assert len(pids) == 1
        assert pids[0] == "pid1"

        # Verify only the first component was called
        mock_component1.score.assert_called_once()
        mock_component2.score.assert_not_called()

    def test_hybrid_retrieval_empty_results(self):
        """Test hybrid retrieval with no results"""
        # Arrange
        mock_component = MagicMock()
        mock_component.score.return_value = {}

        components = [mock_component]
        weights = [1.0]
        query = "test query"
        query_embedding = np.array([0.1, 0.2, 0.3])

        # Act
        pids, scores = hybrid_retrieval(
            query, query_embedding, components, weights, top_k=1
        )

        # Assert
        assert pids == []
        assert scores == []


class TestReorderSearchResultsByRelevancy:
    """Test suite for reorder_search_results_by_relevancy function using LangChain"""

    def setup_method(self):
        """Setup test fixtures"""
        self.sample_results = [
            {
                "Pid": "pid1",
                "Name": "Red Shirt",
                "Brand": "Nike",
                "Category": "Clothing",
                "Color": "Red",
                "Gender": "Male",
                "Size": "M",
            },
            {
                "Pid": "pid2",
                "Name": "Blue Jeans",
                "Brand": "Levi's",
                "Category": "Clothing",
                "Color": "Blue",
                "Gender": "Unisex",
                "Size": "L",
            },
            {
                "Pid": "pid3",
                "Name": "Green Shoes",
                "Brand": "Adidas",
                "Category": "Footwear",
                "Color": "Green",
                "Gender": "Female",
                "Size": "S",
            },
        ]

    @patch("src.backend.retrieval.ChatOpenAI")
    @patch("src.backend.retrieval.PydanticOutputParser")
    @patch("src.backend.retrieval.ChatPromptTemplate")
    def test_reorder_search_results_invalid_indices(
        self, mock_prompt_template, mock_parser, mock_chat_openai
    ):
        """Test reordering with invalid indices from LLM"""
        # Arrange
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        mock_parser_instance = MagicMock()
        mock_parser.return_value = mock_parser_instance

        mock_prompt_instance = MagicMock()
        mock_prompt_template.from_messages.return_value = mock_prompt_instance

        # Mock the chain result with invalid indices
        mock_result = ReorderOutput(
            reordered_indices=[0, 1, 5],  # Invalid index 5
            reasoning="Invalid reordering",
        )

        # Mock the chain invoke
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result
        mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)
        mock_llm.__or__ = MagicMock(return_value=mock_chain)

        query = "test query"

        # Act
        result, _ = reorder_search_results_by_relevancy(
            query, self.sample_results, api_key="test_key"
        )

        # Assert - Should return original order due to invalid indices
        assert result == self.sample_results

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env_api_key"})
    @patch("src.backend.retrieval.ChatOpenAI")
    @patch("src.backend.retrieval.PydanticOutputParser")
    @patch("src.backend.retrieval.ChatPromptTemplate")
    def test_reorder_search_results_env_api_key(
        self, mock_prompt_template, mock_parser, mock_chat_openai
    ):
        """Test reordering using API key from environment"""
        # Arrange
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        mock_parser_instance = MagicMock()
        mock_parser.return_value = mock_parser_instance

        mock_prompt_instance = MagicMock()
        mock_prompt_template.from_messages.return_value = mock_prompt_instance

        # Mock the chain result
        mock_result = ReorderOutput(
            reordered_indices=[0, 1, 2], reasoning="Test reasoning"
        )

        # Mock the chain invoke
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result
        mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)
        mock_llm.__or__ = MagicMock(return_value=mock_chain)

        query = "test query"

        # Act
        result, _ = reorder_search_results_by_relevancy(query, self.sample_results)

        # Assert
        assert len(result) == 3
        # Verify that ChatOpenAI was called with the environment API key
        mock_chat_openai.assert_called_once()
        call_args = mock_chat_openai.call_args
        assert call_args[1]["openai_api_key"] == "env_api_key"


class TestReorderOutput:
    """Test suite for ReorderOutput Pydantic model"""

    def test_reorder_output_valid(self):
        """Test valid ReorderOutput creation"""
        output = ReorderOutput(
            reordered_indices=[2, 0, 1], reasoning="Reordered based on relevance"
        )

        assert output.reordered_indices == [2, 0, 1]
        assert output.reasoning == "Reordered based on relevance"

    def test_reorder_output_invalid(self):
        """Test invalid ReorderOutput creation"""
        with pytest.raises(Exception):  # Pydantic validation error
            ReorderOutput(
                reordered_indices="invalid",  # Should be list
                reasoning="Test reasoning",
            )
