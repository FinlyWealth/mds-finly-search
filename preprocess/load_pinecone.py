import pandas as pd

# import preprocess.load_pinecone as load_pinecone
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import os
from time import sleep


API_KEY = "pcsk_5vb9xu_JLszt4zrHGwpTvXy9EsJQH8KTY49q6RcgXAA8f2YXzUNeTsBg9t8gYPepWSTfic"

COLS = [
    "Name",
    "Category",
    "Price",
    "PriceCurrency",
    "FinalPrice",
    "Discount",
    "isOnSale",
    "IsInStock",
    "Color",
    "Gender",
    "Size",
    "Condition",
    "MergedBrand",
    "combined",
]


class PineconeUploader:
    def __init__(self, environment: str, batch_size: int = 100):
        """
        Initialize the PineconeUploader with API credentials and configuration.

        Args:
            api_key (str): Pinecone API key
            environment (str): Pinecone environment
            batch_size (int): Number of vectors to upload in each batch
        """
        self.batch_size = batch_size
        # Initialize Pinecone
        # load_pinecone = load_pinecone.init(api_key=api_key, environment=environment)
        self.load_pinecone = Pinecone(api_key=API_KEY)

    def create_index(
        self, index_name: str, dimension: int, metric: str = "cosine"
    ) -> None:
        """
        Create a Pinecone index if it doesn't exist.

        Args:
            index_name (str): Name of the index to create
            dimension (int): Dimension of the vectors
            metric (str): Distance metric to use (default: cosine)
        """
        indexes = [index.name for index in self.load_pinecone.list_indexes()]
        if index_name not in indexes:
            print(f"Creating index: {index_name}")
            self.load_pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait for index to be ready
            while not self.load_pinecone.describe_index(index_name).status["ready"]:
                sleep(1)
        else:
            print(f"Index {index_name} already exists")

    def prepare_vectors(
        self,
        df: pd.DataFrame,
        embedding_col: str,
        metadata_cols: List[str],
        id_col: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Prepare vectors and metadata for upload to Pinecone.

        Args:
            df (pd.DataFrame): DataFrame containing embeddings and metadata
            embedding_col (str): Column name containing embeddings
            metadata_cols (List[str]): List of column names to include as metadata
            id_col (str): Column to use as vector IDs (optional)

        Returns:
            List[Dict[str, Any]]: List of vectors ready for upload
        """
        vectors = []

        for idx, row in df.iterrows():
            # Convert embedding to list if it's numpy array
            embedding = (
                row[embedding_col].tolist()
                if isinstance(row[embedding_col], np.ndarray)
                else row[embedding_col]
            )

            # Prepare metadata
            metadata = {col: row[col] for col in metadata_cols if col in df.columns}

            # Use provided ID column or default to index
            vector_id = str(row[id_col]) if id_col else str(idx)

            vectors.append({"id": vector_id, "values": embedding, "metadata": metadata})

        return vectors

    def upload_in_batches(self, vectors: List[Dict[str, Any]], index_name: str) -> None:
        """
        Upload vectors to Pinecone in batches.

        Args:
            vectors (List[Dict[str, Any]]): List of vectors to upload
            index_name (str): Name of the index to upload to
        """
        index = self.load_pinecone.Index(index_name)

        # Process in batches
        for i in tqdm(range(0, len(vectors), self.batch_size)):
            batch = vectors[i : i + self.batch_size]
            try:
                index.upsert(vectors=batch)
            except Exception as e:
                print(f"Error uploading batch {i//self.batch_size}: {str(e)}")
                continue

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_config: Dict[str, Any],
        image_config: Dict[str, Any],
    ) -> None:
        """
        Process a dataframe and upload both text and image embeddings to separate indexes.

        Args:
            df (pd.DataFrame): DataFrame containing embeddings and metadata
            text_config (Dict[str, Any]): Configuration for text embeddings
                {
                    'index_name': str,
                    'embedding_col': str,
                    'metadata_cols': List[str],
                    'dimension': int,
                    'id_col': str (optional)
                }
            image_config (Dict[str, Any]): Configuration for image embeddings
                {
                    'index_name': str,
                    'embedding_col': str,
                    'metadata_cols': List[str],
                    'dimension': int,
                    'id_col': str (optional)
                }
        """
        # Create indexes
        self.create_index(text_config["index_name"], text_config["dimension"])
        self.create_index(image_config["index_name"], image_config["dimension"])

        # Prepare and upload text vectors
        print("Processing text embeddings...")
        text_vectors = self.prepare_vectors(
            df,
            text_config["embedding_col"],
            text_config["metadata_cols"],
            text_config.get("id_col"),
        )
        print("DONE")
        print(len(text_vectors))
        print(self.batch_size)
        print("DONEE")
        self.upload_in_batches(text_vectors, text_config["index_name"])

        # Prepare and upload image vectors
        print("Processing image embeddings...")
        image_vectors = self.prepare_vectors(
            df,
            image_config["embedding_col"],
            image_config["metadata_cols"],
            image_config.get("id_col"),
        )
        self.upload_in_batches(image_vectors, image_config["index_name"])


def main():
    # Example usage
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT")

    # Sample configuration
    text_config = {
        "index_name": "product-text-search",
        "embedding_col": "text_embedding",
        "metadata_cols": COLS,
        "dimension": 512,  # Example dimension for text embeddings
        "id_col": "Pid",
    }

    image_config = {
        "index_name": "product-image-search",
        "embedding_col": "image_embedding",
        "metadata_cols": COLS,
        "dimension": 512,  # Example dimension for image embeddings
        "id_col": "Pid",
    }

    # Initialize uploader
    uploader = PineconeUploader(environment=environment, batch_size=100)

    # Example DataFrame structure
    """
    df = pd.DataFrame({
        'product_id': ['1', '2', '3'],
        'product_name': ['Product A', 'Product B', 'Product C'],
        'description': ['Desc A', 'Desc B', 'Desc C'],
        'category': ['Cat A', 'Cat B', 'Cat C'],
        'image_url': ['url1', 'url2', 'url3'],
        'text_embedding': [np.random.rand(768) for _ in range(3)],
        'image_embedding': [np.random.rand(512) for _ in range(3)]
    })
    """

    # Upload to Pinecone
    data_path = ""  # Replace with actual data path
    df = pd.read_csv(data_path)
    uploader.process_dataframe(df, text_config, image_config)


if __name__ == "__main__":
    main()
