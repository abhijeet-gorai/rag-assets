import os
from typing import List, Dict, Union, Optional, Tuple
from langchain_core.documents import Document
from langchain_milvus.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings


class MilvusRetriever:
    """
    Provides methods for adding, deleting, and retrieving documents from a Milvus vector database.

    Args:
        embeddings_model_name (str, optional): The name of the Hugging Face embeddings model to use. Defaults to "intfloat/multilingual-e5-small".
        collection_name (str, optional): The name of the Milvus collection to use. Defaults to "milvus_collection".
        milvus_uri (str, optional): The URI of the Milvus instance to connect to. Defaults to the value of the environment variable MILVUS_URI.
        milvus_user (str, optional): The username for connecting to the Milvus instance. Defaults to the value of the environment variable MILVUS_USER.
        milvus_password (str, optional): The password for connecting to the Milvus instance. Defaults to the value of the environment variable MILVUS_PASSWORD.
    
    """

    def __init__(
        self,
        embedding_modelname: str = "intfloat/multilingual-e5-small",
        collection_name: str = "milvus_collection",
        milvus_uri: Optional[str] = os.getenv("MILVUS_URI"),
        milvus_user: Optional[str] = os.getenv("MILVUS_USER"),
        milvus_password: Optional[str] = os.getenv("MILVUS_PASSWORD"),
    ) -> None:
        if not all([milvus_uri, milvus_user, milvus_password]):
            raise ValueError(
                "MILVUS_URI, MILVUS_USER, and MILVUS_PASSWORD are required for connecting to the Milvus instance. "
                "Either set them as environment variables or provide them during initialization."
            )
        self.connection_args = {
            "uri": milvus_uri,
            "user": milvus_user,
            "password": milvus_password,
        }
        model_kwargs = {"device": "cpu", "trust_remote_code": True}
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_modelname, model_kwargs=model_kwargs
        )
        self.collection_name = collection_name
        self._vector_db = Milvus(
            self.embeddings_model,
            connection_args=self.connection_args,
            collection_name=self.collection_name,
            auto_id=True,
        )

    def set_collection(self, collection_name: str):
        if collection_name and collection_name != self.collection_name:
            self.collection_name = collection_name
            self._vector_db = Milvus(
                self.embeddings_model,
                connection_args=self.connection_args,
                collection_name=self.collection_name,
                auto_id=True,
            )

    def add_documents(self, docs: List[Document], collection_name: Optional[str] = None) -> List[str]:
        """
        Adds documents to the Milvus vector database.

        Args:
            docs (List[Document]): A list of Document objects to add.
            collection_name (str, optional): If provided, switch to this collection before adding.

        Returns:
            List[str]: A list of document IDs for the added documents.
        """
        if collection_name:
            self.set_collection(collection_name)
        return self._vector_db.add_documents(docs)

    def delete_documents_by_metadata(
        self,
        metadata_filter: Dict[str, Union[str, int, float]],
        collection_name: Optional[str] = None,
    ) -> bool:
        """
        Deletes documents from the Milvus vector database based on their metadata.

        Args:
            metadata_filter (Dict[str, Union[str, int, float]]): A dictionary specifying the metadata conditions for deletion.
            collection_name (str, optional): If provided, switch to this collection before deleting.

        Returns:
            bool: True if the deletion was successful, False otherwise.
        """
        if collection_name:
            self.set_collection(collection_name)
        expr_list = [
            f'{key}=="{value}"' if isinstance(value, str) else f"{key}=={value}"
            for key, value in metadata_filter.items()
        ]
        expr = " and ".join(expr_list) if expr_list else None

        try:
            self._vector_db.delete(expr=expr)
            return True
        except Exception:
            return False
    
    def get_relevant_docs_with_similarity_score(
        self,
        query: str,
        k: int = 10,
        metadata_filter: Dict[str, Union[str, int, float]] = {},
        collection_name: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves relevant documents from the Milvus vector database along with their similarity scores.

        Args:
            query (str): The query string for retrieving relevant documents.
            k (int, optional): The number of documents to retrieve. Defaults to 10.
            metadata_filter (Dict[str, Union[str, int, float]], optional): Metadata conditions to filter results. Defaults to an empty dictionary.
                Example:
                    {
                        "author": "John Doe",
                        "publication_year": 2021
                    }
            collection_name (str, optional): If provided, switch to this collection before querying.

        Returns:
            List[Tuple[Document, float]]: A list of tuples containing relevant documents and their similarity scores.
        """
        if collection_name:
            self.set_collection(collection_name)
        expr_list = [
            f'{key}=="{value}"' if isinstance(value, str) else f"{key}=={value}"
            for key, value in metadata_filter.items()
        ]
        expr = " and ".join(expr_list) if expr_list else None
        return self._vector_db.similarity_search_with_score(query, k=k, expr=expr)

    def get_relevant_docs(
        self,
        query: str,
        k: int = 10,
        metadata_filter: Dict[str, Union[str, int, float]] = {},
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        """
        Retrieves relevant documents from the Milvus vector database based on a query.

        Args:
            query (str): The query string for retrieving relevant documents.
            k (int, optional): The number of documents to retrieve. Defaults to 10.
            metadata_filter (Dict[str, Union[str, int, float]], optional): Metadata conditions to filter results. Defaults to an empty dictionary.
                Example:
                    {
                        "author": "John Doe",
                        "publication_year": 2021
                    }
            collection_name (str, optional): If provided, switch to this collection before querying.

        Returns:
            List[Document]: A list of relevant documents retrieved from the Milvus vector database.
        """
        if collection_name:
            self.set_collection(collection_name)
        docs_with_scores = self.get_relevant_docs_with_similarity_score(query, k, metadata_filter)
        return [doc[0] for doc in docs_with_scores]
