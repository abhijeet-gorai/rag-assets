import os
from typing import List, Dict, Union, Optional, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class ChromaRetriever:
    """
    Provides methods for adding, deleting, and retrieving documents from a Chroma vector database.

    Args:
        embedding_modelname (str, optional): The name of the Hugging Face embeddings model to use.
            Defaults to "intfloat/multilingual-e5-small".
        collection_name (str, optional): The name of the Chroma collection to use. Defaults to "chroma_collection".
        persist_directory (str, optional): The directory to persist the Chroma database.
            Defaults to the value of the environment variable CHROMA_PERSIST_DIRECTORY, or "./chroma_db" if not set.
    """

    def __init__(
        self,
        embedding_modelname: str = "intfloat/multilingual-e5-small",
        collection_name: str = "chroma_collection",
        persist_directory: Optional[str] = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
    ) -> None:
        model_kwargs = {"device": "cpu", "trust_remote_code": True}
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_modelname, model_kwargs=model_kwargs
        )
        self._vector_db = Chroma(
            embedding_function=self.embeddings_model,
            persist_directory=self.persist_directory,
            collection_name=collection_name,
        )

    def set_collection(self, collection_name):
        if collection_name != self.collection_name:
            self.collection_name = collection_name
            self._vector_db = Chroma(
                embedding_function=self.embeddings_model,
                persist_directory=self.persist_directory,
                collection_name=collection_name,
            )

    def _build_filter(self, metadata_filter: Dict[str, Union[str, int, float]]) -> Optional[Dict]:
        """
        Builds a metadata filter dictionary for Chroma queries.
        If more than one filter is provided, combines them using the $and operator.
        """
        if not metadata_filter:
            return None
        if len(metadata_filter) == 1:
            return metadata_filter
        # For multiple filters, use $and to combine conditions.
        return {"$and": [{key: value} for key, value in metadata_filter.items()]}

    def add_documents(self, docs: List[Document]) -> List[str]:
        """
        Adds documents to the Chroma vector database.

        Args:
            docs (List[Document]): A list of Document objects to add.

        Returns:
            List[str]: A list of document IDs for the added documents.
        """
        return self._vector_db.add_documents(docs)

    def delete_documents_by_metadata(self, metadata_filter: Dict[str, Union[str, int, float]]) -> bool:
        """
        Deletes documents from the Chroma vector database based on their metadata.

        Args:
            metadata_filter (Dict[str, Union[str, int, float]]): A dictionary specifying the metadata conditions for deletion.
                Example:
                    {
                        "author": "John Doe",
                        "publication_year": 2021
                    }

        Returns:
            bool: True if the deletion was successful, False otherwise.
        """
        filter_dict = self._build_filter(metadata_filter)
        try:
            self._vector_db.delete(where=filter_dict)
            return True
        except Exception:
            return False

    def get_relevant_docs_with_similarity_score(
        self, query: str, k: int = 10, metadata_filter: Dict[str, Union[str, int, float]] = {}
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves relevant documents from the Chroma vector database along with their similarity scores.

        Args:
            query (str): The query string for retrieving relevant documents.
            k (int, optional): The number of documents to retrieve. Defaults to 10.
            metadata_filter (Dict[str, Union[str, int, float]], optional): Metadata conditions to filter results.
                Defaults to an empty dictionary.
                Example:
                    {
                        "author": "John Doe",
                        "publication_year": 2021
                    }

        Returns:
            List[Tuple[Document, float]]: A list of tuples containing relevant documents and their similarity scores.
        """
        filter_dict = self._build_filter(metadata_filter)
        return self._vector_db.similarity_search_with_score(query, k=k, filter=filter_dict)

    def get_relevant_docs(
        self, query: str, k: int = 10, metadata_filter: Dict[str, Union[str, int, float]] = {}
    ) -> List[Document]:
        """
        Retrieves relevant documents from the Chroma vector database based on a query.

        Args:
            query (str): The query string for retrieving relevant documents.
            k (int, optional): The number of documents to retrieve. Defaults to 10.
            metadata_filter (Dict[str, Union[str, int, float]], optional): Metadata conditions to filter results.
                Defaults to an empty dictionary.
                Example:
                    {
                        "author": "John Doe",
                        "publication_year": 2021
                    }

        Returns:
            List[Document]: A list of relevant documents retrieved from the Chroma vector database.
        """
        docs_with_scores = self.get_relevant_docs_with_similarity_score(query, k, metadata_filter)
        return [doc for doc, _ in docs_with_scores]
