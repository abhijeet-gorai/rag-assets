import os
from typing import List, Dict, Tuple, Optional, Any
from elasticsearch import Elasticsearch
from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore


class ESParentChildRetriever:
    """
    A retriever that manages parent-child document relationships using Elasticsearch for vector search.
    """

    def __init__(
        self,
        parent_doc_path: str = "./parent_docs",
        embedding_modelname: str = "intfloat/multilingual-e5-small",
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 400,
        parent_chunk_overlap: int = 200,
        child_chunk_overlap: int = 40,
        separators: List[str] = ["\n\n", " "],
        es_index: Optional[str] = os.getenv("ES_INDEX"),
        es_url: Optional[str] = os.getenv("ES_URL"),
        es_user: Optional[str] = os.getenv("ES_USER"),
        es_password: Optional[str] = os.getenv("ES_PASSWORD"),
        es_cert: Optional[str] = os.getenv("ES_CERT"),
    ) -> None:
        """
        Initializes the ESParentChildRetriever.

        Args:
            parent_doc_path (str): Path to store parent documents. Defaults to "./parent_docs".
            embedding_modelname (str, optional): Name of the Hugging Face embedding model. Defaults to "intfloat/multilingual-e5-small".
            parent_chunk_size (int, optional): Size of parent document chunks. Defaults to 2000.
            child_chunk_size (int, optional): Size of child document chunks. Defaults to 400.
            parent_chunk_overlap (int, optional): Overlap in characters between parent chunks. Defaults to 200.
            child_chunk_overlap (int, optional): Overlap in characters between child chunks. Defaults to 40.
            separators (List[str], optional): List of separators for chunking. Defaults to ["\n\n", " "].
            es_index (Optional[str], optional): Elasticsearch index name. Defaults to environment variable ES_INDEX.
            es_url (Optional[str], optional): Elasticsearch URL. Defaults to environment variable ES_URL.
            es_user (Optional[str], optional): Elasticsearch username. Defaults to environment variable ES_USER.
            es_password (Optional[str], optional): Elasticsearch password. Defaults to environment variable ES_PASSWORD.
            es_cert (Optional[str], optional): Path to Elasticsearch SSL certificate. Defaults to environment variable ES_CERT.
        """
        if not all([es_url, es_user, es_password, es_index]):
            raise ValueError(
                "ES_URL, ES_USER, ES_PASSWORD and ES_INDEX are required for connecting to Elasticsearch instance. "
                "Either set them as environment variables or provide them during initialization."
            )
        self._elasticsearch_store = self.__get_elasticsearch_store(
            es_url, es_user, es_password, es_index, es_cert, embedding_modelname
        )
        self._docstore = self.__get_parent_docstore(parent_doc_path)
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap, separators=separators
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap, separators=separators
        )
        self._retriever = ParentDocumentRetriever(
            vectorstore=self._elasticsearch_store,
            docstore=self._docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

    def __get_elasticsearch_store(
        self, es_url: str, es_user: str, es_password: str, es_index: str, es_cert: Optional[str], embedding_modelname: str
    ) -> ElasticsearchStore:
        """
        Initializes the Elasticsearch vector store.

        Args:
            es_url (str): Elasticsearch URL.
            es_user (str): Elasticsearch username.
            es_password (str): Elasticsearch password.
            es_index (str): Elasticsearch index name.
            es_cert (Optional[str]): Path to Elasticsearch SSL certificate.
            embedding_modelname (str): Name of the Hugging Face embedding model.

        Returns:
            ElasticsearchStore: Configured Elasticsearch vector store.
        """
        model_kwargs = {"device": "cpu"}
        embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_modelname, model_kwargs=model_kwargs
        )
        es = Elasticsearch(
            [es_url],
            basic_auth=(es_user, es_password),
            ca_certs=es_cert,
            verify_certs=True,
        )
        return ElasticsearchStore(
            es_connection=es, index_name=es_index, embedding=embeddings_model
        )

    def __get_parent_docstore(self, parent_doc_path: str) -> BaseStore[str, Document]:
        """
        Initializes the parent document store.

        Args:
            parent_doc_path (str): Path to store parent documents.

        Returns:
            BaseStore[str, Document]: A key-value document store.
        """
        fs = LocalFileStore(parent_doc_path)
        return create_kv_docstore(fs)

    def add_documents(self, docs: List[Document]) -> None:
        """
        Adds documents to the retriever.

        Args:
            docs (List[Document]): List of documents to add.
        """
        self._retriever.add_documents(docs)

    def get_relevant_docs(self, query: str, metadata_filter: Dict[str, Tuple[Any, bool]] = {}, k: int = 4) -> List[Document]:
        """
        Retrieves relevant documents for a given query.

        Args:
            query (str): Search query.
            metadata_filter (Dict[str, Tuple[Any, bool]], optional): A dictionary mapping metadata fields to a tuple of 
                                                           (value, fuzzy_search), where `fuzzy_search` is a boolean
                                                           indicating if fuzzy matching should be used. Defaults to {}.
                Example:
                    {
                        "author": ("John Doe", True),
                        "publication_year": (2021, False)
                    }
            k (int, optional): Number of top documents to retrieve. Defaults to 4.

        Returns:
            List[Document]: List of relevant documents.
        """
        search_kwargs = {"k": k, "filter": []}
        for metadata_field, (metadata_value, fuzzy_search) in metadata_filter.items():
            filter_clause = (
                {"match": {f"metadata.{metadata_field}": {"query": metadata_value, "fuzziness": "AUTO"}}}
                if fuzzy_search
                else {"term": {f"metadata.{metadata_field}.keyword": metadata_value}}
            )
            search_kwargs["filter"].append(filter_clause)
        if not search_kwargs["filter"]:
            search_kwargs.pop("filter")
        self._retriever.search_kwargs = search_kwargs
        return self._retriever.invoke(query)

    def get_relevant_docs_with_similarity_score(self, query: str, metadata_filter: Dict[str, Tuple[Any, bool]] = {}, k: int = 4) -> List[Tuple[Document, float]]:
        """
        Retrieves relevant documents along with their similarity scores.

        Args:
            query (str): Search query.
            metadata_filter (Dict[str, Tuple[Any, bool]], optional): A dictionary mapping metadata fields to a tuple of 
                                                           (value, fuzzy_search), where `fuzzy_search` is a boolean
                                                           indicating if fuzzy matching should be used. Defaults to {}.
                Example:
                    {
                        "author": ("John Doe", True),
                        "publication_year": (2021, False)
                    }
            k (int, optional): Number of top documents to retrieve. Defaults to 4.

        Returns:
            List[Tuple[Document, float]]: List of tuples containing documents and their similarity scores.
        """
        search_kwargs = {"k": k, "filter": []}
        for metadata_field, (metadata_value, fuzzy_search) in metadata_filter.items():
            filter_clause = (
                {"match": {f"metadata.{metadata_field}": {"query": metadata_value, "fuzziness": "AUTO"}}}
                if fuzzy_search
                else {"term": {f"metadata.{metadata_field}.keyword": metadata_value}}
            )
            search_kwargs["filter"].append(filter_clause)
        if not search_kwargs["filter"]:
            search_kwargs.pop("filter")
        sub_docs = self._elasticsearch_store.similarity_search_with_score(query, **search_kwargs)
        id_key = self._retriever.id_key
        ids = []
        scores = []
        for d, score in sub_docs:
            if id_key in d.metadata and d.metadata[id_key] not in ids:
                ids.append(d.metadata[id_key])
                scores.append(score)
        docs = self._docstore.mget(ids)
        docs_score = list(zip(docs, scores))
        return [d for d, _ in docs_score if d is not None]