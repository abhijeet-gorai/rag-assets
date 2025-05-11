import os
from typing import List, Dict, Tuple, Optional, Union
from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus.vectorstores import Milvus


class MilvusParentChildRetriever:
    def __init__(
        self,
        parent_doc_path: str = "./parent_docs",
        embedding_modelname: str = "intfloat/multilingual-e5-small",
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 400,
        parent_chunk_overlap: int = 200,
        child_chunk_overlap: int = 40,
        separators: List[str] = ["\n\n", " "],
        collection_name: str = "milvus_collection",
        milvus_uri: Optional[str] = os.getenv("MILVUS_URI"),
        milvus_user: Optional[str] = os.getenv("MILVUS_USER"),
        milvus_password: Optional[str] = os.getenv("MILVUS_PASSWORD"),
    ) -> None:
        """
        Initializes the MilvusParentChildRetriever.

        Args:
            parent_doc_path (str): Path to store parent documents. Defaults to "./parent_docs".
            embedding_modelname (str, optional): Name of the Hugging Face embedding model. Defaults to "intfloat/multilingual-e5-small".
            parent_chunk_size (int, optional): Size of parent document chunks. Defaults to 2000.
            child_chunk_size (int, optional): Size of child document chunks. Defaults to 400.
            parent_chunk_overlap (int, optional): Overlap in characters between parent chunks. Defaults to 200.
            child_chunk_overlap (int, optional): Overlap in characters between child chunks. Defaults to 40.
            separators (List[str], optional): List of separators for chunking. Defaults to ["\n\n", " "].
            collection_name (str, optional): Name of the Milvus collection. Defaults to "milvus_collection".
            milvus_uri (Optional[str], optional): Milvus database URI. Defaults to value from environment variable MILVUS_URI.
            milvus_user (Optional[str], optional): Milvus database username. Defaults to value from environment variable MILVUS_USER.
            milvus_password (Optional[str], optional): Milvus database password. Defaults to value from environment variable MILVUS_PASSWORD.
        """
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
        self.collection_name = collection_name
        model_kwargs = {"device": "cpu"}
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_modelname, model_kwargs=model_kwargs
        )
        self._vector_db = Milvus(
            self.embeddings_model,
            connection_args=self.connection_args,
            collection_name=self.collection_name,
            auto_id=True,
        )
        self._docstore = self.__get_parent_docstore(parent_doc_path)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap, separators=separators
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap, separators=separators
        )
        self._retriever = ParentDocumentRetriever(
            vectorstore=self._vector_db,
            docstore=self._docstore,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )

    def set_collection(self, collection_name: str):
        """
        Switch the Milvus collection for both vector store and retriever.
        """
        if collection_name and collection_name != self.collection_name:
            self.collection_name = collection_name
            # recreate vector DB with new collection
            self._vector_db = Milvus(
                self.embeddings_model,
                connection_args=self.connection_args,
                collection_name=self.collection_name,
                auto_id=True,
            )
            # update retriever's vectorstore
            self._retriever.vectorstore = self._vector_db

    def __get_parent_docstore(self, parent_doc_path: str) -> BaseStore[str, Document]:
        fs = LocalFileStore(parent_doc_path)
        return create_kv_docstore(fs)

    def add_documents(self, docs: List[Document], collection_name: Optional[str] = None) -> None:
        """
        Adds documents to the retriever's vector store and docstore.

        Args:
            docs (List[Document]): List of documents to add.
            collection_name (str, optional): If provided, switch to this collection first.
        """
        if collection_name:
            self.set_collection(collection_name)
        self._retriever.add_documents(docs)

    def get_relevant_docs(
        self,
        query: str,
        k: int = 4,
        metadata_filter: Dict[str, Union[str, int, float]] = {},
        collection_name: Optional[str] = None,
    ) -> List[Document]:
        """
        Retrieves relevant documents for a given query.

        Args:
            query (str): Search query.
            k (int, optional): Number of top documents to retrieve. Defaults to 4.
            metadata_filter (Dict[str, Union[str, int, float]], optional): Metadata conditions to filter results. Defaults to {}.
                Example:
                    {
                        "author": "John Doe",
                        "publication_year": 2021
                    }
            collection_name (str, optional): If provided, switch to this collection first.

        Returns:
            List[Document]: List of relevant documents.
        """
        if collection_name:
            self.set_collection(collection_name)
        expr_list = [
            f'{key}=="{value}"' if isinstance(value, str) else f"{key}=={value}"
            for key, value in metadata_filter.items()
        ]
        expr = " and ".join(expr_list) if expr_list else None
        self._retriever.search_kwargs = {"expr": expr, "k": k}
        return self._retriever.invoke(query)

    def get_relevant_docs_with_similarity_score(
        self,
        query: str,
        k: int = 4,
        metadata_filter: Dict[str, Union[str, int, float]] = {},
        collection_name: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves relevant documents along with their similarity scores.

        Args:
            query (str): Search query.
            k (int, optional): Number of top documents to retrieve. Defaults to 4.
            metadata_filter (Dict[str, Union[str, int, float]], optional): Metadata conditions to filter results. Defaults to {}.
                Example:
                    {
                        "author": "John Doe",
                        "publication_year": 2021
                    }
            collection_name (str, optional): If provided, switch to this collection first.

        Returns:
            List[Tuple[Document, float]]: List of tuples containing documents and their similarity scores.
        """
        if collection_name:
            self.set_collection(collection_name)
        expr_list = [
            f'{key}=="{value}"' if isinstance(value, str) else f"{key}=={value}"
            for key, value in metadata_filter.items()
        ]
        expr = " and ".join(expr_list) if expr_list else None
        sub_docs = self._vector_db.similarity_search_with_score(query, k=k, expr=expr)
        id_key = self._retriever.id_key
        ids, scores = [], []
        for d, score in sub_docs:
            if id_key in d.metadata and d.metadata[id_key] not in ids:
                ids.append(d.metadata[id_key])
                scores.append(score)
        docs = self._docstore.mget(ids)
        return [(doc, score) for doc, score in zip(docs, scores) if doc is not None]