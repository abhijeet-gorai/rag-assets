import os
from typing import List, Dict, Tuple, Optional, Union
from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


class ChromaParentChildRetriever:
    def __init__(
        self,
        parent_doc_path: str = "./parent_docs",
        embedding_modelname: str = "intfloat/multilingual-e5-small",
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 400,
        parent_chunk_overlap: int = 200,
        child_chunk_overlap: int = 40,
        separators: List[str] = ["\n\n", " "],
        collection_name: str = "chroma_collection",
        persist_directory: Optional[str] = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
    ) -> None:
        """
        Initializes the ChromaParentChildRetriever.

        Args:
            parent_doc_path (str): Path to store parent documents. Defaults to "./parent_docs".
            embedding_modelname (str, optional): Name of the Hugging Face embedding model.
                Defaults to "intfloat/multilingual-e5-small".
            parent_chunk_size (int, optional): Size of parent document chunks. Defaults to 2000.
            child_chunk_size (int, optional): Size of child document chunks. Defaults to 400.
            parent_chunk_overlap (int, optional): Overlap in characters between parent chunks. Defaults to 200.
            child_chunk_overlap (int, optional): Overlap in characters between child chunks. Defaults to 40.
            separators (List[str], optional): List of separators for chunking. Defaults to ["\n\n", " "].
            collection_name (str, optional): Name of the Chroma collection. Defaults to "chroma_collection".
            persist_directory (Optional[str], optional): Directory to persist the Chroma database.
                Defaults to the value of the environment variable CHROMA_PERSIST_DIRECTORY or "./chroma_db".
        """
        model_kwargs = {"device": "cpu"}
        embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_modelname, model_kwargs=model_kwargs
        )
        self._vector_db = Chroma(
            embedding_function=embeddings_model,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        self._docstore = self.__get_parent_docstore(parent_doc_path)
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap, separators=separators
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap, separators=separators
        )
        self._retriever = ParentDocumentRetriever(
            vectorstore=self._vector_db,
            docstore=self._docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
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

    def _build_filter(self, metadata_filter: Dict[str, Union[str, int, float]]) -> Optional[Dict]:
        """
        Constructs a metadata filter for Chroma.
        When more than one filter is passed, the conditions are combined using the $and operator.
        """
        if not metadata_filter:
            return None
        if len(metadata_filter) == 1:
            return metadata_filter
        return {"$and": [{key: value} for key, value in metadata_filter.items()]}

    def add_documents(self, docs: List[Document]) -> None:
        """
        Adds documents to the retriever.

        Args:
            docs (List[Document]): List of documents to add.
        """
        self._retriever.add_documents(docs)

    def get_relevant_docs(
        self, query: str, k: int = 4, metadata_filter: Dict[str, Union[str, int, float]] = {}
    ) -> List[Document]:
        """
        Retrieves relevant documents for a given query.

        Args:
            query (str): Search query.
            k (int, optional): Number of top documents to retrieve. Defaults to 4.
            metadata_filter (Dict[str, Union[str, int, float]], optional): Metadata conditions to filter results.
                Example:
                    {
                        "author": "John Doe",
                        "publication_year": 2021
                    }

        Returns:
            List[Document]: List of relevant documents.
        """
        filter_dict = self._build_filter(metadata_filter)
        self._retriever.search_kwargs = {"filter": filter_dict}
        return self._retriever.invoke(query)

    def get_relevant_docs_with_similarity_score(
        self, query: str, k: int = 4, metadata_filter: Dict[str, Union[str, int, float]] = {}
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves relevant documents along with their similarity scores.

        Args:
            query (str): Search query.
            k (int, optional): Number of top documents to retrieve. Defaults to 4.
            metadata_filter (Dict[str, Union[str, int, float]], optional): Metadata conditions to filter results.
                Example:
                    {
                        "author": "John Doe",
                        "publication_year": 2021
                    }

        Returns:
            List[Tuple[Document, float]]: List of tuples containing documents and their similarity scores.
        """
        filter_dict = self._build_filter(metadata_filter)
        # Perform similarity search with scores using the Chroma vector store and filter.
        sub_docs = self._vector_db.similarity_search_with_score(query, filter=filter_dict)
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
