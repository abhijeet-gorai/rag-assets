import os
import hashlib
import json
import datetime
from typing import List, Dict, Tuple, Optional, Any
from elasticsearch import Elasticsearch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore


class ElasticsearchRetriever:
    def __init__(
        self,
        embedding_modelname: str = "intfloat/multilingual-e5-small",
        es_index: Optional[str] = os.getenv("ES_INDEX"),
        es_url: Optional[str] = os.getenv("ES_URL"),
        es_user: Optional[str] = os.getenv("ES_USER"),
        es_password: Optional[str] = os.getenv("ES_PASSWORD"),
        es_cert: Optional[str] = os.getenv("ES_CERT"),
    ) -> None:
        """
        Initializes the ElasticsearchRetriever with the provided configuration.

        Args:
            embedding_modelname (str): Name of the embedding model to use. Defaults to "intfloat/multilingual-e5-small".
            es_index (Optional[str]): Elasticsearch index name.
            es_url (Optional[str]): Elasticsearch URL.
            es_user (Optional[str]): Elasticsearch username.
            es_password (Optional[str]): Elasticsearch password.
            es_cert (Optional[str]): Path to Elasticsearch certificate.
        """
        if not all([es_url, es_user, es_password, es_index]):
            raise ValueError(
                "ES_URL, ES_USER, ES_PASSWORD and ES_INDEX are required for connecting to elasticsearch instance. "
                "Either set them as environment variables or provide them during initialization."
            )
        model_kwargs = {"device": "cpu"}
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_modelname, model_kwargs=model_kwargs
        )
        self.es_index = es_index
        self.es = Elasticsearch(
            [es_url],
            basic_auth=(es_user, es_password),
            ca_certs=es_cert,
            verify_certs=True,
        )
        self._retriever = ElasticsearchStore(
            es_connection=self.es, index_name=es_index, embedding=self.embeddings_model
        )

    def _get_unique_id(self, document: Document) -> str:
        """
        Generates a unique ID for a document using SHA-256 hashing.

        Args:
            document (Document): The document to generate an ID for.

        Returns:
            str: The unique document ID.
        """
        content_to_hash = document.page_content + json.dumps(
            document.metadata, sort_keys=True
        )
        document_id = hashlib.sha256(content_to_hash.encode("utf-8")).hexdigest()
        return document_id

    def add_documents(self, docs: List[Document]) -> List[str]:
        """
        Adds a list of documents to the Elasticsearch index.

        Args:
            docs (List[Document]): A list of documents to be added.

        Returns:
            List[str]: A list of document IDs that were added.
        """
        ids = [self._get_unique_id(doc) for doc in docs]
        self._retriever.add_documents(docs, ids=ids)
        return ids

    def get_relevant_docs_with_similarity_score(
        self,
        query: str,
        metadata_filter: Dict[str, Tuple[Any, bool]] = {},
        k: int = 10,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves documents relevant to the provided query along with their similarity scores.

        Args:
            query (str): The query string to search for.
            metadata_filter (Dict[str, Tuple[Any, bool]], optional): A dictionary mapping metadata fields to a tuple of 
                                                           (value, fuzzy_search), where `fuzzy_search` is a boolean
                                                           indicating if fuzzy matching should be used. Defaults to {}.
                Example:
                    {
                        "author": ("John Doe", True),
                        "publication_year": (2021, False)
                    }
            k (int): The number of top relevant documents to return. Defaults to 10.

        Returns:
            List[Tuple[Document, float]]: A list of tuples where each tuple contains a relevant document 
                                          and its corresponding similarity score.
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
        return self._retriever.similarity_search_with_score(query, **search_kwargs)

    def get_relevant_docs(
        self,
        query: str,
        metadata_filter: Dict[str, Tuple[Any, bool]] = {},
        k: int = 10,
    ) -> List[Document]:
        """
        Retrieves documents relevant to the provided query and metadata filter.

        Args:
            query (str): The query string to search for.
            metadata_filter (Dict[str, Tuple[Any, bool]], optional): A dictionary mapping metadata fields to a tuple of
                                                           (value, fuzzy_search), where `fuzzy_search` is a boolean
                                                           indicating if fuzzy matching should be used. Defaults to {}.
                Example:
                    {
                        "author": ("John Doe", True),
                        "publication_year": (2021, False)
                    }
            k (int): The number of top relevant documents to return. Defaults to 10.

        Returns:
            List[Document]: A list of relevant documents.
        """
        docs_with_scores = self.get_relevant_docs_with_similarity_score(query, metadata_filter, k)
        return [doc[0] for doc in docs_with_scores]
    
    def create_document(self, document_id: str, document: Dict[str, Any]) -> None:
        """
        Creates a new document in the Elasticsearch index.

        Args:
            document_id (str): The unique ID of the document.
            document (Dict[str, Any]): The document data to be indexed.
        """
        document["last_updated"] = datetime.datetime.now(datetime.UTC)
        self.es.index(index=self.es_index, id=document_id, document=document)

    def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """
        Updates an existing document in the Elasticsearch index.

        Args:
            document_id (str): The ID of the document to update.
            updates (Dict[str, Any]): The updates to apply to the document.

        Returns:
            bool: True if the update was successful.
        """
        updates["last_updated"] = datetime.datetime.now(datetime.UTC)
        update_body = {"doc": updates}
        self.es.update(index=self.es_index, id=document_id, body=update_body)
        return True

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a document by its ID.

        Args:
            document_id (str): The ID of the document to retrieve.

        Returns:
            Optional[Dict[str, Any]]: The document data if found, otherwise None.
        """
        try:
            result = self.es.get(index=self.es_index, id=document_id)
            return result["_source"]
        except Exception as e:
            print(f"Document not found: {e}")
            return None

    def delete_document(self, document_id: str) -> bool:
        """
        Deletes a document from the Elasticsearch index by its ID.

        Args:
            document_id (str): The ID of the document to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            self.es.delete(index=self.es_index, id=document_id)
            return True
        except Exception as e:
            print(f"Failed to delete document: {e}")
            return False

    def hybrid_search(
        self,
        query: str,
        keyword_field: str,
        vector_field: str,
        num_results: int = 10,
        hybrid_weight: float = 0.7,
    ):
        """
        Retrieves documents relevant to the provided query through hybrid search (keyword + semantic).

        Args:
            query (str): The query string to search for.
            keyword_field (str): The field name on which the keyword search to happen.
            vector_field (str): The field name on which the semantic search to happen.
            num_results (int): The number of top relevant documents to return. Defaults to 10.
            hybrid_weight (float): The hybrid weight for the search. Defaults to 0.7.

        Returns:
            List[Document]: A list of relevant documents.
        """
        
        # Get query embeddings
        query_vector = self.embeddings_model.embed_query(query)

        # Construct hybrid search query
        search_query = {
            "size": num_results,
            "query": {
                "script_score": {
                    "query": {
                        "bool": {
                            "should": [ # must
                                {
                                    "match": {
                                        keyword_field: {
                                            "query": query,
                                            "boost": 1 - hybrid_weight,
                                        }
                                    }
                                },
                            ]
                        }
                    },
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, '{vector_field}') * {hybrid_weight}",
                        "params": {"query_vector": query_vector},
                    },
                }
            },
        }

        try:
            response = self.client.search(index=self.es_index, body=search_query)
            return response

        except Exception as e:
            return []