from typing import Any, List, Tuple
from langchain.schema import Document
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """
    A class for reranking documents using a cross-encoder model.

    Args:
        model_name (str): The name of the cross-encoder model to use. Defaults to "cross-encoder/ms-marco-MiniLM-L-6-v2".

    Attributes:
        _cross_encoder (CrossEncoder): The cross-encoder model to use for similarity calculation.

    Methods:
        calculate_sim(query: str, documents_text_list: List[str]) -> List[float]:
            Calculates the similarity scores between a query and a list of documents.

        rerank(query: str, docs: List[Document], get_sim_scores: bool = False) -> List[Document] | List[(Document, float)]:
            Reranks a list of documents based on their similarity to a query. If get_sim_scores is True, returns a list of tuples containing the documents and their similarity scores.
    """
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._cross_encoder = CrossEncoder(model_name)

    def calculate_sim(self, query: str, documents_text_list: List[str]):
        """
        Calculates the similarity scores between a query and a list of documents.

        Args:
            query (str): The query string.
            documents_text_list (List[str]): A list of document text strings.

        Returns:
            List[float]: A list of similarity scores between the query and each document.
        """
        pairs = [[query, doc] for doc in documents_text_list]
        scores = self._cross_encoder.predict(pairs)
        return scores
    
    def rerank(self, query: str, docs: List[Document], get_sim_scores:bool=False) -> List[Document] | List[Tuple[Document, float]]:
        """
        Reranks a list of documents based on their similarity to a query. If get_sim_scores is True,
        it returns a list of tuples containing the documents and their similarity scores.

        Args:
            query (str): The query string.
            docs (List[Document]): A list of Document objects.
            get_sim_scores (bool, optional): If True, returns a list of tuples containing the documents and their similarity scores. Defaults to False.

        Returns:
            List[Document] | List[Tuple[Document, float]]: A list of reranked documents or a list of tuples containing the documents and their similarity scores, depending on the value of get_sim_scores.
        """
        documents_text_list = [doc.page_content for doc in docs]
        scores = self.calculate_sim(query=query, documents_text_list=documents_text_list)
        docs_with_score = tuple(zip(docs, scores))
        reranked_docs = sorted(docs_with_score, key=lambda x: x[1], reverse=True)
        if not get_sim_scores:
            reranked_docs = [x[0] for x in reranked_docs]
        return reranked_docs