from typing import Any, List, Tuple
from transformers import AutoTokenizer, AutoModel
from langchain_core.documents import Document
import torch

class ColbertReranker:
    """
    A class for reranking documents based on a query using the ColBERT model.

    Attributes:
        _tokenizer (AutoTokenizer): The tokenizer used for tokenizing text.
        _model (AutoModel): The ColBERT model for calculating similarity scores.

    Methods:
        calculate_sim(query: str, documents_text_list: List[str]) -> List[float]:
            Calculates the similarity scores between a query and a list of documents.

        rerank(query: str, docs: List[Document], get_sim_scores: bool = False) -> List[Document] | List[(Document, float)]:
            Reranks a list of documents based on a query and returns the reranked list. If get_sim_scores is True,
            it returns a list of tuples containing the documents and their similarity scores.
    """
    def __init__(self, tokenizer="colbert-ir/colbertv2.0", model="colbert-ir/colbertv2.0"):
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self._model = AutoModel.from_pretrained(model)

    def calculate_sim(self, query: str, documents_text_list: List[str]):
        """
        Calculates the similarity scores between a query and a list of documents.

        Args:
            query (str): The query string.
            documents_text_list (List[str]): A list of document text strings.

        Returns:
            List[float]: A list of similarity scores corresponding to each document in documents_text_list.
        """
        query_encoding = self._tokenizer(query, return_tensors="pt")
        query_embedding = self._model(**query_encoding).last_hidden_state
        rerank_score_list = []

        for document_text in documents_text_list:
            document_encoding = self._tokenizer(
                document_text, return_tensors="pt", truncation=True, max_length=512
            )
            document_embedding = self._model(**document_encoding).last_hidden_state

            sim_matrix = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(2), document_embedding.unsqueeze(1), dim=-1
            )

            # Take the maximum similarity for each query token (across all document tokens)
            # sim_matrix shape: [batch_size, query_length, doc_length]
            max_sim_scores, _ = torch.max(sim_matrix, dim=2)
            rerank_score_list.append(torch.mean(max_sim_scores, dim=1).item())

        return rerank_score_list
    
    def rerank(self, query: str, docs: List[Document], get_sim_scores:bool=False) -> List[Document] | List[Tuple[Document, float]]:
        """
        Reranks a list of documents based on a query and returns the reranked list. If get_sim_scores is True,
        it returns a list of tuples containing the documents and their similarity scores.

        Args:
            query (str): The query string.
            docs (List[Document]): A list of Document objects.
            get_sim_scores (bool, optional): Whether to return the similarity scores along with the reranked documents. Defaults to False.

        Returns:
            List[Document] | List[Tuple[Document, float]]: If get_sim_scores is False, returns a list of reranked Document objects. If get_sim_scores is True, returns a list of tuples containing the reranked Document objects and their similarity scores.
        """
        documents_text_list = [doc.page_content for doc in docs]
        scores = self.calculate_sim(query=query, documents_text_list=documents_text_list)
        docs_with_score = tuple(zip(docs, scores))
        reranked_docs = sorted(docs_with_score, key=lambda x: x[1], reverse=True)
        if not get_sim_scores:
            reranked_docs = [x[0] for x in reranked_docs]
        return reranked_docs