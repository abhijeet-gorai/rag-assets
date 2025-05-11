import ast
import json
import hashlib
from services.retrievers.retriever_chroma import ChromaRetriever
from services.groq import init_chat_model
from utils.file_readers import read_pdf_file, read_docx_file
from typing import List, Tuple, Iterator, Union
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, BaseMessageChunk
from langchain_core.documents import Document
from prompts.rag_prompts import (
    qa_system_prompt, 
    qa_user_prompt, 
    multiple_queries_system_prompt, 
    multiple_queries_user_prompt
)
from .config_schema import AppConfig

class RAGFusion:
    """
    A Retrieval-Augmented Generation (RAG) fusion pipeline that combines multiple related queries and fuses their results.
    
    This implementation performs the following steps:
      1. Reads and splits documents, adding unique IDs to each document based on its content and metadata.
      2. Generates multiple related queries for a given original query.
      3. Retrieves candidate documents for each related query.
      4. Fuses the retrieved documents using reciprocal rank fusion.
      5. Generates an answer using a language model, either as a complete response or in a streaming fashion.
    """

    def __init__(self, config:AppConfig = {}) -> None:
        """
        Initializes the RAGFusion system by setting up the language model and the vector database.
        
        Args:
            config (AppConfig): Application configuration containing keys for LLMs, vector database, 
                                file reader options, and chunking parameters. Defaults to {}.
        """
        llm_config = config.get("llm", {})
        vector_db_config = config.get("vector_db", {})
        self.file_reader = config.get("file_reader", {
            "pdf": {"pdfloader": "PYMUPDF"},
            "docx": {"docxloader": "python-docx"}
        })
        self.chunking_config = config.get("chunking", {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "separators": ["\n\n", " "]
        })
        
        self.llm = init_chat_model(**llm_config)
        self.vector_db = ChromaRetriever(**vector_db_config)
        
    def add_documents(self, file_list: List[str]) -> None:
        """
        Reads and processes documents from the provided file paths, splits them into smaller chunks,
        assigns a unique ID to each chunk based on its content and metadata, and adds them to the vector database.
        
        Args:
            file_list (List[str]): A list of file paths to be added.
        
        Raises:
            ValueError: If a file format is unsupported.
        """
        docs = []
        for file in file_list:
            if file.lower().endswith('.pdf'):
                docs.extend(read_pdf_file(file, **self.file_reader.get("pdf")))
            elif file.lower().endswith('.docx'):
                docs.extend(read_docx_file(file, **self.file_reader.get("docx")))
            else:
                raise ValueError(f"Unsupported file format: {file}")
                
        # Split the documents into smaller chunks for improved retrieval performance.
        splitter = RecursiveCharacterTextSplitter(**self.chunking_config)
        split_docs = splitter.split_documents(docs)
        
        # Add a unique identifier to each document based on its content and sorted metadata.
        for doc in split_docs:
            unique_id = hashlib.md5(
                (doc.page_content + json.dumps(doc.metadata, sort_keys=True)).encode()
            ).hexdigest()
            doc.metadata['id'] = unique_id
            
        self.vector_db.add_documents(split_docs)

    def generate_related_queries(self, original_query: str) -> List[str]:
        """
        Generates a list of related queries based on the original query using the language model.
        
        The language model is prompted with system and user prompts designed for generating multiple queries.
        The response is expected to contain a Python literal (e.g., list) enclosed in triple backticks.
        
        Args:
            original_query (str): The original user query.
        
        Returns:
            List[str]: A list of related queries, or an empty list if generation fails.
        """
        messages = [
            SystemMessage(content=multiple_queries_system_prompt),
            HumanMessage(content=multiple_queries_user_prompt.format(query=original_query))
        ]
        try:
            response = self.llm.invoke(messages)
            # Expecting the generated queries to be within triple backticks.
            queries = response.content.split("```")[1]
            return ast.literal_eval(queries)
        except Exception as e:
            return []
        
    def reciprocal_rank_fusion(self, retrieved_docs: List[List[Document]], max_docs: int = 4) -> List[Document]:
        """
        Fuses multiple lists of retrieved documents using Reciprocal Rank Fusion (RRF).
        
        Each document's score is computed based on its rank in each list, and the documents are then
        sorted based on the aggregated scores.
        
        Args:
            retrieved_docs (List[List[Document]]): A list containing lists of documents retrieved for different queries.
            max_docs (int, optional): The maximum number of documents to return after fusion. Defaults to 4.
        
        Returns:
            List[Document]: A list of fused documents sorted by their aggregated RRF scores.
        """
        doc_scores = {}
        for docs in retrieved_docs:
            for rank, doc in enumerate(docs):
                doc_id = doc.metadata.get('id')
                if doc_id:
                    rrf_score = 1 / (rank + 1)
                    if doc_id in doc_scores:
                        doc_scores[doc_id]['score'] += rrf_score
                    else:
                        doc_scores[doc_id] = {'doc': doc, 'score': rrf_score}
        fused_docs = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)
        return [entry['doc'] for entry in fused_docs[:max_docs]]

    def get_context(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieves and fuses relevant documents for the given query by generating related queries and applying reciprocal rank fusion.
        
        Args:
            query (str): The original user query.
            k (int, optional): The number of documents to retrieve per related query. Defaults to 4.
        
        Returns:
            List[Document]: A list of fused documents relevant to the original query.
        """
        related_queries = self.generate_related_queries(original_query=query)
        all_retrieved_docs = []
        for q in related_queries:
            retrieved_docs = self.vector_db.get_relevant_docs(query=q, k=k)
            all_retrieved_docs.append(retrieved_docs)
        fused_docs = self.reciprocal_rank_fusion(retrieved_docs=all_retrieved_docs, max_docs=k)
        return fused_docs
    
    def get_answer(
        self, 
        query: str, 
        docs: List[Document], 
        stream: bool = False
    ) -> Union[BaseMessage, Iterator[BaseMessageChunk]]:
        """
        Generates an answer to the query based on the provided context from retrieved documents.
        
        The context is formed by concatenating the page content of the documents, which is then provided to the language model
        along with system and user prompts.
        
        Args:
            query (str): The user query.
            docs (List[Document]): The list of documents used to form the context.
            stream (bool, optional): Whether to stream the answer. Defaults to False.
        
        Returns:
            Union[BaseMessage, Iterator[BaseMessageChunk]]:
                - If stream is False, returns a single BaseMessage containing the complete answer.
                - If stream is True, returns an iterator over BaseMessageChunk instances.
        """
        context = "\n\n".join([doc.page_content for doc in docs])
        messages = [
            SystemMessage(content=qa_system_prompt),
            HumanMessage(content=qa_user_prompt.format(query=query, context=context))
        ]
        if stream:
            response = self.llm.stream(messages)
        else:
            response = self.llm.invoke(messages)
        return response

    def respond_to_query(self, query: str, k: int = 4) -> Tuple[BaseMessage, List[Document]]:
        """
        Processes the query by retrieving relevant documents, generating a complete answer, and returning both.
        
        Args:
            query (str): The user query.
            k (int, optional): The number of relevant documents to retrieve per query. Defaults to 4.
        
        Returns:
            Tuple[BaseMessage, List[Document]]:
                A tuple containing:
                - The generated answer as a BaseMessage.
                - The list of relevant documents used to generate the answer.
        """
        similar_docs = self.get_context(query=query, k=k)
        answer = self.get_answer(query, similar_docs)
        return answer, similar_docs

    def stream_answer(self, query: str, k: int = 4) -> Iterator[BaseMessageChunk]:
        """
        Processes the query by retrieving relevant documents and streaming the answer.
        
        Args:
            query (str): The user query.
            k (int, optional): The number of relevant documents to retrieve per query. Defaults to 4.
        
        Returns:
            Iterator[BaseMessageChunk]: An iterator over the generated answer chunks.
        """
        similar_docs = self.get_context(query=query, k=k)
        answer = self.get_answer(query, similar_docs, stream=True)
        return answer
