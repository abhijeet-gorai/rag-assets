from services.retrievers.parent_child_retriever_chroma import ChromaParentChildRetriever
from services.groq import init_chat_model
from utils.file_readers import read_pdf_file, read_docx_file
from typing import List, Tuple, Iterator, Union
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, BaseMessageChunk
from prompts.rag_prompts import qa_system_prompt, qa_user_prompt
from langchain_core.documents import Document
from .config_schema import AppConfig

class ParentChildRAG:
    """
    A Retrieval-Augmented Generation (RAG) pipeline implementation using a parent-child document retrieval strategy.
    
    This class integrates document retrieval with a language model to answer queries based on retrieved documents.
    It uses a parent-child retrieval mechanism where documents are structured with a parent-child relationship.
    """

    def __init__(self, config:AppConfig = {}) -> None:
        """
        Initializes the ParentChildRAG system by setting up the language model and the parent-child vector database.
        
        The parent documents are expected to be stored in the './parent_docs' directory.
        """
        llm_config = config.get("llm", {})
        vector_db_config = config.get("vector_db", {})
        chunking_config = config.get("chunking", {})
        self.file_reader = config.get("file_reader", {
            "pdf": {"pdfloader": "PYMUPDF"},
            "docx": {"docxloader": "python-docx"}
        })
        self.llm = init_chat_model(**llm_config)
        self.vector_db = ChromaParentChildRetriever(**vector_db_config, **chunking_config)
        
    def add_documents(self, file_list: List[str]) -> None:
        """
        Reads and processes documents from the provided file paths, then adds them to the parent-child vector database.
        
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
        self.vector_db.add_documents(docs)

    def get_context(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieves relevant documents based on the query using the parent-child retrieval strategy.
        
        Args:
            query (str): The user query.
            k (int, optional): The number of relevant documents to retrieve. Defaults to 4.
        
        Returns:
            List[Document]: A list of retrieved documents relevant to the query.
        """
        return self.vector_db.get_relevant_docs(query=query, k=k)

    def get_answer(self, query: str, docs: List[Document], stream: bool = False) -> Union[BaseMessage, Iterator[BaseMessageChunk]]:
        """
        Generates an answer to the query based on the provided context from retrieved documents.
        
        Args:
            query (str): The user query.
            docs (List[Document]): The list of documents used to form the context.
            stream (bool, optional): Whether to stream the answer. Defaults to False.
        
        Returns:
            Union[BaseMessage, Iterator[BaseMessageChunk]]:
                - If stream is False, returns a single BaseMessage containing the answer.
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
        Processes the query by retrieving relevant documents and generating a complete answer.
        
        Args:
            query (str): The user query.
            k (int, optional): The number of relevant documents to retrieve. Defaults to 4.
        
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
            k (int, optional): The number of relevant documents to retrieve. Defaults to 4.
        
        Returns:
            Iterator[BaseMessageChunk]: An iterator over the generated answer chunks.
        """
        similar_docs = self.get_context(query=query, k=k)
        answer = self.get_answer(query, similar_docs, stream=True)
        return answer
