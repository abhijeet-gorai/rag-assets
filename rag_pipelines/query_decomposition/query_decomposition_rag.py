import ast
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Iterator, Dict, Union

from services.retrievers.retriever_chroma import ChromaRetriever
from services.groq import init_chat_model
from utils.file_readers import read_pdf_file, read_docx_file
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, BaseMessageChunk
from prompts.rag_prompts import (
    qa_system_prompt, 
    qa_user_prompt,
    sub_query_system_prompt, 
    sub_query_user_prompt
)
from .config_schema import AppConfig
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class QueryDecompositionRAG:
    """
    A Retrieval-Augmented Generation (RAG) pipeline that decomposes a user query into subqueries,
    retrieves relevant documents for each subquery in parallel, and then generates answers for each subquery.
    Finally, it combines the subquery question-answer pairs with the original query to produce a final answer.

    Pipeline Steps:
      1. get_context: Decompose the query and retrieve relevant documents for each subquery.
      2. get_answer: Process each subquery (using its retrieved documents) to generate an answer,
         then combine all subquery Q/A pairs with the original query to generate the final answer.
    """

    def __init__(self, config: AppConfig = {}) -> None:
        """
        Initializes the QueryDecompositionRAG system by setting up the language model and vector database.

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
        Reads and processes documents from provided file paths, splits them into smaller chunks,
        assigns unique IDs, and adds them to the vector database.

        Args:
            file_list (List[str]): List of file paths to be added.

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
        splitter = RecursiveCharacterTextSplitter(**self.chunking_config)
        split_docs = splitter.split_documents(docs)
        self.vector_db.add_documents(split_docs)

    def decompose_query(self, query: str) -> List[str]:
        """
        Decomposes the original query into a list of subqueries using the language model.

        The LLM is prompted with system and user messages. Its output is expected to be a Python literal (e.g., a list)
        enclosed in triple backticks.

        Args:
            query (str): The original user query.

        Returns:
            List[str]: A list of subqueries, or the original query in a list if decomposition fails.
        """
        messages = [
            SystemMessage(content=sub_query_system_prompt),
            HumanMessage(content=sub_query_user_prompt.format(query=query))
        ]
        try:
            response = self.llm.invoke(messages)
            # Expected output format: ```["Subquery 1", "Subquery 2", ...]```
            queries_str = response.content.split("```")[1]
            return ast.literal_eval(queries_str)
        except Exception as e:
            return [query]

    def get_context(self, query: str, k: int = 4) -> Dict[str, List[Document]]:
        """
        Decomposes the query into subqueries and retrieves relevant documents for each subquery in parallel.

        Args:
            query (str): The original user query.
            k (int, optional): The number of relevant documents to retrieve for each subquery. Defaults to 4.

        Returns:
            Dict[str, List[Document]]: A dictionary mapping each subquery to its list of retrieved documents.
        """
        subqueries = self.decompose_query(query)
        subquery_docs: Dict[str, List[Document]] = {}
        
        with ThreadPoolExecutor() as executor:
            future_to_subquery = {
                executor.submit(self.vector_db.get_relevant_docs, query=subq, k=k): subq
                for subq in subqueries
            }
            for future in as_completed(future_to_subquery):
                subq = future_to_subquery[future]
                try:
                    docs = future.result()
                    subquery_docs[subq] = docs
                except Exception as e:
                    subquery_docs[subq] = []  # In case of failure, map subquery to an empty list.
        return subquery_docs

    def process_subquery_with_docs(self, subquery: str, docs: List[Document]) -> Tuple[str, BaseMessage]:
        """
        Generates an answer for a given subquery using the provided list of documents.

        Args:
            subquery (str): The subquery to process.
            docs (List[Document]): List of documents retrieved for this subquery.

        Returns:
            Tuple[str, BaseMessage]: A tuple containing the subquery and the generated answer.
        """
        context = "\n\n".join([doc.page_content for doc in docs])
        messages = [
            SystemMessage(content=qa_system_prompt),
            HumanMessage(content=qa_user_prompt.format(query=subquery, context=context))
        ]
        answer = self.llm.invoke(messages)
        return subquery, answer

    def get_answer(self, query: str, context_dict: Dict[str, List[Document]], k: int = 4, stream: bool = False) -> Union[BaseMessage, Iterator[BaseMessageChunk]]:
        """
        Generates answers for each subquery based on the retrieved documents and combines the subquery
        question-answer pairs with the original query to produce a final answer.

        Args:
            query (str): The original user query.
            context_dict (Dict[str, List[Document]]): A dictionary mapping subqueries to their list of retrieved documents.
            k (int, optional): The number of relevant documents retrieved per subquery (for consistency). Defaults to 4.
            stream (bool, optional): Whether to stream the final answer. Defaults to False.

        Returns:
            Union[BaseMessage, Iterator[BaseMessageChunk]]:
                - If stream is False, returns a single BaseMessage with the complete final answer.
                - If stream is True, returns an iterator over BaseMessageChunk instances.
        """
        subquery_qa_pairs: List[Tuple[str, BaseMessage]] = []
        with ThreadPoolExecutor() as executor:
            future_to_subquery = {
                executor.submit(self.process_subquery_with_docs, subq, docs): subq
                for subq, docs in context_dict.items()
            }
            for future in as_completed(future_to_subquery):
                try:
                    qa_pair = future.result()
                    subquery_qa_pairs.append(qa_pair)
                except Exception as e:
                    logger.error("Error processing a subquery: %s", e, exc_info=True)
        
        combined_text = "\n".join([f"Subquery: {q}\nAnswer: {ans.content}" for q, ans in subquery_qa_pairs])
        messages = [
            SystemMessage(content=qa_system_prompt),
            HumanMessage(content=qa_user_prompt.format(query=query, context=combined_text))
        ]
        if stream:
            return self.llm.stream(messages)
        else:
            return self.llm.invoke(messages)

    def respond_to_query(self, query: str, k: int = 4) -> Tuple[BaseMessage, Dict[str, List[Document]]]:
        """
        Processes the original query by retrieving context for subqueries and generating the final answer.

        Args:
            query (str): The original user query.
            k (int, optional): The number of relevant documents to retrieve per subquery. Defaults to 4.

        Returns:
            Tuple[BaseMessage, Dict[str, List]]:
                - The final answer as a BaseMessage.
                - A dictionary mapping each subquery to its list of retrieved documents.
        """
        context_dict = self.get_context(query, k)
        final_answer = self.get_answer(query, context_dict, k)
        return final_answer, context_dict

    def stream_answer(self, query: str, k: int = 4) -> Iterator[BaseMessageChunk]:
        """
        Processes the query by retrieving context for subqueries and streaming the final answer.

        Args:
            query (str): The original user query.
            k (int, optional): The number of relevant documents to retrieve per subquery. Defaults to 4.

        Returns:
            Iterator[BaseMessageChunk]: An iterator over the final answer chunks.
        """
        context_dict = self.get_context(query, k)
        return self.get_answer(query, context_dict, k, stream=True)
