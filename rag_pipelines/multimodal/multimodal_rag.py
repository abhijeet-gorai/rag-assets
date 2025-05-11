import ast
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Iterator, Union

from services.retrievers.retriever_chroma import ChromaRetriever
from services.groq import init_chat_model
from .multimodal_pdf_reader import MultimodalPdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, BaseMessageChunk
from prompts.rag_prompts import (
    qa_system_prompt, 
    qa_user_prompt, 
    multimodal_qa_system_prompt, 
    multimodal_qa_user_prompt,
    image_answer_system_prompt,
    image_answer_user_prompt
)
from langchain_core.documents import Document
from utils.chat_image import ChatWithImage
from .config_schema import AppConfig
from utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class MultiModalRAG:
    """
    A multimodal Retrieval-Augmented Generation (RAG) pipeline.

    This class integrates document retrieval with both a text-based language model and a vision-capable LLM
    to answer queries based on retrieved text and image content from documents. It is designed to process PDFs 
    (currently the only supported format) in a multimodal fashion, allowing image extraction and concurrent image answer generation.
    """

    def __init__(self, config: AppConfig = {}) -> None:
        """
        Initializes the MultiModalRAG system by setting up the language models, multimodal PDF reader,
        vector database, and rate limiter using the provided configuration.

        Args:
            config (AppConfig): Application configuration containing keys for LLMs, vector database, 
                                multimodal reader options, and chunking parameters. Defaults to {}.
        """
        llm_config = config.get("answer_llm", {})
        vision_llm_config = config.get("vision_llm", {})
        vector_db_config = config.get("vector_db", {})
        context_window = config.get("multimodal_reader", {}).get("context_window", 4)
        self.max_workers = config.get("multimodal_reader", {}).get("max_workers", 4)
        self.group_mode = config.get("multimodal_reader", {}).get("group_mode", "group_by_title")
        self.keep_images_separate = config.get("multimodal_reader", {}).get("keep_images_separate", False)
        self.chunking_config = config.get("chunking", {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "separators": ["\n\n", " "]
        })

        # Initialize text and vision language models.
        self.llm = init_chat_model(**llm_config)
        self.image_llm = ChatWithImage(**vision_llm_config)

        # Initialize the multimodal PDF reader.
        self.multimodal_reader = MultimodalPdfReader(
            vision_llm_config=vision_llm_config, 
            context_window=context_window, 
            max_workers=self.max_workers
        )
        # Initialize the vector database for document retrieval.
        self.vector_db = ChromaRetriever(**vector_db_config)

        # Rate limiter to throttle calls to image-based LLM.
        self.rate_limiter = RateLimiter(max_calls=8, period=1)

    def add_documents(self, file_list: List[str]) -> None:
        """
        Reads and processes PDF documents from the given file paths, extracts multimodal content (text and images),
        splits the text into smaller chunks, and adds them to the vector database.

        Args:
            file_list (List[str]): List of PDF file paths to be added.

        Raises:
            ValueError: If a file format other than PDF is provided.
        """
        docs: List[Document] = []
        for file in file_list:
            if file.lower().endswith('.pdf'):
                # Use the multimodal reader to extract both text and images.
                docs.extend(
                    self.multimodal_reader.read_pdf(
                        file, 
                        group_mode=self.group_mode, 
                        keep_images_separate=self.keep_images_separate
                    )
                )
            else:
                raise ValueError(f"Only pdf file format supported: {file}")
            
        for doc in docs:
            doc.metadata["images"] = str(doc.metadata["images"])

        # Split documents into smaller chunks for better retrieval performance.
        splitter = RecursiveCharacterTextSplitter(**self.chunking_config)
        split_docs = splitter.split_documents(docs)
        self.vector_db.add_documents(split_docs)

    def get_context(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieves relevant documents from the vector database based on the provided query.

        Args:
            query (str): The user query.
            k (int, optional): The number of relevant documents to retrieve. Defaults to 4.

        Returns:
            List[Document]: A list of documents relevant to the query.
        """
        retrieved_docs = self.vector_db.get_relevant_docs(query=query, k=k)
        for doc in retrieved_docs:
            doc.metadata["images"] = ast.literal_eval(doc.metadata["images"])
        return retrieved_docs
    
    def get_answer_from_image(self, query: str, image: str) -> str:
        """
        Generates an answer for a given query from an image using the vision-enabled LLM.

        Applies rate limiting.

        Args:
            query (str): The user query.
            image (str): The base64 image to be processed.

        Returns:
            str: The answer generated by the image LLM.

        Raises:
            Exception: If all retry attempts fail.
        """
        self.rate_limiter.wait()
        user_prompt = image_answer_user_prompt.format(query=query)
        answer = self.image_llm.chat_with_image(
            prompt=user_prompt, 
            image=image, 
            system_message=image_answer_system_prompt,
            convert_image_to_base64=False
        )
        return answer

    def get_answer(
        self, 
        query: str, 
        docs: List[Document], 
        stream: bool = False
    ) -> Union[BaseMessage, Iterator[BaseMessageChunk]]:
        """
        Generates an answer to the query by combining textual context from retrieved documents 
        with answers generated from any images present within those documents.

        The process is as follows:
          1. Concatenate text content from all documents to form the context.
          2. Extract unique images from document metadata.
          3. If images are present, concurrently obtain answers for each image.
          4. Use either a multimodal prompt (if images exist) or a standard QA prompt (if no images) to query the text LLM.
          5. Return the final answer, either as a complete BaseMessage or a stream of BaseMessageChunks.

        Args:
            query (str): The user query.
            docs (List[Document]): List of documents used to form the context.
            stream (bool, optional): Whether to stream the answer. Defaults to False.

        Returns:
            Union[BaseMessage, Iterator[BaseMessageChunk]]:
                - If stream is False, returns a single BaseMessage with the complete answer.
                - If stream is True, returns an iterator over BaseMessageChunk instances.
        """
        # Build text context from document contents.
        context = "\n\n".join([doc.page_content for doc in docs])

        # Extract images from document metadata.
        images = []
        for doc in docs:
            doc_images = doc.metadata.get("images", [])
            doc_image_base64 = [image["image_base64"] for image in doc_images]
            images.extend(doc_image_base64)
        images = list(set(images))  # Remove duplicates.

        # If images are present, generate image-based answers concurrently.
        if images:
            image_answers = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_image = [executor.submit(self.get_answer_from_image, query, image) for image in images]
                for future in as_completed(future_to_image):
                    try:
                        answer = future.result()
                        image_answers.append(answer)
                    except Exception as e:
                        logger.error("Error processing an image: %s", e, exc_info=True)
            # Combine image answers.
            chunk_summaries = "\n\n".join(image_answers)
            messages = [
                SystemMessage(content=multimodal_qa_system_prompt),
                HumanMessage(content=multimodal_qa_user_prompt.format(query=query, context=context, chunk_summaries=chunk_summaries))
            ]
        else:
            messages = [
                SystemMessage(content=qa_system_prompt),
                HumanMessage(content=qa_user_prompt.format(query=query, context=context))
            ]
        # Invoke the text LLM with either a streaming or non-streaming response.
        if stream:
            response = self.llm.stream(messages)
        else:
            response = self.llm.invoke(messages)
        return response

    def respond_to_query(self, query: str, k: int = 4) -> Tuple[BaseMessage, List[Document]]:
        """
        Processes the user query by retrieving relevant documents and generating a complete answer.
        Combines multimodal content (text and images) in the response.

        Args:
            query (str): The user query.
            k (int, optional): The number of relevant documents to retrieve. Defaults to 4.

        Returns:
            Tuple[BaseMessage, List[Document]]:
                - The generated answer as a BaseMessage.
                - The list of relevant documents used to generate the answer.
        """
        similar_docs = self.get_context(query=query, k=k)
        answer = self.get_answer(query, similar_docs)
        return answer, similar_docs

    def stream_answer(self, query: str, k: int = 4) -> Iterator[BaseMessageChunk]:
        """
        Processes the query by retrieving relevant documents and streaming the final answer.
        This method supports streaming responses from the text LLM.

        Args:
            query (str): The user query.
            k (int, optional): The number of relevant documents to retrieve. Defaults to 4.

        Returns:
            Iterator[BaseMessageChunk]: An iterator over the generated answer chunks.
        """
        similar_docs = self.get_context(query=query, k=k)
        answer = self.get_answer(query, similar_docs, stream=True)
        return answer
