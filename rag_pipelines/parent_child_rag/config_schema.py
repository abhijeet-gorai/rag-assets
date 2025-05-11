from typing import TypedDict, List, Literal, Dict, Any

class LLMConfig(TypedDict):
    model_id: str
    max_tokens: int
    temperature: float
    top_p: float
    stop_sequences: List[str]

class ChunkingConfig(TypedDict):
    parent_chunk_size: int
    parent_chunk_overlap: int
    child_chunk_size: int
    child_chunk_overlap: int
    separators: List[str]

class FileReaderConfig(TypedDict):
    pdf: Dict[str, Any]
    docx: Dict[str, Any]

class AppConfig(TypedDict):
    llm: LLMConfig
    vector_db: Dict[str, Any]
    chunking: ChunkingConfig
    file_reader: FileReaderConfig
