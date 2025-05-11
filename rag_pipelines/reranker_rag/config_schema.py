from typing import TypedDict, List, Literal, Dict, Any

class LLMConfig(TypedDict):
    model_id: str
    max_tokens: int
    temperature: float
    top_p: float
    stop_sequences: List[str]

class ChunkingConfig(TypedDict):
    chunk_size: int
    chunk_overlap: int
    separators: List[str]

class FileReaderConfig(TypedDict):
    pdf: Dict[str, Any]
    docx: Dict[str, Any]

class ReRankerConfig(TypedDict):
    type: Literal["colbert", "crossencoder"]
    params: Dict[str, Any]

class AppConfig(TypedDict):
    llm: LLMConfig
    vector_db: Dict[str, Any]
    chunking: ChunkingConfig
    file_reader: FileReaderConfig
    reranker: ReRankerConfig
