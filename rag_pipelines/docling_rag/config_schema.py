from typing import TypedDict, List, Dict, Any

class VisionLLMConfig(TypedDict):
    model_id: str
    max_tokens: int
    temperature: float
    top_p: float
    stop_sequences: List[str]

class AnswerLLMConfig(TypedDict):
    model_id: str
    max_tokens: int
    temperature: float
    top_p: float
    stop_sequences: List[str]

class MultimodalReaderConfig(TypedDict):
    context_window: int
    max_workers: int
    group_mode: str
    keep_images_tables_separate: bool

class ChunkingConfig(TypedDict):
    chunk_size: int
    chunk_overlap: int
    separators: List[str]

class AppConfig(TypedDict):
    vision_llm: VisionLLMConfig
    answer_llm: AnswerLLMConfig
    vector_db: Dict[str, Any]
    multimodal_reader: MultimodalReaderConfig
    chunking: ChunkingConfig
