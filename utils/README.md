# Utils Module

This folder contains utility modules and tools that are used throughout the repository. The utilities include:

- **Audio Detection:** Modules for processing and detecting events in audio files.
  - `audio_detection.py` – Contains functionality to detect audio events.
- **Chat with Image:** Modules to interact with a vision LLM that accepts both text and images.
  - `chat_image.py` – Provides methods to encode images, build chat messages, and manage session histories.
- **File Readers:** Modules to read and process different file formats (e.g., PDFs, DOCX).
  - `file_readers.py` – Implements file reading functions.
- **Rate Limiter:** Utility for rate limiting function calls.
  - `rate_limiter.py` – Provides rate limiting capabilities.
- **Rerankers:** Implementations of reranking algorithms.
  - `rerankers/colbert.py` – ColBERT reranker.
  - `rerankers/cross_encoder.py` – Cross Encoder reranker.
- **Tools:** Miscellaneous helper tools.
  - `tools/code_interpreter.py` – Code interpreter utilities.
  - `tools/code_interpreter_dockerfile/` – Dockerfile for code interpreter deployment.

These utilities help to modularize the code and make it reusable across different services and pipelines.