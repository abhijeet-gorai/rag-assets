# Services Module

- **LLM Services:**
  - `groq.py` – Functions for creating Langchain ChatGroq objects.
- **Retrievers:**
  - Contains various implementations for retrieving documents:
    - `retriever_chroma.py` – Retriever using Chroma.
    - `retriever_es.py` – Retriever using Elasticsearch.
    - `retriever_milvus.py` – Retriever using Milvus.
  - Additionally, there are specific retrievers for parent-child document structures under the `retrievers` subfolder.

These service modules are the backbone for accessing LLM APIs, and retrieval systems, enabling the pipelines and other modules to function seamlessly.
