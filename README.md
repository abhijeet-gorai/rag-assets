# RAG Assets Repository

Welcome to the RAG Assets Repository. This repository contains a collection of modules, pipelines, and example notebooks to help you build and experiment with RAG solutions. The assets include tools for:
- **Retrieval-Augmented Generation (RAG):** Multiple implementations including basic, multimodal, parent-child, query decomposition, fusion, and reranker pipelines.
- **Image & File Processing:** Modules for chat with image, and file reading utilities.
- **Document Retrieval & Reranking:** Implementations using vector databases (Chroma, Elasticsearch, Milvus) and rerankers (ColBERT, Cross Encoder).


## 🚀 Getting Started

**Prerequisite:** Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (a fast Python package manager).

### 1. Clone the Repository

### 2. Install Dependencies

```bash
uv venv                   # Create a virtual environment
uv sync                   # Sync dependencies from pyproject.toml and uv.lock
```

### 3. Explore Example Notebooks
Navigate to the `example_notebooks` folder and open any notebook in Jupyter Notebook to see how the modules work.

## Configuration

Each RAG pipeline read configuration from YAML files (e.g., in `rag_pipelines/basic_rag/config.yaml` or `rag_pipelines/reranker_rag/config.yaml`). Adjust these files as needed for your environment.

## Contributing

Contributions, bug reports, and feature requests are welcome. Please feel free to open an issue or submit a pull request.