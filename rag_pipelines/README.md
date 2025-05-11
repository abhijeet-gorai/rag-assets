# RAG Pipelines

This folder contains various implementations of Retrieval-Augmented Generation (RAG) pipelines. These pipelines combine document retrieval with language model generation to produce context-aware answers.

The subfolders include:

- **basic_rag:** 
  - Contains a basic RAG implementation.
- **multimodal:**
  - Implements a multimodal RAG pipeline that can handle pdfs containing images.
- **parent_child_rag:**
  - Implements a RAG pipeline for parent-child document structures.
- **query_decomposition:**
  - Implements a query decomposition based RAG pipeline.
- **rag_fusion:**
  - Combines multiple related queries and fuses their results.
- **reranker_rag:**
  - Implements a RAG pipeline that leverages a reranking mechanism (using ColBERT or Cross Encoder) to improve document relevance.

Each subfolder is self-contained with its own configuration, schema, and implementation details. Use the example notebooks in the `example_notebooks` folder to see how each pipeline can be utilized.
