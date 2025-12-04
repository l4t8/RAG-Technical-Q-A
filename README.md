# REFRAG Analysis Framework: Benchmarking RAG Strategies

**Q&A system implementation for the 2025-26 Concurso de Modelización de Empresa of Management Solutions (Facultad de Matemáticas, UCM).**

![Python](https://img.shields.io/badge/Python-3.11-blue) ![LangChain](https://img.shields.io/badge/Orchestration-LangChain_v0.2.16-green) ![Gemini](https://img.shields.io/badge/Model-Gemini%202.5%20Pro%20%2F%20Flash-orange)

This repository hosts an experimental evaluation framework designed to empirically measure the performance of different Retrieval-Augmented Generation (RAG) architectures.

The project focuses on the ingestion and analysis of complex technical documentation (Case study: "REFRAG: Rethinking RAG based Decoding" by Meta). The fundamental objective is to quantify how different retrieval strategies—from lexical search to hybrid approaches—impact the accuracy, latency, and faithfulness of an LLM when answering domain-specific questions.

## Technical Architecture

The system is built in a modular fashion to allow for the hot-swapping of retrieval components and inference models.

### Core Components
* **Orchestration:** LangChain v0.2.16 (including `langchain-google-genai` and `langchain-huggingface`) for chain and prompt management.
* **Ingestion & Chunking:** PDF processing using `PyPDFLoader` and recursive segmentation (`RecursiveCharacterTextSplitter`) to optimize the context window.
* **Vector Storage:** Embedding persistence using ChromaDB.
* **Models:**
    * *Inference:* **Google Gemini 2.5** (Pro and Flash versions via API).
    * *Embeddings:* `all-MiniLM-L6-v2` (HuggingFace) for efficient semantic representation.

### Evaluation Pipelines
The framework allows running and comparing four distinct strategies:

1.  **(A) Baseline (Zero-shot):** Direct evaluation of the LLM's parametric knowledge without retrieved context. Serves as a control baseline.
2.  **(B) Sparse Retrieval (BM25):** Retrieval based on term frequency and exact keyword matching. Ideal for specific technical terminology.
3.  **(C) Dense Retrieval:** Semantic search via cosine similarity in vector space.
4.  **(D) Hybrid RAG (Ensemble):** Implementation of a hybrid retriever combining BM25 and Dense Retrieval to mitigate individual weaknesses.

## Conclusions

It is important to note that this project relies on a free tier of the Google API; it is highly probable that all results would improve with a paid tier. Furthermore, the recorded latency times may not accurately reflect those of a production-scale environment.

**Performance by Model (Flash vs. Pro)**
Observations indicate that when using the **Flash model**, the performance gap between BM25 and Hybrid Retrieval is negligible, in contrast to the **Pro model** results. For scenarios prioritizing extreme speed and cost-efficiency, Hybrid Retrieval offers little advantage over the similar performance of BM25. However, with increased computational resources (Pro model), a significant divergence appears: Hybrid RAG outperforms BM25 RAG due to the synergistic effect of Dense Retrieval. While Dense Retrieval alone is inefficient and fails to surpass BM25, their combination achieves high precision across all metrics.

**Metrics Analysis**
Regarding **Source Attribution Accuracy**, Hybrid Retrieval represents a clear improvement for both models; choosing otherwise is suboptimal. Additionally, latency analysis reveals that Hybrid RAG achieves lower response times than both Dense and BM25 RAG for Pro and Flash models. Furthermore, the **Fuzz Score** strongly supports the adoption of Hybrid RAG. The fact that the hybrid model prevails in this lexical metric is a definitive argument: it demonstrates an ability to not only grasp semantic context but also respect exact terminology better than BM25 alone, effectively combining the best of both worlds.

**The Role of BM25 and Dense Retrieval**
In contrast to Hybrid RAG, the primary advantages of **BM25 RAG** lie in its low resource consumption and the lack of requirement for a vector store. While BM25 RAG might serve as a temporary solution for resource-constrained projects, it is difficult to envision a scenario where the computational cost of Hybrid RAG is not justified by the substantial improvement across all metrics.

Conversely, **Dense Retrieval** consistently underperforms compared to BM25 across all metrics. This is attributed to the dataset characteristics: the theoretical depth and highly specialized terminology heavily favor BM25's exact matching. While Dense Retrieval might excel as a "cheap and effective" method in literary datasets where depth relies on semantics rather than morphology, this is a niche case, especially since the marginal cost of upgrading from Dense to Hybrid is minimal.