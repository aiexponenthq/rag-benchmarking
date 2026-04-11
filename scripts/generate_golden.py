#!/usr/bin/env python3
"""
Generate a 50-sample golden evaluation dataset for rag-benchmarking.
Covers 10 domains with 5 samples each. No LLM calls — fully deterministic.
Run: python scripts/generate_golden.py --output data/golden/qa.jsonl
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

SAMPLES = [
    # ── Domain 1: RAG Fundamentals ───────────────────────────────────────────
    {
        "question": "What does RAG stand for and what problem does it solve?",
        "contexts": [
            "RAG stands for Retrieval-Augmented Generation. It combines information retrieval with large language model generation.",
            "Traditional LLMs are limited to knowledge from their training data. RAG allows dynamic access to external knowledge at inference time.",
        ],
        "answer": "RAG stands for Retrieval-Augmented Generation. It solves the hallucination problem by grounding LLM responses in retrieved documents.",
        "ground_truths": ["RAG stands for Retrieval-Augmented Generation and reduces hallucination by grounding responses in retrieved documents."],
        "relevant_doc_ids": ["rag-1", "rag-2"],
    },
    {
        "question": "What are the main components of a RAG pipeline?",
        "contexts": ["A RAG pipeline consists of three main components: a retriever, a vector store, and a generator. The retriever finds relevant documents, the vector store indexes embeddings, and the generator produces the final answer."],
        "answer": "The three main components are: a retriever that finds relevant documents, a vector store that indexes embeddings, and a generator (LLM) that produces the final answer.",
        "ground_truths": ["A RAG pipeline has three components: retriever, vector store, and generator."],
        "relevant_doc_ids": ["rag-3"],
    },
    {
        "question": "What is chunking in RAG systems?",
        "contexts": ["Chunking is the process of splitting documents into smaller pieces for indexing. Chunk size affects retrieval quality — smaller chunks are more precise but may lose context, while larger chunks retain more context but reduce precision."],
        "answer": "Chunking splits documents into smaller pieces for indexing. The chunk size involves a precision-context trade-off.",
        "ground_truths": ["Chunking splits documents into smaller pieces; chunk size trades precision against context retention."],
        "relevant_doc_ids": ["rag-4"],
    },
    {
        "question": "What is the difference between dense and sparse retrieval?",
        "contexts": ["Dense retrieval uses neural embeddings (vectors) to represent documents and queries, enabling semantic search. Sparse retrieval uses keyword-based methods like BM25 that match exact terms. Hybrid retrieval combines both approaches."],
        "answer": "Dense retrieval uses neural embeddings for semantic search. Sparse retrieval uses keyword matching like BM25. Hybrid methods combine both.",
        "ground_truths": ["Dense uses neural embeddings for semantic search; sparse uses keyword matching; hybrid combines both."],
        "relevant_doc_ids": ["rag-5"],
    },
    {
        "question": "What is reranking in RAG and why is it used?",
        "contexts": ["Reranking is a second-stage retrieval step that re-scores retrieved documents using a more powerful cross-encoder model. It improves precision by reordering the initial retrieval results based on relevance to the query."],
        "answer": "Reranking uses a cross-encoder to re-score and reorder retrieved documents, improving precision over first-stage retrieval.",
        "ground_truths": ["Reranking uses a cross-encoder to re-score retrieved documents and improve retrieval precision."],
        "relevant_doc_ids": ["rag-6"],
    },

    # ── Domain 2: Vector Databases ───────────────────────────────────────────
    {
        "question": "What is a vector database and how does it work?",
        "contexts": ["A vector database stores high-dimensional numerical representations (embeddings) of data. It uses approximate nearest neighbor (ANN) algorithms like HNSW or IVF to find the most similar vectors quickly."],
        "answer": "A vector database stores high-dimensional embeddings and uses ANN algorithms (HNSW, IVF) for fast similarity search.",
        "ground_truths": ["A vector database stores embeddings and uses ANN algorithms for fast similarity search."],
        "relevant_doc_ids": ["vdb-1"],
    },
    {
        "question": "What is cosine similarity used for in vector search?",
        "contexts": ["Cosine similarity measures the angle between two vectors, returning a value between -1 and 1. Values close to 1 indicate high similarity. It is commonly used in RAG systems to find documents semantically similar to a query."],
        "answer": "Cosine similarity measures the angle between vectors (ranging from -1 to 1) to find semantically similar documents in RAG systems.",
        "ground_truths": ["Cosine similarity measures vector angle similarity; used to find semantically similar documents."],
        "relevant_doc_ids": ["vdb-2"],
    },
    {
        "question": "What is HNSW and why is it used in vector databases?",
        "contexts": ["HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest neighbor algorithm. It provides sub-linear search time while maintaining high recall, making it suitable for large-scale vector search applications."],
        "answer": "HNSW is a graph-based ANN algorithm providing sub-linear search time with high recall, used for large-scale vector search.",
        "ground_truths": ["HNSW is a graph-based ANN algorithm with sub-linear search time and high recall."],
        "relevant_doc_ids": ["vdb-3"],
    },
    {
        "question": "What are embeddings in the context of NLP?",
        "contexts": ["Embeddings are dense vector representations of text that capture semantic meaning. Words or sentences with similar meanings have vectors that are close together in the embedding space."],
        "answer": "Embeddings are dense vector representations of text capturing semantic meaning; similar texts have nearby vectors.",
        "ground_truths": ["Embeddings are dense vector representations where semantically similar texts have nearby vectors."],
        "relevant_doc_ids": ["vdb-4"],
    },
    {
        "question": "How does Qdrant differ from other vector databases?",
        "contexts": ["Qdrant is an open-source vector database written in Rust. It supports payload filtering alongside vector search, allowing combined semantic and structured filtering. It offers both cloud-hosted and self-hosted options."],
        "answer": "Qdrant is a Rust-based open-source vector database supporting combined vector search and payload filtering, with both cloud and self-hosted options.",
        "ground_truths": ["Qdrant is a Rust-based vector database with payload filtering support and cloud/self-hosted options."],
        "relevant_doc_ids": ["vdb-5"],
    },

    # ── Domain 3: EU AI Act ───────────────────────────────────────────────────
    {
        "question": "What is the EU AI Act and when does it apply?",
        "contexts": ["The EU AI Act (Regulation 2024/1689) is the world's first comprehensive AI regulation. It entered into force on 1 August 2024, with obligations phasing in over three years. It applies to AI systems placed on the EU market or used within the EU."],
        "answer": "The EU AI Act is the world's first comprehensive AI regulation, entering into force August 2024, applicable to AI systems in the EU.",
        "ground_truths": ["The EU AI Act entered into force August 2024 and applies to AI systems placed on the EU market."],
        "relevant_doc_ids": ["euai-1"],
    },
    {
        "question": "What obligations does Article 53 of the EU AI Act impose?",
        "contexts": ["Article 53 imposes transparency obligations on providers of general-purpose AI (GPAI) models. They must publish technical documentation, comply with copyright law, and publish summaries of training data used."],
        "answer": "Article 53 requires GPAI model providers to publish technical documentation, comply with copyright law, and disclose training data summaries.",
        "ground_truths": ["Article 53 requires GPAI providers to publish technical documentation and training data summaries."],
        "relevant_doc_ids": ["euai-2"],
    },
    {
        "question": "What are high-risk AI systems under the EU AI Act?",
        "contexts": ["High-risk AI systems are defined in Annex III of the EU AI Act. They include AI used in critical infrastructure, education, employment, essential services, law enforcement, migration, and administration of justice. These systems must comply with Article 6 requirements by August 2026."],
        "answer": "High-risk AI systems (Annex III) include AI in critical infrastructure, education, employment, law enforcement, and justice. Compliance required by August 2026.",
        "ground_truths": ["High-risk AI systems under Annex III include infrastructure, education, employment, and law enforcement AI; compliance by August 2026."],
        "relevant_doc_ids": ["euai-3"],
    },
    {
        "question": "What does Article 9 of the EU AI Act require?",
        "contexts": ["Article 9 requires providers of high-risk AI systems to implement a risk management system. This includes identification and analysis of risks, estimation of risks, and adoption of risk management measures."],
        "answer": "Article 9 requires high-risk AI providers to implement a risk management system covering identification, analysis, and mitigation of risks.",
        "ground_truths": ["Article 9 requires a risk management system for high-risk AI, covering risk identification, analysis, and mitigation."],
        "relevant_doc_ids": ["euai-4"],
    },
    {
        "question": "What are the fines for non-compliance with the EU AI Act?",
        "contexts": ["Non-compliance with the EU AI Act can result in fines of up to EUR 35 million or 7% of global annual turnover for prohibited AI practices. High-risk AI violations carry fines up to EUR 15 million or 3% of global turnover."],
        "answer": "Prohibited AI practices: up to EUR 35M or 7% of global turnover. High-risk violations: up to EUR 15M or 3%.",
        "ground_truths": ["Fines: up to EUR 35M/7% for prohibited practices, EUR 15M/3% for high-risk violations."],
        "relevant_doc_ids": ["euai-5"],
    },

    # ── Domain 4: RAG Evaluation Metrics ─────────────────────────────────────
    {
        "question": "What is faithfulness in RAG evaluation?",
        "contexts": ["Faithfulness measures whether the generated answer is grounded in the retrieved context. A faithful answer makes only claims that are directly supported by the context, without hallucinating facts not present in the retrieved documents."],
        "answer": "Faithfulness measures whether the answer is grounded in retrieved context, with no hallucinated facts beyond what the context supports.",
        "ground_truths": ["Faithfulness measures whether the answer is fully supported by the retrieved context without hallucination."],
        "relevant_doc_ids": ["eval-1"],
    },
    {
        "question": "What is context relevance in RAG evaluation?",
        "contexts": ["Context relevance measures whether the retrieved contexts are relevant to the query. High context relevance means the retriever is returning documents that actually help answer the question, minimizing noise."],
        "answer": "Context relevance measures whether retrieved documents are relevant to the query, minimizing noise in the retrieved context.",
        "ground_truths": ["Context relevance measures whether retrieved documents are relevant to the query."],
        "relevant_doc_ids": ["eval-2"],
    },
    {
        "question": "What is MRR and when is it used?",
        "contexts": ["Mean Reciprocal Rank (MRR) is a retrieval metric that averages the reciprocal of the rank of the first relevant document across queries. MRR = 1/rank of first relevant document. It is used when there is one clearly most relevant document per query."],
        "answer": "MRR (Mean Reciprocal Rank) averages the reciprocal rank of the first relevant document. Used when one document is most relevant per query.",
        "ground_truths": ["MRR averages 1/rank of first relevant document; used when one document is most relevant per query."],
        "relevant_doc_ids": ["eval-3"],
    },
    {
        "question": "What is NDCG and how does it differ from Precision@K?",
        "contexts": ["NDCG (Normalized Discounted Cumulative Gain) weighs relevant documents by their position in the ranked list, penalizing relevant documents found lower in the ranking. Precision@K simply counts the fraction of relevant documents in the top K results, without considering rank position."],
        "answer": "NDCG weights relevant documents by rank position; Precision@K counts relevant fraction in top-K without considering rank position.",
        "ground_truths": ["NDCG weighs by position; Precision@K counts fraction in top-K without position weighting."],
        "relevant_doc_ids": ["eval-4"],
    },
    {
        "question": "What is the RAG triad of metrics?",
        "contexts": ["The RAG triad consists of three key metrics: Context Relevance (query ↔ context), Faithfulness (context ↔ response), and Answer Relevance (query ↔ response). Together they evaluate the full RAG pipeline from retrieval through generation."],
        "answer": "The RAG triad: Context Relevance (query-context), Faithfulness (context-response), Answer Relevance (query-response).",
        "ground_truths": ["The RAG triad evaluates Context Relevance, Faithfulness, and Answer Relevance."],
        "relevant_doc_ids": ["eval-5"],
    },

    # ── Domain 5: LLMs and Transformers ──────────────────────────────────────
    {
        "question": "What is attention mechanism in transformers?",
        "contexts": ["The attention mechanism allows a model to focus on relevant parts of the input when producing an output. Scaled dot-product attention computes attention scores by taking the dot product of query and key vectors, scaled by the square root of dimension."],
        "answer": "Attention allows models to focus on relevant input parts. Scaled dot-product attention uses query-key dot products scaled by sqrt(dimension).",
        "ground_truths": ["Attention mechanism focuses on relevant inputs; scaled dot-product uses query-key products scaled by sqrt(dim)."],
        "relevant_doc_ids": ["llm-1"],
    },
    {
        "question": "What is the difference between encoder-only, decoder-only, and encoder-decoder models?",
        "contexts": ["Encoder-only models (like BERT) are optimized for understanding tasks. Decoder-only models (like GPT) generate text autoregressively. Encoder-decoder models (like T5) combine both for sequence-to-sequence tasks like translation."],
        "answer": "Encoder-only (BERT): understanding tasks. Decoder-only (GPT): text generation. Encoder-decoder (T5): sequence-to-sequence tasks.",
        "ground_truths": ["Encoder-only for understanding; decoder-only for generation; encoder-decoder for seq2seq."],
        "relevant_doc_ids": ["llm-2"],
    },
    {
        "question": "What is temperature in LLM generation?",
        "contexts": ["Temperature controls the randomness of text generation. A temperature of 0 makes the model deterministic (always choosing the highest probability token). Higher temperatures increase randomness and creativity."],
        "answer": "Temperature controls generation randomness. Temperature 0 = deterministic; higher values increase randomness.",
        "ground_truths": ["Temperature 0 is deterministic; higher values increase generation randomness."],
        "relevant_doc_ids": ["llm-3"],
    },
    {
        "question": "What is context window length and why does it matter for RAG?",
        "contexts": ["Context window length is the maximum number of tokens a model can process at once. For RAG, a larger context window allows more retrieved documents to be included in the prompt, potentially improving answer quality at the cost of higher latency and cost."],
        "answer": "Context window is the maximum tokens a model processes. Larger windows allow more RAG context but increase latency and cost.",
        "ground_truths": ["Context window limits tokens processed; larger windows allow more RAG context but increase cost."],
        "relevant_doc_ids": ["llm-4"],
    },
    {
        "question": "What is hallucination in LLMs?",
        "contexts": ["Hallucination occurs when an LLM generates text that is factually incorrect, fabricated, or not grounded in the provided context. It happens because LLMs are trained to produce plausible text, not necessarily accurate text."],
        "answer": "Hallucination is when LLMs generate factually incorrect or fabricated content, occurring because models optimize for plausible text generation.",
        "ground_truths": ["Hallucination is factually incorrect or fabricated LLM output, caused by optimization for plausible text."],
        "relevant_doc_ids": ["llm-5"],
    },

    # ── Domain 6: Python / FastAPI ─────────────────────────────────────────
    {
        "question": "What is Pydantic and why is it used in FastAPI?",
        "contexts": ["Pydantic is a Python data validation library that uses type annotations. FastAPI uses Pydantic models for request body validation, response serialization, and auto-generated API documentation."],
        "answer": "Pydantic is a Python data validation library using type annotations. FastAPI uses it for request validation, response serialization, and OpenAPI docs.",
        "ground_truths": ["Pydantic validates data using type annotations; FastAPI uses it for request validation and auto-docs."],
        "relevant_doc_ids": ["py-1"],
    },
    {
        "question": "What is dependency injection in FastAPI?",
        "contexts": ["FastAPI's dependency injection system allows declaring dependencies using function parameters. Dependencies can be shared across routes, handle authentication, database connections, and other cross-cutting concerns."],
        "answer": "FastAPI's dependency injection uses function parameters to declare reusable dependencies for authentication, DB connections, and shared logic.",
        "ground_truths": ["FastAPI dependency injection uses function parameters for shared, reusable route dependencies."],
        "relevant_doc_ids": ["py-2"],
    },
    {
        "question": "What is async/await in Python and when should it be used?",
        "contexts": ["async/await enables asynchronous programming in Python. It should be used for I/O-bound operations like HTTP requests, database queries, and file I/O, allowing other tasks to run while waiting."],
        "answer": "async/await enables asynchronous I/O. Use for HTTP requests, DB queries, file I/O — operations where CPU waits for external resources.",
        "ground_truths": ["async/await is for I/O-bound operations enabling concurrent execution while waiting."],
        "relevant_doc_ids": ["py-3"],
    },
    {
        "question": "What is the difference between requirements.txt and pyproject.toml?",
        "contexts": ["requirements.txt is a flat list of pinned dependencies. pyproject.toml is a modern packaging standard (PEP 518/621) that specifies both build system and project metadata, including dependency ranges. pyproject.toml is preferred for new projects."],
        "answer": "requirements.txt is a pinned dependency list. pyproject.toml is the modern standard for build system config and dependency ranges.",
        "ground_truths": ["requirements.txt lists pinned deps; pyproject.toml is the modern packaging standard with build config."],
        "relevant_doc_ids": ["py-4"],
    },
    {
        "question": "What is a Python virtual environment?",
        "contexts": ["A virtual environment is an isolated Python environment with its own packages and interpreter. It prevents dependency conflicts between projects and allows different package versions per project."],
        "answer": "A virtual environment provides isolated Python environments with per-project packages, preventing dependency conflicts.",
        "ground_truths": ["Virtual environments are isolated Python environments preventing dependency conflicts."],
        "relevant_doc_ids": ["py-5"],
    },

    # ── Domain 7: MLOps / Production AI ────────────────────────────────────
    {
        "question": "What is model drift and how is it detected?",
        "contexts": ["Model drift occurs when a model's performance degrades because the real-world data distribution changes from the training distribution. It is detected by monitoring prediction distributions, feature distributions, and model performance metrics over time."],
        "answer": "Model drift is performance degradation due to distribution shift. Detected by monitoring prediction distributions and performance metrics over time.",
        "ground_truths": ["Model drift is performance degradation from distribution shift, detected by monitoring distributions and metrics."],
        "relevant_doc_ids": ["mlops-1"],
    },
    {
        "question": "What is the purpose of an A/B test in ML?",
        "contexts": ["A/B testing compares two model versions by splitting traffic and measuring performance metrics. It ensures a new model is statistically significantly better before full deployment, reducing risk of regression."],
        "answer": "A/B testing splits traffic between model versions to statistically compare performance before full deployment.",
        "ground_truths": ["A/B testing splits traffic between model versions to statistically verify improvements."],
        "relevant_doc_ids": ["mlops-2"],
    },
    {
        "question": "What is a feature store?",
        "contexts": ["A feature store is a centralized repository for storing and serving machine learning features. It enables feature reuse across teams and models, ensures consistency between training and serving, and supports point-in-time correct feature retrieval."],
        "answer": "A feature store is a centralized repository for ML features enabling reuse, training-serving consistency, and point-in-time retrieval.",
        "ground_truths": ["A feature store centralizes ML features for reuse, consistency, and point-in-time retrieval."],
        "relevant_doc_ids": ["mlops-3"],
    },
    {
        "question": "What is containerization and why is it important for ML?",
        "contexts": ["Containerization packages code and dependencies into portable containers (e.g., Docker). For ML, it ensures reproducibility across environments, simplifies deployment, and eliminates 'works on my machine' issues."],
        "answer": "Containerization packages code and dependencies (Docker). For ML it ensures reproducibility, simplifies deployment, eliminates environment inconsistencies.",
        "ground_truths": ["Containerization ensures reproducibility and deployment consistency for ML workloads."],
        "relevant_doc_ids": ["mlops-4"],
    },
    {
        "question": "What is the difference between batch and online inference?",
        "contexts": ["Batch inference processes many samples together offline, optimizing for throughput. Online inference serves individual or small batches of requests in real time, optimizing for latency."],
        "answer": "Batch inference processes many samples offline for throughput. Online inference handles real-time requests with low latency.",
        "ground_truths": ["Batch inference is offline/throughput-optimized; online inference is real-time/latency-optimized."],
        "relevant_doc_ids": ["mlops-5"],
    },

    # ── Domain 8: Security / AI Safety ─────────────────────────────────────
    {
        "question": "What is prompt injection?",
        "contexts": ["Prompt injection is an attack where malicious input manipulates an LLM into following attacker instructions. It can override system prompts, exfiltrate data, or cause the model to take unintended actions."],
        "answer": "Prompt injection manipulates LLMs via malicious input to override system prompts, exfiltrate data, or cause unintended actions.",
        "ground_truths": ["Prompt injection manipulates LLMs via malicious input to override instructions or exfiltrate data."],
        "relevant_doc_ids": ["sec-1"],
    },
    {
        "question": "What is the principle of least privilege in AI systems?",
        "contexts": ["The principle of least privilege means AI agents and systems should only have access to the resources and permissions they need for their specific task. This limits the blast radius of compromised or misbehaving AI components."],
        "answer": "Least privilege means AI systems get only the minimum permissions needed, limiting damage from compromised components.",
        "ground_truths": ["Least privilege restricts AI systems to minimum necessary permissions to limit compromise impact."],
        "relevant_doc_ids": ["sec-2"],
    },
    {
        "question": "What is data poisoning in machine learning?",
        "contexts": ["Data poisoning attacks inject malicious training data to corrupt a model's behavior. Attackers can cause specific inputs to be misclassified or plant backdoors activated by trigger patterns."],
        "answer": "Data poisoning injects malicious training data to corrupt model behavior, causing misclassification or planting backdoor triggers.",
        "ground_truths": ["Data poisoning injects malicious training data to cause misclassification or backdoor behaviors."],
        "relevant_doc_ids": ["sec-3"],
    },
    {
        "question": "What is PII and why must it be handled carefully in RAG systems?",
        "contexts": ["PII (Personally Identifiable Information) includes data that can identify an individual, such as names, emails, SSNs. In RAG systems, if PII is indexed in the knowledge base and retrieved, it may be exposed in model outputs, violating privacy regulations."],
        "answer": "PII is data identifying individuals. In RAG, indexed PII can leak into outputs, violating privacy regulations like GDPR.",
        "ground_truths": ["PII in RAG knowledge bases can leak into model outputs, violating GDPR and other privacy regulations."],
        "relevant_doc_ids": ["sec-4"],
    },
    {
        "question": "What is content security policy (CSP) in web applications?",
        "contexts": ["Content Security Policy is an HTTP security header that controls which resources a browser is allowed to load. It mitigates XSS attacks by restricting inline scripts and external resource origins."],
        "answer": "CSP is an HTTP header controlling allowed browser resources, mitigating XSS by restricting inline scripts and external origins.",
        "ground_truths": ["CSP is an HTTP header that restricts resource loading to mitigate XSS attacks."],
        "relevant_doc_ids": ["sec-5"],
    },

    # ── Domain 9: Data Engineering ──────────────────────────────────────────
    {
        "question": "What is ETL and how does it work?",
        "contexts": ["ETL stands for Extract, Transform, Load. It is a data pipeline pattern where data is extracted from source systems, transformed (cleaned, normalized, aggregated), and loaded into a destination system such as a data warehouse."],
        "answer": "ETL (Extract, Transform, Load) extracts data from sources, transforms it (clean/normalize), and loads it to a data warehouse.",
        "ground_truths": ["ETL extracts, transforms, and loads data from sources to a destination like a data warehouse."],
        "relevant_doc_ids": ["de-1"],
    },
    {
        "question": "What is Apache Kafka used for?",
        "contexts": ["Apache Kafka is a distributed event streaming platform. It is used for real-time data pipelines, stream processing, and event-driven architectures. Kafka stores events durably and allows multiple consumers to read from the same stream."],
        "answer": "Kafka is a distributed event streaming platform for real-time pipelines, stream processing, and event-driven architectures with durable storage.",
        "ground_truths": ["Kafka is a distributed streaming platform for real-time data pipelines and event-driven architectures."],
        "relevant_doc_ids": ["de-2"],
    },
    {
        "question": "What is a data lake vs a data warehouse?",
        "contexts": ["A data lake stores raw, unstructured, and structured data at scale without predefined schemas. A data warehouse stores structured, processed data optimized for analytics and reporting with defined schemas."],
        "answer": "Data lake: raw unstructured data, schema-on-read. Data warehouse: structured, processed data optimized for analytics with defined schemas.",
        "ground_truths": ["Data lakes store raw data with schema-on-read; data warehouses store structured data for analytics."],
        "relevant_doc_ids": ["de-3"],
    },
    {
        "question": "What is dbt (data build tool)?",
        "contexts": ["dbt (data build tool) is a transformation tool that allows data analysts to write modular SQL transformations. It handles dependency management between models, automated testing, and documentation generation for data pipelines."],
        "answer": "dbt enables modular SQL transformations with dependency management, testing, and documentation for data pipelines.",
        "ground_truths": ["dbt is a transformation tool for modular SQL with dependency management and automated testing."],
        "relevant_doc_ids": ["de-4"],
    },
    {
        "question": "What is change data capture (CDC)?",
        "contexts": ["Change Data Capture tracks and captures changes (inserts, updates, deletes) in a database. It enables real-time replication to downstream systems, maintaining data consistency without full table scans."],
        "answer": "CDC tracks database changes (inserts/updates/deletes) for real-time replication to downstream systems without full table scans.",
        "ground_truths": ["CDC tracks database changes for real-time replication without full table scans."],
        "relevant_doc_ids": ["de-5"],
    },

    # ── Domain 10: Responsible AI ───────────────────────────────────────────
    {
        "question": "What is AI bias and how can it be mitigated?",
        "contexts": ["AI bias occurs when a model produces systematically unfair outcomes for certain groups. It can arise from biased training data, model design choices, or feedback loops. Mitigation includes diverse training data, fairness constraints, and regular bias audits."],
        "answer": "AI bias produces unfair outcomes for certain groups from biased data or design. Mitigation: diverse data, fairness constraints, regular audits.",
        "ground_truths": ["AI bias produces unfair outcomes, mitigated by diverse data, fairness constraints, and audits."],
        "relevant_doc_ids": ["rai-1"],
    },
    {
        "question": "What is explainability in AI?",
        "contexts": ["AI explainability (or interpretability) refers to the degree to which humans can understand how an AI system makes decisions. Methods include SHAP values, LIME, and attention visualization. Explainability is required by EU AI Act for high-risk systems."],
        "answer": "AI explainability allows humans to understand model decisions. Methods include SHAP, LIME, attention viz. Required by EU AI Act for high-risk systems.",
        "ground_truths": ["AI explainability helps humans understand model decisions; EU AI Act requires it for high-risk systems."],
        "relevant_doc_ids": ["rai-2"],
    },
    {
        "question": "What is model card documentation?",
        "contexts": ["Model cards are structured documents describing a machine learning model's intended use, training data, evaluation results, and limitations. They enable transparency and help users understand when a model is appropriate for their use case."],
        "answer": "Model cards document ML models: intended use, training data, evaluation results, and limitations for transparency.",
        "ground_truths": ["Model cards document intended use, training data, evaluation, and limitations for model transparency."],
        "relevant_doc_ids": ["rai-3"],
    },
    {
        "question": "What is NIST AI RMF?",
        "contexts": ["NIST AI Risk Management Framework (AI RMF) is a voluntary framework for managing AI risks. It provides guidance across four functions: Govern, Map, Measure, and Manage. Organizations use it to identify and mitigate risks in AI systems."],
        "answer": "NIST AI RMF is a voluntary framework with four functions (Govern, Map, Measure, Manage) for identifying and mitigating AI risks.",
        "ground_truths": ["NIST AI RMF is a voluntary risk management framework with Govern, Map, Measure, and Manage functions."],
        "relevant_doc_ids": ["rai-4"],
    },
    {
        "question": "What is the difference between safety and security in AI?",
        "contexts": ["AI safety addresses unintended harmful behaviors from AI systems, such as misalignment with human values or catastrophic failures. AI security addresses intentional attacks on AI systems, such as adversarial examples and model theft. Both are critical for responsible AI deployment."],
        "answer": "AI safety: unintended harmful behaviors (misalignment, failures). AI security: intentional attacks (adversarial examples, model theft). Both are critical.",
        "ground_truths": ["AI safety is about unintended harmful behavior; AI security is about intentional attacks."],
        "relevant_doc_ids": ["rai-5"],
    },
]


def main(output: str = "data/golden/qa.jsonl", n: int = 50) -> None:
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for sample in SAMPLES[:n]:
            entry = {**sample, "sample_id": str(uuid.uuid4())}
            f.write(json.dumps(entry) + "\n")

    print(f"Written {min(n, len(SAMPLES))} samples to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/golden/qa.jsonl")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    main(args.output, args.n)
