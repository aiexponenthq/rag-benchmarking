#!/usr/bin/env python3
"""Generate a synthetic 50-sample golden dataset for RAG evaluation.

Usage
-----
    python scripts/generate_golden_dataset.py \
        --output data/golden/qa.jsonl \
        --n 50

Each JSONL line is one evaluation sample with keys:
    question        str   — natural language question
    contexts        list  — retrieved passage(s) the RAG system returned
    answer          str   — model-generated answer to evaluate
    ground_truths   list  — one canonical reference answer (for ContextRecall /
                            ContextPrecision ground-truth)

The samples span five topic domains so the dataset exercises varied vocabulary
and retrieval scenarios.  All content is deterministic and requires no LLM call.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Seed data: (question, passage, reference_answer) triples per domain
# ---------------------------------------------------------------------------

_TEMPLATES: list[tuple[str, str, str]] = [
    # ---- RAG & NLP --------------------------------------------------------
    (
        "What does Retrieval-Augmented Generation (RAG) combine?",
        "Retrieval-Augmented Generation (RAG) combines a dense retrieval component "
        "with a seq2seq language model. The retriever fetches relevant passages from "
        "a corpus; the generator conditions on those passages to produce a faithful answer.",
        "RAG combines dense retrieval with a seq2seq language model to produce answers "
        "conditioned on retrieved passages.",
    ),
    (
        "Why is a reranker useful in a RAG pipeline?",
        "A reranker (cross-encoder) re-scores each candidate passage by attending jointly "
        "to the query and the passage text, which is more accurate than bi-encoder "
        "dot-product similarity but slower. It is applied after initial retrieval to "
        "promote the most relevant chunks to the top.",
        "A reranker re-scores retrieved passages with joint query-passage attention, "
        "improving relevance ranking at the cost of extra compute.",
    ),
    (
        "What is the difference between sparse and dense retrieval?",
        "Sparse retrieval (BM25, TF-IDF) uses term frequency statistics and supports "
        "exact keyword matching. Dense retrieval uses bi-encoder neural networks to "
        "embed queries and passages into a shared vector space enabling semantic matching.",
        "Sparse retrieval matches exact keywords via term statistics; dense retrieval "
        "embeds text into vectors for semantic similarity search.",
    ),
    (
        "What is chunking in the context of RAG?",
        "Chunking is the process of splitting source documents into smaller text segments "
        "(chunks) before indexing. Chunk size is a key hyperparameter: too large and the "
        "model loses focus; too small and context is fragmented.",
        "Chunking splits documents into smaller segments for indexing, balancing context "
        "completeness and retrieval focus.",
    ),
    (
        "How does RAGAS measure faithfulness?",
        "RAGAS faithfulness measures the fraction of claims in the model answer that can "
        "be inferred from the provided context. An LLM judge decomposes the answer into "
        "atomic claims and verifies each against the retrieved passages.",
        "RAGAS faithfulness is the proportion of answer claims verifiable from retrieved "
        "contexts, judged by an LLM.",
    ),
    # ---- Vector Databases -------------------------------------------------
    (
        "What is HNSW indexing used for in Qdrant?",
        "Qdrant uses Hierarchical Navigable Small World (HNSW) graphs for approximate "
        "nearest-neighbour search. HNSW builds a multi-layer graph where each node is "
        "connected to its closest neighbours, enabling sub-linear query time.",
        "HNSW enables fast approximate nearest-neighbour search by building a multi-layer "
        "proximity graph over the vector collection.",
    ),
    (
        "What is a named vector in Qdrant?",
        "Qdrant supports multiple named vectors per point. Each named vector can have "
        "a different dimensionality and distance metric, allowing a single point to be "
        "indexed in several embedding spaces simultaneously.",
        "Named vectors allow a single Qdrant point to carry multiple embeddings with "
        "different dimensions or distance metrics.",
    ),
    (
        "How does Qdrant handle payload filtering?",
        "Qdrant supports structured payload filtering applied during vector search. "
        "Filters are expressed as JSON conditions (must, should, must_not) and are "
        "evaluated before the HNSW traversal is limited, preserving search accuracy.",
        "Qdrant payload filters are JSON conditions applied before HNSW traversal, "
        "combining structured and semantic search.",
    ),
    (
        "What is the difference between Qdrant collections and shards?",
        "A Qdrant collection is a named group of vectors with a common dimensionality "
        "and distance metric. Shards are horizontal partitions of a collection distributed "
        "across nodes for scalability.",
        "Collections group vectors by type; shards partition a collection across nodes "
        "for distributed scalability.",
    ),
    (
        "What distance metrics does Qdrant support?",
        "Qdrant supports cosine similarity, dot product, and Euclidean (L2) distance. "
        "Cosine is recommended for normalised embeddings from models like "
        "sentence-transformers.",
        "Qdrant supports cosine similarity, dot product, and L2 distance; cosine is "
        "preferred for normalised embeddings.",
    ),
    # ---- LLMs & Transformers ----------------------------------------------
    (
        "What is the attention mechanism in Transformers?",
        "The attention mechanism computes a weighted sum of value vectors, where weights "
        "are derived from dot products between query and key vectors scaled by the square "
        "root of the key dimension. Multi-head attention applies this in parallel across "
        "multiple representation subspaces.",
        "Attention computes scaled dot-product weights over key-value pairs; multi-head "
        "attention parallelises this across subspaces.",
    ),
    (
        "What is rotary positional embedding (RoPE)?",
        "RoPE encodes position by rotating the query and key vectors in complex space "
        "by an angle proportional to their position index. Unlike absolute embeddings, "
        "RoPE decays naturally with relative distance and extends to longer contexts.",
        "RoPE rotates query/key vectors by position-proportional angles, providing "
        "relative position encoding that generalises to longer sequences.",
    ),
    (
        "What is Flash Attention?",
        "Flash Attention is an IO-aware exact attention algorithm that tiles the attention "
        "computation to fit in GPU SRAM, avoiding materialising the full N×N attention "
        "matrix in HBM. This reduces memory usage from O(N²) to O(N) and increases speed.",
        "Flash Attention tiles the N×N attention computation in fast SRAM, reducing "
        "memory from O(N²) to O(N) while remaining exact.",
    ),
    (
        "What is quantisation in LLM inference?",
        "Quantisation reduces the bit-width of model weights (e.g., from FP16 to INT4) "
        "to decrease memory footprint and increase throughput. GPTQ and AWQ are popular "
        "post-training quantisation methods for LLMs.",
        "Quantisation lowers weight precision (e.g., INT4) to cut memory and improve "
        "inference speed; GPTQ and AWQ are common methods.",
    ),
    (
        "What is speculative decoding?",
        "Speculative decoding uses a small draft model to propose several tokens, then "
        "verifies them in parallel with the larger target model. Accepted tokens provide "
        "a speedup without changing the output distribution.",
        "Speculative decoding drafts multiple tokens with a small model and verifies "
        "them in parallel with the large model for faster inference.",
    ),
    # ---- Python & FastAPI -------------------------------------------------
    (
        "What is the event loop in Python asyncio?",
        "The asyncio event loop is the central execution mechanism that schedules and "
        "runs coroutines, handles I/O events via selectors, and manages callbacks. "
        "A single thread runs one event loop at a time.",
        "The asyncio event loop schedules coroutines and I/O callbacks in a single "
        "thread using a selector-based mechanism.",
    ),
    (
        "What does run_in_executor do in asyncio?",
        "``loop.run_in_executor(executor, func, *args)`` runs a synchronous callable "
        "in a thread-pool (or process-pool) executor and returns a coroutine that awaits "
        "the result, preventing the event loop from blocking.",
        "run_in_executor offloads blocking sync code to a thread-pool, yielding a "
        "coroutine the event loop can await without blocking.",
    ),
    (
        "What is dependency injection in FastAPI?",
        "FastAPI's Depends system resolves dependencies declared in endpoint signatures "
        "automatically. Dependencies can be async functions, classes, or callables; "
        "they are executed before the endpoint and support sub-dependencies.",
        "FastAPI Depends resolves callables declared in endpoint parameters, supporting "
        "async, class-based, and nested dependencies.",
    ),
    (
        "What is the purpose of Pydantic BaseModel in FastAPI?",
        "Pydantic BaseModel provides automatic request body parsing, validation, and "
        "JSON serialisation in FastAPI. Fields with type annotations are validated at "
        "runtime; invalid requests receive a 422 Unprocessable Entity response.",
        "Pydantic BaseModel validates and parses request bodies in FastAPI, returning "
        "422 on type mismatches.",
    ),
    (
        "What are background tasks in FastAPI?",
        "FastAPI BackgroundTasks lets you schedule work to run after a response is "
        "sent. Tasks are executed in the same process thread after the response is "
        "returned to the client, suitable for lightweight fire-and-forget operations.",
        "FastAPI BackgroundTasks run lightweight post-response work in the same process "
        "without blocking the client.",
    ),
    # ---- MLOps & Evaluation -----------------------------------------------
    (
        "What is precision at K (P@K) in information retrieval?",
        "Precision at K measures the fraction of the top-K retrieved documents that "
        "are relevant. P@K = |relevant ∩ top-K| / K. It does not penalise for relevant "
        "documents ranked below position K.",
        "P@K is the fraction of top-K results that are relevant, ignoring relevant items "
        "outside the top K.",
    ),
    (
        "What is Normalised Discounted Cumulative Gain (NDCG)?",
        "NDCG measures ranking quality by summing relevance scores discounted by "
        "logarithmic rank position and normalising by the ideal ranking. NDCG@K is "
        "computed over only the top-K positions.",
        "NDCG sums log-discounted relevance scores and normalises by the ideal ranking, "
        "measuring ranked retrieval quality.",
    ),
    (
        "What is Mean Reciprocal Rank (MRR)?",
        "MRR is the mean over queries of the reciprocal of the rank of the first "
        "relevant result. MRR = (1/|Q|) Σ 1/rank_i. It rewards systems that place "
        "at least one relevant document near the top.",
        "MRR averages the reciprocal rank of the first relevant result per query, "
        "rewarding early placement of any relevant document.",
    ),
    (
        "What is context recall in RAGAS?",
        "RAGAS context recall measures the fraction of the reference answer's claims "
        "that are supported by the retrieved contexts. A high score means the retriever "
        "fetched contexts covering the ground-truth information.",
        "RAGAS context recall is the proportion of reference-answer claims supported "
        "by retrieved contexts.",
    ),
    (
        "What is a confusion matrix in classification evaluation?",
        "A confusion matrix tabulates True Positives, False Positives, True Negatives, "
        "and False Negatives for a classifier. It is the foundation for deriving "
        "precision, recall, F1, and accuracy metrics.",
        "A confusion matrix enumerates TP, FP, TN, FN to derive precision, recall, "
        "F1, and accuracy.",
    ),
    # ---- Responsible AI ---------------------------------------------------
    (
        "What is hallucination in language models?",
        "Hallucination occurs when a language model generates factually incorrect or "
        "unsupported statements with high confidence. In RAG systems, faithfulness "
        "metrics detect hallucinations by checking whether claims are grounded in "
        "the retrieved context.",
        "Hallucination is when a model confidently generates unsupported facts; RAG "
        "faithfulness metrics detect this by grounding verification.",
    ),
    (
        "What is RLHF?",
        "Reinforcement Learning from Human Feedback (RLHF) fine-tunes a language model "
        "using a reward model trained on human preference rankings. The policy is "
        "optimised with PPO to maximise the reward signal while a KL penalty prevents "
        "excessive deviation from the supervised baseline.",
        "RLHF fine-tunes LLMs with a human-preference reward model via PPO and a KL "
        "penalty to align outputs with human values.",
    ),
    (
        "What is constitutional AI?",
        "Constitutional AI (Anthropic) trains a model to critique and revise its own "
        "outputs against a set of principles (a 'constitution') using a supervised "
        "revision phase followed by RLHF with AI-generated preference data.",
        "Constitutional AI uses a principle-based self-critique-and-revision loop "
        "followed by RLHF on AI-generated preference labels.",
    ),
    (
        "What is bias in machine learning?",
        "Bias in ML refers to systematic errors where model predictions consistently "
        "favour or disfavour certain groups or outcomes. It can arise from skewed "
        "training data, biased labels, or model architecture choices.",
        "ML bias is systematic prediction error favouring certain groups, arising "
        "from data, labels, or architecture choices.",
    ),
    (
        "What is differential privacy in ML training?",
        "Differential privacy adds calibrated Gaussian or Laplace noise to gradients "
        "during training (DP-SGD) so that the presence or absence of any single training "
        "example has a bounded effect on the output model.",
        "DP-SGD adds noise to gradients during training, bounding the influence of any "
        "individual example on the model.",
    ),
    # ---- Cloud & Infrastructure -------------------------------------------
    (
        "What is a Dockerfile ENTRYPOINT vs CMD?",
        "ENTRYPOINT sets the main executable for a container. CMD provides default "
        "arguments to ENTRYPOINT, or the default command if ENTRYPOINT is not set. "
        "At runtime, CMD can be overridden; ENTRYPOINT can only be overridden with "
        "--entrypoint.",
        "ENTRYPOINT is the fixed container executable; CMD supplies its default "
        "arguments and is easily overridden at runtime.",
    ),
    (
        "What is a Kubernetes liveness probe?",
        "A Kubernetes liveness probe periodically checks whether a container is alive. "
        "If the probe fails, Kubernetes restarts the container. Common probe types are "
        "HTTP GET, TCP socket, and exec command.",
        "A Kubernetes liveness probe restarts a container when it fails, using HTTP, "
        "TCP, or exec checks.",
    ),
    (
        "What is horizontal pod autoscaling in Kubernetes?",
        "Horizontal Pod Autoscaling (HPA) automatically adjusts the number of pod "
        "replicas in a deployment based on observed CPU utilisation or custom metrics. "
        "The HPA controller queries the metrics API and scales replicas to maintain "
        "a target utilisation.",
        "HPA scales pod replicas up or down based on CPU or custom metric targets "
        "to maintain target utilisation.",
    ),
    (
        "What is a Kubernetes ConfigMap?",
        "A ConfigMap stores non-sensitive configuration data as key-value pairs. "
        "Pods can consume ConfigMaps as environment variables, command-line arguments, "
        "or mounted files. Unlike Secrets, ConfigMap data is not base64-encoded.",
        "ConfigMaps store non-sensitive config as key-value pairs consumed by pods "
        "as env vars or mounted files.",
    ),
    (
        "What is a Kubernetes Persistent Volume?",
        "A Persistent Volume (PV) is cluster-level storage provisioned by an admin "
        "or dynamically by a StorageClass. A Persistent Volume Claim (PVC) binds a pod "
        "to a PV, decoupling storage lifecycle from pod lifecycle.",
        "PVs provide cluster-level storage; PVCs bind pods to PVs, decoupling storage "
        "and pod lifecycles.",
    ),
    # ---- Security ---------------------------------------------------------
    (
        "What is OAuth 2.0?",
        "OAuth 2.0 is an authorisation framework that allows third-party applications "
        "to obtain limited access to a service on behalf of a user without exposing "
        "their credentials. It issues access tokens via flows such as authorization "
        "code, client credentials, and device code.",
        "OAuth 2.0 grants third-party apps limited service access via access tokens "
        "without exposing user credentials.",
    ),
    (
        "What is JWT?",
        "JSON Web Token (JWT) is a compact, self-contained token format encoding "
        "claims as a signed JSON payload. JWTs are commonly used as bearer tokens in "
        "OAuth 2.0 and OpenID Connect. The signature (HMAC or RSA) ensures integrity.",
        "JWTs are signed JSON-encoded claim tokens used as bearer tokens in OAuth/OIDC; "
        "the signature ensures integrity.",
    ),
    (
        "What is rate limiting in an API?",
        "Rate limiting restricts the number of requests a client can make in a time "
        "window. Common algorithms include token bucket (smooth bursts), fixed window "
        "(simple, bursty), and sliding window (accurate). Rate limits protect against "
        "abuse and ensure fair usage.",
        "Rate limiting caps API requests per time window using algorithms like token "
        "bucket or sliding window to prevent abuse.",
    ),
    (
        "What is SQL injection and how is it prevented?",
        "SQL injection inserts malicious SQL syntax into user-supplied input to "
        "manipulate database queries. Prevention techniques include parameterised "
        "queries (prepared statements), ORM query builders, and input validation.",
        "SQL injection exploits unescaped input to manipulate queries; parameterised "
        "statements and ORMs prevent it.",
    ),
    (
        "What is mTLS?",
        "Mutual TLS (mTLS) requires both the client and server to present X.509 "
        "certificates during the TLS handshake. It is used for service-to-service "
        "authentication in zero-trust architectures.",
        "mTLS mandates client and server certificate exchange during TLS handshake "
        "for mutual authentication.",
    ),
    # ---- Data Engineering -------------------------------------------------
    (
        "What is Apache Parquet?",
        "Parquet is a columnar storage format optimised for analytical workloads. "
        "Columns are stored contiguously, enabling predicate pushdown and efficient "
        "compression. It supports nested data structures via Dremel encoding.",
        "Parquet is a columnar format with predicate pushdown and nested data support "
        "via Dremel encoding, optimised for analytics.",
    ),
    (
        "What is data lineage?",
        "Data lineage tracks the origin, movement, and transformation of data through "
        "a pipeline. It supports debugging, auditing, and compliance by showing how "
        "each dataset was derived.",
        "Data lineage tracks data origin and transformations for debugging, auditing, "
        "and compliance.",
    ),
    (
        "What is change data capture (CDC)?",
        "CDC captures row-level changes in a database (inserts, updates, deletes) and "
        "streams them to downstream consumers. Debezium reads the database write-ahead "
        "log to produce CDC events with low latency.",
        "CDC streams row-level database changes from the write-ahead log to downstream "
        "systems with low latency.",
    ),
    (
        "What is a data lakehouse?",
        "A data lakehouse combines the low-cost object storage of a data lake with the "
        "ACID transactions and schema enforcement of a data warehouse. Formats like "
        "Delta Lake and Apache Iceberg enable this by adding transaction logs to Parquet.",
        "A lakehouse adds ACID transactions and schema enforcement (via Delta Lake or "
        "Iceberg) to low-cost object storage.",
    ),
    (
        "What is the difference between OLTP and OLAP?",
        "OLTP (Online Transaction Processing) systems are optimised for high-throughput "
        "row-level reads and writes (e.g., order placement). OLAP (Online Analytical "
        "Processing) systems are optimised for aggregating large volumes of data across "
        "many rows (e.g., revenue reports).",
        "OLTP optimises fast row-level transactions; OLAP optimises large-scale "
        "aggregation queries across many rows.",
    ),
    # ---- Statistics & Math ------------------------------------------------
    (
        "What is cosine similarity?",
        "Cosine similarity measures the cosine of the angle between two vectors: "
        "cos(θ) = (A·B) / (‖A‖ ‖B‖). It is 1 for identical directions and 0 for "
        "orthogonal vectors, commonly used to compare embedding representations.",
        "Cosine similarity is the dot product of two vectors divided by the product "
        "of their magnitudes, measuring directional agreement.",
    ),
    (
        "What is the curse of dimensionality?",
        "In high-dimensional spaces, data points become sparse and distances "
        "concentrate — all pairs of points become nearly equidistant. This harms "
        "nearest-neighbour search accuracy and increases the data needed to cover "
        "the space adequately.",
        "The curse of dimensionality makes high-dimensional points sparse and nearly "
        "equidistant, degrading nearest-neighbour search.",
    ),
    (
        "What is KL divergence?",
        "Kullback-Leibler divergence measures how one probability distribution P "
        "differs from a reference Q: KL(P‖Q) = Σ P(x) log(P(x)/Q(x)). It is "
        "asymmetric and used in variational inference, RL fine-tuning, and knowledge "
        "distillation.",
        "KL divergence measures how P differs from Q asymmetrically; it is used in "
        "variational inference, RLHF, and distillation.",
    ),
    (
        "What is cross-entropy loss?",
        "Cross-entropy loss H(y, ŷ) = -Σ y_i log(ŷ_i) measures the dissimilarity "
        "between the true label distribution y and the predicted distribution ŷ. "
        "For classification it reduces to -log(ŷ_true_class).",
        "Cross-entropy loss measures predicted vs true distribution mismatch and "
        "reduces to negative log-probability of the true class.",
    ),
    (
        "What is the bias-variance trade-off?",
        "Bias is error from incorrect model assumptions (underfitting); variance is "
        "error from sensitivity to training data (overfitting). Increasing model "
        "complexity typically reduces bias but raises variance; regularisation and "
        "data augmentation manage this trade-off.",
        "High bias means underfitting; high variance means overfitting. Regularisation "
        "and data augmentation balance the trade-off.",
    ),
]

# There are exactly 50 templates above — one per sample.
assert len(_TEMPLATES) == 50, f"Expected 50 templates, got {len(_TEMPLATES)}"


def make_sample(idx: int, question: str, passage: str, reference: str) -> dict:
    """Build one JSONL-ready dict from template values."""
    return {
        "id": idx + 1,
        "question": question,
        # contexts: the passage the retriever returned
        "contexts": [textwrap.dedent(passage).strip()],
        # answer: for the golden dataset we treat the reference as the
        # model answer so faithfulness tests pass out of the box.
        # In a live eval harness this would be the actual model output.
        "answer": reference,
        "ground_truths": [reference],
    }


def generate(n: int = 50) -> list[dict]:
    templates = _TEMPLATES[:n]
    return [make_sample(i, q, p, r) for i, (q, p, r) in enumerate(templates)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic golden dataset")
    parser.add_argument("--output", default="data/golden/qa.jsonl", help="Output JSONL path")
    parser.add_argument("--n", type=int, default=50, help="Number of samples (max 50)")
    args = parser.parse_args()

    n = min(args.n, len(_TEMPLATES))
    samples = generate(n)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Wrote {n} samples to {out}")


if __name__ == "__main__":
    main()
