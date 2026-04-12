# RAG Benchmarking

[![PyPI](https://img.shields.io/pypi/v/rag-benchmarking.svg)](https://pypi.org/project/rag-benchmarking/)
[![CI](https://github.com/aiexponenthq/rag-benchmarking/actions/workflows/ci.yml/badge.svg)](https://github.com/aiexponenthq/rag-benchmarking/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![EU AI Act](https://img.shields.io/badge/EU%20AI%20Act-Article%2015-gold.svg)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689)

**A framework-agnostic evaluation harness for RAG and agentic AI systems.**

Bring your own RAG pipeline — LangChain, LlamaIndex, or custom — and benchmark it against classic and agentic-era metrics. Built for teams who need to prove their AI systems work before they ship.

Built by [AiExponent LLC](https://aiexponent.com). Maps to EU AI Act Article 15 (accuracy requirements).

---

## Quick Start

```bash
pip install rag-benchmarking
```

```python
from app.sdk.client import RagEval

client = RagEval(api_url="http://localhost:5001", api_key="your-key")

# Works with LangChain
result = my_chain.invoke({"query": "What is RAG?"})
sample = RagEval.from_langchain(result)

# Or any dict with question / contexts / answer
sample = {
    "question": "What is RAG?",
    "contexts": ["RAG stands for Retrieval-Augmented Generation."],
    "answer": "RAG combines retrieval with LLM generation.",
}

report = client.evaluate([sample], metrics=["faithfulness", "answer_relevancy"])
print(report["scores"])
# {"faithfulness": 0.958, "answer_relevancy": 0.810}
```

```bash
# Start the evaluation server
docker compose up
# API docs: http://localhost:5001/docs
```

---

## Architecture

```mermaid
graph TD
    RAG["Your RAG System\nLangChain · LlamaIndex · Custom"]
    SDK["SDK Adapters\nRagEval.from_langchain()\nRagEval.from_llamaindex()"]
    SCHEMA["EvalSample / AgentTrace\nharness/schemas.py"]
    RUNNER["EvaluationRunner\nharness/runner.py"]

    CLASSIC["Classic Metrics\nfaithfulness · answer_relevancy\ncontext_precision · context_recall"]
    RETRIEVAL["Retrieval Metrics\nPrecision@K · Recall@K\nMRR · NDCG"]
    AGENTIC["Agentic Metrics\nagent_faithfulness · tool_call_accuracy\nretrieval_necessity · source_attribution"]

    REPORT["BenchmarkReport"]
    STORE["SQLite ResultStore\nRun history + comparison"]
    API["REST API\n/v1/evaluate · /v1/evaluate/agent\n/v1/runs · /v1/runs/compare"]

    RAG --> SDK --> SCHEMA --> RUNNER
    RUNNER --> CLASSIC
    RUNNER --> RETRIEVAL
    RUNNER --> AGENTIC
    CLASSIC --> REPORT
    RETRIEVAL --> REPORT
    AGENTIC --> REPORT
    REPORT --> STORE --> API

    style RAG fill:#2d5a2d,color:#fff
    style SDK fill:#1e3a5f,color:#fff
    style SCHEMA fill:#1e3a5f,color:#fff
    style RUNNER fill:#1e3a5f,color:#fff
    style CLASSIC fill:#c9a84c,color:#000
    style RETRIEVAL fill:#c9a84c,color:#000
    style AGENTIC fill:#c9a84c,color:#000
    style REPORT fill:#1e3a5f,color:#fff
    style STORE fill:#1e3a5f,color:#fff
    style API fill:#2d5a2d,color:#fff
```

---

## Metrics

### Classic RAG Metrics

```mermaid
graph LR
    Q["question\ncontexts\nanswer"]

    FAITH["faithfulness\nAre all claims in the\nanswer supported by context?"]
    RELEV["answer_relevancy\nDoes the answer\naddress the question?"]
    CPREC["context_precision\nAre retrieved chunks\nrelevant to the query?"]
    CREC["context_recall\nDoes context contain\nenough to answer?"]

    Q --> FAITH
    Q --> RELEV
    Q --> CPREC
    Q --> CREC

    style Q fill:#1e3a5f,color:#fff
    style FAITH fill:#c9a84c,color:#000
    style RELEV fill:#c9a84c,color:#000
    style CPREC fill:#c9a84c,color:#000
    style CREC fill:#c9a84c,color:#000
```

| Metric | What it measures | LLM judge |
|---|---|---|
| `faithfulness` | Are all claims in the answer supported by context? | Yes |
| `answer_relevancy` | Does the answer address the question? | Yes |
| `context_precision` | Are retrieved chunks relevant to the query? | Yes |
| `context_recall` | Does context contain enough to answer correctly? | Yes |
| `precision_at_k` | Fraction of top-K retrieved docs that are relevant | No |
| `recall_at_k` | Fraction of relevant docs found in top-K | No |
| `mrr` | Reciprocal rank of first relevant doc | No |
| `ndcg_at_k` | Rank-weighted retrieval quality | No |

### Agentic-Era Metrics

For multi-step agents, tool-using systems, and autonomous RAG pipelines:

| Metric | What it measures | LLM judge |
|---|---|---|
| `source_attribution_accuracy` | Did the agent cite sources it actually retrieved? | No — deterministic |
| `agent_faithfulness` | Is every reasoning step faithful to retrieved sources? | Yes |
| `tool_call_accuracy` | Did the agent choose the right tool at the right time? | Yes |
| `retrieval_necessity` | Was retrieval actually needed for this query? | Yes |

### Metric Groups

```python
# Use pre-defined groups
report = client.evaluate(samples, metric_group="classic")
report = client.evaluate(samples, metric_group="retrieval")
report = client.evaluate(samples, metric_group="agentic_v1")
report = client.evaluate(samples, metric_group="full")  # all metrics
```

---

## Benchmarks

Measured on the built-in 50-sample golden dataset (10 domains):

| Metric | Score | Label |
|---|---|---|
| faithfulness | **0.958** | Excellent |
| answer_relevancy | **0.810** | Good |

---

## LLM Backend

Several metrics use an LLM as a judge. Supported providers:

```bash
# .env
LLM_PROVIDER=gemini       # recommended
GEMINI_API_KEY=your-key

# Or OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key
```

**Cost guidance:** A full classic-metrics pass on 50 samples costs ~$0.05–$0.15 with Gemini Flash or GPT-4o-mini. Source attribution accuracy is deterministic and costs nothing.

**Determinism:** Judge calls run at `temperature=0.0`. For CI/CD, flag changes beyond ±0.05 rather than asserting exact scores.

---

## API Reference

```bash
# Evaluate a RAG sample
curl -X POST http://localhost:5001/v1/evaluate \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [{"question": "What is RAG?",
      "contexts": ["RAG is Retrieval-Augmented Generation."],
      "answer": "RAG combines retrieval with generation."}],
    "metrics": ["faithfulness", "answer_relevancy"]
  }'

# Evaluate an agentic trace
curl -X POST http://localhost:5001/v1/evaluate/agent \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "trace": {
      "question": "What is the GPAI deadline?",
      "final_answer": "GPAI obligations apply from August 2025.",
      "tool_calls": [{"tool_name": "retrieve",
        "tool_input": {"query": "GPAI deadline"},
        "tool_output": "Article 53 obligations apply from August 2025.",
        "step_index": 0}]
    },
    "metrics": ["source_attribution_accuracy", "tool_call_accuracy"]
  }'

# Compare runs
curl -X POST http://localhost:5001/v1/runs/compare \
  -H "X-API-Key: your-key" \
  -d '["run-id-a", "run-id-b"]'
```

---

## EU AI Act Article 15

```mermaid
graph LR
    RAG["rag-benchmarking\nevaluation harness"]

    A15["Article 15\nAccuracy · Robustness\nCybersecurity"]
    FAITH2["Faithfulness testing\n→ measures hallucination rate"]
    ROBUST["Robustness testing\n→ adversarial + edge case queries"]
    ATTR["Source attribution\n→ verifies citation accuracy"]
    REPORT2["BenchmarkReport\n→ audit-ready evidence\nfor Article 15 compliance"]

    RAG --> FAITH2
    RAG --> ROBUST
    RAG --> ATTR
    FAITH2 --> REPORT2
    ROBUST --> REPORT2
    ATTR --> REPORT2
    A15 -.->|"requires"| REPORT2

    style RAG fill:#c9a84c,color:#000
    style A15 fill:#1e3a5f,color:#fff
    style REPORT2 fill:#2d5a2d,color:#fff
```

Systematic RAG evaluation produces audit-ready evidence for Article 15's accuracy and robustness requirements.

---

## AiExponent Toolchain

rag-benchmarking feeds accuracy evidence into RiskForge for Article 9 risk management:

```mermaid
graph LR
    LCC["LCC\n(Art. 53 licenses)"]
    RAG["rag-benchmarking\n(Art. 15 accuracy)"]
    RF["RiskForge\n(Art. 9 risk management)"]
    TD["TransparencyDeck\n(Art. 13 docs)"]

    LCC -->|"license evidence"| RF
    RAG -->|"benchmark_report.json\naccuracy evidence"| RF
    RF -->|"rmf.json"| TD

    style RAG fill:#c9a84c,color:#000
    style LCC fill:#1e3a5f,color:#fff
    style RF fill:#1e3a5f,color:#fff
    style TD fill:#1e3a5f,color:#fff
```

---

## Configuration

```bash
# .env
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
OPENAI_API_KEY=...

# Vector store (built-in RAG pipeline only)
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=...

# API authentication
API_KEY=your-secret-key
ENFORCE_API_KEY=true
```

---

## Project Structure

```
src/
  harness/            # Framework-agnostic evaluation harness
    schemas.py        # EvalSample, AgentTrace, BenchmarkReport
    protocol.py       # RAGEvaluable Protocol — the plug-in contract
    runner.py         # EvaluationRunner — orchestrates metrics
    result_store.py   # SQLite persistence
  app/
    api/              # FastAPI endpoints
    eval/             # Metric implementations
    sdk/              # Python SDK (RagEval client)
data/
  golden/qa.jsonl     # 50-sample golden dataset (10 domains)
```

---

## Known Limitations

- English-only benchmark datasets; no multilingual evaluation.
- Custom dataset integration requires manual formatting to the JSONL schema.
- Accuracy metrics only — latency and throughput are not measured.
- LLM-as-judge quality depends on the configured judge model.
- Rate limiting is in-memory and resets on server restart.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and PRs welcome.

```bash
git clone https://github.com/aiexponenthq/rag-benchmarking
cd rag-benchmarking
pip install -e ".[test]"
pytest
```

---

## License

[Apache 2.0](LICENSE) — free to use, modify, and distribute.

Built by [AiExponent LLC](https://aiexponent.com) — `hello@aiexponent.com`

---

*Part of the AiExponent open-source AI governance toolchain:
[license-compliance-checker](https://github.com/aiexponenthq/license-compliance-checker) ·
**rag-benchmarking** ·
[RiskForge](https://github.com/aiexponenthq/riskforge)*
