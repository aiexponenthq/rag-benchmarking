# Manual Testing Guide

Step-by-step instructions to verify the rag-benchmarking tool is working correctly on your machine. No prior experience with the codebase required.

---

## What you need before starting

| Requirement | Where to get it |
|---|---|
| Python 3.11+ | python.org or `brew install python@3.11` |
| A Gemini API key | aistudio.google.com → Get API key (free tier available) |
| Git | Already installed on your machine |

> **Note:** Qdrant (the vector database) is only needed if you want to test the built-in RAG pipeline. The evaluation harness itself works without it — you can skip those steps.

---

## Part 1: Setup (5 minutes)

### Step 1 — Clone and enter the repo

```bash
cd /Users/ajayp/Code/rag-benchmarking
```

### Step 2 — Install dependencies

```bash
pip install -e ".[test]"
```

You should see packages installing. It will take 2–3 minutes the first time.

**Verify it worked:**
```bash
python -c "from app.sdk.client import RagEval; print('OK')"
```
Expected output: `OK`

### Step 3 — Create your `.env` file

```bash
cp .env.example .env
```

Now open `.env` in your editor and fill in your Gemini key:

```bash
GEMINI_API_KEY=your-actual-key-here
LLM_PROVIDER=gemini
API_KEY=test-local-key
ENFORCE_API_KEY=true
```

Leave everything else as-is for now.

---

## Part 2: Run the automated tests (2 minutes)

This verifies all the evaluation logic works correctly without needing any API keys or server.

```bash
python -m pytest tests/unit/ -v
```

**Expected output:** All tests pass. You will see lines like:
```
tests/unit/test_schemas.py::test_eval_sample_minimal PASSED
tests/unit/test_retrieval_metrics.py::test_precision_at_k_perfect PASSED
...
84 passed
```

If any test fails, stop here and check that Step 2 completed successfully.

---

## Part 3: Start the server (1 minute)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 5001
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:5001
INFO:     Application startup complete.
```

Leave this terminal running. Open a **new terminal** for the remaining steps.

### Verify the server is alive

```bash
curl http://localhost:5001/health
```

Expected response:
```json
{"status": "ok"}
```

---

## Part 4: Test the evaluation API

These tests show the core feature — evaluating a RAG system's output.

### Test 4a — Evaluate a single sample (no LLM needed)

This uses `source_attribution_accuracy`, which is fully deterministic (no API key required):

```bash
curl -s -X POST http://localhost:5001/v1/evaluate \
  -H "X-API-Key: test-local-key" \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [{
      "question": "What is the EU AI Act?",
      "contexts": ["The EU AI Act is the world'\''s first comprehensive AI regulation, entering into force in August 2024."],
      "answer": "The EU AI Act is the world'\''s first comprehensive AI regulation."
    }],
    "metrics": ["source_attribution_accuracy"]
  }' | python -m json.tool
```

Expected response — you will see a score of 1.0 (perfect attribution, no hallucinated sources):
```json
{
    "metrics": {
        "source_attribution_accuracy": 1.0
    },
    "skipped_metrics": []
}
```

### Test 4b — Evaluate faithfulness (requires Gemini API key)

```bash
curl -s -X POST http://localhost:5001/v1/evaluate \
  -H "X-API-Key: test-local-key" \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [{
      "question": "What is RAG?",
      "contexts": ["RAG stands for Retrieval-Augmented Generation. It combines information retrieval with LLM generation to reduce hallucinations."],
      "answer": "RAG stands for Retrieval-Augmented Generation. It reduces hallucinations by grounding responses in retrieved documents."
    }],
    "metrics": ["faithfulness"]
  }' | python -m json.tool
```

Expected response — faithfulness score between 0.8 and 1.0:
```json
{
    "metrics": {
        "faithfulness": 0.92
    },
    "skipped_metrics": []
}
```

> If you see an error like `GEMINI_API_KEY is required`, double-check your `.env` file and restart the server.

### Test 4c — Test with ground truth (enables context_precision + context_recall)

```bash
curl -s -X POST http://localhost:5001/v1/evaluate \
  -H "X-API-Key: test-local-key" \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [{
      "question": "What does Article 53 of the EU AI Act require?",
      "contexts": ["Article 53 requires GPAI model providers to publish technical documentation, comply with copyright law, and publish summaries of training data used."],
      "answer": "Article 53 requires GPAI providers to publish technical documentation and training data summaries.",
      "ground_truths": ["Article 53 requires GPAI model providers to publish technical documentation and training data summaries."]
    }],
    "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
  }' | python -m json.tool
```

Expected — all four scores between 0.7 and 1.0.

---

## Part 5: Test the agentic evaluation endpoint

This tests the new agentic feature — evaluating a multi-step agent's reasoning trace:

```bash
curl -s -X POST http://localhost:5001/v1/evaluate/agent \
  -H "X-API-Key: test-local-key" \
  -H "Content-Type: application/json" \
  -d '{
    "trace": {
      "question": "When do GPAI obligations apply under the EU AI Act?",
      "final_answer": "GPAI obligations under Article 53 apply from August 2025.",
      "tool_calls": [{
        "tool_name": "retrieve",
        "tool_input": {"query": "GPAI EU AI Act deadline"},
        "tool_output": "Article 53 GPAI obligations apply from August 2025.",
        "step_index": 0
      }],
      "reasoning_steps": [],
      "retrieved_chunks": []
    },
    "metrics": ["source_attribution_accuracy"]
  }' | python -m json.tool
```

Expected — `source_attribution_accuracy` of 1.0 (no hallucinated sources).

---

## Part 6: Test run history

After running evaluations, you can retrieve and compare runs:

```bash
# List all runs
curl -s http://localhost:5001/v1/runs \
  -H "X-API-Key: test-local-key" | python -m json.tool
```

Expected — a list of run objects, each with a `run_id`, `n_samples`, and `metrics`.

---

## Part 7: Test the Python SDK

Open a Python shell in a new terminal:

```bash
python3
```

Then paste:

```python
from app.sdk.client import RagEval

client = RagEval(api_url="http://localhost:5001", api_key="test-local-key")

sample = {
    "question": "What is a vector database?",
    "contexts": ["A vector database stores high-dimensional embeddings and uses approximate nearest neighbour algorithms for fast similarity search."],
    "answer": "A vector database stores embeddings for fast similarity search using ANN algorithms.",
}

report = client.evaluate([sample], metrics=["source_attribution_accuracy"])
print("Score:", report)
```

Expected — `Score:` followed by a dict with `source_attribution_accuracy: 1.0`.

---

## Part 8: Run the golden dataset benchmark

This runs evaluation against all 50 benchmark samples and saves a report:

```bash
python scripts/evaluate.py data/golden/qa.jsonl \
  --metrics faithfulness answer_relevancy \
  --out-json reports/my-first-run.json
```

This will take 3–8 minutes as it makes LLM API calls for 50 samples.

**View the results:**
```bash
cat reports/my-first-run.json | python -m json.tool | head -30
```

Expected — aggregate scores around faithfulness 0.85–0.92, answer_relevancy 0.88–0.94.

---

## Part 9: Interactive API explorer

FastAPI automatically generates an interactive API explorer. Open in your browser:

```
http://localhost:5001/docs
```

You can click any endpoint, fill in the request body, hit "Execute", and see the live response. This is the easiest way to explore all available endpoints.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `pip install` fails | Ensure Python 3.11+: `python --version` |
| `ModuleNotFoundError: app` | Run `pip install -e .` from the repo root |
| Server won't start | Check `.env` file exists: `ls -la .env` |
| `401 Unauthorized` | Check `X-API-Key` matches `API_KEY` in `.env` |
| `GEMINI_API_KEY is required` | Add your key to `.env`, restart the server |
| Faithfulness score is 0.0 | LLM judge call failed — check GEMINI_API_KEY is valid |
| `Connection refused` | Server isn't running — go back to Part 3 |

---

## What to test next

Once the above steps work:

1. **Bring your own RAG output** — replace the sample dict in Test 4a with output from your actual RAG system
2. **Use the LangChain adapter** — `RagEval.from_langchain(your_chain_output)`
3. **Compare two configurations** — run evaluation before and after a change, use `/v1/runs/compare`
4. **Read the comparison guide** — `docs/comparison.md` explains how this tool compares to RAGAS, TruLens, DeepEval
