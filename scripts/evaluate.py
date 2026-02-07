from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.eval.ragas_runner import run_evaluation
from app.eval.reporting import write_report_files


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation on a JSONL dataset")
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to JSONL with fields: question, contexts, answer, ground_truths",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="(Deprecated) Path to write JSON report"
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="reports/ragas_report.json",
        help="Path to write JSON report",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="reports/ragas_report.md",
        help="Path to write Markdown summary",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0 = all)")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=None,
        help="Metrics to compute (e.g., faithfulness answer_relevancy)",
    )
    args = parser.parse_args()

    samples = load_jsonl(Path(args.dataset))
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    result = run_evaluation(samples, metrics=args.metrics)
    # Backward compatibility: if --out provided, override out-json
    out_json = Path(args.out) if args.out else Path(args.out_json)
    out_md = Path(args.out_md) if args.out_md else None
    written = write_report_files(result, out_json=out_json, out_md=out_md)
    if written.get("json"):
        print("Saved JSON:", written["json"])
    if written.get("md"):
        print("Saved Markdown:", written["md"])


if __name__ == "__main__":
    main()
