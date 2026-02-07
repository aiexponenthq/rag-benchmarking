from __future__ import annotations

import argparse
from pathlib import Path

from app.config.settings import get_settings
from app.retrieval.chunking import TextChunk, recursive_character_chunk
from app.retrieval.embeddings import EmbeddingsClient
from app.retrieval.qdrant_store import ensure_collection, get_qdrant_client, upsert_points


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest plain text/markdown files into Qdrant Cloud"
    )
    parser.add_argument("paths", nargs="+", help="File or directory paths to ingest")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--embeddings", type=str, default="BAAI/bge-base-en-v1.5")
    args = parser.parse_args()

    settings = get_settings()
    client = get_qdrant_client()
    embedder = EmbeddingsClient(model_name=args.embeddings)

    all_chunks: list[TextChunk] = []
    for p in args.paths:
        path = Path(p)
        files: list[Path] = []
        if path.is_dir():
            files = [
                f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in {".txt", ".md"}
            ]
        elif path.is_file():
            files = [path]
        else:
            continue
        for f in files:
            txt = read_text_file(f)
            chunks = recursive_character_chunk(
                txt, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, source_id=str(f)
            )
            all_chunks.extend(chunks)

    if not all_chunks:
        print("No files or chunks to ingest.")
        return

    texts = [c.text for c in all_chunks]
    print(f"Embedding {len(texts)} chunks ...")
    vectors = embedder.embed(texts)

    # Try to ensure a named vector schema 'content' for portability
    collection_name, vector_name = ensure_collection(
        client,
        settings.qdrant_collection,
        vector_size=vectors.shape[1],
        desired_vector_name="content",
    )

    payloads = [
        {"source_id": c.source_id, "chunk_index": c.chunk_index, "text": c.text} for c in all_chunks
    ]

    print("Upserting to Qdrant Cloud ...")
    upsert_points(client, collection_name, vectors, payloads, vector_name=vector_name)
    print("Done.")


if __name__ == "__main__":
    main()
