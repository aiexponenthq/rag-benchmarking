from __future__ import annotations

from app.retrieval.chunking import recursive_character_chunk


def test_chunk_boundaries_and_overlap() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 10
    chunks = recursive_character_chunk(text, chunk_size=50, chunk_overlap=10, source_id="test")
    assert len(chunks) > 1
    # Ensure overlap: end of chunk i intersects with start of chunk i+1
    for i in range(len(chunks) - 1):
        assert chunks[i].text[-10:] == chunks[i + 1].text[:10]
