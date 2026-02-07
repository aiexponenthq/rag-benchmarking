from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextChunk:
    """Represents a chunk of text with its source identifier and position."""

    text: str
    source_id: str
    chunk_index: int


def recursive_character_chunk(
    text: str, *, chunk_size: int = 1000, chunk_overlap: int = 150, source_id: str
) -> list[TextChunk]:
    """Split text into overlapping chunks.

    Parameters
    ----------
    text: str
        Input text to split.
    chunk_size: int
        Target size of each chunk.
    chunk_overlap: int
        Overlap between consecutive chunks.
    source_id: str
        Identifier for the text source (e.g., file path).

    Returns
    -------
    List[TextChunk]
        List of chunks with metadata.
    """

    assert chunk_size > 0 and chunk_overlap >= 0 and chunk_overlap < chunk_size
    chunks: list[TextChunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append(TextChunk(text=chunk_text, source_id=source_id, chunk_index=index))
        if end == len(text):
            break
        start = end - chunk_overlap
        index += 1
    return chunks
