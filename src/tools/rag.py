"""RAG pipeline — embed and retrieve sports science literature."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI


# Default path for the local Chroma DB
CHROMA_DIR = Path(__file__).parent.parent / "data" / ".chroma"
COLLECTION_NAME = "sports_science"


def get_embedding_function() -> OpenAIEmbeddingFunction:
    """Return the OpenAI embedding function for Chroma."""
    return OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model_name="text-embedding-3-small",
    )


def get_collection(
    chroma_dir: Optional[Path] = None,
) -> chromadb.Collection:
    """Get or create the sports science Chroma collection."""
    path = str(chroma_dir or CHROMA_DIR)
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 200,
) -> list[str]:
    """Split text into overlapping chunks by character count.

    Uses a simple sliding window. For production you'd want
    sentence-aware splitting, but this works well for articles.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # Try to break at a sentence boundary
        if end < len(text):
            last_period = chunk.rfind(". ")
            if last_period > chunk_size * 0.5:
                end = start + last_period + 2
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 50]


def ingest_text(
    text: str,
    source: str,
    collection: Optional[chromadb.Collection] = None,
    chunk_size: int = 800,
    overlap: int = 200,
) -> int:
    """Chunk and embed a text document into the collection.

    Args:
        text: The full text content to ingest.
        source: A label for the source (e.g. "Schoenfeld 2017 - Volume").
        collection: Optional Chroma collection (uses default if None).
        chunk_size: Characters per chunk.
        overlap: Overlap between chunks.

    Returns:
        Number of chunks ingested.
    """
    coll = collection or get_collection()
    chunks = chunk_text(text, chunk_size, overlap)

    ids = [f"{source}::chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]

    # Upsert in batches of 50 (Chroma limit)
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        coll.upsert(
            ids=ids[i : i + batch_size],
            documents=chunks[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    return len(chunks)


def ingest_file(
    filepath: str | Path,
    source: Optional[str] = None,
    collection: Optional[chromadb.Collection] = None,
) -> int:
    """Read a .txt or .md file and ingest it.

    For PDFs, convert to text first (e.g. with pdfplumber) and use ingest_text.
    """
    path = Path(filepath)
    text = path.read_text(encoding="utf-8")
    label = source or path.stem
    return ingest_text(text, label, collection)


def ingest_directory(
    directory: str | Path,
    collection: Optional[chromadb.Collection] = None,
) -> dict[str, int]:
    """Ingest all .txt and .md files in a directory."""
    dirpath = Path(directory)
    results: dict[str, int] = {}
    for ext in ("*.txt", "*.md"):
        for filepath in dirpath.glob(ext):
            n = ingest_file(filepath, collection=collection)
            results[filepath.name] = n
    return results


def retrieve(
    query: str,
    n_results: int = 5,
    collection: Optional[chromadb.Collection] = None,
) -> list[dict]:
    """Retrieve the most relevant chunks for a query.

    Returns list of {"text": ..., "source": ..., "score": ...}
    """
    coll = collection or get_collection()
    results = coll.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "text": doc,
            "source": meta.get("source", "unknown"),
            "score": round(1 - dist, 4),  # cosine similarity
        }
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


def format_context(results: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    if not results:
        return "No relevant sports science literature found."

    sections: list[str] = []
    for i, r in enumerate(results, 1):
        sections.append(
            f"[Source {i}: {r['source']}] (relevance: {r['score']})\n{r['text']}"
        )
    return "\n\n---\n\n".join(sections)