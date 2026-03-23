"""Retriever — queries the ChromaDB vector index for relevant code context."""

import os
from typing import List, Optional

from improvement_loop.project_config import get_project_config


# ---------------------------------------------------------------------------
# Defaults — used when ProjectConfig fields are empty
# ---------------------------------------------------------------------------

_DEFAULT_COLLECTION_NAME = "codebase_index"


def _get_collection_name() -> str:
    """Return the ChromaDB collection name from config."""
    pcfg = get_project_config()
    return pcfg.collection_name or _DEFAULT_COLLECTION_NAME


def retrieve(
    query: str,
    repo_root: str,
    n_results: int = 5,
    persist_dir: str = ".chromadb",
    min_relevance: Optional[float] = None,
) -> List[dict]:
    """Retrieve code chunks relevant to *query* from the vector index.

    Parameters
    ----------
    query : str
        Natural-language description of what to search for.
    repo_root : str
        Root of the repository (where ``.chromadb/`` lives).
    n_results : int
        Maximum number of results to return.
    persist_dir : str
        Subdirectory under *repo_root* for the ChromaDB store.
    min_relevance : float, optional
        If set, discard results whose distance exceeds this threshold
        (lower distance = more relevant in ChromaDB's default metric).

    Returns
    -------
    list[dict]
        Each dict has keys: ``file``, ``chunk_id``, ``text``, ``distance``.
    """
    import chromadb

    client = chromadb.PersistentClient(
        path=os.path.join(repo_root, persist_dir)
    )
    collection_name = _get_collection_name()

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return []

    results = collection.query(query_texts=[query], n_results=n_results)

    hits: List[dict] = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = (
                results["distances"][0][i] if results["distances"] else None
            )
            if min_relevance is not None and distance is not None:
                if distance > min_relevance:
                    continue
            hits.append({
                "file": meta.get("file", ""),
                "chunk_id": meta.get("chunk_id", ""),
                "text": doc,
                "distance": distance,
            })

    return hits


def retrieve_context(
    query: str,
    repo_root: str,
    n_results: int = 5,
    persist_dir: str = ".chromadb",
) -> str:
    """Retrieve relevant code and format it as a single context string.

    Convenience wrapper around :func:`retrieve` that returns a
    ready-to-inject prompt fragment.
    """
    hits = retrieve(query, repo_root, n_results=n_results, persist_dir=persist_dir)
    if not hits:
        return ""

    parts: List[str] = []
    for hit in hits:
        header = hit["file"]
        if hit["chunk_id"]:
            header += f" :: {hit['chunk_id']}"
        parts.append(f"=== {header} ===\n{hit['text']}")

    return "\n\n".join(parts)
