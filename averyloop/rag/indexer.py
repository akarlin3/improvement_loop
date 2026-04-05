"""Indexer — builds and queries a ChromaDB vector index of the codebase."""

import os
from typing import List, Optional

from improvement_loop.project_config import get_project_config
from improvement_loop.rag.chunker import chunk_file, discover_files


# ---------------------------------------------------------------------------
# Defaults — used when ProjectConfig fields are empty
# ---------------------------------------------------------------------------

_DEFAULT_COLLECTION_NAME = "codebase_index"
_DEFAULT_PROJECT_NAME = "project"


def _get_collection_name() -> str:
    """Return the ChromaDB collection name from config."""
    pcfg = get_project_config()
    return pcfg.collection_name or _DEFAULT_COLLECTION_NAME


def _get_project_name() -> str:
    """Return the project name for descriptions."""
    pcfg = get_project_config()
    return pcfg.name or _DEFAULT_PROJECT_NAME


def build_index(repo_root: str, persist_dir: str = ".chromadb") -> int:
    """Index the codebase into ChromaDB.

    Returns the number of chunks indexed.
    """
    import chromadb

    client = chromadb.PersistentClient(path=os.path.join(repo_root, persist_dir))
    collection_name = _get_collection_name()
    project_name = _get_project_name()

    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": f"Code index for {project_name}"},
    )

    files = discover_files(repo_root)
    chunk_count = 0

    for rel_path in files:
        full_path = os.path.join(repo_root, rel_path)
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError:
            continue

        chunks = chunk_file(content, rel_path)
        for chunk_id, chunk_text in chunks:
            if not chunk_text.strip():
                continue
            collection.add(
                ids=[f"{rel_path}::{chunk_count}"],
                documents=[chunk_text],
                metadatas=[{
                    "file": rel_path,
                    "chunk_id": chunk_id,
                    "project": project_name,
                }],
            )
            chunk_count += 1

    return chunk_count


def query_index(
    query: str,
    repo_root: str,
    n_results: int = 5,
    persist_dir: str = ".chromadb",
) -> List[dict]:
    """Query the codebase index.

    Returns a list of dicts with keys: file, chunk_id, text, distance.
    """
    import chromadb

    client = chromadb.PersistentClient(path=os.path.join(repo_root, persist_dir))
    collection_name = _get_collection_name()

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return []

    results = collection.query(query_texts=[query], n_results=n_results)

    hits = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else None
            hits.append({
                "file": meta.get("file", ""),
                "chunk_id": meta.get("chunk_id", ""),
                "text": doc,
                "distance": distance,
            })

    return hits
