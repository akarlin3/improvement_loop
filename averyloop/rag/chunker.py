"""Chunker — splits source files into chunks for vector indexing.

Respects project configuration for which directories/extensions to skip.
Keeps MATLAB chunking support as a language feature.
"""

import os
import re
from typing import List, Tuple

from improvement_loop.project_config import get_project_config


# ---------------------------------------------------------------------------
# Default skip lists — used when ProjectConfig fields are empty
# ---------------------------------------------------------------------------

_DEFAULT_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}
_DEFAULT_BINARY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip",
                               ".tar", ".gz", ".bin", ".exe", ".dll", ".so",
                               ".pyc", ".pyo"}


def _get_skip_dirs() -> set:
    """Return directories to skip from ProjectConfig, merging with read-only dirs."""
    pcfg = get_project_config()
    skip = set(pcfg.skip_dirs) if pcfg.skip_dirs else set(_DEFAULT_SKIP_DIRS)
    # Also skip read-only dirs when chunking for the index
    if pcfg.read_only_dirs:
        skip.update(pcfg.read_only_dirs)
    return skip


def _get_binary_extensions() -> set:
    """Return file extensions to skip from ProjectConfig."""
    pcfg = get_project_config()
    if pcfg.skip_extensions:
        return set(pcfg.skip_extensions)
    return set(_DEFAULT_BINARY_EXTENSIONS)


def discover_files(root: str) -> List[str]:
    """Walk *root* and return relative paths of source files to index.

    Skips directories and extensions per ProjectConfig.
    """
    skip_dirs = _get_skip_dirs()
    skip_ext = _get_binary_extensions()
    results = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Filter out skipped directories in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in skip_dirs
            and not os.path.join(dirpath, d).replace(root, "").lstrip(os.sep).rstrip(os.sep)
            in skip_dirs
        ]

        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if ext in skip_ext:
                continue
            rel_path = os.path.relpath(os.path.join(dirpath, fname), root)
            results.append(rel_path)

    return results


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def chunk_python(content: str, file_path: str) -> List[Tuple[str, str]]:
    """Split a Python file into chunks by top-level class/function definitions.

    Returns list of (chunk_id, chunk_text) tuples.
    """
    chunks = []
    lines = content.split("\n")
    current_chunk_lines: list = []
    current_name = file_path

    for line in lines:
        # Detect top-level definitions
        match = re.match(r'^(class |def )(\w+)', line)
        if match:
            # Save previous chunk
            if current_chunk_lines:
                chunk_text = "\n".join(current_chunk_lines)
                if chunk_text.strip():
                    chunks.append((current_name, chunk_text))
            current_name = f"{file_path}::{match.group(2)}"
            current_chunk_lines = [line]
        else:
            current_chunk_lines.append(line)

    # Last chunk
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines)
        if chunk_text.strip():
            chunks.append((current_name, chunk_text))

    return chunks if chunks else [(file_path, content)]


def chunk_matlab(content: str, file_path: str) -> List[Tuple[str, str]]:
    """Split a MATLAB file into chunks by function definitions.

    MATLAB functions are delimited by ``function`` keywords.
    Returns list of (chunk_id, chunk_text) tuples.
    """
    chunks = []
    lines = content.split("\n")
    current_chunk_lines: list = []
    current_name = file_path

    for line in lines:
        match = re.match(r'^\s*function\s+.*?(\w+)\s*\(', line)
        if match:
            if current_chunk_lines:
                chunk_text = "\n".join(current_chunk_lines)
                if chunk_text.strip():
                    chunks.append((current_name, chunk_text))
            current_name = f"{file_path}::{match.group(1)}"
            current_chunk_lines = [line]
        else:
            current_chunk_lines.append(line)

    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines)
        if chunk_text.strip():
            chunks.append((current_name, chunk_text))

    return chunks if chunks else [(file_path, content)]


def chunk_file(content: str, file_path: str) -> List[Tuple[str, str]]:
    """Chunk a file based on its extension.

    Supports Python and MATLAB; other files are returned as a single chunk.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".py":
        return chunk_python(content, file_path)
    elif ext in (".m", ".mlx"):
        return chunk_matlab(content, file_path)
    else:
        # Return the whole file as one chunk
        return [(file_path, content)]
