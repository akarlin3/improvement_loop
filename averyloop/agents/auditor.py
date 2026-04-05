"""Auditor agent — runs LLM-powered code audits against the codebase."""

import os
from typing import List

from improvement_loop.project_config import get_project_config


# ---------------------------------------------------------------------------
# Default audit prompt — used when ProjectConfig.audit_system_prompt is empty
# ---------------------------------------------------------------------------

DEFAULT_AUDIT_PROMPT = """\
You are a senior code auditor. Audit the codebase and return a JSON array of findings.

Each finding must have:
{
  "dimension": one of ("performance", "correctness", "error_handling", "modularity", \
"memory", "code_quality", "test_coverage", "security", "cross_platform"),
  "file": "path/to/file",
  "function_name": "optional — specific function if applicable",
  "description": "what the problem is — cite specific lines",
  "fix": "concise fix description (1-3 sentences, NO full code blocks)",
  "importance": integer 1-10 (8-10 = critical bugs/data integrity, 4-7 = moderate, 1-3 = minor),
  "branch_name": "<prefix><slug>" (slug: lowercase, hyphens, no spaces, max 50 chars after prefix)
}

## Rules
- Be specific: cite files, functions, and line numbers.
- Do NOT suggest changes to read-only directories.
- Do NOT inflate importance scores — style nits are 1-2, not 7-8.
- Cover multiple dimensions if possible (performance, correctness, security, tests, etc.).
- Return ONLY the JSON array — no markdown fences, no commentary.
"""


def get_audit_system_prompt() -> str:
    """Return the audit system prompt from ProjectConfig, or the default."""
    pcfg = get_project_config()
    prompt = pcfg.audit_system_prompt or DEFAULT_AUDIT_PROMPT

    # Inject read-only directories warning if configured
    read_only = pcfg.read_only_dirs
    if read_only:
        read_only_str = ", ".join(f"`{d}`" for d in read_only)
        prompt += f"\n\nIMPORTANT: The following directories are READ-ONLY — " \
                  f"never suggest modifications there: {read_only_str}"

    return prompt


def collect_source_files(repo_root: str, max_file_chars: int = 8000) -> str:
    """Collect key source files as context for the audit.

    Uses ProjectConfig.key_files if configured, otherwise scans source_dirs.
    """
    pcfg = get_project_config()

    key_files: List[str] = pcfg.key_files if pcfg.key_files else []

    # If no key_files configured, discover files from source_dirs
    if not key_files:
        skip_ext = set(pcfg.skip_extensions or [".png", ".jpg", ".pdf"])
        for src_dir in (pcfg.source_dirs or ["src/"]):
            src_path = os.path.join(repo_root, src_dir)
            if not os.path.isdir(src_path):
                continue
            for dirpath, dirnames, filenames in os.walk(src_path):
                # Filter skip_dirs
                dirnames[:] = [
                    d for d in dirnames
                    if d not in (pcfg.skip_dirs or [".git", "__pycache__"])
                ]
                for fname in sorted(filenames):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in skip_ext:
                        rel = os.path.relpath(
                            os.path.join(dirpath, fname), repo_root
                        )
                        key_files.append(rel)

    parts = []
    for rel_path in key_files:
        full_path = os.path.join(repo_root, rel_path)
        if not os.path.exists(full_path):
            continue
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if len(content) > max_file_chars:
                content = content[:max_file_chars] + "\n... [truncated]"
            parts.append(f"=== {rel_path} ===\n{content}")
        except OSError:
            continue
    return "\n\n".join(parts)
