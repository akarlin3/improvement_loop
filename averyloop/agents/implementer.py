"""Implementer agent — generates and applies code fixes for audit findings."""

import os

from improvement_loop.agents._api import api_call_with_retry
from improvement_loop.loop_config import get_config
from improvement_loop.project_config import get_project_config


# ---------------------------------------------------------------------------
# Default fix prompt — used when ProjectConfig.fix_system_prompt is empty
# ---------------------------------------------------------------------------

DEFAULT_FIX_PROMPT = """\
You are a code fixer. Given the original file and a description of the fix, \
return ONLY the complete updated file content. No markdown fences, no commentary, \
no explanation — just the raw file content ready to be written to disk."""


def get_fix_system_prompt() -> str:
    """Return the fix system prompt from ProjectConfig, or the default."""
    pcfg = get_project_config()
    return pcfg.fix_system_prompt or DEFAULT_FIX_PROMPT


def apply_fix(finding, repo_root: str | None = None) -> None:
    """Use Claude to generate and apply a code fix for a single finding.

    Parameters
    ----------
    finding : Finding
        An evaluator Finding with at least ``file``, ``description``, and
        ``fix`` attributes.
    repo_root : str, optional
        Root of the repository.  Defaults to one level up from the package
        directory.
    """
    if repo_root is None:
        repo_root = os.getcwd()

    file_path = os.path.join(repo_root, finding.file)
    if not os.path.exists(file_path):
        print(f"    WARNING: File {finding.file} not found, skipping")
        return

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        original_content = f.read()

    cfg = get_config()
    new_content = api_call_with_retry({
        "model": get_project_config().fix_model or cfg.fix_model,
        "max_tokens": cfg.fix_max_tokens,
        "system": get_fix_system_prompt(),
        "messages": [{
            "role": "user",
            "content": (
                f"File: {finding.file}\n"
                f"Problem: {finding.description}\n"
                f"Fix: {finding.fix}\n\n"
                f"Original file content:\n{original_content}"
            ),
        }],
    })
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        f.write(new_content)
    print(f"    Updated {finding.file}")
