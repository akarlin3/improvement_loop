"""Reviewer agent — evaluates proposed patches for correctness and quality."""

import json
from typing import Optional

from improvement_loop.agents._api import api_call_with_retry
from improvement_loop.loop_config import get_config
from improvement_loop.project_config import get_project_config


# ---------------------------------------------------------------------------
# Default review prompt — used when ProjectConfig.review_system_prompt is empty
# ---------------------------------------------------------------------------

DEFAULT_REVIEW_PROMPT = """\
You are a code reviewer. Evaluate proposed patches for:
- Correctness of the implementation
- No introduction of regressions or security issues
- Adequate test coverage for the change
- Adherence to existing project conventions
- No modifications to read-only directories

Return a JSON object with exactly these keys:
{
  "verdict": "APPROVE" | "REQUEST_CHANGES" | "REJECT",
  "summary": "<what the patch does>",
  "issues": ["<issue 1>", ...],
  "reasoning": "<1-3 sentence justification>"
}

Return ONLY the JSON object — no markdown fences, no commentary."""

VALID_VERDICTS = ("APPROVE", "REQUEST_CHANGES", "REJECT")


def get_review_system_prompt() -> str:
    """Return the review system prompt from ProjectConfig, or the default."""
    pcfg = get_project_config()
    prompt = pcfg.review_system_prompt or DEFAULT_REVIEW_PROMPT

    # Inject read-only directories warning if configured
    read_only = pcfg.read_only_dirs
    if read_only:
        read_only_str = ", ".join(f"`{d}`" for d in read_only)
        prompt += f"\n\nIMPORTANT: The following directories are READ-ONLY — " \
                  f"patches must not modify files in: {read_only_str}"

    return prompt


def _parse_review(raw: str) -> Optional[dict]:
    """Parse and validate a review JSON response."""
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
        else:
            text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    verdict = data.get("verdict", "").upper()
    if verdict not in VALID_VERDICTS:
        return None

    data["verdict"] = verdict
    return data


def review(
    finding,
    diff: str,
    repo_root: str | None = None,
) -> dict:
    """Review a patch for a single finding.

    Parameters
    ----------
    finding : Finding
        The evaluator Finding that produced the patch.
    diff : str
        The git diff of the patch to review.
    repo_root : str, optional
        Repository root (unused here but kept for interface consistency).

    Returns
    -------
    dict
        Keys: ``verdict`` (APPROVE/REQUEST_CHANGES/REJECT), ``summary``,
        ``issues``, ``reasoning``.  On parse failure, returns a dict with
        verdict ``REQUEST_CHANGES`` and an explanation.
    """
    cfg = get_config()
    user_message = (
        f"## Finding\n"
        f"Dimension: {finding.dimension}\n"
        f"File: {finding.file}\n"
        f"Description: {finding.description}\n"
        f"Proposed fix: {finding.fix}\n\n"
        f"## Diff\n```\n{diff}\n```\n\n"
        f"Review this patch and return your verdict as JSON."
    )

    raw = api_call_with_retry({
        "model": get_project_config().fix_model or cfg.fix_model,
        "max_tokens": 2000,
        "system": get_review_system_prompt(),
        "messages": [{"role": "user", "content": user_message}],
    })

    result = _parse_review(raw)
    if result is None:
        return {
            "verdict": "REQUEST_CHANGES",
            "summary": "Review parse failure",
            "issues": ["Could not parse reviewer response as valid JSON"],
            "reasoning": f"Raw response: {raw[:200]}",
        }

    return result
