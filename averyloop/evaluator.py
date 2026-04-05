"""Evaluator — judge scoring and exit-condition logic for the improvement loop."""

import anthropic  # type: ignore
import json
import os
import re
import sys
import time
from typing import List, Literal, Optional

from pydantic import BaseModel, field_validator

from improvement_loop.project_config import get_project_config


VALID_DIMENSIONS = (
    "performance", "correctness", "error_handling", "modularity",
    "memory", "code_quality", "test_coverage", "security", "cross_platform",
)

VALID_STATUSES = ("pending", "implemented", "merged")

# Characters forbidden in git branch names (see git-check-ref-format)
_GIT_BRANCH_INVALID_RE = re.compile(
    r"[\x00-\x1f\x7f ~^:?*\[\\]"  # control chars + special chars
    r"|\.\.+"                        # consecutive dots
    r"|@\{"                          # @{ sequence
    r"|\.$"                          # trailing dot
    r"|\.lock$"                      # .lock suffix
)


class Finding(BaseModel):
    """Typed schema for a single codebase improvement finding."""

    dimension: Literal[
        "performance", "correctness", "error_handling", "modularity",
        "memory", "code_quality", "test_coverage", "security", "cross_platform",
    ]
    file: str
    function_name: Optional[str] = None
    description: str
    fix: str
    importance: int
    branch_name: str
    status: Optional[Literal["pending", "implemented", "merged"]] = None

    @field_validator("importance")
    @classmethod
    def _importance_range(cls, v: int) -> int:
        if not (1 <= v <= 10):
            raise ValueError("importance must be between 1 and 10 inclusive")
        return v

    @field_validator("branch_name")
    @classmethod
    def _valid_branch_name(cls, v: str) -> str:
        prefix = get_project_config().branch_prefix or "improvement/"
        if not v.startswith(prefix):
            raise ValueError(f"branch_name must start with '{prefix}'")
        slug = v[len(prefix):]
        if not slug:
            raise ValueError("branch_name slug must not be empty")
        if len(slug) > 50:
            raise ValueError("branch_name slug must be at most 50 characters")
        if " " in v:
            raise ValueError("branch_name must not contain spaces")
        # No extra slashes beyond the leading prefix
        if "/" in slug:
            raise ValueError(
                f"branch_name must not contain slashes after '{prefix}'"
            )
        if _GIT_BRANCH_INVALID_RE.search(v):
            raise ValueError(
                "branch_name contains characters invalid in git branch names"
            )
        return v

    def to_log_dict(self) -> dict:
        """Serialize to the dict format used by loop_tracker's JSON log."""
        d: dict = {
            "dimension": self.dimension,
            "file": self.file,
            "description": self.description,
            "fix": self.fix,
            "importance": self.importance,
            "branch_name": self.branch_name,
        }
        if self.function_name is not None:
            d["function_name"] = self.function_name
        if self.status is not None:
            d["status"] = self.status
        return d


def _get_client() -> anthropic.Anthropic:
    """Return an Anthropic client, using the config API key if set.

    Resolution order: project_config.yaml → loop config JSON → ANTHROPIC_API_KEY env var.
    """
    from improvement_loop.loop_config import get_config
    from improvement_loop.project_config import get_project_config
    project_cfg = get_project_config()
    loop_cfg = get_config()
    api_key = project_cfg.anthropic_api_key or loop_cfg.anthropic_api_key
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    return anthropic.Anthropic(**kwargs)


# ---------------------------------------------------------------------------
# Default prompts — used when ProjectConfig fields are empty
# ---------------------------------------------------------------------------

DEFAULT_JUDGE_PROMPT = """\
You are an expert code-audit judge. Your job is to evaluate the quality \
of a codebase improvement audit and return a structured JSON score.

## Scoring dimensions (each 0-10)

**specificity** — Are findings concrete and actionable? Do they cite specific files, \
functions, and line numbers?

**accuracy** — Are the findings technically correct? Would implementing them actually \
improve the code?

**coverage** — Does the audit examine all relevant dimensions: performance, correctness, \
error handling, modularity, memory usage, code quality, test coverage, security, and \
cross-platform compatibility?

**prioritization** — Are importance scores well-calibrated? Critical bugs and data-integrity \
issues should score 8-10; minor style nits should score 1-2.

**domain_appropriateness** — Do the findings respect the project domain?

**overall** — Holistic quality of the audit as a guide for the next improvement iteration. \
Weight accuracy and domain_appropriateness most heavily.

## Flags
Return a list of string flags for any of these conditions:
- "LEAKAGE_RISK" — a suggested change could introduce data leakage
- "PHI_RISK" — a suggestion could expose patient data
- "DEPS_MODIFIED" — audit suggests modifying read-only directories
- "INFLATED_SCORES" — importance scores are systematically too high
- "DEFLATED_SCORES" — importance scores are systematically too low
- "MISSING_DIMENSION" — an entire audit dimension was skipped
- "INCORRECT_FINDING" — one or more findings are technically wrong

Return an empty list if no flags apply.

## Output format
Return ONLY a JSON object with exactly these keys:
{
  "specificity": <float 0-10>,
  "accuracy": <float 0-10>,
  "coverage": <float 0-10>,
  "prioritization": <float 0-10>,
  "domain_appropriateness": <float 0-10>,
  "overall": <float 0-10>,
  "flags": [<string>, ...],
  "reasoning": "<1-3 sentence justification for the overall score>"
}
"""

DEFAULT_CALIBRATION = """\
## Calibration examples

### Example 1 — High-quality audit (expected overall ~8.5)
Specific file:line citations, correct semantics, well-calibrated importance.

### Example 2 — Medium-quality audit (expected overall ~5)
Vague findings with no file references; missing dimensions.

### Example 3 — Poor audit with risky suggestions (expected overall ~2)
Suggests removing safety guards, writing to read-only dirs, inflated scores.
"""


def _build_judge_prompt() -> str:
    """Build the full judge system prompt from ProjectConfig or defaults."""
    pcfg = get_project_config()
    judge_prompt = pcfg.judge_system_prompt or DEFAULT_JUDGE_PROMPT
    calibration = pcfg.judge_calibration or DEFAULT_CALIBRATION
    return judge_prompt + "\n\n" + calibration


REQUIRED_KEYS = {
    "specificity", "accuracy", "coverage",
    "prioritization", "domain_appropriateness",
    "overall", "flags", "reasoning"
}

SCORE_KEYS = {
    "specificity", "accuracy", "coverage",
    "prioritization", "domain_appropriateness", "overall"
}

MAX_RETRIES = 3
RETRY_DELAY = 2.0


def parse_and_validate(raw_text: str) -> Optional[dict]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON parse failed: {e}")
        return None

    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        print(f"Missing keys: {missing}")
        return None

    for key in SCORE_KEYS:
        val = data[key]
        if not isinstance(val, (int, float)) or not (0 <= val <= 10):
            print(f"Invalid score for '{key}': {val}")
            return None

    return data


def score_audit(audit_output: str, dry_run: bool = False) -> dict:
    if dry_run:
        return {
            "specificity": 5.0, "accuracy": 5.0, "coverage": 5.0,
            "prioritization": 5.0, "domain_appropriateness": 5.0,
            "overall": 5.0,
            "flags": [],
            "reasoning": "Dry-run mode — skipped API evaluation."
        }
    from improvement_loop.loop_config import get_config
    cfg = get_config()
    client = _get_client()
    full_prompt = _build_judge_prompt()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=get_project_config().judge_model or cfg.judge_model,
                max_tokens=cfg.judge_max_tokens,
                system=full_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"Score this audit:\n\n{audit_output}"
                    }
                ]
            )
            result = parse_and_validate(response.content[0].text)
            if result is not None:
                return result
        except anthropic.APIError as e:
            print(f"Attempt {attempt} API error: {e}")

        time.sleep(RETRY_DELAY * attempt)

    return {
        "specificity": 5.0, "accuracy": 5.0, "coverage": 5.0,
        "prioritization": 5.0, "domain_appropriateness": 5.0,
        "overall": 5.0,
        "flags": ["EVALUATION_FAILED"],
        "reasoning": "Fallback scores — evaluator failed after max retries."
    }


def check_diminishing_returns(log: list, cfg=None) -> bool:
    """Return True (stop the loop) if diminishing returns detected.

    All four conditions must be met simultaneously over the last ``cfg.dr_window``
    iterations (default 4):
    1. Merge rate < ``dr_max_merge_rate`` in every iteration
    2. Average importance across the window < ``dr_max_avg_importance``
    3. Same file appears in >= ``dr_min_file_repeats`` iterations
    4. No iteration has an audit score above ``dr_max_audit_score``

    Parameters
    ----------
    cfg : LoopConfig, optional
        If *None*, the module-level cached config is used.
    """
    from improvement_loop.loop_config import get_config
    if cfg is None:
        cfg = get_config()

    window = cfg.dr_window
    if len(log) < window:
        return False

    recent = log[-window:]

    # Condition 1: merge rate below threshold in every iteration
    for entry in recent:
        created = len(entry.get("branches_created", []))
        merged = len(entry.get("branches_merged", []))
        if created > 0 and (merged / created) >= cfg.dr_max_merge_rate:
            return False

    # Condition 2: average importance below threshold
    all_importances = []
    for entry in recent:
        for f in entry.get("findings", []):
            imp = f.get("importance")
            if imp is not None:
                all_importances.append(imp)
    if not all_importances or (sum(all_importances) / len(all_importances)) >= cfg.dr_max_avg_importance:
        return False

    # Condition 3: same file in at least dr_min_file_repeats iterations
    from collections import Counter
    file_iteration_counts: Counter = Counter()
    for entry in recent:
        files_this_iter = set()
        for f in entry.get("findings", []):
            file_val = f.get("file")
            if file_val:
                files_this_iter.add(file_val)
        for file_val in files_this_iter:
            file_iteration_counts[file_val] += 1
    if not any(c >= cfg.dr_min_file_repeats for c in file_iteration_counts.values()):
        return False

    # Condition 4: no audit score above threshold
    for entry in recent:
        score = entry.get("audit_scores", {}).get("overall", 0)
        if score > cfg.dr_max_audit_score:
            return False

    return True


def should_continue_loop(scores: dict, findings: List[Finding], dry_run: bool = False) -> bool:
    """
    Returns True if the loop should continue, False if it should exit.
    Combines test results with judge scores.

    The ``exit_strategy`` config field controls which exit logic runs:
    - ``"classic"`` — original threshold-only checks
    - ``"diminishing_returns"`` — only the staleness detector
    - ``"both"`` (default) — classic first, then diminishing returns
    """
    from improvement_loop.loop_config import get_config
    cfg = get_config()

    if dry_run:
        print("Exit condition met — dry-run mode")
        return False

    # Evaluator failure is not a valid exit condition — retry next iteration
    if "EVALUATION_FAILED" in scores.get("flags", []):
        print("Continuing — evaluator failed, cannot trust scores as exit signal")
        return True

    use_classic = cfg.exit_strategy in ("classic", "both")
    use_dr = cfg.exit_strategy in ("diminishing_returns", "both")

    # ── Classic exit checks ──────────────────────────────────────────────
    if use_classic:
        high_priority = [f for f in findings if f.importance >= cfg.importance_threshold]
        if high_priority:
            print(f"Continuing — {len(high_priority)} findings at importance >= {cfg.importance_threshold}")
            return True

        if scores["coverage"] < cfg.min_coverage_score:
            print(f"Continuing — audit coverage score {scores['coverage']}/10 is too low")
            return True

        # Check for critical flags from project config
        pcfg = get_project_config()
        critical_flags = frozenset(pcfg.critical_flags) if pcfg.critical_flags else frozenset()
        active_flags = set(scores.get("flags", []))
        if active_flags and "EVALUATION_FAILED" not in active_flags:
            # If critical flags are configured, only continue on those
            if critical_flags:
                if active_flags & critical_flags:
                    print(f"Continuing — judge flagged critical issues: {active_flags & critical_flags}")
                    return True
            else:
                # No critical flags configured — any flag continues
                print(f"Continuing — judge flagged issues: {scores['flags']}")
                return True

    # ── Diminishing returns check ────────────────────────────────────────
    if use_dr:
        from improvement_loop.loop_tracker import load_log
        if check_diminishing_returns(load_log(), cfg=cfg):
            print(f"⚠️  Diminishing returns detected over last {cfg.dr_window} iterations — stopping loop")
            return False

    if use_classic:
        print("Exit condition met — no findings above threshold and audit quality sufficient")
    else:
        print("Exit condition met — diminishing returns not triggered")
    return False
