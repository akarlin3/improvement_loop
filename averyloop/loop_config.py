"""Configuration for the improvement loop.

Loads from ``improvement_loop_config.json`` at the repo root.  Every field
has a built-in default so the file is optional — missing keys are filled in
automatically (same pattern as ``parse_config.m`` for the MATLAB pipeline).
"""

import json
import os
from dataclasses import dataclass, fields
from typing import Literal

CONFIG_PATH = os.path.join(os.getcwd(), "improvement_loop_config.json")


@dataclass
class LoopConfig:
    """All tuneable knobs for the improvement loop."""

    # ── Exit strategy ────────────────────────────────────────────────────
    # "classic"             — original threshold-only logic
    # "diminishing_returns" — only the 4-condition staleness detector
    # "both"                — classic first, then diminishing returns
    exit_strategy: Literal["classic", "diminishing_returns", "both"] = "both"

    # ── Classic exit thresholds ──────────────────────────────────────────
    importance_threshold: int = 2        # findings >= this keep the loop going
    min_coverage_score: float = 6.0      # coverage below this keeps the loop going

    # ── Diminishing returns thresholds ───────────────────────────────────
    dr_window: int = 4                   # how many recent iterations to examine
    dr_max_merge_rate: float = 0.15      # per-iteration merge rate ceiling
    dr_max_avg_importance: float = 3.5   # avg importance ceiling across window
    dr_min_file_repeats: int = 3         # same file must appear in >= N iterations
    dr_max_audit_score: float = 8.5      # no iteration may exceed this score

    # ── API ──────────────────────────────────────────────────────────────
    # If empty, falls back to ANTHROPIC_API_KEY env var (Anthropic SDK default).
    anthropic_api_key: str = ""
    audit_model: str = "claude-opus-4-6"
    fix_model: str = "claude-opus-4-6"
    judge_model: str = "claude-opus-4-6"

    # ── Token limits ────────────────────────────────────────────────────
    audit_max_tokens: int = 32000
    fix_max_tokens: int = 8192
    judge_max_tokens: int = 2000

    # ── Orchestrator knobs ───────────────────────────────────────────────
    max_api_retries: int = 3
    retry_base_delay: float = 30.0
    max_self_heal_attempts: int = 2
    max_file_chars: int = 8000


def load_loop_config(path: str | None = None) -> LoopConfig:
    """Load config from JSON, falling back to defaults for missing keys."""
    cfg_path = path or CONFIG_PATH
    overrides: dict = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            overrides = json.load(f)

    valid_keys = {fld.name for fld in fields(LoopConfig)}
    filtered = {k: v for k, v in overrides.items() if k in valid_keys}
    return LoopConfig(**filtered)


# Module-level singleton — importers get a shared instance.
# Re-call load_loop_config() to refresh from disk if needed.
_cached: LoopConfig | None = None


def get_config(path: str | None = None) -> LoopConfig:
    """Return a cached LoopConfig, loading from disk on first call."""
    global _cached
    if _cached is None:
        _cached = load_loop_config(path)
    return _cached


def reset_config() -> None:
    """Clear the cached config so the next get_config() reloads from disk."""
    global _cached
    _cached = None
