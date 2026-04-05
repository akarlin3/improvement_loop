"""Project-specific configuration for the improvement loop.

Each project using the improvement loop provides a project_config.yaml
that controls which directories to scan, how to run tests, what prompts
to use for auditing/review/judging, and other project-specific settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ProjectConfig:
    """Project-specific configuration loaded from project_config.yaml."""

    name: str = ""
    description: str = ""
    languages: List[str] = field(default_factory=lambda: ["python"])
    default_branch: str = "main"
    branch_prefix: str = "improvement/"
    source_dirs: List[str] = field(default_factory=lambda: ["src/"])
    test_command: str = "python -m pytest tests/ -q --tb=short"
    test_ignores: List[str] = field(default_factory=list)
    read_only_dirs: List[str] = field(default_factory=list)
    skip_dirs: List[str] = field(default_factory=lambda: [".git", "__pycache__"])
    key_files: List[str] = field(default_factory=list)
    audit_system_prompt: str = ""  # loaded from prompts.audit_system
    review_system_prompt: str = ""  # loaded from prompts.review_system
    fix_system_prompt: str = ""  # loaded from prompts.fix_system
    judge_system_prompt: str = ""  # loaded from prompts.judge_system
    judge_calibration: str = ""  # loaded from prompts.judge_calibration
    risk_flags: List[str] = field(
        default_factory=lambda: ["LEAKAGE_RISK", "PHI_RISK"]
    )
    critical_flags: List[str] = field(
        default_factory=lambda: ["LEAKAGE_RISK", "PHI_RISK"]
    )
    forbidden_patterns: List[str] = field(default_factory=list)
    anthropic_api_key: str = ""  # If empty, falls back to loop config
    audit_model: str = ""  # If empty, falls back to loop config default
    fix_model: str = ""  # If empty, falls back to loop config default
    judge_model: str = ""  # If empty, falls back to loop config default
    collection_name: str = "codebase_index"
    skip_extensions: List[str] = field(
        default_factory=lambda: [".png", ".jpg", ".pdf"]
    )


_SEARCH_PATHS = [
    "project_config.yaml",
    "improvement_loop_project.yaml",
]


def load_project_config(path: Optional[str] = None) -> ProjectConfig:
    """Load project config from YAML.

    Search order:
        1. Explicit ``path`` argument
        2. ``PROJECT_CONFIG`` environment variable
        3. ``./project_config.yaml``
        4. ``./improvement_loop_project.yaml``

    If no file is found, returns a default ``ProjectConfig``.
    """
    resolved: Optional[Path] = None

    if path is not None:
        resolved = Path(path)
    elif "PROJECT_CONFIG" in os.environ:
        resolved = Path(os.environ["PROJECT_CONFIG"])
    else:
        for candidate in _SEARCH_PATHS:
            p = Path(candidate)
            if p.exists():
                resolved = p
                break

    if resolved is None or not resolved.exists():
        return ProjectConfig()

    with open(resolved, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    # Flatten nested prompts section into top-level fields
    prompts = raw.pop("prompts", {})
    prompt_mapping = {
        "audit_system": "audit_system_prompt",
        "review_system": "review_system_prompt",
        "fix_system": "fix_system_prompt",
        "judge_system": "judge_system_prompt",
        "judge_calibration": "judge_calibration",
    }
    for yaml_key, field_name in prompt_mapping.items():
        if yaml_key in prompts and field_name not in raw:
            raw[field_name] = prompts[yaml_key]

    # Only pass keys that are valid ProjectConfig fields
    valid_fields = {f.name for f in ProjectConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in valid_fields}

    return ProjectConfig(**filtered)


_cached_project: ProjectConfig | None = None


def get_project_config() -> ProjectConfig:
    """Return the cached project config, loading it on first call."""
    global _cached_project
    if _cached_project is None:
        _cached_project = load_project_config()
    return _cached_project


def reset_project_config_cache() -> None:
    """Reset the cached config. Useful for testing."""
    global _cached_project
    _cached_project = None
