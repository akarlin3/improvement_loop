"""Git utility functions for the improvement loop workflow orchestrator.

Replaces natural-language git instructions with direct subprocess calls.
"""

import logging
import pathlib
import shlex
import shutil
import subprocess
import sys
from typing import List

from improvement_loop.project_config import get_project_config

# Repository root — the current working directory (the target project).
# Previously __file__-relative, which broke when installed as a pip package.
REPO_ROOT = pathlib.Path.cwd()

logger = logging.getLogger(__name__)


def _run(args: List[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git (or other) command, optionally raising on failure."""
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command {args!r} failed (exit {result.returncode}):\n{result.stderr}"
        )
    return result


# ---------------------------------------------------------------------------
# Branch helpers
# ---------------------------------------------------------------------------

def current_branch() -> str:
    """Return the name of the current git branch."""
    result = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return result.stdout.strip()


def branch_exists(branch_name: str) -> bool:
    """Return True if the branch exists locally or on origin."""
    local = _run(
        ["git", "rev-parse", "--verify", branch_name], check=False
    )
    if local.returncode == 0:
        return True
    remote = _run(
        ["git", "rev-parse", "--verify", f"origin/{branch_name}"], check=False
    )
    return remote.returncode == 0


def _default_branch() -> str:
    """Return the project's default branch from config."""
    return get_project_config().default_branch or "main"


def create_branch(branch_name: str, base: str | None = None) -> None:
    """Create and checkout a new branch off *base*.

    If *base* is not specified, uses the project config's default_branch.
    Raises RuntimeError if the branch already exists.
    """
    if base is None:
        base = _default_branch()
    if branch_exists(branch_name):
        raise RuntimeError(f"Branch '{branch_name}' already exists.")
    _run(["git", "checkout", "-b", branch_name, base])


def checkout(branch_name: str) -> None:
    """Checkout an existing branch.

    Raises RuntimeError if the branch does not exist.
    """
    if not branch_exists(branch_name):
        raise RuntimeError(f"Branch '{branch_name}' does not exist.")
    _run(["git", "checkout", branch_name])


# Alias used by orchestrator
switch_branch = checkout


def merge_branch(
    source: str, target: str | None = None, delete_after: bool = True
) -> None:
    """Merge *source* into *target* using ``--no-ff``.

    If *target* is not specified, uses the project config's default_branch.
    Checks out *target* first, merges, then optionally deletes *source*.
    Raises RuntimeError on merge conflict.
    """
    if target is None:
        target = _default_branch()
    _run(["git", "checkout", target])
    result = _run(
        ["git", "merge", "--no-ff", source, "-m", f"Merge {source} into {target}"],
        check=False,
    )
    if result.returncode != 0:
        # Abort the failed merge so the repo is not left in a dirty state.
        _run(["git", "merge", "--abort"], check=False)
        raise RuntimeError(
            f"Merge conflict merging '{source}' into '{target}':\n{result.stderr}"
        )
    if delete_after:
        _run(["git", "branch", "-d", source])


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

def run_syntax_check() -> bool:
    """Run py_compile on all .py files in configured source_dirs.

    Falls back to scanning all .py files under REPO_ROOT if no source_dirs
    are configured.

    Returns True if all files compile, False on the first syntax error.
    """
    pcfg = get_project_config()
    source_dirs = pcfg.source_dirs if pcfg.source_dirs else [""]

    py_files = []
    for src_dir in source_dirs:
        search_root = REPO_ROOT / src_dir if src_dir else REPO_ROOT
        if search_root.is_dir():
            py_files.extend(sorted(search_root.rglob("*.py")))

    for py_file in py_files:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(py_file)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"    ❌  Syntax error in {py_file.relative_to(REPO_ROOT)}:")
            print(f"        {result.stderr.strip()}")
            return False

    print(f"    ✅  Syntax check passed ({len(py_files)} files)")
    return True


def run_python_tests(capture_output: bool = False) -> "bool | tuple[bool, str]":
    """Run the Python test suite using the configured test command.

    If *capture_output* is False (default), streams to stdout and returns bool.
    If *capture_output* is True, returns ``(passed, output_text)`` so callers
    can inspect failure details for self-healing.
    """
    pcfg = get_project_config()
    test_command = pcfg.test_command or "python -m pytest tests/ -q --tb=short"

    # Build the command argv from the config string
    parts = shlex.split(test_command)
    # Replace leading "python" with the current interpreter
    if parts and parts[0] == "python":
        parts[0] = sys.executable

    # Append --ignore flags for configured test_ignores
    for ignore_path in (pcfg.test_ignores or []):
        parts.extend(["--ignore", ignore_path])

    print(f"    [debug] Running tests from: {REPO_ROOT}")
    result = subprocess.run(
        parts,
        cwd=REPO_ROOT,
        check=False,
        capture_output=capture_output,
        text=capture_output,
    )
    print(f"    [debug] pytest exit code: {result.returncode}")
    if capture_output:
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        return result.returncode == 0, combined
    return result.returncode == 0


def run_matlab_tests() -> bool:
    """Run the MATLAB test suite, recording failing test names to a file.

    Writes ``matlab_test_failures.txt`` in the repo root with one failing
    test name per line. The file is removed if all tests pass.

    Returns True if exit code 0, False otherwise.
    If ``matlab`` is not on PATH, logs a warning and returns True
    (non-blocking for environments without MATLAB).
    """
    if shutil.which("matlab") is None:
        logger.warning("matlab not found on PATH — skipping MATLAB tests")
        return True

    failures_path = REPO_ROOT / "matlab_test_failures.txt"

    # MATLAB snippet: run tests, write failing names to file, then exit
    matlab_cmd = (
        "results = runtests('pipeline/tests'); "
        f"fid = fopen('{failures_path.as_posix()}', 'w'); "
        "for i = 1:numel(results), "
        "  if results(i).Failed, "
        "    fprintf(fid, '%s\\n', results(i).Name); "
        "  end; "
        "end; "
        "fclose(fid); "
        "if any([results.Failed]), exit(1); else exit(0); end"
    )

    result = subprocess.run(
        ["matlab", "-batch", matlab_cmd],
        cwd=REPO_ROOT,
        check=False,
    )

    passed = result.returncode == 0

    if passed and failures_path.exists():
        failures_path.unlink()
    elif failures_path.exists():
        content = failures_path.read_text(encoding="utf-8").strip()
        n = len(content.splitlines()) if content else 0
        print(f"    ❌  {n} MATLAB test failure(s) — see {failures_path}")
    else:
        if not passed:
            print("    ❌  MATLAB tests failed (no failure details captured)")

    return passed


# ---------------------------------------------------------------------------
# Branch slug / staging helpers
# ---------------------------------------------------------------------------

def sanitize_branch_slug(text: str, max_len: int = 50) -> str:
    """Convert arbitrary text to a valid git branch slug.

    * Lowercase
    * Replace spaces and non-alphanumeric characters with hyphens
    * Collapse consecutive hyphens
    * Strip leading/trailing hyphens
    * Truncate to *max_len*
    * Never returns an empty string — falls back to ``"fix"``.
    """
    import re

    slug = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.strip("-")
    slug = slug[:max_len].rstrip("-")
    return slug or "fix"


def get_staged_files() -> list[str]:
    """Return list of files currently staged for commit."""
    result = _run(["git", "diff", "--cached", "--name-only"])
    lines = result.stdout.strip()
    return lines.splitlines() if lines else []


def commit_all(message: str) -> None:
    """Stage all modified/new tracked files and commit with *message*.

    No-op if nothing to commit.
    """
    _run(["git", "add", "-A"])
    status = _run(["git", "status", "--porcelain"])
    if not status.stdout.strip():
        return  # nothing to commit
    _run(["git", "commit", "-m", message])
