"""Improvement-loop orchestrator (v2).

Runs repeated audit > implement > review > test-and-merge cycles, stopping
when the evaluator signals that no further improvements are warranted.

All project-specific values are read from ProjectConfig.

Pipeline phases per iteration:
    1. _phase_audit          — call audit API, parse findings
    2. _phase_implement      — call apply_fix for each finding on its own branch
    3. _phase_review         — call reviewer for each implemented finding
    4. _phase_test_and_merge — run tests, merge approved findings
    5. _phase_log            — log iteration, evaluate exit condition
"""

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

from improvement_loop import loop_tracker
from improvement_loop import git_utils
from improvement_loop.evaluator import Finding
from improvement_loop.agents._api import api_call_with_retry as _api_call_with_retry
from improvement_loop.agents.auditor import get_audit_system_prompt, collect_source_files
from improvement_loop.agents.implementer import apply_fix
from improvement_loop.agents.reviewer import review as _review
from improvement_loop.loop_config import get_config as _get_loop_config
from improvement_loop.project_config import get_project_config

# Repo root is the current working directory (the target project).
# This was previously __file__-relative, which broke when the package
# was installed via pip into site-packages.
REPO_ROOT = os.getcwd()


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FindingState:
    """Tracks a single finding through the pipeline phases."""

    finding: Finding
    diff: str = ""
    review_verdict: str = ""      # APPROVE / REQUEST_CHANGES / REJECT
    review_detail: Optional[dict] = None
    tests_passed: Optional[bool] = None
    merged: bool = False
    error: Optional[str] = None


@dataclass
class IterationState:
    """Shared state for one iteration of the loop."""

    iteration: int
    dry_run: bool
    audit_output: str = ""
    finding_states: List[FindingState] = field(default_factory=list)
    original_branch: str = ""
    all_tests_passed: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_critical_flags() -> frozenset:
    """Return the set of critical flags from ProjectConfig."""
    pcfg = get_project_config()
    if pcfg.critical_flags:
        return frozenset(pcfg.critical_flags)
    return frozenset({"LEAKAGE_RISK", "PHI_RISK"})


def _get_diff(branch: str, base: str) -> str:
    """Return the git diff between *base* and *branch*."""
    result = subprocess.run(
        ["git", "diff", f"{base}...{branch}"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout


# ---------------------------------------------------------------------------
# Phase 1: Audit
# ---------------------------------------------------------------------------

def _run_audit(iteration: int, context: str, dry_run: bool) -> str:
    """Run or simulate a code audit.  Returns raw audit text."""
    if dry_run:
        return f"[dry-run] audit output for iteration {iteration}"

    cfg = _get_loop_config()
    source_context = collect_source_files(REPO_ROOT, max_file_chars=cfg.max_file_chars)
    user_message = (
        f"## Iteration context\n{context}\n\n"
        f"## Source files\n{source_context}\n\n"
        "Return your findings as a JSON array."
    )

    return _api_call_with_retry({
        "model": cfg.audit_model,
        "max_tokens": cfg.audit_max_tokens,
        "system": get_audit_system_prompt(),
        "messages": [{"role": "user", "content": user_message}],
    })


def _parse_findings(audit_output: str, dry_run: bool) -> List[Finding]:
    """Parse or simulate findings from audit output."""
    if dry_run:
        return []

    # Strip markdown fences if present
    text = audit_output.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            text = inner.strip()
        elif text.startswith("```"):
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

    # If text still doesn't look like JSON, try to find a JSON array directly
    if not text.startswith("["):
        bracket_pos = text.find("[")
        if bracket_pos >= 0:
            text = text[bracket_pos:]

    # Truncation guard: check if the JSON array appears complete
    if not text.rstrip().endswith("]"):
        print("WARNING: Audit response appears truncated — consider increasing max_tokens")
        import re
        boundaries = [m.end() - 1 for m in re.finditer(r'\}\s*,\s*\{', text)]
        last_brace = text.rfind("}")
        candidates = sorted(set(boundaries + ([last_brace] if last_brace >= 0 else [])),
                            reverse=True)
        for pos in candidates:
            chunk = text[:pos + 1].rstrip().rstrip(",")
            if not chunk.lstrip().startswith("["):
                chunk = "[" + chunk
            chunk = chunk + "]"
            try:
                raw_list = json.loads(chunk)
                if not isinstance(raw_list, list):
                    continue
                print(f"WARNING: Recovered {len(raw_list)} findings from truncated response")
                findings = []
                for i, raw in enumerate(raw_list):
                    try:
                        finding = Finding(**raw)
                        findings.append(finding)
                    except Exception as e:
                        print(f"WARNING: Skipping finding {i}: {e}")
                return findings
            except json.JSONDecodeError:
                continue
        print("WARNING: Could not recover any findings from truncated response")
        print(f"Raw output (last 200 chars): {text[-200:]}")
        return []

    try:
        raw_list = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"WARNING: Could not parse audit findings as JSON: {e}")
        print(f"Raw output (first 500 chars): {audit_output[:500]}")
        return []

    if not isinstance(raw_list, list):
        print(f"WARNING: Expected JSON array, got {type(raw_list).__name__}")
        return []

    findings = []
    for i, raw in enumerate(raw_list):
        try:
            finding = Finding(**raw)
            findings.append(finding)
        except Exception as e:
            print(f"WARNING: Skipping finding {i}: {e}")
    return findings


def _phase_audit(state: IterationState) -> None:
    """Phase 1: audit the codebase and parse findings."""
    print(f"\n[1/5] Gathering context from prior iterations...")
    context = loop_tracker.get_context_for_next_iteration()

    print(f"[2/5] Running code audit via Claude API...")
    state.audit_output = _run_audit(state.iteration, context, state.dry_run)
    print(f"       Audit response: {len(state.audit_output)} chars")

    findings = _parse_findings(state.audit_output, state.dry_run)
    print(f"       Found {len(findings)} valid finding(s)")
    for j, f in enumerate(findings, 1):
        print(f"       {j}. [{f.dimension}] {f.description[:80]}"
              f" (importance={f.importance})")

    state.finding_states = [FindingState(finding=f) for f in findings]


# ---------------------------------------------------------------------------
# Phase 2: Implement
# ---------------------------------------------------------------------------

def _phase_implement(state: IterationState) -> None:
    """Phase 2: generate and apply a fix for each finding on its own branch."""
    if state.dry_run or not state.finding_states:
        return

    print(f"\n[3/5] Implementing fixes...")
    state.original_branch = git_utils.current_branch()

    for fs in state.finding_states:
        finding = fs.finding
        print(f"\n--- Implementing: {finding.branch_name} ---")
        print(f"    {finding.dimension}: {finding.description}")

        try:
            if git_utils.branch_exists(finding.branch_name):
                print(f"    Branch {finding.branch_name} already exists, skipping")
                finding.status = "pending"
                fs.error = "branch already exists"
                continue

            git_utils.create_branch(finding.branch_name, base=state.original_branch)
            apply_fix(finding, repo_root=REPO_ROOT)
            git_utils.commit_all(
                f"improvement: {finding.dimension} — {finding.description[:60]}"
            )
            finding.status = "implemented"

            # Capture diff for reviewer
            fs.diff = _get_diff(finding.branch_name, state.original_branch)

        except Exception as e:
            print(f"    Error implementing fix: {e}")
            finding.status = "pending"
            fs.error = str(e)
            state.all_tests_passed = False
        finally:
            try:
                git_utils.checkout(state.original_branch)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Phase 3: Review
# ---------------------------------------------------------------------------

def _phase_review(state: IterationState) -> None:
    """Phase 3: review each implemented finding via the reviewer agent.

    Verdicts:
        APPROVE          — proceed to test and merge
        REQUEST_CHANGES  — skip (leave branch for manual follow-up)
        REJECT           — delete the branch

    Critical flags from ProjectConfig force rejection.
    """
    if state.dry_run or not state.finding_states:
        return

    print(f"\n[4/5] Reviewing patches...")
    critical_flags = _get_critical_flags()

    for fs in state.finding_states:
        finding = fs.finding
        if finding.status != "implemented":
            continue

        print(f"\n--- Reviewing: {finding.branch_name} ---")

        try:
            result = _review(finding, diff=fs.diff, repo_root=REPO_ROOT)
            fs.review_detail = result
            verdict = result.get("verdict", "REQUEST_CHANGES")
            fs.review_verdict = verdict

            # Critical-flag override: if the review mentions any critical
            # flag keywords, force rejection regardless of the verdict
            issues_text = " ".join(result.get("issues", []))
            reasoning = result.get("reasoning", "")
            combined_text = (issues_text + " " + reasoning).upper()
            flagged = {f for f in critical_flags if f in combined_text}
            if flagged:
                print(f"    Critical flags detected: {flagged} — forcing REJECT")
                fs.review_verdict = "REJECT"
                verdict = "REJECT"

            print(f"    Verdict: {verdict}")
            if result.get("issues"):
                for issue in result["issues"][:3]:
                    print(f"      - {issue[:100]}")

            if verdict == "REJECT":
                print(f"    Deleting branch: {finding.branch_name}")
                finding.status = "pending"
                try:
                    git_utils.checkout(state.original_branch)
                    subprocess.run(
                        ["git", "branch", "-D", finding.branch_name],
                        check=False, capture_output=True,
                    )
                except Exception:
                    pass
            elif verdict == "REQUEST_CHANGES":
                print(f"    Skipping — branch left for manual follow-up")
                finding.status = "pending"

        except Exception as e:
            print(f"    Review error: {e} — treating as REQUEST_CHANGES")
            fs.review_verdict = "REQUEST_CHANGES"
            finding.status = "pending"


# ---------------------------------------------------------------------------
# Phase 4: Test and merge
# ---------------------------------------------------------------------------

def _phase_test_and_merge(state: IterationState) -> None:
    """Phase 4: run tests and merge approved findings."""
    if state.dry_run or not state.finding_states:
        return

    print(f"\n[5/5] Testing and merging approved patches...")

    for fs in state.finding_states:
        finding = fs.finding
        if fs.review_verdict != "APPROVE":
            continue

        print(f"\n--- Testing: {finding.branch_name} ---")

        try:
            git_utils.checkout(finding.branch_name)

            # Syntax check
            print("    Running syntax check...")
            if not git_utils.run_syntax_check():
                print("    Syntax error detected — skipping")
                finding.status = "pending"
                fs.tests_passed = False
                state.all_tests_passed = False
                continue

            # Full test suite
            print("    Running tests...")
            py_ok = git_utils.run_python_tests()
            fs.tests_passed = py_ok

            if not py_ok:
                finding.status = "pending"
                state.all_tests_passed = False
                print("    Tests failed — fix needs review")
                continue

            print("    Tests passed on branch")

            # Rebase onto latest target before merging to pick up
            # changes from previously merged findings in this iteration
            print(f"    Rebasing {finding.branch_name} onto {state.original_branch}...")
            rebase_result = subprocess.run(
                ["git", "rebase", state.original_branch],
                capture_output=True, text=True, check=False,
            )
            if rebase_result.returncode != 0:
                print(f"    Rebase conflict — aborting rebase, skipping merge")
                subprocess.run(
                    ["git", "rebase", "--abort"],
                    capture_output=True, check=False,
                )
                finding.status = "implemented"
                state.all_tests_passed = False
                continue

            # Re-run tests after rebase to ensure compatibility with
            # previously merged findings
            print("    Re-running tests after rebase...")
            post_rebase_ok = git_utils.run_python_tests()
            if not post_rebase_ok:
                print("    Tests failed after rebase — skipping merge")
                finding.status = "implemented"
                fs.tests_passed = False
                state.all_tests_passed = False
                continue

            # Merge
            print(f"    Merging: {finding.branch_name}")
            # Record the pre-merge commit so we can revert if post-merge
            # tests fail
            pre_merge_sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, check=False,
                cwd=REPO_ROOT,
            ).stdout.strip()
            try:
                git_utils.merge_branch(
                    finding.branch_name, target=state.original_branch,
                    delete_after=True,
                )
                finding.status = "merged"
                fs.merged = True
                print(f"    Merged: {finding.branch_name}")

                # Post-merge sanity check
                print(f"    Post-merge test run on {state.original_branch}")
                post_syntax_ok = git_utils.run_syntax_check()
                post_test_ok = post_syntax_ok and git_utils.run_python_tests()
                if post_test_ok:
                    print("    Post-merge tests passed")
                else:
                    reason = "syntax error" if not post_syntax_ok else "test failure"
                    print(f"    Post-merge {reason} — reverting merge")
                    subprocess.run(
                        ["git", "reset", "--hard", pre_merge_sha],
                        capture_output=True, check=False,
                        cwd=REPO_ROOT,
                    )
                    finding.status = "implemented"
                    fs.merged = False
                    state.all_tests_passed = False
            except Exception as e:
                print(f"    Merge failed: {finding.branch_name} — {e}")
                finding.status = "implemented"
                state.all_tests_passed = False

        except Exception as e:
            print(f"    Error during test/merge: {e}")
            finding.status = "pending"
            state.all_tests_passed = False
        finally:
            try:
                git_utils.checkout(state.original_branch)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Phase 5: Log
# ---------------------------------------------------------------------------

def _phase_log(state: IterationState) -> dict:
    """Phase 5: log the iteration and evaluate exit condition."""
    findings = [fs.finding for fs in state.finding_states]

    entry = loop_tracker.log_iteration(
        audit_output=state.audit_output,
        findings=findings,
        tests_passed=state.all_tests_passed,
        dry_run=state.dry_run,
    )
    return entry


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(max_iterations: int = 10, dry_run: bool = False) -> list:
    """Execute the improvement loop up to *max_iterations* times.

    Stops early when ``loop_tracker.log_iteration`` returns an entry
    whose ``exit_condition_met`` field is ``True``.

    Returns the list of log entries produced during this run.
    """
    pcfg = get_project_config()
    project_name = pcfg.name or "project"
    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"\n{'='*60}")
    print(f"  Improvement Loop [{project_name}] — {mode} (max {max_iterations} iterations)")
    print(f"{'='*60}\n")

    entries: list = []

    for i in range(1, max_iterations + 1):
        print(f"\n{'─'*60}")
        print(f"  ITERATION {i}/{max_iterations}")
        print(f"{'─'*60}")

        state = IterationState(iteration=i, dry_run=dry_run)

        # Phase 1: Audit
        _phase_audit(state)

        # Phase 2: Implement
        _phase_implement(state)

        # Phase 3: Review
        _phase_review(state)

        # Phase 4: Test and merge
        _phase_test_and_merge(state)

        # Phase 5: Log
        entry = _phase_log(state)
        entries.append(entry)

        if entry["exit_condition_met"]:
            print(f"\n  Loop complete after {i} iteration(s).")
            break

    _print_run_summary(entries)
    return entries


def _print_run_summary(entries: list) -> None:
    """Print a concise summary of this run's iterations and findings."""
    n = len(entries)
    pcfg = get_project_config()
    project_name = pcfg.name or "project"
    print(f"\n{'='*60}")
    print(f"  IMPROVEMENT LOOP SUMMARY [{project_name}] — {n} iteration(s)")
    print(f"{'='*60}")

    total_findings = 0
    total_implemented = 0
    total_pending = 0
    by_dimension: dict = {}

    for entry in entries:
        for f in entry.get("findings", []):
            total_findings += 1
            status = f.get("status", "unknown")
            dim = f.get("dimension", "unknown")
            by_dimension.setdefault(dim, {"implemented": 0, "pending": 0})
            if status in ("implemented", "merged"):
                total_implemented += 1
                by_dimension[dim]["implemented"] += 1
            else:
                total_pending += 1
                by_dimension[dim]["pending"] += 1

    for entry in entries:
        it = entry["iteration"]
        n_findings = entry["findings_count"]
        n_merged = len(entry.get("branches_merged", []))
        score = entry.get("audit_scores", {}).get("overall", "?")
        tests = "pass" if entry.get("tests_passed") else "FAIL"
        exit_flag = " [EXIT]" if entry.get("exit_condition_met") else ""
        print(f"  Iter {it}: {n_findings} findings, {n_merged} merged, "
              f"score={score}/10, tests={tests}{exit_flag}")

    print(f"\n  Findings:     {total_findings} total, "
          f"{total_implemented} implemented, {total_pending} pending")

    if by_dimension:
        print(f"\n  By dimension:")
        for dim in sorted(by_dimension):
            counts = by_dimension[dim]
            print(f"    {dim}: {counts['implemented']} implemented, "
                  f"{counts['pending']} pending")

    if entries:
        last = entries[-1]
        if last.get("exit_condition_met"):
            print(f"\n  Status: Converged — all findings below threshold")
        elif last.get("tests_passed") is False:
            print(f"\n  Status: Stopped — test failures remain")
        else:
            print(f"\n  Status: Stopped — max iterations reached")

    print(f"{'='*60}\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Improvement loop orchestrator v2")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without API calls or code changes")
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="Maximum number of audit/fix cycles (default: 10)")
    parser.add_argument("--single-iteration", action="store_true",
                        help="Run exactly one iteration (shorthand for --max-iterations 1)")
    args = parser.parse_args()

    max_iter = 1 if args.single_iteration else args.max_iterations
    run_loop(max_iterations=max_iter, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
