"""Improvement-loop orchestrator (v2).

Runs repeated audit > fix > evaluate cycles, stopping when the evaluator
signals that no further improvements are warranted.

All project-specific values are read from ProjectConfig.
"""

import argparse
import json
import os
import sys
from typing import List

from improvement_loop import loop_tracker
from improvement_loop import git_utils
from improvement_loop.evaluator import Finding
from improvement_loop.agents._api import api_call_with_retry as _api_call_with_retry
from improvement_loop.agents.auditor import get_audit_system_prompt, collect_source_files
from improvement_loop.agents.implementer import apply_fix
from improvement_loop.loop_config import get_config as _get_loop_config
from improvement_loop.project_config import get_project_config

# Repo root is one level up from this file's directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_critical_flags() -> frozenset:
    """Return the set of critical flags from ProjectConfig."""
    pcfg = get_project_config()
    if pcfg.critical_flags:
        return frozenset(pcfg.critical_flags)
    return frozenset({"LEAKAGE_RISK", "PHI_RISK"})


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


def _apply_fixes(findings: List[Finding], dry_run: bool) -> bool:
    """Apply fixes for each finding on its own branch. Returns True if all tests pass."""
    if dry_run:
        return True

    if not findings:
        return True

    original_branch = git_utils.current_branch()
    critical_flags = _get_critical_flags()
    all_passed = True

    for finding in findings:
        print(f"\n--- Applying fix: {finding.branch_name} ---")
        print(f"    {finding.dimension}: {finding.description}")

        try:
            if git_utils.branch_exists(finding.branch_name):
                print(f"    Branch {finding.branch_name} already exists, skipping")
                finding.status = "pending"
                continue

            git_utils.create_branch(finding.branch_name, base=original_branch)

            # Use Claude to generate and apply the fix
            apply_fix(finding, repo_root=REPO_ROOT)

            # Commit changes
            git_utils.commit_all(
                f"improvement: {finding.dimension} — {finding.description[:60]}"
            )

            # Run tests
            print("    Running syntax check...")
            if not git_utils.run_syntax_check():
                print("    Syntax error detected — skipping tests")
                finding.status = "pending"
                all_passed = False
                continue

            print("    Running Python tests...")
            py_ok = git_utils.run_python_tests()

            if py_ok:
                print("    Tests passed on branch")

                print(f"    Attempting merge: {finding.branch_name}")
                try:
                    git_utils.merge_branch(
                        finding.branch_name, target=original_branch,
                        delete_after=True,
                    )
                    finding.status = "merged"
                    print(f"    Merged: {finding.branch_name}")

                    # Post-merge sanity check
                    print(f"    Post-merge test run on {original_branch}")
                    if not git_utils.run_syntax_check():
                        print(f"    Post-merge syntax error — merge may have introduced issues")
                        all_passed = False
                        continue
                    post_ok = git_utils.run_python_tests()
                    if post_ok:
                        print(f"    Post-merge tests passed")
                    else:
                        print(f"    Post-merge tests FAILED — merge may have introduced issues")
                        all_passed = False
                except Exception as e:
                    print(f"    Merge failed: {finding.branch_name} — {e}")
                    finding.status = "implemented"
                    all_passed = False
            else:
                finding.status = "pending"
                all_passed = False
                print("    Tests failed — fix needs review")

        except Exception as e:
            print(f"    Error applying fix: {e}")
            finding.status = "pending"
            all_passed = False
        finally:
            try:
                git_utils.checkout(original_branch)
            except Exception:
                pass

    return all_passed


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

        print(f"\n[1/5] Gathering context from prior iterations...")
        context = loop_tracker.get_context_for_next_iteration()

        print(f"[2/5] Running code audit via Claude API...")
        audit_output = _run_audit(i, context, dry_run)
        print(f"       Audit response: {len(audit_output)} chars")

        print(f"[3/5] Parsing findings...")
        findings = _parse_findings(audit_output, dry_run)
        print(f"       Found {len(findings)} valid finding(s)")
        for j, f in enumerate(findings, 1):
            print(f"       {j}. [{f.dimension}] {f.description[:80]}"
                  f" (importance={f.importance})")

        print(f"[4/5] Applying fixes and running tests...")
        tests_passed = _apply_fixes(findings, dry_run)

        print(f"\n[5/5] Logging iteration and evaluating exit condition...")

        entry = loop_tracker.log_iteration(
            audit_output=audit_output,
            findings=findings,
            tests_passed=tests_passed,
            dry_run=dry_run,
        )
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
