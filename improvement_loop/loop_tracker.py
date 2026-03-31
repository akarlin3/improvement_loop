"""Loop tracker — logs iteration results and provides context for subsequent runs."""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Optional  # noqa: F401 — used by callers via import

from improvement_loop.evaluator import Finding, score_audit, should_continue_loop

LOG_FILE = os.path.join(os.getcwd(), "improvement_loop_log.json")


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_log() -> list:
    """Load existing log or return empty list."""
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        content = f.read().strip()
    if not content:
        return []
    return json.loads(content)


def save_log(log: list) -> None:
    """Save log to disk atomically."""
    tmp = LOG_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(log, f, indent=2)
    os.replace(tmp, LOG_FILE)  # atomic on all platforms


def get_current_iteration(log: list) -> int:
    """Return the next iteration number."""
    if not log:
        return 1
    return log[-1]["iteration"] + 1


# ── Core logging ─────────────────────────────────────────────────────────────

def log_iteration(
    audit_output: str,
    findings: List[Finding],
    tests_passed: bool,
    dry_run: bool = False
) -> dict:
    """
    Log one complete loop iteration.
    Derives branches_created and branches_merged from findings.
    Returns the entry so the caller can inspect it.
    """
    log = load_log()
    iteration = get_current_iteration(log)

    # Score the audit
    scores = score_audit(audit_output, dry_run=dry_run)

    # Surface evaluator failures so the loop doesn't silently proceed
    # on default fallback scores
    if not dry_run:
        if "EVALUATION_FAILED" in scores.get("flags", []):
            print("WARNING: Evaluator failed after max retries — using fallback "
                  "scores (all 5.0). Audit quality assessment is unreliable.")
        else:
            # Check for suspiciously uniform scores (sign of malfunction)
            score_vals = [scores.get(k, 0) for k in
                          ["specificity", "accuracy", "coverage",
                           "prioritization", "domain_appropriateness"]]
            if len(set(score_vals)) == 1 and score_vals[0] == 5.0:
                print("WARNING: All audit dimension scores are exactly 5.0 — "
                      "evaluator may have returned defaults without flagging.")

    # Derive branch lists from findings
    branches_created = [f.branch_name for f in findings]
    branches_merged = [f.branch_name for f in findings if f.status == "merged"]

    # Tag each finding with a unique ID and iteration number
    tagged_findings = []
    for i, finding in enumerate(findings):
        tagged_findings.append({
            "id": f"iter{iteration}_{i+1:03d}",
            "iteration": iteration,
            **finding.to_log_dict()
        })

    # Determine exit condition
    exit_condition_met = not should_continue_loop(scores, findings, dry_run=dry_run)

    # Use the same importance threshold as the evaluator's exit logic
    from improvement_loop.loop_config import get_config
    importance_threshold = get_config().importance_threshold

    entry = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "audit_scores": scores,
        "findings": tagged_findings,
        "findings_count": len(findings),
        "high_priority_findings": len(
            [f for f in findings if f.importance >= importance_threshold]
        ),
        "branches_created": branches_created,
        "branches_merged": branches_merged,
        "tests_passed": tests_passed,
        "exit_condition_met": exit_condition_met
    }

    log.append(entry)
    save_log(log)

    # Check for score drift
    check_score_drift(log)

    _print_iteration_summary(entry)
    return entry


# ── Analysis ─────────────────────────────────────────────────────────────────

def get_context_for_next_iteration() -> str:
    """
    Generate a context block to inject into Claude's next iteration prompt.
    Tells Claude what's already been done so it doesn't repeat itself.
    """
    log = load_log()
    if not log:
        return "This is the first iteration. No prior history."

    lines = [
        f"This is iteration {get_current_iteration(log)}.",
        f"Total iterations completed: {len(log)}",
        "",
        "## Already implemented improvements:"
    ]

    all_findings = []
    for entry in log:
        for finding in entry["findings"]:
            if finding.get("status") == "merged":
                all_findings.append(finding)

    if all_findings:
        for f in all_findings:
            lines.append(
                f"- [{f['dimension']}] {f['description']} "
                f"(iter {f['iteration']}, importance {f.get('importance', '?')}/10)"
            )
    else:
        lines.append("- None yet")

    lines += [
        "",
        "## Audit quality trend (overall scores):"
    ]

    scores = [e["audit_scores"]["overall"] for e in log]
    for i, score in enumerate(scores, 1):
        lines.append(f"- Iteration {i}: {score}/10")

    lines += [
        "",
        "## Instructions:",
        "Do NOT re-implement any improvement already listed above.",
        "Focus only on findings not yet addressed.",
        "Your importance scores must be consistent with prior iterations."
    ]

    return "\n".join(lines)


def check_score_drift(log: list) -> bool:
    """Warn if overall scores drift more than 3 points between iterations."""
    if len(log) < 2:
        return False
    try:
        last_two = [e["audit_scores"]["overall"] for e in log[-2:]]
    except (KeyError, TypeError):
        return False
    drift = abs(last_two[0] - last_two[1])
    if drift > 3:
        print(f"WARNING: Score drift of {drift:.1f} detected — "
              f"evaluator may be inconsistent")
        return True
    return False


def get_pending_branches() -> list:
    """Return branch names from the most recent iteration where status != 'merged'."""
    log = load_log()
    if not log:
        return []
    latest = log[-1]
    return [
        f["branch_name"]
        for f in latest["findings"]
        if f.get("status") != "merged" and "branch_name" in f
    ]


def mark_finding_merged(iteration: int, finding_id: str) -> None:
    """Update a specific finding's status to 'merged' in the log file."""
    log = load_log()
    for entry in log:
        if entry["iteration"] != iteration:
            continue
        for finding in entry["findings"]:
            if finding["id"] == finding_id:
                finding["status"] = "merged"
                # Update branches_merged to stay consistent
                entry["branches_merged"] = [
                    f["branch_name"]
                    for f in entry["findings"]
                    if f.get("status") == "merged" and "branch_name" in f
                ]
                save_log(log)
                return
    raise ValueError(
        f"Finding '{finding_id}' not found in iteration {iteration}"
    )


def get_unmerged_findings(iteration: int) -> list:
    """Return all findings from a given iteration where status is not 'merged'."""
    log = load_log()
    for entry in log:
        if entry["iteration"] == iteration:
            return [
                f for f in entry["findings"]
                if f.get("status") != "merged"
            ]
    return []


def get_all_findings_by_dimension() -> dict:
    """Return all findings grouped by dimension across all iterations."""
    log = load_log()
    by_dimension = {}
    for entry in log:
        for finding in entry["findings"]:
            dim = finding.get("dimension", "unknown")
            by_dimension.setdefault(dim, []).append(finding)
    return by_dimension


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_iteration_summary(entry: dict) -> None:
    """Print a concise summary of one iteration to stdout."""
    print(f"\n{'='*60}")
    print(f"Iteration {entry['iteration']} — {entry['timestamp']}")
    print(f"{'='*60}")
    print(f"Audit score:       {entry['audit_scores']['overall']}/10")
    print(f"Findings:          {entry['findings_count']} total, "
          f"{entry['high_priority_findings']} high priority")
    print(f"Branches created:  {len(entry['branches_created'])}")
    print(f"Branches merged:   {len(entry['branches_merged'])}")
    # Show unmerged branches if any exist
    unmerged = [
        f["branch_name"]
        for f in entry["findings"]
        if f.get("status") != "merged" and "branch_name" in f
    ]
    if unmerged:
        print(f"Unmerged branches: {len(unmerged)}")
        for branch in unmerged:
            print(f"  - {branch}")
    print(f"Tests passed:      {entry['tests_passed']}")
    print(f"Exit condition:    {'YES — stopping' if entry['exit_condition_met'] else 'NO — continuing'}")
    if entry['audit_scores']['flags']:
        print(f"Flags:             {entry['audit_scores']['flags']}")
    print(f"{'='*60}\n")


def print_full_summary() -> None:
    """Print a complete summary of all iterations."""
    log = load_log()
    if not log:
        print("No iterations logged yet.")
        return

    print(f"\n{'='*60}")
    print(f"FULL LOOP SUMMARY — {len(log)} iterations")
    print(f"{'='*60}")

    total_findings = sum(e["findings_count"] for e in log)
    total_merged = sum(len(e["branches_merged"]) for e in log)
    avg_score = sum(e["audit_scores"]["overall"] for e in log) / len(log)

    print(f"Total findings:    {total_findings}")
    print(f"Total merges:      {total_merged}")
    print(f"Avg audit score:   {avg_score:.1f}/10")
    print(f"Final status:      "
          f"{'Complete' if log[-1]['exit_condition_met'] else 'In progress'}")
    print(f"\nIteration breakdown:")

    for entry in log:
        status = "DONE" if entry["exit_condition_met"] else ">"
        print(f"  {status} Iter {entry['iteration']}: "
              f"score={entry['audit_scores']['overall']}/10, "
              f"findings={entry['findings_count']}, "
              f"merged={len(entry['branches_merged'])}")

    print(f"\nFindings by dimension:")
    by_dim = get_all_findings_by_dimension()
    for dim, findings in sorted(by_dim.items()):
        print(f"  {dim}: {len(findings)} findings")

    print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loop tracker CLI")
    subparsers = parser.add_subparsers(dest="command")

    # context
    subparsers.add_parser("context", help="Print context for next iteration")

    # log
    log_parser = subparsers.add_parser("log", help="Log a completed iteration")
    log_parser.add_argument("--audit", required=True)
    log_parser.add_argument("--findings", required=True)
    log_parser.add_argument("--tests-passed", required=True, dest="tests_passed")

    # summary
    subparsers.add_parser("summary", help="Print full loop summary")

    args = parser.parse_args()

    if args.command == "context":
        print(get_context_for_next_iteration())

    elif args.command == "log":
        findings = json.loads(args.findings)
        tests_passed = args.tests_passed.lower() == "true"

        entry = log_iteration(
            audit_output=args.audit,
            findings=findings,
            tests_passed=tests_passed
        )

        if entry["exit_condition_met"]:
            print("Exit condition met — stopping")
            sys.exit(0)
        else:
            print("Continuing loop")
            sys.exit(2)

    elif args.command == "summary":
        print_full_summary()

    else:
        parser.print_help()
