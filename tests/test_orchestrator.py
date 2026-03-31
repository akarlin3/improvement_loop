"""Tests for orchestrator_v2.run_loop exit-condition logic."""

import subprocess
import pytest
from unittest.mock import patch

from improvement_loop import loop_tracker
from improvement_loop import orchestrator_v2


@pytest.fixture(autouse=True)
def _isolate_log(tmp_path, monkeypatch):
    """Point loop_tracker at a temp log file for every test."""
    log_file = str(tmp_path / "test_log.json")
    monkeypatch.setattr(loop_tracker, "LOG_FILE", log_file)


# ── Exit on first iteration ─────────────────────────────────────────────────

def test_run_loop_stops_on_first_exit_condition():
    """If log_iteration returns exit_condition_met=True on the first call,
    the loop must execute exactly one iteration."""
    exit_entry = {
        "iteration": 1,
        "exit_condition_met": True,
        "audit_scores": {"overall": 9.0, "flags": []},
        "findings": [],
        "findings_count": 0,
        "high_priority_findings": 0,
        "branches_created": [],
        "branches_merged": [],
        "tests_passed": True,
    }

    with patch.object(
        loop_tracker, "log_iteration", return_value=exit_entry
    ) as mock_log:
        entries = orchestrator_v2.run_loop(max_iterations=10, dry_run=True)

    assert mock_log.call_count == 1
    assert len(entries) == 1
    assert entries[0]["exit_condition_met"] is True


# ── Exit on third iteration ──────────────────────────────────────────────────

def test_run_loop_stops_after_third_iteration():
    """If log_iteration returns exit_condition_met=False twice then True,
    the loop must execute exactly three iterations."""
    continuing_entry = {
        "iteration": 1,
        "exit_condition_met": False,
        "audit_scores": {"overall": 5.0, "flags": []},
        "findings": [],
        "findings_count": 0,
        "high_priority_findings": 0,
        "branches_created": [],
        "branches_merged": [],
        "tests_passed": True,
    }
    exit_entry = {
        **continuing_entry,
        "iteration": 3,
        "exit_condition_met": True,
        "audit_scores": {"overall": 9.0, "flags": []},
    }

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return {**continuing_entry, "iteration": call_count}
        return exit_entry

    with patch.object(
        loop_tracker, "log_iteration", side_effect=side_effect
    ) as mock_log:
        entries = orchestrator_v2.run_loop(max_iterations=10, dry_run=True)

    assert mock_log.call_count == 3
    assert len(entries) == 3
    assert entries[-1]["exit_condition_met"] is True
    assert entries[0]["exit_condition_met"] is False


# ── Phase 4: test_and_merge revert on post-merge failure ─────────────────

class TestPhaseTestAndMerge:
    """Tests for _phase_test_and_merge rebase and revert logic."""

    def _make_finding_state(self, branch="improvement/test-branch"):
        from improvement_loop.evaluator import Finding
        finding = Finding(
            dimension="correctness",
            file="src/module.py",
            description="test fix",
            fix="apply fix",
            importance=5,
            branch_name=branch,
            status="implemented",
        )
        fs = orchestrator_v2.FindingState(finding=finding)
        fs.review_verdict = "APPROVE"
        return fs

    @patch.object(orchestrator_v2.git_utils, "checkout")
    @patch.object(orchestrator_v2.git_utils, "run_syntax_check", return_value=True)
    @patch.object(orchestrator_v2.git_utils, "run_python_tests", return_value=True)
    @patch.object(orchestrator_v2.git_utils, "merge_branch")
    @patch("improvement_loop.orchestrator_v2.subprocess")
    def test_post_merge_failure_reverts(
        self, mock_subprocess, mock_merge, mock_tests, mock_syntax, mock_checkout
    ):
        """When post-merge tests fail, the merge should be reverted."""
        # run_python_tests is called 3 times:
        # 1. pre-merge on branch (pass)
        # 2. after rebase (pass)
        # 3. post-merge (fail)
        mock_tests.side_effect = [True, True, False]

        # Simulate subprocess calls: rebase, rev-parse, reset
        mock_subprocess.run.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),   # rebase
            subprocess.CompletedProcess(args=[], returncode=0, stdout="abc123\n", stderr=""),  # rev-parse
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),   # reset --hard
        ]

        fs = self._make_finding_state()
        state = orchestrator_v2.IterationState(iteration=1, dry_run=False)
        state.original_branch = "main"
        state.finding_states = [fs]

        orchestrator_v2._phase_test_and_merge(state)

        # The finding should NOT be merged because post-merge tests failed
        assert fs.finding.status == "implemented"
        assert fs.merged is False
        assert state.all_tests_passed is False

    @patch.object(orchestrator_v2.git_utils, "checkout")
    @patch.object(orchestrator_v2.git_utils, "run_syntax_check", return_value=True)
    @patch.object(orchestrator_v2.git_utils, "run_python_tests", return_value=True)
    @patch.object(orchestrator_v2.git_utils, "merge_branch")
    @patch("improvement_loop.orchestrator_v2.subprocess")
    def test_rebase_conflict_skips_merge(
        self, mock_subprocess, mock_merge, mock_tests, mock_syntax, mock_checkout
    ):
        """When rebase has conflicts, the finding should be skipped."""
        mock_subprocess.run.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="conflict"),  # rebase fail
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),           # rebase --abort
        ]

        fs = self._make_finding_state()
        state = orchestrator_v2.IterationState(iteration=1, dry_run=False)
        state.original_branch = "main"
        state.finding_states = [fs]

        orchestrator_v2._phase_test_and_merge(state)

        assert fs.finding.status == "implemented"
        assert fs.merged is False
        # merge_branch should never be called when rebase fails
        mock_merge.assert_not_called()
