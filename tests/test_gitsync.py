import subprocess

import pytest

from ttt.gitsync import (
    current_branch, commit_results, push, sync_results, GitSyncError,
)


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), check=True,
                          capture_output=True, text=True)


def _init_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-b", "main"], path)
    _git(["config", "user.email", "test@example.com"], path)
    _git(["config", "user.name", "Test"], path)
    (path / "README").write_text("x")
    _git(["add", "README"], path)
    _git(["commit", "-m", "init"], path)


@pytest.fixture
def repo_with_remote(tmp_path):
    work = tmp_path / "work"
    remote = tmp_path / "remote.git"
    _init_repo(work)
    remote.mkdir()
    _git(["init", "--bare", "-b", "main"], remote)
    _git(["remote", "add", "origin", str(remote)], work)
    return work, remote


def _write_result(work, name="E0"):
    d = work / "results" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "agg.json").write_text("{}")


def test_current_branch_reports_checked_out_branch(repo_with_remote):
    work, _ = repo_with_remote
    assert current_branch(cwd=str(work)) == "main"


def test_commit_results_commits_then_noop_when_unchanged(repo_with_remote):
    work, _ = repo_with_remote
    _write_result(work)
    assert commit_results(["results/E0"], "results: E0", cwd=str(work)) is True
    # Re-running with no changes must be a clean no-op (no empty-commit error).
    assert commit_results(["results/E0"], "results: E0", cwd=str(work)) is False


def test_sync_results_pushes_to_an_empty_remote(repo_with_remote):
    # Remote has no 'main' yet -> push() must skip the rebase and still create it.
    work, remote = repo_with_remote
    _write_result(work)
    sync_results(["results/E0"], "results: E0 sync", cwd=str(work))
    log = subprocess.run(["git", "--git-dir", str(remote), "log", "--oneline"],
                         capture_output=True, text=True)
    assert "results: E0 sync" in log.stdout


def test_push_returns_false_on_bad_remote_without_raising(tmp_path):
    work = tmp_path / "work"
    _init_repo(work)
    _git(["remote", "add", "origin", str(tmp_path / "nope.git")], work)
    assert push(cwd=str(work)) is False  # logged, not raised


def test_sync_results_does_not_raise_on_push_failure(tmp_path):
    # Commit succeeds, push fails (bad remote) -> sync_results must swallow it.
    work = tmp_path / "work"
    _init_repo(work)
    _git(["remote", "add", "origin", str(tmp_path / "nope.git")], work)
    _write_result(work)
    sync_results(["results/E0"], "results: E0", cwd=str(work))  # must not raise
    # The commit still landed locally.
    log = subprocess.run(["git", "-C", str(work), "log", "--oneline"],
                         capture_output=True, text=True)
    assert "results: E0" in log.stdout


def test_commit_results_raises_on_commit_failure(repo_with_remote):
    # A failing pre-commit hook forces `git commit` to exit non-zero, which must
    # surface as GitSyncError (e.g. the missing-identity case in production).
    work, _ = repo_with_remote
    hook = work / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    _write_result(work)
    with pytest.raises(GitSyncError):
        commit_results(["results/E0"], "results: E0", cwd=str(work))
