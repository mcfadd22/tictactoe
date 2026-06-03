"""Best-effort git commit + push of sweep results, for headless VM runs.

Isolated subprocess wrapper around `git` so a long-running sweep can stream each
condition's `results/<name>/` to the current branch as it finishes. Push failures
(network/auth/rebase) are logged and swallowed so a transient blip never kills a
multi-hour run; the results stay committed locally and ride out on the next push.
"""
from __future__ import annotations

import subprocess


class GitSyncError(RuntimeError):
    """Raised when a git step that should not be swallowed fails (e.g. a commit
    failing because the operator never set user.name/user.email)."""


def _git(args, cwd=None, check=True):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=check,
        capture_output=True, text=True,
    )


def current_branch(cwd=None) -> str:
    return _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd).stdout.strip()


def commit_results(paths, message, cwd=None) -> bool:
    """Force-add `paths` (results/ is gitignored) and commit them.

    Returns True if a commit was made, False if nothing was staged (paths
    unchanged since the last commit). Raises GitSyncError if the commit itself
    fails so the cause is explicit rather than silent.
    """
    _git(["add", "-f", *paths], cwd=cwd)
    # `git diff --cached --quiet` exits 0 when nothing is staged, 1 otherwise.
    if _git(["diff", "--cached", "--quiet"], cwd=cwd, check=False).returncode == 0:
        return False
    result = _git(["commit", "-m", message], cwd=cwd, check=False)
    if result.returncode != 0:
        raise GitSyncError(
            "git commit failed (is user.name/user.email configured?):\n"
            f"{result.stderr.strip()}"
        )
    return True


def _remote_branch_exists(branch, cwd=None) -> bool:
    return _git(["ls-remote", "--exit-code", "origin", branch],
                cwd=cwd, check=False).returncode == 0


def push(branch=None, cwd=None, rebase=True) -> bool:
    """Best-effort: pull --rebase (only if the remote branch exists) then push.

    Returns True on success, False on any failure (logged, never raised). A
    failed rebase is aborted so the working tree is left clean.
    """
    if branch is None:
        branch = current_branch(cwd=cwd)
    if rebase and _remote_branch_exists(branch, cwd=cwd):
        pulled = _git(["pull", "--rebase", "origin", branch], cwd=cwd, check=False)
        if pulled.returncode != 0:
            _git(["rebase", "--abort"], cwd=cwd, check=False)
            print(f"  [gitsync] pull --rebase failed; skipping push this round:\n"
                  f"    {pulled.stderr.strip()}", flush=True)
            return False
    pushed = _git(["push", "origin", branch], cwd=cwd, check=False)
    if pushed.returncode != 0:
        print(f"  [gitsync] push failed (results committed locally, will retry "
              f"next condition):\n    {pushed.stderr.strip()}", flush=True)
        return False
    return True


def sync_results(paths, message, cwd=None, branch=None) -> None:
    """Commit the given result paths and best-effort push them. Never raises:
    a commit failure (e.g. unset identity) or a push failure is logged loudly
    but does not interrupt the sweep — results are always saved on disk."""
    try:
        made = commit_results(paths, message, cwd=cwd)
    except GitSyncError as e:
        print(f"  [gitsync] WARNING: could not commit results: {e}\n"
              f"    Results remain on disk; commit/push them manually.", flush=True)
        return
    if not made:
        print("  [gitsync] no result changes to commit", flush=True)
        return
    if push(branch=branch, cwd=cwd):
        print("  [gitsync] results pushed", flush=True)
