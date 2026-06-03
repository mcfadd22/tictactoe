# Results Git-Sync for Headless VM Runs — Design

**Date:** 2026-06-03
**Status:** Approved (design)
**Branch:** `parker/symmetry-experiments`
**Builds on:** `docs/design/specs/2026-06-02-composable-symmetry-experiments-design.md`

## Motivation

The full Phase-1 sweep (`run_experiment.py --condition all`) is long-running and
CPU-bound. To run it on a VM and stream artifacts back for review (by the operator
and a collaborator), the runner should **commit and push each condition's results to
the current branch as it finishes**, incrementally, so a preempted VM still preserves
everything completed so far. Results currently land in `results/<name>/`, which is
gitignored.

## Decisions (resolved with user)

1. **Target:** the **same branch** (`parker/symmetry-experiments`) on `origin` (the
   collaborator's repo `mcfadd22/tictactoe`) — code and results live together; one
   branch to watch.
2. **Gitignore:** `results/` stays ignored for local dev. Force-add (`git add -f`)
   overrides it, and **only** when the new `--push-results` flag is passed.
3. **Cadence:** one commit + push per condition (E0, then E1, …), after that
   condition's `save_condition(...)`.
4. **Robustness:** push is **best-effort** — network/auth/rebase failures are logged
   and the sweep continues (results stay committed locally and ride out on the next
   condition's push). A commit failure (e.g., missing git identity) is logged loudly
   but also does **not** kill the run — results are always on disk via `save_condition`.

## Architecture

### New module `ttt/gitsync.py`

A small, isolated `subprocess`-based wrapper around `git`. No project imports, so it is
independently testable.

- `current_branch(cwd=None) -> str` — `git rev-parse --abbrev-ref HEAD`.
- `commit_results(paths, message, cwd=None) -> bool` — `git add -f <paths>`, then commit
  **iff** something is staged (returns `False` on no-op, avoiding empty-commit errors).
  Raises `GitSyncError` if the commit itself fails (so the cause — e.g. unset
  `user.email` — is explicit).
- `push(branch=None, cwd=None, rebase=True) -> bool` — best-effort. If the remote branch
  exists (`git ls-remote --exit-code origin <branch>`), `git pull --rebase` first to
  absorb upstream code pushes; on rebase failure, `git rebase --abort` and skip. Then
  `git push origin <branch>`. Returns `True`/`False`; never raises.
- `sync_results(paths, message, cwd=None, branch=None) -> None` — orchestrates
  commit + push; catches `GitSyncError` and push failures, logs a clear `[gitsync]`
  line, and **never raises**.

### `run_experiment.py`

- New `--push-results` flag (`store_true`, default **off** — local runs unchanged).
- After each condition's `save_condition(...)`, when the flag is set, call
  `sync_results(["results/<name>"], message)` where `message` =
  `"results: <name> (<G> configs x <S> seeds, encoding=<enc>, head=<head>)"`.
- Wiring is threaded through `run_one` (including the E1 positive-control call).

## Operator setup (VM, out of code scope)

On the VM: clone the repo, `git checkout parker/symmetry-experiments`, create the venv,
`pip install -r requirements.txt`, set `git config user.name`/`user.email`, and configure
push auth (HTTPS PAT or SSH deploy key). Then:
`.\.venv\Scripts\python.exe run_experiment.py --condition all --push-results`.

## Testing

`ttt/gitsync.py` is tested hermetically (no network): a temp `git init` repo plus a local
`--bare` repo as `origin`.
- `commit_results` returns `True` on first commit, `False` on an unchanged re-run.
- `sync_results` pushes a result file; the bare remote's log shows the commit.
- A bad/absent remote makes `push` return `False` without raising.
- First push with no pre-existing remote branch skips the rebase and still succeeds.

`run_experiment.py`: `--push-results` parses and defaults `False`; the existing CONDITIONS
matrix tests are unaffected.

## Scope / non-goals (YAGNI)

- No dedicated results branch, no Git LFS, no retry/backoff loops, no per-epoch streaming.
- The helper does not configure git identity or auth — that is operator setup.
