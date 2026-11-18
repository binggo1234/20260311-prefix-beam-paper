# Prefix Witness Advancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a strict complete saving witness can advance restricted-prefix search early and improve final board count.

**Architecture:** Return an optional legal witness chain and saved completion from the existing simulation loop. Consume it before the prefix beam's ordinary commit checkpoint; all non-witness paths remain unchanged.

**Tech Stack:** Existing Python 3.9, pytest, existing solver/decoder and private batch tools; no dependencies added.

**Spec:** `docs/superpowers/specs/2026-11-11-prefix-witness-design.md`

## Global Constraints

- Keep current candidates, priors/PUCT, completion/warmstart quotas, commit max steps, commit depth/width and geometry.
- Default off; require prefix+staged activation; strict complete board saving only.
- Total 30 s including audit/write, closing 3 s, external watchdog 38 s; private runner cleanup grace 8 s is not solver allowance.
- Concurrent 3 tasks×3 seeds×2 methods plus 3 Greedy, randomized blocks and method order; no selective reruns.
- Preserve original dirty checkout, frozen formal baseline and old evidence; update draft PR only after verification.

## Review Focus

- Staged evaluation updates incumbent before returning: comparison must use pre-evaluation board count.
- Same-board score improvement, incomplete completion or cutoff cannot trigger a false advance.
- PUCT paths beyond the existing commit length/depth limits cannot bypass those limits.
- A complete partial-fidelity result can advance without converting its cache to a full mode group.
- Re-ranking the witness must not start extra completion work or leak it to a later root.

### Task 1: Implement and verify the isolated witness policy

**Files:** Modify `src/inrp/cfg.py`, `src/inrp/runner.py`, `src/inrp/mcts.py`; create `tests/test_prefix_witness.py`.

**Interfaces:** `_run_simulations(root, n_sims)` returns `None` on the old path or `(chain, completed_snapshot)` for an eligible witness; existing non-prefix callers ignore the optional return. `_solve_prefix_beam` consumes it within the same batch. No request schema changes.

- [x] Write failing tests for strict pre-evaluation improvement, disabled/tied/incomplete/expired/terminal cases, existing commit depth/length/admissibility, no extra completion during early commit, resumed deeper complete evaluation, profile isolation and real budgeted handoff.
- [x] Run `python3 -m pytest -q tests/test_prefix_witness.py`; confirm missing behavior causes failures.
- [x] Implement default flag, inherited variant, optional return and same-batch consumption with evidence.
- [x] Run targeted tests, entire `python3 -m pytest -q -rs`, compileall and diff checks; expect all available tests pass, known 3 CutEval skips reported.
- [x] Commit this feature and tests after verification.

### Task 2: Lock and run one paired development batch

**Files:** Private sibling QA runner/summarizer/tests; public `docs/development/prefix-witness-20261111.md`, README.

**Interfaces:** Consume Task 1 runtime commit/variant/meta; use existing handoff and independent geometry auditor. Private progress retains failures plus saved-result identities.

- [x] Test private cleanup boundary/failure reporting before freezing runner; both methods use identical worker lifecycle.
- [x] Lock 21-run manifest and execute once; verify all outcomes including failures, source/result hashes, net/envelope utilization, actual deeper evaluations and Greedy floor.
- [x] Summarize all task×seed outcomes; decide stop/continue using additional final saving evidence, never only deeper search.
- [x] Obtain one fresh code review of the single-feature diff before runtime freeze; address important issues with failing tests. Do not edit runtime during the batch.
- [x] Update docs with exact results/limits, rerun suite and compile/diff checks; commit and push current research branch; update/attach existing draft PR; do not merge master.
