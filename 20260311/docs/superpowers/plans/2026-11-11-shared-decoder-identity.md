# Shared Decoder Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix shared identity caching and measure restricted-prefix results against an explicit short-edge reference.

**Architecture:** One shared cache-key correction, removal of the redundant reference wrapper, and minimal deterministic-order/dispatch additions. A separately locked private batch uses the existing loader, audit and worker lifecycle.

**Tech Stack:** Existing Python/pytest/decoder/budget APIs, no dependencies.

**Spec:** `docs/superpowers/specs/2026-11-11-shared-decoder-identity-design.md`

## Global Constraints

- Keep existing restricted-prefix search settings and geometry unchanged; no new heuristic.
- Shared decoder and all references use the same UID-safe cache contract.
- 15-run development batch only, 30/3/38/8 s lifecycle, no selective reruns or tuning.
- Preserve old evidence, original dirty checkout and private factory inputs; update existing draft only.

## Review Focus

- Shape-equivalent identities permuted across replays: shared and cold layouts/hashes must agree.
- Same UID replay: valid cache reuse must remain available.
- Rotation-only parts and duplicate dimensions: cache reuse must preserve identity and geometry.
- Optional deadline interruption: no contamination of saved complete results or subsequent replay.
- Short-edge tie ordering and adaptive tiers: reference must match the previously defined deterministic order and decoder settings.

### Task 1: Shared fix and deterministic short-edge entry point

**Files:** Modify `src/inrp/mcts.py`, `evaluation_baselines.py`, `ordering.py`, `runner.py`, `handoff.py`; create `tests/test_decoder_identity.py`.

**Interfaces:** Consume `OmnipotentDecoder.replay_order`, `_micro_key`, canonical `baseline_sequence`, existing Greedy runner/handoff dispatch and TotalBudget. Produce shared UID-safe caching; `short_edge_sequence(parts) -> Tuple[int, ...]`; variant `baseline_short_edge` supported by runner and handoff.

- [ ] Add failing regressions for complete identities/real hashes vs cold replay with duplicate-shape and rotation-only parts; interrupted replay followed by a valid replay; same-state/UID reuse; literal short-edge order and actual budgeted handoff/profile dispatch.
- [ ] Run `python3 -m pytest tests/test_decoder_identity.py -q --tb=short`; Expected: UID corruption/hash mismatch and missing short-edge feature failures.
- [ ] Add UID to shared micro key, remove redundant reference subclass, implement the exact deterministic order and minimal dispatch. No other algorithm changes.
- [ ] Run `python3 -m pytest -q -rs`, `python3 -m compileall -q src tests scripts`, `git diff --check`; Expected: all runnable tests pass, only existing three CutEval skips.
- [ ] Commit code/tests/spec/plan; obtain one fresh review of the entire new diff from `02bf332`; fix Important findings through RED→GREEN tests and full suite before freezing.

### Task 2: Concurrent comparison and publication

**Files:** Private QA runner/summarizer/tests and `shared_identity_short_edge_20261111/`; public `docs/development/shared-identity-short-edge-20261111.md`; private roadmap/PR body.

**Interfaces:** Consume Task 1 variant/shared fix and existing private `collect_worker`, loader, area bounds and geometry audit. Produce exact 15-row schedule, all result/failure records and task-level comparisons.

- [ ] Test allocation balance and failed-run summary before implementation; Expected: missing-feature failures then passes.
- [ ] Freeze code/input/script hashes and 15-row schedule with seed 20261113; execute once. Expected: every scheduled row recorded, all failures retained and no runtime mutation.
- [ ] Reaudit all saved outputs and identities; report all three prefix seeds, short-edge/Greedy counts, means, net U/envelope occupancy and honest failure limits. Expected: no best seed passed off as mean, no historical timing comparison.
- [ ] Write aggregate report and six-item roadmap; full tests, compile/diff checks; push existing branch and update existing draft PR. Expected: verified remote HEAD/body match, draft open, master unchanged.

Execute inline under the user's existing delegation; no new permission checkpoint. A fresh reviewer is required by executing-plans, not a team of implementers. Archive the completed execution ledger recoverably with the private batch.
