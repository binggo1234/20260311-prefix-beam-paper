# Evaluation Yardstick Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give restricted-prefix utilization results a fair same-order reference and an explicit relaxation bound.

**Architecture:** Two isolated evaluation modules, minimal runner/handoff dispatch, a private locked development batch and a public aggregate report. Existing algorithms are unchanged.

**Tech Stack:** Existing Python, pytest, Decimal and existing decoder/budget/audit APIs; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-11-evaluation-yardstick-design.md`

## Global Constraints

- Keep existing restricted-prefix search, legacy SA, kerf and trim semantics unchanged.
- All replay calls receive the absolute search deadline; retain completed incumbents on optional interruptions.
- No SOTA/commercial/optimality claim without verified same-input evidence; private orders/layouts stay local.
- 30-run diagnostic batch, 30/3/38/8 s lifecycle, no outcome-dependent retuning or reruns.

## Review Focus

- Optional replay mutates caches before timing out: completed saved snapshots must remain usable.
- Rotation-only parts and trim/kerf boundaries: bounds must match the actual geometry convention.
- Small orders with duplicate deterministic rankings: loops must terminate by time or 6000 proposals.
- SA accepts a worse current solution: published result must still retain Greedy or better.
- Adaptive configuration inheritance: all three methods must decode identically at every order size.

### Task 1: Budgeted reference baselines and area bounds

**Files:** Create `src/inrp/evaluation_baselines.py`, `src/inrp/evaluation_bounds.py`, `tests/test_evaluation_yardstick.py`; modify `src/inrp/runner.py`, `src/inrp/handoff.py`.

**Interfaces:** Consume `baseline_sequence(parts)`, `OmnipotentDecoder.replay_order(order, prefix_snapshot=..., deadline=..., deadline_phase=...)`, `_global_lex_score`, `_scalar_reward`, `BudgetExpired`, `PackingRequest`. Produce `solve_evaluation_baseline(parts, cfg, seed) -> SearchResult`, `packing_area_bounds(request) -> dict`, variants `baseline_order_multistart` and `baseline_sequence_sa_budgeted` through both runner and handoff.

- [ ] Write tests for exact envelope-vs-nominal lower bounds (two 44x88 pieces on 100x100, trim 5, kerf 2 => lower bound 2, net ceiling 0.3872), decimal boundary, invalid/rotation geometry; real budgeted handoff for both variants; shared decoder config at three order sizes; exact deadline interruptions before/after the first complete incumbent; whole-sequence proposal permutations and seeded repeatability.
- [ ] Run `python3 -m pytest tests/test_evaluation_yardstick.py -q`; Expected: missing feature failures.
- [ ] Implement spec in isolated modules and minimal dispatch; no unrelated refactor. Use prefix snapshots only at the first changed index. Add behavioral regression tests for any discovered issue before fixing it.
- [ ] Run `python3 -m pytest -q -rs` and `python3 -m compileall -q src`; Expected: all runnable tests pass, only existing external skips.
- [ ] Commit implementation, tests, plan/spec. Expected: clean runtime checkout.

### Task 2: Freeze, run and report the diagnostic comparison

**Files:** Private QA scripts/manifest/results in `evaluation_yardstick_20261111/`; public `docs/development/evaluation-yardstick-20261111.md`; private roadmap and existing PR description.

**Interfaces:** Consume Task 1 variants and bounds, current private manifest/loader/independent geometry audit, validated prior `collect_worker` lifecycle helper. Produce locked schedule, all 30 result records, task/seed table, bounds, audited aggregates and a GitHub draft update.

- [ ] Add private runner/summarizer, test schedule balance and failed-run reporting before implementation. Run the new harness tests; Expected: fail on missing new behavior, then pass after implementation.
- [ ] Freeze runtime and hashes, then execute all 30 rows once. Expected: every scheduled row has an honest success/failure record and bounded worker lifecycle; no source changes while running.
- [ ] Re-audit recorded geometry/hash identity; summarize each method's seeds/means and sampled feasible best against bounds. Expected: no unverifiable BKS/commercial entries and no fabricated optimality gap.
- [ ] Write public aggregate report and private six-item roadmap update, update existing draft PR after required verification. Expected: remote branch matches committed tested HEAD, draft stays open, master unchanged.

## Execution decisions

The user has delegated routine technical choices and authorized GitHub updates. Execute inline without another design/permission checkpoint. A single fresh reviewer is required by executing-plans before freezing runtime; it reviews this task's diff from b577f516, not the historic branch. Task 2 is an experiment/report deliverable, not an algorithm modification.
