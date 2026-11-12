# Two-completion Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run one matched-initialization two-completion early-prefix comparison.

**Architecture:** Isolated experimental subclass/adapter reuses current solver without changing existing default paths. Private runner freezes one pair and audits captured snapshots without extra packing.

**Tech Stack:** Existing Python dataclasses/pytest, multiprocessing spawn and existing absolute budget/handoff/geometry interfaces; no dependencies.

**Spec:** `docs/superpowers/specs/2026-11-12-two-completion-control-design.md`

## Global Constraints

- Restricted-prefix topic and paused new heuristics remain.
- Two initial decodes separately counted; at most two started full post-initialization completions per arm, including failed starts.
- Existing `rh_mcts_prefix_short_completion_on` selection/profile/geometry unchanged; no known witness identity or position override.
- Per-arm30 s pipeline/3 s closing/38 s parent watchdog/8 s cleanup.
- One739-piece seed1000 pair, schedule seed20261116; no retries/retuning/extra matrix.
- Public code/tests/spec/plan/report only; private raw factory evidence; authorized branch/draft only, no master merge.

## Review Focus

- A third replay/promotion or cache-hit accounting must not silently exceed the allowance or create phantom mode counters.
- Quota, initial-incomplete, replay timeout and absolute close expiry must remain distinct and preserve only complete valid incumbents.
- Instrumentation/reference must not change common initial RNG/cache or the current search's first two choices.
- Unsupported arm/profile must reject before decoding; existing default solver remains unaffected.
- Saved raw snapshots, digests, handoff/record fields and geometry audits must remain bound, including partial/failure outcomes.

### Task 1: Experimental adapter and frozen-run harness

**Files:**
- Create: `src/inrp/early_prefix_control.py`, `tests/test_early_prefix_control.py`.
- Create privately under QA: `run_two_completion_control_20261112.py`, `audit_two_completion_control_20261112.py`, `test_two_completion_control_20261112.py`.
- Existing `cfg.py`, `mcts.py`, `runner.py`, `handoff.py` stay unchanged.

**Interfaces:**
- Consumes: existing `PackingRequest`, caller-created `TotalBudget`, common initialization and `_formal_result_from_incumbent()`.
- Produces: `solve_two_completion_handoff(request, budget, arm)`; normal `PackingHandoff` with experimental metadata and up-to-two saved evaluation receipts. Private runner `main()` creates new output directory and prelocks one pair; private auditor reads saved files without replay.

- [ ] Write tests exercising real toy runs: both arms count2 initial+2 completion starts/successes, zero third replay, identical initial state, reference uses existing two entry UIDs, search first-two trajectory matches an independently truncated original solver; final score is best initial/evaluations.
- [ ] Run `PYTHONPATH=src python3 -m pytest -q tests/test_early_prefix_control.py`; expect failure for the missing adapter, then meaningful assertions as features are introduced.
- [ ] Implement the adapter and receipts, adding replay-boundary expiry/error tests and pre-decode unsupported-arm/profile checks. Ensure second incumbent is installed before quota close and no timeout flag on quota.
- [ ] Write private schedule/record-corruption/failure-classification tests before their implementation. Implement one-pair runner using existing `collect_worker`,30/3/38/8 lifecycle, and auditor binding common/evaluation/final raw identities and geometry. Keep failures without rerun.
- [ ] Run `PYTHONPATH=src python3 -m pytest -q` and private tests; expect all pass, only existing optional CutEval skips. Run compilation and `git diff --check`; expect exit0.
- [ ] Commit code/tests/spec/plan and create a fresh implementation review package covering the entire task plus private harness. Fix Critical/Important findings in one RED→GREEN pass, then rerun the suite. Do not collect factory data before this gate.

### Task 2: One frozen pair, audit and publication

**Files:** private `two_completion_control_20261112/`, QA research roadmap, public `docs/development/two-completion-control-20261112.md`, private prepared PR body.

**Interfaces:**
- Consumes: reviewed Task1 adapter/harness, frozen known739 source identity and historical initial reference.
- Produces: two preserved arm outcomes, independently audited comparison with accepted/incomplete status, aggregate six-part report and publication receipt.

- [ ] Freeze source/dependency/input/reference/workspace identities; run the private runner once. Expect two retained scheduled records, each at most2 completion starts; quality outcome is not specified.
- [ ] Run private auditor; accepted pair requires complete identical initial state, exactly2 successful completions per arm, valid all-layout geometry/digest/record bindings and clean bounded lifecycle. Otherwise report incomplete without retry.
- [ ] Generate factual report and roadmap entry covering version/scope, counts/budgets, quality/time, verification, limits and next priority. Recheck source/dependencies/protected workspaces; compile/diff checks must pass.
- [ ] Commit aggregate report, push existing branch and update existing draft PR with prepared structured body; verify exact remote head/body, draft state, unchanged master and path whitelist. Empty remote checks are not a CI pass.
