# Prefix complement development plan

> **For agentic workers:** Execute these tasks sequentially in this isolated clone using test-driven development and verification-before-completion. No parallel agents are required.

**Goal:** Protect the exact Greedy reference, attribute search costs and improvements, and introduce bounded complementary macro candidates.

**Architecture:** Keep the existing decoder, handoff and restricted prefix search. Share the canonical Greedy ordering between runner and MCTS. Add opt-in complementary pair generation as a distinct prefix variant, retaining the existing candidate count and all completed incumbents.

**Tech Stack:** Python 3.9-compatible implementation, existing OR-Tools dependency, pytest.

**Spec:** User-approved in-chat first-round design, 2026-10-16: exact Greedy floor, stage source/timing evidence, complementary candidates; no long formal batch in this iteration.

## Global constraints

- The frozen baseline is bd5e406e020d56191e92872749f7f43a1cc44afd; its checkout, inputs, manifests and outputs remain unchanged.
- Develop on research/ch2-prefix-complement-20261016, based on GitHub master bd6022e434ad84bbdb25ff9ecdef4b2f78316052.
- Import only baseline runtime code, tests and scripts. Do not add raw factory inputs or experiment outputs to GitHub.
- Preserve nominal dimensions, tool envelopes, trims, identities and allowed rotations.
- Total deadlines include all initial solutions and diagnostic instrumentation.
- Complementary macros are sequences decoded by the existing decoder, not rigid compound shapes.
- Publish the reviewed branch and a draft PR; do not merge into master.

## Review focus

- Greedy survives deadline expiry during later warmstarts or completion.
- Complementary pairs use solver dimensions and allowed rotations; they cannot consume parts twice.
- Every selected action and corresponding metadata fit the existing action cap, including caps of one.
- Disabled complement mode retains candidate behavior; expired deadlines stop candidate generation.
- Improvement timestamps use a stable run origin, and measured phases include interrupted work.

## Task 1: Portable baseline

- [x] Copy frozen runtime/test sources into a fresh clone based on GitHub master.
- [x] Reproduce baseline test failures and remove dependence on private manifest paths using synthetic schedule rows.
- [x] Explicitly skip optional cross-chapter integration when pinned clean repositories are unavailable.
- [x] Run the full baseline suite and commit the code-only baseline import (199 passed, 3 skipped; b972f4b).

## Task 2: Exact Greedy and evidence

Files: src/inrp/ordering.py, runner.py, mcts.py; tests/test_prefix_development.py.

- [x] Write failing tests for exact Greedy injection, deadline survival, evidence source and nonnegative interrupted-phase timings.
- [x] Add baseline_sequence(parts) shared by both consumers, preserving runner._baseline_sequence compatibility.
- [x] Seed the prefix solver with exact Greedy before metadata-ranked initial solutions; record initial/final board counts.
- [x] Record warmstart, prefix-search, candidate-generation and per-mode completion time separately; distinguish nested timers.
- [x] Add source/phase/mode/depth fields to complete-incumbent events without changing the score or handoff schema.
- [x] Run targeted and full tests, then commit (c7a8371; 204 passed, 3 skipped).

## Task 3: Complementary candidates

Files: cfg.py, runner.py, mcts.py; tests/test_prefix_development.py.

- [x] Write failing behavior tests for pair fit, rotation permissions, geometry inflation, action cap, deadline and opt-in handoff.
- [x] Implement _complement_macro_actions(snapshot, remaining_mask, limit) over a bounded deterministic scout subset.
- [x] Rank horizontal/vertical pair envelopes by compactness and actual free-space fit; replay both pair orders through the existing decoder.
- [x] Prefer distinct pair part sets while retaining a single fallback; preserve existing beam-state deduplication. New action-level geometric-state deduplication is deferred.
- [x] Integrate under rh_mcts_prefix_beam_complement; leave the existing variant's candidate behavior unchanged.
- [x] Record pair generation/selection and candidate coverage counters.
- [x] Run targeted and full tests, then commit (82d6efb; 214 passed, 3 skipped).

## Task 4: Verify and publish

- [x] Run small synthetic handoff/deadline checks and a bounded representative development comparison; retain negative results (12 runs, geometry audited).
- [x] Document scope, exact commands, costs, remaining limitations and deferred staged-completion/repair work.
- [x] Add red/green boundary regressions for completed Greedy before cache registration and interrupted completion mode attribution (c4ee424; targeted 24 passed, full 215 passed, 3 skipped).
- [x] Review development diffs and baseline import file list; confirm no factory input/output additions.
- [x] Address independent review's non-formal completed-incumbent closure finding; preserve no-incumbent diagnostics and overall deadline enforcement (aa83e33; full 217 passed, 3 skipped on 2026-10-20).
- [x] Push only the development branch, create and attach [draft PR #1](https://github.com/binggo1234/20260311-prefix-beam-paper/pull/1), verify remote SHA and unchanged master (2026-10-20). Independent review found no remaining blocker in the development diff; prerequisite-import and external-integration merge coverage remain outside that review.
