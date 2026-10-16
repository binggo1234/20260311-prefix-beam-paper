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
- [ ] Run the full baseline suite and commit the code-only baseline import.

## Task 2: Exact Greedy and evidence

Files: src/inrp/ordering.py, runner.py, mcts.py; tests/test_prefix_development.py.

- [ ] Write failing tests for exact Greedy injection, deadline survival, evidence source and nonnegative interrupted-phase timings.
- [ ] Add baseline_sequence(parts) shared by both consumers, preserving runner._baseline_sequence compatibility.
- [ ] Seed the prefix solver with exact Greedy before metadata-ranked initial solutions; record initial/final board counts.
- [ ] Record warmstart, prefix-search, candidate-generation and per-mode completion time separately; distinguish nested timers.
- [ ] Add source/phase/mode/depth fields to complete-incumbent events without changing the score or handoff schema.
- [ ] Run targeted and full tests, then commit.

## Task 3: Complementary candidates

Files: cfg.py, runner.py, mcts.py; tests/test_prefix_development.py.

- [ ] Write failing behavior tests for pair fit, rotation permissions, geometry inflation, action cap, deadline and opt-in handoff.
- [ ] Implement _complement_macro_actions(snapshot, remaining_mask, limit) over a bounded deterministic scout subset.
- [ ] Rank horizontal/vertical pair envelopes by compactness and actual free-space fit; replay both pair orders through the existing decoder.
- [ ] Prefer distinct pair part sets and resulting layout states, while retaining a single fallback.
- [ ] Integrate under rh_mcts_prefix_beam_complement; leave the existing variant's candidate behavior unchanged.
- [ ] Record pair generation/selection and candidate coverage counters.
- [ ] Run targeted and full tests, then commit.

## Task 4: Verify and publish

- [ ] Run small synthetic handoff/deadline checks and a bounded representative development comparison; retain negative results.
- [ ] Document scope, exact commands, costs, remaining limitations and deferred staged-completion/repair work.
- [ ] Review all diffs and confirm no factory input/output additions.
- [ ] Push only the development branch, create and attach a draft PR, verify remote SHA.
