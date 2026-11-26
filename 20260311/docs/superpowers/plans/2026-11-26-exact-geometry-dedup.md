# Exact complete-geometry sequence deduplication implementation plan

**Goal:** Test whether repeated complete geometry sequences can be skipped without changing the represented layouts or their first-best selection.

**Architecture:** Add an explicit, default-off option to the existing single-prefix scan. Compare exact complete sequences of envelope dimensions, nominal dimensions and ordered rotation permissions within one solver instance. Keep the first real UID representative and its real layout. Do not transplant snapshots or cache values between UIDs. Retain the full existing candidate order and separately export skipped UID-to-representative records.

**Tech stack:** Existing Python solver and pytest; private experiment wrappers remain local.

**Spec:** The user-approved next step in `thesis/qa/ch2_material_focus_20261126/材料收益与重复评价核查.md`; restricted prefixes and deterministic short-edge completion remain fixed. No new shape ranking heuristic, deeper prefix, repair or default profile.

## Correctness argument and limits

At equal complete geometry words, deterministic local placement sees the same free rectangles and orientations at each position; local and board selection keys use geometry, area and board position, not UID. Nominal geometry and its accumulation order coincide, so terminal score coincides. UID and source label are copied from the real part at each position. `_update_global_best` uses strict score improvement, so later equal-score representatives cannot replace the first. This argument assumes correct cache results, as does the current decoder; existing rounded/hash cache risks are inherited, not solved or claimed absent here. Never compare or reuse identity hashes as geometry equivalence evidence.

## Task 1: Explicit option and correctness checks

- [ ] Add failing tests for opt-in pruning, unchanged default, identity restoration, exact dimensions, nominal area, rotation permissions and interleaved equal shapes.
- [ ] Add exact `_geometry_sequence_key` and `deduplicate_geometry=False` keyword; record successful first representatives and explicit skips.
- [ ] Test exhaustive small permutations against cold and warmed deterministic decoders, including differently labelled real parts; compare every state modulo positional UID renaming and recompute actual hashes.
- [ ] Test quota, deadline and failure handling, complete identity output and first-best preservation. Run the full existing pytest suite and syntax/whitespace checks.

## Task 2: Frozen minimal development comparison, only if Task 1 passes

- [ ] Freeze source/dependency/input/history hashes, schedule and bounds before new industrial outputs.
- [ ] Fixed work: one pair for all 178-piece K1 candidates, and one pair for the first 12 distinct geometry words of the 2416-piece candidate order. The UID baseline stops after the same distinct-word coverage, including its repeated evaluations. Each arm gets 90 seconds total, 10 seconds closing reserve, 98 seconds parent bound including cleanup; no retry or tuning. Both arms retain the same strong initialization.
- [ ] Fixed time: one 2416-piece pair, seed1000, 30 seconds whole pipeline, 3 seconds closing reserve, 38 seconds parent bound, cleanup capped at 8 seconds. Same current decoder, initializer and full candidate order. Schedule seed20261126; independently shuffle each pair's arm order. No new order or seed.
- [ ] Verify every saved complete layout from source, actual counts, UID mapping, shared representative scores/layouts, request and initial state equality. Count generation, key construction, initialization, incomplete work, export and audits in time. Startup/exit covered by parent bound; do not compare parent and child absolute clocks.
- [ ] Report within-pair timing and material separately. A failed or incomplete arm makes the pair inconclusive. Preserve failures; do not repeat or extend budgets after outcomes. Known developer orders establish no generalization or novel algorithm contribution.

## Task 3: Delivery

- [ ] Preserve all prior outputs and the Word master. Summarize concrete implementation, correctness, timing and board count; no sum of old and new speedups.
- [ ] If implementation and checks pass, update the already authorized research branch/draft only, with aggregate development evidence; no master merge or private industrial data upload.

## Execution status, 2026-11-26

Tasks 1 and 2 completed: 440 production tests passed, three existing optional integrations skipped; seven private protocol checks passed. All six frozen runs and three paired comparisons were valid, with saved-output re-audit. Fixed-time board count did not improve. See the [development result](../../development/exact-geometry-dedup-20261126.md) for complete counts, timing and limitations. The checkboxes above preserve the pre-execution protocol.
