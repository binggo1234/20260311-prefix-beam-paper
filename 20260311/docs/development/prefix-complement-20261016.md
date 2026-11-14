# Restricted-prefix complement development, 2026-10-16

This iteration adds an exact Greedy incumbent, search attribution, and an opt-in complementary-pair candidate generator. It does not establish a new best method or replace frozen paper evidence.

## Provenance and scope

The development branch starts from GitHub master `bd6022e434ad84bbdb25ff9ecdef4b2f78316052`. Commit `b972f4b` ports the audited runtime, tests, and support scripts from the separately frozen local baseline `bd5e406e020d56191e92872749f7f43a1cc44afd` onto that history. This prerequisite import accounts for most of the PR's size; subsequent commits isolate the new algorithm work.

The frozen checkout, its attempt7 manifest/results, and the original working checkout are not modified. No new factory inputs, layouts, or raw run outputs are added to this branch. Existing tracked historical data remain unchanged. The active project root is `20260311/`; the top-level duplicate `src/` is inactive and unchanged. The ignore rule is anchored to `/src/` so it no longer hides newly added active source modules.

There are no new runtime dependencies. The existing OR-Tools requirement is retained; tests use the existing pytest environment.

## Behavior changes

1. `inrp.ordering.baseline_sequence` is shared by the Greedy runner and prefix warmstart path. Its ordering is exactly the previous Greedy nominal-area/aspect/long-edge/UID tie break, rather than the metadata-biased warmstart order. Prefix variants decode it first, including when other warmstarts are disabled. A completed result is retained before optional cache registration, so later deadline expiry cannot discard it. The floor is conditional on that first decode completing within budget; it is not a promise of a feasible result under an arbitrarily short budget.
2. Strict complete-incumbent improvements record their source and timing context. Inclusive phase timers also account for interrupted work. This is instrumentation, not a change to the lexicographic objective or handoff schema.
3. `rh_mcts_prefix_beam_complement` inherits the existing adaptive prefix profile and enables complementary pairs. Other variants do not enable the new candidate generator by default. Both prefix controls now receive the exact Greedy floor, so neither should be mislabeled as frozen attempt7.

The pair generator uses a deterministic scout pool capped at `MACRO_COMPLEMENT_POOL` (default 16, hard maximum 64), permitted orientations, solver-inflated dimensions, and trimmed board dimensions. It ranks feasible horizontal/vertical pair envelopes by existing tail-space fit, envelope compactness, and nominal area. Both pair orders are decoded through the existing decoder; candidates whose two parts land on different boards are rejected. These are placement sequences, not rigid compound shapes or a new exact geometry solver.

Selection reserves a single-part fallback and up to `MACRO_COMPLEMENT_TOPK` distinct complementary part sets (default 2), within the existing action cap. Remaining slots use the original candidate pool with part-set deduplication. Reverse orders are compared but only one action per part set is retained. No new action-level geometric-state equivalence test is introduced; the existing beam-state deduplication remains unchanged. The disabled/zero-quota candidate policy is regression-tested.

The pool and selected-action count are bounded, but pair screening still scans orientations and tail free rectangles, and accepted proposals require extra decoder work. A fixed action cap does not imply equal CPU cost.

## Budgeted usage

Run from the active project root after installing the existing requirements. For example:

```python
from inrp.budget import TotalBudget
from inrp.handoff import PackingPart, PackingRequest, solve_packing_handoff_with_budget

# Create the absolute budget before loading/building the input.
budget = TotalBudget(30.0, 3.0)
request = PackingRequest(
    case_id="synthetic-complement",
    parts=(
        PackingPart("a", "a", 54.0, 94.0, (0,)),
        PackingPart("b", "b", 34.0, 94.0, (0,)),
    ),
    board_width_mm=110.0,
    board_height_mm=110.0,
    trim_mm=5.0,
    tool_diameter_mm=6.0,
    time_limit_s=30.0,
    solver_variant="rh_mcts_prefix_beam_complement",
)
handoff = solve_packing_handoff_with_budget(request, budget)
print(handoff.panel_count, handoff.provenance["solver_meta"])
```

Use `PYTHONPATH=src` when the package is not installed. Request dimensions are nominal; the handoff path handles tool clearance. Do not manually inflate them. Formal result serialization still needs the existing budget-aware writer/watchdog; this example only obtains a validated handoff.

## Evidence fields

All fields are in `handoff.provenance["solver_meta"]`.

| Field | Meaning |
| --- | --- |
| `greedy_floor_panels`, `warmstart_best_panels` | Completed exact-Greedy board count and best warmstart board count; may be absent if no such complete result exists. |
| `global_best_events` | Strict improving complete incumbents, including `source`, `phase`, `completion_mode`, and `prefix_depth`. Generic legacy update paths retain source `search`. |
| `evidence_time_origin` | `solver_init`: event elapsed time is measured from the solver evidence clock initialized in its constructor, not the external input-loading budget. |
| `phase_time_s` | Inclusive accumulated warmstart, prefix search, candidate generation, complement generation, and completion times. Nested phases must not be summed as total wall time. |
| `completion_mode_time_s` | Replay time by completion mode, including an interrupted mode. It excludes suffix ordering and subsequent scoring; an interrupted group is not cached as fully evaluated. |
| `puct_selections` | Actual PUCT child selections, not the number of score-function evaluations. |
| `complement_pairs_considered` | Pair-screening attempts, accumulated across generation calls. |
| `macro_complement_actions` | Selected complementary actions, accumulated across calls; not unique visited states. |
| `candidate_unique_parts_max`, `candidate_unique_parts_total` | Maximum/sum of the distinct-part union in each selected candidate set; not globally unique explored parts. |
| `main_profile_tier` | Adaptive profile tier chosen by the existing runner. |

In interrupted completion groups, `oracle_completion_last_best_mode` is updated to the mode that actually supplied the usable complete result, rather than retaining a previous group's mode. `prefix_depth` in simulation events is tree macro depth, not a part count or a promise that the evaluation leaf obeys the beam commit-depth limit.

## Validation

Final verification on 2026-10-20, locally with Python 3.9.6:

```sh
cd 20260311
python3 -m pytest -q tests/test_prefix_development.py tests/test_completion_deadline_contract.py
python3 -m pytest -q -rs
git diff --check
```

The targeted suite has 26 passing tests; the full suite has 217 passing tests and 3 skips. The skipped tests require clean pinned external CutEval/joint repositories; the local CutEval checkout has user changes and was left untouched. On a machine without these optional repositories, the cross-chapter test module is explicitly skipped. Its source-identity unit checks still run when the source roots exist. Restoring the expected clean external versions and rerunning `tests/test_minibatch.py` is necessary to claim full cross-chapter integration coverage; do not discard user changes to obtain it.

The manifest-schedule test uses synthetic 11-task rows rather than a private absolute manifest path. New tests exercise exact Greedy injection, deadline survival before cache registration and during later replay, stable timestamps, interrupted completion attribution, actual PUCT selection, pair compactness, per-part/global rotation permissions, inflated geometry/trim, candidate caps, disabled/zero-quota behavior, and a real budgeted handoff.

Independent pre-publication review identified a further local-time-limit path: a completed Greedy incumbent existed, but a later `completion_replay` timeout was re-raised rather than closed into a result. Commit `aa83e33` permits closing that incumbent for non-formal prefix calls too, without changing the absolute closing-deadline checks. Non-formal calls with no complete incumbent still re-raise the original timeout; formal metadata/behavior remain compatible. Regression tests cover the usable fallback, absence of misleading formal metadata, and refusal to close after the overall deadline. The broader 135-test prefix/handoff/quality subset also passes.

### Bounded local diagnostic, not formal paper evidence

Three representative local tasks were run once per method, seed 1000, serially in the fixed order Greedy / warmstart-only / corrected prefix / complement prefix. Each run used a 30 s total budget with 3 s closing reserve and an independent geometry audit. Warmstart-only used the corrected prefix configuration but stopped after constructive initialization. Source hashes, event details, and the harness remain in local QA storage, outside this repository.

The diagnostic was run at `82d6efb601198824fb07925ebd2403098f9a894d`, before the separately regression-tested cache-registration/partial-mode attribution boundary fixes in `c4ee424` and local-time-limit closure fix in `aa83e33`. It must not be presented as a final-commit benchmark.

| Parts | Greedy boards | Warmstart-only boards | Corrected prefix boards / elapsed s | Complement prefix boards / elapsed s |
| ---: | ---: | ---: | ---: | ---: |
| 178 | 29 | 29 | 29 / 3.947 | 29 / 4.160 |
| 739 | 98 | 98 | 97 / 27.148 | 97 / 27.166 |
| 2416 | 374 | 374 | 374 / 27.188 | 374 / 27.176 |

All 12 runs passed the independent geometry audit, including part identity, rotations, nominal dimensions, tool envelopes, trim bounds, overlaps, and utilization. Elapsed values include the diagnostic audit; internal search limits are not the full external deadline.

Complementary candidates were exercised on the first two cases, but did not reduce board counts relative to the corrected prefix control. On the 739-part case both methods reached 97 boards through baseline-mode prefix completion; a single observed event timestamp is not evidence of a speed advantage. On the 2416-part case warmstarts used approximately 26.94 s and no search simulations ran. This is an observed initialization bottleneck, not evidence against pair quality. No statistical quality or speed claim follows from three tasks and one seed.

## Deferred work

The first iteration did not implement staged completion, capped warmstart allocation, new repair scheduling, learned ranking, wider beams, or new action-level geometric diversity. The [2026-10-20 follow-up](warmstart-budget-20261020.md) now implements the initialization/search budget split; the historical diagnostic above still belongs only to its stated commit. Completion cost and other items remain deferred. Do not treat this follow-up as replacement formal evidence. Existing repair settings are inherited, not newly evaluated here. Further comparisons require matched total budgets and multiple seeds against Greedy and corrected warmstart-only controls, independently audited geometry, and frozen code/input identities.
