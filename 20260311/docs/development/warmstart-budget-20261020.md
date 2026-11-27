# Preserve prefix-search time with a bounded warmstart stage

This follow-up implements the user-approved initialization budget split. The earlier 2416-part diagnostic exhausted approximately 26.94 s in warmstarts and completed zero simulations under a 30 s total / 3 s closing-reserve budget. This change limits optional initialization, not the total solver budget or the decoder's objective.

## Policy

At warmstart entry, let `available = search_deadline - now`. The local allowance is `min(available * 0.20, 5.0)` seconds, measured from that same entry time. There is no additional allowance after Greedy finishes.

1. Decode the exact canonical Greedy sequence using the existing search deadline. Retain a complete result without invoking optional elite/layout/pattern archives.
2. Greedy may exceed the local allowance, but never receives permission to exceed the search deadline. If it already used the allowance, skip all optional registration/restarts and yield to search.
3. Otherwise use the local deadline for archive/pattern registration, suffix-cache priming, metadata-ranked initial solutions, further constructive packing, and replay normalization.
4. When local work expires, preserve complete incumbents and valid partial cache entries. Resume prefix search using the original, unchanged search deadline and closing reserve. Local expiry does not itself set `solver_timeout_hit`; later search expiry can still set that flag in the final result.

The canonical Greedy floor remains conditional on its initial replay completing. Extremely short budgets or sufficiently expensive single completions may still prevent a completed search simulation. This policy is not a universal guarantee of search progress for every input/budget.

## Configuration and scope

| CFG field | Default | Meaning |
| --- | ---: | --- |
| `PREFIX_WARMSTART_BUDGET_ENABLE` | `True` | Apply the local allowance only when prefix-beam mode is enabled. Set false for the legacy-allocation control. |
| `PREFIX_WARMSTART_BUDGET_FRAC` | `0.20` | Fraction of available search time at initial-stage entry; finite and in [0, 1]. |
| `PREFIX_WARMSTART_BUDGET_MAX_S` | `5.0` | Finite, nonnegative cap in seconds. Zero skips optional work after retaining Greedy. |

These fields can be set on CFG or through the existing runner's `VARIANT_EXTRA_OVERRIDES`. Both corrected prefix and complement prefix receive the same default allowance. Non-prefix variants keep their previous allocation. An unbounded search still caps optional initialization at 5 s; its unbounded remaining-time evidence is JSON `null`, not infinity.

No new dependency, handoff schema, geometry rule, objective, completion policy, beam width, or repair policy is introduced. Frozen attempt7 evidence and its source checkout remain unchanged.

## Evidence

The additive fields live in `handoff.provenance["solver_meta"]`:

- `warmstart_budget_enabled`: whether the prefix-specific local allowance applies.
- `warmstart_budget_s`: computed allowance when enabled, including time used by mandatory Greedy; it is not a hard cap on mandatory Greedy.
- `warmstart_budget_hit`: optional initialization hit the local deadline while search time remained.
- `warmstart_stop_reason`: `completed`, `warmstart_budget`, or `search_deadline`.
- `search_budget_remaining_s`: remaining search time when warmstarts exit.
- `prefix_search_budget_at_entry_s`: remaining search time at actual prefix-search entry, after root warm-start bookkeeping.

Existing `phase_time_s["warmstart"]` measures actual inclusive time. Existing warmstart kept/cache counts are refreshed on interrupted exits, so partial valid caches are not reported as empty. Complete optional candidates are retained even when registration cannot start before the local cutoff.

The local deadline is propagated through optional packing/replay and archive helpers. Checks occur between decoded placements, archived boards/layouts, and suffix-cache entries; there is no hard preemption inside an individual legacy geometry operation, signature/sort, or pattern insertion. Small checkpoint overruns are possible. Absolute total-deadline validation and the existing outer watchdog remain necessary for formal execution.

## Verification

Runtime commits: `fcfc394` introduces the split; `484367f` addresses independent-review findings about optional archive side effects, partial counters, and last-restart timeout attribution.

From the active project root, Python 3.9.6:

```sh
python3 -m pytest -q tests/test_warmstart_budget.py
python3 -m pytest -q -rs
python3 -m compileall -q src tests scripts
git diff --check
```

The budget suite has 17 passing cases. The full suite has 234 passing tests and 3 skips caused by the existing dirty external CutEval checkout; user changes in that checkout are preserved. New coverage exercises fraction/cap selection, mandatory Greedy longer than the local cap, actual prefix-search entry, extra replay and packing expiry, suffix and archive/pattern registration, zero budget without optional archives, retained improved complete candidates, partial counters, opt-out/non-prefix controls, unbounded time, invalid configuration, and mandatory-replay global expiry.

The independent review's blocking archive finding was reproduced with a failing zero-budget test and a three-board archive test, then fixed. Two evidence findings were likewise reproduced and corrected. Review found no remaining blocker in that patch; this is not a recommendation to merge the prerequisite runtime import or to claim external cross-chapter integration coverage.

## Same-input, same-budget diagnostic

The final runtime at `484367f2459481cd582c27037edd644baf994e72` was tested on the same previously stalled 2416-part task, seed 1000, with a 30 s total budget and 3 s closing reserve. Runs were serial in the order shown; the two standard-prefix runs differ only in the cap flag. The local harness uses the existing config/solver/conversion path for all methods and independently audits geometry against the original source. Source identity matches the previous diagnostic. Raw input, layouts, source hashes, detailed events, and harness remain in local QA storage outside the repository.

| Method | Warmstart s | Search s remaining at entry | Completed simulations | Boards | Total elapsed s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact Greedy | — | — | — | 374 | 3.604 |
| Corrected prefix, cap off | 26.941 | 0.000 | 0 | 374 | 27.175 |
| Corrected prefix, cap on | 5.000 | 21.942 | 4 | 374 | 27.188 |
| Complement prefix, cap on | 5.000 | 21.941 | 4 | 374 | 27.187 |

All four runs passed independent part-identity, rotation, nominal-dimension, tool-envelope, trim-bound, non-overlap, and utilization checks within the total budget. Total elapsed includes input loading and the audit. Both capped runs report `warmstart_budget_hit=true` and stop reason `warmstart_budget`; their final `solver_timeout_hit=1` arises later when search reaches its own deadline, not at the initial-stage cutoff.

This meets the scoped acceptance criterion on the stalled task: actual simulations now finish, complete outputs remain feasible, and board counts do not worsen relative to exact Greedy. It does not demonstrate additional board savings or a statistical speed improvement. PUCT selections remain zero; the runs expand/evaluate candidates but do not yet reach a PUCT selection. Complementary actions were selected 10 times in the capped complement run (cumulative across candidate-generation calls, not ten distinct states).

Completion consumed approximately 21.84 s / 21.81 s in the capped standard/complement runs. These are inclusive nested timers, not additive wall-clock phases. Tiny changes in secondary floating-point scores or event ordering are not board-count improvements. An earlier preliminary run at `fcfc394` also produced four completed simulations, but the table above uses only the post-review runtime.

## Interpretation and next scope

The success criterion is a complete, independently validated layout no worse in board count than exact Greedy, with actual search simulations completed within the unchanged total budget on the previously stalled task. It is not additional board savings or a statistical speed/quality claim.

Staged completion remains separate work: leaving time for search does not make each multi-mode completion cheap, and a handful of completed simulations need not exercise PUCT selection. A subsequent evaluation should distinguish completed simulations, PUCT selections, candidate generation, and completion-mode time under matched budgets rather than infer them from the presence of a search timer alone.
