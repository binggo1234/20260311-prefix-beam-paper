# Staged prefix completion — implementation and development check

User-approved scope (2026-10-28): implement factor A only. Reference C is `c32444d6110e66f4d39ab1cf5c4fdcf79cb5a1b8`. Do not change geometry, candidate generation, macro quota, prefix depth/width, warmstarts, score/reward, or repair settings. Keep the frozen attempt7 checkout/results unchanged. Update the existing draft PR only after verification; do not merge.

## First fixed policy (not tuned on the diagnostic)

- Add opt-in `rh_mcts_prefix_beam_staged`, inheriting the corrected standard prefix profile (not the complement profile). Existing variants retain the previous exhaustive completion policy.
- First evaluate each state with the first configured deterministic mode (`baseline` by default). Only a successfully completed suffix is usable. This is a complete packing but a primary-only evaluation, not a fully evaluated mode group.
- Each four completed primary evaluations grant one promotion credit. Use a credit when the primary result is within one board of the incumbent, when no incumbent exists, or when two unused credits have accumulated (exploration independent of that quality threshold). A cache hit does not create another credit. A promoted partial entry may resume without spending another credit.
- Promote by evaluating the remaining configured modes, preserving their order and the existing ranking. Reuse already completed modes; an interrupted mode is not recorded as completed. Only a group with every configured mode completed may enter the existing full-group cache.
- Primary/partial entries use a separate cache. On revisit, a primary-only entry can be reconsidered when credits become available. Honor the existing cache-disable flag. Reject staged+random completion rather than silently changing the random ablation.
- All replay/checkpoint work uses the unchanged search deadline. Preserve completed results on later-mode expiry; first-mode expiry still fails if no usable result exists. Record evaluation tier, promotions/exploration, modes completed, and distinct successfully evaluated states.
- Retain a new complete incumbent immediately after evaluation, including when promotion is triggered during beam ranking rather than a simulation. Do not perform optional archive work just to retain it.

The policy is a heuristic compute-allocation change, not a promise of matching the exhaustive evaluator on every state. It may miss a good mode on a screened-out prefix; the Greedy complete incumbent remains the safety floor. No claim of material savings follows from more evaluations alone.

Select `PackingRequest(..., solver_variant="rh_mcts_prefix_beam_staged")` through the existing budgeted handoff API to enable this policy. `rh_mcts_prefix_beam` remains the exhaustive control. There is no new dependency or handoff schema. Configuration defaults are `PREFIX_BEAM_STAGED_ENABLE=False`, `PREFIX_BEAM_STAGED_PROMOTE_EVERY=4`, `PREFIX_BEAM_STAGED_EXPLORE_CREDITS=2`, and `PREFIX_BEAM_STAGED_PANEL_MARGIN=1`; only the staged variant enables the flag. The existing `VARIANT_EXTRA_OVERRIDES` mechanism may override these fields, but no alternative setting was selected from this diagnostic.

## Implementation / checks

1. Write failing behavior tests for screening, credit quota, exploration, promotion on revisit, full/partial cache semantics, deadline preservation, real feasible output, and profile isolation.
2. Implement the opt-in completion path and additive evidence. Keep existing exhaustive logic and tests as a control.
3. Run targeted/full pytest, compileall and diff checks. Inspect the final diff for unrelated behavior changes.
4. Freeze a runtime commit before the bounded diagnostic, then document all results including regressions. Do not tune the policy after inspecting outcomes in this batch.

## Locked development diagnostic

Question: with the same 30 s total / 3 s closing reserve, does factor A evaluate more complete prefix states without breaking geometry, deadlines or the Greedy floor, and does it change board counts?

- Existing development tasks: 178, 739 and 2416 parts, taken from the unchanged source manifest. These are not unseen generalization data.
- Paired seeds: 1000, 1001, 1002. Methods: corrected prefix C and staged prefix C+A. Greedy is deterministic and is run once per task, not treated as three independent repetitions.
- Reproducible task×seed block ordering and within-block method randomization using schedule seed 20261028. Serial independent child processes on the same host; parent watchdog 38 s. No other benchmark workers run concurrently.
- Freeze input hashes, runtime commit, full settings, schedule and failure policy before execution. Save local progressive records; do not upload factory inputs/layouts/raw outputs.
- Independently audit part identity, rotations, nominal dimensions, tool envelopes, trim bounds, overlap, utilization and total elapsed. Record net utilization and envelope occupancy separately without changing the solver objective.
- Report every task's per-seed panels and task means, wins/ties/losses and all failures. Record successful distinct completion states, modes completed, simulations and actual PUCT separately. The three seeds are algorithm repetitions within a task, not independent production orders.
- An internal search cutoff with a usable complete result is not a failed run. Parent watchdog, no usable incumbent, serialization/audit timeout or invalid geometry is a failure; never impute or selectively rerun it.
- This is phase-B mechanism validation, not the formal 10-task×5-seed material-quality gate. The 120 s selection matrix and 420 s formal batch remain later stages; do not launch them automatically in response to disappointing results.

The separate entry contains only completed per-mode outputs plus a promotion flag. Full-group cache hits still mean all configured modes completed; `primary_only` and `promoted_partial` are explicit tiers. With caching disabled, repeated primary evaluations are real work and can create credits again, but no tier is stored. Distinct-state evidence uses the existing state key (layout hash, ordered hash, remaining mask, prefix depth), not a geometric-equivalence class of prefixes. Per-mode timers exclude suffix ordering in both paths; the inclusive completion timer includes that cost.

## Verified results (2026-10-28)

Frozen runtime: `e6a996a876c578dfc34a7f3001b2c07b4163642d`. C uses the `c32444d` exhaustive-completion policy on this same runtime, with additive instrumentation; C+A enables staging. Subsequent documentation-only commits do not change the tested runtime. Input identity, ordering and settings were locked before the first run. All 21 scheduled runs completed: 3 deterministic Greedy references and 18 paired method runs. There were no selective reruns or post-outcome policy changes.

Seeds in each cell below are ordered 1000 / 1001 / 1002. Because every seed within a task returned the same panel count, the shown utilization is also the task mean. Percentages use all used boards, including the tail board.

| Parts | Greedy boards | C boards | C+A boards | Net utilization C → C+A | Change, percentage points | Envelope occupancy C → C+A |
| --- | ---: | --- | --- | --- | ---: | --- |
| 178 | 29 | 29 / 29 / 29 | 29 / 29 / 29 | 83.7932% → 83.7932% | 0 | 85.5369% → 85.5369% |
| 739 | 98 | 97 / 97 / 97 | 97 / 97 / 97 | 89.3805% → 89.3805% | 0 | 91.3207% → 91.3207% |
| 2416 | 374 | 374 / 374 / 374 | 373 / 373 / 373 | 89.1561% → 89.3952% | +0.2390 | 90.8932% → 91.1369% |

Net utilization is `sum(w*h)/(N*W*H)`. Envelope occupancy is `sum((w+6)*(h+6))/(N*W*H)` for this unchanged rectangular reservation model, not measured cutter-path loss. Trim remains 5 mm, boards 2440×1220 mm, and rotations 0°/90°. Neither metric changes the optimization objective.

| Parts | Mean distinct successful states C → C+A | Mean simulations C → C+A | Mean actual PUCT C → C+A | Mean total seconds C → C+A |
| --- | --- | --- | --- | --- |
| 178 | 77 → 77 | 80 → 80 | 76 → 76 | 4.080 → 2.494 |
| 739 | 45 → 80 | 45 → 80 | 25.33 → 64 | 27.531 → 26.235 |
| 2416 | 4 → 8 | 4 → 8 | 0 → 0 | 27.227 → 27.227 |

The unchanged simulation cap is 80. The small task hits that cap in both methods: less elapsed time, not more evaluated states. The two larger tasks evaluate more states, but the largest still makes zero PUCT selections. Mean promotions are 19 / 20 / 2 for the three staged tasks. The first complete below-Greedy layout on the large task is retained at 21.919 / 21.642 / 21.475 s (solver-evidence origin, not external budget origin). More evaluations alone remain a mechanism measurement, not material savings.

All 21 results passed independent checks of source part identity, allowed rotation, nominal dimensions, tool envelopes, trim bounds, non-overlap and utilization; those checks were repeated against the saved results. Each result hash, runtime commit and source hash was rechecked. All minimal results were atomically serialized within the 30 s total budget; the largest observed elapsed time after write was 27.780888 s. Internal search cutoffs retained usable layouts and are not watchdog/total-budget failures. All method results were no worse than their saved exact Greedy floor. There were **0 failures, 0 of 9 paired board-count regressions**, and task-level **1 win / 2 ties / 0 losses**. The sum of task mean board-count changes is −1 (not a pooled utilization estimate or an independent-order significance test).

Local evidence is retained under the QA batch `staged_completion_20261028`: locked manifest, progressive records, all 21 result files and `verified_summary.json`, plus the private run/verification harnesses. Factory inputs, source identifiers, layouts and raw outputs are not added to this public repository.

## Regression verification and acceptance

From `20260311/`, Python 3.9.6:

```sh
python3 -m pytest -q tests/test_staged_completion.py
python3 -m pytest -q -rs
python3 -m compileall -q src tests scripts
git diff --check
```

The full suite passes **253 tests with 3 external-integration skips**. The existing dirty external CutEval checkout is preserved; those skips do not establish cross-chapter integration coverage. Regression coverage includes first-mode expiry, partial promotion resumption, cache opt-out, full-cache fidelity, revisited-state promotion, exploration quota, immediate ranking-time incumbent retention, real budgeted handoff, and unchanged small/medium/large profile and candidate generation settings. No new dependency is required.

Code correctness and scoped phase-B mechanism validation pass on these development tasks. This is also a reproducible short-budget board-saving signal on one known task, **not formal material-quality acceptance**. Only one of three tasks improves; the approved gate requires at least three main-task improvements, a strictly smaller sum of task means and wins greater than losses under the frozen formal matrix. The 10-task×5-seed formal matrix and unseen-order validation have not run. These seeds are within-task algorithm repetitions, not nine independent production orders. The old frozen solver already reached 373 on this large task with 420 s; this result is not a new unrestricted record or evidence of superiority at 420 s.

Remaining risk: screening can omit a mode that would improve a rejected prefix, and the Greedy floor cannot prevent regressions against exhaustive C on new tasks. Hard preemption inside a geometry operation is not added. Keep staging opt-in, retain the unchanged control, and do not widen/deepen search or enable extra repairs based on these outcomes. The next sole research priority is stage C, global candidate coverage as a separate factor, before the planned C / C+A / C+B / C+A+B selection; it is not implemented or launched in this change.
