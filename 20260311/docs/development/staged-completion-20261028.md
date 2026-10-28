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

Runtime regression checks: pending final verification. Diagnostic results: pending.
