# Fixed single reference for restricted-prefix research — 2026-11-13

Project decision: retain **the existing two empty-root singles followed by canonical short-edge completion** as a fixed strong reference, using arm `single_reference`. The research topic remains restricted prefixes. New heuristics remain paused. This decision fixes a comparison policy; it does not change the default solver or authorize another experiment.

## 1. Frozen definition and implementation

Implementation anchor: runtime `57d4672b524d3b62eca3ba672bb2acf9c57c10ec`, [experimental adapter](https://github.com/binggo1234/20260311-prefix-beam-paper/blob/57d4672b524d3b62eca3ba672bb2acf9c57c10ec/20260311/src/inrp/early_prefix_control.py), profile `rh_mcts_prefix_short_completion_on`, arm `single_reference`. Existing MCTS/configuration/runner/handoff implementation remains unchanged from `e716957` in that runtime.

1. Run the existing Greedy and short-edge common initializer, retaining the best complete initial incumbent.
2. At the empty root, call existing `_single_macro_actions(full_remaining_mask, 2)` once. Preserve the returned two-entry order. These are the existing single-entry candidates, not the first two parts of the short-edge order.
3. Independently place each returned single at the empty root and complete its remaining order with existing canonical `short_edge`. This is two one-action alternatives, not a combined two-piece prefix.
4. Retain the best complete initial/evaluated incumbent and close without another packing replay or optional repair.

Do not hardcode the known saving part, its dimensions or its position. Do not replace the existing entry ordering, select the best two after inspecting new outcomes, add completion modes or refill the allowance. Changes to this definition or source anchor require an explicit new protocol/version; preserve this reference and its evidence.

## 2. Evaluation and time accounting

The frozen early-budget comparison uses the following convention:

| Item | Fixed convention |
| --- | --- |
| Common initialization | Two full replays, separately counted; identical for both arms |
| Post-initial allowance | At most two started full completion replays per arm |
| Failed/expired replay | Consumes a start; no replacement attempt |
| Candidate/local placement/cache/ranking work | Included in wall time; no full-replay credit unless a full replay starts |
| Initial floor | Retain the best complete initial solution |
| Per-arm pipeline / closing reserve | 30 s /3 s |
| Parent watchdog / bounded cleanup | 38 s /8 s |

Complete common initialization and exactly two successful post-initial completions per arm are required to accept this paired early-budget comparison. Match the input, geometry/rotation/clearance semantics, full initial identity/score, initial RNG/cache state and total time convention. Report actual starts, successes, time and stop reason; do not fill unused time. Preserve failures and exclude incomplete/invalid pairs from quality conclusions.

This fixes the existing two-evaluation protocol. It does not prescribe a budget for an unimplemented longer-search or PUCT experiment; those require a separate design and declared comparison conditions.

## 3. Material metric and comparison rule

For the same order, board dimensions and constraints, compare panel count and **net material utilization** first. Report reserved-envelope occupancy as an auxiliary measure of the rectangular cutting-clearance allowance; it is not net product yield or exact cutter-path loss. A secondary-score change at the same panel count is not a net material-utilization gain on this fixed input.

Any additional material-gain claim for a more complex method must include this reference under the declared comparison conditions. Beating plain short-edge initialization alone does not establish additional value beyond this stronger reference. Keep secondary layout quality and descriptive timing visible, without relabeling them as material gain.

## 4. Established evidence and interpretation boundary

The accepted [739-piece seed1000 pair](two-completion-control-20261113.md) produced:

| Result | Fixed single reference | Current search, two completions |
| --- | ---: | ---: |
| Common initial panels | 96 | 96 |
| Evaluation panels | 96,95 | 96,96 |
| Final panels | **95** | 96 |
| Net utilization | **91.2622%** | 90.3116% |
| PUCT selections | **0** | **0** |

The reference's second complete order/raw layout matches the independently cold-reproduced saving witness. The definition is fixed after known-order development evidence; its generality has not been established on unseen orders.95 is an observed result on this order, not a universal acceptance target or an optimality certificate.

Both arms performed zero PUCT selections. The pair therefore does not assess PUCT necessity, PUCT benefit or longer-search value. Earlier longer-search results and cross-order ties remain separate evidence; the formal quality gate remains unchanged and unpassed. Future claims must name their order/seed/budget scope rather than extrapolate from this pair.

## 5. Current work status

The strong reference is fixed for research comparisons. New heuristics, additional search mechanisms, expanded matrices and formal runs are not started by this decision. Defaults, runtime code, geometry and prior evidence remain unchanged. The active topic is still restricted prefixes. Questions about PUCT or longer search remain open until separately authorized evidence addresses them.

## 6. Documentation verification

This update adds this agreement only. Verify its definitions against the existing adapter/source anchor and saved pair, repeat saved receipt/geometry checks without solver construction or replay, and verify source/evidence/workspace identities plus documentation links/whitespace. No new factory run, packing replay, search or candidate evaluation; no runtime-suite rerun or new CI-success claim for this documentation change. Raw factory inputs, identities, dimensions, sequences and layouts remain private. Publication uses the existing authorized draft branch; master is not merged.
