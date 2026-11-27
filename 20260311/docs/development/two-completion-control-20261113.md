# Two-completion early-prefix control — 2026-11-13

On the known739-piece order, the existing two-single reference reaches **95 panels**, while current search truncated to the same two complete evaluations retains **96 panels**. Both use identical complete strong initialization, initial RNG/cache state and the same total time limit. The pair passes independent receipt/geometry audits. Current search has no early material advantage in this pair; the restricted-prefix witness remains valid. Neither arm performs a PUCT selection within these two evaluations, so this is not a PUCT ablation.

## 1. Version and implementation

Frozen runtime checkout `57d4672b524d3b62eca3ba672bb2acf9c57c10ec`, built from `8ae915b`. The protocol was prepared Nov12 and executed Nov13; private protocol/script names retain Nov12. New opt-in [experimental adapter](../../src/inrp/early_prefix_control.py) exposes `solve_two_completion_handoff(request, budget, arm)` and reuses `rh_mcts_prefix_short_completion_on`. Existing `mcts.py`, `cfg.py`, `runner.py` and `handoff.py` remain identical to frozen `e716957`; no default/profile, geometry, candidate, scoring or heuristic change.

The reference calls existing `_single_macro_actions(empty_root_remaining, 2)` once and evaluates those two entries in returned order using existing short-edge completion. There is no witness identity/position override. The search arm uses existing search/selection until its allowance ends. Both retain the common incumbent and use existing close without reconstruction replay, SA or optional repair. Actual snapshots are retained and exported under the absolute deadline, including every normally returned final incumbent even when initialization is incomplete.

## 2. Frozen scope, counts and acceptance

Exactly one known-order pair, seed1000. Schedule seed20261116 orders current search first, reference second. Input/reference/source/script/dependency identities and schedule are frozen before outcomes. No retries, tuning, new orders/seeds or matrix expansion.

Each independent arm uses **30 s pipeline,3 s closing reserve,38 s parent watchdog,8 s bounded cleanup**. The two common Greedy/short-edge full replays are separately counted; after initialization, at most **two full completion starts**, including failed starts, are permitted. Candidate generation, partial-prefix placement and cache/ranking work consume wall time. Cache reuse without a full replay does not spend a credit. No work is added to fill unused time or discover an extra solution after exhaustion. Two credits do not reach the unchanged fourth-primary promotion.

Acceptance requires identical complete initial sequence/layout/full score/candidate panel ledger, initial RNG/cache keys/counts, two successful post-initial completions per arm, valid captured/final geometry and clean output within budget. Quality is measured, not a hardcoded95 acceptance target. Credit exhaustion is distinct from replay timeout or absolute close expiry. The pipeline includes input/preparation, initialization, evaluation, evidence, validation, checkpoint/audits and serialization/hash; startup/runtime imports are covered by the parent watchdog.

## 3. Material results and actual early evaluations

| Item | Two existing singles + short-edge | Current search, two completions |
| --- | ---: | ---: |
| Common Greedy / best initial panels | 98 /96 | 98 /96 |
| Post-initial full starts / successes | 2 /2 | 2 /2 |
| Panels at evaluation1, evaluation2 | **96,95** | 96,96 |
| Actual prefix piece counts | 1,1 | 4,6 |
| Actual action depths | 1,1 | 1,1 |
| Final panels | **95** | 96 |
| Net material utilization | **91.2622%** | 90.3116% |
| Reserved-envelope occupancy | 93.2433% | 92.2720% |
| Simulations / PUCT selections | 0 /0 | 2 /0 |

The reference saves one panel relative to the common initial floor and the truncated search: **+0.9506483 percentage points net utilization**. Its second evaluation exactly matches the historical first-saving sequence and raw layout digests. The search's two recorded full orders do not match that witness. Both end as `completion_quota`, not timeout; both evaluate only `short_edge`, twice. The current early schedule spends its two evaluations on4-piece and6-piece prefixes instead of the covered saving single; macro kinds and complete noisy-prior histories are not inferred from prefix length.

Boards2440×1220 mm,5 mm trim,6 mm tool allowance and0/90° rotations are unchanged. Envelope occupancy counts reserved rectangular clearance and is not net product yield or exact cutter-path loss. A secondary-score improvement at96 panels is not a material-utilization gain on this fixed input.

## 4. Timing and independent checks

| Descriptive timing | Single reference | Truncated search |
| --- | ---: | ---: |
| Common initialization | 0.796080041 s | 0.804630125 s |
| Handoff runtime, before independent audits | 1.310241250 s | 1.247504083 s |
| Complete pipeline through serialization/hash | 1.557467875 s | 1.401793333 s |

This is one pair, with budgeted diagnostic/export and audit work. The reference also receives a separate pure source-arithmetic entry-order audit. These observations support budget compliance, not a controlled speed advantage. No time limit was extended or reset.

Each arm's best initial, two evaluated and final geometry records is independently checked against the source, totaling8 logical audit records, including repeated final layouts. Checks cover exact part identity/multiplicity, dimensions, permitted rotation, trim/boundaries and tool-clearance non-overlap. Raw-envelope→nominal placement bindings and complete sequence/layout digests pass; secondary scores are independently recomputed within floating tolerance. Final results equal the best complete initial/evaluated score. Initial identities also match the frozen historical96-panel reference; historical results are identity checks, not concurrent performance controls.

Saved records/handoff/provenance, mode starts/successes, runtime/schema/feasibility, frozen progress/version and deadline relationships agree. Both workers exit cleanly; no failed, incomplete or orphan outcome is counted as a comparison. Frozen code/dependencies,497 historical JSON files and both protected workspaces remain unchanged.

## 5. Tests and review

17 new runtime checks cover real-decoder2+2 replay counts/no third replay, common RNG/cache equality, exact existing reference entries, parity with original first-two search evaluations, cache reuse, unsupported configuration, partial initialization, replay errors/timeouts, search-boundary quota close and absolute close expiry. Full runtime: **387 passed/3 existing optional CutEval skips**.32 private schedule/evidence/corruption/failure checks plus5 lifecycle checks pass; combined **424 passed/3 skipped**. Compilation and whitespace checks pass.

One fresh read-only review independently ran the pre-fix404/3 suite and found3 Important issues: quota classification at the second-success/search-deadline boundary, loss of completed evidence on exceptions, and missing final receipt/bindings after partial initialization. The normative-envelope/protocol gap initially graded Minor was elevated to Important because invalid feasibility/schema/runtime could otherwise count as eligible evidence. All were fixed before factory data in one RED→GREEN pass; relevant regressions reproduced the faults. No second independent review is claimed.

Returned handoffs are checkpointed before independent audit; failures retain existing file bindings, counts/stage and exports created while the deadline remains open. Offline inspection audits failed receipts but excludes failed executions. If the absolute deadline prevents new export/write, raw unavailability is explicit. The implementation does not promise complete recovery after hard process/OS failure. This pair exercised normal successful paths; failure behavior is established by toy/fault-injection/lifecycle tests. No remote-CI pass claim.

## 6. Interpretation and next priority

The existing entry rule and simple restricted-prefix policy already recover this known95-panel witness within two completions. Current search's initial evaluation schedule does not beat that reference on this order/seed/budget. This raises the relevant comparison baseline beyond plain short-edge96; reproducing95 alone is not evidence of additional value from a more complex search.

Freeze this simple reference as a strong control for future claims and continue pausing new heuristics. Any further claim about material gain from the current search must compare against it under a declared budget; no additional matrix or mechanism is launched here. Restricted-prefix topic remains unchanged. This is not a judgment of longer-search value: the earlier longer739 runs also reached95 but were separate batches. One known-order pair cannot establish unseen/cross-order mean gain, optimality, PUCT necessity or a formal quality pass. Earlier cross-order ties and the unpassed formal gate remain unchanged.

Execution rulings, in order: reuse the authorized isolated branch/draft (visibility limited to that development branch); count started completions including failures, initialization separate and other work wall-budgeted (conclusion depends on this accounting); review implementation before outcomes and executor-audit later artifacts (no independent outcome review); elevate invalid normative/protocol evidence to Important and fix (additional audit maintenance); verify the formerly unjudged factory pair through executor receipts (no independent performance review); exclude other orders/seeds/long-search/optimality/causal claims (untested); verify publication separately and never treat empty checks as CI success (no independent report/publication review); support budgeted checkpoints/counts rather than guaranteed full hard-crash recovery (some failed raw receipts may be unavailable); use full regressions and source-based geometry audits rather than re-proving the entire original solver (no global correctness proof). No deferred minors.

Related preserved evidence: [candidate coverage audit](single-candidate-path-audit-20261112.md), [first-saving receipt](first-saving-receipt-20261112.md), [cold reproduction](cold-prefix-replay-20261112.md). Public code/tests/protocol/aggregate report only; raw factory identities, dimensions, orders/layouts and detailed private run scripts/receipts remain local. Existing draft only; no merge to master.
