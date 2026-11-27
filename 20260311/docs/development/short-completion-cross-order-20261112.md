# Short-completion cross-order check — 2026-11-12

The fixed short-edge-primary candidate produces **no additional panel savings on either newly checked known order at30 s**. All three seeds tie at29 panels for178 pieces and372 for2416 pieces. The prespecified cross-order development signal fails; the earlier739-piece96→95 result remains a single-order positive example. Pause matrix expansion and new heuristics. Restricted-prefix search remains the topic; formal acceptance is not achieved.

## 1. Version and scope

Frozen checkout `5403f049c1e6a201a05c8d65a1f2723162fbe9b9`; production source/tests/scripts are identical to `810e43966dcb1b8292b2a9cf14184dce308f57ca`. This update adds protocol/aggregate documentation and a private experiment harness; **no production algorithm, default, dependency or heuristic change**.

Reuse `rh_mcts_prefix_short_completion_on/off`: identical Greedy→short-edge strong initialization and active completion tuple `(short_edge,long_edge,strip_bias)`. Only the existing search gate differs between arms. On starts the existing prefix procedure from an empty root; off returns the saved complete initial solution without search/idle padding. Candidates, scoring, staged quotas, adaptive tiers, depth/width/commit rules and geometry remain fixed; complement/type-fair/witness mechanisms stay off.

Two remaining already-used orders,178 and2416 pieces, were selected before outcomes. Neither is unseen validation. The previously positive739-piece order is not rerun or pooled into this batch's decision.

## 2. Locked design and decision

Two orders×three paired seeds1000/1001/1002×on/off=**12 runs, once each**. Six order_seed blocks and within-block arms randomized with seed20261116. Before outcomes, lock actual checkout/source/input/dependency/script hashes, historical initial references, schedule and analysis. No retuning, selective reruns or additional matrix.

Each run:30 s pipeline allowance including input loading/construction, both initial decodes, enabled search, validation, independent geometry audit and minimal serialization;3 s closing reserve;38 s absolute parent watchdog;8 s cleanup within that watchdog. Process startup is outside the internal timer and covered by the watchdog. Off returns early under the same maximum budget; equal consumed CPU time is not the design.

Initial panels/full score/candidate ledger/sequence/raw-layout receipts must match across arms and all seeds, and also the prelocked historical same-order common seeds29/372. Historical outputs verify initialization identity only, not treatment efficacy or timing. Off must return the initial solution unchanged and perform zero actual completion/search evaluations. On must actually evaluate short-edge primary completion.

Failed/incomplete/mismatched/protocol-invalid rows remain recorded and withhold the affected order's means. Valid-order descriptive means can remain when another order is invalid, but overall replication cannot pass. Signal requires both orders eligible, no per-seed regression and at least one strictly improved order mean. All ties pause escalation; invalid execution is inconclusive. Use three-seed order means, never best seeds or pooled panel-weighted quality. The formal gate is unchanged.

## 3. Every seed and material results

| Pieces | Seed | Off panels | On panels | Off total s | On total s |
| --- | --- | ---: | ---: | ---: | ---: |
| 178 | 1000 | 29 | 29 | 0.118 | 2.486 |
| 178 | 1001 | 29 | 29 | 0.119 | 2.592 |
| 178 | 1002 | 29 | 29 | 0.118 | 2.489 |
| 2416 | 1000 | 372 | 372 | 9.528 | 27.240 |
| 2416 | 1001 | 372 | 372 | 9.400 | 27.239 |
| 2416 | 1002 | 372 | 372 | 9.423 | 27.236 |

| Pieces | Mean panels off/on | Mean net U, both | Envelope occupancy, both | Net-U gain | Mean seconds off/on |
| --- | --- | ---: | ---: | ---: | --- |
| 178 | 29 /29 | 83.7932% | 85.5369% | 0pp | 0.118 /2.522 |
| 2416 | 372 /372 | 89.6355% | 91.3819% | 0pp | 9.450 /27.238 |

**0 task wins/2 ties/0 losses**, zero additional material savings and zero per-seed regression. Net U uses nominal product area/full purchased-board area. Envelope occupancy reserves `(w+6)*(h+6)` and is distinct from product yield/exact cutter-path loss. Board2440×1220 mm,5 mm trim/edge,6 mm allowance, permitted0/90° rotations unchanged; no excluded tail board or relaxed geometry.

Every off final sequence/layout/full score equals initialization. All six on final layout fingerprints differ from initialization: the search changes layouts and secondary quality, while panel count and global material utilization stay unchanged. Layout changes alone do not establish savings.

## 4. Execution diagnostics

Counts below follow seeds1000/1001/1002 and do not substitute for the material result.

| Pieces | On completed simulations | Distinct completed states | PUCT selections | short-edge attempts | long-edge /strip-bias attempts |
| --- | --- | --- | --- | --- | --- |
| 178 | 80/80/80 | 65/65/65 | 83/83/83 | 65/65/65 | 16/16 each, every seed |
| 2416 | 7/6/7 | 7/6/7 | 1/0/0 | 8/7/8 | 1/1 each, every seed |

The extra large-order primary attempt in each run expires at the search deadline; completed-state counts exclude it, and the complete incumbent survives. Initialization takes approximately0.096–0.107 s for178 pieces and9.161–9.292 s for2416 pieces. Large-order search therefore still completes few states. The small order also has no panel gain despite much more exploration. Retained beam depth is3 for178 and0 for2416; this is not a hard maximum for every evaluated leaf. `commit_steps=0` does not count all beam retention events. No isolated PUCT-necessity or throughput-causality claim.

## 5. Verification and acceptance

**12/12** clean exits and saved-output audits;6/6 eligible paired initial matches and both cross-seed/reference matches. Zero failures, protocol-invalid rows, floor violations or orphan saved files. Max post-serialization elapsed **27.239745541 s<30 s**. Every payload's request case/seed/profile/time limit, board/trim/tool/rotation settings, saved budget totals/reserve/deadlines and actual completion counters are bound to the locked protocol. Geometry/source/runtime/schedule/result/dependency identities pass. All12 final raw-layout fingerprints are reconstructed from exported placements/source/sequence, and both material metrics recomputed without packing replay. Floating material comparisons use1e-12 absolute tolerance for accumulation-order tail differences; raw fingerprints remain exact.

One fresh read-only review of the new private harness found3 Important gaps: skipped checks on saved failures/orphans, missing payload-to-protocol identity checks, and possible off actual-counter/aggregate-counter contradictions. All3 were accepted and fixed in one test-driven pass before outcomes. Added reproductions failed then passed; no second-review claim. Failed/orphan outputs are now reconciled/audited without promoting failed execution to success, actual off counters must be zero, and incorrect request/geometry/budget/accounting is rejected.

**41 new private tests**, plus25 reused private checks and347 runtime tests: **413 passed/3 existing optional CutEval skips**. Compileall/py_compile/diff checks pass. The unchanged production algorithm retains its previous independent review; this turn's fresh reviewer covers harness integration. Post-fix execution/audit/report/publication verification is by the executor. No remote-CI pass is assumed from an empty check list.

Execution/correctness pass; **cross-order material-gain criterion does not pass**. Formal acceptance remains not achieved. No formal420 s or unseen-order run, no concurrent area-primary comparator, no isolated component causal claim, optimality/SOTA/commercial superiority claim or master merge. Old results/failures remain unchanged.

## 6. Decision and next priority

Keep the opt-in candidate and the739 positive evidence, but do not promote defaults or expand the quality matrix on these two null results. Next priority is a bounded **read-only comparison of the saved739 saving path and these two no-saving search traces**, separating large-order throughput limits from candidate/completion coverage. That audit has not been launched here; new heuristics remain paused and the restricted-prefix theme stays.

Execution rulings: select the two remaining known orders and execute inline (limited scope/power, no unseen inference); review private harness before freezing and verify subsequent outcomes separately (executor post-fix verification only); accept the reviewer's exclusion of pending performance (no independent performance review), formal/unseen/PUCT claims (no broader inference), future publication/CI (verify remote receipts separately), and unchanged algorithm re-review (source identity plus prior review, no new algorithm review). No deferred minors. Public changes are protocol/aggregate docs only; private inputs, layouts, sequences, scripts/tests and detailed records stay local.
