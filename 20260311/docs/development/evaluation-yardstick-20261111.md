# Same-order utilization yardstick — 2026-11-11

The current restricted-prefix staged method loses to ordinary whole-order references on two of three known orders at the same 30 s budget. This establishes a concrete attainable reference, not a field-wide utilization threshold or a verdict that restricted-prefix search is invalid. No restricted-prefix algorithm or default was changed in this work.

## 1. Version and implementation scope

Frozen experiment runtime: `120d29436f27446fc5e92441840284d0159104d9`. Implementation commits: `a531bf6` (isolated references/bounds), `09c00d8` (exact arithmetic and envelope SA energy), `120d294` (identity-safe reference micro-cache). Later report commits do not change runtime sources.

- `baseline_order_multistart`: exact canonical Greedy, up to six distinct deterministic alternatives (long edge, short edge, perimeter, aspect, width, height), then area-biased randomized whole-order rankings.
- `baseline_sequence_sa_budgeted`: the same deterministic portfolio, then global swap, relocation, reversal and block relocation; energy is minus panels plus 0.01 times mean squared envelope occupancy, temperature 0.25 cooled by 0.995 per completed SA proposal with floor 0.01. Best complete solution is retained independently of the accepted current state.
- Both allow at most **6000 stochastic proposal attempts after the deterministic portfolio**. No-op proposals also consume attempts. Both share staged placement rules and adaptive decoder settings; neither executes restricted-prefix MCTS. These are reproducible ordinary references, not implementations claimed to be SOTA.
- Every replay receives the absolute search deadline; an interrupted optional candidate retains the saved complete incumbent, with no final replay. An incomplete initial Greedy is a failed run. Legacy `baseline_sa` is preserved and is not used here.
- New references append UID to micro-cache keys, because cached post-placement hashes include UID. This changes cache sharing, not placement/scoring rules. The research control's existing cache implementation was deliberately left unchanged; its identity issue is a correctness risk to address separately before formal evaluation.
- `packing_area_bounds(request)` uses exact rational arithmetic over decimal input representations. It reports nominal and envelope area bounds, their maximum, and the implied nominal-utilization ceiling. It rejects unfit/invalid geometry.

## 2. Locked comparison and budget

Three known orders (178/739/2416 parts), seeds 1000/1001/1002, staged and the two references, plus one deterministic Greedy per order: **30 runs**. Seeded task×seed blocks and within-block method order, schedule seed **20261112**. Experimental unit is the order; seeds describe algorithm variability, not independent orders.

Each run has 30 s total including loading/construction, solver, validation, independent audit and minimal serialization; 3 s closing reserve; 38 s absolute parent watchdog; 8 s cleanup grace bounded by that watchdog. Algorithms may finish before their allowance; evaluation counts and consumed time are not forced equal. Hardware/environment, source/input/script hashes and the complete schedule were frozen before this replacement batch. No policy retuning or selective reruns.

Geometry: full board 2440×1220 mm, trim 5 mm per edge, tool clearance 6 mm, permitted 0°/90° rotations as supplied by each part. Envelope dimensions are `(w+6, h+6)` within the trimmed board. No shared-cut or exact cutter-path area claim.

## 3. All seeds and utilization results

Entries within a cell follow seeds **1000 / 1001 / 1002**. Greedy was run only once per order.

| Parts | Greedy panels | Staged panels | Multistart panels | Portfolio + SA panels | Staged mean net U | Multistart mean net U | SA mean net U |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| 178 | 29 | 29 / 29 / 29 | 29 / 29 / 29 | 29 / 29 / 29 | 83.7932% | 83.7932% | 83.7932% |
| 739 | 98 | 97 / 97 / 97 | 96 / 96 / 96 | 96 / 96 / 95 | 89.3805% | 90.3116% | 90.6285% |
| 2416 | 374 | 373 / 373 / 373 | 372 / 372 / 372 | 372 / 372 / 372 | 89.3952% | 89.6355% | 89.6355% |

Relative to multistart, staged has **0 task wins / 1 tie / 2 losses**, sum of task-mean panel differences **+2**. Relative to portfolio+SA, it has **0 / 1 / 2**, sum **+2⅓**. The medium-order SA mean is **95⅔ panels**, distinct from its best single seed of **95 panels / 91.2622%**. Mean utilization is the mean of per-run ratios, not the ratio formed from mean panel count.

| Parts | Staged envelope occupancy | Multistart envelope occupancy | SA mean envelope occupancy |
| --- | ---: | ---: | ---: |
| 178 | 85.5369% | 85.5369% | 85.5369% |
| 739 | 91.3207% | 92.2720% | 92.5958% |
| 2416 | 91.1369% | 91.3819% | 91.3819% |

Net utilization is `sum(w*h)/(N*W*H)`; envelope occupancy is `sum((w+g)*(h+g))/(N*W*H)`. The latter includes reserved clearance and is not effective product yield.

Attribution matters: all three multistart results on the medium and large orders are exactly the **short-edge-descending** order. The large portfolio+SA runs also return that exact order and complete **zero SA proposals**: their six completed evaluations are the deterministic portfolio, not evidence of annealing effectiveness. The medium 95-panel seed completes 89 SA proposals and reaches 95 panels at solver-local elapsed 7.388864 s, after first reaching 96 through the portfolio. Its success is one seed, not universal stability. No new attribution runs were needed; saved sequence identities and counters establish this.

## 4. Bounds and the resulting assessment standard

Let nominal area be A, inflated-envelope area E, full-board area B and trimmed usable area C. The model-specific panel relaxation is `LB = max(ceil(A/B), ceil(E/C))`. Its implied net ceiling `A/(LB*B)` may be unattainable because shape/rotation constraints remain relaxed.

| Parts | Nominal-area LB | Envelope-area LB / final LB | Best audited feasible panels in this batch | Best net U | Net U ceiling from LB | Best gap to area LB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 178 | 25 | 26 | 29 | 83.7932% | 93.4617% | 11.5385% |
| 739 | 87 | 90 | 95 | 91.2622% | 96.3324% | 5.5556% |
| 2416 | 334 | 345 | 372 | 89.6355% | 96.6504% | 7.8261% |

The last column is `(best feasible N − LB)/LB`, **not the true optimality gap**. No optimum is proven. “Best” means sampled, independently feasible, in this batch; it is not a verified external BKS. Same-order commercial and verified-BKS entries remain **unavailable/null** for all three orders.

A uniform **95% net utilization** gate is invalid: the 178-piece model cannot exceed even the optimistic 93.4617% ceiling. Cross-paper percentages also do not supply a gate: the 2026 furniture-remnant study reports different orders/constraints ([official paper](https://www.sciengine.com/JFE/doi/10.13360/j.issn.2096-1359.202412001)); the insulation result permits splitting pieces ([CP 2025 paper](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CP.2025.7)). They provide context, not same-input competitors.

Use the following yardstick: valid complete geometry and lifecycle first; same-order panel count/net U against a strong simple reference second; a relaxation bound as an explicitly optimistic diagnostic; formal multi-order/seed and unseen-order evidence before general claims. The previously approved formal acceptance gate is unchanged. Current staged results do **not** establish additional quality over these references.

## 5. Failures, review and verification

The first runtime (`09c00d8`) was **aborted**, not silently replaced: row 6 failed identity validation (two missing and two duplicate parts). Ten parent records remain (nine passed, one failed), row 11 was interrupted, rows 12–30 were unexecuted. Those partial outcomes are excluded from comparative conclusions. Original scripts, locked manifest, progress and saved outputs remain local.

Root cause was reproduced with three synthetic parts: cached geometry-equivalent micro results contained a UID-dependent post-placement hash; replaying `(1,3,2)` then `(2,3,1)` could restore `[1,3,1]`. The isolated reference decoder now keys these results by UID. Two new regression cases failed before the fix and pass afterwards. The replacement batch reran **all 30 rows**, same schedule, budgets and search policies, under a new identity. This is a correctness replacement, not selective outcome rerunning.

Replacement batch: **30/30 passed**, clean worker exits; all 30 saved layouts reaudited independently with runtime/source/input/result/schedule hashes checked; all 30 area ceilings independently recomputed; no orphan output, Greedy-floor violation or incomplete comparison. Maximum post-serialization total elapsed **28.172176167 s < 30 s**. Zero corrected-batch failures does not erase the aborted batch's failure.

Code validation: **22 targeted evaluation tests**, **311 full-suite passes / 3 existing skips**, compileall and diff checks. The skips are the existing optional minibatch integrations requiring a clean external CutEval checkout; user modifications were preserved. Nine private checks cover allocation, incomplete/failure summaries and the reused worker lifecycle.

One fresh reviewer examined `b577f516..a531bf6` and found two Important issues: default Decimal precision could overstate the bound; SA incorrectly reused the decoder's nominal-area/gamma statistic. Both received failing regressions, exact fixes and a green whole suite. Subsequent identity isolation was RED→GREEN tested and self-verified, not independently re-reviewed. No remote CI claim.

Deferred review coverage: no deterministic forced-worse-SA retention test; no ≤60-piece adaptive-tier regression (the three experimental tiers are checked). Proposal-count wording is clarified here: ≤6 distinct deterministic alternatives plus 6000 stochastic attempts.

## 6. Decision and next priority

The evaluation yardstick is implemented and this diagnostic comparison is complete. Restricted-prefix material-quality acceptance is **not passed**: current staged trails ordinary full-order references on two known orders. These reference layouts are not counted as gains of the restricted-prefix method. No formal 420 s matrix, unseen-order validation, algorithm-default change or master merge occurred.

Next sole implementation priority: repair and regression-test the **shared decoder's part-identity cache contract** separately before any formal comparison. Then keep short-edge ordering/whole-order references as the quality floor when testing additional restricted-prefix benefit. Pause further candidate/commit heuristics. The theme stays restricted-prefix; the 95-panel and 372-panel feasible layouts provide concrete targets, while the 178-piece 28-panel target remains unproven feasible.

Execution rulings: carry out the user-approved plan inline (reference choices may require later revision); restrict UID isolation to the new references (shared-cache risk remains and baseline throughput can differ); retain the aborted batch and repeat every row after the correctness fix (extra computation, no result erasure); exclude historical work from the fresh review and validate this new batch independently (claims are limited to this audited batch and must be withdrawn if its evidence fails). Factory inputs, layouts and detailed evidence stay private.
