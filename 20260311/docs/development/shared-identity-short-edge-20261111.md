# Shared decoder identity repair and short-edge control — 2026-11-11

The shared decoder's identity-cache defect is repaired. With the corrected decoder used by every method, the current staged restricted-prefix system still trails deterministic short-edge ordering on two of three known orders. Correctness passed; additional material-utilization benefit over this strong reference was not demonstrated. No new search heuristic was added.

## 1. Runtime and exact change

Frozen runtime: `26c7fdf545cbd4e1d6633a4aa9fe6b932b1f7b47`, from `02bf332`. Later reporting commits do not alter runtime sources.

`OmnipotentDecoder._micro_key` now includes the requested part UID. Previously, shape-equivalent parts could share a `MicroDecodeResult`, whose `board_hash_after` nevertheless contained the first part's identity. This could contaminate layout hashes and cause a complete-cache replay to restore missing/duplicated identities. The three-part order pair `(1,3,2)` then `(2,3,1)` reproduced `[1,3,1]` before repair. The shared class now owns the same safe key semantics previously confined to evaluation references; that redundant subclass has been removed.

Existing placement/scoring rules, geometry, candidate selection, prefix depth/width, initialization policy, staged completion, commit policy and default flags are unchanged. Same-part/state micro and complete-cache reuse remains valid. The cache fix can change cache hits, intermediate states and throughput; no equivalence to the buggy runtime or universal performance gain is claimed.

Add `baseline_short_edge` through runner and budgeted handoff. Its exact order is nominal short edge descending, nominal area descending, UID ascending, matching the preceding evaluation portfolio. It uses the same corrected decoder and the existing deterministic replay/deadline/delivery path. The research system retains its canonical area-order Greedy floor; **short-edge was not inserted into its warmstart or completion policy**.

## 2. Locked concurrent comparison

Three known orders: 178/739/2416 parts. Per order: canonical Greedy once, short-edge once, staged prefix seeds 1000/1001/1002. Total **15 runs**. Task-block order and within-block method/seed order were shuffled with fixed seed **20261113**. Deterministic references were not repeated as independent seeds.

All methods use a 30 s end-to-end pipeline allowance including input loading/construction, solving, validation, independent audit and minimal serialization; 3 s closing reserve; 38 s absolute parent watchdog; 8 s cleanup grace bounded by that watchdog. Python process startup is outside the internal pipeline timer and subject to the parent watchdog. Algorithms may finish early. Runtime, inputs, audit/worker dependencies, runner/summarizer and allocation hashes were frozen before outcomes. No selective reruns or policy tuning.

Geometry remains 2440×1220 mm boards, 5 mm trim per edge, 6 mm inflated-envelope tool clearance, and supplied permitted rotations. Raw orders/layouts stay local. These three previously inspected orders are development cases, not unseen validation.

## 3. All outcomes and both material metrics

Prefix cells show seeds **1000 / 1001 / 1002**; deterministic columns are one run each.

| Parts | Greedy panels | Short-edge panels | Prefix panels | Short-edge net U | Prefix mean net U | Prefix − short-edge panels | Prefix − short-edge net U |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 178 | 29 | 29 | 29 / 29 / 29 | 83.7932% | 83.7932% | 0 | 0 pp |
| 739 | 98 | 96 | 97 / 97 / 97 | 90.3116% | 89.3805% | +1 | −0.9310 pp |
| 2416 | 374 | 372 | 373 / 373 / 373 | 89.6355% | 89.3952% | +1 | −0.2403 pp |

Task-level prefix outcome: **0 wins / 1 tie / 2 losses**, sum of task-mean panel differences **+2**. No prefix seed beats short-edge on any of the three tasks; the best prefix seed equals its mean panel count for each task. All prefix results retain the canonical Greedy floor.

| Parts | Greedy envelope occupancy | Short-edge envelope occupancy | Prefix mean envelope occupancy |
| --- | ---: | ---: | ---: |
| 178 | 85.5369% | 85.5369% | 85.5369% |
| 739 | 90.3889% | 92.2720% | 91.3207% |
| 2416 | 90.8932% | 91.3819% | 91.1369% |

Net U is nominal area divided by full purchased-board area. Envelope occupancy additionally counts the reserved 6 mm part envelopes; it is not useful product yield or exact cutter-path loss. No denominator excludes tail boards.

These are current-system vs deterministic-order results, **not a causal isolation of the prefix component**. Their initialization and evaluation paths differ as specified. A later common-strong-initialization prefix-on/off experiment would be needed to isolate additional prefix benefit; this batch did not perform it. Prior SA best-seed layouts belong to a different frozen batch and are not substituted into this comparison.

## 4. Failures and verification

**15/15** rows completed with clean worker exits, independent geometry/identity audits and minimal serialization within budget. All saved outputs were independently reaudited; input/runtime/result/schedule/dependency identities match. All three short-edge sequences were checked against an independently constructed literal ordering from saved nominal dimensions. No missing/duplicate parts, Greedy-floor violation, failed/watchdog/cleanup run, missing comparison or orphan saved output. Maximum post-serialization total elapsed **27.271525458 s < 30 s**.

Regression sequence: 10 failing tests and two existing-contract passes before implementation; **12/12** pass after repair. Covered shared-vs-cold identities, geometry and actual hashes with same-shape and rotation-only parts; interrupted replay followed by a valid replay; same-state/UID cache hits; literal tie ordering; all four adaptive tiers (20/178/739/2416); runner/handoff dispatch and exact deadline interruption. Whole suite: **323 passed / 3 existing external CutEval skips**; compileall and diff checks pass. Ten private behavioral checks cover scheduling, failure/incomplete summaries, task-mean rather than best-seed comparison, and reused worker lifecycle.

Fresh independent review of exactly `02bf332..26c7fdf` approved the fix with no Critical/Important findings. It independently ran targeted tests and additionally checked all 120 permutations of a five-part multi-board fixture, including duplicate dimensions and rotation-only parts: shared/cold identities, geometry, layout/ordered hashes, global scores and scalar rewards agree. Same-UID micro-cache reuse across distinct empty board IDs also passed. The reviewer did not independently rerun the whole suite or review private experimental outputs.

Deferred minor: retained automated regressions use single-board fixtures and do not snapshot/compare the full saved incumbent geometry/hash/score across interruption. The reviewer's extra multi-board check was in memory, not added as a retained regression. No remote-CI or unavailable external-integration claim.

Historical failure records and results are unchanged. This follow-up resolves the **shared-cache risk** left open by [the preceding yardstick report](evaluation-yardstick-20261111.md); it does not retroactively change that runtime's results or convert its interrupted first batch into successful evidence.

## 5. Acceptance and scope

Shared-cache correctness and this development comparison are complete. Material quality relative to the strong short-edge reference **does not pass**. No formal 420 s matrix, unseen-order validation, new SOTA/commercial/optimality claim or master merge. The previously approved formal acceptance gate and the restricted-prefix theme remain unchanged.

The next useful comparison is a **common strong initial solution with prefix enabled/disabled**, keeping all other settings matched, before introducing another candidate/commit heuristic. Merely retaining the best of short-edge and the existing system would improve the delivered floor; it would not demonstrate a research contribution from prefix search. This next comparison is not implemented or run here.

## 6. Publication and execution choices

Tested code and this aggregate report update the existing draft PR. Original dirty checkout and frozen formal evidence are preserved. Detailed orders, layouts, result files and audit manifests remain private.

Execution choices: implement the explicitly authorized repair/comparison inline without another planning checkpoint (reference scope may later need revision); restrict the fresh review to this new code range and independently audit the private experiment/publication (withdraw affected comparisons if that evidence fails). The deferred regression coverage is disclosed above. Private execution records retain the exact decisions and verification commands.
