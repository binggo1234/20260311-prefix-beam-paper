# Cold replay of the captured one-piece prefix — 2026-11-12

The known739-piece order reproduces **96 panels for canonical short-edge order and95 panels for the captured single-piece-front order**, using two independent fresh decoder instances. Both complete sequences and raw layouts exactly match their corresponding historical digests, and both pass independent geometry audits. The recovered first-saving witness therefore reproduces without search-warmed decoder caches. This is one fixed-order reproducibility check, not a new search-quality experiment.

## 1. Version and scope

Run checkout `08c742620867935457661dfbe825ed7c75d84372`; production source remains identical to frozen runtime `e71695771a6ec871d4fdbc474e47fca4a8a5aeb7`. This batch adds only this public report. Private execution/audit/test scripts and raw evidence stay local. No solver, default, candidate, ranking, completion, geometry or dependency changes.

The inputs are the previously captured first-saving receipt and its original common short-edge reference. Before any new decode, the receipt's event/raw-layout bindings and exact equation were verified: take the existing piece at original short-edge position497, move it to the front, and preserve the relative order of the remaining738 pieces. The position identifies this known witness; it is not a selection heuristic.

## 2. Prelocked execution

Exactly one pair, in declared order: original short-edge, then captured single-front. Each arm prepares its own parts/configuration and constructs its own decoder with a new empty starting layout. Configuration values are identical, while decoder/configuration objects and all three cache containers are distinct. Initial cache sizes and all hit/miss counters are zero. No prior snapshot is supplied.

Runtime/input/receipt/script/dependency hashes, target counts and arm order were written before execution. Shared pair budget: **30 s pipeline,3 s closing reserve,38 s parent watchdog and8 s bounded cleanup**. Both replays use the same absolute27 s replay boundary; there is no per-arm budget reset. The pipeline covers source/receipt handling, preparation, both decodes, validation, independent geometry audits and minimal writes after runtime imports; the parent watchdog also covers startup/imports. No retry, retuning or additional factory decode.

MCTS construction and solver entry points are guarded against invocation. There is no initialization search, completion-mode selection, candidate generation, SA or repair. The diagnostic profile supplies configuration only; exported metadata labels the execution as fixed-order cold replay. Acceptance separates complete valid execution from whether historical counts/layouts reproduce, so a genuine miss would remain a recorded result.

## 3. Material and layout results

| Item | Original short-edge | Captured single-front |
| --- | ---: | ---: |
| Panels | **96** | **95** |
| Net material utilization | 90.3116% | **91.2622%** |
| Reserved-envelope occupancy | 92.2720% | 93.2433% |
| Fixed-order decode time | 0.377334333 s | 0.363265042 s |
| Full sequence matches historical reference | Yes | Yes |
| Raw layout matches historical reference | Yes | Yes |
| Independent geometry audit | Passed | Passed |

Saving: **one panel, +0.9506483 percentage points net utilization**. Envelope occupancy rises by0.9712842 points; it counts the6 mm rectangular clearance allowance and is not net product yield or exact cutter-path loss. Boards2440×1220 mm,5 mm trim,6 mm tool allowance,0/90° rotations and other geometry settings are unchanged.

The pair finishes in **0.861162 s** from budget creation through post-serialization completion, with clean worker exit. Timing is descriptive for this fixed-order check. The supplied witness already contains the answer; this is not the time needed to discover it, and does not establish a speedup over the earlier search.

## 4. Cache and independent evidence checks

Both arms start with empty board-term, micro and decode caches and zero counters. Each finishes with739 decode misses and zero decode hits. Observed micro hits are also zero; micro misses are53,247 and53,211 respectively. These observations distinguish an empty-cache replay from the earlier receipt's merely fresh completion-mode evaluation. There is no transferred decoder/snapshot/cache between arms.

Post-run audit performs no packing replay. It binds saved UID order and raw layout digests to their historical references, reconstructs nominal placements from raw envelopes, checks the exported sequence/placements and repeats source-based geometry audits. Each contains739 requested pieces exactly once with matching dimensions, permitted rotations, trim/boundary feasibility and tool-clearance non-overlap. Serialized pair/progress records, actual counts, reproduction flags, budgets and frozen file identities agree. Execution accepted; both target counts and layouts reproduced; no audit finding.

## 5. Validation and limits

**18 private saved-evidence/guard checks pass**, including corruption rejection and a valid execution with an honest reproduction miss. Before data collection, the valid toy case exposed an identity-counting error in the new private auditor; changing the counter input from mapping values to UID keys made the suite pass. No factory outcome existed when this was corrected. **24 existing decoder-identity/deadline/budget checks pass**; Python compilation and Git whitespace checks pass. No production code changed, so the full runtime suite was not rerun for this documentation batch. There was no independent reviewer for the private harness and no remote-CI pass claim.

All483 preflight historical JSON files and the original dirty/formal frozen workspaces retain their identities. Two fixed decodes of one known order are not independent order-level replicates. This result establishes the feasibility and cold reproducibility of this witness; it does not establish unseen-order gains, average search benefit, PUCT necessity, an optimal panel count or a general rule based on position497. Earlier cross-order ties and the unpassed formal gate remain unchanged.

## 6. Progress and next priority

The evidence chain now connects an actual one-action restricted-prefix event to its saved95-panel layout and to an exact cold-decoder reproduction. The unresolved problem is **finding useful short prefixes reliably and cheaply**, not whether this particular witness survives fresh decoding.

Keep the restricted-prefix topic, current defaults and paused new heuristics. The next bounded priority is a read-only audit of the existing single-action candidate coverage and evaluation path: establish what the saved evidence/code can say about how this witness became available and selected, and mark missing candidate-history information as unknown. Do not turn the known piece into a heuristic, enlarge the quality matrix or launch the formal gate as part of this batch.

Related evidence: [actual first-saving receipt](first-saving-receipt-20261112.md), [paired739-piece development check](short-edge-completion-ablation-20261112.md), [cross-order check](short-completion-cross-order-20261112.md). Raw factory identities, dimensions, sequences and layouts remain private. Existing draft PR only; no merge to master.
