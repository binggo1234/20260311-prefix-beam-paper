# Factory data ASH/prefix screen, frozen before packing outcomes

2026-11-26. User explicitly requested the factory data at `/Users/binggo/Desktop/graduate/data/` after the public ASH comparison. This authorizes this new factory screen; the preceding pause on unsolicited matrix expansion does not prohibit this requested run. No production algorithm or thesis edits, no new heuristic, no default solver change, no factory raw-data publication.

## Data and geometry

Use all27 actual `txpnx/*.csv` tasks identified by `data_source_scope_20261126/audit.json`:39267 parts,10 batches,1–2819 parts per task. Do not count `nj`, ZIP copies or temporary files as extra inputs. Hash every source and verify count and exact Upi/Fleng/Fwidth binding. One row is one part. Preflight established each task has one material/thickness group and every part fits the stock in at least one allowed orientation.

Keep established factory configuration:2440×1220mm stock, trim5mm on every edge, tool diameter6mm, no grain restriction,0/90 rotations. Effective packing envelope per part is nominal width+6 by nominal height+6; nominal part area divided by gross purchased board area is the core utilization metric. Never count tool clearance as useful product area. Source README metadata cannot override user-confirmed parameters. Preserve raw files, existing experiments and Word master.

All27 tasks have prior project exposure, not unseen industrial validation. Public reporting uses F01–F27 assigned by sorted relative source path; full mapping remains private. No claim that single orders are independent of factory batches.

## Fixed methods and budget

Reuse unchanged `inrp.sequence_reference.run_reference` at the current research revision, with the same decoder, common Greedy+short-edge initial solution and normal intra-run caches. SINGLE keeps the existing early single action order; PAIR lexicographically enumerates distinct ordered UID pairs with H suffix; ASH uses the already validated source-policy transplant, alpha.1/.002/.65 and strict board-only base updates. Full UID-sequence exact dedup only, no geometry dedup, no historical solution injection. All initialization uses seed1000; ASH stochastic stream seed1000 only in this first full-data screen. No seed-stability or state-of-the-art claim.

Exactly81 independently cold-started runs:27 tasks×3 arms. Randomize task blocks and arm order within each task before executing, seed20261127. Sequential worker execution to avoid intentional solver contention. Each run has30s child pipeline total including solver/helper imports, input load, initialization, generation/filtering, replay, receipts and final validation/export/write;3s closing reserve, so search deadline at27s. Parent startup/exit measured separately with35s watchdog and clean-exit requirement. Compare durations within their own process, never cross-process absolute perf_counter values.

Ordinary run cap1000000 full replay starts, raw proposal cap100000, both include the declared counting semantics from the public experiment. Only the one-part task uses cap2 for all three arms: two identical initializer calls already cover its sole permutation; report as degenerate no-search control, not algorithm improvement. No early stopping based on observed gains. No retries, enlarged limits, changed cohort or adaptive seeding after outcomes.

## Endpoints and qualification

Primary:final boards and nominal/gross-board utilization at30s; case-level ASH vsSINGLE and ASH vsPAIR wins/ties/losses plus aggregate boards and area-weighted utilization across matched eligible tasks. This includes material savings relative to the freshly matched common initialization and direct methods, distinct from beating historical best known witnesses.

Secondary predeclared checkpoint:12 total completed evaluations INCLUDING2 initializer calls (=10 new candidates), projected from the same timed trace. Candidate generation and adaptation do not depend on future remaining quota. This is a fixed-evaluation checkpoint within timed runs, not a separately executed count-limited experiment. If fewer than12 complete, mark unavailable/time-censored; never replace it by an earlier final result, and restrict paired checkpoint comparisons to tasks where both arms reach12. The one-part task is structurally exhausted, not a time-censored failure. Do not equalize counts after observing outcomes by choosing a smaller checkpoint.

Record first-saving candidate and total evaluation, first-saving elapsed, complete/started/interrupted work, proposals/duplicates, actual times and initial-state hashes. ONE ASH seed is descriptive only, not stable superiority. Even full12-call ties cannot evaluate long search quality. Pair/global ASH express different spaces; this is a direct reference, not a same-two-piece selector ablation.

## Implementation, verification and delivery

Only a private factory input/driver/auditor adapter is added; public solver policy remains unchanged. Test source/count/geometry checks and checkpoint censoring before execution, run existing relevant/full tests. Freeze source inputs, code/dependencies, protocol, prior public experiment records and master. Preserve first execution and all failures.

Save every complete layout, full sequence and score in private per-run JSONL, counting capture/write cost in the run budget. Offline audit forbids decoder/search execution; source-bound geometry and kerf envelopes checked for every complete receipt. Independently transcribe all proposal streams, ASH random/update history, best-layout provenance and counts. Match two initial receipts and full decoder cache entry across the three arms of each task. Failure/timeout is reported, not dropped or retried; audit available complete receipts separately if a pipeline fails.

For progress and early error visibility, perform the same saved-only three-arm audit between completed task blocks, with no timed worker running. It cannot change the remaining frozen schedule or trigger a retry. A final whole-batch audit repeats qualification against the original hashes.

Publish only protocol, anonymous task-level aggregate CSV and report to the existing research branch after verification. No raw factory parts/UIDs/layout coordinates, paper PDF or master merge. Report actual utilization gain or absence, budget coverage and limitations without treating experiment volume or engineering work as an algorithm innovation.
