# Short-edge primary completion: prefix on/off — 2026-11-12

On the one preselected739-piece order, all three paired seeds use **96 panels with prefix off and95 with prefix on**. Mean net material utilization increases **90.3116%→91.2622%, +0.9506 percentage points**. This passes the prespecified development-signal rule, not the unchanged formal quality gate. Restricted-prefix search remains the research topic.

## 1. Version and minimal change

Frozen runtime: `810e43966dcb1b8292b2a9cf14184dce308f57ca`, based on `72d8a83`. Reporting changes do not alter runtime sources. Two opt-in profiles:

- `rh_mcts_prefix_short_completion_off`, inheriting `rh_mcts_prefix_common_init_off`.
- `rh_mcts_prefix_short_completion_on`, inheriting `rh_mcts_prefix_common_init_on`.

The only configuration change relative to each parent is primary completion: `(baseline,long_edge,strip_bias)` becomes `(short_edge,long_edge,strip_bias)`. Register the previously absent oracle-mode name, sort by raw nominal short edge descending, nominal area descending and UID ascending, and record the normalized active modes in metadata. The order exactly matches the existing strong initializer, including tiny positive dimensions and remaining subsets. No dependency change.

Both arms still decode Greedy then short-edge initialization and retain the better complete incumbent with identical decoder/cache/RNG state. The existing search gate distinguishes on/off. On starts from the existing empty root; off returns the saved initialization without search or idle padding. Candidate generation, scoring, adaptive tiers, prefix depth/width/commit rules, staged promotion quotas, both fallback modes, deadlines and geometry remain unchanged. Earlier complement/type-fair/witness mechanisms stay disabled. Old/default profiles retain their old completion tuple and initialization behavior.

## 2. Locked scope and budget

One known739-piece order, seeds1000/1001/1002, two arms: **six runs, once each**. Schedule seed20261115 randomizes seed blocks and arm order. Sources, runtime/input/dependency hashes, schedule, eligibility and decision rules were frozen before outcomes. No retuning, selective rerun, added heuristic or larger matrix.

Each run has a30 s pipeline allowance including input loading/construction, both initial decodes, enabled search, validation, independent geometry audit and minimal serialization;3 s closing reserve;38 s parent watchdog;8 s cleanup within that watchdog. Process startup is outside the internal timer and covered by the watchdog. Both arms have the same maximum allowance; off returns early. This estimates the benefit/cost of enabling the existing prefix procedure with aligned completion, not PUCT alone or equal consumed CPU time.

All arms/seeds must match initial panels, full score, candidate-panel ledger, sequence and raw-layout receipts, also matching the prior frozen same-order96-panel common seed. Initial Greedy/short-edge counts are98/96. Incomplete, failed, mode-mismatched or floor-violating rows remain recorded and withhold the order-level mean. On must actually evaluate short-edge primary completion. Development signal requires a valid mean_on<96; best seed is never substituted for the mean.

## 3. Every seed and material result

| Seed | Off panels | On panels | Off total seconds | On total seconds | First95-panel event, solver-origin seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1000 | 96 | 95 | 0.874 | 25.260 | 3.555 |
| 1001 | 96 | 95 | 0.877 | 25.733 | 3.488 |
| 1002 | 96 | 95 | 0.881 | 25.397 | 3.394 |
| Mean | **96** | **95** | **0.877** | **25.463** | — |

| Material metric | Off, every seed and mean | On, every seed and mean | Difference |
| --- | ---: | ---: | ---: |
| Net utilization | 90.3116% | 91.2622% | +0.9506pp |
| Reserved-envelope occupancy | 92.2720% | 93.2433% | +0.9713pp |

One task win, zero ties/losses; three paired improvements, no floor regression. The three seeds are repeated algorithm runs on **one order**, not three independent orders. Each on final sequence/layout differs from initialization; each off final sequence/layout/full score equals initialization exactly.

Net utilization uses nominal product area/full purchased-board area. Envelope occupancy uses reserved `(w+6)*(h+6)` area and is distinct from product yield and exact cutter-path loss. Boards2440×1220 mm,5 mm trim/edge,6 mm allowance, permitted0/90° rotations unchanged; no tail-board exclusion or geometry relaxation. Area lower bound90 is a relaxation, not an attainable optimum or proof95 is optimal.

## 4. Execution and source of improvement

On completed80 simulations and80 distinct completion states for every seed; PUCT selections are49/50/50 in seed order. Each on evaluated short_edge80 times, long_edge20 and strip_bias20, baseline0. Off simulation/PUCT/completion counters are all zero. Initialization takes0.803–0.822 s across all six runs.

All first95-panel events are recorded as `prefix_completion`, winning mode `short_edge`, evaluated prefix depth1. Later best events also remain short-edge completions. These are complete incumbent events, followed by final geometry auditing. They show material improvement after the common seed through this prefix/completion path, without SA or optional repair; they do not isolate which selection component is necessary. Recorded retained beam depth reaches3. This does not impose depth3 on every simulation leaf, and `commit_steps=0` is not a count of all beam-root retention events.

The result provides a legal restricted-prefix/completion95-panel example under the new suffix family. It need not reproduce the previous whole-sequence SA witness. The prior witness audit's724/725/725 requirements applied to that particular sequence and the old suffix modes; they were never an impossibility claim for all95-panel layouts.

## 5. Verification and acceptance

**6/6** clean workers and independently reaudited saved outputs,3/3 eligible initial-matched pairs, cross-seed initial match,0 execution failures,0 protocol-invalid rows,0 floor violations and0 orphan outputs. Initial receipts also match the frozen historical common seed; that reference verifies identity, not historical treatment efficacy. Source/input/result/schedule/dependency hashes match. Max post-serialization elapsed25.732635166 s<30 s.

The executor separately reconstructed all six raw final-layout fingerprints from exported placements, source dimensions and sequence order, recomputed both material metrics and checked actual mode counters and event provenance. This supplementary audit adds zero packing runs and does not change the locked analysis. All results and failures from older experiments remain untouched.

Nine new runtime tests were observed failing before implementation, then passed. Full suite **347 passed/3 existing optional CutEval skips**; compileall and diff checks pass. Twenty-five private checks include nine new schedule/protocol/mean tests plus reused pair-identity and bounded worker-lifecycle tests. Skips preserve existing dirty external integration sources and are not counted as passes.

One fresh read-only reviewer checked `72d8a83..810e439`, found no Critical/Important/Minor findings, and independently ran347 passed/3 skips. The reviewer covered production code; private execution/performance/publication were explicitly left to the executor's tests/audits. No independent private-harness review or remote-CI pass is claimed.

Implementation, bounded experiment and the prespecified **development signal pass**. Formal acceptance remains **not achieved**: no formal420 s matrix, cross-order replication or unseen-order validation in this batch. This six-run design does not concurrently compare area-primary and short-edge-primary, so historical null results do not establish a causal effect of changing the primary mode. No SOTA, commercial superiority, optimality or general reliability claim; no master merge.

## 6. Decision and next scope

Freeze this opt-in candidate and retain the complete evidence. The next planning priority is a prespecified cross-order check with the same common initialization and on/off protocol; it has not been launched. Additional heuristics remain paused. This development signal justifies further validation, not immediate default adoption or a formal-quality claim.

Execution rulings: reuse the established isolated clone and authorized existing draft (cost if authorization were misunderstood: branch visibility); review production before freezing, then separately test/audit the private batch (cost: no independent harness/performance review); accept the reviewer's exclusion of pending experiment/publication only because those are verified separately before publication (same limitation). No deferred minors. Public changes contain code/tests/spec/plan/aggregate report only; factory inputs, layouts, sequences and detailed audit files stay private.
