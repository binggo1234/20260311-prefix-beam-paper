# Common strong initialization: prefix on/off — 2026-11-11

After exactly matched strong initialization, the existing restricted-prefix procedure produces **no additional board savings on these three known orders at 30 s**. All nine on/off seed pairs tie; all nine on results retain the exact initial layout. The ablation is implemented and validated; incremental material-quality acceptance is not passed. This is a scoped negative result, not a proof that restricted-prefix search is generally ineffective.

## 1. Version and implementation

Frozen runtime: `4acad1344c485f02ef9e09ce3a8a57dfa9d59f27`, based on `d33a834`. Reporting commits do not change runtime sources. New opt-in, budgeted-only variants:

- `rh_mcts_prefix_common_init_off`: common initialization, then return the saved complete incumbent immediately.
- `rh_mcts_prefix_common_init_on`: identical initialization, then the existing staged restricted-prefix search.

Both inherit the same staged/adaptive configuration and differ only in variant identity and `PREFIX_ABLATION_SEARCH_ENABLE`. Both keep `PREFIX_BEAM_ENABLE=True`; setting it false would enter another legacy MCTS/SA/postprocessing pipeline and would not implement this ablation. `PREFIX_COMMON_INIT_ABLATION=False` and `PREFIX_ABLATION_SEARCH_ENABLE=True` are the defaults, so existing variants retain their original initializer/policies.

The common protocol decodes canonical Greedy then exact short-edge-descending order with the same corrected shared decoder, retaining the lexicographically better complete layout. Both decodes use the absolute search deadline and count against the run's budget. Additional constructive/random warmstarts, archive registration and suffix priming are bypassed identically in both arms. The common initial solution is an incumbent, not a new committed prefix; on still begins its existing prefix search from an empty root. No geometry, prefix depth/width, candidate/commit/completion heuristic, search RNG or default research-policy change.

The two profiles require a finite absolute formal deadline and prefix-beam mode, rejecting incompatible configuration before decoding. Off returns before root construction/warmstart and runs no completion, search, SA or repair. On uses the existing formal prefix close, which skips optional final repair/SA. All close paths retain a saved complete snapshot without final replay.

If Greedy cannot complete, the solver fails. If the second initialization replay expires, complete Greedy may still be delivered, but initialization is marked incomplete, search is skipped and the experimental pair is ineligible. Initial/final sequence and raw-layout SHA256 receipts include UID, board ID, unrounded coordinates/dimensions and rotation. These receipts are for the same frozen runtime; no cross-runtime bitwise-portability claim.

## 2. Locked paired design

Three known orders (178/739/2416 pieces), three paired seeds (1000/1001/1002), two arms: **18 runs**. Task×seed blocks and arm order randomized with seed **20261114**. Off is deterministic; its per-pair repeats verify initialization identity and are not independent production orders. The order is the experimental unit.

Each run: 30 s internal pipeline allowance including input loading/construction, initialization, search when enabled, validation, independent geometry audit and minimal serialization; 3 s closing reserve; 38 s absolute parent watchdog; 8 s cleanup within that watchdog. Python process startup is outside the internal timer and covered by the watchdog. Off returns early rather than padding idle time. This measures the benefit and cost of adding prefix search after a common initialization under the same maximum allowance; it does not isolate PUCT from other internals of the existing prefix procedure.

Source/input/dependency/script hashes, the complete schedule and analysis rules were frozen before outcomes. No selective reruns, retuning or larger matrix. Invalid/incomplete/mismatched pairs withhold the order-level comparison and remain in the evidence. Within an order, initialization must match both across arms and across all seeds.

## 3. Every seed and material results

Entries follow seeds **1000 / 1001 / 1002**.

| Parts | Common Greedy / short-edge panels | Off panels | On panels | Mean net U, both | Envelope occupancy, both | On − off panels / net U |
| --- | --- | --- | --- | ---: | ---: | --- |
| 178 | 29 / 29 | 29 / 29 / 29 | 29 / 29 / 29 | 83.7932% | 85.5369% | 0 / 0 pp |
| 739 | 98 / 96 | 96 / 96 / 96 | 96 / 96 / 96 | 90.3116% | 92.2720% | 0 / 0 pp |
| 2416 | 374 / 372 | 372 / 372 / 372 | 372 / 372 / 372 | 89.6355% | 91.3819% | 0 / 0 pp |

Task outcome: **0 wins / 3 ties / 0 losses**; sum of task-mean panel differences **0**. Every on best seed equals its task mean. All on final layout fingerprints equal their initial fingerprints: no retained layout change, not merely equal panel counts. Off final sequence/layout/score equals the common initial solution exactly, and all off simulation/PUCT/completion counters are zero.

Net U uses nominal product area over full purchased-board area; envelope occupancy includes reserved `(w+6)*(h+6)` area and is not effective product yield or exact cutter-path loss. Boards remain 2440×1220 mm with 5 mm trim/edge and the supplied rotations. No tail-board exclusion or geometry relaxation.

The stronger floor delivers 96/372 panels, but those layouts already exist before prefix search. Their quality must be attributed to deterministic initialization, **not to prefix research contribution**. Earlier system-vs-short-edge tables remain historical evidence and are not treated as paired timing controls here.

## 4. Search actually executed

| Parts | On completed simulations by seed | On distinct completion states | On PUCT selections | On total seconds by seed | Off mean total seconds |
| --- | --- | --- | --- | --- | ---: |
| 178 | 80 / 80 / 80 | 75 / 75 / 75 | 99 / 99 / 99 | 2.458 / 2.514 / 2.436 | 0.116 |
| 739 | 79 / 78 / 80 | 79 / 78 / 80 | 59 / 59 / 60 | 27.219 / 27.200 / 27.206 | 0.891 |
| 2416 | 7 / 8 / 8 | 7 / 8 / 8 | 0 / 0 / 0 | 27.243 / 27.243 / 27.242 | 9.347 |

On initialization takes approximately 0.094–0.097 s / 0.790–0.818 s / 8.998–9.096 s for the three sizes. The largest order still completes only 7–8 states with zero PUCT selections. The smaller orders also show no material gain despite more evaluated states, so the whole negative result cannot be attributed solely to large-order throughput. These counts verify execution; they are not substitutes for material quality.

## 5. Verification and acceptance

**18/18** clean worker exits, successful independent geometry/identity audits and minimal outputs within budget. Every saved output was reaudited; source/input/result/schedule/dependency hashes match. All nine pairs match initial sequence/layout receipts, panel counts, full score and candidate panel ledger; all three orders also match initialization across seeds. No invalid/incomplete pair, failure, floor violation, missing comparison or orphan saved result. Max post-serialization total elapsed **27.243417958 s < 30 s**.

TDD: 13 absent-feature/wrong-pipeline failures before implementation, then 15 targeted passes; an additional missing final-receipt failure preceded that receipt implementation. Full suite **338 passed / 3 existing optional CutEval skips**, compileall and diff checks pass. Sixteen private behavioral checks cover schedule balance, missing/duplicate identities, failures, initial mismatch, incomplete seeds, off-state mutation/search, cross-seed mismatch and task-mean vs best-seed reporting, plus the reused worker lifecycle.

Fresh independent review of exactly `d33a834..4acad13` found no Critical/Important/Minor issues and independently ran the 15 targeted tests and full338/three-skip suite. It reviewed initialization parity, exact fingerprint serialization, immutability, deadlines and the early off gate. Private batch outputs were independently audited by the main executor, not reviewed by that code reviewer. External CutEval user modifications remain preserved; skipped integrations and remote CI are not claimed to pass.

Implementation and controlled-ablation execution pass. Additional material savings from the current prefix procedure after common initialization **are not demonstrated**, and the existing formal material-quality gate is not satisfied. No formal420 s matrix, unseen-order validation, SOTA/commercial/optimality claim or master merge. The restricted-prefix theme and all earlier negative/failure records remain unchanged.

## 6. Decision and next scope

Keep the new profiles opt-in for reproducibility. Do not adopt a new heuristic or extend a long matrix on the strength of this null result. The next bounded diagnostic should compare the previously audited 95-panel medium-order witness sequence with the current legal-prefix and completion-order rules, using saved evidence first. This can clarify what the current representation/evaluation would need to cover; a missing hit alone does not prove reachability or impossibility. That diagnostic is not implemented/run in this batch.

Execution choices: act within the approved ablation without another design checkpoint (the initialization protocol may later need revision); independently freeze/test/audit the private experiment and preserve historical evidence (withdraw affected comparisons if audit evidence fails); restrict fingerprints to the frozen runtime and preserve skipped dirty integrations (no cross-runtime or external-integration claim); interpret the treatment as the whole existing prefix-search procedure, not PUCT alone (no component-specific causal claim). No deferred review findings. Only code/tests/aggregate documentation are public; factory data/layouts and detailed audit files stay local.
