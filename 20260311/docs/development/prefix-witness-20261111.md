# Complete board-saving witness advancement: mechanism works, quality ties

Decision (2026-11-11): keep `PREFIX_BEAM_WITNESS_ADVANCE=False`. The opt-in variant is retained for reproducing a development result, not adopted as a validated material-utilization improvement. Do not launch its longer selection/formal matrix. Restricted-prefix search remains the topic and original staged completion remains the research control.

Runtime frozen before the batch: `5b6e7b27400112e30b8e08821b7a84f98ca3f646` (feature `6bab7d2`, followed only by additive fresh-evaluation evidence/tests). Concurrent control is `rh_mcts_prefix_beam_staged`; treatment is `rh_mcts_prefix_beam_staged_witness` on the same runtime. Historical `48b527a` is the reference policy identity, not a historical timing control.

## Why this single factor

A read-only audit found the large task exhausted search time before the existing commit checkpoint: batches request 8 simulations, the round floor is 16, and only 8 succeeded in the earlier matched diagnostic. All selected leaves were depth 1. A refreshed seven-entry candidate list represented eight distinct actions over time, explaining zero PUCT selections without assuming a fixed candidate set. The complete 373-board incumbent was retained, but its prefix was never committed.

The new policy tests one remedy: after an existing prefix produces a complete layout with strictly fewer boards than the pre-evaluation incumbent, advance that legal path immediately instead of waiting for the ordinary checkpoint. It does not lower visit thresholds generally or add candidate rules.

## Exact implementation scope

- Default-off `PREFIX_BEAM_WITNESS_ADVANCE`; enabled only with prefix search and staged completion. The new variant inherits the staged profile, differing only in this flag and identity. Adaptive-profile isolation is tested at 6/178/739/2416 parts.
- Capture the incumbent board count before evaluation; staged evaluation already updates the incumbent internally. Secondary-score changes with equal boards do not trigger.
- Require a complete sequence covering the order exactly, successful backup, remaining search time, a nonterminal leaf and an admissible existing path. Its length cannot exceed `COMMIT_MAX_STEPS`, and its absolute leaf depth cannot exceed `PREFIX_BEAM_DEPTH`; every edge obeys the existing single fallback rule.
- Return an optional witness path plus its already completed snapshot from `_run_simulations`. Consume it in the same beam-root iteration before the round-floor check. No pending candidate leaks into another batch/root.
- Use the existing detach/warmstart path and compute the witness rank from its saved completion afterward. This particular commit ranking does not run another completion or promote partial fidelity to a full cache entry. Ordinary non-witness ranking is unchanged.
- Preserve candidate generation/limits, priors, PUCT, RNG, staged promotion quotas, initial Greedy/optional warmstarts, commit depth/width, geometry, objective and repairs. Early concentration can lose alternatives; final regressions remain an explicit risk.

Evidence: `prefix_witness_enabled`, `prefix_witness_triggers`, `prefix_witness_advances`, `prefix_witness_deeper_evals`, `prefix_witness_deeper_fresh_evals`, `prefix_witness_deeper_depth_max`, and at most 32 `prefix_witness_events`. Complete deeper returns can include cache hits; the separate fresh counter requires at least one successfully completed new completion-mode replay. Neither count implies extra board savings.

## Locked paired check

Known development tasks: 178/739/2416 parts; seeds 1000/1001/1002. Task×seed blocks and within-block method order randomized with seed 20261111. Two methods per block plus Greedy once per task: **21 runs**. Seeds are repeated algorithm runs on three known orders, not nine independent orders.

Fixed 30 s total / 3 s closing reserve, including input loading, solving, independent geometry audit and atomic minimal serialization; absolute parent watchdog 38 s. Before freezing the batch, five private lifecycle tests verified a shared 8 s post-result cleanup grace bounded by that watchdog. Cleanup is outside, and never extends, the solver's 30 s budget. Earlier diagnostic failures with a two-second exit grace remain unchanged in local history.

Runtime/source/input identities, environment, runner/summarizer hashes and schedule were locked before execution. No policy retuning or selective reruns. No code changes during the batch. All raw factory inputs, layouts and detailed results remain private.

## Final quality: all seeds tie

Triplets are seed 1000 / 1001 / 1002. Utilization includes every used full board, including the tail board.

| Parts | Greedy boards | Staged boards | Witness boards | Net utilization, both | Envelope occupancy, both |
| --- | ---: | --- | --- | ---: | ---: |
| 178 | 29 | 29 / 29 / 29 | 29 / 29 / 29 | 83.7932% | 85.5369% |
| 739 | 98 | 97 / 97 / 97 | 97 / 97 / 97 | 89.3805% | 91.3207% |
| 2416 | 374 | 373 / 373 / 373 | 373 / 373 / 373 | 89.3952% | 91.1369% |

Task outcomes **0 wins / 3 ties / 0 losses**; all nine paired board differences zero; sum of task-mean board differences zero; all net-utilization differences **0 percentage points**. Existing advantages over Greedy are shared by the control and treatment, not new witness-policy gains.

Net utilization uses nominal part area / `(boards*2440*1220)`. Envelope occupancy uses `(w+6)*(h+6)` reservation area under unchanged 6 mm clearance and 5 mm trim; it is not measured cutter-path loss. No metric or geometry constraint was changed to improve the numbers.

## Mechanism: early commit really happened

Every value below was identical across the three seeds except elapsed time.

| Parts | Simulations staged → witness | Distinct complete states | PUCT selections | Committed depth | Witness advances | Fresh deeper evaluations after witnessed commit | Max deeper evaluation depth |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 178 | 80 → 80 | 77 → 77 | 76 → 76 | 3 → 3 | 0 | 0 | not applicable |
| 739 | 80 → 71 | 80 → 71 | 64 → 46 | 3 → 3 | 1 | 16 | 4 |
| 2416 | 8 → 8 | 8 → 8 | 0 → 0 | 0 → 1 | 1 | 1 | 2 |

In the large task, the witness committed depth 0→1 at 21.6128 / 21.7400 / 21.6874 s (solver-evidence origin), then completed one new depth-2 evaluation in every seed. This is actual fresh decoding, not just a depth counter or cache-only return. It replaces one shallow evaluation without increasing the total eight states; no extra board was saved.

The medium task advanced depth 1→2 on its 98→97 complete witness, then completed 16 fresh deeper evaluations. It finished the beam flow with fewer total simulations, not with more evaluated states. No universal throughput claim follows.

`prefix_depth_reached` is **committed** depth, not the maximum MCTS evaluation depth. The existing commit-depth limit is 3; ordinary MCTS selection can evaluate deeper leaves (including depth 4 here). This existing distinction is not modified by the witness flag. Likewise, 80 is the small task's actual flow count, not an independent total-simulation hard cap.

| Parts | Mean total s staged → witness | Maximum total s, both |
| --- | --- | ---: |
| 178 | 2.4827 → 2.4983 | 2.5154 |
| 739 | 26.3345 → 25.1690 | 26.4183 |
| 2416 | 27.2296 → 27.2238 | 27.2367 |

First complete below-Greedy event seconds (1000 / 1001 / 1002):

- 739 control: 11.0691 / 10.8712 / 10.8977; witness: 11.1181 / 10.8629 / 10.8282.
- 2416 control: 21.5328 / 21.6529 / 21.4296; witness: 21.6045 / 21.7315 / 21.6791.

Events originate at solver evidence initialization, not external input loading. The witness policy does not consistently reach the first saving earlier. Inclusive phase timers are nested and must not be added as wall time.

## Correctness, execution and code verification

All **21/21** runs completed with clean process exits and no watchdog/cleanup failures. All outputs passed independent identity/rotation/dimension/clearance/trim/bounds/non-overlap/utilization audits, then were reaudited after saving. Source/runtime/result/schedule identities matched; no unassociated saved output was omitted. All minimal results were written within 30 s; maximum observed elapsed after serialization **27.236704333 s**. No Greedy-floor violation or paired board-count regression.

The prior instrumented read-only audit had three failed historical-quality reproductions and two process-exit failures. This concurrent batch does not relabel them or establish the cause of that earlier throughput change; it supplies a fresh control under the present execution conditions.

From `20260311/`:

```sh
python3 -m pytest -q tests/test_prefix_witness.py
python3 -m pytest -q -rs
python3 -m compileall -q src tests scripts
git diff --check
```

**21 witness tests** pass; full suite **289 passed / 3 skipped**. Skips are the existing dirty external CutEval checkout, preserved untouched. Five private runner lifecycle tests also pass. Tests cover pre-evaluation comparison, opt-out, equal-board/noncomplete/failed/expired/terminal cases, path/depth/admissibility limits, partial fidelity, cached-vs-fresh evidence, actual deeper continuation without witness recompletion, profile isolation and real budgeted handoff.

A fresh read-only review of `48b527a..6bab7d2` found no Critical/Important/Minor blockers and independently ran the then-current 288-test suite. The additive fresh-evaluation evidence follow-up `5b6e7b2` was subsequently RED→GREEN tested and self-verified, not independently re-reviewed. There is no remote-CI or unavailable external-integration coverage claim.

Implementation choices: retain the existing isolated development checkout; reuse the complete snapshot for this commit rather than re-completing it; distinguish cache-only returns from fresh evaluation. Their relevant remaining risk is early search concentration, not relaxed geometry or extra solver allowance. The original dirty checkout, frozen formal source and historical evidence are unchanged.

## Acceptance and next priority

Scoped implementation/correctness and fresh-deeper-search mechanism checks pass. **Material-quality improvement does not pass.** Keep the witness flag off and the original staged variant as control. Under the predeclared stop rule, do not launch a longer matrix for this policy. No claim of improved utilization, new record, unseen-order generalization or completed formal acceptance is supported.

The next sole priority is a read-only cost audit of complete suffix decoding: the large task still completes only eight distinct states and only one fresh deeper evaluation. Identify repeated decoding/copying work and the cost of optional modes before proposing another isolated change. Do not introduce a new candidate/commit heuristic, extend the budget, or modify geometry in this follow-up. Formal acceptance remains the separately approved multi-task, multi-seed quality gate.
