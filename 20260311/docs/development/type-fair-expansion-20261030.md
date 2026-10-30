# Type-fair expansion: fixed single-factor development check

Scope (2026-10-30): keep restricted-prefix search. Reference is staged completion at `c94a579`. Do not add candidates, widen/deepen search, change completion/warmstart quotas, modify geometry/reward, or enable repair. Existing variants keep their selection policy.

## Policy fixed before the diagnostic

The new opt-in variant is `rh_mcts_prefix_beam_staged_type_fair`, inheriting `rh_mcts_prefix_beam_staged`. The only non-identity override is `PREFIX_BEAM_TYPE_FAIR_EXPANSION=True` (default false).

At each expansion, use only the existing progressive-widening allowed set. Among its unexpanded actions choose a kind with the smallest number of already expanded children in that same allowed set. Break kind ties in fixed order single / pattern / anchor / tail; unknown kinds follow those. Within one kind retain the original prior/action ordering. This is a deterministic first policy, not selected from a tuning sweep.

Do not change the allowed set, candidate generator, priors, root noise, RNG consumption, within-kind ranking, or the PUCT formula/path once allowed actions have been expanded. The policy is active only in prefix+macro mode. Explicit no-prior ablation plus the enabled flag is rejected instead of silently changing its shuffled-order control.

Add successful expansion counts by kind, how often the chosen action differs from the original first action, and the first 32 expansion kinds. These are mechanism evidence, not quality metrics. No new dependency or handoff schema is needed.

Balanced kind access may delay a useful high-prior pattern and can regress against the staged control. The exact Greedy complete incumbent remains the unchanged floor, conditional on completing its initial decode. Fairness does not imply global optimality or board savings.

## Locked diagnostic and decision rule

- Same known development tasks: 178, 739 and 2416 parts. Seeds 1000/1001/1002, paired by task×seed, schedule seed 20261030, randomized method order within blocks.
- Reference: staged prefix. Treatment: staged prefix + type-fair expansion. Greedy run once per task. Total 21 runs.
- 30 s total budget, 3 s closing reserve, 38 s external watchdog. Input load, search, independent audit and atomic result serialization are included in the total budget. Keep failures and all regressions; no selective reruns or post-outcome policy changes.
- Freeze runtime commit, source hashes, settings, schedule and environment before execution. Reaudit saved outputs and verify result identities afterward. Private inputs/layouts/raw outputs remain outside the public repository.
- Primary result: task mean boards and net utilization. Secondary: envelope occupancy, first saving event, completion states, PUCT and type-access evidence. Seeds are within-task repetitions, not independent orders.
- Continue to later selection only if there is additional board-saving evidence against staged control; report all task wins/ties/losses and worst seeds. If merely earlier equal-quality results or more evaluations occur, stop this policy as a material-quality proposal and do not launch a longer matrix. Formal acceptance remains the separately approved 10-task×5-seed gate, including at least three improving main tasks.

## Verification

From `20260311/`:

```sh
python3 -m pytest -q tests/test_type_fair_expansion.py
python3 -m pytest -q -rs
python3 -m compileall -q src tests scripts
git diff --check
```

Before freezing runtime: 14 targeted cases passed; full suite 267 passed / 3 external-integration skips (existing dirty CutEval checkout preserved). Checks cover original order when disabled, type fairness/within-kind ordering, progressive-widening boundaries, PUCT preservation, excluded children, a single kind, bounded evidence, profile/candidate isolation at three scales, activation gates, random-ablation protection and real budgeted feasible handoff.

## Results and decision (2026-10-30)

Runtime frozen before execution: `257b27b113fffcfa30cba1f70de574b9f429f2ad`. Both methods use that runtime; the staged control has the `c94a579` policy plus additive evidence only. The following documentation and extra 178-part profile regression do not alter the tested algorithm. Input hashes and the full 21-run order were locked in advance. No policy retuning or selective reruns occurred.

Each triplet below is seed 1000 / 1001 / 1002. Net utilization is nominal part area divided by all used full-board area, including the tail board. Envelope occupancy uses the unchanged `(w+6)*(h+6)` reservation model, not measured cutter-path loss.

| Parts | Greedy boards | Staged boards | Type-fair boards | Net utilization, both methods | Envelope occupancy, both methods |
| --- | ---: | --- | --- | ---: | ---: |
| 178 | 29 | 29 / 29 / 29 | 29 / 29 / 29 | 83.7932% | 85.5369% |
| 739 | 98 | 97 / 97 / 97 | 97 / 97 / 97 | 89.3805% | 91.3207% |
| 2416 | 374 | 373 / 373 / 373 | 373 / 373 / 373 | 89.3952% | 91.1369% |

Task-level outcome: **0 wins / 3 ties / 0 losses**, sum of task mean board-count differences 0, all net-utilization differences 0 percentage points, and 0 of 9 paired board-count regressions. There is no additional material-saving signal against staged completion.

| Parts | Mean distinct successful states staged → fair | Mean actual PUCT staged → fair | Mean total s staged → fair | First below-Greedy complete event s, staged / fair |
| --- | --- | --- | --- | --- |
| 178 | 77 → 77 | 76 → 76 | 2.524 → 2.376 | none / none |
| 739 | 80 → 79.67 | 64 → 60 | 25.712 → 26.979 | 11.012, 10.808, 10.665 / 10.641, 10.560, 9.631 |
| 2416 | 8 → 8 | 0 → 0 | 27.244 → 27.229 | 21.475, 21.415, 21.545 / 6.954, 19.439, 19.349 |

Event times originate at solver evidence initialization, not input loading. The new policy can reach an equal-quality incumbent earlier, especially in one large-task seed, but that is not an improved final board count or a universal runtime gain. Medium-task total time increased in this batch and one seed completed 79 rather than 80 states. The small task still hits the unchanged 80-simulation cap. The largest still has no PUCT selection and no committed beam-depth advance (`prefix_depth_reached=0`), despite evaluated depth-1 children. Do not infer deep-prefix exploration from an active search timer.

The type-fair counters show mean reordered expansion choices of 47 / 40 / 3 across the three sizes; its first choices include single / pattern / anchor. This verifies the intervention occurred, not its utility. Different subsequent states can naturally generate different candidates, but the candidate generator and per-state limits are unchanged.

All **21/21** outputs passed independent part-identity, permitted-rotation, nominal-dimension, 6 mm tool-envelope, 5 mm trim-bound, non-overlap and utilization checks, then were reaudited after saving. Result/source hashes, runtime identity and schedule identity matched. All minimal files were atomically written within 30 s; maximum observed elapsed time after write was **27.449537458 s**. No failures/watchdog terminations or Greedy-floor violations occurred. Internal search cutoffs with a saved complete layout remain valid runs.

Final verification: **15 targeted tests**, **268 full-suite passes / 3 external-integration skips**, compileall and diff checks passed. The extra targeted case covers the actual 178-part adaptive profile; it was added after the runtime freeze without changing solver code. External dirty CutEval changes, the original dirty checkout, the frozen source and old evidence are preserved. There is no independent review or remote-CI claim for this patch.

Private evidence stays in local QA batch `type_fair_20261030`: locked manifest, progressive records, 21 full run files and `verified_summary.json`, with private run/reverification scripts. No new factory inputs, layouts or detailed outputs are uploaded to the public repository. Existing factory tasks were already examined; this does not establish unseen-order generalization.

**Decision:** scoped implementation/correctness checks pass; material-quality improvement does not. Keep type-fair expansion disabled and retain the original staged variant as the research control. The opt-in variant is retained only to reproduce this negative development result, not promoted to the default or described as a validated utilization optimization. Per the locked rule, do not launch the longer selection/formal matrix for this policy. The separately approved formal acceptance gate is unchanged and has not run.

The next sole priority is a read-only audit of actual prefix advancement and retention: identify why the large task completes only eight evaluations without PUCT/committed deeper search before proposing another policy. This does not authorize changing depth, commitment thresholds, scoring or geometry in the present patch. Candidate-feedback and type-access additions remain paused; restricted-prefix search remains the topic.
