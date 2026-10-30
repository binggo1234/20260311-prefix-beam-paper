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

Runtime commit and diagnostic results will be recorded after the fixed batch. No efficacy result is claimed by this implementation note.
