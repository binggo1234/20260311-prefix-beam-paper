# Complete board-saving witness advancement

The user approved the single-factor proposal and delegated implementation choices. Keep restricted-prefix search, staged completion and the existing geometry/budget/Greedy floor. Build in the existing isolated development checkout, not the original dirty checkout.

## Approaches and decision

Lowering all visit gates would also advance unproven prefixes. Adding more candidates does not address the observed missed commit checkpoint. Use only a strict, complete board-saving witness to interrupt the current simulation batch and advance its existing path. This is the selected approach; no new thresholds or dependencies.

## Contract

- Default-off `PREFIX_BEAM_WITNESS_ADVANCE`; variant `rh_mcts_prefix_beam_staged_witness` inherits staged completion. Activate only in prefix+staged mode; original variants remain unchanged.
- Capture incumbent panel count before evaluation (staged evaluation updates the incumbent internally). After successful completion and backup, trigger only if the returned full sequence covers the order exactly and uses strictly fewer boards than that prior incumbent.
- The entire selected path must be a legal existing commit chain, length at most `COMMIT_MAX_STEPS`, leaf depth at most `PREFIX_BEAM_DEPTH`, every edge admissible under existing single fallback restrictions. A terminal leaf or an expired search budget does not trigger advancement. A deeper inadmissible witness still keeps its complete incumbent but does not bypass limits.
- Return an optional witness chain plus its already completed snapshot from `_run_simulations`. The prefix beam consumes it before the unchanged round-floor check, detaches it through the existing path, and ranks it from that saved complete snapshot. Do not call another completion or mark partial fidelity as a full cache entry. No pending witness state crosses batches or roots.
- All ordinary paths, including disabled/no-witness paths and ordinary commit ranking, remain unchanged. Early concentration may lose alternatives; the experiment must report regressions.
- Add bounded event evidence and counts for strict triggers, advances, and successful deeper evaluations after a witnessed commit. Separately count deeper complete returns that performed at least one fresh successful completion-mode replay; cache-only returns are not fresh work. A depth counter alone is not mechanism success.

## Experiment and acceptance

Freeze runtime, hashes, schedule and rules before results. Concurrent staged control vs witnessed treatment: 178/739/2416 parts, seeds 1000/1001/1002, 30 s total / 3 s closing, randomized method order within task×seed blocks. Greedy once per task; 21 runs. Keep every failure and no selective reruns or retuning.

Solver/input/audit/minimal serialization remain inside 30 s. External watchdog 38 s. Correct the private diagnostic's overly short process-exit grace before locking the new batch: use 8 s post-result cleanup grace for both methods, never extra solver time; track result validity and process exit separately. Preserve all old failures. Small known cleanup test must verify grace expiration/failure retention before the batch.

Mechanism evidence requires a triggered early commit followed by an actually completed deeper evaluation (absolute depth at least 2). Quality evidence is additional final board savings versus concurrent staged control, never historical 373-board timing. No extra savings means stop this policy's long-matrix escalation, even if deeper search works. Formal multi-task acceptance remains unchanged and has not run. Inputs/layouts/raw results stay private; publish only code/tests/aggregate results to the existing draft PR, not master.
