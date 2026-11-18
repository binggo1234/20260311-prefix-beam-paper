# Common Initialization Prefix Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the existing prefix-search procedure's incremental material savings after identical strong deterministic initialization.

**Architecture:** Opt-in profiles share one initializer in RecedingHorizonMCTS; off returns before any search, on uses the existing formal prefix path. A private frozen paired batch audits initial identity and final outputs.

**Tech Stack:** Existing Python/pytest/decoder/budget/audit APIs and hashlib; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-11-common-init-prefix-ablation-design.md`

## Global Constraints

- Keep restricted-prefix theme/limits, geometry, search heuristics and existing defaults unchanged.
- Common initializer is Greedy then short-edge; skip all other initialization in both arms.
- Budgeted-only profiles; every decode uses the absolute search deadline; retain only completed solutions.
- 18-run paired diagnostic, 30/3/38/8 s lifecycle; no retuning/selective reruns; initial mismatches invalidate comparisons.
- Preserve dirty checkout, frozen evidence and private factory data; update existing draft only.

## Review Focus

- Off must never fall through to alternative MCTS/beam/SA/repair or even root warmstart.
- Deadline expires during second initial order: preserve complete Greedy but flag an invalid ablation pair.
- Complete common floor remains immutable after partial/timed-out search.
- Paired RNG state, decoder configuration and initial fingerprints agree; only search gate differs.
- Missing formal deadline/non-prefix profiles reject before any work; defaults keep the original warmstart path.

### Task 1: Common initializer and strict search gate

**Files:** Modify `src/inrp/cfg.py`, `mcts.py`, `runner.py`; create `tests/test_common_initial_ablation.py`.

**Interfaces:** Consume baseline_sequence/short_edge_sequence, decoder.replay_order, _update_global_best(record_archive=False), _formal_result_from_incumbent and existing formal prefix solver. Produce PREFIX_COMMON_INIT_ABLATION=False/PREFIX_ABLATION_SEARCH_ENABLE=True defaults and two matching profiles; initializer metadata and exact on/off gate.

- [ ] Write tests for fixture [(30,50),(15,35),(55,50),(30,35),(20,70),(15,75)] on100x100: Greedy2/short-edge1; both share identical seed geometry/score/fingerprints and RNG. Off invokes no alternative pipeline; on reaches existing prefix gate. Include deadline interruptions before/after first complete seed, floor immutability after search interruption, profile equality across tiers, invalid configurations and real budgeted handoff.
- [ ] Run `python3 -m pytest tests/test_common_initial_ablation.py -q --tb=short`; Expected: absent common metadata/gate/profile failures.
- [ ] Implement only specified initializer/gates/profiles; run full `python3 -m pytest -q -rs`, compileall and diff checks; Expected: green suite with existing three external skips only.
- [ ] Commit, obtain one fresh review of new range from d33a834, RED→GREEN any Important corrections, and freeze runtime.

### Task 2: Paired experiment, audit and publication

**Files:** Private QA runner/summarizer/tests and `common_init_prefix_ablation_20261111/`; public development report; private roadmap/PR body.

**Interfaces:** Consume two profiles/metadata and existing private loader, worker lifecycle and geometry audit. Produce locked18-row schedule, all outputs/failures and initial-matched task-level outcome.

- [ ] Add failing tests for allocation, initial mismatch/incomplete pair exclusion, no best-seed-as-mean reporting and failures. Expected: missing-feature failures, then pass after implementation.
- [ ] Freeze runtime/dependency/input hashes and schedule seed20261114, execute18 rows once. Expected: one result/failure per row, no runtime mutation.
- [ ] Reaudit every saved layout and initial fingerprints; assert off no search and exact initial result, on floor preserved. Expected: complete/matched pairs only contribute task comparisons; all failures/mismatches retained.
- [ ] Write all seeds/means/both material metrics and six-item progress, full final checks, push/update existing draft. Expected: remote HEAD/body match, draft remains open and master unchanged.

Execute inline under explicit user authorization and prior delegation, without another design checkpoint. One fresh code reviewer is required by executing-plans. Recoverably archive this plan's ledger with its private batch.
