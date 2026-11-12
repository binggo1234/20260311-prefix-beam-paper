# Short-Edge Primary Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test prefix increment over the shared96-panel seed with short-edge primary completion on one order only.

**Architecture:** Minimal mode registration/key/receipt and two derived opt-in profiles. Reuse the frozen common-init worker/pairing checks in a separately identified six-run private batch.

**Tech Stack:** Existing Python/pytest, deterministic ordering/decoder and lifecycle/audit APIs; no dependency changes.

**Spec:** `docs/superpowers/specs/2026-11-12-short-edge-completion-design.md`

## Global Constraints

- Only primary completion changes: (baseline,long_edge,strip_bias)→(short_edge,long_edge,strip_bias).
- Preserve initialization, search settings, geometry and default profiles; no new candidate/scoring heuristic.
- One739-piece order, three paired seeds, six runs,30/3/38/8 s lifecycle, no retuning/selective reruns.
- Mean below96 is a development signal only; unchanged formal gate. No primary-mode causal claim from historical timing.
- Preserve private data and historical/dirty/frozen workspaces; existing draft only.

## Review Focus

- Unsupported mode filtering must not silently remove short_edge.
- Short-edge sorting must exactly match initialization for all remaining subsets and UID ties, including tiny positive dimensions.
- Staged primary/cache/deadline handling and both fallback modes must remain intact.
- New profiles must differ from their parent only by mode tuple/identity and match each other except search gate at all tiers.
- Default area-primary profiles and the shared common-init/off-floor contracts must remain unchanged.

### Task 1: Register mode and isolated profiles

**Files:** Modify `src/inrp/mcts.py`, `src/inrp/runner.py`; create `tests/test_short_edge_completion.py`.

**Interfaces:** Consume existing short_edge_sequence/_baseline_completion_order/staged evaluation/common-init profiles. Produce new short_edge primary key, active-mode metadata, rh_mcts_prefix_short_completion_on/off with identical adaptation.

- [ ] Write failing tests for exact subset/tie/tiny-dimension ordering; parsed/recorded modes; first real staged evaluation, fourth-state promotion retaining two fallbacks; interrupted primary replay retaining the complete floor; profile parity/old defaults and real common-init receipts.
- [ ] Run `python3 -m pytest tests/test_short_edge_completion.py -q --tb=short`; Expected: absent/filtered mode and missing profile failures.
- [ ] Implement minimal branch/allowlist/receipt and derived profiles. Run full `python3 -m pytest -q -rs`, compileall and diff checks; Expected: runnable suite green, only existing3 CutEval skips.
- [ ] Commit, obtain one fresh code review from72d8a83; RED→GREEN Important fixes if any, then freeze runtime.

### Task 2: Six-run paired development check and publication

**Files:** Private QA runner/summarizer/tests and short_edge_completion_ablation_20261112/; public development report; private roadmap/PR body.

**Interfaces:** Consume Task1 modes/profiles/receipts and prior frozen common-init worker/pair checks. Produce six locked rows, all saved outputs/audits and complete-seed mean comparison.

- [ ] Add failing tests for six-row allocation, wrong mode/floor exclusion, failed seed withholding and mean-only signal; implement/test reused-harness wrappers. Expected: missing-feature failures then passes.
- [ ] Freeze sources/input/dependencies/schedule seed20261115, execute six rows once. Expected: all outcomes recorded; no runtime mutation, retries or outcome tuning.
- [ ] Reaudit source/result identities, active modes, initial historical/paired receipts, off no-search and final geometry. Expected: all-seed means only when protocol complete; all failures retained; signal only if mean_on<96.
- [ ] Write all per-seed/material/timing/limitations and six-item progress; final checks; push/update draft. Expected: verified remoteHEAD/body, draft open, master unchanged.

Execute inline under the user's approved scope and prior technical delegation; no new design/permission checkpoint. One fresh reviewer is required by executing-plans. Archive this plan's ledger recoverably with the private batch.
