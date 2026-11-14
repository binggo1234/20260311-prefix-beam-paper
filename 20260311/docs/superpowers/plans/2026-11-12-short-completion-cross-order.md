# Short-Completion Cross-Order Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement inline task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Check whether the fixed739-piece improvement replicates on the two remaining known orders.

**Architecture:** Preserve all production code; reuse the immutable common-init worker/lifecycle/pair checks behind a new private two-order schedule and reference-aware summary. Freeze once after harness review, execute12 rows and audit all outcomes.

**Tech Stack:** Existing Python/pytest/multiprocessing and geometry/hash APIs; no dependencies added.

**Spec:** `docs/superpowers/specs/2026-11-12-short-completion-cross-order-design.md`

## Global Constraints

- Production src/tests/scripts identical to810e439; same on/off profiles and three completion modes.
- Exactly178/2416 pieces, seeds1000/1001/1002,12 runs once, schedule20261116,30/3/38/8 s lifecycle.
- Freeze reference29/372 panels and all identities; no historical efficacy control, retuning or selective reruns.
- Overall signal requires both eligible orders, no per-seed regression, at least one improved task mean; ties pause escalation, invalid rows make overall result inconclusive.
- Formal gate unchanged; no unseen-order, PUCT-only, primary-mode causal or optimality claim.
- Existing draft only; private raw data/results remain local; no master merge or new heuristic/matrix.

## Review Focus

- Cross-order aggregation must not let a small-order win hide invalid large-order rows or substitute pooled/best-seed counts for means.
- Historical reference mismatch in both arms must still invalidate an order, while preserving execution statuses.
- Missing/duplicate/misidentified scheduled rows and zero on primary evaluations must not pass.
- Spawn must route results to the new directory without modifying historical evidence or granting extra time.
- Geometry/mode/floor/off identity and bounded lifecycle checks must apply to every output, including saved failed rows.

### Task 1: Private cross-order harness and review

**Files:** Create QA/run_short_completion_cross_order_20261112.py, summarize_short_completion_cross_order_20261112.py, test_short_completion_cross_order_20261112.py; public spec/plan only. QA is the existing private ch2_prefix_upgrade_20261016 directory.

**Interfaces:** Consume previous.worker(conn,row,commit), collect_worker(...watchdog_s=38,cleanup_grace_s=8), INITIAL_KEYS and summarize_pairs(manifest,records). Produce build_schedule(tasks), summarize(manifest,records), new batch short_completion_cross_order_20261112/ with historical_initial keyed by task_id.

- [ ] Write RED tests:12 balanced strict rows; reject wrong task counts/duplicates; task-mean win vs best seed; all ties; failure/incomplete/mode/primary/reference/floor/off mutation withholding; valid task retained but overall invalid; missing/duplicate/method mismatch rejection.
- [ ] Run `python3 -m pytest test_short_completion_cross_order_20261112.py -q --tb=short` from QA. Expected: absent-module failures, then passes after minimal wrappers/guards.
- [ ] Implement strict schedule20261116, prelocked refs run002/run007, separate child OUT and source/dependency/HEAD checks. Summary reuses pairing, adds per-order protocol guards/global decision and all-output audit, no production edits.
- [ ] Run new plus prior25 private checks and project full suite; py_compile/compileall/diff checks. Expected: all runnable pass; only existing3 CutEval skips.
- [ ] Commit spec/plan, supply private harness paths/hashes and review focus to one fresh read-only reviewer before outcomes. Important/Critical fixes get one RED→GREEN pass. Expected: ready to freeze, no unresolved blocking findings.

### Task 2: Frozen12-run check, audit and draft update

**Files:** New private batch only; public docs/development/short-completion-cross-order-20261112.md; private roadmap, PR body, review/ledger/publication receipts.

**Interfaces:** Consume frozen Task1 schedule/reference/summary; produce12 immutable outcomes, verified_summary and aggregate progress/report without production changes.

- [ ] Lock hashes/runtime/refs/schedule before first outcome, run new runner once. Expected:12 outcomes retained, no retries/retuning; failures allowed but disclosed.
- [ ] Run summary to reaudit each output, initial pairs/cross-seed/reference/off, geometry/material/layout fingerprints/modes/budget/source identity. Expected:12 audit results or explicitly withheld comparisons with all failures preserved; no new packing replay.
- [ ] Write every seed/mean/net/envelope/time/failure and exact development decision with formal/unseen/causal limits. Update six-item roadmap and existing PR body. Expected: report equals saved verified summary; no outcome-dependent extra matrix.
- [ ] Verify combined tests, compile/diff, public-path whitelist; commit/push authorized draft and verify remoteHEAD/body/master/clean state. Expected: OPEN/draft, masterbd6022e unchanged, no remote-CI pass assumption. Archive ledger recoverably in batch.

Self-review: spec requirements map to these two tasks; interfaces use the existing worker row protocol; task-level means/global eligibility and all five review risks are covered by tests/audit. No new execution-approval checkpoint under the user's standing technical delegation.
