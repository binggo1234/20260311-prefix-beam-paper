# First-Saving Receipt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover an independently auditable first-saving prefix without changing search policy.

**Architecture:** Default-off diagnostic flag/profile; node path metadata passed to accepted-incumbent capture; save one persistent snapshot and serialize during existing close. One private frozen worker audits both saved first and final outputs without replay.

**Tech Stack:** Existing Python/pytest/dataclasses, persistent layouts and budget/audit APIs. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-12-first-saving-receipt-design.md`

## Global Constraints

- Same common short-edge on search; diagnostic profile differs only by flag/identity. No new heuristic/RNG/cache-key/geometry/search/deadline changes.
- Default-off; first accepted improvement below completed common floor only, capture once, no replay, no later overwrite.
- Exact actual prefix/macro/mode/cache/event/sequence/layout association; incomplete context is explicit diagnostic failure.
- All capture/export/audits/write inside existing30/3 s;38/8 s worker lifecycle. No deadline extension.
- One739 seed1000 factory run once, no retries/outcome tuning/larger matrix. Initial96 receipt identity frozen; formal gate unchanged.
- Existing draft only, preserve dirty/frozen/historical workspaces; private raw receipts stay local.

## Review Focus

- An initialization or equal-panel improvement must not become the first-saving event; later improvements must leave its snapshot/receipt unchanged.
- Root detachment and ranking-time completion must retain actual macro ancestry; depth cannot substitute for piece count.
- Staged/full cache reuse must distinguish winning-mode freshness from other modes' evaluations.
- Off/default profiles must retain behavior, RNG/cache state and adaptive settings; close export never decodes or extends budget.
- First receipt and final result must each be audited; missing/invalid/late results remain recorded, not silently repaired or rerun.

### Task 1: Default-off capture, context and budgeted export

**Files:** Modify src/inrp/cfg.py, mcts.py, runner.py; create tests/test_first_saving_receipt.py; spec/plan.

**Interfaces:** Consume existing SearchNode/MacroAction, _evaluate_prefix_with_baseline/_update_global_best/_formal_result_from_incumbent and handoff meta. Produce diagnostic chain/context, meta first_saving_receipt and opt-in rh_mcts_prefix_short_receipt; handoff API unchanged.

- [ ] Write real decoder RED tests using seven-piece common2→prefix1 and ten-piece common3→prefix2 toy fixtures: exact first capture/common-floor exclusion/default-off, later strictly better same-panel incumbent, detached roots/ranking context, fresh/staged/full cache classification, JSON export/no replay/expiry, all-tier profile parity and real budgeted handoff.
- [ ] Run `python3 -m pytest tests/test_first_saving_receipt.py -q --tb=short`; Expected: absent receipt/profile/context failures, then passes after minimal implementation.
- [ ] Implement one-capture snapshot/context and close serializer; add profile/flag only. Run full `python3 -m pytest -q -rs`, compileall/diff. Expected: all runnable pass; only existing3 CutEval skips.
- [ ] Add/test private strict one-row runner and receipt reconstruction audit before review. Expected: corrupt prefix/macro/hash/event/origin/geometry/budget fails, non-reproduction retained.
- [ ] Commit code/spec/plan/tests; fresh read-only review of branch/private harness, Important fixes get one RED→GREEN pass; then freeze. Expected: no unresolved blocking findings and no factory outcome yet.

### Task 2: One frozen diagnostic and publication

**Files:** Private run_first_saving_receipt_20261112.py, audit_first_saving_receipt_20261112.py, test_first_saving_diagnostic_20261112.py and first_saving_receipt_20261112/; public docs/development/first-saving-receipt-20261112.md; private roadmap/body/ledger.

**Interfaces:** Consume first_saving_receipt, frozen context/profile/runtime and existing geometry/lifecycle APIs. Produce one immutable output with separately audited first/final layouts, reconstruction certificate and diagnostic status.

- [ ] Lock739 input/seed1000/profile/sources/dependencies/known common floor96/reference run006, then run exactly once30/3/38/8. Expected: one retained outcome, no retry or outcome-driven source edit.
- [ ] Reaudit first/final identities/geometry/fingerprints, exact prefix+suffix reconstruction, macro flattening and first-event association plus budgets. Expected: passed diagnostic only with real first95 receipt; otherwise preserve missing/invalid/incomplete status and report it.
- [ ] Report version/scope/budget/prefix/material/time/audits/failures/limits and next bounded recommendation, update six-item roadmap/PR body. Expected: raw IDs/layouts private; no new quality/formal/causal claim.
- [ ] Run final relevant tests/compile/diff/public-path whitelist; commit/push/update existing draft and verify head/body/master/clean state. Expected: OPEN/draft, masterbd6022e unchanged, no remote-CI pass assumption. Archive ledger recoverably.

Self-review: both tasks use the same receipt/profile/schema; context and cache/immutability/deadline risks have tests; no separate handoff API or unrelated refactor. Continue under the user's approved diagnostic and standing technical delegation.
