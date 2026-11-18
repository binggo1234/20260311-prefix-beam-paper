# Matched prefix evaluation budget implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline, one fresh read-only review before factory data.

**Goal:** Establish whether current search adds material value over simple restricted-prefix scanning at twelve complete evaluations.

**Architecture:** Keep production source frozen. Build a private current-search instrumentation subclass and existing capped scan wrapper; save exact full-layout receipts, audit them independently and compare one frozen paired run.

**Tech Stack:** Existing Python/pytest, existing V3 serializer and geometry auditor.

**Spec:** [Protocol](../../development/matched-prefix-budget-20261113.md).

## Global Constraints

- Frozen production source f01b2fc; same initializer, decoder, geometry and entry RNG/cache.
- Exactly one739-piece order / seed1000 / twelve post-initial full starts per arm;30/3/38/8 s budgets.
- Preserve staged modes and pre-cap behavior; no new heuristics, increased depth or padding of inactive controls.
- Best completed capture retained only at closing; never feed early updates into promotion.
- Require complete equal counts for eligibility; retain failures and all prior evidence; no public raw data or merge.

## Review Focus

- Mid-promotion quota stop preserves completed modes and counters without a thirteenth evaluation.
- Metadata names true modes and full suffixes; matching first-two behavior alone does not validate twelve.
- Replay timeout or partial initialization cannot count as successful paired evidence.
- Common initial identity and RNG/cache match despite using separate instrumentation classes.
- Every saved receipt is bound to source geometry/score and audited without solver execution; failures cannot disappear from the schedule.

## Task 1: Frozen paired current-search vs scan experiment

Files: create private `matched_prefix_probe_20261113.py`, `test_matched_prefix_probe_20261113.py`, `run_matched_prefix_budget_20261113.py`, `audit_matched_prefix_budget_20261113.py` under the existing QA directory; update report/README only in public checkout. No runtime implementation change.

Interfaces: existing RecedingHorizonMCTS completion methods and decoder replay; existing `_SinglePrefixScanSolver` with limit12; V3 JSON serializer and existing receipt/geometry auditors. Private schema `inrp-matched-prefix-budget-v1` normalizes count/evidence fields while preserving original receipts and modes.

- [x] Write failing tests for cap12 vs cap2/no cap, actual staged modes and exact first-twelve parity; then implement minimum instrumentation. Expected: tests fail for missing cap behavior then pass with exact parity.
- [x] Test synthetic full worker plus saved-output audits, source/receipt corruption, initial mismatch, missing/failed runs and timeout handling. Expected: valid toy passes and all invalid variants reject or remain ineligible. Run full runtime suite unchanged.
- [x] Obtain one fresh read-only review; fix material findings before inputs. Commit public protocol, freeze all inputs/source/helper hashes and exact schedule.
- [x] Run exactly the two declared arms, independently audit all saved receipts and preserve all results; report fixed-count eligibility and actual material/efficiency outcomes without claiming prefix on/off or PUCT attribution.
- [x] Run task-done verification, publish aggregate report to existing draft, verify remote head/body/state, archive this task's scratch ledger.
