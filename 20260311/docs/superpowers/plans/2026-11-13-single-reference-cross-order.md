# Fixed Single Reference Cross-order Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run exactly the two approved known-order single-reference validations and distinguish execution validity from material gain.

**Architecture:** New private schedule/routing/audit wrapper reuses the unchanged two-completion worker, solver adapter and raw-receipt auditor. Public changes are this protocol/plan and a factual report.

**Tech Stack:** Existing Python/pytest/multiprocessing/absolute budgets; no new dependency.

**Spec:** `docs/superpowers/specs/2026-11-13-single-reference-cross-order-design.md`

## Global Constraints

- Exactly178/2416-piece existing orders, seed1000 once each, only single_reference; schedule seed20261117.
-30 s pipeline/3 s closing/38 s parent watchdog/8 s bounded cleanup.
- Two common initial replays separately counted; at most two later full starts, failed starts included; no refill/retry/retuning.
- Runtime source identical to57d4672; no new heuristics/default/profile/geometry changes, old scripts/results immutable.
- Source/ordered-parts/full initial identity matched to historical29/372 references; historical search is background only.
- Both eligible with at least one panel saving = positive known-order signal; both ties = no replication; any invalid/incomplete = inconclusive.
- Private raw data/harness; public protocol/aggregate report; authorized draft only, no master merge.

## Review Focus

- Extra/duplicate/missing/wrong-order/seed/arm rows must not enlarge scope or create a false two-order signal.
- Historical reference seed1002 must bind deterministic parts/initial identity without becoming an efficacy/time/RNG comparator.
- Spawn child output routing and failure/orphan recovery must preserve new output isolation without editing old workers/results.
- Full receipt/geometry/budget/record validation must precede any material conclusion; incomplete execution cannot become a negative.
- Source/reference/dependency changes and unclaimed artifacts must be visible rather than silently excluded.

### Task 1: Protocol-specific private harness and tests

**Files:** create privately under QA `run_single_reference_cross_order_20261113.py`, `audit_single_reference_cross_order_20261113.py`, `test_single_reference_cross_order_20261113.py`; public spec/plan above. Existing runtime/tests/scripts stay unchanged.

**Interfaces:** consume previous `worker(connection,row,commit)`, `collect_worker`, `inspect_payload`, `validate_run_protocol`, `audit_failed_record`; produce `build_schedule(tasks)`, protocol-specific saved-record validation and `summarize(reports, failures)` plus one-shot runner/auditor mains.

- [ ] Write failing tests for exactly2 deterministic reference-only rows; reject wrong sizes/duplicate identities/extra rows; genuine generated-input output routing; identity/protocol corruptions and strict gain/tie/incomplete classifications.
- [ ] Run private pytest; expect missing/stub functionality failures. Implement minimal wrapper and validator, keeping all source/historical data immutable.
- [ ] Run private tests plus existing32 receipt and5 lifecycle checks and full runtime suite; expect all pass/only existing3 optional skips. Compile and whitespace checks exit0.
- [ ] Commit public protocol/plan and request one fresh implementation review covering both private harness and whole new public range. Fix Critical/Important findings RED→GREEN before outcomes; no second review.

### Task 2: Freeze, run once, audit and publish

**Files:** private `single_reference_cross_order_20261113/`, QA roadmap; public `docs/development/single-reference-cross-order-20261113.md`; private prepared PR body.

**Interfaces:** consume reviewed frozen Task1 code and historical references; produce two preserved outcomes, independently verified valid/tied/saving/inconclusive decision, six-part aggregate report and remote publication receipt.

- [ ] Freeze input/source/dependencies/references/protected workspaces and run one two-row batch. Expect exactly2 retained scheduled records, at most2 later replay starts each; quality not predetermined.
- [ ] Run saved-output auditor without packing replay. Require ordered parts/full initial identity, budget/protocol/geometry/record bindings; failures/orphans preserved and excluded. Report incomplete without retry when necessary.
- [ ] Write factual report/roadmap with scope, counts/budgets, panel/net/envelope metrics, validation, limits and decision. Recheck all frozen identities; compile/diff checks exit0.
- [ ] Commit report, push authorized branch and update existing draft with prepared structured body. Verify exact remote head/body, unchanged master, clean checkout and public path whitelist; empty checks are not CI success.
