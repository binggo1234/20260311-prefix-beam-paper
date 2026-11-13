# Budgeted single-prefix scan implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan inline. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether extending existing single-entry coverage, without a new ranking heuristic, delivers savings within the existing 30 s pipeline budget.

**Architecture:** Add an opt-in experimental handoff adapter. Reuse the common Greedy/short-edge initializer, decoder, complete-incumbent scoring and absolute deadlines. Visit every UID at most once in the existing full early-entry sort order, move it to the front, preserve the canonical short-edge suffix, and stop at the deadline or an optional explicit evaluation cap. Preserve the frozen two-completion adapter and all defaults.

**Tech Stack:** Existing Python package, pytest, existing TotalBudget/geometry auditors/process collector. No new dependencies.

**Spec:** [Single-prefix landscape and next priority](../../development/single-prefix-landscape-20261113.md), especially sections 3, 5 and 6; latest user authorization permits autonomous implementation and necessary verification.

## Global constraints

- Restricted-prefix topic; actual prefix length exactly one; no new shape rule, witness override, deeper tree, PUCT, repair or completion mode.
- Same existing early-entry key; bypass scout truncation only. Every UID remains distinct.
- Same complete common initialization and initial cache/RNG state as frozen two-single reference. Verify first two outputs at matched quota.
- Absolute pipeline 30 s / closing 3 s; parent watchdog 38 s / cleanup 8 s. Failed replay starts count. Final incumbent never worsens the completed initial floor.
- Keep historical controls unchanged. Compare fixed two-single against full scan under the same total cap; explicitly report different evaluation counts, not a matched-count or new heuristic claim.
- Factory data/layouts/identities and detailed receipts remain private; use existing isolated checkout/draft, never modify master or protected user checkouts.

## Review focus

- Pool truncation accidentally excludes IDs or changes the original first two entries: verify beyond configured scout limits and compare real outputs with frozen reference.
- Last replay times out after earlier savings: retain completed best, distinguish started/completed/failed, never retry.
- Initialization partially completes: retain available complete floor and label incomplete initialization, not an eligible quality pair.
- Evidence export/geometry/serialization overruns: keep absolute closing deadline; checkpoint returned handoff before audits and exclude incomplete runs.
- Invalid profile, empty/zero quota and repeated identities: reject unsupported inputs, allow an explicitly zero post-initial quota, enumerate all UID alternatives once.

## Task 1: Adapter, regression coverage, bounded comparison and report

**Files:**
- Create `src/inrp/single_prefix_scan.py`: `solve_single_prefix_scan_handoff(request, budget, *, max_evaluations=None)`.
- Create `tests/test_single_prefix_scan.py`: real-decoder parity, full coverage and deadline/failure contracts.
- Create `docs/development/single-prefix-scan-20261113.md`: protocol before data, then six-part aggregate outcome report.
- Modify `README.md` only to link the new report.
- Private QA scripts reuse existing loading, evidence audit and process collection; do not rewrite older files.

**Interfaces:** Uses PackingRequest/PackingHandoff, TotalBudget and RecedingHorizonMCTS initialization/decoder/close. Exports versioned `solver_meta.single_prefix_scan` initial/final/evaluation receipts, entry identities, counts and stop reason. Does not modify `early_prefix_control` or its schema.

- [ ] Write failing tests, run pytest, confirm missing scan behavior is the failure.
- [ ] Implement minimal opt-in adapter, run new tests plus frozen two-completion tests.
- [ ] Prepare fixed six-run schedule: known 178/739/2416 orders, seed1000, reference and scan once each, deterministic randomized block/arm order; freeze source/input/worker/auditor identities before execution. No retries or extra probes.
- [ ] Run full runtime suite and meaningful private schedule/receipt checks; have one fresh read-only review of this change, fix material findings before data.
- [ ] Commit clean runtime/protocol, run six bounded workers serially, independently audit all saved complete layouts and initial/first-two parity. Preserve failures; classify wins/ties/inconclusive honestly.
- [ ] Write aggregate report; check links/whitespace/source/evidence identities; push existing development branch and update existing draft PR. Report material result, actual evaluation budget and limitations; no merge or formal/unseen claim.

Expected validation: new regression failures first, then all relevant tests pass; complete eligible pairs share initialization and first-two identities; measured board count/net utilization decide outcome, never hardcode desired 28/95 into method acceptance.
