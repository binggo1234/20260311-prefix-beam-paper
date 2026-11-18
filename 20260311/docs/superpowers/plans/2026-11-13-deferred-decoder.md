# Deferred decoder materialization implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline, one fresh read-only review before factory data.

**Goal:** Remove losing-candidate layout construction while preserving exact decode semantics and increasing useful evaluation capacity.

**Architecture:** Score existing/new-board candidates using the original arithmetic and micro cache. Retain winner data, then materialize its board list, layout hashes and DecodeResult once. Freeze the former decode method as a test oracle; leave geometry, caches, ranking and defaults unchanged.

**Tech Stack:** Existing Python/pytest; no dependency changes.

**Spec:** [Deferred decoder protocol](../../development/deferred-decoder-20261113.md).

## Global constraints

- Exact part identity, permitted rotation, geometry and lexicographic tie order; same UID cache keys, calls and contents.
- Preserve evaluation of the new-board candidate and its error behavior even when an old board fits.
- Keep per-board deadline checks and usable prefix snapshot/cache after interruption.
- No new prefix/shape/search heuristic; touch decoder construction only.
- Fixed ten cold replays, then one2416 paired scan under the declared budgets. No selection of favorable reruns or pooling old profiles into timing evidence.
- Existing isolated development checkout/draft; protected workspaces/raw inputs remain untouched/private.

## Review focus

- Multiple feasible boards and tied scores must pick identical first winner.
- Floating accumulation order and every snapshot/hash must match the frozen oracle.
- Duplicate shapes, asymmetric rotations and shared-cache revisits must preserve UIDs.
- No feasible existing board, oversized new-board failure and deadline/cache-hit boundaries must match prior behavior.
- Evidence overhead remains inside timed pipeline; fixed-work and fixed-time claims are distinct.

## Task 1: Exact deferred construction and bounded verification

Files: modify `src/inrp/mcts.py::OmnipotentDecoder.decode`; add `tests/decoder_eager_reference.py` and `tests/test_decoder_deferred.py`; update linked report and README. Private diagnostics reuse the V3 serializer and geometry auditors without modifying frozen scripts.

- [x] Freeze old decode; write and observe failing winner-only construction test, plus oracle equivalence cases.
- [x] Implement minimal deferred construction; run targeted and full tests.
- [x] Review this change once in fresh context; resolve material findings before data.
- [x] Lock code/input/dependency identities and schedules; run the declared fixed-work and fixed-time comparisons, independently audit saved outputs and preserve failures.
- [x] Report exactness, actual performance/coverage/material outcomes and limitations; verify and publish existing draft branch.

Execution ruling: the fixed-time pair discovered a new371-panel result at evaluation12. Freeze that first saving sequence and perform exactly two supplied-sequence cold confirmations (old/new) under one60 s diagnostic cap; these are reproducibility checks, not extra search candidates, timing replicates or reruns of the budget pair. Both exactly reproduce the witness. Final saved-output audit also rebuilds the full frozen source-hash map, addressing the review's Minor runner limitation.
