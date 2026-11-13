# Board-local blueprint reuse implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline, one fresh read-only review before factory data.

**Goal:** Reduce identical local geometry computations while retaining exact UID-safe decoder behavior.

**Architecture:** Memoize pure placement blueprints on unchanged Board objects; keep per-UID transitions and hashes independent. Freeze the former micro method as oracle and reuse existing private replay/scan auditors.

**Tech Stack:** Existing Python and pytest; no new dependency.

**Spec:** [Protocol](../../development/blueprint-memo-20261113.md).

## Global Constraints

- Exact query-time dimensions, rotations and mode; no rounded shape equivalence.
- Board mutation clears memo; clones start empty; initialization during query cannot detach stored results.
- Preserve UID cache keys/contents/counters, scoring, identity and deadline behavior.
- No new heuristics, longer prefixes, changed budget or merged PR; raw factory data stays private.
- Six shuffled fixed-work pairs; retain only on exactness and lower median baseline time, then one fixed-time large-order pair. No favorable reruns.

## Review Focus

- Mutable parts queried after decoder construction must use current nominal dimensions and rotations.
- Shared prefix boards used by another decoder must respect mode and global rotation differences.
- Failed fit reuse and free-rectangle initialization must not bypass invalidation.
- Rounded-near shapes and equal geometry with distinct UIDs must retain independent post-placement hashes.
- Evidence serialization and auditing must fit the unchanged pipeline deadline; saved source/dependency hashes must be rechecked.

## Task 1: UID-safe geometry memo and bounded verification

Files: modify `src/inrp/packer.py::Board/_invalidate_board_cache` and `src/inrp/mcts.py::_micro_decode`; add `tests/decoder_blueprint_reference.py`, `tests/test_decoder_blueprint_memo.py`; update report and README. Private helpers remain outside the public repository.

Interfaces: existing `best_local_placement`, `apply_local_blueprint`, UID micro caches and V3 evidence schema remain unchanged.

- [ ] Freeze old micro method; add a failing repeated-query test requiring one geometry calculation with different actual UID/hash results; cover invalidation, clone, exact shape/nominal/rotation keys, fresh decoder and legacy cache equality.
- [ ] Run `PYTHONPATH=src python3 -m pytest -q tests/test_decoder_blueprint_memo.py`; expect the reuse test to fail before implementation, then all cases pass after the minimal cache change. Run full suite, expect no new failures/skips.
- [ ] Prepare and test frozen private schedules/gates; obtain one fresh read-only review before private inputs; fix material findings.
- [ ] Commit source/protocol, lock all hashes, run six fixed-work pairs; if gate passes run the one timed pair. Independently audit saved output, all legacy source/dependency/reference and historical JSON hashes. Preserve every failure.
- [ ] Report measured effects and limits, run task-done full suite, publish to existing draft branch and verify remote state. No formal efficacy or novelty claim.
