# CLASS06 Feasibility Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this bounded research diagnostic task by task. User has explicitly opted out of further permission/design questions; implement in the current research checkout.

**Goal:** Determine whether `cl_06_100_02` has a fixed-orientation, zero-clearance packing into at most three 300×300 boards.

**Architecture:** Standard CP-SAT mandatory rectangles in a strip of three adjacent boards. For each item of size (w,h), x belongs to the union `[bW,bW+W-w]` for b=0..2 and y belongs to `[0,H-h]`. One NoOverlap2D covers all rectangles. The disjoint x domains forbid crossing board boundaries. A separate, solver-free verifier checks any returned local-board witness.

**Tech Stack:** Existing Python 3.9, OR-Tools 9.15.6755, pytest; no installations or production imports.

**Spec:** `docs/development/opportunity-certificates-20261126.md`, final section; user approved its three-outcome protocol. Implementation choices and proof below are the agent's choices, not additional user-specified requirements.

## Global Constraints

- Research reference only; retain restricted-prefix theme, existing baseline and feedback negative result. Do not change default decoder, factory data or main Word document.
- Only original `CLASS/CLASS/cl_06_100_02.ins2D` from the previously fingerprinted CLASS.zip. Native dimensions and complete UID inventory; fixed orientation, trim/kerf 0, no guillotine constraint.
- Exactly one real-case solver invocation: 60-second solver limit, 8 workers, seed 20261126, no hint/objective/manual search strategy. Save model, parameters, logs, source hashes, build/solve/process timings; no follow-up run to improve result.
- FEASIBLE/OPTIMAL plus independently valid witness => optimum3 from area LB3. INFEASIBLE of complete model plus old valid4 witness => optimum4. UNKNOWN/timeout => remain [3,4]. MODEL_INVALID or verification failure => invalid diagnostic.
- Tiny synthetic validation solves are separate from the one CLASS06 run; no statistical/generalization claim.

## Equivalence and integer coordinates

Each board packing maps into the strip by x'=x+bW; conversely each domain-legal rectangle has one board ID and local x. Rectangles on different boards have disjoint x interiors, while same-board separation is exactly NoOverlap2D. No item is optional and no rotation is introduced.

Integer coordinates do not exclude real-coordinate packings of integer-sized axis-aligned rectangles. Given any packing, for each pair on a board choose one true non-overlap direction. The chosen horizontal inequalities form an acyclic graph because widths are positive; set each x to its longest-path sum from origin. Coordinates become integer and do not exceed original x. Do the same independently for vertical inequalities. Each chosen separation remains true and right/top bounds still hold. Thus the integer model is equivalent for feasibility, not a rounded approximation.

## Review Focus

1. Across-boundary placements must be rejected even when the enlarged strip could fit them; test three3×2 pieces in two5×2 boards (infeasible).
2. Non-area geometric infeasibility: four6×4 pieces do not fit one10×10 board, but do fit two.
3. Full identity, fixed orientation, exact boundary contact and same-board overlap: independent verifier must reject corruptions and accept legal edge contact.
4. UNKNOWN must carry no witness or infeasibility claim; exercise with zero time limit on a synthetic case.
5. Source substitution or malformed original member must fail before solve; test fingerprint and count checks. No original-case solve in tests.

### Task 1: Standard model, witness checker and bounded runner

**Files:** Create `research/class06_feasibility_20261126/{model.py,verify_witness.py,run_reference.py,test_reference.py,README.md}`.

**Interfaces:** `build_model(shapes,width,height,boards)` returns model and per-UID x/y variables; `solve_reference(...,time_limit_s,workers,seed)` returns solver status, optional local placements and timings. `verify_witness(shapes,width,height,boards,placements)` returns boards used or raises ValueError. `run(archive,out)` validates fixed original source and writes immutable one-run artifacts.

- [ ] Write tests first; run and record expected missing-implementation failure.
- [ ] Implement minimal model/verifier/runner. Validate generated model before solve; preserve full status and optional witness.
- [ ] Run tests including tiny grid exhaustive oracle comparisons; no production imports. Ensure runner exercises real file I/O, source rejection, and create-only output directory using a tiny source fixture with diagnostic solve stub disabled by source mismatch where appropriate.
- [ ] Independent read-only review before real run; fix correctness issues, rerun relevant tests.
- [ ] Freeze source, protocol, dependency fingerprints, original inputs, previous results, production tree and Word digest in local QA. Export and hash actual model before real solve. Run one bounded CLASS06 invocation; parent timeout90s, report if triggered.
- [ ] Verify any witness without solver or source model helpers; preserve unknown/infeasible statuses correctly.
- [ ] Run repository pytest, report skips/failures accurately. Publish research reference, outcome and receipt to existing branch. Check remote head and clean checkout.

### Task 2: Decision report

**Files:** Add `docs/development/class06-feasibility-results-20261126.md`, result assets, README link; append dated outcome to previous certificate report rather than replacing its historical table.

- [ ] Report exact outcome, verified bounds, validation, source/model limits and no new factory result.
- [ ] Fix next research question from this outcome; do not call the standard reference an innovation or tune a failed feedback mechanism.
- [ ] Review final report against saved artifacts; check links, JSON and whitespace; commit and push exact task files.
