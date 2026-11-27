# ASH sequence-policy reference, frozen before new packing outcomes

2026-11-26. User approved the preceding direct-reference plan. Scope: implement a reference policy, not a new heuristic or default solver. Theme remains restricted prefixes. No factory matrix, no thesis edits, no merge to master. Existing factory no-grain/trim5/tool6 settings remain unchanged.

## Source and adaptation

Oliveira, Gamboa, Silva (2021), DOI10.1007/978-3-030-85476-8_8, local full-text `method_difference_20261114/sequence2021_pdf.pdf`, algorithms1–2 and section4.1. Generate output by resetting j to the first remaining item at each output position, accepting if random>=alpha, otherwise moving cyclically. Remove accepted item and stop the inner loop before modulo on an empty queue. Start alpha=.1, add .002 after a non-improving full sequence, cap .65; on strictly fewer boards adopt that sequence as base and reset .1. Same-board secondary improvements do not update ASH's base. These rules preserve the paper's board-count objective.

Two explicit adaptations: replace the original ERS geometric constructor by our existing decoder, and replace the original efficiency initialization by the common existing Greedy+short-edge strong initializer. Call it an ASH sequence-policy transplant, never full ASH reproduction or state of the art. Exact UID-sequence deduplication is shared across arms; known duplicate results update ASH as non-improvements without another decode, with generation and lookup costs charged. No geometry-word dedup or external historical layout reuse.

## Cohort, arms, costs

Exactly the already-exposed public cases cl_03_100_01, cl_04_100_01, cl_07_100_01: known negative K2, known K1 gain, known K2 gain respectively. No unseen/generalization claims. Each is100 parts, fixed direction, original stock sizes, zero trim/gap. No case replacement or new case after outcomes.

Arms: SINGLE uses the existing early single candidate order and H suffix; PAIR uses all ordered distinct UID pairs in lexicographic UID order and the same H suffix (includes sequences also expressible by K1; only already evaluated full sequences are skipped); ASH uses the above policy initialized from the best common initializer. SINGLE and PAIR each seed1000 once; ASH seeds1000,1001,1002. Deterministic controls are shared within a case, not counted as3 independent repetitions. No UID7/87 or observed witness is injected.

Two separate regimes,30 runs in total (3 cases ×5 arm/seed settings ×2 regimes):

- EVAL: maximum1000 full replay starts INCLUDING two common initial starts;60s total qualification bound with5s closing reserve. Compare1000-evaluation caps; SINGLE can exhaust early and carries its incumbent forward with unspent evaluations, explicitly not claiming1000 actual evaluations. Any time-censored EVAL run is not a completed quota comparison.
- TIME:5s total from child before data loading through result export/write, with1s closing reserve. No small evaluation cap (1000000 starts), raw proposal cap100000. SINGLE may exhaust early; PAIR finite exhaust is also eligible. Report actual wall time and counts. Parent watchdog total+5s including normal exit. Import/setup shared and charged after worker start; interpreter spawn time separately reported by parent.

Each arm begins a fresh solver/decoder, runs the exact common initializer, and has identical UID input/decoder configuration. Initial solver seed fixed1000 for all arms (deterministic initial state); ASH random stream alone uses row seed. Candidate calls replay complete sequences from empty layouts through the same decoder with its normal intra-run caches; no prefix snapshots or warm historical caches. Cache entry state after common initialization must match across arms/regimes. Every completed layout is captured during its budget; offline geometry and lineage auditing cannot generate/search/decode. No per-candidate physical-fit screen, no multi-fidelity calls.

Counters: initial/candidate starts and completions, interrupted generation and replay, full proposals, exact duplicates, random draws, generation/filter/replay/receipt times. Candidate receipt capture and append cost belongs to solver time; minimal final validation and export belong to closing reserve. Full layout receipts streamed to each private JSONL trace, no retained per-evaluation snapshot list or callback cycle. Preserve first run outputs/failures and hashes; no automatic retries, retuning, changing checkpoints or enlarging budget.

## Endpoints and interpretation

Primary: final board count at the declared cap/deadline. Secondary: first fewer-board candidate evaluation (candidate-only and total including2 initial), first-saving elapsed time, per-budget incumbent trajectory, utilization, actual counts/time. Separate case-level wins/ties/losses; ASH3 seeds are a stability range, not3 independent instances. No significance claim with3 cases. Differences in expressive range (K1/K2/global permutation) are explicit; PAIR/ASH is not a same-two-piece-range selector ablation. This comparison establishes a direct reference and limits existing claims, not a new mechanism by itself.

Decision: retain observed per-case evidence. If ASH matches/beats prefixes, do not claim unique prefix efficiency. A prefix advantage on these known cases is development evidence only, requiring an actual new mechanism and later independent validation before innovation claims. Do not silently resume PUCT, static residual scoring, deeper enumeration or factory tests.

## Implementation and verification plan

1. Test-first standalone ASH policy: hand-scripted random decisions/cyclic wrap, threshold boundary, unique identity preservation, seeded reproducibility, strict improvement adoption/reset, ties/cap, deadline cancellation, invalid states.
2. Minimal reference runner uses existing shared initialization/decoder APIs; test real toy geometry, quota, duplicate handling, equal initial entry state, time interruption, candidate-order inventory and trace accounting.
3. Run full repository suite, freeze source/input/protocol/test hashes and randomized30-run schedule (seed20261126), execute once with per-run watchdog and checkpoint.
4. Independently audit every saved complete layout, initial equality, proposals/policy trajectory via separate source transcription, best-result provenance, starts/time bounds and first-saving endpoints; verify prior data/master hashes. Final code review and relevant regression checks before publication.
5. Publish code, tests, protocol and aggregate report on the existing research branch; no factory raw files or paper PDF.
