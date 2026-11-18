# Saved prefix path audit — 2026-11-12

The739-piece95-panel result is a concrete short-prefix/completion positive example, and the other two orders also exercise front promotion without saving panels. The large order has a measured throughput constraint; the small order finishes its current search well before its deadline. **The missing evidence is the actual first-saving prefix and the identities/results of all evaluated prefixes.** Next priority is a logging-only diagnostic to recover that receipt, keeping the algorithm fixed.

This audit adds **zero packing replays, searches or candidate evaluations**. It reads the18 saved outputs from the six-run739 batch and12-run178/2416 batch. Production `src/tests/scripts` remain identical to `810e439`; inspected checkout `203dba1fd5bcaa16d6f48fed7efb67ac8d9cb01c`. Only this aggregate report is public; new audit scripts/tests and detailed certificates stay private. No new quality experiment or acceptance result.

## 1. What the saved evidence supports

Every output's source/input/runtime/dependency/schedule/result identity, locked request/geometry/budget/counters, initial reference/paired state and final sequence/raw-layout receipts was rechecked. All18 saved geometries are independently feasible. Initialization is the canonical short-edge order on each order, and paired/cross-seed initial identities match. Final event full scores match the delivered incumbents. The inspected close path returns the saved global sequence/snapshot without final replay.

Events contain elapsed time, winning mode/tier, action depth, panels, score and internal layout hashes. They do **not** contain actual prefix UIDs, prefix piece count, macro boundaries/kinds or a log of all evaluated candidate states. Only final complete sequences/placements are exported. Internal event hashes also include unexported free rectangles; this audit does not invert hashes or reconstruct free rectangles by replay. Exact raw final SHA256 receipts are separately reconstructed from the exported placements/source/sequence.

## 2. Final sequence factorization and first-saving event are different evidence

For each final sequence, find the smallest number of its front pieces that can be fixed freely while the rest remains the canonical short-edge order filtered to the remaining UIDs. This is an **exact final-sequence factorization**, not the minimum number of pieces necessary to save a panel, and not recovery of actual macro grouping. Each certificate reconstructs the exact residual suffix; one fewer front piece fails. Independent forward enumeration and a last-descent check agree.

| Pieces | Seeds1000/1001/1002 panels, off→on | Minimum free front pieces for final sequence | Last accepted event depth | Possible actual prefix length from that event |
| --- | --- | --- | --- | --- |
| 739 | 96→95 /96→95 /96→95 | 16 /16 /16 | 3 /3 /3 | 16–18 each |
| 178 | 29→29 /29→29 /29→29 | 3 /3 /3 | 2 /2 /2 | 3–12 each |
| 2416 | 372→372 /372→372 /372→372 | 1 /4 /4 | 1 /1 /1 | 1–6 /4–6 /4–6 |

The lower ends describe exact UID sequence reproduction. Upper ends follow the observed event action depth and the inspected≤6-piece macro cap. The intervals do not identify which factorization/grouping was used. Equal ordered nominal dimensions/allowed-rotation labels give the same minimum front counts. UID permutation LCS lengths were cross-checked with rank-LIS; the minimum arbitrary single-item relocation counts also equal16/3/(1,4,4), respectively. Neither metric is an iteration count or a universal savings bound.

For739, the final16-piece factorization draws from canonical short-edge positions8–686 and from **five different initial boards**. It induces686 differing sequence positions, and509 exported placements differ when panel identity is included. The residual723-piece suffix remains canonical after removing the front pieces. These are descriptions of a saved feasible95-panel order; the audit does not attribute necessity to all16 pieces or to an individual shape/board.

Crucially, the **first95-panel event occurred at action depth1**, at solver-origin3.555/3.488/3.394 s in seed order. Registered macro actions are truncated to≤6 pieces; `_expand` increases depth once per action. Thus the first saving already occurred with **at most6 front pieces** under this implementation. The later final events occurred at9.116/9.017/8.822 s and depth3. The first-saving complete sequence/placements were not retained separately, so the final16-piece sequence cannot be substituted for that earlier prefix or described as necessary to save a panel.

Source semantics: [macro cap](https://github.com/binggo1234/20260311-prefix-beam-paper/blob/810e43966dcb1b8292b2a9cf14184dce308f57ca/20260311/src/inrp/mcts.py#L1976), [action-depth increment](https://github.com/binggo1234/20260311-prefix-beam-paper/blob/810e43966dcb1b8292b2a9cf14184dce308f57ca/20260311/src/inrp/mcts.py#L2951), [event fields](https://github.com/binggo1234/20260311-prefix-beam-paper/blob/810e43966dcb1b8292b2a9cf14184dce308f57ca/20260311/src/inrp/mcts.py#L3254), [saved-incumbent close](https://github.com/binggo1234/20260311-prefix-beam-paper/blob/810e43966dcb1b8292b2a9cf14184dce308f57ca/20260311/src/inrp/mcts.py#L4046).

## 3. What can and cannot be concluded about coverage

The178 final factorization fronts three pieces from original short-edge positions2–118. The2416 results front one piece from position2057, or four pieces spanning positions1029–2162. Thus all three orders have retained examples of promoting pieces from later in the canonical order to the front. It would be incorrect to explain both negative orders as never trying this broad operation type. The pieces are different orders' own parts; no cross-order identity or equal-shape causal matching is claimed.

The staged completion winner, global lexicographic score and prefix rank prioritize panel count. Completed returned prefix evaluations update the incumbent, and the formal close preserves it. Along that inspected valid-completion path, a strictly lower-panel result would not be rejected because of a worse secondary score. The final29/372 counts support that the completed returned evaluations did not deliver a retained lower-panel solution. This is not an inventory of every constructed partial candidate or an assertion about unfinished/unpromoted modes.

However, aggregated65 or6–7 distinct-state counts do not identify the actual prefixes. `candidate_unique_parts_max` is a per-generated-list maximum, and its total sums such counts across calls; neither is the global union of parts visited. Without generated/selected/evaluated identity receipts, one cannot determine whether a specific promising shape combination was never generated, generated but not selected, evaluated without gain, left unpromoted, or interrupted. Nor can the audit infer that a lower-panel layout is unreachable.

Source anchors: [returned evaluations retain complete incumbents](https://github.com/binggo1234/20260311-prefix-beam-paper/blob/810e43966dcb1b8292b2a9cf14184dce308f57ca/20260311/src/inrp/mcts.py#L2449), [panel-first global score](https://github.com/binggo1234/20260311-prefix-beam-paper/blob/810e43966dcb1b8292b2a9cf14184dce308f57ca/20260311/src/inrp/mcts.py#L506), [per-list counters](https://github.com/binggo1234/20260311-prefix-beam-paper/blob/810e43966dcb1b8292b2a9cf14184dce308f57ca/20260311/src/inrp/mcts.py#L2030).

## 4. Large-order throughput is constrained; small-order stopping is different

| Pieces | Completed simulations | Distinct completed states | Search phase seconds | Completion/search phase share | Retained beam depth |
| --- | --- | --- | --- | --- | --- |
| 178 | 80/80/80 | 65/65/65 | 2.353/2.443/2.352 | 85.4–85.6% | 3/3/3 |
| 739 | 80/80/80 | 80/80/80 | 24.245/24.707/24.388 | 93.88–93.93% | 3/3/3 |
| 2416 | 7/6/7 | 7/6/7 | 17.749/17.718/17.710 | 98.78–99.06% | 0/0/0 |

These inclusive phase times overlap: do not sum phases or interpret ratios as a controlled timing effect. Different order sizes are not causal controls for one another.

The2416 on runs spend9.198–9.237 s on common initialization, entering prefix search with17.704–17.744 s left before the search deadline. Completion consumes17.519–17.552 s; candidate generation is approximately0.101–0.140 s. All three runs time out during one short-edge completion while retaining a complete incumbent. Each completes fewer than the first requested8 simulations. The existing first-round gate is16 requested simulations before retention checks, witness advance is disabled, only one beam round occurs and no successor root is retained. Under the inspected control flow, time expires before reaching that first retention checkpoint. This establishes limited evaluation opportunity, **not** that faster completion or a deeper root would save a panel.

The178 runs instead finish their current beam procedure after about2.35–2.44 s, from approximately26.9 s available at entry, with no timeout and retained roots reaching depth3. Their null result is not due to exhausting the30 s allowance. More wall time alone does not force this terminated control flow to explore more states. Depth3 limits retained roots; `_select_leaf` has no corresponding hard depth3 stop, so do not call every leaf≤18 pieces or80 a standalone global simulation cap.

Source anchors: [first-round gate](https://github.com/binggo1234/20260311-prefix-beam-paper/blob/810e43966dcb1b8292b2a9cf14184dce308f57ca/20260311/src/inrp/mcts.py#L3768), [leaf selection](https://github.com/binggo1234/20260311-prefix-beam-paper/blob/810e43966dcb1b8292b2a9cf14184dce308f57ca/20260311/src/inrp/mcts.py#L2992).

## 5. Verification and limits

**18/18 saved outputs** reaudited; exact final raw-layout/sequence receipts and both material metrics verified; all old evidence files/dependencies, current production sources and protected original/frozen workspaces unchanged. Guards prohibit solver construction, decoding/replay, search, packing and candidate-generation/evaluation entry points; **zero forbidden calls during the audit**.

Eleven new pure-audit tests were observed failing for the missing audit implementation, then passed. Together with seven existing exhaustive mathematical tests: **18 passed**. Independent forward matching/last-descent and LCS-DP/rank-LIS agree on actual artifacts; pure math tests cover duplicates, wrong multisets, event/depth confusion and throughput inference. Private scripts compile and public diff checks pass. No new independent algorithm/performance review is claimed.

The runtime suite is deliberately not rerun because it executes packing excluded by the user. Previous413-pass/three-existing-CutEval-skip evidence remains historical, not a fresh result. This audit changes no material-quality finding:739 remains a single-order positive, the other two remain ties, and the formal gate remains unchanged/not achieved. No new matrix, unseen-order, primary-mode causal, PUCT-only or optimality claim.

## 6. One minimal next verification

Keep the restricted-prefix algorithm fixed. First add **bounded diagnostic receipts only** for the actual prefix UIDs/piece count, macro grouping/source, winning completion mode/tier/cache origin, and the first-saving complete sequence/layout. Then diagnose the739 order at fixed seed1000 once under the same30 s allowance, counting logging/serialization inside that allowance. Acceptance is that the first95-panel event can be tied to a reconstructible actual prefix and independently auditable complete layout; a missed event is recorded rather than selectively rerun. This is a proposed future diagnostic, **not implemented or run here**, and is not a new material-quality comparison.

That receipt can ground a later matched coverage check on the other orders. Until it exists, do not introduce another heuristic or expand the quality matrix from an assumed missing-prefix explanation.
