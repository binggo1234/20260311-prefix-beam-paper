# Exact board-local blueprint reuse

## Scope and acceptance protocol

The next engineering step reuses UID-independent local geometry for equal
parts on the same unchanged Board object. It does not change restricted
prefixes, candidate order, completion, scoring, rotation or spacing rules.
The incumbent development evidence remains 28/95/371 panels for the separate
178/739/2416-piece batches; the formal evidence gate remains unpassed.

`OmnipotentDecoder._micro_decode` currently computes local geometry on every
UID-cache miss. A diagnostic of the fixed 2416-piece baseline order counted
669,769 calls, including 359,417 repeated exact Board/shape queries (25,295
repeated feasible queries). This diagnostic is not a timing comparison.

Use a lazy Board-local blueprint dictionary, cleared with existing geometry
cache invalidation and omitted from clones and dataclass comparison. Its
exact query-time key includes envelope and nominal dimensions, permitted
rotations, global rotation and placement mode. Retain the UID-specific micro
cache, actual-UID placement, post-placement area and identity hash. Store
after local placement, which may initialize free rectangles and invalidate
the board's caches. Never reuse MicroDecodeResult between identities.

Acceptance first requires equality to the frozen pre-change micro method:
every selected placement, complete order/layout/score, legacy cache entries
and counters, with mutation, rotation, nominal-area and duplicate-UID tests.
No production dependencies or default solver changes.

Before private factory verification, freeze the source, inputs, helper hashes,
schedule and all existing JSON hashes. Run one cold old/new pair for each
178/739 witness, three pairs for the fixed 2416-piece 372-panel baseline,
and one pair for the already known 2416-piece 371-panel witness. Each pair
has a 60 s diagnostic budget, 10 s closing reserve and 68 s watchdog.
Shuffle pair/arm order with seed 20261120. These are supplied-sequence
replays, not search. Retain the optimization only if fixed-work exactness
passes and the three baseline pairs show a lower median decode time.

Only after that gate run one 2416-piece scan per arm, same seed 1000 and
30 s total / 3 s closing / 38 s watchdog / 8 s cleanup, using the existing
V3 private harness and independent saved-output audit. Do not rerun a
disappointing result or tune candidate order. Count complete receipts only;
compare the shared evaluated prefix, initializer and legacy caches/RNG.
Report time, coverage and net material utilization separately. Any newly
saving order can have exactly one supplied-order confirmation per decoder;
those confirmations are neither timing replicates nor extra search trials.
Keep failures and all raw inputs/layouts private. Publish aggregate results
and tested source to the existing draft branch; never merge.

The cache trades extra per-live-board memory for less geometry work. A
single-machine development comparison establishes neither generalization
nor an algorithmic novelty claim.

## Verified development result

Runtime anchor: `02ef39a4296716a8e06f80520a1648b23d128b4b`;
frozen UID-only micro oracle: `90499abc1277c3103cbccf5bd8044c2c7a27043c`.
Both arms use the already deferred successor construction. This comparison
measures the additional blueprint memo, rather than repeating the earlier
eager/deferred comparison.

All twelve supplied-order cold replays exactly reproduced their reference
sequence, layout and score. Each used a fresh decoder. The three paired
2416-piece 372-panel baseline times were:

| Pair | UID-only micro (s) | Blueprint memo (s) |
| --- | ---: | ---: |
| 1 | 4.020494 | 3.335725 |
| 2 | 3.997492 | 3.291348 |
| 3 | 4.063493 | 3.360179 |

The median paired speed ratio is 1.209308, equivalent to a 17.3081%
reduction in decode time. The single witness pairs (not timing replicates)
also matched exactly: 178/28 at 0.039660→0.031014 s, 739/95 at
0.329422→0.283173 s, and the known 2416/371 order at
3.950707→3.245296 s.

The one predeclared fixed-time large-order pair passed:

| Metric | UID-only micro | Blueprint memo |
| --- | ---: | ---: |
| Common initializer time (s) | 8.209706 | 6.803367 |
| Candidate completions started | 13 | 16 |
| Candidate completions finished | 12 | 15 |
| Final panels | 371 | 371 |
| Net material utilization | 89.877087% | 89.877087% |
| Full pipeline time (s) | 27.515126 | 27.572552 |

Both stopped at the search deadline. The initializer solution, legacy
cache/RNG entry state and first twelve completed candidate receipts were
exactly equal. The previously observed 13 completions in the deferred batch
are a separate run; the concurrent control here finished twelve. The
within-pair increase is twelve to fifteen, not thirteen to fifteen.

Independent saved-output checks ran with solver construction, decoding,
candidate generation and search forbidden. They reconstructed receipts,
checked source geometry and complete identities, and passed 43 geometry
audits (12 cold plus 14/17 timed). All 760 pre-existing JSON files were
unchanged; the full runtime source map, reference files and helper hashes
were rechecked. No reruns, candidate retuning or new saving witness occurred.

## Validation and disposition

- Correctness: 428 tests passed; the same three optional pinned CutEval
  integration tests remain skipped because their source checkout is dirty.
- Private experiment protocol: nine tests passed, including both-arm
  synthetic saved-output audit and corruption rejection.
- Fresh read-only review found no introduced runtime defect. Two Important
  preparation gaps were fixed before factory execution: freeze all local
  Python helper dependencies, and complete the dedicated saved-output
  acceptance/comparison. Targeted regression checks were observed RED→GREEN.
  No Critical issues, deferred minors or declined-to-judge items remained.
- Retain the small exact performance optimization. It improves evaluation
  capacity, but **adds zero panels saved and zero utilization percentage
  points** against this concurrent control.
- The known development witnesses remain 28/95/371 panels across the three
  separate batch sizes. Default search policies, the restricted-prefix
  theme and the pause on additional heuristics remain unchanged. The formal
  evidence gate remains unpassed.

The next research gate is an equal-complete-evaluation comparison after a
common strong initializer, to isolate the restricted-prefix contribution.
Freeze this implementation for that comparison; do not count cache speed
as algorithmic novelty. Additional cache memory was not benchmarked here;
the dictionary is lazy and lives only with its unchanged Board object.
Results describe these development orders and this machine, not unseen
orders or a repeated fixed-time performance distribution.
