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
