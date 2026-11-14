# Matched complete-evaluation prefix comparison

## Frozen development protocol

The literal prefix-off arm stops after common initialization and has zero
post-initial evaluations. Repeating that solution to pad a quota would not
be an informative equal-evaluation control. The existing simple single-item
scan is itself a restricted-prefix policy. This experiment therefore tests
the **increment of current search over simple restricted-prefix scanning**,
not a prefix on/off causal effect and not PUCT alone.

Freeze production source at `f01b2fc4bc17600614fc634ad610997747ee29a3`.
Use exactly one existing 739-piece development order, seed1000, two arms:

- Simple scan: the unchanged full early-entry single-item order with fixed
  canonical short-edge suffix, capped at twelve full evaluation starts.
- Current search: unchanged existing candidate generation, tree selection,
  prefix constraints, staged promotion and completion modes, instrumented
  to stop after twelve full evaluation starts. Cache reuse spends no credit.

Twelve is an executor choice made before this comparison, using the prior
large-order concurrent control's completed coverage as a modest fixed quota.
It is not a tuned optimum. No other orders/seeds/quotas, favorable reruns,
new heuristic, larger time cap or deeper-prefix policy are authorized here.
Both arms have exactly the same complete Greedy then short-edge initializer,
decoder, geometry, RNG and legacy cache entry state. Count those two initial
replays separately. Both have a30 s full pipeline /3 s closing reserve /
38 s watchdog /8 s cleanup. Stop early on quota exhaustion; do not pad time.
Input loading, candidate generation, cached/partial placement, promotion,
evidence construction, geometry validation/audit and writing count in time.

Instrumentation captures the active completion mode before replay, the
actual prefix and suffix, complete snapshot and scalar score. Check credit
before mode counters increment; crossing the cap mid-promotion must not
start a thirteenth replay or corrupt counters. Do not update incumbents
early, as that could change promotion/candidate decisions. At closing only,
retain the best complete evaluated layout and common initial floor, including
any completed mode in an interrupted final group. Record if this close-only
retention changes the original incumbent. No reconstruction replay.

Before factory execution, synthetic tests must show exact prefix/mode/order/
layout/score equality between the instrumented current search's first twelve
replays and the original unbounded search, plus cap boundaries through a
promotion, incomplete initialization, timeout and corruption rejection.
A fresh read-only review checks the entire private protocol. Freeze source,
helper/reference/input hashes and all historical JSON before execution.
Shuffle the two arms with seed20261121 and retain every failure.

Execution eligibility requires both complete initial replays and exactly
twelve successful post-initial evaluations in each arm; matched initial
solution and entry state; valid full identities, rotations, boundaries and
reserved-tool envelopes for every captured and final layout; correct saved
receipt/score/provenance bindings; clean output inside the original deadline.
If either arm fails or has fewer completions, the equal-count comparison is
inconclusive. Never promote failed/incomplete evidence into a material win.

Independently reload and audit every saved receipt with solver construction,
decoding and search forbidden. Report per-arm panels, nominal product area
over full purchased-board area, first evaluation improving the common initial
panel count, actual mode counts, prefix piece counts, simulations/PUCT and
pipeline time. Repeated/cached/duplicate-shape evaluations are reported rather
than reinterpreted as independent orders. More evaluations or a secondary
score improvement at equal panels are not higher material utilization.

Decision: fewer panels supports only a known-order fixed-quota whole-policy
advantage. A tie/loss supports no extra material gain at this quota; earlier
attainment of the same saving is descriptive evaluation efficiency. Neither
can establish general superiority, PUCT necessity, optimality, novelty or
the unchanged formal quality gate. In a tie/loss, keep the simple scan as the
strong baseline and pause investment in this complex route; further method
work needs a concrete difference from nearby literature and a testable gain.

Keep all source factory identities, dimensions, layouts and detailed private
helpers local. Publish only this protocol, aggregate results and links on the
existing draft branch; no merge and no changes to protected user checkouts.

## Verified result and decision

Executed once under the frozen protocol at documentation commit
`49c166136f2ac0419ae942a41d23bfbf9eb65302`; production source/tests/scripts
remain identical to `f01b2fc`. Current search ran first, simple scan second.
Both workers exited cleanly, exhausted their evaluation quota rather than
their time allowance, and produced valid saved outputs. No retry or tuning.

| Metric | Current search | Simple single-prefix scan |
| --- | ---: | ---: |
| Complete common initializer | 96 panels | 96 panels |
| Post-initial starts / completions | 12 / 12 | 12 / 12 |
| Final panels | **96** | **95** |
| Net material utilization | **90.311584%** | **91.262233%** |
| First evaluation saving a panel | None within12 | **2** |
| Short-edge / long-edge / strip-bias replays | 8 / 2 / 2 | 12 / 0 / 0 |
| Completed simulations / PUCT selections | 8 / 0 | 0 / 0 |
| Common initialization (s) | 0.611425 | 0.612162 |
| Full pipeline (s) | 2.884517 | 2.749048 |
| Close-only retention changed incumbent | No | No |

The complete initial receipt and entry RNG/legacy-cache state match exactly.
Against simple scan, current search uses one additional panel and loses
0.9506483 net-utilization percentage points. This is the same known medium
order used by earlier diagnostics, not an independent replication order.
The saving95-panel layout is already known; this experiment adds controlled
comparison evidence, not a new saving witness.

The current arm's twelve evaluated panel counts are
`96,96,96,96,99,104,96,96,96,96,98,104`. Prefix piece counts are
`4,6,5,5,5,5,5,4,4,1,1,1`; these are not macro depths. Eight distinct
prefixes are evaluated, with four credits spent on secondary completion
modes. Its only single-item primary evaluation is number10 and corresponds
to scan entry rank1. It does not evaluate the simple scan's second single
entry, which already saves a panel. This describes the observed coverage
gap; it does not prove that all deeper prefixes or secondary modes are
unhelpful. Neither arm performs a PUCT selection, so there is no PUCT-value
conclusion or longer-search conclusion.

Independent saved-output checks, with solver construction/decoding/search
forbidden, passed28 source-geometry audits: initial, twelve completed and
final receipts in each arm. Every full UID permutation, source dimension,
allowed rotation, trim/tool envelope, suffix mode, layout digest, recomputed
score and final-best binding passed. All778 inventoried historical JSON
files, frozen production source, local helper modules and reference hashes
remain unchanged. Full production suite: **428 passed,3 existing optional
pinned CutEval integration skips**. Private protocol: **15 passed**, including
exact first-twelve parity to original search and whole-audit failure cases.

One fresh read-only review found no Critical issue. Its two Important gaps
were fixed in one RED→GREEN pass before factory data: independently audit
failed checkpoints/partial receipts with sufficient bindings; recover and
audit scheduled watchdog checkpoints while retaining ineligible status.
Unrelated extra files reject. Available counts survive expired failures
without writing after the deadline. These failure cases are established by
synthetic tests; both actual factory runs succeeded.

Deferred Minor: the generic auditor does not yet guard some redundant
provenance fields (`case_id`, geometry/tool metadata), final ablation
fingerprints and equality of original/normalized scan evidence. The executor
explicitly checked all of those fields in both actual saved results and they
agree. Generic regression coverage for those redundant fields remains future
hardening. No second independent review or remote-CI result is claimed.

**Disposition:** keep the simple scan as the strong baseline and pause further
investment in this current complex early-search route. At twelve complete
evaluations it has no additional material value on this order and misses a
saving single entry already reached at evaluation2. Do not respond by adding
another selection heuristic, increasing prefix depth or silently increasing
the evaluation quota. The restricted-prefix theme remains: all three known
development sizes have saving single-prefix witnesses, but method novelty,
independent-order benefit and the existing formal gate remain unproven.

The next research deliverable is a focused difference audit against recursive
Pilot and sequence-neighborhood methods, centered on full-order short-prefix
coverage and complete-evaluation efficiency. That audit should identify an
actual methodological gap before another search mechanism is implemented.
