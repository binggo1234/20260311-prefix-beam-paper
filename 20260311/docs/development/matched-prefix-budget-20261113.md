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
