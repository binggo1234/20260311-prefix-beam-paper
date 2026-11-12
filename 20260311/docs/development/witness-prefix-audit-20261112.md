# Saved 95-panel witness vs prefix/completion rules — 2026-11-12

The saved95-panel witness is feasible for the same739-piece order and geometry as the96-panel common initial solution. Its ordering is **not a short-front-prefix change followed by any currently configured completion order**. Reproducing this specific UID sequence, or even its shape-label sequence, would require freely specifying the first724–725 pieces. This establishes a mismatch with this witness; it does **not** establish that every95-panel layout is unreachable.

This is an exploratory, read-only saved-evidence audit. No runtime/configuration/heuristic changes, packing replays, candidate evaluations or expanded experiment matrix. Only this aggregate document is added publicly; factory sequences and layouts remain private.

## 1. Evidence identities and geometry

- Witness: corrected yardstick runtime `120d29436f27446fc5e92441840284d0159104d9`, saved run020, whole-sequence SA seed1002, **95 panels / 91.2622% net utilization**.
- Common initial reference and three paired prefix-on runs: common-init runtime `4acad1344c485f02ef9e09ce3a8a57dfa9d59f27`, **96 panels / 90.3116%**. Difference is one panel / approximately **0.9506 percentage points**.
- Current inspected checkout: `884c6505a0371aa409644793f9684447b19cf7f2`; its runtime sources match the common-init experiment. Historical source hashes were checked against their actual Git blobs, not against whichever files happen to be live.

All seven relevant saved outputs (one witness, three off and three on) were independently geometry/identity reaudited. Input hashes, requested parts, permitted rotations, board2440×1220 mm, trim5 mm/edge and tool clearance6 mm agree exactly. Result/schedule/runtime/dependency identities match. This confirms a feasible95-panel order, not its optimality or same-runtime timing superiority. The witness's historical89 completed SA proposals and7.388864 s first95 event are provenance only; its intermediate accepted proposal sequences were not saved, so an exact mutation history cannot be reconstructed.

## 2. Difference from the saved short-edge initial sequence

| Metric | Exact UID order | Ordered dimensions + permitted rotations | Allowed envelope-pose class relaxation |
| --- | ---: | ---: | ---: |
| Pieces | 739 | 739 | 739 |
| Equal positions | 55 | 241 | 241 |
| Different positions | 684 | 498 | 498 |
| First / last different positions, one-based | 11 / 725 | 11 / 725 | 11 / 725 |
| Longest common subsequence length | 654 | 654 | 654 |
| Minimum arbitrary single-piece relocations | 85 | 85 | 85 |

There are329 ordered shape classes. Merging identical shapes reduces apparent position differences, but does not turn this witness into a few-item ordering change. The pose-class column further relaxes identity/orientation labels using equal sets of permitted inflated rectangles; it does not assert identical deterministic placement behavior after every relabeling.

The85 figure is `739 − LCS`, the minimum number of single items deleted/reinserted **anywhere** to reproduce that sequence. It is not85 front-prefix actions,85 SA iterations,85 causally critical pieces, or a minimum number of block moves. One block move may change many positions. Nor does it prove that all of these ordering differences are necessary to obtain95 panels: the saved sequence may contain irrelevant changes.

## 3. Exact-front-prefix certificate

Current active deterministic completion orders are `baseline` (nominal area/aspect/long edge), `long_edge`, and `strip_bias` (aspect/long edge/area), with UID tie breaks. The common short-edge seed **does not change these completion orders**; short-edge is not an active completion mode. See [the completion keys/order](../../src/inrp/mcts.py#L2168).

For a target sequence T and a mode's complete sorted order B, ask for the smallest k such that `T[k:]` is a subsequence of B. Then there exists an identity assignment/removal for a *freely ordered* front prefix `T[:k]` followed by that mode's remaining sorted suffix. With unique UIDs, this is exactly the minimum prefix needed for the concatenation to equal T. With repeated shape tokens, the subsequence test permits reassignment within equal classes, making it a relaxed necessary condition.

| Current completion | Minimum freely specified front pieces, UID | Ordered-shape relaxation | Pose-class relaxation | Remaining sorted suffix | Minimum macro actions at ≤6 pieces/action |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 724 | 724 | 724 | 15 | ≥121 |
| long_edge | 725 | 725 | 725 | 14 | ≥121 |
| strip_bias | 725 | 725 | 725 | 14 | ≥121 |

Each result has two checks: the reported minimum suffix is a subsequence, and making the prefix one shorter fails. The backwards-greedy calculation was independently verified by scanning every possible prefix length with a forward subsequence assignment, for all three identity/shape descriptions. This permits arbitrary front-prefix ordering and ignores macro candidate legality; actual candidate restrictions can only make exact reproduction harder. It is not a claim that121 legal macro actions exist.

This says the **saved ordering** cannot be reproduced by a tiny front-prefix edit plus an unchanged current suffix. Different sequences may produce the same95-panel layout, and other95-panel layouts may exist. Neither is ruled out by these certificates.

## 4. What the current prefix limits really mean

Effective medium-order settings: beam retained-root depth3, width2, branch top3, commit-chain length1, macro size≤6, staged deterministic completion. With one action per retained-root advance, a depth3 retained root contains at most18 pieces. That is much smaller than724; however, **18 is not a universal cap on evaluated simulation prefixes**.

[The beam loop stops advancing a retained root at depth3](../../src/inrp/mcts.py#L3768), but [`_select_leaf` traverses until a terminal/unexpanded node](../../src/inrp/mcts.py#L2989) and has no corresponding depth3 stop. [A tree expansion adds one action/depth](../../src/inrp/mcts.py#L2945). Therefore configured beam depth and deepest evaluated leaf must not be conflated. No saved maximum-leaf-depth trace exists for these runs, so this audit does not invent one or claim an observed violation. If a manuscript states that every evaluated prefix has depth≤3, the current code does not provide that guarantee. This semantic point is documented without changing the algorithm.

A separate, conservative run-specific bound is available from the saved counts. Common-init starts with an empty root and no preexisting children; one selection creates at most one new tree node, each action contains at most6 pieces, and the global-deadline interruption permits at most one additional uncompleted expansion. Thus a root-to-leaf prefix has at most `(completed simulations + 1) × 6` pieces, even allowing every expansion to lie on one path:

| Seed | Completed simulations | Conservative maximum prefix pieces from expansion count |
| --- | ---: | ---: |
| 1000 | 79 | ≤480 |
| 1001 | 78 | ≤474 |
| 1002 | 80 | ≤486 |

These are derived upper bounds under the inspected frozen control flow, not observed prefix lengths. All are below724. Consequently, those saved30 s runs could not reproduce this particular complete ordering through their prefix-plus-current-suffix evaluations. This is a run-specific representation/work bound, **not a general95-panel infeasibility result or an isolated primary-bottleneck diagnosis**.

## 5. Verification and unchanged scope

- Seven mathematical audit tests were written failing first, then passed; minimum-prefix results match exhaustive toy subsequence assignments including duplicates. LCS lengths also match exhaustive common-subsequence enumeration on toy inputs.
- Actual739-piece minima were independently checked by a different forward-scanning method; all actual LCS values were independently checked using matching-position LIS, agreeing at654.
- Seven historical outputs were reaudited; historical source/dependency/input/result hashes match. Current runtime source hashes before/after are identical.
- Audit guards prohibit decoder decode/replay, solver execution, macro candidate construction and packing calls. Recorded counts: **0 packing replays,0 search runs,0 new candidate evaluations**. Toy tests manipulate token sequences only.
- No runtime files or factory results changed. Runtime tests retain their preceding338-pass/three-existing-skip evidence; they were not rerun here because this audit makes no runtime change or new packing execution. No fresh independent code-review or remote-CI claim.

Private reproducibility artifacts: `witness_prefix_audit_20261112/audit.json`, `manifest.json`, `independent_combinatorial_check.json`, plus the audit script, seven-test file and scope record in the private QA directory. This is post-hoc witness analysis, not a newly preregistered efficacy comparison. Only aggregate documentation is public; old failure/negative records and original dirty/frozen checkouts remain untouched.

## 6. Decision

The95-panel artifact proves same-order material headroom, but it is not a demonstrated legal short-prefix positive example for the current completion family. We cannot use failure to rediscover its exact global sequence as a verdict against restricted-prefix research, nor use it to justify an untested increase in depth, time or candidate heuristics.

Keep the theme and pause algorithm/matrix expansion. Before proposing another mechanism, distinguish a witness inside the intended prefix/completion family from a witness produced by unrestricted whole-sequence moves, and state retained-root vs evaluated-leaf depth accurately. This audit found no alternative95-panel short-prefix witness and made no attempt to search for one; that existence question remains open.
