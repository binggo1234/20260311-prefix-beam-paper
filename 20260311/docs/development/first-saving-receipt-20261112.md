# First-saving prefix receipt — 2026-11-12

The one prelocked diagnostic captured and independently audited the actual first95-panel solution. Its prefix is **one existing `single` action**: move the piece at position497 in the original short-edge order to the front, then retain canonical short-edge order for the remaining738 pieces. Common initialization uses96 panels. Both the captured first solution and the separately delivered final solution use95 panels and pass geometry audits. This is a recovered feasible witness on the known order, not a new quality comparison or formal acceptance.

## 1. Version and implementation

Frozen runtime `e71695771a6ec871d4fdbc474e47fca4a8a5aeb7`, based on `a983348`. New default-false `PREFIX_FIRST_SAVING_RECEIPT_ENABLE` and opt-in `rh_mcts_prefix_short_receipt`, inheriting `rh_mcts_prefix_short_completion_on`. Profile parity tests cover20/178/739/2416-piece adaptation: only flag and variant identity differ.

Before the first saving, nodes carry a diagnostic macro chain that survives root detachment. The first accepted prefix-completion incumbent strictly below the **completed common initial floor** saves one persistent snapshot/sequence and actual prefix/macro/mode/cache provenance. Greedy→short-edge initialization improvement is excluded. Capture-time exact sequence/layout SHA256s are stamped into the actual incumbent event and its receipt. Later improvements do not replace this capture.

Existing formal close exports raw unrounded envelope placements from that saved snapshot, without decoder replay. The receipt travels through existing handoff metadata; no handoff API/dependency change. Only one full witness is retained; no complete evaluation history is logged. Missing context is explicit, not inferred from depth/final sequence. Close export checks the existing absolute deadline and receives no extra time.

Existing profiles keep the flag off. Candidate generation, rankings/reward, common initialization, staged quotas, RNG/cache keys, adaptive policies, depth/width/commit, short_edge/long_edge/strip_bias modes and geometry are unchanged. Logging uses existing wall time, so identical timed trajectories or speed improvements are not assumed. Other experimental heuristics remain off.

## 2. One frozen diagnostic

Exactly one factory run: known739-piece order, seed1000, diagnostic profile, **30 s pipeline/3 s closing/38 s parent watchdog/8 s bounded cleanup**. Source/input/script/dependency/reference identities and the single schedule row were frozen before execution. No retry, selective rerun, outcome tuning or larger matrix. Toy tests are separate from this factory-run count.

The pipeline includes input handling, initialization/search, receipt capture/export, existing final validation, independent first/final geometry audits and minimal serialization. Process startup is covered by the parent watchdog. Initial sequence/layout/full score/candidate-panel ledger match the prelocked96-panel historical common seed; that reference is an identity check, not a historical efficacy/timing control.

Acceptance was recovery of the actual first95 event, exact prefix/macro/full-sequence association, independently feasible captured geometry and budgeted clean output. No receipt with no saving is `not_reproduced`; a saving event/result with missing/corrupt receipt is `invalid`. Both remain recorded, with no retry. Execution status is separate from diagnostic status.

## 3. Recovered first-saving witness

| Receipt item | Observed value |
| --- | --- |
| Common Greedy/short-edge floor | 98 /96 panels |
| First captured saving | 95 panels |
| Actual prefix pieces / action depth | **1 /1** |
| Actual macro kind / source | `single` / `single` |
| Original short-edge position, one-based | **497** |
| Original initial-board location | 87 of96 |
| Residual order | Remaining738 pieces in unchanged canonical short-edge order |
| Completion mode / tier | `short_edge` / `primary_only` |
| Winning completion origin | Fresh mode evaluation; no full-group/staged-entry reuse |
| First-saving event | Index3, solver-origin **3.488436958 s** |
| Post-serialization total elapsed | **24.877846916 s** |

The exact equation `first_sequence = [captured UID] + initial_short_edge_sequence_without_that_UID` is checked against the complete739-UID permutation. The raw captured placements match the exact digest stored at the actual first event and the receipt. Prefix count equals recorded macro flattening, and action count equals depth. The selected identity/dimensions/full sequence/layout remain in the private receipt; position497 describes this order's original sort and is not a general selection rule.

| Material metric | Common96-panel initial solution | Captured first95 / final95 | Difference |
| --- | ---: | ---: | ---: |
| Net utilization | 90.3116% | 91.2622% | +0.9506pp |
| Reserved-envelope occupancy | 92.2720% | 93.2433% | +0.9713pp |

Boards2440×1220 mm,5 mm trim,6 mm tool allowance and allowed0/90° rotations are unchanged. Envelope occupancy is not product yield or exact cutter-path loss. First and final solutions have different sequence and layout digests; later same-panel secondary improvements remain separate from the first saving. The earlier16-piece final-sequence certificate therefore does not identify this first-saving prefix or establish a necessary savings length.

The complete run performs80 simulations/80 distinct completion states,49 PUCT selections and80/20/20 short_edge/long_edge/strip_bias mode evaluations. Receipt capture and placement export timers total about**1.353 ms**; this excludes separately unmeasured node-chain/context/serialization overhead, which still counts inside total elapsed. There is no controlled overhead/speed comparison.

## 4. Independent verification and first/final separation

The worker audits both captured and final layouts before minimal write; both contain every requested part exactly once and satisfy source dimensions, rotations, trim/boundaries and tool-clearance non-overlap. Post-run audit repeats both geometry checks without packing replay. Initial/reference, payload/request/budget, final raw receipt and first event→receipt→raw snapshot bindings pass. Saved initial/final/diagnostic fields are checked against solver metadata and recomputed diagnosis/audits, including hashes.

**Execution passed; diagnostic passed; clean process exit; zero audit findings.** Exactly one saved factory outcome. The first captured snapshot was not substituted by the later final layout. Frozen runtime/dependencies remain unchanged through execution;86 historical evidence files and original dirty/frozen workspaces preserve their preflight identities.

## 5. Tests, review and limits

23 new runtime tests cover first/common-floor exclusion, later strict-improvement immutability, detached/ranking-time macro context, fresh/staged/full and cached-winner-with-fresh-fallback origins, default-off RNG/cache/profile parity, JSON handoff and deadline-preserving export without replay. Full runtime suite: **370 passed/3 existing optional CutEval skips**.29 private diagnostic checks plus5 reused lifecycle checks pass; combined validation **404 passed/3 skipped**. Compileall/py_compile/diff checks pass.

The originally selected reviewer could not run because of model quota. One actual fresh read-only review by the available fallback model independently passed23 runtime and16 private+5 lifecycle tests, found2 Important audit gaps and no Critical/Minor issue. Before the factory outcome, both were fixed in one RED→GREEN pass: missing capture is distinguished from true non-reproduction, and capture/event/raw/saved-record metadata bindings are enforced. A runtime assertion and13 new private cases failed before these fixes, then passed; full suite is green. No second-review or independent post-fix performance-review claim.

This run confirms a valid restricted-prefix witness and its provenance on one known order. It does not establish order-level mean gain, unseen generalization, PUCT necessity, a primary-mode causal effect, optimality or formal acceptance. The formal gate is unchanged/not achieved; previous cross-order ties remain unchanged. A cold decoder replay was **not** performed in this diagnostic, and the full generated/selected/evaluated prefix inventory remains unlogged.

## 6. Next bounded priority

Keep this one-piece prefix as a fixed positive witness. Next recommended verification is a **cold-decoder original-order versus captured-single-front-order check**, with fresh decoder instances and unchanged geometry, to establish whether the95-panel sequence reproduces independently of warmed search caches. This is not implemented/run here. Do not turn original position497 into a heuristic or enlarge the quality matrix before that check; restricted-prefix topic and paused new heuristics remain.

Execution rulings: use the established authorized branch/draft (branch visibility if scope misunderstood); retain only the first snapshot and node provenance (observational cost can alter timing, no speed/quality causal comparison); review before freeze then verify outcome separately (no second independent outcome review); substitute an available reviewer after quota prevented the original reviewer from working (no review by that unavailable model); accept excluded pre-data reproduction/timing as executor-verified only, excluded quality/formal/unseen/causal claims as untested, and pending publication/CI as separately checked without treating empty checks as a pass. No deferred minors. Only code/tests/spec/plan/aggregate report are public; raw factory receipts stay local.
