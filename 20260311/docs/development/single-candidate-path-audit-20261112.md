# Existing single-action coverage of the saving witness — 2026-11-12

The cold-reproduced95-panel witness is already **second in the existing early single-entry order**, despite being497th in canonical short-edge order and53rd in static-score order. It is covered by the current two-single entry cap. The empty-root macro pool has an upper bound of7 actions, below both the12-action mix limit and initial progressive width13. This establishes existing candidate coverage on the known739-piece order; it does not recover the historical evaluation chronology or establish the value of PUCT.

## 1. Version, scope and acceptance

Read-only checkout `abc84ff16127c8bcbd9906813dfe3c04c3f39ccb`; source remains identical to frozen runtime `e71695771a6ec871d4fdbc474e47fca4a8a5aeb7`. This batch adds only this public report. Private audit/tests/evidence remain local. No production code, candidate rule, default, completion, geometry, heuristic or dependency change.

Scope: inspect the existing generator/selection/evaluation code, derive empty-root coverage from the original input metadata, and bind that derivation to the saved first-saving and cold-replay receipts. Acceptance requires agreement between independent score arithmetic and the existing pure source expressions, valid receipt/geometry bindings, explicit separation of inference from observation, and zero new packing/search/candidate evaluations. Decoder and solver construction, candidate generators, selection, expansion and completion entry points are guarded against invocation. Only preparation, pure metadata arithmetic/sorting and saved-layout auditing run.

## 2. Derived entry coverage

The unchanged diagnostic configuration uses the large tier, macro actions enabled, early single-only ratio0, `MACRO_SINGLE_TOPK=2`, geometric priors enabled and complement/type-fair mechanisms disabled. Preprocessing still annotates shape metadata when dimension preprocessing is disabled; it does not quantize the input dimensions. Restored configuration values exactly match the previously saved cold-replay configuration.

| Existing order or gate | Derived value for the known witness |
| --- | ---: |
| Canonical short-edge position | 497 |
| Static-score position | 53 |
| Early single-entry position | **2** |
| Ties on its early key before UID tie-break | 1, including itself |
| Single entry cap | **2** |
| Scout head / appended tail samples | 12 /4 |
| Scout list / retained single entries | 16 /2 |
| Early empty-root macro pool upper bound | 7 |
| Macro mix limit / initial progressive width | 12 /13 |

Static score combines nominal area/board area, aspect ratio and metadata bonus. Early scouting first sorts by a metadata bonus reduced by duplicate penalties, then static score and UID. These are distinct from short-edge completion order. Existing `_single_macro_actions` takes only the first two entries of the scout list; appended tail samples do not reach that slice in this configuration. This is a root-entry observation about the rule, not evidence that tail samples are ineffective everywhere.

The early root permits at most4 memory patterns,1 anchor when patterns are present and2 singles; without patterns, at most2 anchors plus2 singles. Tail actions are disabled at this stage and complement is off. Hence the conservative maximum is7. Both the mix and initial widening limits fit the entire generated root pool, so neither numerical cap excludes this single. Actual registration/retention is also evidenced by its saved `single`/`single` macro ancestry. The audit does not rerun local placement or produce a new candidate pool.

Source: [static score](../../src/inrp/mcts.py#L1084), [metadata and duplicate bonuses](../../src/inrp/mcts.py#L1316), [stage limits](../../src/inrp/mcts.py#L1514), [single entry slice](../../src/inrp/mcts.py#L1926), [scouting](../../src/inrp/mcts.py#L2053).

## 3. Saved path and unknown chronology

The saved first accepted saving has one prefix piece and action depth1, kind/source `single`/`single`, completed with `short_edge`, `primary_only`, fresh mode evaluation and no full-group/staged-entry reuse. It occurs at incumbent-event index3 and solver-origin3.488436958 s. The index counts incumbent events, not candidate evaluations. Its full order is exactly that single followed by the unchanged canonical order of the remaining738 pieces.

The first-saving receipt and both existing cold outputs pass source-based geometry and identity audits without replay; full sequence/raw-layout bindings match. Earlier cold results remain96 versus95 panels, net utilization90.3116% versus91.2622%, a0.9506483-point gain. This batch produces no new quality result.

The selection code generates/registers macro actions, mixes them, sorts by prior/rank, may apply root Dirichlet noise, restricts to progressive width and expands an unexpanded action before using PUCT among expanded children. The root noise is enabled here; type-fair expansion is off. A root frontier can be regenerated when its stored width is below the requested width. Thus a static entry rank is not an observed post-noise rank or evaluation position.

Unavailable in saved evidence: chronological root candidate IDs/priors, the witness's rank after noise or regeneration, its simulation/full-evaluation index, the exact simulation-versus-ranking call site of the first saving, and the full generated/selected/evaluated prefix inventory. Aggregate49 PUCT selections cannot assign causal credit for this witness. The audit neither reconstructs missing history nor concludes that PUCT or layout feedback is unnecessary.

Source: [candidate preparation/noise](../../src/inrp/mcts.py#L2855), [leaf selection](../../src/inrp/mcts.py#L3038), [simulation evaluation](../../src/inrp/mcts.py#L4044), [shared evaluation/incumbent update](../../src/inrp/mcts.py#L2460).

## 4. Existing evaluation cost

| Saved diagnostic timer | Seconds | Share of prefix-search phase |
| --- | ---: | ---: |
| Prefix search | 23.867984 | 100% |
| Completion | 22.424470 | **93.9521%** |
| Candidate generation | 1.184630 | 4.9633% |

These are inclusive/nested phase timers and must not be summed as independent wall times. Existing completion-mode replay timers are short-edge12.054711 s, long-edge5.062442 s and strip-bias5.124479 s; they do not include all completion overhead. Counts are80 distinct completed states/80 simulations and80/20/20 mode evaluations.139 oracle calls include reuse and are not139 distinct full decodes. Likewise,900 generated macro-action events and258 single-action events are cumulative generation counters, not unique pieces or distinct evaluated prefixes. No per-witness candidate-generation or complete discovery-cost breakdown is saved.

Completion dominates this saved run's measured search phase. This supports testing a small existing-candidate reference before adding selection complexity; it does not establish a speedup obtainable by removing any mechanism.

## 5. Verification and remaining limits

**16 private checks pass**: independent-versus-source score agreement on varied/duplicate toy shapes, opposing UID tie-breaks, the two-entry slice, unsupported-profile rejection, six execution guards and saved-JSON restoration. The first audit attempt stopped during request validation because saved rotation arrays needed restoration to the interface's tuple type. The attempt, manifest and exact script versions are preserved locally. A regression case reproduced the failure; the private reader was fixed, tests passed and the read-only audit completed. No solver code or factory search/replay ran during either attempt.

Compilation and Git whitespace checks pass. Frozen source/dependencies,491 preflight historical JSON files and the original dirty/formal frozen workspaces retain their identities. No full runtime suite was rerun for this report-only batch; no independent private-script review or remote-CI pass is claimed. The evidence remains one known-order witness, with no new cross-order, unseen, optimality, mean-gain or formal-acceptance claim. Earlier cross-order ties and the unpassed formal gate remain unchanged.

## 6. Next bounded priority

The existing entry rule already covers this positive witness. The next useful control is **the two existing empty-root single entries, evaluated directly with canonical short-edge completion**, with no new candidate family, ranking or known-piece override. Predeclare identical strong initialization, a two-full-completion allowance and the same total pipeline limit for this reference and a truncated current-search arm. Keep the initial-floor fallback and record actual calls/time; do not fill unused time with extra evaluations. This would test early gain within two completions on the known order, not the eventual value of a longer search or a formal PUCT ablation. It is a proposal only; not implemented or executed here.

Keep the restricted-prefix topic and paused new heuristics. Do not hardcode position497 or the captured identity; do not expand the matrix or launch formal acceptance in this batch. Earlier [first-saving receipt](first-saving-receipt-20261112.md) and [cold replay](cold-prefix-replay-20261112.md) remain separate preserved evidence. Raw factory identities, dimensions, sequences and layouts remain private; existing draft only, no merge to master.
