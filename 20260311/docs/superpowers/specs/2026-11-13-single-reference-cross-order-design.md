# Fixed single reference on two other known orders

## Authorized question

The user approved exactly two new factory runs of the frozen `single_reference`: existing178-piece and2416-piece orders, seed1000 once per order, identical existing strong initialization, two post-initial full-completion starts and30 s total pipeline. Test whether the reference saves panels outside the known739-piece example. The research topic remains restricted prefixes and new heuristics stay paused. This is a separately bounded protocol; it does not change the [fixed reference](../../development/fixed-single-reference-20261113.md) or authorize a larger matrix, current-search rerun, PUCT/long-search test or formal acceptance run.

## Implementation choice and dependencies

Reuse unchanged `early_prefix_control.solve_two_completion_handoff` at source anchor57d4672 and the already tested private two-completion worker, receipt auditor and bounded process collector. A new private wrapper routes that worker to a new output directory inside each spawn child; a new auditor validates this protocol and per-order initial identity. No runtime/default/profile/geometry/candidate/scoring/dependency change. Duplicating the adapter or editing old frozen scripts would add unnecessary divergence; those approaches are rejected. Only protocol and aggregate report are public; raw factory information and detailed harness remain private.

## Frozen input and schedule

- Exactly two distinct already-used orders:178 and2416 parts. Fixed seed1000, arm `single_reference`, profile `rh_mcts_prefix_short_completion_on` for both. Randomize only order with schedule seed20261117; do not add seeds/arms.
- Reuse the source manifest and previously frozen references `common_init_prefix_ablation_20261111/run_002.json` (178,29 panels) and `run_007.json` (2416,372 panels), with their existing exact checksums. These historical references used seed1002; they bind deterministic initial identity and parts order only, not historical RNG/cache state, current efficacy or timing. The prior cross-seed audits established deterministic initial equality; recheck actual fingerprints here.
- Freeze source/dependency/input/reference/protected-workspace identities and the two rows before outcomes. No factory timing probe, retry, retuning or replacement run.

## Budgets and evidence

Each run has30 s total pipeline,3 s closing reserve,38 s absolute parent watchdog and8 s bounded cleanup. Initialization's Greedy/short-edge two complete replays are counted separately. At most two later full replay starts, including errors/timeouts; no replacement credit or idle filling. Local prefix/candidate/cache work, evidence capture/export, validation, checkpoint/audits, serialization and result hashing count inside pipeline; startup/runtime imports are covered by parent watchdog. Use the existing complete initial floor and best-incumbent close without new replay.

Retain the matched best initial, each completed evaluation and final raw receipts; bind source IDs/dimensions/order, nominal placements, sequence/layout digests, complete scores, quota counts and exact row/protocol/budget metadata. Initial identity includes panels/full score/candidate ledger/sequence/layout digests and ordered part-list hash. Geometry stays2440×1220 mm,trim5,tool6,0/90° rotations,nominal-with-tool-clearance semantics. New entry definitions and witness identity/position overrides are prohibited.

Reuse budgeted failure checkpoints and explicit deadline-limited raw unavailability. Reconcile saved failure/orphan files and audit recoverable receipts without counting failed execution as a valid outcome. An absent/duplicate/wrong row, geometry/identity violation or incomplete budgeted execution withholds the overall two-order signal. Preserve any valid per-order descriptive result and all failure records.

Inventory run and failure artifacts by actual claims, including expected-number unclaimed failures beside passed rows; filename membership alone is not acceptance. Recoverable orphan failure metadata is explicitly post-failure evidence, never a successful outcome. With an intact verified manifest, missing/unparseable parent progress becomes an explicit inconclusive error and triggers artifact inspection. Replace parent progress atomically to reduce truncation risk; no guarantee of complete recovery after arbitrary OS/storage failure is made.

## Decision before outcomes

Execution acceptance requires both scheduled runs clean, complete initialization matching their frozen references, exactly2 successful post-initial starts/completions, independent source-based geometry/receipt checks and output within budget. Compare each reference result with its own actual matched common initial solution. Panel count/net material utilization are primary; reserved-envelope occupancy is auxiliary, not product yield or exact cutter-path loss. Secondary score/layout changes at the same panel count are not material gain.

- Both eligible and at least one strictly lower panel count: positive development signal on another already-used order, not unseen generalization or formal acceptance.
- Both eligible and both tied: two-order material replication fails; preserve the739 positive example and continue pausing algorithm expansion.
- Any missing/failed/incomplete/invalid outcome: inconclusive overall; do not call execution failure an algorithm negative and do not selectively rerun.

No two-order pooled panel-weighted mean, best-seed selection, statistical significance or independent-order inflation. Historical current-search results are context only, not a new concurrent comparator. PUCT, longer-search value, optimality and the formal quality gate remain untested/unchanged.

## Validation and publication

Use RED→GREEN tests for schedule rejection, identity/protocol binding, missing/duplicate/failed/orphan handling and gain/tie/inconclusive decisions; reuse existing adapter/receipt/lifecycle checks and run full project tests. One fresh read-only implementation/harness review before data; fix Critical/Important findings in one verified pass. Executor audits later data/report/publication without claiming a second outcome review. Publish protocol/aggregate report to the authorized existing branch/draft, verify exact head/body, retain master/protected workspaces unchanged and do not treat empty remote checks as CI success.
