# Two-completion early-prefix control

## Intent and delegated choices

Implement the user's approved comparison: existing two empty-root singles plus short-edge completion versus current search, identical strong initialization, two complete post-initialization evaluations and the same total time limit. Restricted-prefix topic and paused new heuristics remain. Routine implementation and publication choices are delegated by the existing conversation; no additional design/permission round is needed.

## Scope and approaches

Use an isolated experimental adapter `src/inrp/early_prefix_control.py` around the existing short-completion-on profile. A small subclass reuses common initialization, candidate entry, staged completion, selection and incumbent close. Existing source/default paths remain unchanged. This is the executor's implementation choice: modifying global decoder or MCTS defaults would increase unrelated regression risk; duplicating a search implementation would weaken the control. The reference calls the existing `_single_macro_actions(full_mask, 2)` once at the empty root, in returned order, with no identity/position override.

Public interface: `solve_two_completion_handoff(request, budget, arm)`, where arm is `single_reference` or `current_search`, request uses `rh_mcts_prefix_short_completion_on`, and budget is caller-created before input handling. Validate arm/profile before decoding. Require the unchanged deterministic staged profile, primary `short_edge`, promotion every4 and common initial/search gates enabled. Reject unsupported configuration instead of silently changing it.

## Evaluation and time accounting

- Each independent arm runs the same existing Greedy and short-edge common initializer and retains its best complete solution. The two initial full decodes are separately counted and excluded from the post-initialization allowance.
- At most **two started full post-initialization completion replays** per arm; a timeout/error consumes a start. Exactly two successfully completed replays plus complete common initialization are needed for an accepted factory pair. Do not refill failures or unused time.
- Partial-prefix local placement, candidate generation, cache hits and ranking without a new full replay do not consume a replay credit; all consume wall time. Close immediately at the exhausted allowance before another selection/evaluation. No extra close/reconstruction replay, SA/repair or time extension.
- Two credits cannot reach the unchanged fourth-primary staged promotion. Record actual mode starts/completions and assert agreement with adapter receipts. A credit stop is separate from time expiry and must not set timeout flags.
- Use independent solver/decoder/config objects. Bind equality of initial sequence/layout/full score/candidate panel ledger, RNG state, decoder cache keys/counts and configuration values. No shared layouts/cache across arms.
- Per-arm30 s caller-created pipeline,3 s closing reserve,38 s parent watchdog,8 s bounded cleanup. Include data/preparation, initialization, search/reference, diagnostic capture/export, validation, independent geometry audits and minimal serialization. Parent watchdog covers process startup/imports; pipeline timing begins after runtime imports.

## Evidence and acceptance

Retain each completed evaluation's actual prefix/depth, complete sequence/raw snapshot, score, mode, replay time and exact digests, plus the common initial and final result. Capture snapshots during the run; export without replay under the absolute deadline. Evidence overhead is budgeted and comparable instrumentation applies to both arms; do not claim controlled speedup from one pair.

Every normally returned final incumbent has a raw receipt even if only Greedy initialization completed. Write the returned handoff checkpoint within budget before independent auditing. Exceptions preserve start/completion counts and available exports in failure evidence; offline inspection checks those receipts but never accepts a failed execution as a quality comparison. When the absolute deadline prevents export/write, record the failure phase and explicit raw-receipt unavailability; do not add time or promise complete evidence after hard process/OS failure.

Freeze source/scripts/input hashes and one739-piece seed1000 paired schedule before outcomes, randomizing arm order with schedule seed20261116. No other orders/seeds, retry, retuning or matrix expansion. Failed/partial outputs remain evidence; do not count an incomplete arm as a valid comparison. Compare panel count/net utilization first and secondary score/time descriptively. No hardcoded95 success requirement. Audit every completed evaluation and initial/final output against source dimensions, identity, orientation, trim and tool-clearance geometry, with raw↔nominal/digest bindings and best-of-initial/evaluations score consistency.

This tests early gain within two completions on a known order. It is not a formal PUCT ablation, an assessment of longer-search value, unseen generalization, optimality or formal acceptance. Existing cross-order ties and formal gate remain unchanged. Public code/tests/spec/plan/aggregate report only; raw factory IDs/dimensions/orders/layouts and private run receipts remain local. Update the existing authorized branch/draft; do not merge master or touch protected workspaces.

## Validation and review

Test quota exclusion of common initialization, no third evaluation/phantom counter/close replay, unchanged first-two search trajectory, exact existing reference entries, unchanged initial RNG/cache state, best-incumbent floor, unsupported profile/arm rejection, initial/completion/close expiry and partial evidence. Tests use real toy decoders; fault injection occurs only at actual replay/clock boundaries. Run the full project suite, compilation and whitespace checks. Review implementation and private harness before factory data; executor audits later outcomes/report/publication separately, without a second independent outcome-review claim.
