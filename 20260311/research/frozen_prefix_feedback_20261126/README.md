# Frozen bounded-prefix feedback policy

Archive of the policy tested in the 2026-11-26 public validation. It is not registered as a production solver. The frozen overlap policy did not establish an increment over direct random pair visitation; see the [results](../../docs/development/public-feedback-validation-results-20261126.md).

`policy.py` and `test_policy.py` are byte-identical to the locally frozen research version. Policy SHA-256: `0a505e143f8b6bc51229f0594edcc7831712a8bbc90a53e2f37208217342f9b1`.

Run the standalone policy checks from the project directory:

```sh
python3 -m pytest -q research/frozen_prefix_feedback_20261126/test_policy.py
```

The caller supplies a fixed complete UID order H and target-capacity scan feedback as counts of fixed-orientation `(width, height)` types. Short prefixes have depth at most two. Initial full-layout exclusions and target feedback have different semantics; an ordinary full layout does not supply a missing-item signature. Each policy instance belongs to one fixed input/decoder/target context.

This directory archives the pure visitation policy and its checks. It does not include the private source adapter, capped geometry evaluator, metered process runner, or full-layout receipts, and is not presented as a complete end-to-end reproduction bundle. The [public protocol](../../docs/development/public-feedback-validation-protocol-20261126.md) and [aggregate records](../../docs/development/assets/public-feedback-validation-20261126.csv) describe the evaluated setup. Original CLASS data remain available from [2DPackLib](https://site.unibo.it/operations-research/en/research/2dpacklib).

The active FIFO window is limited to 16 signatures; that is not a bound on total cached signatures or memory. The feedback-reuse API produced no reuse events in these evaluated runs. No industrial rotation, trim or kerf adaptation was validated by this policy archive.
