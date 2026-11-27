# Fixed initial-anchor reference

Standard move-to-front composition for the SINGLE reference: preserve its original UID visit order, freeze the selected common initializer sequence once, and keep all non-prefix UIDs in that captured order. No adaptation after an improvement. This is a reference control, not a new heuristic or a registered production solver.

`anchor_stream.py` and `test_anchor_stream.py` are byte-identical archives of the tested composition used in the [factory F15 experiment](../../docs/development/factory-greedy-anchor-results-20261127.md). They do not include customer data or any original factory IDs. The factory input adapter, operational worker, full layout traces and saved-only geometry audit remain in local QA; this is not a complete end-to-end experiment package.

An isolated worker wrapped the unchanged `inrp.sequence_reference._prefix_stream` with `fixed_initial_stream(original_stream, solver, arm, h)` and temporarily substituted it for that process only. The existing reference retained two common initializers, decoder, visit order, counters, deadlines, receipts and validation. Production files and defaults were not modified.

Run the synthetic behavior tests from the project root:

```sh
python3 -m pytest -q research/fixed_initial_anchor_20261127/test_anchor_stream.py
```

Tests verify actual completed sequences and fixed-anchor semantics. They do not run the original factory case. The archived helper validates SINGLE arm and complete UID inventory; supported inputs are those supplied by the existing reference stream.
