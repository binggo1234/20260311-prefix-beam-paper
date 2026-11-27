# CLASS06 standard geometry reference

This standalone research diagnostic asks whether the original `cl_06_100_02` fits into at most three 300×300 boards. It is a standard CP-SAT feasibility reference, not a prefix policy or a new packing method. It imports no `inrp` production code and is not registered as a solver variant.

Input: the original [2DPackLib CLASS archive](https://site.unibo.it/operations-research/en/research/2dpacklib), SHA-256 `9f5ca94d6a9ee5c9273302b8fbddf028c3f6a1d595e80622aafe79ec55589f9c`. The runner additionally checks the fixed member path and digest. No factory source data are included.

Model: every item is a mandatory fixed-size rectangle. Boards are tiled horizontally; each x variable has a union of board-specific legal intervals that forbids crossing the boundaries. All y variables stay within board height. One NoOverlap2D prevents overlaps. Fixed native orientation, zero trim/kerf, no guillotine restriction, no objective/hints/manual search strategy. See the [plan and completeness argument](../../docs/superpowers/plans/2026-11-26-class06-feasibility.md).

One fixed run uses OR-Tools 9.15.6755, 60 seconds solver time, 8 workers and seed20261126. The parent process applies a separate90-second watchdog. Parallel solving is not guaranteed deterministic across reruns or machines. The observed result is one bounded diagnostic, not a statistical comparison or a claim of optimum search speed.

From the project root:

```sh
python3 -m pytest -q research/class06_feasibility_20261126/test_reference.py
python3 research/class06_feasibility_20261126/run_reference.py --archive /absolute/path/to/CLASS.zip --out /absolute/path/to/new-output-directory
```

The second command is the runner only; an invoking controller must enforce the90-second process watchdog. It refuses existing output directories. It exports `model.pbtxt`, `parameters.txt` and `pre_solve.json` before its sole solve, followed by solver log, response and result. Its source validation happens before creating the directory. This is a pinned experiment tool for the fingerprinted source, not a general malformed-input API; some malformed Python-level objects can raise TypeError/KeyError instead of ValueError.

FEASIBLE/OPTIMAL needs an independently verified full native-board witness. With the known area lower bound3, a verified3-board witness proves optimum3. INFEASIBLE refers to the complete geometric model and, together with the saved valid4-board witness, proves optimum4. UNKNOWN leaves [3,4] unresolved. MODEL_INVALID, failed verification or a watchdog interruption is not evidence of infeasibility. See [official solver status meanings](https://developers.google.com/optimization/cp/cp_solver).

Synthetic tests include an independent cell-occupancy oracle, cross-board-boundary traps, non-area geometric obstruction, exact touching, full UID inventory, rotation/overlap/bounds corruption, source substitution and zero-budget UNKNOWN. No test invokes the original CLASS06 solve.
