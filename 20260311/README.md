# Project Layout

This directory is the actual project root used by the solver and experiment scripts.

## Main folders

- `src/inrp/`
  - core algorithm package
- `data/`
  - industrial and public benchmark inputs
- `tests/`
  - regression and unit tests
- `outputs/`
  - experiment outputs
- `experiments/`
  - small helper experiment scripts

## Core files

- `src/inrp/mcts.py`
  - main implementation of `rh_mcts_ref`, `rh_mcts_beam`, `rh_mcts_prefix_beam`, and the opt-in `rh_mcts_prefix_beam_complement`
- `src/inrp/runner.py`
  - variant registry and run orchestration
- `src/inrp/cfg.py`
  - shared configuration
- `run_benchmark_suite.py`
  - suite-based benchmark runner
- `run_paper_experiments.py`
  - paper table generation entry point

## Output policy

- Final paper results live in `outputs/paper_final/`
- Older exploratory outputs live in `outputs/archive_exploration/`

## Development notes

- [Restricted-prefix complement iteration, 2026-10-16](docs/development/prefix-complement-20261016.md): scope, budgeted usage, evidence fields, tests, and limitations. This development variant is not a replacement for frozen paper results.
- [Warmstart budget split, 2026-10-20](docs/development/warmstart-budget-20261020.md): preserve search time with a 20% / 5 s optional-initialization allowance while retaining exact Greedy.
