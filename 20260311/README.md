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
  - main implementation of `rh_mcts_ref`, `rh_mcts_beam`, and `rh_mcts_prefix_beam`
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
