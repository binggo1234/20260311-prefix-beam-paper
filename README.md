# 20260311 Prefix-Beam Paper Repo

This repository currently uses `20260311/` as the real project root.

## Where to look first

- Main project root: `20260311/`
- Core algorithm code: `20260311/src/inrp/`
- Main solver file: `20260311/src/inrp/mcts.py`
- Solver orchestration: `20260311/src/inrp/runner.py`
- Main configuration: `20260311/src/inrp/cfg.py`
- Benchmark runner: `20260311/run_benchmark_suite.py`
- Paper experiment runner: `20260311/run_paper_experiments.py`

## Paper-facing outputs

Final paper results were curated into:

- `20260311/outputs/paper_final/`

Historical exploratory runs were moved to:

- `20260311/outputs/archive_exploration/`

## Notes

- The top-level duplicate `src/` directory is not part of the active project layout.
- If you only need to read or modify the algorithm, work inside `20260311/src/inrp/`.
