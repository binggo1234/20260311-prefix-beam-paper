# Strong Baseline Experiments Export

This folder contains only the code related to the newly added strong-baseline
experiment batches.

The copied files keep their original repository-relative paths under the
`20260311/` subtree so they can be merged back into the main project layout
without ambiguity.

Included files:

- `20260311/src/inrp/runner.py`
  - adds `baseline_pilot` and `baseline_full_beam`
- `20260311/tests/test_mcts_quality.py`
  - regression tests for the new variants
- `20260311/tools/run_strong_baseline_batches.py`
  - unified entry point for compare / time-scan / repeatability / param-grid
- `20260311/tools/run_paper_parallel_plan.py`
  - parallel lane scheduling updated with strong baseline tasks
- `20260311/tools/monitor_paper_extended_plan.py`
  - monitor updated with new experiment outputs
- `20260311/run_paper_extended_plan.ps1`
  - new stages: `strong_external`, `strong_time`, `strong_repeat`, `strong_params`, `strong_baselines`
- `20260311/EXPERIMENT_EXECUTION_PLAN.md`
  - updated execution notes for the new experiment groups

This export intentionally excludes unrelated workspace changes.
