# Paper Experiment Execution Plan

This file translates the latest experiment design into commands that match the
current repository state.

## 1. Main Results with `baseline_sa`

Industrial main table, 4 methods:

```powershell
python 20260311\run_paper_experiments.py robustness `
  --suite industrial_xlsx6 `
  --exp-name exp_main_ind_all_1seed_300s `
  --variants baseline_greedy baseline_sa rh_mcts_ref rh_mcts_prefix_beam `
  --n-seeds 1 `
  --seed0 1000 `
  --solver-time-limit-s 300
```

Public main table, 4 methods:

```powershell
python 20260311\run_paper_experiments.py robustness `
  --suite public_compatible `
  --exp-name exp_main_pub_all_1seed_300s `
  --variants baseline_greedy baseline_sa rh_mcts_ref rh_mcts_prefix_beam `
  --n-seeds 1 `
  --seed0 1000 `
  --solver-time-limit-s 300
```

Outputs:

- `results_long.csv`
- `robustness_summary.csv`
- per-case `totals_*.csv`
- per-seed `placements.csv`, `boards_metrics.csv`

## 2. Robustness

Minimum recommended robustness run:

```powershell
python 20260311\run_paper_experiments.py robustness `
  --suite industrial_xlsx6 `
  --exp-name exp_robust_ind_5seed_1200s `
  --variants baseline_sa rh_mcts_ref rh_mcts_prefix_beam `
  --n-seeds 5 `
  --seed0 1000 `
  --solver-time-limit-s 1200
```

Stronger version:

```powershell
python 20260311\run_paper_experiments.py robustness `
  --suite industrial_xlsx6 `
  --exp-name exp_robust_ind_5seed_3600s `
  --variants baseline_sa rh_mcts_ref rh_mcts_prefix_beam `
  --n-seeds 5 `
  --seed0 1000 `
  --solver-time-limit-s 3600
```

## 3. Built-in Ablation

Industrial:

```powershell
python 20260311\run_paper_experiments.py ablation `
  --suite industrial_xlsx6 `
  --exp-name exp_ablation_ind_3seed_600s `
  --n-seeds 3 `
  --seed0 1000 `
  --solver-time-limit-s 600
```

Public:

```powershell
python 20260311\run_paper_experiments.py ablation `
  --suite public_compatible `
  --exp-name exp_ablation_pub_1seed_300s `
  --n-seeds 1 `
  --seed0 1000 `
  --solver-time-limit-s 300
```

## 4. Strong Ablation

The repository now supports:

- `ours_nomacro`
- `ours_nolightrepair`

Industrial:

```powershell
python 20260311\run_paper_experiments.py robustness `
  --suite industrial_xlsx6 `
  --exp-name exp_ablation_strong_ind `
  --variants rh_mcts_ref rh_mcts_full ours_nomacro ours_nolightrepair rh_mcts_prefix_beam `
  --n-seeds 3 `
  --seed0 1000 `
  --solver-time-limit-s 600
```

Public:

```powershell
python 20260311\run_paper_experiments.py robustness `
  --suite public_compatible `
  --exp-name exp_ablation_strong_pub `
  --variants rh_mcts_ref rh_mcts_full ours_nomacro ours_nolightrepair rh_mcts_prefix_beam `
  --n-seeds 1 `
  --seed0 1000 `
  --solver-time-limit-s 300
```

## 5. Strong External Baselines

The repository now supports two stronger external baselines aligned with the
latest Excel plan:

- `baseline_pilot`
- `baseline_full_beam`

`baseline_pilot` uses the current prefix-beam engine in a depth-1 / width-1
recursive pilot configuration with one fixed completion oracle.

`baseline_full_beam` uses the current full-sequence beam solver with a coarse
default of `beam width = 4`, `branch top-k = 8`.

Pilot comparison, industrial:

```powershell
python 20260311\tools\run_strong_baseline_batches.py compare `
  --baseline-kind pilot `
  --suite industrial_xlsx6 `
  --exp-name exp_strong_pilot_ind_1seed_300s `
  --n-seeds 1 `
  --seed0 1000 `
  --solver-time-limit-s 300
```

Pilot comparison, public:

```powershell
python 20260311\tools\run_strong_baseline_batches.py compare `
  --baseline-kind pilot `
  --suite public_compatible `
  --exp-name exp_strong_pilot_pub_1seed_300s `
  --n-seeds 1 `
  --seed0 1000 `
  --solver-time-limit-s 300
```

Full Beam comparison, industrial:

```powershell
python 20260311\tools\run_strong_baseline_batches.py compare `
  --baseline-kind full_beam `
  --suite industrial_xlsx6 `
  --exp-name exp_strong_full_beam_ind_1seed_300s `
  --n-seeds 1 `
  --seed0 1000 `
  --solver-time-limit-s 300
```

Full Beam comparison, public:

```powershell
python 20260311\tools\run_strong_baseline_batches.py compare `
  --baseline-kind full_beam `
  --suite public_compatible `
  --exp-name exp_strong_full_beam_pub_1seed_300s `
  --n-seeds 1 `
  --seed0 1000 `
  --solver-time-limit-s 300
```

Outputs:

- `results_long.csv`
- `summary_by_case_variant.csv`
- `comparison_wide.csv`

## 6. Time-Budget Scan

Industrial 6 cases, 3 seeds, budgets `10/30/60/120/300`:

```powershell
python 20260311\tools\run_strong_baseline_batches.py time_scan `
  --suite industrial_xlsx6 `
  --exp-name exp_strong_budget_ind_3seed `
  --n-seeds 3 `
  --seed0 1000 `
  --budgets 10 30 60 120 300
```

Public 10 representative cases, 1 seed, budgets `10/30/60/120/300`:

```powershell
python 20260311\tools\run_strong_baseline_batches.py time_scan `
  --suite public_compatible `
  --exp-name exp_strong_budget_pub10_1seed `
  --n-seeds 1 `
  --seed0 1000 `
  --budgets 10 30 60 120 300
```

Outputs:

- `results_long.csv`
- `time_scan_summary.csv`

## 7. Industrial Repeatability

Industrial 6 cases, 5 seeds:

```powershell
python 20260311\tools\run_strong_baseline_batches.py repeatability `
  --suite industrial_xlsx6 `
  --exp-name exp_strong_repeat_ind_5seed_300s `
  --n-seeds 5 `
  --seed0 1000 `
  --solver-time-limit-s 300
```

Outputs:

- `results_long.csv`
- `repeatability_summary.csv`
- `comparison_wide.csv`

## 8. External Baseline Parameter Grids

Pilot coarse tuning, industrial representatives:

```powershell
python 20260311\tools\run_strong_baseline_batches.py param_grid `
  --target pilot `
  --suite industrial_xlsx6 `
  --exp-name exp_strong_param_pilot_ind `
  --n-seeds 3 `
  --seed0 1000 `
  --solver-time-limit-s 300
```

Pilot coarse tuning, public representatives:

```powershell
python 20260311\tools\run_strong_baseline_batches.py param_grid `
  --target pilot `
  --suite public_compatible `
  --exp-name exp_strong_param_pilot_pub5 `
  --n-seeds 3 `
  --seed0 1000 `
  --solver-time-limit-s 300
```

Full Beam coarse tuning, industrial representatives:

```powershell
python 20260311\tools\run_strong_baseline_batches.py param_grid `
  --target full_beam `
  --suite industrial_xlsx6 `
  --exp-name exp_strong_param_full_beam_ind `
  --n-seeds 3 `
  --seed0 1000 `
  --solver-time-limit-s 300
```

Full Beam coarse tuning, public representatives:

```powershell
python 20260311\tools\run_strong_baseline_batches.py param_grid `
  --target full_beam `
  --suite public_compatible `
  --exp-name exp_strong_param_full_beam_pub5 `
  --n-seeds 3 `
  --seed0 1000 `
  --solver-time-limit-s 300
```

Outputs:

- `results_long.csv`
- `param_grid_summary.csv`

## 9. Prefix Parameter Sensitivity

Representative cases:

- `data2`
- `data6`
- `data3`
- `gcut6`
- `cgcut1`

Depth:

```powershell
python 20260311\tools\run_prefix_param_grid.py `
  --suite all_known `
  --exp-name exp_prefix_depth_grid `
  --variants rh_mcts_prefix_beam `
  --case-ids data2 data6 data3 gcut6 cgcut1 `
  --parameter PREFIX_BEAM_DEPTH `
  --values 1 2 3 4 5 `
  --n-seeds 1 `
  --seed0 1000 `
  --solver-time-limit-s 300
```

Width:

```powershell
python 20260311\tools\run_prefix_param_grid.py `
  --suite all_known `
  --exp-name exp_prefix_width_grid `
  --variants rh_mcts_prefix_beam `
  --case-ids data2 data6 data3 gcut6 cgcut1 `
  --parameter PREFIX_BEAM_WIDTH `
  --values 1 2 3 4 `
  --n-seeds 1 `
  --seed0 1000 `
  --solver-time-limit-s 300
```

Branch top-k:

```powershell
python 20260311\tools\run_prefix_param_grid.py `
  --suite all_known `
  --exp-name exp_prefix_branch_grid `
  --variants rh_mcts_prefix_beam `
  --case-ids data2 data6 data3 gcut6 cgcut1 `
  --parameter PREFIX_BEAM_BRANCH_TOPK `
  --values 2 3 4 5 `
  --n-seeds 1 `
  --seed0 1000 `
  --solver-time-limit-s 300
```

Outputs:

- `results_long.csv`
- `param_grid_summary.csv`

## 10. Layout Visualization

Current repo already contains tail-region visualization through:

- `20260311/tools/render_paper_figures.py`

For a future general layout comparison tool, recommended interface:

```powershell
python tools/render_layout_compare.py `
  --run-dir outputs/exp_main_ind_all_1seed_300s `
  --case-id data6 `
  --variants baseline_sa rh_mcts_ref rh_mcts_prefix_beam `
  --seed 1000 `
  --out figures/layout_data6_compare.png
```

## Recommended Execution Order

1. Main results with `baseline_sa`
2. Built-in ablation
3. Strong ablation
4. Strong external baselines (`Pilot`, `Full Beam`)
5. Time-budget scan
6. Industrial repeatability
7. External baseline parameter grids
8. Robustness
9. Prefix parameter sensitivity
