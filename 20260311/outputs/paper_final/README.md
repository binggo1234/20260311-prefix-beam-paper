# Paper Final Outputs

This folder keeps the output directories that are worth citing in the paper draft.

## Main result root

- `formal_results/paper_overnight_formal_v2_20260318`

Key result files inside that directory:

- Industrial main table, `1 seed / 300s`:
  - `formal_results/paper_overnight_formal_v2_20260318/main_industrial_xlsx6_1seed_300s/suite_comparison_table.csv`
- Industrial formal table, `3 seeds / 3600s`:
  - `formal_results/paper_overnight_formal_v2_20260318/main_industrial_xlsx6_3seed_3600s/suite_comparison_table.csv`
- Public generalization table:
  - `formal_results/paper_overnight_formal_v2_20260318/generalization_public_1seed_300s/suite_comparison_table.csv`
- Ablation table:
  - `formal_results/paper_overnight_formal_v2_20260318/paper_ablation_industrial_xlsx6_1seed_300s/ablation_wide.csv`
- Robustness tables:
  - `formal_results/paper_overnight_formal_v2_20260318/paper_robustness_data3_5seed_900s/robustness_summary.csv`
  - `formal_results/paper_overnight_formal_v2_20260318/paper_robustness_data6_5seed_900s/robustness_summary.csv`
- Sensitivity tables:
  - `formal_results/paper_overnight_formal_v2_20260318/paper_sensitivity_data3/sensitivity_summary.csv`
  - `formal_results/paper_overnight_formal_v2_20260318/paper_sensitivity_data6/sensitivity_summary.csv`

## Current paper-facing conclusion

- Main method: `rh_mcts_prefix_beam`
- Industrial set `data1-data6`, `1 seed / 300s`:
  - wins vs `baseline_greedy`: `5`
  - ties vs `baseline_greedy`: `1`
  - losses vs `baseline_greedy`: `0`
  - wins vs `rh_mcts_ref`: `6`
- Industrial set `data1-data6`, `3 seeds / 3600s`:
  - `data1`: `342` vs baseline `343`
  - `data2`: `391` vs baseline `392`
  - `data3`: `174` vs baseline `175`
  - `data4`: `239` vs baseline `239`
  - `data5`: `198` vs baseline `199`
  - `data6`: best `150`, otherwise `151`, vs baseline `151`

## History files

- `history/data1_optimization_history.csv`
- `history/data2_optimization_history.csv`

## Notes

- Older exploratory outputs were not deleted. They were moved to `../archive_exploration`.
- This folder is intended to be the only output subtree needed during paper writing.
