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

- [30 秒全订单单件前缀扫描，2026-11-13](docs/development/single-prefix-scan-20261113.md)：按需调用的扫描接口；三个已知订单配对一胜两平，178 件达到 28 张，并报告评价覆盖和收尾成本。
- [全订单单件前置诊断，2026-11-13](docs/development/single-prefix-landscape-20261113.md)：178 件订单遍历后找到 29→28 张见证，完成独立冷重放，并定位现有两 single 入口的覆盖缺口。
- [当前研究定位与创新证据门槛，2026-11-13](docs/development/restricted-prefix-contribution-20261113.md)：受限前缀的节材表达能力与预算效率；明确近邻文献、现有证据及尚未成立的贡献。
- [Restricted-prefix complement iteration, 2026-10-16](docs/development/prefix-complement-20261016.md): scope, budgeted usage, evidence fields, tests, and limitations. This development variant is not a replacement for frozen paper results.
- [Warmstart budget split, 2026-10-20](docs/development/warmstart-budget-20261020.md): preserve search time with a 20% / 5 s optional-initialization allowance while retaining exact Greedy.
- [Staged prefix completion, 2026-10-28](docs/development/staged-completion-20261028.md): opt-in single-mode screening and quota-controlled promotion, separate fidelity caches, paired development results and acceptance limits.
- [Type-fair expansion, 2026-10-30](docs/development/type-fair-expansion-20261030.md): isolated action-access policy and matched negative quality result; remains disabled, not adopted as a material-utilization improvement.
- [Complete saving witness advancement, 2026-11-11](docs/development/prefix-witness-20261111.md): default-off early prefix advancement; actual fresh deeper evaluation verified, but all final board counts tie the concurrent staged control, so no material-quality promotion.
