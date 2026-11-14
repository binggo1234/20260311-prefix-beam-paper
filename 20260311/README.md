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

- [近邻方法差异审查，2026-11-14](docs/development/method-difference-audit-20261114.md)：取得2021非断头多板序列方法全文；受限前缀可精确嵌入BubbleSearch决策向量框架。现有表示不足以单独证明创新，生产代码继续冻结。
- [等12次完整评价的前缀策略对照，2026-11-13](docs/development/matched-prefix-budget-20261113.md)：739件已知订单上，当前搜索96张，简单短前缀扫描95张，后者第2次评价省板；统一强初解与入口状态，暂停继续投入当前复杂路线。不是前缀开关或PUCT消融。
- [身份安全的几何蓝图复用，2026-11-13](docs/development/blueprint-memo-20261113.md)：固定大订单重放耗时中位降幅约17.31%；同30 s配对完成候选12→15，双方均371张、净利用率89.877%，本次没有新增省料收益。
- [延迟构造与 2416 件 371 张见证，2026-11-13](docs/development/deferred-decoder-20261113.md)：同序列结果保持一致，固定工作量重放耗时中位降幅约12.35%，同30 s预算完成候选10→13并发现372→371张方案。
- [30 秒全订单单件前缀扫描，2026-11-13](docs/development/single-prefix-scan-20261113.md)：按需调用的扫描接口；三个已知订单配对一胜两平，178 件达到 28 张，并报告评价覆盖和收尾成本。
- [全订单单件前置诊断，2026-11-13](docs/development/single-prefix-landscape-20261113.md)：178 件订单遍历后找到 29→28 张见证，完成独立冷重放，并定位现有两 single 入口的覆盖缺口。
- [当前研究定位与创新证据门槛，2026-11-13](docs/development/restricted-prefix-contribution-20261113.md)：受限前缀的节材表达能力与预算效率；明确近邻文献、现有证据及尚未成立的贡献。
- [Restricted-prefix complement iteration, 2026-10-16](docs/development/prefix-complement-20261016.md): scope, budgeted usage, evidence fields, tests, and limitations. This development variant is not a replacement for frozen paper results.
- [Warmstart budget split, 2026-10-20](docs/development/warmstart-budget-20261020.md): preserve search time with a 20% / 5 s optional-initialization allowance while retaining exact Greedy.
- [Staged prefix completion, 2026-10-28](docs/development/staged-completion-20261028.md): opt-in single-mode screening and quota-controlled promotion, separate fidelity caches, paired development results and acceptance limits.
- [Type-fair expansion, 2026-10-30](docs/development/type-fair-expansion-20261030.md): isolated action-access policy and matched negative quality result; remains disabled, not adopted as a material-utilization improvement.
- [Complete saving witness advancement, 2026-11-11](docs/development/prefix-witness-20261111.md): default-off early prefix advancement; actual fresh deeper evaluation verified, but all final board counts tie the concurrent staged control, so no material-quality promotion.
