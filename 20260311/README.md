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

- [创新边界与目标反馈小诊断，2026-11-26](docs/development/innovation-scope-feedback-diagnostic-20261126.md)：CLASS07的100个单件目标容量扫描中，已知联合成员7排48–55、87与47件并列第一；单件缺件面积不能清楚提示该组合。停止直接按此评分截取首件的方案，已有思想保留参照；8项私有检查和104份布局只读审计通过，生产代码/工厂数据/主稿未改，无省板新发现或创新成立结论。
- [定向文献与机制决策，2026-11-26](docs/development/mechanism-literature-decision-20261126.md)：核对GDRR、GeoNest及神经启发式参数搜索等方法和两个作者仓库。优先研究目标板数下的未放件反馈，明确与已停用剪枝/阻塞规则的区别、评价接口变化和停止条件；尚非新创新，无代码改动或新增利用率结果。
- [工厂尺寸角色核查，2026-11-26](docs/development/factory-input-dimension-contract-20261126.md)：原81次实验保留为Fleng/Fwidth名义模型对照。9份可解析XML的17147件轴对齐矩形，其CutLength/CutWidth均对应RawLength/RawWidth，仅1403件也对应Fleng/Fwidth。需显式区分尺寸角色及边界语义，不能自动声称实际下料制造验证；5/6毫米值、默认算法及主稿未改，未产生新算法或省板成果。
- [全部27个工厂任务直接对照，2026-11-26](docs/development/factory-ash-reference-results-20261126.md)：81次30秒运行及10068份布局记录核验通过。共同初解5381张，单件5365、两件5363、ASH5364；两件相对强单件净少2张、整体净利用率增加0.033735个百分点。ASH仅单种子，无稳定优势或新创新结论；2416件本轮未覆盖旧371张序列，旧见证复核仍有效。
- [同解码器ASH序列策略直接对照，2026-11-26](docs/development/ash-prefix-reference-results-20261126.md)：30次冻结运行全部有效；1000评价组CLASS07的PAIR和ASH三种子均27张，5秒组均28张；CLASS04中ASH第1–2次新评价省板，单件扫描第29次。466项回归通过、4项私有计时回归通过；无新工厂结果，不成立前缀独有评价效率或新创新。
- [余件—旧板面积流检验，2026-11-26](docs/development/residual-area-flow-probe-20261126.md)：12个保存状态诊断；五个两件状态的下界全部23.5954，未通过预定早期区分检验，停止该评分。9项原型测试及独立四割验证通过，无新工业搜索或材料收益；下一步仅补齐同解码器ASH序列策略参照，尚未实施。
- [CLASS07前缀联合效应，2026-11-26](docs/development/class07-prefix-interaction-20261126.md)：五条给定序列冷重放，两个单件均28张、正反两件前缀均27张；新定位同面积余料对69×15零件接纳能力的差异。用于研究联合选择及余件—余料适配，尚非新方法、工业增益或泛化证据。
- [目标板数可行性机制核查，2026-11-26](docs/development/target-feasibility-probe-20261126.md)：普通剩余面积界不区分前缀；181条保存序列的封闭板界相对开板上限额外提前量为0，停止该候选。11项原型检查、440项生产回归通过；无工厂新搜索、无新增利用率结果。
- [完整几何序列去重，2026-11-26](docs/development/exact-geometry-dedup-20261126.md)：默认关闭的精确候选去重；440项测试通过，三组开发配对有效。固定工作量178件流水线耗时约降49.89%、2416件约降6.83%；同30秒不同几何覆盖12→13，双方仍371张，无新增材料收益。用户转达的工厂口径已确认：不考虑纹理、修边5毫米、刀径6毫米，与现有实验设置一致；此前纹理和修边缺口判断已撤回。
- [三篇近邻的机制筛选，2026-11-18](docs/development/near-neighbor-mechanism-20261118.md)：取得FAEO作者接收稿并核对续算机制；不同单件前置候选的共同序列开头为零，不直接移植其存档查找。Pilot与2026预见剪枝仍缺完整方法正文；无新排样实验。
- [跨行业排样文献研判，2026-11-17](docs/development/cross-industry-literature-20261117.md)：扩展至玻璃、船舶钣金、服装皮革、物流及芯片/PCB；16项分层条目、12个DOI核对、7个仓库快照。区分目标可行性、评价精度、布局修正和几何学习；本轮不编码新机制、不运行排样。
- [受限前缀章节取舍与证据边界，2026-11-17](docs/development/restricted-prefix-chapter-disposition-20261117.md)：固定强初解＋简单扫描，停止当前复杂路线追加；本地形成章稿及创新证据表，分开报告材料潜力、预算效率和工程优化。独立算法创新及新工业订单有效性尚未成立；默认求解器未改。
- [K2生命周期修复及同预算复核，2026-11-17](docs/development/k2-lifecycle-repair-20261117.md)：消除记录钩子的引用环；含CLASS05单列中断补做，原5例均在120秒内正常退出，旧记录保留。CLASS03完整9900候选无省板见证，其余4例时间删失；72项相关测试通过。
- [K=2表达增量诊断，2026-11-16](docs/development/k2-reachability-20261116.md)：CLASS07由完整K=1的28张降为27张，1例有效正例；5例退出失败，不作阴性结论。47,036份保存布局审计通过；修正跨进程绝对时钟误比，原记录保留。
- [完整K=1可达性诊断，2026-11-16](docs/development/k1-reachability-20261116.md)：7例各100候选全覆盖，693次新序列重放、714份布局审计；CLASS04在第29次评价从4张降至可证明最优的3张，其余6例无单件省板见证。未新增启发式。
- [外部CLASS算例筛查，2026-11-15](docs/development/external-sequence-screen-20261115.md)：10个项目新公开实例、30次有效运行、450份布局审计；简单前缀相对每例最强参照0胜10平0负，未通过预定继续筛查门槛。现有27份工业任务均已历史暴露，不能再称未见集。
- [同解码器序列邻域实测结果，2026-11-15](docs/development/sequence-neighborhood-results-20261115.md)：已知739件订单、每臂12次新序列评价，简单单件前缀95张，一般插入及Kendall穷举访问均96张；45份布局审计通过。保留简单强基线，尚未证明创新或独立订单优势。
- [执行前冻结的序列邻域对照协议v1，2026-11-15](docs/development/sequence-neighborhood-protocol-20261115.md)：三个参照的候选顺序、统一去重及预算规则；执行状态见上述结果报告。
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
