from pathlib import Path
import sys
sys.path.insert(0, str(Path('20260311/src').resolve()))
from inrp.cfg import CFG
from inrp.runner import run_stepF

cfg = CFG()
cfg.CASE_NAME = 'data1_formal_fullcpu_20260313'
cfg.SAMPLE_CSV = r'20260311/outputs/ad_hoc_inputs/data1.csv'
cfg.OUT_ROOT = r'20260311/outputs'
cfg.VARIANTS = ('baseline_greedy', 'rh_mcts_main')
cfg.N_SEEDS = 1
cfg.SEED0 = 1000
cfg.N_JOBS = 2
cfg.POST_REPAIR_CP_NUM_WORKERS = 14
cfg.SOLVER_TIME_LIMIT_S = 900.0
run_stepF(cfg)
