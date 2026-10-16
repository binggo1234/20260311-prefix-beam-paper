from collections import defaultdict
from types import SimpleNamespace
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from inrp.budget import BudgetExpired
from inrp.mcts import RecedingHorizonMCTS


def solver_for(outcomes):
    solver = RecedingHorizonMCTS.__new__(RecedingHorizonMCTS)
    solver.meta = defaultdict(float)
    solver.oracle_completion_cache = {}
    solver.oracle_completion_mode_cache = {}
    solver.prefix_beam_completion_policy = 'deterministic'
    solver.prefix_beam_oracle_modes = ('baseline', 'long_edge', 'strip_bias')
    solver.prefix_beam_oracle_cache_enable = True
    solver.search_deadline = 999999999.0
    solver._baseline_completion_order = lambda remaining_mask, mode: (1, 2)
    solver._terminal_score = lambda snapshot: (-snapshot.boards,)
    solver._prefix_beam_completed_rank = lambda snapshot, prefix_depth: (-snapshot.boards,)
    sequence = iter(outcomes)
    def replay(*args, **kwargs):
        outcome = next(sequence)
        if outcome is None:
            raise BudgetExpired('completion_replay')
        return SimpleNamespace(boards=outcome, layout_hash=(outcome, 0), ordered_hash=outcome), None
    solver.decoder = SimpleNamespace(replay_order=replay)
    return solver


def complete(solver):
    return solver._complete_with_baseline(
        SimpleNamespace(layout_hash=(0, 0), ordered_hash=0), 3, prefix_depth=0)


@pytest.mark.parametrize('outcomes,expected', [([9, None], 9), ([9, 8, None], 8), ([8, 9, None], 8)])
def test_completed_candidate_survives_later_deadline(outcomes, expected):
    solver = solver_for(outcomes)
    suffix, candidate, score = complete(solver)
    assert candidate.boards == expected and suffix == (1, 2)
    assert score == (-expected,)
    assert solver.meta['solver_timeout_hit'] == 1
    assert solver.meta['oracle_completion_partial_returned'] == 1
    assert solver.meta['oracle_completion_partial_layout_hash'] == [expected, 0]
    assert not solver.oracle_completion_cache  # never cache an incomplete mode group


def test_first_mode_deadline_still_raises():
    solver = solver_for([None])
    with pytest.raises(BudgetExpired):
        complete(solver)
    assert not solver.oracle_completion_cache


def test_completion_accounting_distinguishes_started_and_finished_modes():
    solver = solver_for([9, None])
    complete(solver)
    assert solver.meta["oracle_completion_mode_evals"] == 2
    assert solver.meta["oracle_completion_mode_completed"] == 1
    assert solver.meta["oracle_completion_mode_timeouts"] == 1

    finished = solver_for([9, 8, 7])
    complete(finished)
    assert finished.meta["oracle_completion_mode_evals"] == 3
    assert finished.meta["oracle_completion_mode_completed"] == 3
    assert finished.meta["oracle_completion_mode_timeouts"] == 0


def test_complete_modes_keep_existing_ranking_cache_and_ties():
    solver = solver_for([9, 8, 8])
    first = complete(solver)
    assert first[1].boards == 8
    assert solver.meta['oracle_completion_last_best_mode'] == 'long_edge'
    assert complete(solver) == first
    assert solver.meta['oracle_completion_cache_hits'] == 1


@pytest.mark.parametrize('outcomes,expected', [([9, None], 9), ([None], 10)])
def test_simulation_preserves_incumbent_at_deadline(outcomes, expected):
    solver = solver_for(outcomes)
    leaf = SimpleNamespace(prefix_sequence=(), snapshot=SimpleNamespace(layout_hash=(0, 0), ordered_hash=0),
                           remaining_mask=3, depth=0)
    solver.prefix_beam_enable = True
    solver.beam_enable = False
    solver.global_best_snapshot = SimpleNamespace(boards=10)
    solver._select_leaf = lambda root: [leaf]
    solver._record_layout_candidate = lambda *args, **kwargs: None
    solver._reward_scalar = lambda *args: 1.0
    solver._backup = lambda *args: None
    def update(sequence, candidate):
        if candidate.boards < solver.global_best_snapshot.boards:
            solver.global_best_snapshot = candidate
    solver._update_global_best = update
    solver._run_simulations(leaf, 1)
    assert solver.global_best_snapshot.boards == expected
