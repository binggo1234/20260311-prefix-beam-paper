import random

import pytest


def policy(sequence=(1, 2, 3), panels=4, seed=17):
    from inrp import ash_sequence
    return ash_sequence.ASHSequencePolicy(sequence, panels, seed)


class Draws:
    def __init__(self, values):
        self.values = iter(values)

    def random(self):
        return next(self.values)


def test_module_is_present():
    import importlib.util
    assert importlib.util.find_spec('inrp.ash_sequence') is not None, 'ASH policy is missing'


def test_reset_to_first_remaining_item_after_acceptance():
    p = policy()
    p.rng = Draws([.05, .9, .8, .9])
    assert p.propose() == (2, 1, 3)
    assert p.random_draws == 4


def test_rejections_wrap_cyclically_and_threshold_equality_accepts():
    p = policy()
    p.rng = Draws([0, 0, 0, .1, .1, .1])
    assert p.propose() == (1, 2, 3)
    assert p.random_draws == 6


def test_strict_board_improvement_adopts_and_resets_but_ties_do_not():
    p = policy()
    p.rng = Draws([.05, .9, .8, .9] * 3)
    first = p.propose()
    p.observe(4)
    assert p.base_sequence == (1, 2, 3) and p.alpha == pytest.approx(.102)
    second = p.propose()
    p.observe(3)
    assert p.base_sequence == second and p.best_panels == 3 and p.alpha == .1
    p.propose()
    p.observe(4)
    assert p.base_sequence == second and p.best_panels == 3


def test_seeded_permutations_and_saturation():
    a, b = policy(), policy()
    for _ in range(400):
        sa, sb = a.propose(), b.propose()
        assert sa == sb and sorted(sa) == [1, 2, 3]
        a.observe(4)
        b.observe(4)
    assert a.alpha == b.alpha == .65
    assert a.random_draws == b.random_draws


def test_pending_result_must_be_observed_once():
    p = policy()
    with pytest.raises(RuntimeError):
        p.observe(3)
    p.propose()
    with pytest.raises(RuntimeError):
        p.propose()
    p.observe(3)
    with pytest.raises(RuntimeError):
        p.observe(2)


def test_cancellation_checks_inside_random_rejection_loop():
    p = policy()
    p.rng = Draws([0] * 10)
    calls = []
    def check():
        calls.append(1)
        if len(calls) == 4:
            raise TimeoutError('cancel')
    with pytest.raises(TimeoutError):
        p.propose(check)
    assert p.random_draws == 3 and p.pending_sequence is None
    assert p.base_sequence == (1, 2, 3) and p.alpha == .1


@pytest.mark.parametrize('sequence,panels', [((), 1), ((1, 1), 1), ((1, 2), 0), ((1,), True)])
def test_invalid_initial_state(sequence, panels):
    with pytest.raises(ValueError):
        policy(sequence, panels)


def test_invalid_result_does_not_consume_pending_proposal():
    p = policy()
    pending = p.propose()
    with pytest.raises(ValueError):
        p.observe(-1)
    assert p.pending_sequence == pending
    p.observe(3)
