import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def implementation():
    path = Path(__file__).with_name('anchor_stream.py')
    assert path.exists(), 'anchor-only stream implementation is missing'
    spec = importlib.util.spec_from_file_location('factory_anchor_stream', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixed_initial_stream


def old_stream(solver, arm, h):
    for uid in (2, 1, 3):
        yield {'prefix': [uid]}, (uid,) + tuple(u for u in h if u != uid)


def test_replaces_only_completion_and_preserves_original_uid_order():
    solver = SimpleNamespace(global_best_sequence=[3, 1, 2])
    result = list(implementation()(old_stream, solver, 'SINGLE', (1, 2, 3)))
    assert result == [({'prefix': [2]}, (2, 3, 1)),
                      ({'prefix': [1]}, (1, 3, 2)),
                      ({'prefix': [3]}, (3, 1, 2))]


def test_anchor_stays_fixed_after_better_solver_sequence_appears():
    solver = SimpleNamespace(global_best_sequence=[3, 1, 2])
    stream = implementation()(old_stream, solver, 'SINGLE', (1, 2, 3))
    assert next(stream) == ({'prefix': [2]}, (2, 3, 1))
    solver.global_best_sequence[:] = [1, 2, 3]
    assert next(stream) == ({'prefix': [1]}, (1, 3, 2))


@pytest.mark.parametrize('arm,anchor', [('PAIR', [3, 1, 2]),
                                      ('SINGLE', [3, 1, 1]),
                                      ('SINGLE', [3, 1, 4])])
def test_rejects_wrong_arm_or_incomplete_anchor(arm, anchor):
    with pytest.raises(ValueError):
        list(implementation()(old_stream, SimpleNamespace(global_best_sequence=anchor),
                              arm, (1, 2, 3)))
