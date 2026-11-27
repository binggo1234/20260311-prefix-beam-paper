"""Behavioral validation of the standalone, fixed-orientation reference."""
import importlib.util
import hashlib
from itertools import product
from pathlib import Path
import random

import pytest


def load(name):
    path = Path(__file__).with_name(name + '.py')
    assert path.exists(), 'reference implementation is missing: ' + name
    spec = importlib.util.spec_from_file_location('class06_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def solved(shapes, width, height, boards, limit=2):
    return load('model').solve_reference(shapes, width, height, boards,
                                         time_limit_s=limit, workers=1, seed=7)


def exhaustive(shapes, width, height, boards):
    # Independent cell-occupancy oracle, not the model's separation formula.
    ordered = sorted(shapes.values(), key=lambda wh: wh[0] * wh[1], reverse=True)
    if sum(w*h for w, h in ordered) > width*height*boards:
        return False

    def visit(index, masks):
        if index == len(ordered):
            return True
        w, h = ordered[index]
        for b in range(boards):
            if b and masks[b] == masks[b-1]:
                continue
            for x, y in product(range(width-w+1), range(height-h+1)):
                mask = sum(1 << (v*width+u) for u in range(x, x+w) for v in range(y, y+h))
                if masks[b] & mask:
                    continue
                changed = list(masks)
                changed[b] |= mask
                if visit(index+1, changed):
                    return True
        return False

    return visit(0, [0] * boards)


def test_forbids_crossing_between_adjacent_boards():
    result = solved({i: (3, 2) for i in range(1, 4)}, 5, 2, 2)
    assert result['status'] == 'INFEASIBLE'
    assert result['placements'] is None


def test_non_area_obstruction_and_feasible_two_board_witness():
    shapes = {i: (6, 4) for i in range(1, 5)}
    assert solved(shapes, 10, 10, 1)['status'] == 'INFEASIBLE'
    result = solved(shapes, 10, 10, 2)
    assert result['status'] in ('OPTIMAL', 'FEASIBLE')
    assert load('verify_witness').verify_witness(shapes, 10, 10, 2, result['placements']) == 2


def test_edge_contact_and_native_orientation():
    result = solved({1: (3, 2), 2: (3, 2)}, 3, 4, 1)
    assert result['status'] in ('OPTIMAL', 'FEASIBLE')
    assert load('verify_witness').verify_witness({1: (3, 2), 2: (3, 2)}, 3, 4, 1,
                                               result['placements']) == 1
    with pytest.raises(ValueError):
        solved({1: (4, 3)}, 3, 4, 1)


def test_small_model_matches_independent_grid_oracle():
    rng = random.Random(20261126)
    for _ in range(32):
        shapes = {i: (rng.randint(1, 4), rng.randint(1, 4)) for i in range(1, 5)}
        boards = rng.choice([1, 2])
        expected = exhaustive(shapes, 4, 4, boards)
        result = solved(shapes, 4, 4, boards)
        assert result['status'] != 'UNKNOWN'
        assert (result['status'] in ('OPTIMAL', 'FEASIBLE')) == expected
        if expected:
            load('verify_witness').verify_witness(shapes, 4, 4, boards, result['placements'])


def test_zero_budget_is_unknown_and_not_infeasible():
    result = solved({1: (2, 2), 2: (2, 2)}, 4, 4, 1, limit=0)
    assert result['status'] == 'UNKNOWN'
    assert result['placements'] is None


@pytest.mark.parametrize('bad', [
    [{'uid': 1, 'board_id': 1, 'x': 0, 'y': 0, 'rotated': False}],
    [{'uid': 1, 'board_id': 1, 'x': 0, 'y': 0, 'rotated': False},
     {'uid': 1, 'board_id': 1, 'x': 2, 'y': 0, 'rotated': False}],
    [{'uid': 1, 'board_id': 1, 'x': 0, 'y': 0, 'rotated': True},
     {'uid': 2, 'board_id': 1, 'x': 2, 'y': 0, 'rotated': False}],
    [{'uid': 1, 'board_id': 1, 'x': 0, 'y': 0, 'rotated': False},
     {'uid': 2, 'board_id': 1, 'x': 1, 'y': 0, 'rotated': False}],
    [{'uid': 1, 'board_id': 1, 'x': 0, 'y': 0, 'rotated': False},
     {'uid': 2, 'board_id': 2, 'x': 0, 'y': 0, 'rotated': False}],
    [{'uid': 1, 'board_id': 1, 'x': 0, 'y': 0, 'rotated': False},
     {'uid': 2, 'board_id': 1, 'x': 3, 'y': 0, 'rotated': False}],
    [{'uid': 1, 'board_id': 1, 'x': 0, 'y': 0, 'rotated': False},
     {'uid': 2, 'board_id': 1, 'x': float('nan'), 'y': 0, 'rotated': False}],
])
def test_verifier_rejects_missing_duplicate_rotated_overlapping_or_outside_items(bad):
    with pytest.raises(ValueError):
        load('verify_witness').verify_witness({1: (2, 2), 2: (2, 2)}, 4, 2, 1, bad)


def test_verifier_accepts_touching_and_distinct_board_coordinates():
    check = load('verify_witness').verify_witness
    p = [{'uid': 1, 'board_id': 1, 'x': 0, 'y': 0, 'rotated': False},
         {'uid': 2, 'board_id': 1, 'x': 2, 'y': 0, 'rotated': False}]
    assert check({1: (2, 2), 2: (2, 2)}, 4, 2, 1, p) == 1
    p[1].update(board_id=2, x=0)
    assert check({1: (2, 2), 2: (2, 2)}, 4, 2, 2, p) == 2


def test_source_parser_preserves_native_sizes_and_rejects_partial_inventory():
    parse = load('run_reference').parse_member
    raw = b'2\n5 4\n1 4 3 1 1 0\n2 1 2 1 1 0\n'
    assert parse(raw) == (5, 4, {1: (4, 3), 2: (1, 2)})
    for bad in (raw.replace(b'2\n5', b'3\n5'),
                raw.replace(b'2 1 2', b'1 1 2'),
                raw.replace(b'1 1 0', b'1 1 1'),
                raw.replace(b'4 3', b'6 3')):
        with pytest.raises(ValueError):
            parse(bad)


def test_runner_rejects_source_substitution_and_existing_output(tmp_path):
    runner = load('run_reference')
    wrong = tmp_path / 'wrong.zip'
    wrong.write_bytes(b'wrong archive')
    assert hashlib.sha256(wrong.read_bytes()).hexdigest() != runner.ARCHIVE_SHA256
    with pytest.raises(ValueError, match='archive fingerprint'):
        runner.run(wrong, tmp_path / 'new')
    assert not (tmp_path / 'new').exists()
    with pytest.raises(FileExistsError):
        runner.run(wrong, tmp_path)
