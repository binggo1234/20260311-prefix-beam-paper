"""Solver-free native-board geometry check; does not import the CP model."""
from collections import Counter
from itertools import combinations


def verify_witness(shapes, width, height, boards, placements):
    if not placements or Counter(p['uid'] for p in placements) != Counter(shapes.keys()):
        raise ValueError('missing, duplicate or foreign item identity')
    for p in placements:
        if p['rotated'] is not False:
            raise ValueError('rotation is forbidden')
        if any(type(p[k]) is not int for k in ('uid', 'board_id', 'x', 'y')):
            raise ValueError('integer identity, board ID and coordinates required')
        if not 1 <= p['board_id'] <= boards:
            raise ValueError('board ID outside target')
        w, h = shapes[p['uid']]
        if not (0 <= p['x'] <= width - w and 0 <= p['y'] <= height - h):
            raise ValueError('item outside its own board')
    for p, q in combinations(placements, 2):
        if p['board_id'] != q['board_id']:
            continue
        w, h = shapes[p['uid']]
        v, k = shapes[q['uid']]
        if not (p['x'] + w <= q['x'] or q['x'] + v <= p['x'] or
                p['y'] + h <= q['y'] or q['y'] + k <= p['y']):
            raise ValueError('same-board overlap')
    return len({p['board_id'] for p in placements})
