import random

import pytest

from decoder_blueprint_reference import BlueprintReferenceDecoder
from test_decoder_deferred import cfg, fingerprint
from inrp.dataio import Part
import inrp.mcts as mcts
from inrp.packer import (Board, _place_one, apply_local_blueprint,
                         board_signature, clone_board, make_empty_board)


def record_geometry(monkeypatch):
    calls = []
    original = mcts.best_local_placement

    def record(*args, **kwargs):
        calls.append(args[1].uid)
        return original(*args, **kwargs)

    monkeypatch.setattr(mcts, "best_local_placement", record)
    return calls


@pytest.mark.parametrize("size", [(10, 10), (60, 60)])
def test_equal_shapes_reuse_only_geometry_and_keep_uid_hashes(monkeypatch, size):
    parts = {i: Part(i, str(i), *size, *size) for i in (1, 2)}
    decoder = mcts.OmnipotentDecoder(parts, cfg())
    board = make_empty_board(1, 50, 40, trim=1)
    calls = record_geometry(monkeypatch)
    left = decoder._micro_decode(board, parts[1])
    right = decoder._micro_decode(board, parts[2])
    assert calls == [1], "same board and shape must compute geometry once"
    assert decoder.micro_misses == 2 and decoder.micro_hits == 0
    assert len(decoder.micro_cache) == 2
    if size == (60, 60):
        assert left is right is None
    else:
        assert left.blueprint == right.blueprint
        assert left.board_hash_after != right.board_hash_after
        assert left.board_term_after == right.board_term_after == .05 ** 2
        a = apply_local_blueprint(board, parts[1], left.blueprint)
        b = apply_local_blueprint(board, parts[2], right.blueprint)
        assert [p.uid for p in a.placed] == [1]
        assert [p.uid for p in b.placed] == [2]


def test_first_maxrects_initialization_keeps_the_memo(monkeypatch):
    parts = {i: Part(i, str(i), 10, 10, 10, 10) for i in (1, 2)}
    decoder = mcts.OmnipotentDecoder(parts, cfg())
    board = Board(1, 50, 40, trim=1)
    calls = record_geometry(monkeypatch)
    assert decoder._micro_decode(board, parts[1]) is not None
    assert decoder._micro_decode(board, parts[2]) is not None
    assert calls == [1]


def test_mutation_invalidates_and_clones_do_not_inherit_blueprints(monkeypatch):
    parts = {i: Part(i, str(i), 10, 10, 10, 10) for i in range(1, 5)}
    decoder = mcts.OmnipotentDecoder(parts, cfg())
    board = make_empty_board(1, 50, 40, trim=1)
    calls = record_geometry(monkeypatch)
    first = decoder._micro_decode(board, parts[1])
    clone = clone_board(board)
    mcts._board_state_hash(clone, decoder.sig_nd)
    assert clone == board  # transient memo is excluded from equality
    assert decoder._micro_decode(clone, parts[2]) is not None
    assert calls == [1, 2]
    assert _place_one(board, parts[3], True)
    after = decoder._micro_decode(board, parts[4])
    assert calls == [1, 2, 4]
    assert after.blueprint.rect != first.blueprint.rect
    assert board_signature(clone) != board_signature(board)


@pytest.mark.parametrize("change", ["w", "h", "w0", "h0", "rotations"])
def test_exact_query_time_shape_key_includes_nominal_and_rotation(change, monkeypatch):
    parts = {i: Part(i, str(i), 20, 10, 18, 8) for i in (1, 2)}
    decoder = mcts.OmnipotentDecoder(parts, cfg())
    reference = BlueprintReferenceDecoder(parts, cfg())
    board = make_empty_board(1, 50, 40, trim=1)
    calls = record_geometry(monkeypatch)
    decoder._micro_decode(board, parts[1])
    # Mutate a not-yet-queried UID after construction: a constructor-time key
    # would silently reuse the other UID's nominal area or rotation policy.
    if change == "rotations":
        parts[2].allowed_rotations_deg = (90,)
    else:
        setattr(parts[2], change, getattr(parts[2], change) + .00001)
    actual = decoder._micro_decode(board, parts[2])
    assert calls == [1, 2]
    assert actual == reference._micro_decode(clone_board(board), parts[2])


@pytest.mark.parametrize("mode,allow_rot", [("maxrects_bssf", True), ("blf", True),
                                          ("maxrects_baf", False)])
def test_shared_board_reused_by_fresh_decoder_respects_configuration(mode, allow_rot, monkeypatch):
    parts = {i: Part(i, str(i), 20, 10, 18, 8) for i in (1, 2)}
    board = make_empty_board(1, 50, 40, trim=1)
    first_cfg = cfg(); second_cfg = cfg(mode); second_cfg.ALLOW_ROT = allow_rot
    calls = record_geometry(monkeypatch)
    mcts.OmnipotentDecoder(parts, first_cfg)._micro_decode(board, parts[1])
    actual = mcts.OmnipotentDecoder(parts, second_cfg)._micro_decode(board, parts[2])
    assert calls == [1, 2]
    assert actual == BlueprintReferenceDecoder(parts, second_cfg)._micro_decode(clone_board(board), parts[2])


@pytest.mark.parametrize("mode", ["maxrects_baf", "maxrects_bssf", "blf"])
def test_replay_every_step_and_legacy_cache_equal_uid_only_reference(mode):
    rng = random.Random(20261120)
    parts = {i: Part(i, str(i), w, h, w-1, h-1, allowed_rotations_deg=rot)
             for i, w, h, rot in [(1, 19, 13, (0, 90)), (2, 19, 13, (0, 90)),
                                  (3, 28, 18, (0,)), (4, 18, 28, (90,)),
                                  (5, 23, 23, (0, 90)), (6, 10, 14, (0, 90))]}
    old = BlueprintReferenceDecoder(parts, cfg(mode, 1.3))
    new = mcts.OmnipotentDecoder(parts, cfg(mode, 1.3))
    for _ in range(12):
        order = list(parts); rng.shuffle(order)
        a, b = mcts._empty_snapshot(), mcts._empty_snapshot()
        for uid in order:
            left, right = old.decode(a, uid), new.decode(b, uid)
            assert fingerprint(left) == fingerprint(right)
            a, b = left.snapshot, right.snapshot
        for name in ("micro_cache", "board_term_cache"):
            assert getattr(old, name) == getattr(new, name)
        assert old.decode_cache.keys() == new.decode_cache.keys()
        assert (old.micro_hits, old.micro_misses, old.decode_hits, old.decode_misses) == (
            new.micro_hits, new.micro_misses, new.decode_hits, new.decode_misses)
        assert sorted(p.uid for board in b.boards for p in board.placed) == list(parts)
