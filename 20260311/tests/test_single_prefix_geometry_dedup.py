from dataclasses import replace
from itertools import permutations

import pytest

from inrp.budget import BudgetExpired, TotalBudget
from inrp.dataio import Part
from inrp.handoff import PackingPart, PackingRequest
from inrp.mcts import OmnipotentDecoder, _global_lex_score
from inrp.cfg import CFG
from inrp.packer import board_signature
import inrp.single_prefix_scan as scan


def request():
    sizes = [(20, 30), (20, 30), (30, 20), (20, 30), (50, 45), (50, 45)]
    return PackingRequest(case_id="geometry-dedup-toy", parts=tuple(
        PackingPart(str(i), "source-" + str(i), w, h)
        for i, (w, h) in enumerate(sizes, 1)),
        board_width_mm=100, board_height_mm=100, trim_mm=2, tool_diameter_mm=3,
        solver_variant="rh_mcts_prefix_short_completion_on", seed=17, time_limit_s=10)


_DEFAULT = object()


def solve(deduplicate=_DEFAULT, limit=None):
    kwargs = {} if deduplicate is _DEFAULT else {"deduplicate_geometry": deduplicate}
    return scan.solve_single_prefix_scan_handoff(
        request(), TotalBudget(10, .5), max_evaluations=limit, **kwargs)


def evidence(result):
    return result.provenance["solver_meta"]["single_prefix_scan"]


def test_opt_in_skips_exact_words_preserves_real_first_best_and_default():
    ordinary = solve()
    explicit = solve(False)
    dedup = solve(True)
    a, b = evidence(ordinary), evidence(dedup)
    assert ordinary.sequence == explicit.sequence
    assert ordinary.placements == explicit.placements
    assert a["completion_completed"] == len(request().parts)
    assert b["completion_completed"] < a["completion_completed"]
    assert b["geometry_deduplication"] is True
    assert b["completion_completed"] + len(b["geometry_skips"]) == len(b["candidate_order"])
    assert b["candidate_order"] == a["candidate_order"]
    assert b["initial_receipt"] == a["initial_receipt"]
    assert dedup.sequence == ordinary.sequence
    assert dedup.placements == ordinary.placements
    assert dedup.score_global == ordinary.score_global
    assert {p.part_id for p in dedup.placements} == {p.part_id for p in request().parts}
    prior = {r["prefix_sequence"][0]: r for r in a["evaluations"]}
    for skipped in b["geometry_skips"]:
        assert prior[skipped["uid"]]["score"] == prior[skipped["representative_uid"]]["score"]


def test_key_uses_entire_word_and_exact_nominal_envelope_rotation_values():
    assert hasattr(scan, "_geometry_sequence_key"), "exact geometry key is missing"
    parts = {1: Part(1, "a", 23, 33, 20, 30),
             2: Part(2, "b", 23, 33, 20, 30),
             3: Part(3, "c", 33, 23, 30, 20)}
    key = scan._geometry_sequence_key
    assert key((1, 2, 3), parts) == key((2, 1, 3), parts)
    assert key((1, 3, 2), parts) != key((2, 1, 3), parts)
    for changes in ({"w": 23 + 1e-10}, {"w0": 20 + 1e-10},
                    {"h": 33 + 1e-10}, {"h0": 30 + 1e-10},
                    {"allowed_rotations_deg": (90,)}, {"allowed_rotations_deg": (90, 0)}):
        altered = {**parts, 2: replace(parts[2], **changes)}
        assert key((1,), altered) != key((2,), altered)


def test_quota_counts_actual_evaluations_and_zero_keeps_initial():
    zero = evidence(solve(True, 0))
    assert zero["completion_starts"] == 0 and zero["geometry_skips"] == []
    assert zero["initial_receipt"] == zero["final_receipt"]
    limited = evidence(solve(True, 2))
    assert limited["completion_starts"] == limited["completion_completed"] == 2
    assert limited["stop_reason"] == "completion_quota"


@pytest.mark.parametrize("flag", [1, None, "yes"])
def test_non_boolean_opt_in_rejected(flag):
    with pytest.raises(ValueError, match="deduplicate_geometry"):
        solve(flag)


def test_timeout_consumes_attempt_and_keeps_complete_real_incumbent(monkeypatch):
    replay = OmnipotentDecoder.replay_order
    attempts = []

    def interrupt(self, order, **kwargs):
        if kwargs.get("deadline_phase") == "single_prefix_scan_replay":
            attempts.append(tuple(order))
            if len(attempts) == 2:
                raise BudgetExpired("single_prefix_scan_replay")
        return replay(self, order, **kwargs)

    monkeypatch.setattr(OmnipotentDecoder, "replay_order", interrupt)
    result = solve(True)
    e = evidence(result)
    assert e["completion_starts"] == 2 and e["completion_completed"] == 1
    assert e["evaluations"][-1]["status"] == "timed_out"
    assert e["stop_reason"] == "search_deadline"
    assert result.score_global == max(tuple(r["score"]) for r in
                                     [e["initial_receipt"], e["evaluations"][0]])
    assert len(result.placements) == len(request().parts)


def test_failure_is_not_turned_into_a_reusable_representative(monkeypatch):
    replay = OmnipotentDecoder.replay_order

    def broken(self, order, **kwargs):
        if kwargs.get("deadline_phase") == "single_prefix_scan_replay":
            raise ValueError("invalid geometry")
        return replay(self, order, **kwargs)

    monkeypatch.setattr(OmnipotentDecoder, "replay_order", broken)
    with pytest.raises(ValueError, match="invalid geometry") as caught:
        solve(True)
    counts = caught.value.single_prefix_scan_failure
    assert counts["completion_starts"] == 1 and counts["completion_completed"] == 0


def test_equal_geometry_output_keeps_each_actual_source_label():
    req = request()
    req = replace(req, parts=tuple(replace(p, nominal_width_mm=20, nominal_height_mm=30)
                                   for p in req.parts))
    result = scan.solve_single_prefix_scan_handoff(req, TotalBudget(10, .5),
                                                 deduplicate_geometry=True)
    e = evidence(result)
    assert e["completion_completed"] == 1 and len(e["geometry_skips"]) == 5
    assert {p.part_id: p.source_part_id for p in result.placements} == {
        p.part_id: p.source_part_id for p in req.parts}


def normalized(snapshot, sequence):
    position = {uid: i for i, uid in enumerate(sequence)}
    return tuple((b.bid, tuple((position[p.uid], p.rect.x, p.rect.y, p.rect.w,
                 p.rect.h, p.w0, p.h0, p.rotated) for p in b.placed),
                 tuple((r.x, r.y, r.w, r.h) for r in b.free_rects))
                 for b in snapshot.boards)


@pytest.mark.parametrize("rotations", [(0,), (90,), (0, 90)])
def test_all_small_geometry_equal_permutations_match_cold_and_warmed_states(rotations):
    assert hasattr(scan, "_geometry_sequence_key"), "exact geometry key is missing"
    parts = {1: Part(1, "left", 23, 33, 20, 30, allowed_rotations_deg=rotations),
             2: Part(2, "right", 23, 33, 20, 30, allowed_rotations_deg=rotations),
             3: Part(3, "third", 38, 28, 35, 25),
             4: Part(4, "fourth", 38, 28, 35, 25)}
    cfg = CFG(BOARD_W=80, BOARD_H=70, TRIM=2, TOOL_D=3,
              BINARY_SPACING=False, SAFE_GAP=0, ALLOW_ROT=True)
    shared = OmnipotentDecoder(parts, cfg)
    groups = {}
    for sequence in permutations(parts):
        final, states = shared.replay_order(sequence)
        cold, cold_states = OmnipotentDecoder(parts, cfg).replay_order(sequence)
        assert normalized(final, sequence) == normalized(cold, sequence)
        assert [normalized(s, sequence) for s in states] == [normalized(s, sequence) for s in cold_states]
        key = scan._geometry_sequence_key(sequence, parts)
        signature = (tuple(normalized(s, sequence) for s in states), _global_lex_score(final))
        if key in groups:
            assert signature == groups[key]
        else:
            groups[key] = signature
        assert {p.uid for b in final.boards for p in b.placed} == set(parts)
        assert all(p.pid_raw == parts[p.uid].pid_raw for b in final.boards for p in b.placed)
        for state in states:
            hashes = [hash(board_signature(b, nd=shared.sig_nd, include_bid=False)) & ((1 << 64) - 1)
                      for b in state.boards]
            xor = 0
            for value in hashes:
                xor ^= value
            assert state.layout_hash == (sum(hashes) & ((1 << 64) - 1), xor)
            assert state.ordered_hash == sum(value * shared.pos_hash[i]
                                            for i, value in enumerate(hashes)) & ((1 << 64) - 1)
    assert len(groups) == 6
