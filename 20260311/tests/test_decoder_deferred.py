from dataclasses import asdict
from itertools import permutations
import random

import pytest

from decoder_eager_reference import EagerReferenceDecoder
from inrp.budget import BudgetExpired
from inrp.cfg import CFG
from inrp.dataio import Part
import inrp.mcts as mcts
from inrp.packer import board_signature, make_empty_board


def cfg(mode="maxrects_baf", gamma=2.0):
    return CFG(BOARD_W=50, BOARD_H=40, TRIM=1, TOOL_D=0, BINARY_SPACING=False,
               SAFE_GAP=0, BASELINE_PLACE_MODE=mode, UTIL_GAMMA=gamma)


def fingerprint(result):
    s=result.snapshot
    return (tuple(board_signature(b) for b in s.boards),s.sum_u_gamma,s.layout_hash,s.ordered_hash,
            result.action,result.board_index,result.is_new_board,asdict(result.blueprint),result.lex_score,result.avg_u_gamma)


def test_only_winning_candidate_materializes_a_successor(monkeypatch):
    part=Part(1,"p",10,10,10,10);decoder=mcts.OmnipotentDecoder({1:part},cfg())
    boards=[make_empty_board(i,50,40,trim=1) for i in (1,2)]
    snapshot=mcts.LayoutSnapshot(mcts.PersistentBoards.empty().append(boards[0]).append(boards[1]),0.,(0,0),0)
    # Prewarm identical micro transitions so this measures final successor work,
    # not the separate micro-cache computation that is intentionally retained.
    for board in [*boards,make_empty_board(3,50,40,trim=1)]:decoder._micro_decode(board,part)
    calls=[];apply=mcts.apply_local_blueprint
    def record(board,*args):
        calls.append(board.bid)
        return apply(board,*args)
    monkeypatch.setattr(mcts,"apply_local_blueprint",record)
    result=decoder.decode(snapshot,1)
    assert result.board_index==0 and not result.is_new_board
    assert calls==[1],"losing existing/new-board candidates must not build snapshots"


@pytest.mark.parametrize("mode",["maxrects_baf","maxrects_bssf","blf"])
@pytest.mark.parametrize("gamma",[1.3,2.0])
def test_every_step_and_cache_match_frozen_eager_decoder(mode,gamma):
    rng=random.Random(20261119)
    parts=[Part(i,str(i),w,h,w-1,h-1,allowed_rotations_deg=rotations) for i,w,h,rotations in
           [(1,19,13,(0,90)),(2,19,13,(0,90)),(3,28,18,(0,)),(4,18,28,(90,)),
            (5,23,23,(0,90)),(6,10,14,(0,90)),(7,29,17,(0,90)),(8,18,12,(0,90))]]
    by_id={p.uid:p for p in parts};old=EagerReferenceDecoder(by_id,cfg(mode,gamma));new=mcts.OmnipotentDecoder(by_id,cfg(mode,gamma))
    for _ in range(10):
        order=list(by_id);rng.shuffle(order)
        a,b=mcts._empty_snapshot(),mcts._empty_snapshot()
        for uid in order:
            left,right=old.decode(a,uid),new.decode(b,uid)
            assert fingerprint(left)==fingerprint(right)
            a,b=left.snapshot,right.snapshot
        for name in ("micro_cache","board_term_cache"):
            assert getattr(old,name)==getattr(new,name)
        assert old.decode_cache.keys()==new.decode_cache.keys()
        assert (old.micro_hits,old.micro_misses,old.decode_hits,old.decode_misses)==(new.micro_hits,new.micro_misses,new.decode_hits,new.decode_misses)
        # Revisit a completed state/action to exercise the complete cache path.
        assert fingerprint(old.decode(mcts._empty_snapshot(),order[0]))==fingerprint(new.decode(mcts._empty_snapshot(),order[0]))


def test_all_duplicate_shape_permutations_keep_real_identities():
    parts={uid:Part(uid,str(uid),20,15,20,15) for uid in range(1,5)}
    old,new=EagerReferenceDecoder(parts,cfg()),mcts.OmnipotentDecoder(parts,cfg())
    for order in permutations(parts):
        a,_=old.replay_order(order);b,_=new.replay_order(order)
        assert mcts._complete_solution_fingerprints(order,a)==mcts._complete_solution_fingerprints(order,b)
        assert sorted(p.uid for board in b.boards for p in board.placed)==list(parts)


def test_new_board_candidate_error_is_not_skipped_when_existing_board_fits():
    part=Part(1,"large",60,20,60,20,allowed_rotations_deg=(0,))
    for cls in (EagerReferenceDecoder,mcts.OmnipotentDecoder):
        decoder=cls({1:part},cfg())
        board=make_empty_board(1,100,100)
        snapshot=mcts.LayoutSnapshot(mcts.PersistentBoards.empty().append(board),0.,(0,0),0)
        with pytest.raises(RuntimeError,match="cannot place part"):
            decoder.decode(snapshot,1)


def test_deadline_hit_before_cache_lookup_still_rejects_cached_action(monkeypatch):
    part=Part(1,"p",10,10,10,10)
    for cls in (EagerReferenceDecoder,mcts.OmnipotentDecoder):
        decoder=cls({1:part},cfg());decoder.decode(mcts._empty_snapshot(),1)
        with monkeypatch.context() as changes:
            changes.setattr(mcts.time,"perf_counter",lambda:10.)
            with pytest.raises(BudgetExpired):decoder.decode(mcts._empty_snapshot(),1,deadline=10.)
