import importlib
from pathlib import Path
from collections import Counter
from itertools import permutations

import pytest


def module():
    assert Path(__file__).with_name('policy.py').exists(), 'bounded feedback policy is missing'
    return importlib.import_module('policy')


def test_lazy_permutation_visits_every_index_once_and_scales_by_draws():
    m=module()
    for seed in range(3):
        assert sorted(list(m.LazyPermutation(9,seed)))==list(range(9))
    sampler=m.LazyPermutation(10**12,1000)
    values=[next(sampler) for _ in range(50)]
    assert len(set(values))==50 and all(0<=v<10**12 for v in values)
    assert len(sampler.swaps)<=50


def test_canonical_keys_are_exact_for_the_whole_short_prefix_family():
    m=module();h=(3,1,4,2)
    groups={}
    for k in range(3):
        for prefix in permutations(h,k):
            sequence=prefix+tuple(u for u in h if u not in prefix)
            key=m.prefix_key(h,prefix)
            assert groups.setdefault(key,sequence)==sequence
    assert m.family_key(h,(2,4,3,1))==(2,4)
    assert m.family_key(h,(2,4,1,3)) is None


def test_stream_starts_pair_work_without_waiting_for_whole_inventory():
    m=module();h=tuple(range(1,101))
    p=m.StreamPolicy(h,'STREAM_OVERLAP',1000,known={h[:2]})
    probes=0
    while True:
        a=p.next_action()
        if a['phase']=='pair':break
        assert a['phase']=='single_probe'
        probes+=1;p.observe(Counter({(2,3):1}),6,probes+2)
    assert probes<=3 and len(p.window)<=16
    assert p.max_candidates_checked<=240


def test_pair_batch_size_and_fifo_do_not_expand_to_all_pairs():
    m=module();h=tuple(range(1,81))
    p=m.StreamPolicy(h,'STREAM_RANDOM',1001,known={h[:2]})
    consecutive=0;count=0
    for evaluation in range(3,150):
        a=p.next_action()
        if a['phase']=='feedback_reuse':continue
        if a['phase']=='single_probe':consecutive=0;count+=1
        else:consecutive+=1;assert consecutive<=2
        p.observe(Counter({(2,3):1}),6,evaluation)
        assert len(p.window)<=16 and p.max_candidates_checked<=240
    assert count>16 and len(p.window)==16


def test_initial_full_result_does_not_supply_single_feedback():
    m=module();h=tuple(range(1,10))
    p=m.StreamPolicy(h,'STREAM_OVERLAP',1000,known=set(permutations(h,2)))
    a=p.next_action()
    assert a['phase']=='single_probe' and not p.single_cache


def test_a_pair_alias_can_supply_future_single_feedback_without_new_decode():
    m=module();h=tuple(range(1,8))
    p=m.StreamPolicy(h,'STREAM_OVERLAP',1000,known=set())
    p.record_target((4,1),Counter({(2,3):2}),12,9)
    assert p.single_cache[4]['source_evaluation']==9
    assert p.single_cache[4]['area']==12
    assert (4,1) in p.seen


def test_uniform_direct_pair_stream_never_materializes_all_pairs():
    m=module();pairs=list(m.direct_pairs((1,2,3,4),1000))
    assert len(pairs)==12 and len(set(pairs))==12
    assert all(u!=v for u,v in pairs)
    huge=m.direct_pairs(tuple(range(2000)),1000)
    first=[next(huge) for _ in range(10)]
    assert len(set(first))==10


def test_pending_result_cannot_be_overwritten_by_another_action():
    m=module();p=m.StreamPolicy(tuple(range(1,8)),'STREAM_OVERLAP',1000,known=set())
    p.next_action()
    with pytest.raises(RuntimeError):p.next_action()


def test_first_pair_bound_even_when_both_full_initial_words_are_in_family():
    m=module();h=(1,2,3,4)
    for extra in permutations(h,2):
        for seed in range(10):
            p=m.StreamPolicy(h,'STREAM_OVERLAP',seed,known={h[:2],extra})
            probes=0
            while True:
                a=p.next_action()
                if a['phase']=='pair':break
                probes+=1;p.observe(Counter({(1,1):1}),1,probes+2)
            assert probes<=3


def test_reuse_event_has_source_and_overlap_really_changes_pair_priority():
    m=module();h=tuple(range(1,8))
    p=m.StreamPolicy(h,'STREAM_OVERLAP',1000,known=set())
    p.record_target((4,1),Counter({(2,3):2}),12,9)
    for evaluation in range(10,40):
        a=p.next_action()
        if a['phase']=='feedback_reuse':
            assert a['prefix']==(4,) and a['source_evaluation']==9
            assert 4 in p.window and p.pending is None
            break
        p.observe(Counter({(2,3):1}),6,evaluation)
    else:raise AssertionError('expected cached single was never visited')
    p=m.StreamPolicy(h,'STREAM_OVERLAP',1000,known=set())
    p._add_window(1,{'counts':Counter({(2,3):1})})
    p._add_window(2,{'counts':Counter({(2,3):1})})
    p._add_window(3,{'counts':Counter({(4,5):1})})
    assert 3 in p._best_pair()
