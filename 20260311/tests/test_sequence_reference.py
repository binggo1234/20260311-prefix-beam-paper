import importlib
import importlib.util
import time

import pytest

from inrp.budget import BudgetExpired, TotalBudget
from inrp.handoff import PackingPart, PackingRequest


def module():
    assert importlib.util.find_spec('inrp.sequence_reference') is not None, 'reference runner is missing'
    return importlib.import_module('inrp.sequence_reference')


def request():
    return PackingRequest(case_id='reference-toy', parts=tuple(
        PackingPart(str(i), str(i), w, h) for i, (w, h) in enumerate(
            [(41, 21), (32, 29), (25, 35), (19, 17)], 1)),
        board_width_mm=100, board_height_mm=100, trim_mm=0, tool_diameter_mm=0,
        allow_rotation=False, seed=1000, time_limit_s=10,
        solver_variant='rh_mcts_prefix_short_completion_on')


def run(arm, cap=8, seed=1000, proposal_limit=1000):
    rows = []
    result = module().run_reference(request(), TotalBudget(10, .5), arm,
                                    seed=seed, max_evaluations=cap, emit=rows.append,
                                    proposal_limit=proposal_limit)
    return result, rows


def test_common_initial_layout_and_cache_entry_match_across_policies():
    traces = [run(arm, cap=2)[1] for arm in ('SINGLE', 'PAIR', 'ASH')]
    for rows in traces[1:]:
        assert [r['receipt'] for r in rows if r['type'] == 'initial'] == [
            r['receipt'] for r in traces[0] if r['type'] == 'initial']
        assert next(r['state'] for r in rows if r['type'] == 'entry') == next(
            r['state'] for r in traces[0] if r['type'] == 'entry')


@pytest.mark.parametrize('arm', ['PAIR', 'ASH'])
def test_quota_includes_initial_calls_and_only_fresh_sequences_are_decoded(arm):
    result, rows = run(arm)
    initial = [r for r in rows if r['type'] == 'initial']
    proposals = [r for r in rows if r['type'] == 'proposal']
    completed = [r for r in proposals if r['status'] == 'completed']
    assert result['initial_starts'] == result['initial_completed'] == len(initial) == 2
    assert result['candidate_starts'] == result['candidate_completed'] == len(completed) == 6
    assert result['stop_reason'] == 'evaluation_quota'
    seen = {tuple(r['receipt']['sequence']) for r in initial}
    for row in proposals:
        seq = tuple(row['sequence'])
        assert sorted(seq) == [1, 2, 3, 4]
        assert (row['status'] == 'duplicate') == (seq in seen)
        if row['status'] == 'completed':
            seen.add(seq)
            assert row['receipt']['sequence'] == list(seq)
    assert result['final_receipt']['score'] == max(
        r['receipt']['score'] for r in initial + completed)


def test_single_exhausts_with_unused_quota_and_pair_order_is_lexicographic():
    single, rows = run('SINGLE', cap=100)
    assert single['stop_reason'] == 'candidate_exhausted'
    assert single['proposals'] == 4 and single['candidate_completed'] <= 4
    _, rows = run('PAIR', cap=6)
    pairs = [r['parameters']['prefix'] for r in rows if r['type'] == 'proposal']
    assert pairs[:4] == [[1, 2], [1, 3], [1, 4], [2, 1]]


def test_duplicate_ash_proposal_updates_policy_without_spending_credit():
    result, rows = run('ASH', seed=17, proposal_limit=1)
    proposal = next(r for r in rows if r['type'] == 'proposal')
    # Seed17's first four draws all exceed .1: unchanged base sequence.
    assert proposal['status'] == 'duplicate'
    assert result['candidate_starts'] == result['candidate_completed'] == 0
    assert result['ash_alpha'] == pytest.approx(.102)
    assert result['stop_reason'] == 'proposal_quota'


def test_generation_interruption_preserves_initial_and_spends_no_replay_credit(monkeypatch):
    from inrp.ash_sequence import ASHSequencePolicy
    def expired(self, check=None):
        raise BudgetExpired('reference_generation')
    monkeypatch.setattr(ASHSequencePolicy, 'propose', expired)
    result, rows = run('ASH')
    assert result['stop_reason'] == 'search_deadline'
    assert result['interrupted_generation'] == 1
    assert result['candidate_starts'] == result['candidate_completed'] == 0
    assert result['final_receipt'] == max(
        (r['receipt'] for r in rows if r['type'] == 'initial'), key=lambda r: r['score'])


def test_replay_interruption_is_charged_and_keeps_complete_incumbent(monkeypatch):
    from inrp.mcts import OmnipotentDecoder
    replay = OmnipotentDecoder.replay_order
    def expire(self, order, **kwargs):
        if kwargs.get('deadline_phase') == 'reference_replay':
            raise BudgetExpired('reference_replay')
        return replay(self, order, **kwargs)
    monkeypatch.setattr(OmnipotentDecoder, 'replay_order', expire)
    result, rows = run('PAIR')
    assert result['candidate_starts'] == result['interrupted_replays'] == 1
    assert result['candidate_completed'] == 0 and result['stop_reason'] == 'search_deadline'
    assert next(r for r in rows if r['type'] == 'proposal')['status'] == 'timed_out'


def test_unexpected_decoder_error_is_preserved_and_recorded(monkeypatch):
    from inrp.mcts import OmnipotentDecoder
    replay = OmnipotentDecoder.replay_order
    def fail(self, order, **kwargs):
        if kwargs.get('deadline_phase') == 'reference_replay':
            raise ValueError('decoder failure sentinel')
        return replay(self, order, **kwargs)
    monkeypatch.setattr(OmnipotentDecoder, 'replay_order', fail)
    rows = []
    with pytest.raises(ValueError, match='decoder failure sentinel'):
        module().run_reference(request(), TotalBudget(10, .5), 'PAIR', seed=1000,
                               max_evaluations=8, emit=rows.append)
    assert rows[-1]['status'] == 'failed' and rows[-1]['error'] == 'ValueError'


def test_initializer_does_not_leave_a_bound_method_cycle():
    rows = []
    def entry(solver):
        assert 'replay_order' not in solver.decoder.__dict__
        return {'checked': True}
    module().run_reference(request(), TotalBudget(10, .5), 'PAIR', seed=1000,
                           max_evaluations=2, emit=rows.append, entry_capture=entry)


def test_end_write_is_included_in_returned_total_and_trace_cost():
    rows = []
    def emit(row):
        rows.append(row)
        if row['type'] == 'end':
            time.sleep(.02)
    result = module().run_reference(request(), TotalBudget(10, .5), 'SINGLE', seed=1000,
                                    max_evaluations=2, emit=emit)
    assert result['elapsed_s'] >= .02 and result['trace_write_s'] >= .02
    assert result['end_record_write_s'] >= .02
    assert rows[-1]['summary_scope'] == 'before_end_record_write'
    assert result['elapsed_s'] >= rows[-1]['summary']['elapsed_s'] + .02


def test_failed_trace_write_cannot_mask_original_decoder_error(monkeypatch):
    from inrp.mcts import OmnipotentDecoder
    replay = OmnipotentDecoder.replay_order
    def fail(self, order, **kwargs):
        if kwargs.get('deadline_phase') == 'reference_replay':
            raise ValueError('original decoder failure')
        return replay(self, order, **kwargs)
    def emit(row):
        if row.get('status') == 'failed':
            raise BudgetExpired('failed_trace_write')
    monkeypatch.setattr(OmnipotentDecoder, 'replay_order', fail)
    with pytest.raises(ValueError, match='original decoder failure') as caught:
        module().run_reference(request(), TotalBudget(10, .5), 'PAIR', seed=1000,
                               max_evaluations=8, emit=emit)
    assert 'failed_trace_write' in caught.value.reference_trace_error


@pytest.mark.parametrize('arm,cap', [('unknown', 8), ('SINGLE', 1), ('ASH', True)])
def test_invalid_reference_parameters(arm, cap):
    with pytest.raises(ValueError):
        run(arm, cap)
