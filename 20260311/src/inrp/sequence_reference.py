"""Opt-in, metered ASH/SINGLE/PAIR development reference; no solver registry change."""

from __future__ import annotations

import hashlib
import time

from .ash_sequence import ASHSequencePolicy
from .budget import BudgetExpired
from .early_prefix_control import PROFILE, _TwoCompletionSolver
from .handoff import _build_solver_config, _build_solver_input, _validate_request
from .mcts import RecedingHorizonMCTS
from .ordering import short_edge_sequence
from .validate import validate_solution


def _digest(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()


def _entry_state(solver):
    decoder = solver.decoder
    return {
        "initial_sequence": list(solver.global_best_sequence),
        "initial_score": list(solver.global_best_score),
        "cache_keys": {name: _digest(tuple(sorted(getattr(decoder, name))))
                       for name in ("board_term_cache", "micro_cache", "decode_cache")},
        "cache_sizes": {name: len(getattr(decoder, name))
                        for name in ("board_term_cache", "micro_cache", "decode_cache")},
        "cache_counters": [decoder.micro_hits, decoder.micro_misses,
                           decoder.decode_hits, decoder.decode_misses],
    }


def _prefix_stream(solver, arm, h):
    if arm == "SINGLE":
        order = sorted(solver.part_ids, key=lambda uid: (
            solver._part_order_bonus(uid, stage="early"), solver.static_action_score[uid], uid), reverse=True)
        for uid in order:
            yield {"prefix": [uid]}, (uid,) + tuple(x for x in h if x != uid)
    else:
        for u in solver.part_ids:
            for v in solver.part_ids:
                if u != v:
                    yield {"prefix": [u, v]}, (u, v) + tuple(x for x in h if x not in (u, v))


def run_reference(request, budget, arm, *, seed, max_evaluations, emit,
                  proposal_limit=100000, entry_capture=None):
    """Emit complete receipts under one deadline; count both initializer calls.

    `emit` serialization/write costs are charged. Full offline source auditing
    is separate and must never call the decoder. `entry_capture` can attach a
    stronger cache-state audit without affecting candidate selection.
    """
    if arm not in ("SINGLE", "PAIR", "ASH"):
        raise ValueError("unsupported reference arm")
    if type(max_evaluations) is not int or max_evaluations < 2:
        raise ValueError("full replay cap must include two initializer starts")
    if type(proposal_limit) is not int or proposal_limit < 1:
        raise ValueError("proposal cap must be a positive integer")
    if request.solver_variant != PROFILE or request.seed != 1000:
        raise ValueError("reference requires the common profile and initializer seed1000")
    budget.require_open("reference_request")
    _validate_request(request)
    parts, _ = _build_solver_input(request)
    parts, cfg = _build_solver_config(request, parts)
    cfg.FORMAL_TOTAL_DEADLINE = budget.deadline
    cfg.FORMAL_CLOSING_RESERVE_S = budget.closing_reserve_s
    solver = RecedingHorizonMCTS(parts, cfg, 1000)
    solver._configure_solve_deadlines()
    stats = dict(initial_starts=0, initial_completed=0, candidate_starts=0,
                 candidate_completed=0, proposals=0, duplicates=0,
                 interrupted_generation=0, interrupted_replays=0,
                 generation_filter_s=0., replay_s=0., receipt_s=0., trace_write_s=0.)
    known = {}
    initial = []
    replay = solver.decoder.replay_order

    def write(record):
        budget.require_open("reference_trace")
        started = time.perf_counter()
        emit(record)
        stats["trace_write_s"] += time.perf_counter() - started
        budget.require_open("reference_trace")

    def capture(sequence, snapshot):
        started = time.perf_counter()
        receipt = _TwoCompletionSolver._export_capture(solver, (tuple(sequence), snapshot))
        stats["receipt_s"] += time.perf_counter() - started
        return receipt

    def initial_replay(order, **kwargs):
        stats["initial_starts"] += 1
        started = time.perf_counter()
        snapshot, states = replay(order, **kwargs)
        stats["replay_s"] += time.perf_counter() - started
        stats["initial_completed"] += 1
        receipt = capture(order, snapshot)
        known[tuple(order)] = receipt["panels"]
        initial.append(receipt)
        write({"type": "initial", "index": stats["initial_completed"], "receipt": receipt})
        return snapshot, states

    # Remove the temporary bound-method wrapper even on failure: no callback
    # cycle is kept alive through search or process shutdown.
    solver.decoder.replay_order = initial_replay
    try:
        solver._prime_common_initial_solution()
    finally:
        del solver.decoder.replay_order
    entry = entry_capture(solver) if entry_capture else _entry_state(solver)
    write({"type": "entry", "state": entry})
    best_initial = max(initial, key=lambda row: row["score"])
    policy = ASHSequencePolicy(best_initial["sequence"], best_initial["panels"], seed) if arm == "ASH" else None
    stream = None if policy else iter(_prefix_stream(solver, arm, short_edge_sequence(parts)))
    stop = "evaluation_quota"

    def check():
        solver.decoder._require_before(budget.search_deadline, "reference_search")

    while stats["initial_starts"] + stats["candidate_starts"] < max_evaluations:
        if stats["proposals"] >= proposal_limit:
            stop = "proposal_quota"
            break
        try:
            check()
        except BudgetExpired:
            stop = "search_deadline"
            break
        when = time.perf_counter()
        try:
            if policy:
                parameters = {"alpha": policy.alpha, "base_sha256": _digest(policy.base_sequence)}
                sequence = policy.propose(check)
            else:
                parameters, sequence = next(stream)
            duplicate = sequence in known
        except StopIteration:
            stop = "candidate_exhausted"
            break
        except BudgetExpired:
            stats["interrupted_generation"] += 1
            stop = "search_deadline"
            break
        finally:
            stats["generation_filter_s"] += time.perf_counter() - when
        stats["proposals"] += 1
        row = {"type": "proposal", "proposal_index": stats["proposals"],
               "parameters": parameters, "sequence": list(sequence)}
        if duplicate:
            stats["duplicates"] += 1
            if policy:
                policy.observe(known[sequence])
            row.update(status="duplicate", known_panels=known[sequence])
            write(row)
            continue
        try:
            check()
        except BudgetExpired:
            row["status"] = "not_started"
            write(row)
            stop = "search_deadline"
            break
        stats["candidate_starts"] += 1
        row.update(index=stats["candidate_starts"], elapsed_start_s=budget.elapsed_s())
        when = time.perf_counter()
        try:
            snapshot, _ = replay(sequence, deadline=budget.search_deadline,
                                 deadline_phase="reference_replay")
        except BudgetExpired:
            row["replay_s"] = time.perf_counter() - when
            stats["interrupted_replays"] += 1
            row["status"] = "timed_out"
            stop = "search_deadline"
        except Exception as error:
            row.update(replay_s=time.perf_counter() - when, status="failed", error=type(error).__name__)
            try:
                write(row)
            except Exception as trace_error:
                error.reference_trace_error = repr(trace_error)
            raise
        else:
            row["replay_s"] = time.perf_counter() - when
            stats["candidate_completed"] += 1
            receipt = capture(sequence, snapshot)
            known[sequence] = receipt["panels"]
            if policy:
                policy.observe(receipt["panels"])
            solver._update_global_best(sequence, snapshot, source="sequence_reference", record_archive=False)
            row.update(status="completed", receipt=receipt)
        finally:
            row["replay_and_capture_s"] = time.perf_counter() - when
            stats["replay_s"] += row["replay_s"]
            row["elapsed_complete_s"] = budget.elapsed_s()
        write(row)
        if row["status"] == "timed_out":
            break
    validate_solution(parts, list(solver.global_best_snapshot.boards), cfg.BOARD_W, cfg.BOARD_H,
                      trim=cfg.TRIM, safe_gap=0., touch_tol=cfg.TOUCH_TOL)
    final = capture(solver.global_best_sequence, solver.global_best_snapshot)
    write({"type": "final", "receipt": final})
    result = {**stats, "arm": arm, "seed": seed, "max_evaluations": max_evaluations,
              "proposal_limit": proposal_limit, "stop_reason": stop, "initial_panels": best_initial["panels"],
              "final_receipt": final, "elapsed_s": budget.elapsed_s(),
              "ash_alpha": policy.alpha if policy else None,
              "ash_random_draws": policy.random_draws if policy else 0}
    write({"type": "end", "summary_scope": "before_end_record_write",
           "summary": {k: v for k, v in result.items() if k != "final_receipt"}})
    budget.require_open("reference_finished")
    return {**result, "trace_write_s": stats["trace_write_s"], "elapsed_s": budget.elapsed_s(),
            "end_record_write_s": stats["trace_write_s"] - result["trace_write_s"]}
