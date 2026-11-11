"""Deadline-safe whole-order references, independent of restricted-prefix search."""

from __future__ import annotations

import math
import random
import time
from typing import Sequence, Tuple

from .budget import BudgetExpired
from .mcts import OmnipotentDecoder, SearchResult, _global_lex_score, _scalar_reward
from .ordering import baseline_sequence


def _sequence_proposal(sequence: Tuple[int, ...], rng: random.Random) -> Tuple[int, ...]:
    """Sample a global permutation move; no fixed prefix is protected."""
    if len(sequence) < 2:
        return sequence
    result = list(sequence)
    left, right = sorted(rng.sample(range(len(result)), 2))
    move = rng.randrange(4)
    if move == 0:
        result[left], result[right] = result[right], result[left]
    elif move == 1:
        source, target = (left, right) if rng.random() < 0.5 else (right, left)
        result.insert(target, result.pop(source))
    elif move == 2:
        result[left:right + 1] = reversed(result[left:right + 1])
    else:
        block = result[left:right + 1]
        del result[left:right + 1]
        target = rng.randrange(len(result) + 1)
        result[target:target] = block
    return tuple(result)


def _constructive_orders(parts: Sequence) -> Tuple[Tuple[int, ...], ...]:
    keys = (
        lambda p: max(p.w0, p.h0), lambda p: min(p.w0, p.h0),
        lambda p: p.w0 + p.h0, lambda p: max(p.w0, p.h0) / min(p.w0, p.h0),
        lambda p: p.w0, lambda p: p.h0,
    )
    orders = []
    seen = {baseline_sequence(parts)}
    for key in keys:
        order = tuple(p.uid for p in sorted(
            parts, key=lambda p: (key(p), p.w0 * p.h0, -p.uid), reverse=True))
        if order not in seen:
            orders.append(order)
            seen.add(order)
    return tuple(orders)


def solve_evaluation_baseline(parts: Sequence, cfg, seed: int) -> SearchResult:
    """Retain the best complete layout under the same decoder and total deadline."""
    kind = str(getattr(cfg, "SOLVER_KIND", ""))
    if kind not in {"baseline_order_multistart", "baseline_sequence_sa_budgeted"}:
        raise ValueError(f"unsupported evaluation baseline: {kind}")
    started = time.perf_counter()
    local_limit = float(getattr(cfg, "SOLVER_TIME_LIMIT_S", 0.0))
    local_deadline = started + local_limit if math.isfinite(local_limit) and local_limit > 0 else started
    deadline = min(local_deadline, float(getattr(cfg, "FORMAL_TOTAL_DEADLINE", math.inf))
                   - float(getattr(cfg, "FORMAL_CLOSING_RESERVE_S", 0.0)))
    rng = random.Random(seed)
    decoder = OmnipotentDecoder({int(p.uid): p for p in parts}, cfg)
    current_sequence = baseline_sequence(parts)
    current_snapshot, current_snapshots = decoder.replay_order(
        current_sequence, deadline=deadline, deadline_phase="evaluation_greedy_replay")
    best_sequence, best_snapshot = current_sequence, current_snapshot
    best_score = _global_lex_score(best_snapshot)
    greedy_panels = len(best_snapshot.boards)
    events = [{"elapsed_s": time.perf_counter() - started, "panel_count": greedy_panels}]
    attempted = completed = 1
    interrupted = sa_completed = accepted_worse = 0
    orders = _constructive_orders(parts)
    is_sa = kind == "baseline_sequence_sa_budgeted"

    def energy(snapshot):
        count = len(snapshot.boards)
        return -float(count) + 0.01 * snapshot.sum_u_gamma / max(1, count)

    for iteration in range(len(orders) + 6000):
        if time.perf_counter() >= deadline:
            break
        portfolio = iteration < len(orders)
        if portfolio:
            candidate = orders[iteration]
        elif is_sa:
            candidate = _sequence_proposal(current_sequence, rng)
        else:
            candidate = tuple(p.uid for _, p in sorted(
                ((p.w0 * p.h0 * rng.uniform(0.5, 1.5), p) for p in parts),
                key=lambda item: (item[0], -item[1].uid), reverse=True))
        if candidate == current_sequence:
            continue
        first_changed = next(i for i, pair in enumerate(zip(candidate, current_sequence)) if pair[0] != pair[1])
        attempted += 1
        try:
            snapshot, suffix_snapshots = decoder.replay_order(
                candidate[first_changed:], prefix_snapshot=current_snapshots[first_changed],
                deadline=deadline, deadline_phase="evaluation_candidate_replay")
        except BudgetExpired:
            interrupted += 1
            break
        completed += 1
        candidate_snapshots = current_snapshots[:first_changed] + suffix_snapshots
        score = _global_lex_score(snapshot)
        improved = score > best_score
        if improved:
            if len(snapshot.boards) < len(best_snapshot.boards):
                events.append({"elapsed_s": time.perf_counter() - started,
                               "panel_count": len(snapshot.boards)})
            best_sequence, best_snapshot, best_score = candidate, snapshot, score
        accept = improved
        if is_sa and not portfolio:
            temperature = max(0.01, 0.25 * 0.995 ** sa_completed)
            delta = energy(snapshot) - energy(current_snapshot)
            accept = delta >= 0 or rng.random() < math.exp(delta / temperature)
            accepted_worse += int(accept and delta < 0)
            sa_completed += 1
        if accept:
            current_sequence, current_snapshot, current_snapshots = candidate, snapshot, candidate_snapshots

    return SearchResult(
        boards=list(best_snapshot.boards), sequence=best_sequence, score_global=best_score,
        scalar_reward=_scalar_reward(best_snapshot, float(getattr(cfg, "SCALAR_BIG_M", 1000.0))),
        meta={
            "solver_kind": kind, "decoder_place_mode": decoder.place_mode,
            "baseline_attempted_evaluations": attempted,
            "baseline_completed_evaluations": completed,
            "baseline_interrupted_evaluations": interrupted,
            "baseline_sa_completed_proposals": sa_completed,
            "baseline_sa_accepted_worse": accepted_worse,
            "baseline_best_panels": len(best_snapshot.boards), "greedy_floor_panels": greedy_panels,
            "global_best_events": events, "search_deadline": deadline,
            "solver_timeout_hit": time.perf_counter() >= deadline,
            "decode_cache_hits": decoder.decode_hits, "decode_cache_misses": decoder.decode_misses,
            "micro_cache_hits": decoder.micro_hits, "micro_cache_misses": decoder.micro_misses,
        },
    )
