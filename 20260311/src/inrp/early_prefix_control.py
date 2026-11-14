"""Opt-in experimental two-completion control; not a default solver profile."""

from __future__ import annotations

import hashlib
import time

from .budget import BudgetExpired, TotalBudget
from .handoff import (PackingRequest, PackingHandoff, _validate_request,
                      _build_solver_input, _build_solver_config, _convert_result)
from .mcts import RecedingHorizonMCTS, _complete_solution_fingerprints
from .validate import validate_solution


ARMS = ("single_reference", "current_search")
PROFILE = "rh_mcts_prefix_short_completion_on"


class _CompletionQuotaReached(Exception):
    """A credit stop is deliberately not a wall-clock BudgetExpired."""


def _digest(value):
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


class _TwoCompletionSolver(RecedingHorizonMCTS):
    def __init__(self, parts, cfg, seed, arm):
        super().__init__(parts, cfg, seed)
        self.control_arm = arm
        self.initial_starts = self.initial_completed = 0
        self.completion_starts = self.completion_completed = 0
        self.initial_capture = None
        self.entry_state = None
        self.evaluation_captures = []
        self.completion_context = None
        self.control_stop_reason = None
        self.latest_evidence_export = None
        replay = self.decoder.replay_order

        def counted_replay(order, **kwargs):
            phase = kwargs.get("deadline_phase")
            if phase in ("common_initial_greedy", "common_initial_short_edge"):
                self.initial_starts += 1
                result = replay(order, **kwargs)
                self.initial_completed += 1
                return result
            if phase != "completion_replay" or self.completion_context is None:
                raise RuntimeError("unmetered replay in two-completion experiment")
            self._require_credit()
            prefix, depth = self.completion_context
            sequence = tuple(prefix) + tuple(int(uid) for uid in order)
            if not self._is_complete_sequence(sequence):
                raise RuntimeError("completion credit must cover a complete requested sequence")
            self.completion_starts += 1
            record = {"index": self.completion_starts, "status": "started",
                      "prefix_sequence": list(prefix), "prefix_depth": depth,
                      "mode": "short_edge"}
            self.evaluation_captures.append(record)
            started = time.perf_counter()
            try:
                snapshot, trace = replay(order, **kwargs)
            except Exception as error:
                record.update(status="failed", error=type(error).__name__)
                raise
            else:
                self.completion_completed += 1
                record.update(status="completed", capture=(sequence, snapshot))
                return snapshot, trace
            finally:
                record["replay_s"] = max(0.0, time.perf_counter() - started)

        self.decoder.replay_order = counted_replay

    def _require_credit(self):
        if self.completion_starts >= 2:
            raise _CompletionQuotaReached()

    def _prime_common_initial_solution(self):
        super()._prime_common_initial_solution()
        if self.meta.get("common_initial_complete"):
            self.initial_capture = (tuple(self.global_best_sequence), self.global_best_snapshot)

    def _select_leaf(self, root):
        self._require_credit()
        return super()._select_leaf(root)

    def _evaluate_prefix_with_baseline(self, prefix_sequence, prefix_snapshot, remaining_mask,
                                       *, prefix_depth=None, receipt_macro_chain=None):
        self._require_credit()
        previous = self.completion_context
        self.completion_context = (tuple(prefix_sequence), int(prefix_depth if prefix_depth is not None else len(prefix_sequence)))
        try:
            return super()._evaluate_prefix_with_baseline(
                prefix_sequence, prefix_snapshot, remaining_mask, prefix_depth=prefix_depth,
                receipt_macro_chain=receipt_macro_chain)
        finally:
            self.completion_context = previous

    def _solve_prefix_beam(self, root):
        self.entry_state = {
            "rng_sha256": _digest(self.rng.getstate()),
            "cache_sizes": {name: len(getattr(self.decoder, name))
                            for name in ("board_term_cache", "micro_cache", "decode_cache")},
            "cache_keys_sha256": {name: _digest(tuple(sorted(getattr(self.decoder, name))))
                                  for name in ("board_term_cache", "micro_cache", "decode_cache")},
            "counters": {name: getattr(self.decoder, name)
                         for name in ("micro_hits", "micro_misses", "decode_hits", "decode_misses")},
        }
        try:
            if self.control_arm == "current_search":
                return super()._solve_prefix_beam(root)
            entries = self._single_macro_actions(root.remaining_mask, 2)
            for _, sequence, _ in entries:
                if time.perf_counter() >= self.search_deadline:
                    raise BudgetExpired("completion_replay")
                applied = self._apply_sequence(root.snapshot, sequence, key=int(sequence[0]))
                remaining = int(root.remaining_mask)
                for uid in sequence:
                    remaining &= ~int(self.part_bit[int(uid)])
                self._evaluate_prefix_with_baseline(sequence, applied.snapshot, remaining, prefix_depth=1)
            self.control_stop_reason = "completion_quota" if self.completion_starts == 2 else "reference_exhausted"
            return self._formal_result_from_incumbent()
        except _CompletionQuotaReached:
            self.control_stop_reason = "completion_quota"
            return self._formal_result_from_incumbent()

    def _export_capture(self, capture):
        if time.perf_counter() >= self.solve_deadline:
            raise BudgetExpired("two_completion_evidence_export")
        sequence, snapshot = capture
        sequence_sha, layout_sha = _complete_solution_fingerprints(sequence, snapshot)
        rows = [{"board_id": int(board.bid), "uid": int(p.uid),
                 "x": p.rect.x, "y": p.rect.y, "w": p.rect.w, "h": p.rect.h,
                 "w0": p.w0, "h0": p.h0, "rotated": bool(p.rotated)}
                for board in snapshot.boards for p in board.placed]
        if time.perf_counter() >= self.solve_deadline:
            raise BudgetExpired("two_completion_evidence_export")
        return {"sequence": list(sequence), "panels": len(snapshot.boards),
                "score": list(self._terminal_score(snapshot)), "sequence_sha256": sequence_sha,
                "layout_sha256": layout_sha, "placements": rows}

    def _export_evidence(self, reason):
        evidence = {"schema_version": "inrp-two-completion-control-v1", "arm": self.control_arm,
                    "limit": 2, "initial_starts": self.initial_starts, "initial_completed": self.initial_completed,
                    "completion_starts": self.completion_starts, "completion_completed": self.completion_completed,
                    "stop_reason": reason, "entry_state": self.entry_state,
                    "initial_receipt": None, "final_receipt": None,
                    "evaluations": [], "export_complete": False}
        self.latest_evidence_export = evidence
        if self.initial_capture is not None:
            evidence["initial_receipt"] = self._export_capture(self.initial_capture)
        if self.global_best_sequence is not None and self.global_best_snapshot is not None:
            evidence["final_receipt"] = self._export_capture((tuple(self.global_best_sequence), self.global_best_snapshot))
        for record in self.evaluation_captures:
            row = {key: value for key, value in record.items() if key != "capture"}
            if "capture" in record:
                row.update(self._export_capture(record["capture"]))
            evidence["evaluations"].append(row)
        if self.meta["oracle_completion_mode_evals"] != self.completion_starts:
            raise RuntimeError("completion-start counters disagree")
        if self.meta["oracle_completion_mode_completed"] != self.completion_completed:
            raise RuntimeError("completion-success counters disagree")
        if time.perf_counter() >= self.solve_deadline:
            raise BudgetExpired("two_completion_evidence_export")
        evidence["export_complete"] = True
        return evidence

    def solve(self):
        try:
            result = super().solve()
            if not self.meta.get("common_initial_complete"):
                reason = "initial_incomplete"
            elif self.meta.get("solver_timeout_hit"):
                reason = "search_deadline"
            elif self.completion_starts == self.completion_completed == 2:
                reason = "completion_quota"
            else:
                reason = self.control_stop_reason or "natural_return"
            evidence = self._export_evidence(reason)
            self.meta["two_completion_control"] = result.meta["two_completion_control"] = evidence
            if reason == "completion_quota":
                self.meta["result_source"] = result.meta["result_source"] = "two_completion_quota_incumbent"
            return result
        except Exception as error:
            failure = {"schema_version": "inrp-two-completion-failure-v1", "arm": self.control_arm,
                       "phase": getattr(error, "phase", "solve_or_export"), "error_type": type(error).__name__,
                       "initial_starts": self.initial_starts, "initial_completed": self.initial_completed,
                       "completion_starts": self.completion_starts, "completion_completed": self.completion_completed,
                       "raw_receipts_status": "unavailable_deadline"}
            # Preserve only exports made while the original absolute deadline is
            # open. No packing/replay is performed in this failure path.
            if time.perf_counter() < self.solve_deadline:
                try:
                    self._export_evidence("failed")
                except Exception as export_error:
                    failure["export_error"] = type(export_error).__name__
            if self.latest_evidence_export is not None:
                failure["evidence"] = self.latest_evidence_export
                failure["raw_receipts_status"] = "exported" if self.latest_evidence_export["export_complete"] else "partial_export_before_deadline"
            error.two_completion_failure = failure
            raise


def solve_two_completion_handoff(request: PackingRequest, budget: TotalBudget, arm: str) -> PackingHandoff:
    """Run one experimental arm; input handling and output writing share budget."""
    if arm not in ARMS or request.solver_variant != PROFILE:
        raise ValueError("two-completion control requires a supported arm and short-completion-on profile")
    budget.require_open("two_completion_request")
    _validate_request(request)
    parts, by_uid = _build_solver_input(request)
    processed, cfg = _build_solver_config(request, parts)
    if not (cfg.PREFIX_COMMON_INIT_ABLATION and cfg.PREFIX_ABLATION_SEARCH_ENABLE
            and cfg.PREFIX_BEAM_STAGED_ENABLE and cfg.PREFIX_BEAM_COMPLETION_POLICY == "deterministic"
            and tuple(cfg.PREFIX_BEAM_ORACLE_MODES) == ("short_edge", "long_edge", "strip_bias")
            and cfg.PREFIX_BEAM_STAGED_PROMOTE_EVERY == 4):
        raise ValueError("unsupported configuration for the two-completion experiment")
    cfg.FORMAL_TOTAL_DEADLINE = budget.deadline
    cfg.FORMAL_CLOSING_RESERVE_S = budget.closing_reserve_s
    solver = _TwoCompletionSolver(processed, cfg, int(request.seed), arm)
    result = solver.solve()
    budget.require_open("two_completion_validation")
    validate_solution(processed, result.boards, cfg.BOARD_W, cfg.BOARD_H,
                      trim=cfg.TRIM, safe_gap=0.0, touch_tol=cfg.TOUCH_TOL)
    handoff = _convert_result(request, result, by_uid, budget.elapsed_s())
    budget.require_open("two_completion_handoff_ready")
    return handoff
