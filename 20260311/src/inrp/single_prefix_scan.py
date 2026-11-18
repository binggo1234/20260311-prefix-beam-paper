"""Opt-in full-order single-prefix scan; not a default solver profile."""

from __future__ import annotations

import time

from .budget import BudgetExpired, TotalBudget
from .early_prefix_control import PROFILE, _digest
from .handoff import (PackingRequest, PackingHandoff, _validate_request,
                      _build_solver_input, _build_solver_config, _convert_result)
from .mcts import RecedingHorizonMCTS, _complete_solution_fingerprints
from .ordering import short_edge_sequence
from .validate import validate_solution


class _SinglePrefixScanSolver(RecedingHorizonMCTS):
    def __init__(self, parts, cfg, seed, limit):
        super().__init__(parts, cfg, seed)
        self.scan_limit = limit
        self.initial_starts = self.initial_completed = 0
        self.completion_starts = self.completion_completed = 0
        self.initial_capture = self.entry_state = None
        self.candidate_order = []
        self.short_edge_order = short_edge_sequence(self.parts)
        self.evaluations = []
        self.scan_stop_reason = None
        replay = self.decoder.replay_order

        def count_initial(order, **kwargs):
            if kwargs.get("deadline_phase") in ("common_initial_greedy", "common_initial_short_edge"):
                self.initial_starts += 1
                result = replay(order, **kwargs)
                self.initial_completed += 1
                return result
            return replay(order, **kwargs)

        self.decoder.replay_order = count_initial

    def _prime_common_initial_solution(self):
        super()._prime_common_initial_solution()
        if self.meta.get("common_initial_complete"):
            self.initial_capture = (tuple(self.global_best_sequence), self.global_best_snapshot)

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
        # Exact existing early _scout_actions ordering, before its pool cap and
        # tail sampling. Retain every UID, including equal-shaped identities.
        self.candidate_order = sorted(self.part_ids, key=lambda uid: (
            self._part_order_bonus(uid, stage="early"), self.static_action_score[uid], uid), reverse=True)
        self.scan_stop_reason = "candidate_exhausted"
        for uid in self.candidate_order:
            if self.scan_limit is not None and self.completion_starts >= self.scan_limit:
                self.scan_stop_reason = "completion_quota"
                break
            if time.perf_counter() >= self.search_deadline:
                self.scan_stop_reason = "search_deadline"
                break
            sequence = (uid,) + tuple(item for item in self.short_edge_order if item != uid)
            self.completion_starts += 1
            self.meta["oracle_completion_mode_evals"] += 1.0
            self.meta["oracle_completion_mode_eval_short_edge"] = float(
                self.meta.get("oracle_completion_mode_eval_short_edge", 0.0)) + 1.0
            row = {"index": self.completion_starts, "prefix_sequence": [uid], "prefix_depth": 1,
                   "mode": "short_edge", "status": "started"}
            self.evaluations.append(row)
            started = time.perf_counter()
            try:
                snapshot, _ = self.decoder.replay_order(sequence, deadline=self.search_deadline,
                                                       deadline_phase="single_prefix_scan_replay")
            except BudgetExpired:
                row["status"] = "timed_out"
                self.meta["oracle_completion_mode_timeouts"] += 1.0
                self.meta["solver_timeout_hit"] = 1.0
                self.scan_stop_reason = "search_deadline"
                break
            except Exception:
                row["status"] = "failed"
                raise
            else:
                self.completion_completed += 1
                self.meta["oracle_completion_mode_completed"] += 1.0
                row.update(status="completed", capture=(sequence, snapshot))
                self._update_global_best(sequence, snapshot, source="single_prefix_scan", record_archive=False)
            finally:
                row["replay_s"] = max(0.0, time.perf_counter() - started)
        if self.scan_limit == 0:
            self.scan_stop_reason = "completion_quota"
        return self._formal_result_from_incumbent()

    def _capture(self, capture):
        if time.perf_counter() >= self.solve_deadline:
            raise BudgetExpired("single_prefix_scan_export")
        sequence, snapshot = capture
        sequence_sha, layout_sha = _complete_solution_fingerprints(sequence, snapshot)
        rows = [{"board_id": int(board.bid), "uid": int(p.uid),
                 "x": p.rect.x, "y": p.rect.y, "w": p.rect.w, "h": p.rect.h,
                 "w0": p.w0, "h0": p.h0, "rotated": bool(p.rotated)}
                for board in snapshot.boards for p in board.placed]
        if time.perf_counter() >= self.solve_deadline:
            raise BudgetExpired("single_prefix_scan_export")
        return {"sequence": list(sequence), "panels": len(snapshot.boards),
                "score": list(self._terminal_score(snapshot)), "sequence_sha256": sequence_sha,
                "layout_sha256": layout_sha, "placements": rows}

    def solve(self):
        try:
            result = super().solve()
            reason = self.scan_stop_reason if self.meta.get("common_initial_complete") else "initial_incomplete"
            evidence = {"schema_version": "inrp-single-prefix-scan-v1", "limit": self.scan_limit,
                        "initial_starts": self.initial_starts, "initial_completed": self.initial_completed,
                        "completion_starts": self.completion_starts, "completion_completed": self.completion_completed,
                        "candidate_order": self.candidate_order, "short_edge_order": list(self.short_edge_order),
                        "stop_reason": reason, "entry_state": self.entry_state, "initial_receipt": None,
                        "final_receipt": self._capture((tuple(result.sequence), self.global_best_snapshot)),
                        "evaluations": [], "export_complete": False}
            if self.initial_capture is not None:
                evidence["initial_receipt"] = self._capture(self.initial_capture)
            for record in self.evaluations:
                row = {key: value for key, value in record.items() if key != "capture"}
                if "capture" in record:
                    row.update(self._capture(record["capture"]))
                evidence["evaluations"].append(row)
            if time.perf_counter() >= self.solve_deadline:
                raise BudgetExpired("single_prefix_scan_export")
            evidence["export_complete"] = True
            self.meta["single_prefix_scan"] = result.meta["single_prefix_scan"] = evidence
            self.meta["result_source"] = result.meta["result_source"] = "single_prefix_scan_incumbent"
            return result
        except Exception as error:
            error.single_prefix_scan_failure = {
                "initial_starts": self.initial_starts, "initial_completed": self.initial_completed,
                "completion_starts": self.completion_starts, "completion_completed": self.completion_completed,
                "phase": getattr(error, "phase", "solve_or_export"), "error_type": type(error).__name__}
            raise


def solve_single_prefix_scan_handoff(request: PackingRequest, budget: TotalBudget,
                                     *, max_evaluations=None) -> PackingHandoff:
    """Scan the uncapped existing entry order under one absolute budget."""
    if max_evaluations is not None and (type(max_evaluations) is not int or max_evaluations < 0):
        raise ValueError("max_evaluations must be a nonnegative integer or None")
    if request.solver_variant != PROFILE:
        raise ValueError("single-prefix scan requires the short-completion-on profile")
    budget.require_open("single_prefix_scan_request")
    _validate_request(request)
    parts, by_uid = _build_solver_input(request)
    processed, cfg = _build_solver_config(request, parts)
    if not (cfg.PREFIX_COMMON_INIT_ABLATION and cfg.PREFIX_ABLATION_SEARCH_ENABLE
            and cfg.PREFIX_BEAM_ENABLE and not cfg.DISABLE_GEOM_PRIOR and cfg.MACRO_STAGE_EARLY_RATIO > 0):
        raise ValueError("unsupported configuration for the single-prefix scan")
    cfg.FORMAL_TOTAL_DEADLINE = budget.deadline
    cfg.FORMAL_CLOSING_RESERVE_S = budget.closing_reserve_s
    result = _SinglePrefixScanSolver(processed, cfg, int(request.seed), max_evaluations).solve()
    budget.require_open("single_prefix_scan_validation")
    validate_solution(processed, result.boards, cfg.BOARD_W, cfg.BOARD_H,
                      trim=cfg.TRIM, safe_gap=0.0, touch_tol=cfg.TOUCH_TOL)
    handoff = _convert_result(request, result, by_uid, budget.elapsed_s())
    budget.require_open("single_prefix_scan_handoff_ready")
    return handoff
