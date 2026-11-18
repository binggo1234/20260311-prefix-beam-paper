"""Frozen eager decode from3db2960; test-only semantic reference."""

import math
from typing import Optional
from inrp.mcts import (OmnipotentDecoder,DecodeResult,LayoutSnapshot,_avg_u_gamma,
    _board_state_hash,_layout_hash_replace,_ordered_hash_replace,_layout_hash_append,_ordered_hash_append)
from inrp.packer import apply_local_blueprint,make_empty_board


class EagerReferenceDecoder(OmnipotentDecoder):
    def decode(
        self,
        snapshot: LayoutSnapshot,
        action: int,
        *,
        deadline: float = math.inf,
        deadline_phase: str = "completion_replay",
    ) -> DecodeResult:
        self._require_before(deadline, deadline_phase)
        decode_key = (snapshot.layout_hash, snapshot.ordered_hash, len(snapshot.boards), int(action))
        cached = self.decode_cache.get(decode_key)
        if cached is not None:
            self.decode_hits += 1
            return cached
        self.decode_misses += 1
        part = self.parts_by_id[int(action)]
        boards = snapshot.boards
        n_before = len(boards)
        best: Optional[DecodeResult] = None

        for idx, board in boards.enumerate():
            self._require_before(deadline, deadline_phase)
            micro = self._micro_decode(board, part)
            if micro is None:
                continue
            bp = micro.blueprint
            board_hash_before = _board_state_hash(board, self.sig_nd)
            board_after = apply_local_blueprint(board, part, bp)
            new_sum = snapshot.sum_u_gamma - self._board_term(board) + micro.board_term_after
            boards_after = boards.set(idx, board_after)
            snapshot_after = LayoutSnapshot(
                boards=boards_after,
                sum_u_gamma=new_sum,
                layout_hash=_layout_hash_replace(
                    snapshot.layout_hash,
                    board_hash_before,
                    micro.board_hash_after,
                ),
                ordered_hash=_ordered_hash_replace(
                    snapshot.ordered_hash,
                    board_hash_before,
                    micro.board_hash_after,
                    self.pos_hash[idx],
                ),
            )
            avg_gamma = _avg_u_gamma(snapshot_after)
            lex = (
                -float(n_before),
                avg_gamma,
                1.0,
                float(bp.cavity_ratio),
                -float(bp.fragmentation_penalty),
                -float(bp.area_fit),
                -float(idx),
            )
            cand = DecodeResult(
                action=int(action),
                snapshot=snapshot_after,
                board_index=int(idx),
                is_new_board=False,
                blueprint=bp,
                lex_score=lex,
                avg_u_gamma=avg_gamma,
            )
            if best is None or cand.lex_score > best.lex_score:
                best = cand

        self._require_before(deadline, deadline_phase)
        empty_board = make_empty_board(
            n_before + 1,
            self.cfg.BOARD_W,
            self.cfg.BOARD_H,
            trim=self.trim,
            safe_gap=self.safe_gap,
            touch_tol=float(getattr(self.cfg, "TOUCH_TOL", 1e-6)),
            place_mode=self.place_mode,
        )
        micro_new = self._micro_decode(empty_board, part)
        if micro_new is None:
            raise RuntimeError(f"Decoder cannot place part uid={part.uid} on a new board.")
        bp_new = micro_new.blueprint
        board_after_new = apply_local_blueprint(empty_board, part, bp_new)
        new_sum = snapshot.sum_u_gamma + micro_new.board_term_after
        boards_after = boards.append(board_after_new)
        snapshot_after = LayoutSnapshot(
            boards=boards_after,
            sum_u_gamma=new_sum,
            layout_hash=_layout_hash_append(snapshot.layout_hash, micro_new.board_hash_after),
            ordered_hash=_ordered_hash_append(snapshot.ordered_hash, micro_new.board_hash_after, self.pos_hash[n_before]),
        )
        avg_gamma = _avg_u_gamma(snapshot_after)
        cand_new = DecodeResult(
            action=int(action),
            snapshot=snapshot_after,
            board_index=int(n_before),
            is_new_board=True,
            blueprint=bp_new,
            lex_score=(
                -float(n_before + 1),
                avg_gamma,
                0.0,
                float(bp_new.cavity_ratio),
                -float(bp_new.fragmentation_penalty),
                -float(bp_new.area_fit),
                -float(n_before + 1),
            ),
            avg_u_gamma=avg_gamma,
        )
        if best is None or cand_new.lex_score > best.lex_score:
            best = cand_new

        if best is None:
            raise RuntimeError(f"Decoder failed to produce action={action}")
        self.decode_cache[decode_key] = best
        return best
