"""Frozen UID-only micro method from 90499abc; test oracle, never production."""
from typing import Optional
from inrp.mcts import (OmnipotentDecoder, MicroDecodeResult, _board_state_hash,
                       best_local_placement, apply_local_blueprint)
from inrp.packer import Board


class BlueprintReferenceDecoder(OmnipotentDecoder):
    def _micro_decode(self, board: Board, part) -> Optional[MicroDecodeResult]:
        key = self._micro_key(board, part)
        if key in self.micro_cache:
            self.micro_hits += 1
            return self.micro_cache[key]
        self.micro_misses += 1
        bp = best_local_placement(board, part, self.allow_rot, place_mode=self.place_mode)
        if bp is None:
            self.micro_cache[key] = None
            return None
        board_after = apply_local_blueprint(board, part, bp)
        out = MicroDecodeResult(
            blueprint=bp,
            board_term_after=self._board_term(board_after),
            board_hash_after=_board_state_hash(board_after, self.sig_nd),
        )
        self.micro_cache[key] = out
        return out
