"""Canonical constructive orders shared by reference and search solvers."""

from typing import Sequence, Tuple

from .dataio import Part


def baseline_sequence(parts: Sequence[Part]) -> Tuple[int, ...]:
    ranked = sorted(
        parts,
        key=lambda part: (
            float(part.w0) * float(part.h0),
            max(float(part.w0), float(part.h0)) / max(1e-9, min(float(part.w0), float(part.h0))),
            float(max(part.w0, part.h0)),
            -int(part.uid),
        ),
        reverse=True,
    )
    return tuple(int(part.uid) for part in ranked)
