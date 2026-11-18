#!/usr/bin/env python3
"""Audit a completed batch and emit the Chapter 2/3/4 shared layout manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inrp.formal_audit import audit_batch
from inrp.formal_batch import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=6)
    parser.add_argument("--expected-count", type=int, default=110)
    args = parser.parse_args()
    audit, layouts = audit_batch(
        args.manifest,
        args.out_root,
        expected_attempt=args.attempt,
        expected_count=args.expected_count,
        data_root=args.data_root,
        authorization_path=args.authorization,
    )
    layout_path = args.out_root / "chapter2_3_4_shared_layout_manifest.csv"
    with layout_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(layouts[0]))
        writer.writeheader()
        writer.writerows(layouts)
    audit["shared_layout_manifest"] = str(layout_path.resolve())
    audit["shared_layout_manifest_sha256"] = sha256_file(layout_path)
    audit_path = args.out_root / "formal_completion_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
