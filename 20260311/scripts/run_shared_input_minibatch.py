#!/usr/bin/env python3
"""Run the six-piece shared-input integration using explicitly pinned sources."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CUT_ROOT = Path("/Users/binggo/Desktop/graduate/cut_eval")
DEFAULT_JOINT_ROOT = Path("/Users/binggo/Desktop/graduate/joint_nesting_cuteval_hde_validation")


def _pin_imports(cut_root: Path, joint_root: Path) -> None:
    source_roots = (PROJECT_ROOT / "src", cut_root / "src", joint_root / "src")
    for source_root in reversed(source_roots):
        source_text = str(source_root.resolve())
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
    for name, root in (("cut_eval", cut_root), ("joint_nesting_cuteval", joint_root)):
        module = importlib.import_module(name)
        module_path = Path(str(module.__file__)).resolve()
        if root.resolve() / "src" not in module_path.parents:
            raise AssertionError(f"{name} imported from unexpected path: {module_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--cut-root", type=Path, default=DEFAULT_CUT_ROOT)
    parser.add_argument("--joint-root", type=Path, default=DEFAULT_JOINT_ROOT)
    args = parser.parse_args()
    _pin_imports(args.cut_root, args.joint_root)
    from thesis_input.minibatch import run_minibatch

    result = run_minibatch(
        args.bundle_path,
        args.output_dir,
        cut_root=args.cut_root,
        joint_root=args.joint_root,
    )
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
