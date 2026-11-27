"""One fixed CLASS06 diagnostic, reading the original archive by fingerprint."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import zipfile

ARCHIVE_SHA256 = '9f5ca94d6a9ee5c9273302b8fbddf028c3f6a1d595e80622aafe79ec55589f9c'
MEMBER = 'CLASS/CLASS/cl_06_100_02.ins2D'
MEMBER_SHA256 = 'c7b4c78d57526f89146bfee7acbf40e335ff331ef81db1b06063c4df89bb9379'


def parse_member(raw):
    try:
        rows = [list(map(int, line.split())) for line in raw.decode().splitlines() if line.strip()]
        if len(rows) < 3 or len(rows[0]) != 1 or len(rows[1]) != 2:
            raise ValueError('invalid header')
        count = rows[0][0]
        width, height = rows[1]
        items = rows[2:]
        if count <= 0 or min(width, height) <= 0 or len(items) != count:
            raise ValueError('invalid dimensions or count')
        if any(len(item) != 6 or item[3:] != [1, 1, 0] for item in items):
            raise ValueError('unsupported quantity or orientation flags')
        shapes = {uid: (w, h) for uid, w, h, *_ in items}
        if len(shapes) != count or set(shapes) != set(range(1, count + 1)):
            raise ValueError('invalid identity inventory')
        if any(not (0 < w <= width and 0 < h <= height) for w, h in shapes.values()):
            raise ValueError('invalid native size')
        return width, height, shapes
    except (UnicodeError, IndexError, TypeError) as exc:
        raise ValueError('invalid CLASS member') from exc


def run(archive, out):
    started = time.perf_counter()
    archive, out = Path(archive), Path(out)
    if out.exists():
        raise FileExistsError('refusing to overwrite a diagnostic run: ' + str(out))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError('original archive fingerprint mismatch')
    with zipfile.ZipFile(archive) as source:
        raw = source.read(MEMBER)
    if hashlib.sha256(raw).hexdigest() != MEMBER_SHA256:
        raise ValueError('original member fingerprint mismatch')
    width, height, shapes = parse_member(raw)
    if (width, height, len(shapes)) != (300, 300, 100):
        raise ValueError('unexpected CLASS06 geometry')
    from model import solve_reference
    from verify_witness import verify_witness
    import ortools

    out.mkdir(parents=True)
    inputs = {'case_id': 'cl_06_100_02', 'archive_sha256': digest,
              'original_member': MEMBER, 'member_sha256': MEMBER_SHA256,
              'stock': [width, height], 'items': len(shapes),
              'native_fixed_orientation': True, 'trim': 0, 'kerf': 0,
              'target_boards': 3, 'solver_time_limit_s': 60, 'workers': 8,
              'seed': 20261126, 'hint': False, 'objective': False,
              'python': sys.version, 'ortools': ortools.__version__,
              'platform': platform.platform(), 'machine': platform.machine()}
    (out / 'inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
    result = solve_reference(shapes, width, height, 3, time_limit_s=60,
                             workers=8, seed=20261126, artifact_dir=out)
    if result['status'] in ('OPTIMAL', 'FEASIBLE'):
        used = verify_witness(shapes, width, height, 3, result['placements'])
        result['verified_boards_used'] = used
        result['target_feasibility'] = 'FEASIBLE_WITH_VERIFIED_WITNESS'
    elif result['status'] == 'INFEASIBLE':
        result['target_feasibility'] = 'PROVEN_INFEASIBLE_BY_CP_SAT'
    elif result['status'] == 'UNKNOWN':
        result['target_feasibility'] = 'UNKNOWN'
    else:
        result['target_feasibility'] = 'INVALID_DIAGNOSTIC'
    result['process_work_s'] = time.perf_counter() - started
    result['real_case_solver_calls'] = 1
    result['production_decoder_calls'] = 0
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('placements', 'response_statistics')}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.archive, args.out)
