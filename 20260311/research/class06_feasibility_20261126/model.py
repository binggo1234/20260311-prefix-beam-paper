"""Standard exact feasibility model; deliberately separate from inrp decoding."""
import hashlib
import json
import math
from pathlib import Path
import time

from ortools.sat.python import cp_model


def build_model(shapes, width, height, boards):
    if (not shapes or any(type(v) is not int or v <= 0 for v in (width, height, boards))
            or any(type(uid) is not int or uid <= 0 for uid in shapes)):
        raise ValueError('positive integer board geometry and nonempty UID inventory required')
    if any(type(v) is not int or v <= 0 for wh in shapes.values() for v in wh):
        raise ValueError('positive integer native dimensions required')
    if any(len(wh) != 2 or wh[0] > width or wh[1] > height for wh in shapes.values()):
        raise ValueError('each item must fit in its fixed native orientation')
    model = cp_model.CpModel()
    positions = {}
    x_intervals, y_intervals = [], []
    for uid, (w, h) in sorted(shapes.items()):
        domain = cp_model.Domain.from_intervals(
            [[b * width, b * width + width - w] for b in range(boards)])
        x = model.new_int_var_from_domain(domain, f'x_{uid}')
        y = model.new_int_var(0, height - h, f'y_{uid}')
        positions[uid] = (x, y)
        x_intervals.append(model.new_fixed_size_interval_var(x, w, f'ix_{uid}'))
        y_intervals.append(model.new_fixed_size_interval_var(y, h, f'iy_{uid}'))
    model.add_no_overlap_2d(x_intervals, y_intervals)
    error = model.validate()
    if error:
        raise ValueError('invalid CP-SAT geometry model: ' + error)
    return model, positions


def solve_reference(shapes, width, height, boards, *, time_limit_s, workers, seed,
                    artifact_dir=None):
    if (not math.isfinite(time_limit_s) or time_limit_s < 0 or
            type(workers) is not int or workers <= 0 or type(seed) is not int):
        raise ValueError('finite nonnegative time limit, positive workers and integer seed required')
    started = time.perf_counter()
    model, positions = build_model(shapes, width, height, boards)
    built_s = time.perf_counter() - started
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    log = None
    if artifact_dir is not None:
        folder = Path(artifact_dir)
        folder.mkdir(exist_ok=True)
        solver.parameters.log_search_progress = True
        solver.parameters.log_to_stdout = False
        if not model.export_to_file(str(folder / 'model.pbtxt')):
            raise RuntimeError('model export failed before solve')
        (folder / 'parameters.txt').write_text(str(solver.parameters))
        before = {'model_sha256': hashlib.sha256((folder / 'model.pbtxt').read_bytes()).hexdigest(),
                  'parameters_sha256': hashlib.sha256((folder / 'parameters.txt').read_bytes()).hexdigest(),
                  'model_statistics': model.model_stats(), 'model_validated': True,
                  'build_s': built_s}
        (folder / 'pre_solve.json').write_text(json.dumps(before, indent=2) + '\n')
        log = (folder / 'solver.log').open('x')
        solver.log_callback = log.write
    solve_started = time.perf_counter()
    try:
        status = solver.solve(model)
    finally:
        if log is not None:
            log.close()
    solve_elapsed = time.perf_counter() - solve_started
    placements = None
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        placements = []
        for uid, (x, y) in positions.items():
            global_x = solver.value(x)
            placements.append({'uid': uid, 'board_id': global_x // width + 1,
                               'x': global_x % width, 'y': solver.value(y), 'rotated': False})
    result = {'status': solver.status_name(status), 'placements': placements,
              'build_s': built_s, 'solve_elapsed_s': solve_elapsed,
              'solver_wall_s': solver.wall_time, 'build_and_solve_s': time.perf_counter() - started,
              'response_statistics': solver.response_stats()}
    if artifact_dir is not None:
        (folder / 'response.pbtxt').write_text(str(solver.response_proto))
    return result
