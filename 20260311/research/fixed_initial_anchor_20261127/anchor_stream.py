"""Reference composition: preserve visits, freeze the selected initial anchor."""


def fixed_initial_stream(original_stream, solver, arm, h):
    if arm != 'SINGLE':
        raise ValueError('this control is defined only for SINGLE')
    anchor = tuple(solver.global_best_sequence)
    if len(anchor) != len(h) or len(set(anchor)) != len(anchor) or set(anchor) != set(h):
        raise ValueError('selected anchor must preserve the complete UID inventory')
    for parameters, _ in original_stream(solver, arm, h):
        uid, = parameters['prefix']
        if uid not in anchor:
            raise ValueError('foreign UID in original proposal')
        yield parameters, (uid,) + tuple(u for u in anchor if u != uid)
