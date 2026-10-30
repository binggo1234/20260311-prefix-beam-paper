"""Type fairness changes unexpanded-action choice, not the candidate universe."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inrp.budget import TotalBudget
from inrp.cfg import CFG
from inrp.dataio import Part
from inrp.handoff import PackingPart, PackingRequest, solve_packing_handoff_with_budget
from inrp.mcts import RecedingHorizonMCTS, SearchNode, _empty_snapshot
from inrp.runner import _case_stats, _variant_cfg


def parts():
    return [Part(i, str(i), w, h, w, h) for i, (w, h) in enumerate(((4, 7), (6, 8), (3, 6), (2, 7), (8, 3), (5, 5)), 1)]


def solver(enabled=True):
    cfg = CFG()
    cfg.BOARD_W = cfg.BOARD_H = 10.0
    cfg.TRIM = cfg.GAP = cfg.TOOL_D = cfg.SAFE_GAP = 0.0
    cfg.BINARY_SPACING = False
    cfg.PREFIX_BEAM_ENABLE = True
    cfg.MACRO_ACTION_ENABLE = True
    cfg.PREFIX_BEAM_TYPE_FAIR_EXPANSION = enabled
    return RecedingHorizonMCTS(parts(), cfg, 17)


def fake_tree(subject, monkeypatch, allowed_count=6):
    node = SearchNode(_empty_snapshot(), subject.full_remaining_mask)
    node.candidate_actions = [1, 2, 4, 3, 5, 6]
    node.priors = {1: .9, 2: .8, 4: .7, 3: .1, 5: .09, 6: .05}
    node.action_meta = {uid: SimpleNamespace(kind=kind) for uid, kind in (
        (1, "pattern"), (2, "pattern"), (3, "single"), (4, "anchor"), (5, "single"), (6, "tail"))}
    monkeypatch.setattr(subject, "_ensure_candidate_actions", lambda root, limit: None)
    monkeypatch.setattr(subject, "_pw_limit", lambda root: allowed_count)

    def expand(parent, action):
        child = SearchNode(_empty_snapshot(), 0, depth=parent.depth+1, parent=parent,
                           action_from_parent=action, visits=1)
        parent.children[action] = child
        return child

    monkeypatch.setattr(subject, "_expand", expand)
    return node


def test_type_fair_order_preserves_within_kind_prior(monkeypatch):
    subject = solver()
    node = fake_tree(subject, monkeypatch)
    before = (list(node.candidate_actions), dict(node.priors), subject.rng.getstate())
    chosen = [subject._select_leaf(node)[-1].action_from_parent for _ in range(6)]
    assert chosen == [3, 1, 4, 6, 5, 2]
    assert before == (node.candidate_actions, node.priors, subject.rng.getstate())
    assert subject.meta["prefix_type_fair_expansions"] == 6
    assert subject.meta["prefix_type_fair_first_kinds"] == ["single", "pattern", "anchor", "tail", "single", "pattern"]


def test_disabled_policy_preserves_original_prior_order(monkeypatch):
    subject = solver(False)
    node = fake_tree(subject, monkeypatch)
    assert [subject._select_leaf(node)[-1].action_from_parent for _ in range(6)] == [1, 2, 4, 3, 5, 6]
    assert subject.meta["prefix_type_fair_expansions"] == 0


def test_fairness_never_reaches_outside_progressive_widening_set(monkeypatch):
    subject = solver()
    node = fake_tree(subject, monkeypatch, allowed_count=3)
    assert [subject._select_leaf(node)[-1].action_from_parent for _ in range(3)] == [1, 4, 2]
    assert set(node.children) == {1, 2, 4}


def test_puct_path_is_unchanged_after_allowed_actions_are_expanded(monkeypatch):
    subject = solver()
    node = fake_tree(subject, monkeypatch)
    for _ in range(6):
        subject._select_leaf(node)
    expected = max(node.candidate_actions, key=lambda action: subject._puct_score(node, action, sum(node.priors.values())))
    count = subject.meta["prefix_type_fair_expansions"]
    assert subject._select_leaf(node)[-1].action_from_parent == expected
    assert subject.meta["puct_selections"] == 1
    assert subject.meta["prefix_type_fair_expansions"] == count


def test_expanded_children_outside_allowed_set_do_not_bias_local_fairness(monkeypatch):
    subject = solver()
    node = fake_tree(subject, monkeypatch)
    node.action_meta[99] = SimpleNamespace(kind="single")
    node.children[99] = SearchNode(_empty_snapshot(), 0)
    assert subject._select_leaf(node)[-1].action_from_parent == 3


def test_single_available_kind_keeps_prior_order(monkeypatch):
    subject = solver()
    node = fake_tree(subject, monkeypatch)
    for action in node.action_meta.values():
        action.kind = "pattern"
    assert [subject._select_leaf(node)[-1].action_from_parent for _ in range(6)] == [1, 2, 4, 3, 5, 6]


def test_type_evidence_is_bounded(monkeypatch):
    subject = solver()
    for _ in range(40):
        subject._select_leaf(fake_tree(subject, monkeypatch))
    assert subject.meta["prefix_type_fair_expansions"] == 40
    assert len(subject.meta["prefix_type_fair_first_kinds"]) == 32


@pytest.mark.parametrize("n_parts", [6, 739, 2416])
def test_variant_changes_only_expansion_policy_and_identity(n_parts):
    cfg = CFG()
    stats = _case_stats(parts(), cfg)
    stats["n_parts"] = n_parts
    control = _variant_cfg(cfg, "rh_mcts_prefix_beam_staged", stats)
    fair = _variant_cfg(cfg, "rh_mcts_prefix_beam_staged_type_fair", stats)
    ignored = {"PREFIX_BEAM_TYPE_FAIR_EXPANSION", "SOLVER_ROLE", "SOLVER_BUNDLE", "VARIANTS"}
    assert {k:v for k,v in vars(control).items() if k not in ignored} == {k:v for k,v in vars(fair).items() if k not in ignored}
    assert not control.PREFIX_BEAM_TYPE_FAIR_EXPANSION
    assert fair.PREFIX_BEAM_TYPE_FAIR_EXPANSION
    assert fair.PREFIX_BEAM_STAGED_ENABLE and not fair.MACRO_COMPLEMENT_ENABLE
    solvers = [RecedingHorizonMCTS(parts(), config, 41) for config in (control, fair)]
    generated = [subject._generate_macro_actions(_empty_snapshot(), subject.full_remaining_mask, 8) for subject in solvers]
    assert generated[0][0:2] == generated[1][0:2]
    assert {key:value.snapshot.layout_hash for key,value in generated[0][2].items()} == {key:value.snapshot.layout_hash for key,value in generated[1][2].items()}


@pytest.mark.parametrize("gate", ["PREFIX_BEAM_ENABLE", "MACRO_ACTION_ENABLE"])
def test_nonprefix_or_nonmacro_ignores_type_flag(gate):
    cfg = CFG()
    cfg.PREFIX_BEAM_ENABLE = cfg.MACRO_ACTION_ENABLE = cfg.PREFIX_BEAM_TYPE_FAIR_EXPANSION = True
    setattr(cfg, gate, False)
    subject = RecedingHorizonMCTS(parts(), cfg, 1)
    assert not subject.prefix_type_fair_expansion_enable


def test_no_prior_ablation_is_not_silently_changed():
    cfg = CFG()
    cfg.PREFIX_BEAM_ENABLE = cfg.MACRO_ACTION_ENABLE = cfg.PREFIX_BEAM_TYPE_FAIR_EXPANSION = True
    cfg.DISABLE_GEOM_PRIOR = True
    with pytest.raises(ValueError, match="type-fair"):
        RecedingHorizonMCTS(parts(), cfg, 1)


def test_real_budgeted_handoff_retains_feasible_greedy_floor():
    request = PackingRequest(case_id="type-fair", parts=(PackingPart("a", "a", 54, 94), PackingPart("b", "b", 34, 94)),
                             board_width_mm=110, board_height_mm=110,
                             solver_variant="rh_mcts_prefix_beam_staged_type_fair", time_limit_s=1.0)
    result = solve_packing_handoff_with_budget(request, TotalBudget(2.0, .5))
    meta = result.provenance["solver_meta"]
    assert result.panel_count == 1 and result.panel_count <= meta["greedy_floor_panels"]
    assert meta["prefix_type_fair_expansion_enabled"]
    assert meta["prefix_type_fair_expansions"] > 0
    assert {part.part_id for part in result.placements} == {"a", "b"}
