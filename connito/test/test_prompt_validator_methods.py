import hashlib
import json
from types import SimpleNamespace

import pytest
import requests
from pydantic import ValidationError
from test_prompt_validator_lesson_length import _target_head
from test_prompt_validator_scores import BASE, CYCLE_ID, _scores, config, fake  # noqa: F401 (fixtures)

from connito.prompt_validator import methods, run, snapshot
from connito.prompt_validator.config import PromptValidatorCfg, ScoringMethodCfg
from connito.prompt_validator.methods import REGISTRY, Scored
from connito.prompt_validator.methods import reference_pairwise as rp
from connito.prompt_validator.methods.reference_pairwise import ReferenceFileError, answer_side, expected_level
from connito.prompt_validator.stage_io import read_rows, stage_path, write_rows

REF = "the reference course"
TASK = "t_ref"  # a task with no format or length rule: only its method differs


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _refs_file(tmp_path, rows) -> tuple:
    path = tmp_path / "refs" / "references.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _pairwise_cfg(path, sha, weight=1.0, orders="both") -> ScoringMethodCfg:
    return ScoringMethodCfg(method="reference_pairwise", settings={
        "references_path": str(path), "references_sha256": sha, "reference_weight": weight, "orders": orders})


class PairJev:
    """Answers pairwise questions: `levels[(answer, answer_first)]` is the level for that answer in that order. Any
    other call goes to the plain fake."""

    def __init__(self, plain, levels):
        self.plain, self.levels, self.bodies = plain, levels, []

    def post(self, url, json=None, headers=None, timeout=None):
        if not any("course_a" in q["instructions"] for q in json["questions"].values()):
            return self.plain.post(url, json=json, headers=headers, timeout=timeout)
        self.bodies.append(json)
        answers = {}
        for qid, q in json["questions"].items():
            a, b = q["instructions"]["course_a"], q["instructions"]["course_b"]
            assert q["instructions"]["question"] == rp.PAIR_QUESTION and q["criteria"] == rp.PAIR_LEVELS
            level = self.levels[(a, True)] if b == REF else self.levels[(b, False)]
            answers[qid] = {"score": float(level), "confidence": 0.9,
                            "probabilities": {str(i): float(i == level) for i in range(7)}}
        payload = {"model": "jev-1.14.0", "answers": answers, "usage": {"input_tokens": 1000}}
        return SimpleNamespace(status_code=200, text="", json=lambda: payload)


def _setup(config, tmp_path, monkeypatch, fake, levels, refs_for=("p1",), tasks=None, weight=1.0,  # noqa: F811
           orders="both"):
    """Prompts get content_hash and a task (default: p1 TASK, p2 none); refs_for's get a reference; TASK is scored by
    reference_pairwise; the snapshot is re-hashed."""
    pv = config.prompt_validator
    tasks = tasks or {"p1": TASK}
    snapshot_dir = stage_path(pv.runs_dir, CYCLE_ID, "snapshot").parent
    prompts = [{**p, "content_hash": _sha(p["prompt"]), "customer_task_id": tasks.get(p["prompt_id"])}
               for p in read_rows(snapshot_dir / "prompts.jsonl")]
    write_rows(snapshot_dir / "prompts.jsonl", prompts)
    path, sha = _refs_file(tmp_path, [{"prompt_id": f"v6-{p['prompt_id']}", "content_hash": p["content_hash"],
                                       "text": REF, "ok": True} for p in prompts if p["prompt_id"] in refs_for])
    pv.scoring_methods = {TASK: _pairwise_cfg(path, sha, weight, orders)}
    write_rows(snapshot_dir / "snapshot.jsonl", [{**BASE, "rules_sha256": snapshot.rules_sha256(pv, BASE)}])
    pair = PairJev(fake, levels)
    monkeypatch.setattr(requests, "post", pair.post)
    return pair


LEVELS = {("4", True): 1, ("4", False): 4, ("2", True): 3, ("2", False): 3}


# The registry and per-task selection

def test_the_registry_holds_today_s_methods_behind_one_interface():
    assert sorted(REGISTRY) == ["eps_error", "plain_levels", "reference_pairwise"]
    for name, method in REGISTRY.items():
        assert method.name == name and method.RULES and callable(method.score) and callable(method.check)
        assert set(method.SETTINGS_NOT_HASHED) <= set(method.Settings.model_fields)


def test_an_unknown_method_or_a_bad_setting_fails_the_config_load():
    with pytest.raises(ValidationError, match="unknown scoring method"):
        ScoringMethodCfg(method="brier")
    with pytest.raises(ValidationError):
        ScoringMethodCfg(method="plain_levels", settings={"weight": 1})
    with pytest.raises(ValidationError):
        ScoringMethodCfg(method="reference_pairwise", settings={"references_path": "refs.jsonl",
                                                                "references_sha256": "a" * 64})
    with pytest.raises(ValidationError):
        ScoringMethodCfg(method="reference_pairwise", settings={"references_path": "/r.jsonl",
                                                                "references_sha256": "A" * 64})
    with pytest.raises(ValidationError):
        _pairwise_cfg("/r.jsonl", "a" * 64, weight=1.5)


def test_a_new_method_registered_is_selected_for_its_task_only(config, fake, monkeypatch):  # noqa: F811
    class Constant:
        name, RULES, DETAILS = "constant", "every answer 3.5", ("note",)
        Settings = REGISTRY["plain_levels"].Settings
        SETTINGS_NOT_HASHED = frozenset()

        def check(self, settings):
            pass

        def score(self, items, ctx):
            return [Scored(score=3.5, method=self.name, details={"note": ctx.task}) for _ in items]

    monkeypatch.setitem(REGISTRY, "constant", Constant())
    pv = config.prompt_validator
    snapshot_dir = stage_path(pv.runs_dir, CYCLE_ID, "snapshot").parent
    write_rows(snapshot_dir / "prompts.jsonl", [{**p, "customer_task_id": TASK if p["prompt_id"] == "p1" else None}
                                                for p in read_rows(snapshot_dir / "prompts.jsonl")])
    pv.scoring_methods = {TASK: ScoringMethodCfg(method="constant")}
    write_rows(snapshot_dir / "snapshot.jsonl", [{**BASE, "rules_sha256": snapshot.rules_sha256(pv, BASE)}])

    run.run_stages(config, CYCLE_ID, "scores")

    rows = {(r["uid"], r["prompt_id"]): r for r in _scores(config)}
    assert (rows[(1, "p1")]["method"], rows[(1, "p1")]["expected_score"], rows[(1, "p1")]["note"]) == \
           ("constant", 0.7, TASK)
    assert (rows[(2, "p2")]["method"], rows[(2, "p2")]["expected_score"], rows[(2, "p2")]["note"]) == \
           ("plain_levels", 1.0, None)  # p2's task is not listed: plain_levels
    assert len(fake.bodies) == 1  # only p2 went to Jev


def test_plain_levels_selected_by_name_scores_as_the_default_does(config, fake):  # noqa: F811
    run.run_stages(config, CYCLE_ID, "scores")
    default = _scores(config)
    pv = config.prompt_validator
    pv.scoring_methods = {"none_such": ScoringMethodCfg(method="plain_levels")}  # listed: rows gain `method`
    snapshot_dir = stage_path(pv.runs_dir, CYCLE_ID, "snapshot").parent
    write_rows(snapshot_dir / "prompts.jsonl", [{**p, "customer_task_id": "none_such"}
                                                for p in read_rows(snapshot_dir / "prompts.jsonl")])
    write_rows(snapshot_dir / "snapshot.jsonl", [{**BASE, "rules_sha256": snapshot.rules_sha256(pv, BASE)}])

    run.run_stages(config, CYCLE_ID, "scores")

    selected = _scores(config)
    assert [{k: v for k, v in r.items() if k != "method"} for r in selected] == default
    assert {r["method"] for r in selected} == {"plain_levels"}
    assert "method" not in default[0]  # with no task listed, the rows are as before judging methods


# reference_pairwise

def test_each_order_maps_the_answer_s_side_onto_0_to_5_from_the_whole_distribution():
    cases = [(3, True, 2.5), (3, False, 2.5),  # about equal: 2.5 of 5
             (0, True, 5.0), (6, True, 0.0),  # the answer as course_a: level 0 is the answer much better
             (6, False, 5.0), (0, False, 0.0),  # the answer as course_b: level 6 is
             (1, True, 25 / 6), (4, False, 20 / 6)]
    for expected, answer_first, side in cases:
        assert answer_side(expected, answer_first) == pytest.approx(side)
    assert expected_level([0, 0, 0, 0.5, 0.5, 0, 0]) == pytest.approx(3.5)


@pytest.mark.parametrize("orders, four, two, course_a_is_ref", [
    # p1's "4": as course_a clearly better (level 1: 25/6), as course_b slightly better (level 4: 20/6). "2": equal.
    ("both", (4.1667, 3.3333, 3.75), (2.5, 2.5, 2.5), [{False}, {True}]),
    ("a", (4.1667, None, 4.1667), (2.5, None, 2.5), [{False}]),
    ("b", (None, 3.3333, 3.3333), (None, 2.5, 2.5), [{True}]),
])
def test_each_order_asked_is_its_own_call_and_the_comparison_is_their_mean(config, fake, tmp_path,  # noqa: F811
                                                                           monkeypatch, orders, four, two,
                                                                           course_a_is_ref):
    pair = _setup(config, tmp_path, monkeypatch, fake, LEVELS, orders=orders)

    run.run_stages(config, CYCLE_ID, "scores")

    rows = {(r["uid"], r["prompt_id"]): r for r in _scores(config)}
    got = {u: (rows[(u, "p1")]["pairwise_ab"], rows[(u, "p1")]["pairwise_ba"], rows[(u, "p1")]["comparison_score"])
           for u in (1, 2)}
    assert got == {1: four, 2: two}
    first = rows[(1, "p1")]
    assert (first["method"], first["reference_id"]) == ("reference_pairwise", "v6-p1")
    assert first["expected_score"] == first["score"] == round(four[2] / 5, 4)  # at weight 1, the comparison
    assert rows[(3, "p1")] | {"uid": 1, "hotkey": "hk1"} == first  # uid 3 is uid 1's model
    assert (rows[(2, "p1")]["rank"], first["rank"]) == (1, 0)
    assert list(first)[-5:] == ["method", "pairwise_ab", "pairwise_ba", "comparison_score", "reference_id"]
    assert all(b["state"] == {"prompt": "prompt one"} for b in pair.bodies)
    assert sorted(({q["instructions"]["course_a"] == REF for q in b["questions"].values()} for b in pair.bodies),
                  key=str) == course_a_is_ref
    assert not any("prompt one" == b["state"].get("prompt") for b in fake.bodies)  # p1 never graded plain


def test_a_reference_weight_below_1_blends_in_plain_levels(config, fake, tmp_path, monkeypatch):  # noqa: F811
    _setup(config, tmp_path, monkeypatch, fake, LEVELS, weight=0.5)

    run.run_stages(config, CYCLE_ID, "scores")

    rows = {(r["uid"], r["prompt_id"]): r for r in _scores(config)}
    assert rows[(1, "p1")]["expected_score"] == 0.775  # (3.75 / 5 + 0.8) / 2
    assert rows[(2, "p1")]["expected_score"] == 0.45  # (0.5 + 0.4) / 2


def test_a_prompt_with_no_reference_falls_back_to_plain_levels_and_says_so(config, fake, tmp_path,  # noqa: F811
                                                                           monkeypatch):
    # Both prompts are TASK's; only p1 has a reference.
    pair = _setup(config, tmp_path, monkeypatch, fake, LEVELS, tasks={"p1": TASK, "p2": TASK})

    run.run_stages(config, CYCLE_ID, "scores")

    rows = {(r["uid"], r["prompt_id"]): r for r in _scores(config)}
    p2 = rows[(2, "p2")]
    assert (p2["method"], p2["expected_score"], p2["reference_id"], p2["comparison_score"]) == \
           ("plain_levels", 1.0, None, None)
    assert rows[(1, "p1")]["method"] == "reference_pairwise"
    assert all(q["instructions"]["course_a"] in ("4", "2", REF) for b in pair.bodies for q in b["questions"].values())


def test_a_lesson_answer_scores_the_comparison_times_its_length_multiplier(config, fake, tmp_path,  # noqa: F811
                                                                         monkeypatch):
    pv = config.prompt_validator
    pv.lesson_length = pv.lesson_length.model_copy(update={"enabled": True, "aggregate": "mean"})
    lesson = "exp_lesson_residual"
    snapshot_dir = stage_path(pv.runs_dir, CYCLE_ID, "snapshot").parent
    write_rows(snapshot_dir / "prompts.jsonl", [{**p, "prompt": _target_head(200)} if p["prompt_id"] == "p1" else p
                                                for p in read_rows(snapshot_dir / "prompts.jsonl")])
    # h1's lessons average 245 words, past 200 +- 20: x0. h2's average 200: x1.
    courses = {h: json.dumps({"lessons": [{"lesson_number": i, "title": "t", "body": " ".join(["w"] * n),
                                           "source_ids": []} for i, n in enumerate(bodies, 1)]})
               for h, bodies in {"h1": [245, 245, 245], "h2": [200, 190, 210]}.items()}
    answers_path = stage_path(pv.runs_dir, CYCLE_ID, "answers")
    write_rows(answers_path, [{**a, "text": courses[a["model_hash"]]} if a["prompt_id"] == "p1" else a
                              for a in read_rows(answers_path)])
    _setup(config, tmp_path, monkeypatch, fake, {(courses["h1"], True): 3, (courses["h1"], False): 3,
                                                 (courses["h2"], True): 1, (courses["h2"], False): 5},
           tasks={"p1": lesson})
    pv.scoring_methods = {lesson: pv.scoring_methods.pop(TASK)}
    write_rows(snapshot_dir / "snapshot.jsonl", [{**BASE, "rules_sha256": snapshot.rules_sha256(pv, BASE)}])

    run.run_stages(config, CYCLE_ID, "scores")

    rows = {(r["uid"], r["prompt_id"]): r for r in _scores(config)}
    h1, h2 = rows[(1, "p1")], rows[(2, "p1")]
    assert (h1["comparison_score"], h1["jev_expected_score"], h1["length_multiplier"], h1["expected_score"]) == \
           (2.5, 0.5, 0.0, 0.0)
    assert (h2["comparison_score"], h2["jev_expected_score"], h2["length_multiplier"], h2["expected_score"]) == \
           (4.1667, 0.8333, 1.0, 0.8333)
    assert (h1["word_target"], h1["word_counts"]) == (200, [245, 245, 245])


def test_a_missing_or_changed_references_file_stops_the_snapshot_and_scores(config, fake, tmp_path,  # noqa: F811
                                                                        monkeypatch):
    pair = _setup(config, tmp_path, monkeypatch, fake, {})
    path = tmp_path / "refs" / "references.jsonl"
    path.write_text(path.read_text() + "\n")  # one byte more: another file

    with pytest.raises(ReferenceFileError, match="not the pinned"):
        run.run_stages(config, CYCLE_ID, "scores")
    path.unlink()
    with pytest.raises(ReferenceFileError, match="is missing"):
        run.run_stages(config, CYCLE_ID, "scores")

    monkeypatch.setattr(snapshot, "sync_owner_task", lambda cfg: None)
    with pytest.raises(ReferenceFileError, match="is missing"):
        snapshot.run_snapshot(config, CYCLE_ID)

    assert fake.bodies == [] and pair.bodies == []
    assert not stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "scores").exists()
    assert issubclass(ReferenceFileError, methods.MethodStop)  # the loop stops on it (test_prompt_validator_loop)


def test_load_references_keys_by_content_hash_and_refuses_two_for_one_prompt(tmp_path):
    path, sha = _refs_file(tmp_path, [{"prompt_id": "v6-0", "content_hash": "a" * 64, "text": "x", "ok": True},
                                      {"prompt_id": "v6-1", "content_hash": "b" * 64, "text": "y", "ok": False}])
    assert rp.load_references(_pairwise_cfg(path, sha).parsed()) == {"a" * 64: {"reference_id": "v6-0", "text": "x"}}
    path, sha = _refs_file(tmp_path, [{"prompt_id": "v6-0", "content_hash": "a" * 64, "text": "x"}] * 2)
    with pytest.raises(ReferenceFileError, match="two references"):
        rp.load_references(_pairwise_cfg(path, sha).parsed())


# The rules hash

def test_the_rules_hash_holds_each_listed_task_s_method_settings_and_rules_but_not_the_path(tmp_path, monkeypatch):
    def rules(**scoring_methods):
        pv = PromptValidatorCfg(runs_dir=tmp_path / "runs", prompts_path=tmp_path / "p.jsonl")
        pv.scoring_methods = scoring_methods
        return snapshot.rules_sha256(pv, BASE)

    on = _pairwise_cfg("/r/refs.jsonl", "a" * 64)
    before = rules(exp_lesson_residual=on)
    assert rules() != before  # with no task listed, the hash is as before judging methods
    assert before == rules(exp_lesson_residual=_pairwise_cfg("/elsewhere/refs.jsonl", "a" * 64))
    assert before != rules(exp_lesson_residual=_pairwise_cfg("/r/refs.jsonl", "b" * 64))
    assert before != rules(exp_lesson_residual=_pairwise_cfg("/r/refs.jsonl", "a" * 64, weight=0.5))
    assert before != rules(exp_lesson_residual=ScoringMethodCfg(method="plain_levels"))
    assert before != rules(other_task=on)
    assert len({rules(exp_lesson_residual=_pairwise_cfg("/r/refs.jsonl", "a" * 64, orders=o)) for o in "ab"} | {before}) == 3
    monkeypatch.setattr(REGISTRY["reference_pairwise"], "RULES", rp.ReferencePairwise.RULES + " changed")
    assert rules(exp_lesson_residual=on) != before
