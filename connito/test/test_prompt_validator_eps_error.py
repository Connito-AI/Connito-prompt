import hashlib
import json

import pytest
from pydantic import ValidationError
from test_prompt_validator_scores import BASE, CYCLE_ID, _scores, config, fake  # noqa: F401 (fixtures)

from connito.prompt_validator import methods, run, snapshot
from connito.prompt_validator.config import PromptValidatorCfg, ScoringMethodCfg
from connito.prompt_validator.methods import REGISTRY, Context, Item
from connito.prompt_validator.methods import eps_error as eps
from connito.prompt_validator.methods.eps_error import AnswersFileError, parse_forecast
from connito.prompt_validator.stage_io import read_rows, stage_path, write_rows

TASK = "earnings_forecast"
REASONING = "Deposit costs rose and fee income fell.\n"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _answers_file(tmp_path, rows) -> tuple:
    # Synthetic answers only: the real ones stay validator-side, out of the repository.
    path = tmp_path / "answers" / "answers.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _eps_cfg(path, sha) -> ScoringMethodCfg:
    return ScoringMethodCfg(method="eps_error", settings={"answers_path": str(path), "answers_sha256": sha})


def _ranked(tmp_path, answers, actuals=None):
    """eps_error's Scored for (prompt, model_hash, forecast text) triples; each prompt's actual change from actuals,
    else 0.0."""
    prompts = sorted({p for p, _, _ in answers})
    actuals = actuals or {}
    path, sha = _answers_file(tmp_path, [{"prompt_id": f"ans-{p}", "content_hash": _sha(p),
                                          "eps_pct_change": actuals.get(p, 0.0)} for p in prompts])
    items = [Item({"prompt_id": p, "content_hash": _sha(p)}, text, model_hash) for p, model_hash, text in answers]
    return eps.EpsError().score(items, Context(TASK, _eps_cfg(path, sha).parsed(), None))


def _f(value: float) -> str:
    return REASONING + f"FORECAST: {value:+.1f}"


def _setup(config, tmp_path, texts, actual=-41.22, answer_for=("p1",)):  # noqa: F811
    """p1 and p2 become TASK's prompts; answer_for's get an actual change; texts[model_hash] is that model's answer to
    both; the snapshot is re-hashed."""
    pv = config.prompt_validator
    snapshot_dir = stage_path(pv.runs_dir, CYCLE_ID, "snapshot").parent
    prompts = [{**p, "content_hash": _sha(p["prompt"]), "customer_task_id": TASK}
               for p in read_rows(snapshot_dir / "prompts.jsonl")]
    write_rows(snapshot_dir / "prompts.jsonl", prompts)
    path, sha = _answers_file(tmp_path, [{"prompt_id": f"ans-{p['prompt_id']}", "content_hash": p["content_hash"],
                                          "eps_pct_change": actual} for p in prompts if p["prompt_id"] in answer_for])
    pv.scoring_methods = {TASK: _eps_cfg(path, sha)}
    write_rows(snapshot_dir / "snapshot.jsonl", [{**BASE, "rules_sha256": snapshot.rules_sha256(pv, BASE)}])
    answers_path = stage_path(pv.runs_dir, CYCLE_ID, "answers")
    write_rows(answers_path, [{**a, "text": texts[a["model_hash"]]} for a in read_rows(answers_path)])
    return path


@pytest.mark.parametrize("text, forecast", [
    (REASONING + "FORECAST: +15.0", 15.0),
    (REASONING + "FORECAST: -8.5\n\n", -8.5),  # trailing blank lines are not a line
    ("FORECAST: +0.0", 0.0), ("FORECAST: -0.0", 0.0),  # 0 with either sign
    (REASONING + "FORECAST: 15.0", None),  # no sign
    (REASONING + "FORECAST: +15", None),  # no decimal
    (REASONING + "FORECAST: +15.00", None),  # two decimals
    (REASONING + "FORECAST: +15.0%", None),  # extra text
    (REASONING + "Forecast: +15.0", None),  # wrong case
    (REASONING + " FORECAST: +15.0", None),  # not at the line's start
    ("FORECAST: +15.0\nThat is my view.", None),  # not the last line
    ("FORECAST: +10.0\nFORECAST: +15.0", None),  # two forecasts
    (REASONING + "Revenue looks", None),  # cut off at the token limit: no final line
    ("", None),
])
def test_the_forecast_line_must_be_the_last_and_only_one_exactly_in_form(text, forecast):
    assert parse_forecast(text) == forecast


def test_forecasts_are_ranked_closest_first_and_ranks_1_to_9_earn_9_to_1_points(tmp_path):
    scored = _ranked(tmp_path, [("p", f"h{n}", _f(n)) for n in (3, 1, 2)])
    assert [s.details["error_rank"] for s in scored] == [3, 1, 2]
    assert [s.details["rank_points"] for s in scored] == [7, 9, 8]
    assert [s.score for s in scored] == pytest.approx([5 * 7 / 9, 5.0, 5 * 8 / 9])
    assert [s.details["abs_error_pp"] for s in scored] == [3.0, 1.0, 2.0]
    assert {s.method for s in scored} == {"eps_error"}


def test_the_tenth_closest_and_beyond_earn_nothing(tmp_path):
    scored = _ranked(tmp_path, [("p", f"h{n}", _f(n)) for n in range(1, 13)])
    assert [s.details["rank_points"] for s in scored] == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0]
    assert [s.details["error_rank"] for s in scored] == list(range(1, 13))
    assert scored[-1].score == 0.0


def test_equal_errors_share_the_best_rank_they_span(tmp_path):
    # +2.0 and -2.0 are both 2 off an actual of 0: both rank 2; the next is rank 4.
    scored = _ranked(tmp_path, [("p", "a", _f(1)), ("p", "b", _f(2)), ("p", "c", _f(-2)), ("p", "d", _f(4))])
    assert [(s.details["error_rank"], s.details["rank_points"]) for s in scored] == [(1, 9), (2, 8), (2, 8), (4, 6)]
    # A tie spanning ranks 9 and 10 shares rank 9: a point each.
    scored = _ranked(tmp_path, [*(("p", f"h{n}", _f(n)) for n in range(1, 9)), ("p", "x", _f(9)), ("p", "y", _f(-9))])
    assert [(s.details["error_rank"], s.details["rank_points"]) for s in scored[-2:]] == [(9, 1), (9, 1)]
    # Errors equal to 4 decimals are equal, whatever the float noise.
    actual = 0.1 + 0.2  # 0.30000000000000004: errors of 0.2 and 0.4 differ by float noise only
    assert abs(0.2 - actual) != abs(0.4 - actual)
    scored = _ranked(tmp_path, [("p", "a", _f(0.0)), ("p", "b", _f(0.2)), ("p", "c", _f(0.4))], {"p": actual})
    assert [s.details["error_rank"] for s in scored] == [3, 1, 1]


def test_an_answer_with_no_valid_forecast_scores_0_takes_no_rank_and_moves_no_one(tmp_path):
    scored = _ranked(tmp_path, [("p", "a", REASONING + "FORECAST: 1.0"), ("p", "b", _f(2)), ("p", "c", "cut off"),
                                ("p", "d", _f(3))])
    assert [(s.details["error_rank"], s.details["rank_points"], s.score) for s in scored] == \
           [(None, None, 0.0), (1, 9, 5.0), (None, None, 0.0), (2, 8, pytest.approx(5 * 8 / 9))]
    assert (scored[0].details["forecast_pct"], scored[0].details["abs_error_pp"]) == (None, None)


def test_copies_share_their_original_s_place_and_push_no_one_down(tmp_path):
    scored = _ranked(tmp_path, [("p", "h1", _f(1)), ("p", "h1", _f(1)), ("p", "h1", _f(1)), ("p", "h2", _f(2))])
    assert [(s.details["error_rank"], s.details["rank_points"]) for s in scored] == [(1, 9)] * 3 + [(2, 8)]
    # With no model_hash, each answer is its own place.
    scored = _ranked(tmp_path, [("p", None, _f(1)), ("p", None, _f(1)), ("p", None, _f(2))])
    assert [s.details["error_rank"] for s in scored] == [1, 1, 3]


def test_each_prompt_is_ranked_on_its_own_against_its_own_actual(tmp_path):
    scored = _ranked(tmp_path, [("p1", "a", _f(10)), ("p1", "b", _f(0)), ("p2", "a", _f(10)), ("p2", "b", _f(0))],
                     {"p1": 9.0, "p2": -1.0})
    assert [s.details["error_rank"] for s in scored] == [1, 2, 2, 1]
    assert [s.details["answer_id"] for s in scored] == ["ans-p1", "ans-p1", "ans-p2", "ans-p2"]


def test_the_registry_and_config_take_eps_error_and_refuse_bad_settings():
    assert "eps_error" in REGISTRY and REGISTRY["eps_error"].name == "eps_error"
    _eps_cfg("/a.jsonl", "a" * 64)
    for settings in ({"answers_path": "a.jsonl", "answers_sha256": "a" * 64},
                     {"answers_path": "/a.jsonl", "answers_sha256": "A" * 64},
                     {"answers_path": "/a.jsonl", "answers_sha256": "a" * 64, "cap_pp": 25},  # no cap any more
                     {"answers_path": "/a.jsonl", "answers_sha256": "a" * 64, "weight": 1}):
        with pytest.raises(ValidationError):
            ScoringMethodCfg(method="eps_error", settings=settings)


def test_the_scores_stage_scores_each_forecast_against_the_actual_change_with_no_jev_call(config, fake,  # noqa: F811
                                                                                         tmp_path):
    _setup(config, tmp_path, {"h1": REASONING + "FORECAST: -35.0", "h2": REASONING + "FORECAST: 15.0"},
           answer_for=("p1", "p2"))

    run.run_stages(config, CYCLE_ID, "scores")

    rows = {(r["uid"], r["prompt_id"]): r for r in _scores(config)}
    one, two = rows[(1, "p1")], rows[(2, "p1")]
    assert (one["method"], one["forecast_pct"], one["abs_error_pp"], one["error_rank"], one["rank_points"],
            one["answer_id"]) == ("eps_error", -35.0, 6.22, 1, 9, "ans-p1")
    assert one["expected_score"] == one["score"] == 1.0  # 9 points of 9
    assert (two["forecast_pct"], two["error_rank"], two["expected_score"]) == (None, None, 0.0)  # no sign: format
    assert rows[(3, "p1")] | {"uid": 1, "hotkey": "hk1"} == one  # uid 3 is uid 1's model: its place, not another
    assert (one["rank"], two["rank"]) == (0, 1)
    assert fake.bodies == []  # no Jev call at all


def test_the_scores_stage_ranks_the_second_closest_model_second(config, fake, tmp_path):  # noqa: F811
    _setup(config, tmp_path, {"h1": REASONING + "FORECAST: -35.0", "h2": REASONING + "FORECAST: -50.0"},
           answer_for=("p1", "p2"))

    run.run_stages(config, CYCLE_ID, "scores")

    rows = {(r["uid"], r["prompt_id"]): r for r in _scores(config)}
    assert [(rows[(u, "p1")]["error_rank"], rows[(u, "p1")]["expected_score"]) for u in (1, 2, 3)] == \
           [(1, 1.0), (2, round(8 / 9, 4)), (1, 1.0)]  # the copy (uid 3) takes no place


def test_a_prompt_with_no_answer_in_the_file_stops_the_stage(config, fake, tmp_path):  # noqa: F811
    _setup(config, tmp_path, {"h1": "FORECAST: +1.0", "h2": "FORECAST: +1.0"}, answer_for=("p1",))

    with pytest.raises(AnswersFileError, match=r"no answer for prompts \['p2'\]"):
        run.run_stages(config, CYCLE_ID, "scores")
    assert not stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "scores").exists()


def test_a_missing_or_changed_answers_file_stops_the_snapshot_and_scores(config, fake, tmp_path,  # noqa: F811
                                                                         monkeypatch):
    path = _setup(config, tmp_path, {"h1": "FORECAST: +1.0", "h2": "FORECAST: +1.0"}, answer_for=("p1", "p2"))
    path.write_text(path.read_text() + "\n")  # one byte more: another file

    with pytest.raises(AnswersFileError, match="not the pinned"):
        run.run_stages(config, CYCLE_ID, "scores")
    path.unlink()
    with pytest.raises(AnswersFileError, match="is missing"):
        run.run_stages(config, CYCLE_ID, "scores")
    monkeypatch.setattr(snapshot, "sync_owner_task", lambda cfg: None)
    with pytest.raises(AnswersFileError, match="is missing"):
        snapshot.run_snapshot(config, CYCLE_ID)

    assert fake.bodies == []
    assert not stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "scores").exists()
    assert issubclass(AnswersFileError, methods.MethodStop)  # the loop exits 78 on it (test_prompt_validator_loop)


def test_load_answers_keys_by_content_hash_ignores_other_fields_and_refuses_two_for_one_prompt(tmp_path):
    path, sha = _answers_file(tmp_path, [{"prompt_id": "ans-01", "content_hash": "a" * 64, "eps_pct_change": -9.1,
                                          "company": "any", "links": {"10q": "https://example.com"}}])
    assert eps.load_answers(_eps_cfg(path, sha).parsed()) == {"a" * 64: {"answer_id": "ans-01",
                                                                          "eps_pct_change": -9.1}}
    path, sha = _answers_file(tmp_path, [{"prompt_id": "ans-01", "content_hash": "a" * 64,
                                          "eps_pct_change": 1.0}] * 2)
    with pytest.raises(AnswersFileError, match="two answers"):
        eps.load_answers(_eps_cfg(path, sha).parsed())


def test_the_rules_hash_holds_the_listed_task_s_file_hash_and_rules_but_not_the_path(tmp_path, monkeypatch):
    def rules(**scoring_methods):
        pv = PromptValidatorCfg(runs_dir=tmp_path / "runs", prompts_path=tmp_path / "p.jsonl")
        pv.scoring_methods = scoring_methods
        return snapshot.rules_sha256(pv, BASE)

    on = rules(earnings_forecast=_eps_cfg("/a/answers.jsonl", "a" * 64))
    assert on != rules()
    assert on == rules(earnings_forecast=_eps_cfg("/elsewhere/answers.jsonl", "a" * 64))
    assert on != rules(earnings_forecast=_eps_cfg("/a/answers.jsonl", "b" * 64))
    monkeypatch.setattr(REGISTRY["eps_error"], "RULES", eps.EpsError.RULES + " changed")
    assert rules(earnings_forecast=_eps_cfg("/a/answers.jsonl", "a" * 64)) != on


@pytest.mark.parametrize("line, match", [
    (b"not json", "line 2 is not JSON"),
    (b"[1, 2]", "line 2 is not a JSON object"),
    (b'{"prompt_id": "ans-02", "eps_pct_change": 1.0}', "line 2 has no content_hash"),
    (b'{"prompt_id": "ans-02", "content_hash": "' + b"b" * 64 + b'", "eps_pct_change": "1.0"}',
     "line 2 has no eps_pct_change"),
    (b'{"prompt_id": "ans-02", "content_hash": "' + b"b" * 64 + b'", "eps_pct_change": true}',
     "line 2 has no eps_pct_change"),
    (b'{"prompt_id": "ans-02", "content_hash": "' + b"b" * 64 + b'", "eps_pct_change": NaN}',
     "line 2 has no eps_pct_change"),
    (b"\xff", "is not UTF-8"),
])
def test_a_malformed_answers_line_stops_with_its_line_number_not_its_text(tmp_path, line, match):
    path = tmp_path / "answers.jsonl"
    good = json.dumps({"prompt_id": "ans-01", "content_hash": "a" * 64, "eps_pct_change": -9.1}).encode()
    path.write_bytes(good + b"\n" + line + b"\n")
    settings = _eps_cfg(path, hashlib.sha256(path.read_bytes()).hexdigest()).parsed()

    with pytest.raises(AnswersFileError, match=match) as e:
        eps.load_answers(settings)
    assert "-9.1" not in str(e.value)
