import json
import logging
from time import sleep as real_sleep  # bound before the tests patch time.sleep
from types import SimpleNamespace

import pytest
import requests

from connito.prompt_validator import jev
from connito.prompt_validator.config import JevCfg
from connito.prompt_validator.stage_io import read_rows, stage_path, write_rows
from connito.shared import helper

REQUEST = {"goal": "goal one", "examples": ["example one"], "tone": "plain"}
API_KEY = "test-key"


class FakeJev:
    """The Jev endpoint: scores each answer "n" as n (an empty one as 0), after replying `statuses` first (an
    exception is raised).

    Each answer's probabilities put it all on its own level, unless `probabilities` is given. `quality` replaces
    every answer's {score, confidence, probabilities}; `missing` leaves those question ids out of the response.
    A call holding an answer in `refuse` gets a 401.
    """

    def __init__(self, resolved="jev-1.14.0", statuses=(), usage=True, quality=None, slow_low_scores=False,
                 error_text="slow down", missing=(), extra=(), probabilities=None):
        self.resolved = resolved
        self.statuses = list(statuses)
        self.usage = usage
        self.quality = quality
        self.slow_low_scores = slow_low_scores
        self.error_text = error_text
        self.missing = set(missing)
        self.refuse = set()
        self.extra = list(extra)
        self.probabilities = probabilities
        self.bodies = []

    def post(self, url, json=None, headers=None, timeout=None):
        assert headers == {"Authorization": f"Bearer {API_KEY}"}
        self.bodies.append(json)
        if self.statuses:
            status = self.statuses.pop(0)
            if isinstance(status, Exception):
                raise status
            return SimpleNamespace(status_code=status, text=self.error_text)
        if any(q["instructions"]["answer"] in self.refuse for q in json["questions"].values()):
            return SimpleNamespace(status_code=401, text="unauthorized")
        scores = {qid: float(q["instructions"]["answer"] or 0) for qid, q in json["questions"].items()}
        if self.slow_low_scores:
            real_sleep(0.01 * (5 - min(scores.values())))
        answers = {
            qid: {"score": score, "confidence": 0.9, "probabilities": self._probabilities(score)}
            if self.quality is None else self.quality
            for qid, score in scores.items()
            if qid not in self.missing
        }
        answers.update({qid: {"score": 3.0, "confidence": 0.9, "probabilities": self._probabilities(3)}
                        for qid in self.extra})
        payload = {"model": self.resolved, "answers": answers}
        if self.usage:
            payload["usage"] = {"input_tokens": 1000}
        return SimpleNamespace(status_code=200, text="", json=lambda: payload)

    def _probabilities(self, score):
        if self.probabilities is not None:
            return self.probabilities
        return {str(level): float(level == round(score)) for level in range(6)}


def score_one(client, answer):
    return client.score_many([(REQUEST, "prompt one", answer)])[0]


@pytest.fixture
def serve(monkeypatch):
    monkeypatch.setenv(jev.API_KEY_ENV, API_KEY)
    monkeypatch.setattr(helper.time, "sleep", lambda s: None)

    def serve(**kwargs):
        fake = FakeJev(**kwargs)
        monkeypatch.setattr(requests, "post", fake.post)
        return fake

    return serve


@pytest.fixture
def client(tmp_path):
    def client(**cfg):
        return jev.JevClient(JevCfg(**cfg), tmp_path / "jev_cache", tmp_path / "7" / "assess" / "jev_calls.jsonl")

    return client


def test_cache_hit_makes_no_call(serve, client):
    fake = serve()
    first = score_one(client(), "4")
    again = score_one(client(), "4")
    assert len(fake.bodies) == 1
    assert (first["cached"], again["cached"]) == (False, True)
    assert {**first, "cached": True} == again
    assert first == {"score": 0.8, "expected_score": 0.8, "jev_call_id": first["jev_call_id"], "cached": False}


def test_expected_score_is_the_expected_level_over_jevs_probabilities(serve, client):
    # As Jev answered one live call: score 4.71, and these probabilities, keyed like its legend.
    serve(probabilities={"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.02, "4": 0.22, "5": 0.76})
    row = score_one(client(), "4.71")
    assert row["score"] == 0.942  # 4.71 / 5
    assert row["expected_score"] == 0.948  # (3×0.02 + 4×0.22 + 5×0.76) / 5 = 4.74 / 5


def test_expected_score_reads_a_list_and_normalises_by_the_sum(serve, client):
    serve(probabilities=[0, 0, 0, 0, 1, 1])  # sums to 2: half on 4, half on 5
    assert score_one(client(), "4")["expected_score"] == 0.9


@pytest.mark.parametrize("probabilities", [
    None, {"0": 1.0}, [0.5] * 5, {str(i): 0.0 for i in range(6)}, {**{str(i): 0.2 for i in range(5)}, "5": -0.1},
    {**{str(i): 0.2 for i in range(5)}, "5": "0.1"}, {"a": 1, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0},
])
def test_probabilities_not_one_per_level_raise_and_cache_nothing(serve, client, tmp_path, probabilities):
    serve(quality={"score": 4.0, "confidence": 0.9, "probabilities": probabilities})
    with pytest.raises(RuntimeError, match=r"no probability for each level 0 to 5 for answer_1 \(item [0-9a-f]{12}\)"):
        score_one(client(), "4")
    assert not (tmp_path / "jev_cache").exists()


def test_a_row_cached_before_expected_score_is_scored_again(serve, client, tmp_path):
    fake = serve()
    score_one(client(), "4")
    [path] = (tmp_path / "jev_cache").rglob("*.jsonl")
    path.write_text(json.dumps({"score_raw": 4.0, "score": 0.8, "confidence": 0.9, "jev_call_id": "old"}) + "\n")
    again = score_one(client(), "4")
    assert len(fake.bodies) == 2 and again["cached"] is False and again["expected_score"] == 0.8
    assert score_one(client(), "4")["cached"] is True


def test_resolved_version_mismatch_raises_and_caches_nothing(serve, client, tmp_path):
    serve(resolved="jev-1.15.0")
    with pytest.raises(jev.JevVersionError, match="jev-1.15.0"):
        score_one(client(), "4")
    assert not (tmp_path / "jev_cache").exists()


def test_429_is_retried_then_succeeds(serve, client):
    fake = serve(statuses=[429, 429])
    assert score_one(client(), "3")["score"] == 0.6
    assert len(fake.bodies) == 3


def test_non_retryable_401_raises_after_one_call(serve, client):
    fake = serve(statuses=[401])
    with pytest.raises(RuntimeError, match="Jev call failed"):
        score_one(client(), "4")
    assert len(fake.bodies) == 1


def test_a_timed_out_call_is_not_retried_and_stops_the_stage(serve, client, tmp_path, caplog):
    fake = serve(statuses=[requests.exceptions.ReadTimeout("read timed out")])
    with caplog.at_level(logging.DEBUG), pytest.raises(RuntimeError, match="Jev call failed"):
        score_one(client(), "4")
    assert len(fake.bodies) == 1
    assert "Request timed out, not retrying" in caplog.text
    assert not (tmp_path / "jev_cache").exists()


def test_a_rejected_request_keeps_its_echoed_body_out_of_the_logs(serve, client, caplog):
    tag = "private-prompt-7f3c"
    serve(statuses=[422], error_text=json.dumps({"detail": [{"input": {"state": {"prompt": tag}}}]}))
    with caplog.at_level(logging.DEBUG), pytest.raises(RuntimeError, match="Jev call failed"):
        score_one(client(), "4")
    assert "422" in caplog.text
    assert tag not in caplog.text


def test_response_without_usage_is_logged_then_raises(serve, client, tmp_path):
    serve(usage=False)
    with pytest.raises(RuntimeError, match="usage.input_tokens"):
        score_one(client(), "4")
    assert len(read_rows(tmp_path / "7" / "assess" / "jev_calls.jsonl")) == 1
    assert not (tmp_path / "jev_cache").exists()


def test_spend_cap_stops_further_calls(serve, client):
    fake = serve()
    capped = client(spend_cap_usd=0.001, price_per_m=1.0)  # one call of 1000 tokens spends the whole cap
    score_one(capped, "4")
    with pytest.raises(RuntimeError, match="spend cap"):
        score_one(capped, "5")
    assert len(fake.bodies) == 1


def test_spend_starts_from_the_cycles_calls_log(serve, client, tmp_path):
    fake = serve()
    write_rows(tmp_path / "7" / "assess" / "jev_calls.jsonl", [{"call_id": "earlier", "cost": 5.0}])
    with pytest.raises(RuntimeError, match="spend cap"):
        score_one(client(), "4")
    assert fake.bodies == []


def test_each_live_call_appends_one_row_to_the_calls_log(serve, client, tmp_path):
    serve()
    c = client()
    results = [score_one(c, answer) for answer in ("4", "2", "4")]
    log = tmp_path / "7" / "assess" / "jev_calls.jsonl"
    rows = read_rows(log)
    assert [row["call_id"] for row in rows] == [r["jev_call_id"] for r in results[:2]]
    assert all(row["model_resolved"] == "jev-1.14.0" and row["input_tokens"] == 1000 for row in rows)
    assert API_KEY not in log.read_text()


@pytest.mark.parametrize("quality", [{"score": 7.0}, {"confidence": 0.9}])
def test_invalid_score_raises_and_caches_nothing(serve, client, tmp_path, quality):
    serve(quality=quality)
    with pytest.raises(RuntimeError, match=r"score .* for answer_1 \(item [0-9a-f]{12}\), expected 0 to 5"):
        score_one(client(), "4")
    assert not (tmp_path / "jev_cache").exists()


def test_score_many_keeps_input_order(serve, client):
    serve(slow_low_scores=True)  # earlier calls finish last
    answers = ["0", "1", "2", "3", "4", "5", "1"]
    items = [(REQUEST, f"Prompt {i % 3}", a) for i, a in enumerate(answers)]  # three prompts, so three calls
    results = client(concurrency=6).score_many(items)
    assert [r["score"] for r in results] == [int(a) / 5 for a in answers]


def test_score_many_sends_each_distinct_answer_once(serve, client):
    fake = serve()
    items = [(REQUEST, "prompt one", "4"), (REQUEST, "prompt one", "2"), (REQUEST, "prompt one", "4")]
    results = client().score_many(items)
    assert len(fake.bodies) == 1
    assert sorted(q["instructions"]["answer"] for q in fake.bodies[0]["questions"].values()) == ["2", "4"]
    assert results[0] == results[2]


def test_answers_to_one_prompt_share_one_call_each_as_a_score_question(serve, client):
    fake = serve()
    results = client().score_many([(REQUEST, "prompt one", a) for a in ("4", "1", "5")])

    [body] = fake.bodies
    # The state holds what the answers share; each answer is its own score question, named without the miner.
    assert body["state"] == {"customer_request": REQUEST, "prompt": "prompt one"}
    assert list(body["questions"]) == ["answer_1", "answer_2", "answer_3"]
    [question] = [q for q in body["questions"].values() if q["instructions"]["answer"] == "1"]
    assert question == {
        "type": "score",
        "instructions": {"question": jev.QUALITY_QUESTION, "answer": "1"},
        "criteria": jev.SCORE_LEVELS,
    }
    assert "aimed at you or another grader" in jev.QUALITY_QUESTION and "score 0" in jev.QUALITY_QUESTION
    assert "aimed at the grader" in jev.SCORE_LEVELS[0]
    assert [r["score"] for r in results] == [0.8, 0.2, 1.0]
    assert len({r["jev_call_id"] for r in results}) == 1


def test_each_prompt_gets_its_own_call(serve, client):
    fake = serve()
    client().score_many([(REQUEST, "prompt one", "4"), (REQUEST, "prompt two", "4"), (REQUEST, "prompt one", "2")])

    assert sorted(len(body["questions"]) for body in fake.bodies) == [1, 2]
    assert {body["state"]["prompt"] for body in fake.bodies} == {"prompt one", "prompt two"}


def test_a_call_holds_at_most_max_answers_per_call(serve, client):
    fake = serve()
    results = client(max_answers_per_call=2).score_many([(REQUEST, "prompt one", str(a)) for a in range(5)])

    assert sorted(len(body["questions"]) for body in fake.bodies) == [1, 2, 2]
    assert [r["score"] for r in results] == [0.0, 0.2, 0.4, 0.6, 0.8]


def test_only_answers_not_yet_cached_are_sent(serve, client):
    fake = serve()
    c = client()
    c.score_many([(REQUEST, "prompt one", "4")])

    results = c.score_many([(REQUEST, "prompt one", a) for a in ("4", "2")])

    assert [q["instructions"]["answer"] for q in fake.bodies[1]["questions"].values()] == ["2"]
    assert [r["cached"] for r in results] == [True, False]


def test_a_response_missing_one_answer_caches_none_of_the_call(serve, client, tmp_path):
    serve(missing={"answer_2"})
    with pytest.raises(RuntimeError, match=r"score None for answer_2 \(item [0-9a-f]{12}\)"):
        client().score_many([(REQUEST, "prompt one", a) for a in ("4", "1")])
    assert not (tmp_path / "jev_cache").exists()
    assert len(read_rows(tmp_path / "7" / "assess" / "jev_calls.jsonl")) == 1  # billed, so counted


def test_two_requests_with_the_same_prompt_are_separate_calls(serve, client):
    fake = serve()
    other = {**REQUEST, "goal": "goal two"}
    client().score_many([(REQUEST, "prompt one", "4"), (other, "prompt one", "4")])

    assert sorted(body["state"]["customer_request"]["goal"] for body in fake.bodies) == [
        "goal one", "goal two",
    ]


def test_the_pinned_version_is_requested_by_name_and_logged(serve, client, tmp_path):
    fake = serve(resolved="jev-1.13.0")
    score_one(client(resolved="jev-1.13.0"), "4")

    assert [body["model"] for body in fake.bodies] == ["jev-1.13.0"]
    [row] = read_rows(tmp_path / "7" / "assess" / "jev_calls.jsonl")
    assert (row["model_requested"], row["model_resolved"]) == ("jev-1.13.0", "jev-1.13.0")


def test_another_pinned_version_is_a_cache_miss(serve, client):
    fake = serve(resolved="jev-1.13.0")
    score_one(client(resolved="jev-1.13.0"), "4")
    fake.resolved = "jev-1.14.0"

    assert score_one(client(resolved="jev-1.14.0"), "4")["cached"] is False
    assert [body["model"] for body in fake.bodies] == ["jev-1.13.0", "jev-1.14.0"]


def test_a_batched_call_is_logged_and_charged_once(serve, client, tmp_path):
    serve()
    c = client(price_per_m=1.0)
    c.score_many([(REQUEST, "prompt one", a) for a in ("4", "1", "5")])

    [row] = read_rows(tmp_path / "7" / "assess" / "jev_calls.jsonl")
    assert (row["answers"], row["input_tokens"], row["cost"]) == (3, 1000, 0.001)
    assert c.spent == 0.001


def test_an_answers_position_does_not_follow_input_order(serve, tmp_path):
    fake = serve()
    answers = ["1", "2", "3", "4"]
    for i, order in enumerate([answers, answers[::-1]]):  # two validators, fresh caches, opposite input orders
        jev.JevClient(JevCfg(), tmp_path / f"cache{i}", tmp_path / f"calls{i}.jsonl").score_many(
            [(REQUEST, "prompt one", a) for a in order]
        )

    first, second = ([q["instructions"]["answer"] for q in body["questions"].values()] for body in fake.bodies)
    assert first == second


@pytest.mark.parametrize("value", [0, -1])
def test_max_answers_per_call_below_one_does_not_load(value):
    with pytest.raises(ValueError, match="max_answers_per_call"):
        JevCfg(max_answers_per_call=value)


def test_answers_to_questions_not_asked_are_logged_and_ignored(serve, client, caplog):
    serve(extra=["answer_9"])
    with caplog.at_level(logging.DEBUG):
        assert score_one(client(), "4")["score"] == 0.8
    assert "Jev answered questions it was not asked" in caplog.text


@pytest.mark.no_jev_price
def test_without_a_price_no_client_is_made_and_nothing_is_called(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_DIRECT_API_KEY", "key")
    monkeypatch.setattr(jev, "get_with_retry", lambda *a, **k: pytest.fail("Jev was called"))
    with pytest.raises(RuntimeError, match="jev.price_per_m is not set: the spend cap needs Jev's price"):
        jev.JevClient(JevCfg(), tmp_path / "jev_cache", tmp_path / "jev_calls.jsonl")


# --- levels by the prompt's task ---

LESSON = "exp_lesson_residual"


def _criteria(body):
    return [q["criteria"] for q in body["questions"].values()]


def test_a_lesson_prompt_is_scored_on_the_lesson_levels(serve, client):
    fake = serve()
    result = client().score_many([(None, "Teach it", "4")], tasks=[LESSON])[0]

    [body] = fake.bodies
    assert _criteria(body) == [jev.LESSON_SCORE_LEVELS]
    # Only the levels change: the question and its score-0 rule are the generic ones.
    assert [q["instructions"]["question"] for q in body["questions"].values()] == [jev.QUALITY_QUESTION_NO_REQUEST]
    assert "aimed at the grader" in jev.LESSON_SCORE_LEVELS[0]
    assert result["score"] == 0.8


@pytest.mark.parametrize("task", ["exp_pmc_residual", None, "", "a_task_not_mapped"])
def test_any_other_task_or_none_is_scored_on_the_generic_levels(serve, client, task):
    fake = serve()
    client().score_many([(None, "Explain it", "4"), (REQUEST, "Explain it", "4")], tasks=[task, task])

    assert sorted(map(str, (c for body in fake.bodies for c in _criteria(body)))) == sorted(
        map(str, [jev.SCORE_LEVELS_NO_REQUEST, jev.SCORE_LEVELS]))


def test_without_tasks_every_item_gets_the_generic_levels_and_its_old_cache_key(serve, client, tmp_path):
    fake = serve()
    c = client()
    c.score_many([(None, "Explain it", "4")])
    # The generic levels' cache key did not change with CRITERIA_BY_TASK: a score cached before it is reused.
    assert c.score_many([(None, "Explain it", "4")], tasks=["exp_pmc_residual"])[0]["cached"] is True
    assert c.score_many([(None, "Explain it", "4")], tasks=[None])[0]["cached"] is True
    assert len(fake.bodies) == 1


def test_a_mixed_batch_picks_the_levels_per_prompt(serve, client):
    fake = serve()
    items = [(None, "Teach it", "4"), (None, "Explain it", "3"), (None, "Teach it", "2"), (None, "Explain it", "5")]
    results = client().score_many(items, tasks=[LESSON, "exp_pmc_residual", LESSON, None])

    by_prompt = {body["state"]["prompt"]: body for body in fake.bodies}
    assert set(by_prompt) == {"Teach it", "Explain it"} and len(fake.bodies) == 2
    assert _criteria(by_prompt["Teach it"]) == [jev.LESSON_SCORE_LEVELS] * 2
    assert _criteria(by_prompt["Explain it"]) == [jev.SCORE_LEVELS_NO_REQUEST] * 2
    assert [r["score"] for r in results] == [0.8, 0.6, 0.4, 1.0]


def test_one_prompt_text_under_two_tasks_is_scored_twice_on_each_tasks_levels(serve, client):
    fake = serve()
    results = client().score_many([(None, "Same text", "4")] * 2, tasks=[LESSON, "exp_pmc_residual"])

    assert sorted(map(str, (c for body in fake.bodies for c in _criteria(body)))) == sorted(
        map(str, [jev.LESSON_SCORE_LEVELS, jev.SCORE_LEVELS_NO_REQUEST]))
    assert [r["cached"] for r in results] == [False, False]


def test_every_task_s_levels_are_on_the_generic_scale():
    for levels in jev.CRITERIA_BY_TASK.values():
        assert len(levels) == jev.MAX_LEVEL + 1
        assert [level.split(" ", 1)[0] for level in levels] == [str(n) for n in range(jev.MAX_LEVEL + 1)]


def test_the_scores_stage_passes_each_prompt_s_own_task(monkeypatch, tmp_path):
    from connito.prompt_validator import scores

    snapshot_dir = stage_path(tmp_path, 7, "snapshot").parent
    write_rows(snapshot_dir / "prompts.jsonl", [
        {"prompt_id": "1", "prompt": "Teach it", "request": None, "customer_task_id": LESSON},
        {"prompt_id": "2", "prompt": "Explain it", "request": None, "customer_task_id": "exp_pmc_residual"},
        {"prompt_id": "3", "prompt": "Old row", "request": None},
    ])
    write_rows(stage_path(tmp_path, 7, "answers"), [
        # The lesson prompt's answer a course, so it passes the lesson task's format check (answer_format.py).
        {"uid": u, "hotkey": f"hk{u}", "model_hash": f"m{u}", "prompt_id": p,
         "text": json.dumps({"lessons": [{"lesson_number": 1, "title": "t", "body": "b", "source_ids": []}] * 3})
         if p == "1" else "4"}
        for u in (1, 2) for p in ("1", "2", "3")
    ])
    seen = {}

    class Client:
        def score_many(self, items, tasks=None):
            seen["tasks"] = tasks
            return [{"score": 0.8, "expected_score": 0.8, "jev_call_id": "c"} for _ in items]

    monkeypatch.setattr(scores, "snapshot_meta", lambda config, cycle: {})
    monkeypatch.setattr(scores, "selected_prompts", lambda pv, cycle: [])
    monkeypatch.setattr(scores, "cycle_client", lambda pv, cycle: Client())
    config = SimpleNamespace(prompt_validator=SimpleNamespace(runs_dir=tmp_path))

    assert len(scores.run_scores(config, 7)) == 6
    assert seen["tasks"] == [LESSON, "exp_pmc_residual", None] * 2
