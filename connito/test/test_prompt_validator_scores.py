from types import SimpleNamespace

import pytest
import requests
from test_prompt_validator_jev import API_KEY, FakeJev

from connito.prompt_validator import jev, run, snapshot
from connito.prompt_validator.config import PromptValidatorCfg
from connito.prompt_validator.stage_io import read_rows, stage_path, write_rows
from connito.shared import helper
from connito.test.conftest import needs_judge_rules

CYCLE_ID = 1258
# The snapshot's base model rows; base_rows rebuilds the same from config unless a test changes them.
BASE = {"base_model": "Qwen/Qwen3.6-35B-A3B", "base_model_revision": "0" * 40, "chat_template_sha256": "1" * 64}
REQUESTS = {
    "r1": {"goal": "goal one", "examples": ["example one"], "tone": "plain"},
    "r2": {"goal": "goal three", "examples": [], "tone": "warm"},
}
PROMPTS = [("p1", "r1", "prompt one"), ("p2", "r2", "prompt three")]
# uid -> model_hash -> its answer to each prompt. uids 1 and 3 committed the same weights, so the same answers.
ANSWERS = {"h1": {"p1": "4", "p2": ""}, "h2": {"p1": "2", "p2": "5"}}
HASHES = {1: "h1", 2: "h2", 3: "h1"}


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv(jev.API_KEY_ENV, API_KEY)
    monkeypatch.setattr(helper.time, "sleep", lambda s: None)
    fake = FakeJev()
    monkeypatch.setattr(requests, "post", fake.post)
    return fake


@pytest.fixture
def config(tmp_path, monkeypatch):
    pv = PromptValidatorCfg(runs_dir=tmp_path / "runs", prompts_path=tmp_path / "p.jsonl")
    config = SimpleNamespace(prompt_validator=pv)
    monkeypatch.setattr(snapshot, "base_rows", lambda cfg: dict(BASE))  # the real one downloads the chat template
    snapshot_dir = stage_path(pv.runs_dir, CYCLE_ID, "snapshot").parent
    write_rows(snapshot_dir / "prompts.jsonl", [
        {"prompt_id": p, "request_id": r, "prompt": text, "request": REQUESTS[r]} for p, r, text in PROMPTS
    ])
    write_rows(snapshot_dir / "snapshot.jsonl", [{**BASE, "rules_sha256": snapshot.rules_sha256(pv, BASE)}])
    write_rows(stage_path(pv.runs_dir, CYCLE_ID, "answers"), [
        {"uid": uid, "hotkey": f"hk{uid}", "model_hash": h, "prompt_id": p, "text": ANSWERS[h][p], "n_tokens": 1,
         "finish_reason": "stop"}
        for uid, h in HASHES.items() for p, _, _ in PROMPTS
    ])
    return config


def _scores(config):
    return read_rows(stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "scores"))


def _calls_log(config):
    return read_rows(stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "scores").parent / "jev_calls.jsonl")


def test_every_answer_is_scored_once_and_rows_join_to_their_answer(config, fake):
    run.run_stages(config, CYCLE_ID, "scores")

    # Six answers, four distinct: uid 3's answers are uid 1's, and cost nothing more. One call per prompt.
    request_of = {text: REQUESTS[r] for _, r, text in PROMPTS}
    assert len(fake.bodies) == 2
    assert all(b["state"]["customer_request"] == request_of[b["state"]["prompt"]] for b in fake.bodies)
    sent = sorted(
        (b["state"]["prompt"], q["instructions"]["answer"]) for b in fake.bodies for q in b["questions"].values()
    )
    assert sent == [("prompt one", "2"), ("prompt one", "4"), ("prompt three", ""), ("prompt three", "5")]

    rows = _scores(config)
    # rank is per prompt, 0 for the best; uids 1 and 3 are one model (h1), so they share a rank and take one place.
    assert [(r["uid"], r["hotkey"], r["model_hash"], r["prompt_id"], r["score"], r["expected_score"], r["rank"])
            for r in rows] == [
        (1, "hk1", "h1", "p1", 0.8, 0.8, 0), (1, "hk1", "h1", "p2", 0.0, 0.0, 1),
        (2, "hk2", "h2", "p1", 0.4, 0.4, 1), (2, "hk2", "h2", "p2", 1.0, 1.0, 0),
        (3, "hk3", "h1", "p1", 0.8, 0.8, 0), (3, "hk3", "h1", "p2", 0.0, 0.0, 1),
    ]
    assert set(rows[0]) == {
        "uid", "hotkey", "model_hash", "prompt_id", "score", "expected_score", "rank", "jev_call_id",
    }
    # Answers to one prompt share a call: p1's rows (0, 2, 4) one id, p2's another.
    assert rows[0]["jev_call_id"] == rows[2]["jev_call_id"] == rows[4]["jev_call_id"] != rows[1]["jev_call_id"]
    assert {r["jev_call_id"] for r in rows} == {c["call_id"] for c in _calls_log(config)}  # on a fresh cache
    path = stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "scores")
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_rank_follows_expected_score_not_score(config, fake):
    # p1's "2" gets probabilities whose expected level (4.5) is above that of "4" (4.0): it ranks first.
    fake._probabilities = lambda score: (
        {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0.5, "5": 0.5} if score == 2 else
        {str(level): float(level == round(score)) for level in range(6)}
    )
    run.run_stages(config, CYCLE_ID, "scores")

    p1 = {r["uid"]: (r["score"], r["expected_score"], r["rank"]) for r in _scores(config) if r["prompt_id"] == "p1"}
    assert p1 == {1: (0.8, 0.8, 1), 2: (0.4, 0.9, 0), 3: (0.8, 0.8, 1)}


def test_a_rerun_makes_no_live_calls_and_writes_the_same_rows(config, fake):
    run.run_stages(config, CYCLE_ID, "scores")
    first = _scores(config)

    run.run_stages(config, CYCLE_ID, "scores")

    assert len(fake.bodies) == 2
    assert _scores(config) == first  # jev_call_id included: a cached score keeps its call's id
    assert len(_calls_log(config)) == 2


def test_an_abort_writes_no_marker_and_a_rerun_calls_only_what_is_not_cached(config, fake):
    fake.refuse = {"5"}  # p2's call is refused; p1's, in the pool too, is scored and cached

    with pytest.raises(RuntimeError, match="Jev call failed"):
        run.run_stages(config, CYCLE_ID, "scores")
    assert not stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "scores").exists()
    assert len(_calls_log(config)) == 1

    fake.refuse = set()
    fake.bodies.clear()
    run.run_stages(config, CYCLE_ID, "scores")

    # Only p2's call again, whole: a call is cached all or nothing.
    [body] = fake.bodies
    assert sorted(q["instructions"]["answer"] for q in body["questions"].values()) == ["", "5"]
    assert len(_scores(config)) == 6


def test_a_jev_pin_changed_since_the_snapshot_is_refused(config, fake):
    config.prompt_validator.jev.resolved = "jev-1.15.0"

    with pytest.raises(RuntimeError, match="rerun --stage snapshot"):
        run.run_stages(config, CYCLE_ID, "scores")
    assert fake.bodies == []


def test_a_plain_run_runs_every_stage_in_order(config, fake, monkeypatch):
    ran = []
    stages = [(name, (lambda name: lambda cfg, cycle: ran.append(name) or [{}])(name)) for name, _ in run.STAGES]
    monkeypatch.setattr(run, "STAGES", stages)

    run.run_stages(config, CYCLE_ID)

    # With the answers stage in, nothing is left to --stage: scores and reward follow it.
    assert ran == ["cutoff", "snapshot", "fetch", "answers", "scores", "reward"]


def test_scoring_without_an_answers_file_stops(config, fake):
    stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "answers").unlink()

    with pytest.raises(RuntimeError, match="no answers to score"):
        run.run_stages(config, CYCLE_ID, "scores")
    assert fake.bodies == []


@needs_judge_rules
def test_with_the_switch_on_a_selected_miner_prompt_is_scored_on_the_prompt_alone(config, fake, write_selection):
    pv = config.prompt_validator
    pv.judge_miner_prompts = True
    snapshot_dir = stage_path(pv.runs_dir, CYCLE_ID, "snapshot").parent
    write_rows(snapshot_dir / "snapshot.jsonl", [{**BASE, "rules_sha256": snapshot.rules_sha256(pv, BASE)}])
    write_selection(pv, CYCLE_ID, {"m1": "Miner one", "m2": "Miner two"}, ["m1"])
    answers_path = stage_path(pv.runs_dir, CYCLE_ID, "answers")
    write_rows(answers_path, [*read_rows(answers_path), {"uid": 2, "hotkey": "hk2", "model_hash": "h2",
                                                          "prompt_id": "m1", "text": "3", "n_tokens": 1,
                                                          "finish_reason": "stop"}])
    run.run_stages(config, CYCLE_ID, "scores")
    assert [(r["uid"], r["expected_score"]) for r in _scores(config) if r["prompt_id"] == "m1"] == [
        (2, pytest.approx(0.6))]
    # No customer request, as the pool's dataless prompts have none.
    assert [b["state"] for b in fake.bodies if b["state"]["prompt"] == "Miner one"] == [{"prompt": "Miner one"}]
