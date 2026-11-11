import hashlib
import json
import logging
import stat
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import bittensor
import pytest
import requests

from connito.prompt_validator import dataless, jev, report, snapshot
from connito.prompt_validator.config import DatalessCfg, PromptValidatorCfg, PromptValidatorChainCfg
from connito.prompt_validator.stage_io import read_rows, stage_path, write_rows
from connito.shared.config import CycleCfg
from connito.test.test_prompt_validator_cutoff import OWNER_PERIODS

CYCLE_ID = 1258
# The snapshot's base model rows; base_rows rebuilds the same from config unless a test changes them.
BASE = {"base_model": "Qwen/Qwen3.6-35B-A3B", "base_model_revision": "0" * 40, "chat_template_sha256": "1" * 64}


@pytest.fixture(autouse=True)
def fixed_base_rows(monkeypatch):
    monkeypatch.setattr(snapshot, "base_rows", lambda cfg: dict(BASE))  # the real one downloads the chat template
URL = "https://dataless.test"
LENGTH = 6812  # blocks per prompt cycle
CYCLE = CycleCfg(**OWNER_PERIODS)  # 524 blocks per training cycle: 13 of them make LENGTH
REAL_CLIENT = dataless.DatalessClient  # the fixtures replace it on the module


def _content_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _prompt(prompt_id, text, source="internal", task="t-legal", hotkey=None):
    return {"prompt_id": prompt_id, "hotkey": hotkey, "customer_task_id": task, "source": source, "text": text,
            "content_hash": _content_hash(text), "block": 100}


PROMPTS = [
    _prompt(11, "Prompt eleven, a stand-in."),
    _prompt(12, "Prompt twelve, a stand-in."),
    _prompt(20, "Prompt twenty, a miner's stand-in.", source="miner", hotkey="hk-miner-set"),
]


class Answer:
    def __init__(self, status, payload=None, headers=None, raw=None):
        self.status_code, self.headers = status, headers or {}
        self.content = raw if raw is not None else (b"" if payload is None else json.dumps(payload).encode())

    def json(self):
        return json.loads(self.content)


class FakeDataless:
    """The dataless-api as the validator sees it: the cycle view, paged and per-task prompt lists, and the run's
    write-once rows. `fail` maps a path suffix to statuses to answer first; every request is recorded."""

    def __init__(self, prompts=PROMPTS, open_cycle=CYCLE_ID + 1, page_size=2, tasks_required=False, fail=None,
                 heads=None, publish_answer_lost=False):
        self.prompts, self.open_cycle, self.page_size = prompts, open_cycle, page_size
        self.tasks_required, self.fail = tasks_required, fail or {}
        # Each cycle-view read takes the next head; by default 20 blocks into the open cycle, past the delay.
        self.heads = list(heads) if heads else []
        self.head = self.open_cycle * LENGTH + 20
        self.publish_answer_lost = publish_answer_lost
        self.requests, self.run, self.model_ids = [], None, {}
        self.summary, self.model_scores = {}, {}

    def request(self, method, url, data=None, headers=None, timeout=None, allow_redirects=True):
        assert allow_redirects is False
        parts = urlsplit(url)
        body = json.loads(data) if data else None
        self.requests.append({"method": method, "path": parts.path, "query": parts.query, "headers": headers,
                              "data": data, "body": body})
        for suffix, statuses in self.fail.items():
            if parts.path.endswith(suffix) and statuses:
                status = statuses.pop(0)
                return Answer(status, {"detail": {"code": "busy", "message": "try later"}}, {"Retry-After": "0"})
        if parts.path == "/v1/prompt-cycle":
            if self.heads:
                self.head = self.heads.pop(0)
            self.open_cycle = self.head // LENGTH
            start = self.open_cycle * LENGTH
            return Answer(200, {"head": self.head, "cycle_id": self.open_cycle, "start_block": start,
                                "cutoff_block": start + LENGTH - 1, "cycle_length": LENGTH, "tasks": [],
                                "policy": {"rollout_phase": 2, "max_prompts_per_hotkey": 20}})
        assert headers and "X-Bittensor-Signature" in headers, "a validator call must be signed"
        cycle = int(parts.path.split("/")[4])
        if parts.path.endswith("/prompts"):
            return self._list(cycle, parse_qs(parts.query))
        if self.run and self.run["status"] == "published":
            return Answer(409, {"detail": {"code": "cycle_published", "message": "final"}})
        if parts.path.endswith("/status"):
            if body["status"] == "scoring":
                if cycle != self.open_cycle - 1:
                    return Answer(409, {"detail": {"code": "cycle_not_current", "message": "too late",
                                                   "current_cycle_id": self.open_cycle - 1}})
                self.run = {**body, "cycle_id": cycle}
            else:
                self.run["status"] = body["status"]
                if self.publish_answer_lost and body["status"] == "published":
                    self.publish_answer_lost = False
                    return Answer(503, {"detail": {"code": "unavailable", "message": "lost"}}, {"Retry-After": "0"})
            return Answer(200, {**self.run, "validator_hotkey": "5Validator", "block": self.head,
                                "updated_at": "2026-10-19T12:00:00Z"})
        if parts.path.endswith("/models"):
            for row in body["rows"]:
                self.model_ids.setdefault(row["hotkey"], 1000 + len(self.model_ids))
            return Answer(200, {"model_ids": {r["hotkey"]: self.model_ids[r["hotkey"]] for r in body["rows"]}})
        table, key = {"summary": (self.summary, "hotkey"), "model-scores": (self.model_scores, None)}[
            parts.path.rsplit("/", 1)[1]]
        changed = []
        for i, row in enumerate(body["rows"]):
            k = row[key] if key else (row["model_id"], row["prompt_id"])
            if k in table and table[k] != row:
                changed.append(i)
        if changed:
            return Answer(409, {"detail": {"code": "already_written", "message": "differs", "rows": changed}})
        for row in body["rows"]:
            table[row[key] if key else (row["model_id"], row["prompt_id"])] = row
        return Answer(200, {"rows": len(body["rows"])})

    def _list(self, cycle, query):
        if cycle >= self.open_cycle or self.head < (cycle + 1) * LENGTH - 1 + 10:
            return Answer(409, {"detail": {"code": "before_cutoff", "message": "not yet", "head": self.head,
                                           "ready_at_block": (cycle + 1) * LENGTH - 1 + 10}})
        if cycle != self.open_cycle - 1:
            return Answer(409, {"detail": {"code": "cycle_not_current", "message": "no"}})
        tasks = sorted({p["customer_task_id"] for p in self.prompts})
        task = query.get("customer_task_id", [None])[0]
        if self.tasks_required and task is None:
            return Answer(409, {"detail": {"code": "task_required", "message": "pick one", "customer_task_ids": tasks}})
        rows = sorted((p for p in self.prompts if task is None or p["customer_task_id"] == task),
                      key=lambda p: p["prompt_id"])
        start = int(query.get("page_token", ["0"])[0])
        page = rows[start: start + self.page_size]
        token = str(start + self.page_size) if start + self.page_size < len(rows) else None
        return Answer(200, {"cycle_id": cycle, "snapshot_hash": dataless.snapshot_hash(self.prompts),
                            "snapshot_count": len(self.prompts), "prompts": page, "next_page_token": token})


class Signer:
    def __init__(self):
        self.calls = []

    def __call__(self, method, target, body):
        self.calls.append((method, target, body))
        return {"X-Bittensor-Hotkey": "5Validator", "X-Bittensor-Nonce": str(len(self.calls)),
                "X-Bittensor-Signature": "0x" + "ab" * 64, "X-Bittensor-Version": "1"}


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(dataless.time, "sleep", lambda s: None)


def _client(fake, signer=None, **cfg):
    return dataless.DatalessClient(DatalessCfg(url=URL, **cfg), signer or Signer(), session=fake)


# --- the client ---

def test_every_validator_call_is_signed_on_the_exact_path_and_body_and_the_cycle_view_is_not(no_sleep):
    fake, signer = FakeDataless(), Signer()
    client = _client(fake, signer)
    client.prompt_cycle()
    client.start_run(CYCLE_ID, {"rollout_phase": 2})

    view, start = fake.requests
    assert "X-Bittensor-Signature" not in (view["headers"] or {})
    assert signer.calls == [("POST", f"/v1/validator/cycles/{CYCLE_ID}/status", start["data"])]
    assert start["data"] == b'{"status":"scoring","rollout_phase":2}'  # compact JSON: the bytes signed are sent


def test_a_retry_is_signed_afresh_after_retry_after(no_sleep):
    fake, signer = FakeDataless(fail={"/status": [429, 503]}), Signer()
    _client(fake, signer).start_run(CYCLE_ID, {"rollout_phase": 2})
    assert len(fake.requests) == 3 and len(signer.calls) == 3
    assert len({r["headers"]["X-Bittensor-Nonce"] for r in fake.requests}) == 3


def test_retry_after_is_honoured_capped_and_an_http_date_falls_back(monkeypatch):
    sleeps = []
    monkeypatch.setattr(dataless.time, "sleep", sleeps.append)
    fake = FakeDataless(fail={"/status": [429, 429, 429]})
    statuses = iter(["7", "3600", "Wed, 21 Oct 2026 07:28:00 GMT"])
    real = fake.request

    def with_retry_after(*a, **k):
        answer = real(*a, **k)
        if answer.status_code == 429:
            answer.headers = {"Retry-After": next(statuses)}
        return answer

    fake.request = with_retry_after
    _client(fake).start_run(CYCLE_ID, {"rollout_phase": 2})
    assert sleeps == [7.0, 60.0, 4.0]  # as asked, capped at a minute, then the backoff 2 ** 2


def test_a_redirect_is_an_error_not_a_success(no_sleep):
    fake = FakeDataless()
    fake.request = lambda *a, **k: Answer(307, raw=b"")
    with pytest.raises(dataless.DatalessError, match="307"):
        _client(fake).start_run(CYCLE_ID, {})


def test_the_snapshot_count_is_checked_as_well_as_the_hash():
    fake = FakeDataless()
    real = fake._list

    def miscounted(cycle, query):
        page = real(cycle, query).json()
        page["snapshot_count"] += 1
        return Answer(200, page)

    fake._list = miscounted
    with pytest.raises(RuntimeError, match="the server says 4"):
        _client(fake).list_prompts(CYCLE_ID)


def test_the_url_must_be_https_except_on_this_machine():
    assert DatalessCfg(url="http://127.0.0.1:8000").url == "http://127.0.0.1:8000"
    for bad in ("http://dataless-api.connito.ai", "http://localhost.evil.example", "http://127.0.0.1@evil.example/",
                "https://", "ftp://dataless-api.connito.ai"):
        with pytest.raises(ValueError, match="must be https"):
            DatalessCfg(url=bad)


def test_an_error_keeps_the_servers_code_and_extra_and_no_body_is_logged(no_sleep, caplog):
    fake = FakeDataless(fail={"/status": [409]})
    with caplog.at_level(logging.DEBUG), pytest.raises(dataless.DatalessError) as e:
        _client(fake, retries=0).start_run(CYCLE_ID, {"rollout_phase": 2, "secret": "a private prompt"})
    assert (e.value.status, e.value.code) == (409, "busy")
    assert "a private prompt" not in caplog.text


def test_an_unreachable_api_is_retried_then_named_without_the_query(no_sleep, monkeypatch):
    warned = []
    monkeypatch.setattr(dataless.logger, "warning", lambda event, **kw: warned.append(kw))

    class Down:
        calls = 0

        def request(self, *a, **k):
            Down.calls += 1
            raise requests.ConnectionError("refused")

    with pytest.raises(RuntimeError, match=r"GET /v1/validator/cycles/1258/prompts failed: ConnectionError"):
        _client(Down(), retries=2).list_prompts(CYCLE_ID)
    assert Down.calls == 3
    assert [(w["route"], w["attempt"]) for w in warned] == [(f"/v1/validator/cycles/{CYCLE_ID}/prompts", n)
                                                            for n in (1, 2)]


def test_a_validator_call_without_a_signer_stops():
    client = dataless.DatalessClient(DatalessCfg(url=URL), None, session=FakeDataless())
    with pytest.raises(RuntimeError, match="no signer"):
        client.start_run(CYCLE_ID, {})


def test_list_prompts_reads_every_page_and_checks_the_snapshot_hash():
    fake = FakeDataless(page_size=1)
    listing = _client(fake).list_prompts(CYCLE_ID)
    assert [p["prompt_id"] for p in listing["prompts"]] == [11, 12, 20]
    assert listing["snapshot_count"] == 3 and listing["snapshot_hash"] == dataless.snapshot_hash(PROMPTS)
    assert [r["query"] for r in fake.requests] == ["", "page_token=1", "page_token=2"]


def test_with_several_tasks_each_task_is_listed_and_the_hash_covers_all():
    prompts = [*PROMPTS, _prompt(30, "Prompt thirty, a stand-in.", task="t-tax")]
    fake = FakeDataless(prompts=prompts, tasks_required=True)
    listing = _client(fake).list_prompts(CYCLE_ID)
    assert sorted(p["prompt_id"] for p in listing["prompts"]) == [11, 12, 20, 30]
    assert "customer_task_id=t-tax" in {r["query"] for r in fake.requests}


def test_a_snapshot_that_does_not_hash_to_the_servers_is_refused():
    fake = FakeDataless()
    real = fake._list
    fake._list = lambda cycle, query: _drop_last(real(cycle, query))
    with pytest.raises(RuntimeError, match="snapshot does not check out: 1 prompts"):
        _client(fake).list_prompts(CYCLE_ID)


def _drop_last(answer):
    page = answer.json()
    page["prompts"] = page["prompts"][:-1]
    page["next_page_token"] = None
    return Answer(200, page)


def test_a_prompt_whose_text_does_not_match_its_content_hash_is_refused():
    tampered = [{**PROMPTS[0], "text": "changed"}, *PROMPTS[1:]]
    with pytest.raises(RuntimeError, match="prompt 11: its text does not match its content_hash"):
        _client(FakeDataless(prompts=tampered)).list_prompts(CYCLE_ID)


def test_snapshot_hash_is_id_8_bytes_big_endian_then_content_hash_in_id_order():
    rows = [{"prompt_id": 2, "content_hash": "22" * 32}, {"prompt_id": 1, "content_hash": "11" * 32}]
    want = hashlib.sha256((1).to_bytes(8, "big") + bytes.fromhex("11" * 32)
                          + (2).to_bytes(8, "big") + bytes.fromhex("22" * 32)).hexdigest()
    assert dataless.snapshot_hash(rows) == want


def test_rows_are_chunked_by_count_and_by_body_size(monkeypatch):
    assert [len(b) for b in dataless.chunk_rows([{"i": i} for i in range(2500)], lambda b: {"rows": b})] == [
        1000, 1000, 500]
    monkeypatch.setattr(dataless, "BODY_CAP", 100)
    batches = dataless.chunk_rows([{"t": "x" * 30} for _ in range(5)], lambda b: {"rows": b})
    assert all(len(dataless.encode_body({"rows": b})) <= 100 for b in batches) and sum(map(len, batches)) == 5
    with pytest.raises(ValueError, match="over the 100-byte body cap"):
        dataless.chunk_rows([{"t": "x" * 200}], lambda b: {"rows": b})


# --- the config and the snapshot ---

def test_scoring_cycles_are_13_training_cycles_and_a_file_source_needs_its_path(tmp_path):
    pv = PromptValidatorCfg(prompt_source="dataless")
    assert (pv.scoring_cycle_k, pv.prompts_path, pv.dataless.url) == (13, None, "https://dataless-api.connito.ai")
    with pytest.raises(ValueError, match='prompts_path is required with prompt_source "file"'):
        PromptValidatorCfg()


@pytest.fixture
def snapshot_api(monkeypatch):
    def serve(**kwargs):
        fake = FakeDataless(**kwargs)
        monkeypatch.setattr(snapshot.dataless, "DatalessClient",
                            lambda cfg, signer: REAL_CLIENT(cfg, Signer(), session=fake))
        monkeypatch.setattr(snapshot.dataless, "wallet_signer", lambda wallet: None)
        monkeypatch.setattr(bittensor, "Wallet", lambda **kw: None)
        fake.sleeps = []
        monkeypatch.setattr(dataless.time, "sleep", fake.sleeps.append)
        monkeypatch.setattr(snapshot.time, "sleep", fake.sleeps.append)
        return fake

    return serve


def _config(tmp_path):
    pv = PromptValidatorCfg(runs_dir=tmp_path / "runs", prompt_source="dataless", dataless=DatalessCfg(url=URL),
                            write_root=tmp_path)
    chain = PromptValidatorChainCfg(coldkey_name="c", hotkey_name="h")
    return SimpleNamespace(prompt_validator=pv, chain=chain, cycle=CYCLE)


def test_the_snapshot_keeps_the_owner_prompts_with_no_request(tmp_path, snapshot_api):
    snapshot_api()
    prompts, meta, miner, request = snapshot.load_prompts(_config(tmp_path), CYCLE_ID)
    assert prompts == [
        {"prompt_id": "11", "request_id": "t-legal", "prompt": PROMPTS[0]["text"], "request": None,
         "customer_task_id": "t-legal", "content_hash": PROMPTS[0]["content_hash"]},
        {"prompt_id": "12", "request_id": "t-legal", "prompt": PROMPTS[1]["text"], "request": None,
         "customer_task_id": "t-legal", "content_hash": PROMPTS[1]["content_hash"]},
    ]
    assert meta == {"prompt_source": "dataless", "dataless_snapshot_hash": dataless.snapshot_hash(PROMPTS),
                    "dataless_snapshot_count": 3, "rollout_phase": 2, "dataless_set_hotkeys": ["hk-miner-set"]}
    assert (miner, request) == (None, None)  # kept only with judge_miner_prompts


def test_the_snapshot_waits_for_the_prompt_cycle_to_close_and_be_listed(tmp_path, snapshot_api):
    # Right after cutoff the cycle is still open; then it has closed but is inside the snapshot delay.
    cutoff = (CYCLE_ID + 1) * LENGTH - 1
    fake = snapshot_api(heads=[cutoff - 142, cutoff + 4, cutoff + 10])
    prompts = snapshot.load_prompts(_config(tmp_path), CYCLE_ID).prompts
    assert len(prompts) == 2
    assert fake.sleeps == [600.0, 72.0]  # 152 blocks capped at 10 minutes, then 6 blocks of 12 s


def test_a_longer_server_delay_is_waited_out_on_its_ready_block(tmp_path, snapshot_api):
    cutoff = (CYCLE_ID + 1) * LENGTH - 1
    fake = snapshot_api(heads=[cutoff + 10, cutoff + 10, cutoff + 12])
    real = fake._list
    calls = []

    def later(cycle, query):  # the server asks for 2 more blocks than we wait
        calls.append(1)
        if len(calls) == 1:
            return Answer(409, {"detail": {"code": "before_cutoff", "message": "not yet", "head": cutoff + 10,
                                           "ready_at_block": cutoff + 12}})
        return real(cycle, query)

    fake._list = later
    prompts = snapshot.load_prompts(_config(tmp_path), CYCLE_ID).prompts
    assert len(prompts) == 2 and fake.sleeps == [24.0]


def test_a_prompt_cycle_the_api_has_moved_past_stops_the_snapshot(tmp_path, snapshot_api):
    snapshot_api(heads=[(CYCLE_ID + 2) * LENGTH + 20])
    with pytest.raises(RuntimeError, match="lists prompt cycle 1259 now, not 1258"):
        snapshot.load_prompts(_config(tmp_path), CYCLE_ID)


@pytest.mark.parametrize("change", [
    {"start_block": (CYCLE_ID + 1) * LENGTH - 100},  # a grid with another origin
    {"cycle_length": LENGTH - 524},  # 12 training cycles a cycle
    {"cutoff_block": (CYCLE_ID + 2) * LENGTH - 1 - 524},  # a cutoff a training cycle before the cycle's end
])
def test_a_prompt_cycle_off_the_scoring_cycle_grid_stops_the_snapshot_before_listing(tmp_path, snapshot_api, change):
    fake = snapshot_api()
    real = fake.request

    def off_grid(method, url, **kwargs):
        answer = real(method, url, **kwargs)
        if urlsplit(url).path == "/v1/prompt-cycle":
            answer = Answer(200, {**answer.json(), **change})
        return answer

    fake.request = off_grid
    with pytest.raises(RuntimeError, match="a scoring cycle here is 6812 blocks from block 0, cut off at its last"):
        snapshot.load_prompts(_config(tmp_path), CYCLE_ID)
    assert not [r for r in fake.requests if r["path"].endswith("/prompts")]


def test_a_cycle_with_no_owner_prompts_stops_the_snapshot(tmp_path, snapshot_api):
    snapshot_api(prompts=[PROMPTS[2]])
    with pytest.raises(RuntimeError, match="no internal prompts for prompt cycle 1258"):
        snapshot.load_prompts(_config(tmp_path), CYCLE_ID)


def test_a_file_source_reads_the_private_prompt_file(tmp_path):
    path = tmp_path / "p.jsonl"
    row = {"prompt_id": "p1", "request_id": "r1", "prompt": "Hi", "request": {"goal": "g", "examples": [], "tone": "t"}}
    path.write_text(json.dumps(row) + "\n")
    config = SimpleNamespace(prompt_validator=PromptValidatorCfg(prompts_path=path))
    assert snapshot.load_prompts(config, CYCLE_ID) == ([row], {}, None, None)


# --- Jev without a customer request ---

def test_a_prompt_with_no_request_gets_the_question_without_the_customer():
    state = jev.score_state(None, "prompt one")
    assert state == {"prompt": "prompt one"}
    question = jev.score_question("an answer", state)
    assert question["instructions"]["question"] == jev.QUALITY_QUESTION_NO_REQUEST
    assert question["criteria"] == jev.SCORE_LEVELS_NO_REQUEST
    assert "customer" not in jev.QUALITY_QUESTION_NO_REQUEST and not any(
        "customer" in level for level in jev.SCORE_LEVELS_NO_REQUEST)
    with_request = jev.score_question("an answer", jev.score_state({"goal": "g", "examples": [], "tone": "t"}, "p"))
    assert with_request["instructions"]["question"] == jev.QUALITY_QUESTION


def test_both_questions_are_in_the_rules_hash(monkeypatch):
    pv = PromptValidatorCfg(prompt_source="dataless")
    before = snapshot.rules_sha256(pv, BASE)
    monkeypatch.setitem(snapshot.SCORE_QUESTIONS["quality_no_request"], "instructions", "another question")
    assert snapshot.rules_sha256(pv, BASE) != before


def test_the_levels_by_task_are_in_the_rules_hash(monkeypatch):
    pv = PromptValidatorCfg(prompt_source="dataless")
    before = snapshot.rules_sha256(pv, BASE)
    monkeypatch.setitem(snapshot.CRITERIA_BY_TASK, "exp_lesson_residual", ["0 Bad", "1 Good"])
    assert snapshot.rules_sha256(pv, BASE) != before


def test_the_format_rules_by_task_are_in_the_rules_hash(monkeypatch):
    pv = PromptValidatorCfg(prompt_source="dataless")
    before = snapshot.rules_sha256(pv, BASE)
    monkeypatch.setitem(snapshot.FORMAT_RULES, "exp_lesson_residual", "exactly 4 lessons")
    assert snapshot.rules_sha256(pv, BASE) != before


# --- the report ---

MODELS = [
    {"uid": 1, "hotkey": "hk1", "hf_repo_id": "miner/one", "hf_revision": "a" * 40, "model_hash": "1" * 48},
    {"uid": 2, "hotkey": "hk2", "hf_repo_id": "miner/two", "hf_revision": "b" * 40, "model_hash": "2" * 64},
    {"uid": 3, "hotkey": "hk-miner-set", "hf_repo_id": "miner/three", "hf_revision": "c" * 40,
     "model_hash": "3" * 64},
    {"uid": 4, "hotkey": "hk4", "hf_repo_id": "miner/four", "hf_revision": "main", "model_hash": "4" * 64},
]


@pytest.fixture
def cycle(tmp_path, monkeypatch, no_sleep):
    """A scoring cycle up to reward, as the stages before report leave it, and the fake API it reports to."""
    config = _config(tmp_path)
    pv = config.prompt_validator
    config.chain.coldkey_name, config.chain.hotkey_name = "c", "h"
    snap = stage_path(pv.runs_dir, CYCLE_ID, "snapshot").parent
    write_rows(snap / "models.jsonl", MODELS)
    write_rows(snap / "snapshot.jsonl", [{
        **BASE, "rules_sha256": snapshot.rules_sha256(pv, BASE), "active_task": "legal",
        "commit_2_block": 9235881, "prompt_source": "dataless", "dataless_snapshot_hash": "f" * 64,
        "dataless_snapshot_count": 3, "rollout_phase": 2, "dataless_set_hotkeys": ["hk-miner-set"],
    }])
    write_rows(stage_path(pv.runs_dir, CYCLE_ID, "fetch"), [
        {"uid": 1, "hotkey": "hk1", "path": "/m1", "reason": None},
        {"uid": 2, "hotkey": "hk2", "path": None, "reason": "hash"},
        {"uid": 3, "hotkey": "hk-miner-set", "path": "/m3", "reason": None},
        {"uid": 4, "hotkey": "hk4", "path": "/m4", "reason": None},
    ])
    write_rows(stage_path(pv.runs_dir, CYCLE_ID, "answers"), [
        {"uid": uid, "hotkey": hk, "model_hash": "", "prompt_id": p, "text": f"answer {uid} to {p}"}
        for uid, hk in ((1, "hk1"), (3, "hk-miner-set")) for p in ("11", "12")
    ])
    write_rows(stage_path(pv.runs_dir, CYCLE_ID, "scores"), [
        {"uid": uid, "hotkey": hk, "model_hash": "", "prompt_id": p, "score": 0.8, "expected_score": e,
         "rank": 0, "jev_call_id": "c"}
        for uid, hk, e in ((1, "hk1", 0.9), (3, "hk-miner-set", 0.7)) for p in ("11", "12")
    ])
    write_rows(stage_path(pv.runs_dir, CYCLE_ID, "reward"), [
        {"rank": 0, "uid": 1, "hotkey": "hk1", "model_hash": "", "n_scored": 2, "qualified": True,
         "mean_score": 0.9, "weight": 1.0},
        {"rank": None, "uid": 2, "hotkey": "hk2", "model_hash": "", "n_scored": 0, "qualified": False,
         "mean_score": 0.0, "weight": 0.0},
        {"rank": 1, "uid": 3, "hotkey": "hk-miner-set", "model_hash": "", "n_scored": 2, "qualified": True,
         "mean_score": 0.7, "weight": 0.0},
    ])
    fake = FakeDataless()
    monkeypatch.setattr(report.dataless, "DatalessClient",
                        lambda cfg, signer: REAL_CLIENT(cfg, Signer(), session=fake))
    monkeypatch.setattr(report.dataless, "wallet_signer", lambda wallet: None)
    monkeypatch.setattr(bittensor, "Wallet", lambda **kw: None)
    return config, fake


def test_report_posts_the_run_models_summary_scores_with_outputs_then_publishes(cycle):
    config, fake = cycle
    rows = report.run_report(config, CYCLE_ID)

    paths = [r["path"].rsplit("/", 1)[1] for r in fake.requests]
    assert paths == ["status", "models", "summary", "model-scores", "status"]
    assert fake.run == {"status": "published", "cycle_id": CYCLE_ID, "rollout_phase": 2, "snapshot_hash": "f" * 64,
                        "jev_version": "jev-1.14.0", "validator_code_version": "prompt-validator",
                        "rules_sha256": snapshot.rules_sha256(config.prompt_validator, BASE),
                        "top_k": 0, "max_tokens": 1024, "seed": 0}
    models = fake.requests[1]["body"]["rows"]
    assert [(m["uid"], m["fetch_result"]) for m in models] == [(1, "ok"), (2, "hash_mismatch"), (3, "ok")]
    assert models[0] == {"hotkey": "hk1", "uid": 1, "training_task_id": "legal", "hf_repo": "miner/one",
                         "revision": "a" * 40, "model_hash": "1" * 48, "commit_block": 9235881, "fetch_result": "ok"}
    # uid 3 sent a prompt set, which nothing here judges, so the API would take no summary for it.
    assert list(fake.summary) == ["hk1", "hk2"]
    assert fake.summary["hk2"]["model_score"] is None and fake.summary["hk1"]["model_score"] == 0.9
    scores = fake.model_scores
    # The server keeps outputs for max(ceil(3/10), 10) = all 3 models, uid 3's too, though it has no summary row.
    assert {k: (r["jev_score"], r.get("output")) for k, r in scores.items()} == {
        (1000, 11): (0.9, "answer 1 to 11"), (1000, 12): (0.9, "answer 1 to 12"),
        (1002, 11): (0.7, "answer 3 to 11"), (1002, 12): (0.7, "answer 3 to 12"),
    }
    assert rows == [{"cycle_id": CYCLE_ID, "status": "published", "models": 3, "summary": 2, "model_scores": 4,
                     "outputs": 4}]
    assert fake.requests[-2]["headers"]["Content-Type"] == "application/json"


def test_a_model_whose_commit_the_api_cannot_take_is_left_out_and_said(cycle, caplog):
    config, fake = cycle
    with caplog.at_level(logging.WARNING):
        report.run_report(config, CYCLE_ID)
    assert "hk4" not in fake.model_ids and "model not reported" in caplog.text and "revision" in caplog.text


def test_a_report_stopped_part_way_is_rerun_to_the_end(cycle):
    config, fake = cycle
    fake.fail = {"/model-scores": [500, 500, 500, 500]}
    with pytest.raises(dataless.DatalessError, match="500"):
        report.run_report(config, CYCLE_ID)
    assert fake.run["status"] == "scoring"
    fake.fail = {}
    assert report.run_report(config, CYCLE_ID)[0]["status"] == "published"  # the re-posted rows are no-ops


def test_a_report_whose_run_start_is_now_too_late_goes_on_with_the_run(cycle):
    config, fake = cycle
    fake.fail = {"/model-scores": [500, 500, 500, 500]}
    with pytest.raises(dataless.DatalessError):
        report.run_report(config, CYCLE_ID)
    fake.fail, fake.head = {}, (CYCLE_ID + 2) * LENGTH + 20  # the next prompt cycle has closed
    fake.open_cycle = CYCLE_ID + 2
    assert report.run_report(config, CYCLE_ID)[0]["status"] == "published"


def test_a_late_rerun_of_a_cycle_published_elsewhere_is_recorded_as_done(cycle, caplog):
    config, fake = cycle
    report.run_report(config, CYCLE_ID)
    fake.open_cycle = CYCLE_ID + 2  # the next cycle closed: start answers cycle_not_current before "published"
    real = fake.request

    def late(method, url, **kw):
        if url.endswith("/status") and json.loads(kw["data"])["status"] == "scoring":
            return Answer(409, {"detail": {"code": "cycle_not_current", "message": "late"}})
        return real(method, url, **kw)

    fake.request = late
    ledger = config.prompt_validator.runs_dir / "published.jsonl"
    ledger.unlink()  # as from another runs_dir or host
    sent = len(fake.requests)
    with caplog.at_level(logging.WARNING):
        assert report.run_report(config, CYCLE_ID) == [{"cycle_id": CYCLE_ID, "status": "published",
                                                        "published_before": "server"}]
    assert "already published on the dataless-api; recorded as done, not reported again" in caplog.text
    # The late run start is answered before the fake records it; the models post is refused: nothing more is sent.
    assert [r["path"].rsplit("/", 1)[1] for r in fake.requests[sent:]] == ["models"]
    [row] = read_rows(ledger)
    assert row["source"] == "server" and row["validator_hotkey"] is None


def test_a_run_never_started_in_time_is_named_as_such(cycle):
    config, fake = cycle
    fake.open_cycle = CYCLE_ID + 2
    real = fake.request

    def missing(method, url, **kw):
        if url.endswith("/models"):
            return Answer(409, {"detail": {"code": "cycle_run_missing", "message": "no run"}})
        return real(method, url, **kw)

    fake.request = missing
    with pytest.raises(RuntimeError, match="no run was started for prompt cycle 1258 before the next one closed"):
        report.run_report(config, CYCLE_ID)


def test_a_publish_whose_answer_was_lost_counts_as_published(cycle):
    config, fake = cycle
    fake.publish_answer_lost = True
    assert report.run_report(config, CYCLE_ID)[0]["status"] == "published"
    [row] = read_rows(config.prompt_validator.runs_dir / "published.jsonl")
    assert row["source"] == "this_run" and row["published_at"]  # the local time: the lost answer had the server's


def test_a_cycle_with_no_model_to_report_is_not_published(cycle, monkeypatch):
    config, fake = cycle
    monkeypatch.setattr(report, "model_rows", lambda meta, models, fetch: [])
    with pytest.raises(RuntimeError, match="no model of scoring cycle 1258 can be reported"):
        report.run_report(config, CYCLE_ID)
    assert fake.requests == []


def test_fetch_results_map_to_the_apis_and_a_missing_row_is_invalid():
    meta = {"active_task": "legal", "commit_2_block": 1}
    models = [{**MODELS[0], "uid": uid, "hotkey": f"hk{uid}"} for uid in range(6)]
    fetch = [{"uid": uid, "hotkey": f"hk{uid}", "reason": reason} for uid, reason in
             enumerate([None, "signature", "repo_unavailable", "timeout", "bad_residual_shard"])]
    assert [r["fetch_result"] for r in report.model_rows(meta, models, fetch)] == [
        "ok", "hash_mismatch", "not_found", "timeout", "invalid", "invalid"]


@pytest.mark.parametrize("repo", ["miner/one two", "-miner/one", "miner/" + "x" * 97, "a/b/c", "miner/one!"])
def test_a_repo_name_the_server_would_refuse_leaves_that_model_out(repo):
    meta = {"active_task": "legal", "commit_2_block": 1}
    assert report.model_rows(meta, [{**MODELS[0], "hf_repo_id": repo}], []) == []
    assert len(report.model_rows(meta, [{**MODELS[0], "hf_repo_id": "Miner_1/one.v2-x"}], [])) == 1


@pytest.mark.parametrize("model_hash", ["1" * 40, "1" * 56, "1" * 66, "A" * 48, "0x" + "1" * 48, "g" * 48])
def test_a_model_hash_the_server_would_refuse_leaves_that_model_out(model_hash):
    meta = {"active_task": "legal", "commit_2_block": 1}
    assert report.model_rows(meta, [{**MODELS[0], "model_hash": model_hash}], []) == []


@pytest.mark.parametrize("model_hash", ["2c97128f7190" + "0" * 36, "1" * 64])
def test_a_miners_24_byte_model_hash_is_reported_and_32_bytes_too(model_hash):
    meta = {"active_task": "legal", "commit_2_block": 1}
    [row] = report.model_rows(meta, [{**MODELS[0], "model_hash": model_hash}], [])
    assert row["model_hash"] == model_hash


def test_a_published_cycle_is_not_reported_again(cycle):
    config, fake = cycle
    report.run_report(config, CYCLE_ID)
    sent = len(fake.requests)
    assert report.run_report(config, CYCLE_ID) == [{"cycle_id": CYCLE_ID, "status": "published",
                                                    "published_before": "ledger"}]
    assert len(fake.requests) == sent  # from the ledger: not asked again


def test_a_cycle_published_elsewhere_is_recorded_as_done_and_nothing_is_posted(cycle):
    config, fake = cycle
    report.run_report(config, CYCLE_ID)
    (config.prompt_validator.runs_dir / "published.jsonl").unlink()  # a new install, a moved runs_dir, another host
    sent = len(fake.requests)
    assert report.run_report(config, CYCLE_ID)[0]["published_before"] == "server"
    assert [r["path"].rsplit("/", 1)[1] for r in fake.requests[sent:]] == ["status"]  # the run start, refused
    assert report.run_report(config, CYCLE_ID)[0]["published_before"] == "ledger"
    assert len(fake.requests) == sent + 1


def test_a_publish_is_appended_to_the_ledger_with_the_run_it_published_and_no_secret(cycle, monkeypatch):
    config, fake = cycle
    pv = config.prompt_validator
    monkeypatch.setenv("JEV_DIRECT_API_KEY", "jev-secret")
    report.run_report(config, CYCLE_ID)
    ledger = pv.runs_dir / "published.jsonl"
    assert read_rows(ledger) == [{
        "cycle_id": CYCLE_ID, "test": False, "source": "this_run", "published_at": "2026-10-19T12:00:00Z",
        "snapshot_sha256": None, "rules_sha256": snapshot.rules_sha256(pv, BASE), "dataless_snapshot_hash": "f" * 64,
        "validator_hotkey": "5Validator", "run_block": fake.head, "url": pv.dataless.url,
    }]
    assert stat.S_IMODE(ledger.stat().st_mode) == 0o600 and "secret" not in ledger.read_text()


def test_an_output_loses_its_nul_characters_which_the_server_refuses():
    assert report._output("a\x00b") == "ab"


def test_an_output_over_the_cap_is_cut_at_a_character_boundary():
    text = "é" * (dataless.OUTPUT_MAX_BYTES // 2 + 3)
    cut = report._output(text)
    assert len(cut.encode()) <= dataless.OUTPUT_MAX_BYTES and text.startswith(cut)
    assert report._output("short") == "short"


def test_outputs_are_kept_for_the_top_models_by_model_score_as_the_server_keeps_them():
    assert [report.keep_count(n) for n in (3, 10, 11, 120)] == [3, 10, 10, 12]
    # 150 models, 40 of them without a summary row: the server keeps 15, the best 15 by model_score.
    hotkeys = {f"h{i:03d}" for i in range(150)}
    summary = [{"hotkey": f"h{i:03d}", "model_score": float(i)} for i in range(110)]
    kept = report.kept_hotkeys(hotkeys, summary)
    assert kept == {f"h{i:03d}" for i in range(95, 110)}
    # Ties by hotkey bytes; a null score comes after every number.
    tied = [{"hotkey": h, "model_score": 1.0} for h in ("b", "a")] + [{"hotkey": "c", "model_score": None}]
    assert report.kept_hotkeys({"a", "b", "c"}, tied) == {"a", "b", "c"}
    many = {f"x{i:02d}" for i in range(11)}
    assert report.kept_hotkeys(many, [{"hotkey": h, "model_score": 1.0} for h in many]) == many - {"x10"}


def test_only_a_dataless_snapshot_is_reported(cycle):
    config, _ = cycle
    path = stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "snapshot")
    meta = read_rows(path)[0]
    write_rows(path, [{**meta, "prompt_source": "file"}])
    with pytest.raises(RuntimeError, match='only a snapshot taken with prompt_source "dataless"'):
        report.run_report(config, CYCLE_ID)


def test_report_needs_every_earlier_stage(cycle):
    config, fake = cycle
    stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "reward").unlink()
    with pytest.raises(RuntimeError, match="nothing to report yet: no reward"):
        report.run_report(config, CYCLE_ID)
    assert fake.requests == []


def _log_levels(monkeypatch):
    logged = []
    for level in ("debug", "info", "warning"):
        monkeypatch.setattr(dataless.logger, level,
                            lambda event, level=level, **kw: logged.append((level, event, kw)))
    return logged


def test_each_request_is_logged_with_its_route_status_rows_and_attempt_and_nothing_it_carries(no_sleep, monkeypatch):
    logged = _log_levels(monkeypatch)
    fake = FakeDataless(fail={"/summary": [503], "/prompt-cycle": [503]})
    client = _client(fake)
    client.list_prompts(CYCLE_ID)  # two pages: the second asks with a page token
    client.prompt_cycle()  # a read that is retried
    client.post_summary(CYCLE_ID, [{"hotkey": "hk1", "uid": 1, "model_score": 1.0, "model_weight": 0.5,
                                    "note": "a private answer"}, {"hotkey": "hk2", "uid": 2}])
    client.post_models(CYCLE_ID, [{"hotkey": "hk1"}])

    requests_logged = [(level, kw) for level, event, kw in logged if event == "dataless-api request"]
    prompts = f"/v1/validator/cycles/{CYCLE_ID}/prompts"
    summary = f"/v1/validator/cycles/{CYCLE_ID}/summary"
    # Reads that went through first time are debug; every write, failure and retry is info.
    assert requests_logged == [
        ("debug", {"method": "GET", "route": prompts, "status": 200, "attempt": 1}),
        ("debug", {"method": "GET", "route": prompts, "status": 200, "attempt": 1}),
        ("info", {"method": "GET", "route": "/v1/prompt-cycle", "status": 503, "attempt": 1}),
        ("info", {"method": "GET", "route": "/v1/prompt-cycle", "status": 200, "attempt": 2}),
        ("info", {"method": "POST", "route": summary, "status": 503, "attempt": 1, "rows": 2}),
        ("info", {"method": "POST", "route": summary, "status": 200, "attempt": 2, "rows": 2}),
        ("info", {"method": "POST", "route": f"/v1/validator/cycles/{CYCLE_ID}/models", "status": 200,
                  "attempt": 1, "rows": 1}),
    ]
    assert len(requests_logged) == len(fake.requests)  # one line per request sent
    assert any(r["query"] for r in fake.requests)  # the page token, left out of the route
    text = repr(logged)
    for private in ["a private answer", "page_token", "X-Bittensor", "ab" * 64, *(p["text"] for p in PROMPTS)]:
        assert private not in text


@pytest.mark.parametrize(("test_mode", "level"), [(True, "debug"), (False, "info")])
def test_only_test_modes_404_while_no_test_cycle_is_open_is_a_debug_line(no_sleep, monkeypatch, test_mode, level):
    logged = _log_levels(monkeypatch)
    with pytest.raises(dataless.DatalessError):
        _client(FakeDataless(fail={"/prompt-cycle": [404]}), test_mode=test_mode).prompt_cycle()
    assert [(lvl, kw["status"]) for lvl, event, kw in logged if event == "dataless-api request"] == [(level, 404)]
