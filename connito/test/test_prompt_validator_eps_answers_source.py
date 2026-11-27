"""eps_error's answers_source: the pinned file (the default, hashed as before) or the answers prompt-record serves
with each drawn pool prompt on the validator's list, taken by the snapshot stage and pinned in its meta."""
import hashlib
import json
from urllib.parse import urlsplit

import pytest
from pydantic import ValidationError
from test_prompt_validator_scores import BASE as SCORES_BASE
from test_prompt_validator_scores import CYCLE_ID, _scores, config, fake  # noqa: F401 (fixtures)

from connito.prompt_validator import dataless, run, snapshot
from connito.prompt_validator.config import PromptValidatorCfg, ScoringMethodCfg
from connito.prompt_validator.methods import REGISTRY, hashed
from connito.prompt_validator.methods import eps_error as eps
from connito.prompt_validator.methods.eps_error import AnswersFileError, answers_file_name
from connito.prompt_validator.stage_io import read_rows, stage_path, write_rows
from connito.test.test_prompt_validator_dataless import (
    REAL_CLIENT,
    Answer,
    FakeDataless,
    Signer,
    _config,
    _prompt,
    fixed_base_rows,  # noqa: F401 (fixture)
    snapshot_api,  # noqa: F401 (fixture)
)

TASK = "earnings_forecast"


def _answer(value):
    """An answer as prompt-record's earnings_forecast pool file holds it (prompts/earnings_forecast/build.py): the
    actual eps_pct_change among the record's other fields, made up here."""
    return {"announcement_date": "2026-01-01", "cik": "0000000001", "company": "Example Co",
            "earnings_event_id": "ex-1", "eps_concept": "EarningsPerShareDiluted", "eps_diluted": 1.0,
            "eps_pct_change": value, "news_articles": [], "news_provider": "none", "q4_derived": False,
            "quarter_end": "2025-12-31", "quarter_start": "2025-10-01", "revenue_yoy_changes": [],
            "sec_accession": "0000000001-26-000001", "sector": "Example", "ticker": "EXC",
            "year_ago_eps_diluted": 2.0, "year_ago_quarter_end": "2024-12-31"}


def _served(prompt_id, text, value):
    """A drawn pool prompt as GET /v1/validator/cycles/{id}/prompts serves it since prompt-record 0027."""
    return {**_prompt(prompt_id, text, task=TASK), "answer": value if value is None else _answer(value)}


SERVED = [_served(12, "Headlines for the Company, quarter two.", 5.0),
          _served(11, "Headlines for the Company, quarter one.", -41.22),
          {**_prompt(20, "A miner's prompt.", source="miner", task=TASK, hotkey="hk-miner-set"), "answer": None}]


def _source_cfg(**settings) -> ScoringMethodCfg:
    return ScoringMethodCfg(method="eps_error", settings={"answers_source": "prompt_record", **settings})


# --- settings and the rules hash ---

def test_file_is_the_default_and_each_source_takes_only_its_own_settings():
    assert ScoringMethodCfg(method="eps_error", settings={"answers_path": "/a.jsonl", "answers_sha256": "a" * 64}
                            ).parsed().answers_source == "file"
    assert _source_cfg().parsed().answers_source == "prompt_record"
    for settings in ({}, {"answers_path": "/a.jsonl"}, {"answers_sha256": "a" * 64},  # file needs both
                     {"answers_source": "prompt_record", "answers_sha256": "a" * 64},  # prompt_record takes neither
                     {"answers_source": "prompt_record", "answers_path": "/a.jsonl"},
                     {"answers_source": "api"}):
        with pytest.raises(ValidationError):
            ScoringMethodCfg(method="eps_error", settings=settings)


def test_file_mode_hashes_exactly_as_before_answers_source_existed():
    settings = ScoringMethodCfg(method="eps_error", settings={"answers_path": "/a.jsonl", "answers_sha256": "a" * 64,
                                                              "answers_source": "file"}).parsed()
    # What hashed() gave for eps_error before: its settings bar answers_path, and RULES.
    assert hashed("eps_error", settings) == {"method": "eps_error", "settings": {"answers_sha256": "a" * 64},
                                             "rules": eps.EpsError.RULES}


def test_prompt_record_mode_hashes_its_source_and_its_own_rules_text(tmp_path):
    def rules(**scoring_methods):
        pv = PromptValidatorCfg(runs_dir=tmp_path / "runs", prompts_path=tmp_path / "p.jsonl")
        pv.scoring_methods = scoring_methods
        return snapshot.rules_sha256(pv, SCORES_BASE)

    entry = hashed("eps_error", _source_cfg().parsed())
    assert entry["settings"] == {"answers_source": "prompt_record"}
    assert "the answer prompt-record serves with the drawn prompt" in entry["rules"]
    assert "pinned file" not in entry["rules"]
    from_file = rules(earnings_forecast=ScoringMethodCfg(
        method="eps_error", settings={"answers_path": "/a.jsonl", "answers_sha256": "a" * 64}))
    assert rules(earnings_forecast=_source_cfg()) not in (from_file, rules())


# --- the snapshot stage ---

def _source_config(tmp_path):
    cfg = _config(tmp_path)
    cfg.prompt_validator.scoring_methods = {TASK: _source_cfg()}
    return cfg


def _loaded(tmp_path, snapshot_api, served=SERVED):  # noqa: F811
    snapshot_api(prompts=served)
    return snapshot.load_prompts(_source_config(tmp_path), CYCLE_ID).prompts


def test_the_list_answers_become_the_snapshot_file_in_id_order_and_never_reach_prompts_jsonl(tmp_path, snapshot_api):  # noqa: F811
    prompts = _loaded(tmp_path, snapshot_api)
    assert [p["answer"]["eps_pct_change"] for p in prompts] == [-41.22, 5.0]  # carried from the list, id order

    files, meta = snapshot.take_answers(_source_config(tmp_path), prompts)

    assert not any("answer" in p for p in prompts)  # what run_snapshot writes as prompts.jsonl next
    rows = [json.loads(line) for line in files[TASK].decode().splitlines()]
    assert rows == [{"content_hash": SERVED[1]["content_hash"], "prompt_id": "11", "eps_pct_change": -41.22},
                    {"content_hash": SERVED[0]["content_hash"], "prompt_id": "12", "eps_pct_change": 5.0}]
    assert meta == {"prompt_answers": {TASK: {"count": 2, "file_sha256": hashlib.sha256(files[TASK]).hexdigest()}}}


def test_the_answers_are_dropped_unread_for_a_task_in_file_mode_or_not_listed(tmp_path, snapshot_api):  # noqa: F811
    prompts = _loaded(tmp_path, snapshot_api)
    cfg = _source_config(tmp_path)
    cfg.prompt_validator.scoring_methods = {TASK: ScoringMethodCfg(
        method="eps_error", settings={"answers_path": "/a.jsonl", "answers_sha256": "a" * 64})}
    assert snapshot.take_answers(cfg, prompts) == ({}, {})
    assert not any("answer" in p for p in prompts)
    other = [{"prompt_id": "11", "customer_task_id": "t-legal", "content_hash": "0" * 64}]
    assert snapshot.take_answers(_source_config(tmp_path), other) == ({}, {})


@pytest.mark.parametrize("answer, match", [
    (None, "prompt 12 without a finite answer.eps_pct_change"),  # a pool prompt added with no answer
    ({"eps_diluted_pct_change": 2.0}, "without a finite answer.eps_pct_change"),
    ({"eps_pct_change": "2.0"}, "without a finite"),
    ({"eps_pct_change": True}, "without a finite"),
])
def test_a_listed_prompt_without_a_usable_answer_stops_the_snapshot(tmp_path, snapshot_api, answer, match):  # noqa: F811
    served = [SERVED[1], {**SERVED[0], "answer": answer}]
    prompts = _loaded(tmp_path, snapshot_api, served)
    # A MethodStop: the loop exits 78 (EX_CONFIG) rather than retry, as the draw and its answers never change.
    with pytest.raises(AnswersFileError, match=match):
        snapshot.take_answers(_source_config(tmp_path), prompts)


def test_a_missing_answer_exits_the_loop_78_rather_than_retrying(tmp_path, monkeypatch):
    def stop(cfg, cycle_id):
        raise AnswersFileError("prompt-record served earnings_forecast prompt 12 without a finite answer")

    cfg = tmp_path / "pv.yaml"
    cfg.write_text("")
    monkeypatch.setattr(run, "load_config", lambda args, parser: _source_config(tmp_path))
    monkeypatch.setattr(run, "require_gpu_env", lambda: None)
    monkeypatch.setattr(run, "next_cycle", lambda cfg: 1405)
    monkeypatch.setattr(run, "run_cycle", stop)
    monkeypatch.setattr("sys.argv", ["run", "--path", str(cfg), "--loop"])
    with pytest.raises(SystemExit) as e:
        run.main()
    assert e.value.code == 78


class HashedList(FakeDataless):
    """FakeDataless whose list pages carry an answers_hash, as prompt-record may add beside snapshot_hash."""

    def __init__(self, served_hash=None, **kwargs):
        super().__init__(**kwargs)
        self.served_hash = served_hash

    def _list(self, cycle, query):
        answer = super()._list(cycle, query)
        if answer.status_code == 200:
            page = answer.json()
            page["answers_hash"] = self.served_hash or dataless.answers_hash(self.prompts)
            answer = Answer(200, page)
        return answer


def _hashed(tmp_path, monkeypatch, served_hash=None):
    api = HashedList(served_hash, prompts=SERVED)
    monkeypatch.setattr(snapshot.dataless, "DatalessClient",
                        lambda cfg, signer: REAL_CLIENT(cfg, Signer(), session=api))
    monkeypatch.setattr(snapshot.dataless, "wallet_signer", lambda wallet: None)
    import bittensor
    monkeypatch.setattr(bittensor, "Wallet", lambda **kw: None)
    return snapshot.load_prompts(_source_config(tmp_path), CYCLE_ID)


def test_an_answers_hash_on_the_list_is_kept_in_the_meta_and_checked(tmp_path, monkeypatch):
    prompts, meta, _, _ = _hashed(tmp_path, monkeypatch)
    assert meta["dataless_answers_hash"] == dataless.answers_hash(SERVED)
    files, _ = snapshot.take_answers(_source_config(tmp_path), prompts, meta)
    assert len(files[TASK].splitlines()) == 2


def test_answers_that_do_not_match_the_listed_answers_hash_stop(tmp_path, monkeypatch):
    prompts, meta, _, _ = _hashed(tmp_path, monkeypatch, served_hash="0" * 64)
    with pytest.raises(AnswersFileError, match="the dataless-api says 000000000000"):
        snapshot.take_answers(_source_config(tmp_path), prompts, meta)


def test_answers_hash_is_id_content_hash_then_the_canonical_answer_length_prefixed_in_id_order():
    h = hashlib.sha256()
    for p in (SERVED[1], SERVED[0]):  # id order; the miner's prompt has no answer and is left out
        answer = json.dumps(p["answer"], sort_keys=True, separators=(",", ":")).encode()
        h.update(int(p["prompt_id"]).to_bytes(8, "big") + bytes.fromhex(p["content_hash"])
                 + len(answer).to_bytes(4, "big") + answer)
    assert dataless.answers_hash(SERVED) == h.hexdigest()


def test_without_an_answers_hash_on_the_list_nothing_is_checked(tmp_path, snapshot_api):  # noqa: F811
    snapshot_api(prompts=SERVED)
    prompts, meta, _, _ = snapshot.load_prompts(_source_config(tmp_path), CYCLE_ID)
    assert "dataless_answers_hash" not in meta
    assert snapshot.take_answers(_source_config(tmp_path), prompts, meta)[1]["prompt_answers"][TASK]["count"] == 2


class SampleApi(FakeDataless):
    """FakeDataless plus GET /v1/tasks/{task}/sample, shaped as prompt-record serves it (fields as the live route's,
    values made up)."""

    def __init__(self, sample, **kwargs):
        super().__init__(**kwargs)
        self.sample = sample

    def request(self, method, url, data=None, headers=None, timeout=None, allow_redirects=True):
        path = urlsplit(url).path
        if not path.endswith("/sample"):
            return super().request(method, url, data, headers, timeout, allow_redirects)
        assert headers and "X-Bittensor-Signature" in headers
        self.requests.append({"method": method, "path": path})
        return Answer(200, self.sample)


def _sample(value=-41.22, text="Headlines for the Company, quarter three."):
    return {"customer_task_id": TASK, "text": text, "content_hash": hashlib.sha256(text.encode()).hexdigest(),
            "answer": _answer(value), "sample_at": "2026-11-24T18:00:00Z"}


@pytest.mark.parametrize("sample, ok", [
    (_sample(), True),
    ({**_sample(), "content_hash": "0" * 64}, False),
    ({**_sample(), "answer": {"eps_diluted_pct_change": 1.0}}, False),
    ({**_sample(), "answer": None}, False),
])
def test_the_sample_format_check_reads_the_task_sample_and_reports_no_values(tmp_path, monkeypatch, sample, ok):
    api = SampleApi(sample)
    monkeypatch.setattr(snapshot.dataless, "validator_client", lambda cfg: REAL_CLIENT(
        dataless.DatalessCfg(url="https://dataless.test"), Signer(), session=api))
    got = snapshot.sample_format_check(_source_config(tmp_path), TASK)
    assert api.requests[-1]["path"] == f"/v1/tasks/{TASK}/sample"
    assert (got["content_hash_matches_text"] and got["eps_pct_change_ok"]) is ok
    assert got["fields"] == ["answer", "content_hash", "customer_task_id", "sample_at", "text"]
    assert sample["text"] not in json.dumps(got) and "-41.22" not in json.dumps(got)


def test_prompt_record_answers_need_the_dataless_prompt_source(tmp_path):
    cfg = _source_config(tmp_path)
    cfg.prompt_validator.prompt_source = "file"
    listed = [{"prompt_id": "11", "customer_task_id": TASK, "content_hash": "0" * 64}]
    with pytest.raises(AnswersFileError, match="needs prompt_source dataless"):
        snapshot.take_answers(cfg, listed)


# --- the scores stage ---

def _setup(config, answer_for=("p1", "p2"), actual=-41.22):  # noqa: F811
    """The scores fixture's p1 and p2 become TASK's prompts, read from prompt-record: the snapshot's answers file and
    meta written as run_snapshot would. Returns the file."""
    pv = config.prompt_validator
    snapshot_dir = stage_path(pv.runs_dir, CYCLE_ID, "snapshot").parent
    prompts = [{**p, "content_hash": hashlib.sha256(p["prompt"].encode()).hexdigest(), "customer_task_id": TASK}
               for p in read_rows(snapshot_dir / "prompts.jsonl")]
    write_rows(snapshot_dir / "prompts.jsonl", prompts)
    data = "".join(json.dumps({"content_hash": p["content_hash"], "prompt_id": p["prompt_id"],
                               "eps_pct_change": actual}) + "\n" for p in prompts if p["prompt_id"] in answer_for)
    path = snapshot_dir / answers_file_name(TASK)
    path.write_text(data)
    pv.scoring_methods = {TASK: _source_cfg()}
    meta = {**SCORES_BASE, "rules_sha256": snapshot.rules_sha256(pv, SCORES_BASE),
            "prompt_answers": {TASK: {"count": len(answer_for),
                                      "file_sha256": hashlib.sha256(data.encode()).hexdigest()}}}
    write_rows(snapshot_dir / "snapshot.jsonl", [meta])
    answers_path = stage_path(pv.runs_dir, CYCLE_ID, "answers")
    texts = {"h1": "Costs rose.\nFORECAST: -35.0", "h2": "Costs fell.\nFORECAST: +15.0"}
    write_rows(answers_path, [{**a, "text": texts[a["model_hash"]]} for a in read_rows(answers_path)])
    return path


def test_the_scores_stage_ranks_against_the_answers_the_snapshot_took(config, fake):  # noqa: F811
    _setup(config)

    run.run_stages(config, CYCLE_ID, "scores")

    rows = {(r["uid"], r["prompt_id"]): r for r in _scores(config)}
    one, two = rows[(1, "p1")], rows[(2, "p1")]
    assert (one["method"], one["forecast_pct"], one["abs_error_pp"], one["error_rank"], one["answer_id"]) == \
           ("eps_error", -35.0, 6.22, 1, "p1")
    assert (two["error_rank"], two["expected_score"]) == (2, round(8 / 9, 4))
    assert fake.bodies == []


def test_a_changed_or_missing_snapshot_answers_file_or_meta_stops_the_scores_stage(config, fake):  # noqa: F811
    path = _setup(config)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(AnswersFileError, match="not the pinned"):
        run.run_stages(config, CYCLE_ID, "scores")
    path.unlink()
    with pytest.raises(AnswersFileError, match="is missing"):
        run.run_stages(config, CYCLE_ID, "scores")

    _setup(config)
    snapshot_dir = stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "snapshot").parent
    meta = read_rows(snapshot_dir / "snapshot.jsonl")[0]
    write_rows(snapshot_dir / "snapshot.jsonl", [{k: v for k, v in meta.items() if k != "prompt_answers"}])
    with pytest.raises(AnswersFileError, match="the snapshot took no prompt-record answers"):
        run.run_stages(config, CYCLE_ID, "scores")
    assert not stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "scores").exists()


def test_a_prompt_the_snapshot_has_no_answer_for_stops_the_scores_stage(config, fake):  # noqa: F811
    _setup(config, answer_for=("p1",))
    with pytest.raises(AnswersFileError, match=r"no answer for prompts \['p2'\]"):
        run.run_stages(config, CYCLE_ID, "scores")


def test_check_reads_no_file_in_prompt_record_mode():
    REGISTRY["eps_error"].check(_source_cfg().parsed())  # nothing to load before the snapshot
