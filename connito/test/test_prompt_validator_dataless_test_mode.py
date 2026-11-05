import json
import logging
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit

import bittensor
import pytest
import yaml
from test_prompt_validator_dataless import (
    CYCLE_ID,
    PROMPTS,
    REAL_CLIENT,
    URL,
    Answer,
    FakeDataless,
    Signer,
    cycle,  # noqa: F401  (a fixture)
    fixed_base_rows,  # noqa: F401  (autouse here too: the rules check reads the base rows)
    no_sleep,  # noqa: F401  (a fixture, which cycle uses)
)

from connito.prompt_validator import cutoff, dataless, report, run, snapshot
from connito.prompt_validator.config import DatalessCfg, PromptValidatorCfg, PromptValidatorChainCfg
from connito.prompt_validator.stage_io import stage_path, write_rows
from connito.shared.config import CycleCfg
from connito.test.test_prompt_validator_cutoff import OWNER_PERIODS, _api_phase

TEST_CYCLE = 1_000_000_003


class FakeTestServer:
    """The dataless-api's /v1/test copy: the real routes (FakeDataless) under /v1/test, and a test-cycle view that
    the owner moves by hand (`views`: one per read; 404 when None, or an Answer as it is)."""

    def __init__(self, views, real=None):
        self.views, self.real, self.paths = list(views), real or FakeDataless(open_cycle=TEST_CYCLE + 1), []

    def request(self, method, url, data=None, headers=None, timeout=None, allow_redirects=True):
        parts = urlsplit(url)
        self.paths.append((method, parts.path, headers))
        assert parts.path.startswith("/v1/test/"), f"a test-mode client called {parts.path}"
        if parts.path == "/v1/test/prompt-cycle":
            view = self.views.pop(0) if len(self.views) > 1 else self.views[0]
            if view is None:
                return Answer(404, {"detail": {"code": "not_found", "message": "no test cycle"}})
            if isinstance(view, Answer):
                return view
            return Answer(200, {"head": 9308184, "tasks": [], "policy": {"rollout_phase": 2}, **view})
        real_url = urlunsplit(parts._replace(path=parts.path.replace("/v1/test/", "/v1/", 1)))
        return self.real.request(method, real_url, data=data, headers=headers, timeout=timeout,
                                 allow_redirects=allow_redirects)


def _test_client(server, signer=None):
    return REAL_CLIENT(DatalessCfg(url=URL, test_mode=True, test_poll_sec=0), signer or Signer(), session=server)


@pytest.fixture
def sleeps(monkeypatch):
    out = []
    monkeypatch.setattr(dataless.time, "sleep", out.append)
    return out


def test_every_call_goes_to_the_test_routes_and_signs_the_test_path(sleeps):
    server, signer = FakeTestServer([{"cycle_id": TEST_CYCLE, "state": "closed"}]), Signer()
    client = _test_client(server, signer)
    client.prompt_cycle()
    client.start_run(TEST_CYCLE, {"rollout_phase": 2})
    assert [p for _, p, _ in server.paths] == ["/v1/test/prompt-cycle",
                                               f"/v1/test/validator/cycles/{TEST_CYCLE}/status"]
    # The test cycle view is signed too (/v1/test serves its test hotkeys only), on its /v1/test path.
    assert signer.calls[0] == ("GET", "/v1/test/prompt-cycle", b"")
    assert signer.calls[1][:2] == ("POST", f"/v1/test/validator/cycles/{TEST_CYCLE}/status")
    assert server.paths[0][2]["X-Bittensor-Signature"] and server.paths[0][2]["X-Bittensor-Hotkey"] == "5Validator"


def test_the_real_cycle_view_stays_unsigned():
    fake, signer = FakeDataless(), Signer()
    REAL_CLIENT(DatalessCfg(url=URL), signer, session=fake).prompt_cycle()
    assert signer.calls == [] and fake.requests[0]["path"] == "/v1/prompt-cycle"
    assert not any(h.startswith("X-Bittensor") for h in fake.requests[0]["headers"])


NOT_A_TESTER = Answer(403, {"detail": {"code": "role_forbidden",
                                       "message": "This hotkey is not on the test allowlist (TEST_HOTKEYS)."}})


def test_a_hotkey_off_test_hotkeys_is_a_clear_error_never_waited_on_as_no_test_cycle(sleeps, caplog):
    server = FakeTestServer([NOT_A_TESTER, {"cycle_id": TEST_CYCLE, "state": "closed"}])
    with caplog.at_level(logging.INFO), pytest.raises(dataless.TesterRefused) as refused:
        _test_client(server).next_closed_test_cycle(lambda c: False)
    assert str(refused.value) == (
        "dataless-api 403 role_forbidden: hotkey 5Validator refused on GET /v1/test/prompt-cycle: This hotkey is not "
        "on the test allowlist (TEST_HOTKEYS). (/v1/test needs the hotkey on TEST_HOTKEYS, and its validator routes "
        "also a whitelisted validator)")
    assert len(server.paths) == 1 and sleeps == []  # not retried, not slept on
    assert "shows no test cycle" not in caplog.text and "0x" + "ab" * 64 not in caplog.text


def test_a_validator_route_refused_in_test_mode_leads_with_the_servers_reason():
    # On TEST_HOTKEYS but not a whitelisted validator: the server's own words say which, never a blanket blame.
    no_role = Answer(403, {"detail": {"code": "role_forbidden", "message": "This hotkey does not have the role."}})
    server = FakeTestServer([{"cycle_id": TEST_CYCLE, "state": "closed"}])
    server.real = SimpleNamespace(request=lambda *a, **kw: no_role)
    with pytest.raises(dataless.TesterRefused) as refused:
        _test_client(server).start_run(TEST_CYCLE, {"rollout_phase": 2})
    assert str(refused.value).startswith(
        "dataless-api 403 role_forbidden: hotkey 5Validator refused on POST "
        "/v1/test/validator/cycles/1000000003/status: This hotkey does not have the role. (")


def test_an_unsigned_test_request_refused_says_unsigned():
    server = FakeTestServer([NOT_A_TESTER])
    with pytest.raises(dataless.TesterRefused, match="403 role_forbidden: an unsigned request refused on GET "):
        _test_client(server).request("GET", "/v1/test/prompt-cycle", signed=False)


def test_a_real_route_refused_is_a_plain_error():
    fake = FakeDataless()
    fake.request = lambda *a, **kw: NOT_A_TESTER
    with pytest.raises(dataless.DatalessError) as refused:
        REAL_CLIENT(DatalessCfg(url=URL), Signer(), session=fake).start_run(CYCLE_ID, {})
    assert type(refused.value) is dataless.DatalessError


def test_the_real_client_never_uses_the_test_routes():
    client = REAL_CLIENT(DatalessCfg(url=URL), Signer(), session=FakeDataless())
    assert (client.api, client.prefix) == ("/v1", "/v1/validator")


def test_the_run_waits_for_a_closed_test_cycle_not_reported_yet(sleeps):
    server = FakeTestServer([None, {"cycle_id": TEST_CYCLE - 1, "state": "closed"},
                             {"cycle_id": TEST_CYCLE, "state": "open"}, {"cycle_id": TEST_CYCLE, "state": "closed"}])
    reported = {TEST_CYCLE - 1}
    assert _test_client(server).next_closed_test_cycle(lambda c: c in reported) == TEST_CYCLE
    assert len(sleeps) == 3  # none opened yet, the last one already reported, the new one still open


def test_a_closed_test_cycle_is_run_while_the_next_one_is_open(sleeps):
    # The owner opened N+1 right after closing N: the API shows N+1, and still lists N until N+1 closes.
    server = FakeTestServer([{"cycle_id": TEST_CYCLE + 1, "state": "open"}])
    client = _test_client(server)
    assert client.next_closed_test_cycle(lambda c: False) == TEST_CYCLE
    assert client.wait_until_listed(TEST_CYCLE)["cycle_id"] == TEST_CYCLE + 1 and sleeps == []


def test_the_snapshot_waits_for_its_test_cycle_to_close_and_stops_once_a_newer_one_closed(sleeps):
    server = FakeTestServer([{"cycle_id": TEST_CYCLE, "state": "open"}, {"cycle_id": TEST_CYCLE, "state": "closed"}])
    assert _test_client(server).wait_until_listed(TEST_CYCLE)["state"] == "closed" and len(sleeps) == 1
    for view in ({"cycle_id": TEST_CYCLE + 1, "state": "closed"}, {"cycle_id": TEST_CYCLE + 2, "state": "open"}):
        with pytest.raises(RuntimeError, match="lists test cycle 1000000004 now, not 1000000003"):
            _test_client(FakeTestServer([view])).wait_until_listed(TEST_CYCLE)


def test_a_test_cycle_view_that_stays_404_warns_that_test_cycles_may_be_off(sleeps, caplog):
    server = FakeTestServer([None] * 3 + [{"cycle_id": TEST_CYCLE, "state": "closed"}])
    with caplog.at_level(logging.WARNING):
        assert _test_client(server).next_closed_test_cycle(lambda c: False) == TEST_CYCLE
    assert "test cycles are off on the server" in caplog.text


def test_the_snapshot_lists_the_test_cycles_owner_prompts(tmp_path, monkeypatch, sleeps):
    server = FakeTestServer([{"cycle_id": TEST_CYCLE, "state": "closed"}])
    monkeypatch.setattr(snapshot.dataless, "DatalessClient", lambda cfg, signer: _test_client(server))
    monkeypatch.setattr(snapshot.dataless, "wallet_signer", lambda wallet: None)
    monkeypatch.setattr(bittensor, "Wallet", lambda **kw: None)
    pv = PromptValidatorCfg(runs_dir=tmp_path, write_root="/", prompt_source="dataless",
                            dataless=DatalessCfg(url=URL, test_mode=True))
    config = SimpleNamespace(prompt_validator=pv, chain=PromptValidatorChainCfg(coldkey_name="c", hotkey_name="h"))

    prompts, meta, _, _ = snapshot.load_prompts(config, TEST_CYCLE)

    assert [p["prompt_id"] for p in prompts] == ["11", "12"] and meta["dataless_snapshot_count"] == len(PROMPTS)
    assert f"/v1/test/validator/cycles/{TEST_CYCLE}/prompts" in {p for _, p, _ in server.paths}


def test_the_report_runs_the_whole_lifecycle_under_the_test_routes(cycle, monkeypatch):  # noqa: F811
    config, fake = cycle
    server = FakeTestServer([{"cycle_id": CYCLE_ID, "state": "closed"}], real=fake)
    config.prompt_validator.dataless = DatalessCfg(url=URL, test_mode=True)
    monkeypatch.setattr(report.dataless, "DatalessClient", lambda cfg, signer: _test_client(server))

    assert report.run_report(config, CYCLE_ID)[0]["status"] == "published"
    assert [p.rsplit("/", 1)[1] for _, p, _ in server.paths] == ["status", "models", "summary", "model-scores",
                                                                "status"]
    assert all(p.startswith(f"/v1/test/validator/cycles/{CYCLE_ID}/") for _, p, _ in server.paths)
    assert json.loads(fake.requests[-1]["data"]) == {"status": "published"}  # no weights_block: none were set


# --- the cutoff in test mode: no waiting, the latest real cutoff ---

def _cutoff_config(tmp_path, test_mode=True):
    pv = PromptValidatorCfg(runs_dir=tmp_path, write_root="/", prompt_source="dataless",
                            dataless=DatalessCfg(url=URL, test_mode=test_mode))
    return SimpleNamespace(cycle=CycleCfg(**OWNER_PERIODS), prompt_validator=pv)


@pytest.fixture
def chain_at(monkeypatch):
    """cycle-api's /get_phase at a given block, with the real phase arithmetic."""
    def at(block, **overrides):
        monkeypatch.setattr(cutoff, "get_phase_from_api", lambda cfg: _api_phase(block, **overrides))
        monkeypatch.setattr(cutoff, "check_phase_clock", lambda cfg, phase: phase)
        monkeypatch.setattr(cutoff, "wait_till", lambda *a: pytest.fail("a test cycle must not wait"))
    return at


@pytest.mark.parametrize("offset, back", [(0, 0), (40, 0), (-1, 1)])
def test_a_test_cycles_models_come_from_the_latest_real_cutoff_without_waiting(tmp_path, chain_at, offset, back):
    config = _cutoff_config(tmp_path)
    this = cutoff.training_cycle_blocks(config, TEST_CYCLE, 17630)
    chain_at(this.commit_2_block + offset)  # at the cutoff block itself the cutoff has passed, as in a real run

    [row] = cutoff.run_cutoff(config, TEST_CYCLE)

    want = cutoff.training_cycle_blocks(config, TEST_CYCLE, 17630 - back)
    assert row == {**want.__dict__, "test_cycle": True}


def test_a_cycle_index_that_does_not_count_from_block_0_stops_the_test_cutoff(tmp_path, chain_at):
    config = _cutoff_config(tmp_path)
    chain_at(cutoff.training_cycle_blocks(config, TEST_CYCLE, 17630).commit_2_block, cycle_index=5)
    with pytest.raises(RuntimeError, match="cycle_index 5 is not block"):
        cutoff.run_cutoff(config, TEST_CYCLE)


def test_with_test_mode_off_the_cutoff_is_the_scoring_cycles_own(tmp_path, chain_at):
    config = _cutoff_config(tmp_path, test_mode=False)
    own = cutoff.cutoff_blocks(config, 1258)
    chain_at(own.commit_2_block + 1)
    assert cutoff.run_cutoff(config, 1258) == [own.__dict__]


def test_test_mode_needs_the_dataless_prompt_source(tmp_path):
    with pytest.raises(ValueError, match='test_mode needs prompt_source "dataless"'):
        PromptValidatorCfg(prompts_path=tmp_path / "p.jsonl", dataless=DatalessCfg(url=URL, test_mode=True))


def test_the_test_cycle_id_is_taken_from_the_api_when_none_is_given(tmp_path, monkeypatch):
    pv = PromptValidatorCfg(runs_dir=tmp_path, write_root="/", prompt_source="dataless",
                            dataless=DatalessCfg(url=URL, test_mode=True))
    write_rows(stage_path(pv.runs_dir, TEST_CYCLE - 1, "report"), [{}])  # already reported
    config = SimpleNamespace(prompt_validator=pv)
    monkeypatch.setattr(run.PromptValidatorConfig, "from_path", lambda path, active_task: config)
    seen = {}

    class Client:
        def __init__(self, cfg):
            pass

        def next_closed_test_cycle(self, done):
            seen["done"] = (done(TEST_CYCLE - 1), done(TEST_CYCLE))
            return TEST_CYCLE

    monkeypatch.setattr(run, "validator_client", Client)
    monkeypatch.setattr(run, "run_stages", lambda cfg, cycle_id, stage: seen.update(cycle=cycle_id, stage=stage))
    monkeypatch.setattr(run, "get_phase_from_api", lambda cfg: pytest.fail("a test cycle has no phase"))
    monkeypatch.setattr("sys.argv", ["run", "--path", "x.yaml"])
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    run.main()

    assert seen == {"done": (True, False), "cycle": TEST_CYCLE, "stage": None}


@pytest.mark.parametrize("pick", ["loop", "plain run"])
def test_the_test_cycle_pick_is_signed_with_the_configs_hotkey(tmp_path, monkeypatch, sleeps, pick):
    pv = PromptValidatorCfg(runs_dir=tmp_path, write_root="/", prompt_source="dataless",
                            dataless=DatalessCfg(url=URL, test_mode=True))
    config = SimpleNamespace(prompt_validator=pv, chain=PromptValidatorChainCfg(
        coldkey_name="c", hotkey_name="h", wallet_path="/wallets"))
    server, signer, wallets = FakeTestServer([{"cycle_id": TEST_CYCLE, "state": "closed"}]), Signer(), []
    monkeypatch.setattr(bittensor, "Wallet", lambda **kw: wallets.append(kw) or "the wallet")
    monkeypatch.setattr(dataless, "wallet_signer", lambda wallet: signer if wallet == "the wallet" else None)
    monkeypatch.setattr(dataless.requests, "Session", lambda: server)
    if pick == "loop":
        assert run.next_cycle(config) == TEST_CYCLE
    else:
        ran = []
        monkeypatch.setattr(run.PromptValidatorConfig, "from_path", lambda path, active_task: config)
        monkeypatch.setattr(run, "run_stages", lambda cfg, cycle_id, stage: ran.append(cycle_id))
        monkeypatch.setattr("sys.argv", ["run", "--path", "x.yaml"])
        monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
        run.main()
        assert ran == [TEST_CYCLE]
    assert wallets == [{"name": "c", "hotkey": "h", "path": "/wallets"}]
    assert signer.calls == [("GET", "/v1/test/prompt-cycle", b"")]
    assert server.paths[0][2]["X-Bittensor-Signature"]


def _start(monkeypatch, pv, *args):
    """Run main() with args on a config holding pv; returns what run_stages was given, or None if it never ran."""
    monkeypatch.setattr(run.PromptValidatorConfig, "from_path", lambda path, active_task: SimpleNamespace(
        prompt_validator=pv))
    monkeypatch.setattr(run, "get_phase_from_api", lambda cfg: pytest.fail("a test cycle has no phase"))
    seen = {}

    class Client:
        def __init__(self, cfg):
            pass

        def next_closed_test_cycle(self, done):
            return TEST_CYCLE

    monkeypatch.setattr(run, "validator_client", Client)
    monkeypatch.setattr(run, "run_stages", lambda cfg, cycle_id, stage: seen.update(
        cycle=cycle_id, test_mode=cfg.prompt_validator.dataless.test_mode))
    monkeypatch.setattr("sys.argv", ["run", "--path", "x.yaml", *args])
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    run.main()
    return seen or None


def test_the_test_flag_turns_test_mode_on_for_a_dataless_config(tmp_path, monkeypatch):
    pv = PromptValidatorCfg(runs_dir=tmp_path, write_root="/", prompt_source="dataless", dataless=DatalessCfg(url=URL))

    assert _start(monkeypatch, pv, "--test") == {"cycle": TEST_CYCLE, "test_mode": True}


def test_the_test_flag_is_refused_without_the_dataless_prompt_source(tmp_path, monkeypatch, capsys):
    pv = PromptValidatorCfg(runs_dir=tmp_path, write_root="/", prompts_path=tmp_path / "p.jsonl")

    with pytest.raises(SystemExit) as exit_:
        _start(monkeypatch, pv, "--test")

    assert exit_.value.code == 2 and '--test needs prompt_source "dataless"' in capsys.readouterr().err
    assert pv.dataless.test_mode is False


# --- the example config in the repo ---

EXAMPLE = Path(run.__file__).with_name("config.example.yaml")


@pytest.mark.parametrize("test_mode", [False, True])
def test_the_example_config_is_valid_and_its_test_mode_line_turns_test_mode_on(test_mode):
    text = EXAMPLE.read_text()
    if test_mode:
        assert text.count("# test_mode: true") == 1
        text = text.replace("# test_mode: true", "test_mode: true")
    data = yaml.safe_load(text)

    pv = PromptValidatorCfg(**data["prompt_validator"])  # the whole config would read the chain

    assert set(data) == {"chain", "prompt_validator"} and set(data["chain"]) == {"coldkey_name", "hotkey_name"}
    assert (pv.prompt_source, pv.dataless.url, pv.dataless.test_mode) == ("dataless", DatalessCfg().url, test_mode)
    assert pv.runs_dir == Path("/scratch/connito/prompt-validator/runs")
