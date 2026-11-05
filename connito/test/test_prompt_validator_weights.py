import json
import logging
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from connito.prompt_validator import run, snapshot, weights
from connito.prompt_validator.config import PromptValidatorCfg, WeightsCfg
from connito.prompt_validator.stage_io import read_rows, stage_path, write_rows
from connito.prompt_validator.status import loop_status, stage_counts
from connito.shared import bt_compat
from connito.test import test_prompt_validator_loop as loop_tests

looping = loop_tests.looping
CYCLE = 1259
ME = 5  # this validator's uid
HOTKEYS = ["owner", "a", "b", "c", "d", "me"]
BASE = {"base_model": "tiny", "base_model_revision": "0" * 40, "chat_template_sha256": "1" * 64}


def _fake_chain(monkeypatch, row, hotkeys=HOTKEYS, reads=None):
    """A lite metagraph at block 1000 (every last_update 900), and uid ME's weights as raw u16 pairs from `row`."""
    metagraph = SimpleNamespace(block=1000, hotkeys=list(hotkeys), uids=np.arange(len(hotkeys)),
                                last_update=np.full(len(hotkeys), 900))

    def get_metagraph(subtensor, netuid):
        if reads is not None:
            reads.append(("metagraph", netuid))
        return metagraph

    def neuron_for_uid(subtensor, uid, netuid):
        if reads is not None:
            reads.append(("neuron", uid, netuid))
        return SimpleNamespace(hotkey=hotkeys[uid], weights=[(u, w) for u, w in enumerate(row) if w])

    monkeypatch.setattr(bt_compat, "get_metagraph", get_metagraph)
    monkeypatch.setattr(bt_compat, "neuron_for_uid", neuron_for_uid)
    monkeypatch.setattr(weights.bittensor, "Subtensor", lambda network: SimpleNamespace(network=network))


def _config(tmp_path, prompt_share=0.2):
    pv = PromptValidatorCfg(runs_dir=tmp_path, write_root="/", prompt_source="dataless",
                            weights=WeightsCfg(mode="dry_run", prompt_share=prompt_share))
    wallet = SimpleNamespace(hotkeypub=SimpleNamespace(ss58_address="me"))
    return SimpleNamespace(prompt_validator=pv, chain=SimpleNamespace(netuid=102, network="archive",
                                                                      wallet=lambda: wallet))


def _rewards(tmp_path, rewards):
    write_rows(stage_path(tmp_path, CYCLE, "reward"),
               [{"uid": 99, "hotkey": hotkey, "weight": weight} for hotkey, weight in rewards.items()])


@pytest.fixture
def dry_run(tmp_path, monkeypatch):
    """Run the weights stage against an on-chain row of u16 values and reward weights by hotkey; returns its row."""
    def start(chain_row, rewards, prompt_share=0.2):
        _rewards(tmp_path, rewards)
        _fake_chain(monkeypatch, chain_row)
        [row] = weights.run_weights(_config(tmp_path, prompt_share), CYCLE)
        assert "error" not in row
        return row

    return start


def test_at_the_defaults_the_owner_keeps_25_and_prompt_and_training_split_the_rest_15_to_60(dry_run):
    row = dry_run([16384, 32768, 16384], {"c": 1.0})
    assert row["training"] == pytest.approx({0: 0.25, 1: 0.5, 2: 0.25})
    assert row["prompt"] == {3: 1.0}
    assert row["combined"] == pytest.approx({0: 0.25, 1: 0.4, 2: 0.2, 3: 0.15})
    assert row["l1_to_chain"] == pytest.approx(0.1 + 0.05 + 0.15)
    assert [uid for uid, _ in row["top10"]] == [1, 0, 2, 3]
    assert [w for _, w in row["top10"]] == pytest.approx([0.4, 0.25, 0.2, 0.15])
    assert (row["block"], row["uid"], row["hotkey"], row["last_update_age_blocks"]) == (1000, ME, "me", 100)
    assert (row["guards"], row["dropped_hotkeys"], row["prompt_share"]) == ([], [], 0.2)


def test_both_sides_are_renormalized_before_the_split(dry_run):
    # A u16 row whose owner part is not exactly a quarter, and reward weights that do not sum to 1.
    row = dry_run([16390, 24000, 24000], {"a": 2.0, "b": 6.0}, prompt_share=0.5)
    assert row["prompt"] == {1: 0.25, 2: 0.75}
    assert row["combined"] == pytest.approx({0: 0.25, 1: 0.75 * (0.5 * 0.5 + 0.5 * 0.25),
                                             2: 0.75 * (0.5 * 0.5 + 0.5 * 0.75)})
    assert sum(row["combined"].values()) == pytest.approx(1.0)


def test_a_rewarded_hotkey_no_longer_registered_is_dropped_and_counted(dry_run):
    row = dry_run([1, 3], {"gone": 3.0, "b": 1.0, "c": 0.0})
    assert row["prompt"] == {2: 1.0}
    assert row["dropped_hotkeys"] == ["gone"]
    assert row["combined"] == pytest.approx({0: 0.25, 1: 0.6, 2: 0.15})


def test_a_prompt_reward_to_the_owner_uid_never_raises_the_owner_above_its_share(dry_run):
    row = dry_run([1, 3], {"owner": 1.0, "c": 1.0})
    assert row["prompt"] == {3: 1.0}
    assert row["combined"] == pytest.approx({0: 0.25, 1: 0.6, 3: 0.15})


@pytest.mark.parametrize("chain_row", [[], [65535]], ids=["empty", "all_to_uid_0"])
def test_a_training_row_with_no_miner_weight_gives_the_prompt_part_all_the_miners_weight(dry_run, chain_row):
    row = dry_run(chain_row, {"a": 1.0, "b": 3.0})
    assert row["combined"] == pytest.approx({0: 0.25, 1: 0.1875, 2: 0.5625})
    assert row["guards"] == ["no miner weight in the training row: the prompt part takes all the miners' weight"]


def test_no_prompt_reward_leaves_the_training_row_as_it_is(dry_run):
    row = dry_run([1, 2, 1], {"a": 0.0, "b": 0.0})
    assert row["prompt"] == {}
    assert row["combined"] == pytest.approx(row["training"])
    assert row["l1_to_chain"] == pytest.approx(0.0, abs=1e-9)
    assert row["guards"] == ["no prompt reward: the training part takes all the miners' weight"]


def test_with_neither_side_everything_goes_to_the_subnet_owner(dry_run):
    row = dry_run([65535], {})
    assert row["combined"] == {0: 1.0}
    assert row["guards"][0].startswith("no training miner weight and no prompt reward")


def test_the_stage_summary_line_carries_the_key_numbers(dry_run, tmp_path):
    row = dry_run([1, 2, 1], {"c": 1.0, "gone": 1.0})
    counts = stage_counts("weights", [row], tmp_path)
    assert counts == {"uid": ME, "update_age_blocks": 100, "l1_to_chain": 0.3, "owner": 0.25, "dropped_hotkeys": 1,
                      "guards": []}


class PubOnly:
    """A wallet with a public hotkey file: opening the private hotkey fails the test."""

    hotkeypub = SimpleNamespace(ss58_address="me")

    @property
    def hotkey(self):
        raise AssertionError("the private hotkey was opened for an address")


class NoPubFile:
    """A wallet from an older btcli, with no hpub.txt."""

    hotkey = SimpleNamespace(ss58_address="me")

    @property
    def hotkeypub(self):
        raise FileNotFoundError("keyfile at '.../hpub.txt' does not exist")


@pytest.mark.parametrize("wallet", [PubOnly(), NoPubFile()], ids=["pub_file", "no_pub_file"])
def test_the_address_comes_from_the_public_hotkey_file_when_there_is_one(wallet):
    assert weights.hotkey_address(wallet) == "me"


def test_the_stage_reads_the_chain_and_never_writes_to_it(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _rewards(tmp_path, {"c": 1.0})
    touched, reads = [], []

    class ReadOnlySubtensor:
        def __init__(self, network):
            reads.append(network)

        def __getattr__(self, name):  # the fake reads need none of it: any use is a call to the chain
            touched.append(name)
            raise AssertionError(f"subtensor.{name} used")

    def write(name):
        def refuse(*args, **kwargs):
            touched.append(name)
            raise AssertionError(f"{name} called")
        return refuse

    _fake_chain(monkeypatch, [1, 3], reads=reads)
    for name in ("set_weights", "aset_weights", "set_commitment", "aset_commitment", "serve_axon"):
        monkeypatch.setattr(bt_compat, name, write(name))
    monkeypatch.setattr(weights.bittensor, "Subtensor", ReadOnlySubtensor)
    run.run_stages(config, CYCLE, "weights")

    [row] = read_rows(stage_path(tmp_path, CYCLE, "weights"))
    assert touched == [] and "error" not in row
    # config.chain.network, a lite metagraph, then this hotkey's neuron alone.
    assert reads == ["archive", ("metagraph", 102), ("neuron", ME, 102)]


@pytest.mark.parametrize("failure", ["chain", "unregistered", "no_reward"])
def test_a_failed_dry_run_is_logged_and_recorded_and_never_raised(tmp_path, monkeypatch, caplog, failure):
    config = _config(tmp_path)
    if failure != "no_reward":
        _rewards(tmp_path, {"c": 1.0})
    _fake_chain(monkeypatch, [1, 3], hotkeys=HOTKEYS[:ME] if failure == "unregistered" else HOTKEYS)
    if failure == "chain":
        monkeypatch.setattr(bt_compat, "get_metagraph", _unreachable)
    with caplog.at_level(logging.WARNING):
        run.run_stages(config, CYCLE, "weights")

    [row] = read_rows(stage_path(tmp_path, CYCLE, "weights"))
    expected = {"chain": "ConnectionError: archive unreachable",
                "unregistered": "RuntimeError: hotkey me is not registered on subnet 102",
                "no_reward": "FileNotFoundError"}[failure]
    assert row["error"].startswith(expected) and row["scoring_cycle_id"] == CYCLE
    assert "weights dry run failed" in caplog.text


def _unreachable(subtensor, netuid):
    raise ConnectionError("archive unreachable at wss://user:secret@node")


def test_a_chain_read_past_the_timeout_is_recorded_as_timeout_and_left_behind(tmp_path, monkeypatch):
    _rewards(tmp_path, {"c": 1.0})
    _fake_chain(monkeypatch, [1, 3])
    release = threading.Event()
    monkeypatch.setattr(bt_compat, "get_metagraph", lambda subtensor, netuid: release.wait(10))
    monkeypatch.setattr(weights, "READ_TIMEOUT_S", 0.05)
    try:
        [row] = weights.run_weights(_config(tmp_path), CYCLE)
    finally:
        release.set()
    assert row["error"] == "timeout"


def test_status_json_carries_the_error_type_alone(tmp_path, monkeypatch):
    _rewards(tmp_path, {"c": 1.0})
    _fake_chain(monkeypatch, [1, 3])
    monkeypatch.setattr(bt_compat, "get_metagraph", _unreachable)
    with loop_status(tmp_path):
        run.run_stages(_config(tmp_path), CYCLE, "weights")
    status = (tmp_path / "status.json").read_text()
    assert json.loads(status)["last_stage"]["error"] == "ConnectionError" and "secret" not in status
    [row] = read_rows(stage_path(tmp_path, CYCLE, "weights"))
    assert "secret" in row["error"]  # the whole message stays in weights.jsonl


def test_an_earlier_stage_rerun_removes_weights_and_names_its_stage(tmp_path, monkeypatch, caplog):
    config = _config(tmp_path)
    monkeypatch.setattr(run, "STAGES", [(name, lambda cfg, c: [{}]) for name, _ in run.STAGES])
    for stage in ("reward", "weights"):
        write_rows(stage_path(tmp_path, CYCLE, stage), [{}])
    with caplog.at_level(logging.WARNING):
        run.run_stages(config, CYCLE, "reward")
    assert not stage_path(tmp_path, CYCLE, "weights").exists()
    assert "stage=weights" in caplog.text and "--stage weights" in caplog.text


def test_a_real_dry_run_error_row_lets_the_loop_go_on_to_the_next_cycle(tmp_path, looping, monkeypatch):
    _fake_chain(monkeypatch, [1, 3])
    monkeypatch.setattr(bt_compat, "get_metagraph", _unreachable)
    rows = []

    def measure(cfg, cycle_id, name):
        if name == "weights":
            rows.extend(weights.run_weights(cfg, cycle_id))  # looping stands in for run_stages, and its clock

    def make_config():
        config = _dry_run_config(tmp_path)
        config.chain = _config(tmp_path).chain
        return config

    calls, _ = looping(make_config, cycles=2, stage=measure)
    first = loop_tests.TEST_CYCLE
    assert calls == [(first, None), (first, "report"), (first, "weights"), (first + 1, None), (first + 1, "report")]
    assert rows[0]["error"].startswith("ConnectionError")



def test_with_mode_off_the_loop_runs_no_weights_stage(tmp_path, looping):
    calls, _ = looping(lambda: loop_tests._config(tmp_path, test_mode=True), cycles=2)
    assert [name for _, name in calls] == [None, "report", None, "report"]
    assert not list(tmp_path.glob("*/weights"))


def _dry_run_config(tmp_path):
    config = loop_tests._config(tmp_path, test_mode=True)
    config.prompt_validator.weights = WeightsCfg(mode="dry_run")
    return config


def test_with_dry_run_the_loop_runs_weights_after_report(tmp_path, looping):
    calls, _ = looping(lambda: _dry_run_config(tmp_path), cycles=2)
    first = loop_tests.TEST_CYCLE
    assert calls == [(first, None), (first, "report"), (first, "weights"), (first + 1, None), (first + 1, "report")]


def test_a_weights_stage_that_raises_neither_fails_the_cycle_nor_stops_the_loop(tmp_path, looping, caplog):
    def broken(cfg, cycle_id, name):
        if name == "weights":
            raise OSError("disk full")

    # looping fails on any retry sleep: the cycle counts as done, and the next one runs.
    with caplog.at_level(logging.WARNING):
        calls, _ = looping(lambda: _dry_run_config(tmp_path), cycles=2, stage=broken)
    assert [name for _, name in calls] == [None, "report", "weights", None, "report"]
    assert "weights dry run not written" in caplog.text and "cycle failed" not in caplog.text


def test_stage_weights_needs_dry_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run.PromptValidatorConfig, "from_path",
                        lambda path, active_task: loop_tests._config(tmp_path))
    monkeypatch.setattr(run, "run_stages", lambda *a: pytest.fail("a stage ran"))
    monkeypatch.setattr("sys.argv", ["run", "--path", "x.yaml", "--cycle-id", "7", "--stage", "weights"])
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(SystemExit) as exit_:
        run.main()
    assert exit_.value.code == 2 and "--stage weights needs prompt_validator.weights.mode dry_run" in \
        capsys.readouterr().err


def test_a_plain_run_never_runs_the_weights_stage(tmp_path, monkeypatch):
    ran = []
    stages = [(name, (lambda name: lambda cfg, c: ran.append(name) or [{}])(name)) for name, _ in run.STAGES]
    monkeypatch.setattr(run, "STAGES", stages)
    run.run_stages(_config(tmp_path), CYCLE)
    assert "weights" not in ran and "report" not in ran


def test_mode_is_off_by_default_and_on_is_refused(tmp_path):
    assert WeightsCfg().mode == "off" and WeightsCfg().prompt_share == 0.2
    # YAML reads a bare on and off as booleans.
    assert WeightsCfg(**yaml.safe_load("mode: off")).mode == "off"
    for text in ("mode: on", "mode: 'on'", "mode: set"):
        with pytest.raises(ValueError, match="weights.mode must be off or dry_run, got .*sets no chain weights"):
            WeightsCfg(**yaml.safe_load(text))
    with pytest.raises(ValueError, match="got on"):
        PromptValidatorCfg(prompts_path=tmp_path / "p.jsonl", weights=yaml.safe_load("mode: on"))


@pytest.mark.parametrize("settings", [{"prompt_share": 1.5}, {"prompt_share": -0.1}, {"prompt_shar": 0.3}])
def test_a_share_out_of_range_or_a_misspelled_key_is_refused(settings):
    with pytest.raises(ValueError):
        WeightsCfg(**settings)


def test_the_weights_settings_are_not_in_the_rules_hash(tmp_path):
    pv = PromptValidatorCfg(prompts_path=tmp_path / "p.jsonl")
    changed = pv.model_copy(update={"weights": WeightsCfg(mode="dry_run", prompt_share=0.5)})
    assert snapshot.rules_sha256(changed, BASE) == snapshot.rules_sha256(pv, BASE)
