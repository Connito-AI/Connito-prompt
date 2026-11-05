"""Every wallet the prompt validator opens is under chain.wallet_path, and an empty one keeps bittensor's default."""
from types import SimpleNamespace

import bittensor
import pytest
import yaml
from pydantic import ValidationError
from test_prompt_validator_dataless import (
    CYCLE_ID,
    _config,
    cycle,  # noqa: F401  (a fixture)
    fixed_base_rows,  # noqa: F401  (autouse here too: the report's rules check reads the base rows)
    no_sleep,  # noqa: F401  (a fixture, which cycle uses)
    snapshot_api,  # noqa: F401  (a fixture)
)
from test_prompt_validator_dataless_test_mode import EXAMPLE
from test_prompt_validator_run import EXPERT_GROUPS, _write_config

from connito.prompt_validator import report, snapshot
from connito.prompt_validator.config import PromptValidatorChainCfg, PromptValidatorConfig
from connito.shared import config as shared_config
from connito.shared.config import ChainCfg, ValidatorConfig, WorkerConfig, convert_to_str

WALLETS = "/scratch/pv/wallets"


def _record(monkeypatch) -> list[dict]:
    """Replaces bittensor.Wallet with a fake holding no key; returns the keyword arguments of every call."""
    calls = []

    def wallet(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(hotkey_file=SimpleNamespace(exists_on_device=lambda: True, is_encrypted=lambda: False),
                               hotkey=SimpleNamespace(ss58_address="hk"), coldkeypub=SimpleNamespace(ss58_address="ck"))

    monkeypatch.setattr(bittensor, "Wallet", wallet)
    return calls


def _expected(name, hotkey, wallet_path):
    return {"name": name, "hotkey": hotkey, **({"path": wallet_path} if wallet_path else {})}


both = pytest.mark.parametrize("wallet_path", ["", WALLETS])


@both
def test_the_snapshot_signs_with_the_wallet_under_wallet_path(tmp_path, snapshot_api, monkeypatch,  # noqa: F811
                                                              wallet_path):
    snapshot_api()
    calls = _record(monkeypatch)
    config = _config(tmp_path)
    config.chain.wallet_path = wallet_path

    snapshot.load_prompts(config, CYCLE_ID)

    assert calls == [_expected("c", "h", wallet_path)]


@both
def test_the_report_signs_with_the_wallet_under_wallet_path(cycle, monkeypatch, wallet_path):  # noqa: F811
    config, _ = cycle
    calls = _record(monkeypatch)
    config.chain.wallet_path = wallet_path

    report.run_report(config, CYCLE_ID)

    assert calls == [_expected("c", "h", wallet_path)]


@both
def test_the_config_checks_the_hotkey_under_wallet_path(tmp_path, monkeypatch, wallet_path):
    calls = _record(monkeypatch)
    chain = {"coldkey_name": "cold", "hotkey_name": "hot", "coldkey_ss58": "ck", "hotkey_ss58": "hk"}
    path = _write_config(tmp_path, chain={**chain, **({"wallet_path": wallet_path} if wallet_path else {})})

    PromptValidatorConfig.from_path(path, active_task=None)

    assert calls == [_expected("cold", "hot", wallet_path)]


@pytest.mark.parametrize(("chain", "wallet_path"), [
    (ChainCfg(coldkey_name="cold", hotkey_name="hot"), ""),
    (PromptValidatorChainCfg(coldkey_name="cold", hotkey_name="hot"), ""),
    (PromptValidatorChainCfg(coldkey_name="cold", hotkey_name="hot", wallet_path=WALLETS), WALLETS),
])
def test_the_chain_lookup_opens_the_wallet_under_wallet_path(monkeypatch, chain, wallet_path):
    calls = _record(monkeypatch)
    monkeypatch.setattr(bittensor, "Subtensor", lambda **kwargs: None)
    monkeypatch.setattr(shared_config, "get_metagraph", lambda subtensor, netuid: SimpleNamespace(hotkeys=["x", "hk"]))
    config = SimpleNamespace(chain=chain.model_copy())

    WorkerConfig._fill_wallet_data(config)

    assert calls == [_expected("cold", "hot", wallet_path)]
    assert (config.chain.hotkey_ss58, config.chain.coldkey_ss58, config.chain.uid) == ("hk", "ck", 1)


# The chain keys a training validator's or miner's config has before this field: a shared one would be asked for.
CHAIN_KEYS = ("netuid", "uid", "hotkey_ss58", "coldkey_ss58", "ip", "port", "coldkey_name", "hotkey_name", "network",
              "lite_network")


def test_a_training_validator_config_has_no_wallet_path_and_asks_for_no_new_field(tmp_path, monkeypatch):
    config = ValidatorConfig(run={"root_path": str(tmp_path)}, chain={"coldkey_ss58": "c", "hotkey_ss58": "h"},
                             task={"base_path": str(EXPERT_GROUPS)})
    data = convert_to_str(config.model_dump())
    data["chain"] = {key: data["chain"][key] for key in CHAIN_KEYS}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    before = path.read_text()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail(f"the config asked: {a}"))

    loaded = ValidatorConfig.from_path(path, active_task=None)

    assert type(loaded.chain) is ChainCfg and set(ChainCfg.model_fields) == set(CHAIN_KEYS)
    assert path.read_text() == before


def test_the_example_configs_wallet_path_line_sets_it():
    text = EXAMPLE.read_text()
    assert text.count("# wallet_path: ") == 1

    chain = yaml.safe_load(text.replace("# wallet_path: ", "wallet_path: "))["chain"]

    assert PromptValidatorChainCfg(**chain).wallet_path == "/path/to/wallets"


@pytest.mark.parametrize("wallet_path", ["wallets", "./wallets", "../scratch/wallets"])
def test_a_relative_wallet_path_is_refused(wallet_path):
    with pytest.raises(ValidationError, match="chain.wallet_path must be absolute"):
        PromptValidatorChainCfg(wallet_path=wallet_path)


@pytest.mark.parametrize("wallet_path", ["", "/scratch/pv/wallets", "~/wallets"])
def test_an_absolute_or_home_wallet_path_is_kept_as_written(wallet_path):
    assert PromptValidatorChainCfg(wallet_path=wallet_path).wallet_path == wallet_path


def test_a_wallet_path_with_no_hotkey_is_refused_before_anything_else_reads_the_wallet(tmp_path):
    wallets = tmp_path / "wallets"
    chain = {"coldkey_name": "cold", "hotkey_name": "hot", "wallet_path": str(wallets),
             "coldkey_ss58": "ck", "hotkey_ss58": "hk"}

    with pytest.raises(ValueError, match=f"no hotkey at {wallets}/cold/hotkeys/hot: check chain.wallet_path"):
        PromptValidatorConfig.from_path(_write_config(tmp_path, chain=chain), active_task=None)


def test_a_config_whose_plain_hotkey_is_under_wallet_path_loads(tmp_path, pv_chain):
    config = PromptValidatorConfig.from_path(_write_config(tmp_path, chain=pv_chain), active_task=None)

    assert config.chain.wallet().hotkey_file.exists_on_device()
