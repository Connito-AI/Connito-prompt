import ctypes
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import bittensor
import pytest
import yaml
from bittensor.sp_core import Keypair, save_password_to_environment

from connito.prompt_validator import run
from connito.prompt_validator.config import PromptValidatorCfg, PromptValidatorConfig
from connito.prompt_validator.gpu_env import require_gpu_env

REPO = Path(__file__).resolve().parents[2]
EXPERT_GROUPS = REPO / "expert_groups"

PCI = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PV_ALLOWED_GPUS": "2,3,4"}


@pytest.mark.parametrize("gpu", ["2", "3", "4"])
def test_one_of_the_allowed_gpus_in_pci_bus_order_is_accepted(gpu):
    require_gpu_env({**PCI, "CUDA_VISIBLE_DEVICES": gpu})


@pytest.mark.parametrize("allowed", [None, "", " , ", "two", "2,x", "02", "-1"])
def test_with_no_allowed_gpus_named_every_gpu_is_refused(allowed, capsys):
    env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "2", "PV_HOST_GPU": "2"}
    if allowed is not None:
        env["PV_ALLOWED_GPUS"] = allowed
    for in_container in (False, True):
        with pytest.raises(SystemExit) as exit_:
            require_gpu_env(env, in_container=in_container)
        assert exit_.value.code == 2
    assert "set PV_ALLOWED_GPUS to the GPUs this process may use" in capsys.readouterr().err


def test_the_allowed_gpus_are_the_ones_named():
    env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PV_ALLOWED_GPUS": " 1, 7 "}
    for gpu in ("1", "7"):
        require_gpu_env({**env, "CUDA_VISIBLE_DEVICES": gpu})
    for gpu in ("0", "2", "1,7"):
        with pytest.raises(SystemExit):
            require_gpu_env({**env, "CUDA_VISIBLE_DEVICES": gpu})


def test_the_compose_file_names_the_allowed_gpus():
    # Without it the container's loop refuses to start: the deployment, not the code, says which GPUs are ours.
    path = REPO / "connito" / "prompt_validator" / "docker" / "docker-compose.yml"
    if not path.exists():
        pytest.skip("no deployment files in this checkout")
    compose = path.read_text()
    assert 'PV_ALLOWED_GPUS: "2,3,4"' in compose


@pytest.mark.parametrize("env", [
    {"PV_ALLOWED_GPUS": "2,3,4"},
    PCI,
    {**PCI, "CUDA_VISIBLE_DEVICES": ""},
    {**PCI, "CUDA_VISIBLE_DEVICES": "0"},
    {**PCI, "CUDA_VISIBLE_DEVICES": "1"},
    {**PCI, "CUDA_VISIBLE_DEVICES": "5"},
    {**PCI, "CUDA_VISIBLE_DEVICES": "02"},
    {**PCI, "CUDA_VISIBLE_DEVICES": "2,3"},
    {**PCI, "CUDA_VISIBLE_DEVICES": "0,1"},
    {**PCI, "CUDA_VISIBLE_DEVICES": "GPU-8f1c2a7e"},
    {"CUDA_VISIBLE_DEVICES": "2", "PV_ALLOWED_GPUS": "2,3,4"},
    {"CUDA_DEVICE_ORDER": "FASTEST_FIRST", "CUDA_VISIBLE_DEVICES": "2", "PV_ALLOWED_GPUS": "2,3,4"},
])
def test_any_other_gpu_env_is_refused_with_status_2_and_no_traceback(env, capsys):
    with pytest.raises(SystemExit) as exit_:
        require_gpu_env(env)

    err = capsys.readouterr().err
    assert exit_.value.code == 2 and "Traceback" not in err
    assert "one of the GPUs PV_ALLOWED_GPUS allows (2, 3, 4)" in err and "a .env is read too late" in err


@pytest.mark.parametrize("gpu", ["2", "3", "4"])
def test_in_a_container_device_0_is_accepted_as_the_host_gpu_named(gpu):
    require_gpu_env({"CUDA_VISIBLE_DEVICES": "0", "PV_HOST_GPU": gpu, "PV_ALLOWED_GPUS": "2,3,4"}, in_container=True)


@pytest.mark.parametrize("env, in_container", [
    ({"CUDA_VISIBLE_DEVICES": "0", "PV_HOST_GPU": "4"}, False),  # on the host, device 0 is GPU 0
    ({"CUDA_VISIBLE_DEVICES": "0"}, True),
    ({"CUDA_VISIBLE_DEVICES": "0", "PV_HOST_GPU": "0"}, True),
    ({"CUDA_VISIBLE_DEVICES": "0", "PV_HOST_GPU": "1"}, True),
    ({"CUDA_VISIBLE_DEVICES": "0", "PV_HOST_GPU": "5"}, True),
    ({"CUDA_VISIBLE_DEVICES": "0,1", "PV_HOST_GPU": "4"}, True),
    ({"CUDA_VISIBLE_DEVICES": "1", "PV_HOST_GPU": "4"}, True),
    ({"PV_HOST_GPU": "4"}, True),
])
def test_in_a_container_anything_else_is_refused(env, in_container):
    with pytest.raises(SystemExit) as exit_:
        require_gpu_env({**env, "PV_ALLOWED_GPUS": "2,3,4"}, in_container=in_container)
    assert exit_.value.code == 2


def test_the_entry_point_refuses_a_bad_gpu_env_before_it_imports_the_stages(tmp_path):
    # As python -m runs it; the stages import fla, which starts CUDA. Nothing here may see a GPU, so CUDA_* is unset.
    code = (
        "import runpy, sys\n"
        "sys.argv = ['run', '--path', 'x.yaml']\n"
        "try:\n"
        "    runpy.run_module('connito.prompt_validator.run', run_name='__main__')\n"
        "except SystemExit as e:\n"
        "    print(e.code, 'connito.prompt_validator.answers' in sys.modules, 'torch' in sys.modules)\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUDA_")}
    env.update(PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1", PV_ALLOWED_GPUS="2,3,4")

    done = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=120)

    assert done.stdout.split() == ["2", "False", "False"]
    assert "one of the GPUs PV_ALLOWED_GPUS allows" in done.stderr


def test_the_run_checks_the_gpu_env_before_it_reads_the_config(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(run.PromptValidatorConfig, "from_path", lambda *a, **k: pytest.fail("config read"))
    monkeypatch.setattr("sys.argv", ["run", "--path", "x.yaml", "--stage", "report"])

    with pytest.raises(SystemExit) as exit_:
        run.main()
    assert exit_.value.code == 2


def test_set_weights_is_off_and_cannot_be_turned_on(tmp_path):
    assert PromptValidatorCfg(prompts_path=tmp_path / "p.jsonl").set_weights is False
    with pytest.raises(ValueError, match="set_weights must be false: the prompt validator sets no chain weights"):
        PromptValidatorCfg(prompts_path=tmp_path / "p.jsonl", set_weights=True)


def _write_config(tmp_path, **sections):
    data = {
        "run": {"root_path": str(tmp_path)},
        "chain": {"coldkey_ss58": "cold", "hotkey_ss58": "hot"},  # skips the chain lookup
        "task": {"base_path": str(EXPERT_GROUPS)},
        "prompt_validator": {"prompts_path": "/abs/prompts.jsonl", "write_root": str(tmp_path)},
        **sections,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_a_config_whose_locked_fields_differ_from_the_code_loads_without_asking_or_rewriting_it(tmp_path, monkeypatch,
                                                                                               pv_chain):
    # The owner's live cycle lengths differ from the code's defaults; cutoff checks them against cycle-api instead.
    path = _write_config(tmp_path, cycle={"train_period": 340, "commit_period": 11, "submission_period": 60},
                         chain={**pv_chain, "network": "wss://archive.example:443"})
    before = path.read_text()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("the config asked on stdin"))

    config = PromptValidatorConfig.from_path(path, active_task=None)

    assert (config.cycle.train_period, config.chain.network) == (340, "wss://archive.example:443")
    assert path.read_text() == before


def test_the_production_jev_section_loads_and_a_leftover_jev_model_is_refused(tmp_path, pv_chain):
    pv = {"prompts_path": "/abs/prompts.jsonl", "write_root": str(tmp_path)}
    path = _write_config(tmp_path, chain=pv_chain,
                         prompt_validator={**pv, "jev": {"spend_cap_usd": 5.0, "resolved": "jev-1.14.0"}})
    assert PromptValidatorConfig.from_path(path, active_task=None).prompt_validator.jev.resolved == "jev-1.14.0"

    path = _write_config(tmp_path, chain=pv_chain, prompt_validator={**pv, "jev": {"model": "jev-latest"}})
    with pytest.raises(ValueError, match="jev.model is gone: Jev is asked for jev.resolved by name"):
        PromptValidatorConfig.from_path(path, active_task=None)


PASSWORD = "secret-password"


def _encoded(name, password):
    """bittensor's encoding of a password for the variable `name`, which it sets only in the C-level environment."""
    save_password_to_environment(name, password)
    libc = ctypes.CDLL(None)
    libc.getenv.restype = ctypes.c_char_p
    value = libc.getenv(name.encode()).decode()
    os.unsetenv(name)
    return value


@pytest.fixture
def encrypted_hotkey(tmp_path, monkeypatch):
    """A config whose wallet, in a wallet_path under tmp_path, has an encrypted hotkey: (config path, variable)."""
    wallets = str(tmp_path / "wallets")
    hotkey = bittensor.Wallet(name="cold", hotkey="hot", path=wallets).hotkey_file
    hotkey.set_keypair(Keypair.create_from_mnemonic(Keypair.generate_mnemonic()), encrypt=True, password=PASSWORD)
    monkeypatch.setenv(hotkey.env_var_name(), "")
    monkeypatch.delenv(hotkey.env_var_name())  # absent now, and removed again after the test whoever set it
    for prompt in ("bittensor.keyfiles.masked_input", "getpass.getpass"):
        monkeypatch.setattr(prompt, lambda *a, **k: pytest.fail("asked for the password"))
    path = _write_config(tmp_path, chain={"coldkey_name": "cold", "hotkey_name": "hot", "wallet_path": wallets,
                                          "coldkey_ss58": "cold", "hotkey_ss58": "hot"})
    return path, hotkey.env_var_name()


def test_an_encrypted_hotkey_opens_with_bittensors_encoded_password(encrypted_hotkey, monkeypatch, capsys, caplog):
    path, var = encrypted_hotkey
    encoded = _encoded(var, PASSWORD)
    monkeypatch.setenv(var, encoded)

    assert PromptValidatorConfig.from_path(path, active_task=None).chain.hotkey_name == "hot"
    logged = capsys.readouterr().out + caplog.text
    assert PASSWORD not in logged and encoded not in logged


@pytest.mark.parametrize("value", [
    None,
    "",
    " ",  # decodes to an empty password, on which bittensor would ask
    "\t",
    PASSWORD,  # the plain password, which is not bittensor's encoding
    "cGFzc3dvcmQ=",  # a valid encoding of another password
])
def test_an_encrypted_hotkey_the_variable_does_not_open_is_refused(encrypted_hotkey, monkeypatch, value):
    path, var = encrypted_hotkey
    if value is not None:
        monkeypatch.setenv(var, value)

    with pytest.raises(ValueError, match=f"{var} must hold bittensor's encoded password") as refused:
        PromptValidatorConfig.from_path(path, active_task=None)
    assert PASSWORD not in str(refused.value) and (not (value or "").strip() or value not in str(refused.value))


# --- the env file ---

SECRET = "jev-secret-value"


class Loaded(Exception):
    pass


@pytest.fixture
def started(tmp_path, monkeypatch):
    """Run main() with args, up to reading the config; returns JEV_DIRECT_API_KEY as main() left it."""
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("JEV_DIRECT_API_KEY", "")
    monkeypatch.delenv("JEV_DIRECT_API_KEY")  # absent now, and removed again after the test whoever set it

    def from_path(path, active_task):
        raise Loaded(os.environ.get("JEV_DIRECT_API_KEY"))

    monkeypatch.setattr(run.PromptValidatorConfig, "from_path", from_path)

    def start(*args):
        monkeypatch.setattr("sys.argv", ["run", "--path", str(tmp_path / "config.yaml"), *args])
        with pytest.raises(Loaded) as loaded:
            run.main()
        return loaded.value.args[0]

    return start


def test_the_env_file_next_to_the_config_is_loaded_and_its_values_never_logged(tmp_path, started, capsys, caplog):
    (tmp_path / ".env").write_text(f"JEV_DIRECT_API_KEY={SECRET}\n")

    assert started() == SECRET
    logged = capsys.readouterr().out + caplog.text
    assert "env file loaded" in logged and SECRET not in logged


def test_a_variable_already_set_is_not_overridden_by_the_env_file(tmp_path, started, monkeypatch):
    (tmp_path / ".env").write_text(f"JEV_DIRECT_API_KEY={SECRET}\n")
    monkeypatch.setenv("JEV_DIRECT_API_KEY", "from-the-environment")

    assert started() == "from-the-environment"


def test_env_file_names_another_file(tmp_path, started):
    (tmp_path / ".env").write_text("JEV_DIRECT_API_KEY=next-to-the-config\n")
    (tmp_path / "secrets.env").write_text(f"JEV_DIRECT_API_KEY={SECRET}\n")

    assert started("--env-file", str(tmp_path / "secrets.env")) == SECRET


def test_no_env_file_is_fine_but_a_named_one_must_exist(tmp_path, started):
    assert started() is None
    with pytest.raises(SystemExit) as exit_:
        started("--env-file", str(tmp_path / "missing.env"))
    assert exit_.value.code == 2


def test_keys_read_on_import_are_ignored_with_a_warning_that_names_them(tmp_path, started, monkeypatch, capsys,
                                                                      caplog):
    late = ("CUDA_LAUNCH_BLOCKING", "HF_HOME", "HF_HUB_OFFLINE")
    for key in (*late, "HF_TOKEN"):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)  # absent now, and removed again after the test whoever set it
    (tmp_path / ".env").write_text(
        f"JEV_DIRECT_API_KEY={SECRET}\nHF_TOKEN=hf-token-value\nHF_HOME=/hf-home-value\nHF_HUB_OFFLINE=1\n"
        "CUDA_VISIBLE_DEVICES=0\nCUDA_LAUNCH_BLOCKING=1\n")

    assert started() == SECRET
    assert os.environ["HF_TOKEN"] == "hf-token-value" and os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    assert not any(key in os.environ for key in late)
    logged = capsys.readouterr().out + caplog.text
    assert "env file keys ignored" in logged and all(key in logged for key in (*late, "CUDA_VISIBLE_DEVICES"))
    assert not any(value in logged for value in ("/hf-home-value", "hf-token-value", SECRET))


def test_hf_token_from_the_env_file_reaches_huggingface_hub_for_the_fetch_and_is_never_logged(tmp_path, started,
                                                                                               monkeypatch, capsys,
                                                                                               caplog):
    import huggingface_hub

    from connito.prompt_validator import fetch
    from connito.shared import hf_distribute
    from connito.shared.config import HfCfg

    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.delenv("HF_TOKEN")  # absent now, and removed again after the test whoever set it
    (tmp_path / ".env").write_text("HF_TOKEN=hf_token_from_the_env_file\n")
    sent = []
    # What the fetch's download child runs, in this process: it resolves the token from the environment it inherits.
    monkeypatch.setattr(fetch, "download_checkpoint_from_hf_subprocess",
                        lambda timeout_sec, **kwargs: hf_distribute.download_checkpoint_from_hf(**kwargs))
    monkeypatch.setattr(hf_distribute, "hf_hub_download", lambda **kwargs: sent.append(kwargs["token"]))

    def from_path(path, active_task):  # once the env file is read, as the stages run after it
        config = SimpleNamespace(hf=HfCfg(), evaluation=SimpleNamespace(per_miner_download_timeout_sec=60))
        reason = fetch._download(config, "miner/repo", "0" * 40, "model.safetensors", tmp_path / "dl")
        # The base model and tokenizer load through huggingface_hub's own lookup, read when each call is made.
        raise Loaded((reason, huggingface_hub.get_token()))

    monkeypatch.setattr(run.PromptValidatorConfig, "from_path", from_path)
    assert started() == (None, "hf_token_from_the_env_file")
    assert sent == ["hf_token_from_the_env_file"]
    assert "hf_token_from_the_env_file" not in capsys.readouterr().out + caplog.text


# --- logging from the process environment ---

def test_log_env_sets_the_logging_and_unset_keeps_the_default(tmp_path, started, monkeypatch):
    chosen = []
    monkeypatch.setattr(run, "configure_logging", lambda **kwargs: chosen.append(kwargs))
    for key in ("LOG_FORMAT", "LOG_LEVEL", "LOG_UTC"):
        monkeypatch.delenv(key, raising=False)

    started()
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_UTC", "0")
    started()
    assert chosen == [{"fmt": "pretty", "level": "INFO", "use_utc": True},
                      {"fmt": "json", "level": "DEBUG", "use_utc": False}]


@pytest.mark.parametrize(("key", "value"), [("LOG_FORMAT", "jsonl"), ("LOG_LEVEL", "info"), ("LOG_UTC", "yes")])
def test_a_bad_log_env_value_exits_2_before_the_config_is_read(tmp_path, started, monkeypatch, capsys, key, value):
    monkeypatch.setattr(run, "configure_logging", lambda **kwargs: pytest.fail("configured"))
    monkeypatch.setenv(key, value)

    with pytest.raises(SystemExit) as exit_:
        started()
    assert exit_.value.code == 2 and repr(value) in capsys.readouterr().err


def test_log_keys_in_the_env_file_are_ignored_with_a_warning(tmp_path, started, monkeypatch, capsys, caplog):
    monkeypatch.setattr(run, "configure_logging", lambda **kwargs: None)
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    (tmp_path / ".env").write_text("LOG_FORMAT=json\n")

    started()
    assert "LOG_FORMAT" not in os.environ
    logged = capsys.readouterr().out + caplog.text
    assert "env file keys ignored" in logged and "LOG_FORMAT" in logged


@pytest.mark.parametrize(("level", "accepted"), [("DEBUG", True), ("WARNING", True), ("CRITICAL", True),
                                                 ("info", False), ("VERBOSE", False), ("10", False)])
def test_the_log_level_check_needs_nothing_newer_than_python_3_10(tmp_path, started, monkeypatch, level, accepted):
    monkeypatch.delattr(logging, "getLevelNamesMapping", raising=False)  # Python 3.11+
    chosen = []
    monkeypatch.setattr(run, "configure_logging", lambda **kwargs: chosen.append(kwargs["level"]))
    monkeypatch.setenv("LOG_LEVEL", level)

    if accepted:
        started()
        assert chosen == [level]
    else:
        with pytest.raises(SystemExit) as exit_:
            started()
        assert exit_.value.code == 2 and not chosen


@pytest.mark.parametrize("layout", [False, True])
def test_in_a_container_the_config_takes_our_images_layout_only_when_it_is_there(tmp_path, monkeypatch, pv_chain,
                                                                                 layout):
    # Our images have /data and /app/expert_groups; a checkout run in a plain python container has neither and keeps
    # its own root and task configs.
    from connito.shared import config as shared_config
    data, groups = tmp_path / "data", tmp_path / "app" / "expert_groups"
    if layout:
        (data / "checkpoints").mkdir(parents=True)
        (groups / "exp_pmc_residual").mkdir(parents=True)
        (groups / "exp_pmc_residual" / "config.yaml").write_text((EXPERT_GROUPS / "exp_pmc_residual" /
                                                                  "config.yaml").read_text())
    monkeypatch.setattr(shared_config, "is_running_in_docker", lambda: True)
    monkeypatch.setattr(shared_config, "CONTAINER_DATA_ROOT", data)
    monkeypatch.setattr(shared_config, "CONTAINER_EXPERT_GROUPS", groups)

    config = PromptValidatorConfig.from_path(_write_config(tmp_path, chain=pv_chain), active_task=None)

    assert (config.run.root_path, config.task.base_path) == ((data, groups) if layout else (tmp_path, EXPERT_GROUPS))
