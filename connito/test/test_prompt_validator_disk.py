"""A published cycle's shards are deleted, and nothing else; the validator writes and deletes only under write_root."""
import argparse
import logging
import re
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub.constants
import pytest
from pydantic import ValidationError
from test_prompt_validator_dataless import (
    CYCLE_ID,
    LENGTH,
    cycle,  # noqa: F401  (a fixture)
    fixed_base_rows,  # noqa: F401  (autouse here too: the report's rules check reads the base rows)
    no_sleep,  # noqa: F401  (a fixture, which cycle uses)
)
from test_prompt_validator_run import _write_config

from connito.prompt_validator import fetch, report, run, snapshot
from connito.prompt_validator.config import PromptValidatorCfg, PromptValidatorConfig
from connito.prompt_validator.stage_io import (
    append_published,
    published_cycles,
    read_rows,
    run_lock,
    stage_path,
    write_rows,
)

SHARD = b"x" * 1000


@pytest.fixture
def config(cycle):  # noqa: F811  (the imported fixture)
    """A scoring cycle up to reward, reporting to a fake dataless-api, with write_root its tmp_path."""
    return cycle[0]


def _logged(caplog, event: str, fields: str) -> bool:
    """`event` was logged with `fields` first among its key=value fields (the renderer pads the event)."""
    return re.search(rf"{re.escape(event)}\s+{re.escape(fields)}", caplog.text) is not None


def _shards(pv, cycle_id, sources=("aaaa", "bbbb")) -> Path:
    """Fetch's shard dirs for `cycle_id`, one file of 1000 bytes in each."""
    models = stage_path(pv.runs_dir, cycle_id, "fetch").parent / "models"
    for source in sources:
        (models / source).mkdir(parents=True)
        (models / source / "model_expgroup_9.safetensors").write_bytes(SHARD)
    return models


def _keep(tmp_path, pv) -> list[Path]:
    """Files the prune must never touch: this cycle's other stage files, another cycle's shards, a base model in an
    HF cache and a checkpoint, all under the same root."""
    other = _shards(pv, CYCLE_ID - 1)
    base = tmp_path / "hf" / "hub" / "models--Qwen--Qwen3.6-35B-A3B" / "snapshots" / "0" / "model.safetensors"
    ckpt = tmp_path / "ckpt" / "model.safetensors"
    for path in (base, ckpt):
        path.parent.mkdir(parents=True)
        path.write_bytes(SHARD)
    fetch_dir = stage_path(pv.runs_dir, CYCLE_ID, "fetch").parent
    (fetch_dir / "progress.jsonl").write_text("{}\n")
    stages = [stage_path(pv.runs_dir, CYCLE_ID, name) for name in ("snapshot", "fetch", "answers", "scores", "reward")]
    # The snapshot's models, and the chain commits cached for later snapshots' lookback.
    models, cache = stages[0].parent / "models.jsonl", Path(pv.runs_dir) / "chain_commits" / "1-2.jsonl"
    cache.parent.mkdir()
    cache.write_text("")
    return [*other.glob("*/*"), base, ckpt, fetch_dir / "progress.jsonl", *stages, models, cache]


CACHE_ENV = ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH")
VLLM_ENV = ("VLLM_CACHE_ROOT", "VLLM_CONFIG_ROOT")  # checked only with engine "vllm"


def _caches_under(monkeypatch, root):
    """Every cache and temporary directory this process's libraries write, under root."""
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE"):
        monkeypatch.setattr(huggingface_hub.constants, name, str(root / name))
    for name in CACHE_ENV + VLLM_ENV:
        monkeypatch.setenv(name, str(root / name))
    monkeypatch.setattr(tempfile, "tempdir", str(root / "tmp"))  # what gettempdir() made of TMPDIR


def _main(monkeypatch, tmp_path, config, *args):
    _caches_under(monkeypatch, config.prompt_validator.write_root)
    monkeypatch.setattr(run.PromptValidatorConfig, "from_path", lambda path, active_task: config)
    monkeypatch.setattr("sys.argv", ["run", "--path", str(tmp_path / "config.yaml"), *args])
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    run.main()


def test_a_plain_report_run_prunes_only_its_cycles_shards_once_published(config, tmp_path, monkeypatch, caplog):
    pv = config.prompt_validator
    models = _shards(pv, CYCLE_ID)
    keep = _keep(tmp_path, pv)

    with caplog.at_level(logging.INFO):
        _main(monkeypatch, tmp_path, config, "--cycle-id", str(CYCLE_ID), "--stage", "report")

    assert CYCLE_ID in published_cycles(pv.runs_dir)
    assert not models.exists()
    assert all(path.is_file() for path in keep)
    assert stage_path(pv.runs_dir, CYCLE_ID, "report").exists()
    assert _logged(caplog, "shards pruned", f"bytes=2000 files=2 scoring_cycle_id={CYCLE_ID}")


def _loop_once(monkeypatch, tmp_path, config):
    """`--loop --max-cycles 1` with CYCLE_ID, every stage done but its report, as the cycle before the open one."""
    write_rows(stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "cutoff"), [{}])
    block = (CYCLE_ID + 1) * LENGTH + 2000
    phase = SimpleNamespace(block=block, cycle_index=block // 524)
    monkeypatch.setattr(run, "get_phase_from_api", lambda cfg: phase)
    monkeypatch.setattr(run, "check_phase_clock", lambda cfg, p: p)
    # A failed cycle fails the test: no_sleep would let the loop retry it forever.
    monkeypatch.setattr(run, "time", SimpleNamespace(sleep=lambda sec: pytest.fail("a cycle failed"),
                                                     monotonic=time.monotonic))
    _main(monkeypatch, tmp_path, config, "--loop", "--max-cycles", "1")


def test_the_loop_prunes_cycles_published_before_it_started_and_each_it_publishes(config, tmp_path, monkeypatch,
                                                                                  caplog):
    pv = config.prompt_validator
    models = _shards(pv, CYCLE_ID)
    keep = _keep(tmp_path, pv)  # CYCLE_ID - 1 is not published: its shards stay
    old, busy = _shards(pv, 1250), _shards(pv, 1251)
    for published in (1250, 1251):
        append_published(pv.runs_dir, {"cycle_id": published})

    with run_lock(pv.runs_dir / "1251" / ".lock", "a --stage run"), caplog.at_level(logging.INFO):
        _loop_once(monkeypatch, tmp_path, config)

    assert not old.exists() and not models.exists()
    assert len(list(busy.glob("*/*"))) == 2  # its lock was held: never pruned under a running stage
    assert _logged(caplog, "shards not pruned: the cycle is in use", "scoring_cycle_id=1251")
    assert all(path.is_file() for path in keep)
    assert read_rows(pv.runs_dir / "published.jsonl")[-1]["cycle_id"] == CYCLE_ID
    # The sweep runs first, then the cycle, which prunes its own right after its ledger row.
    pruned = [line.split("scoring_cycle_id=")[1][:4] for line in caplog.text.splitlines() if "shards pruned" in line]
    assert pruned == ["1250", str(CYCLE_ID)]


def test_a_sweep_that_cannot_lock_a_cycle_is_a_warning_and_the_loop_goes_on(config, tmp_path, monkeypatch, caplog):
    pv = config.prompt_validator
    old, models = _shards(pv, 1250), _shards(pv, CYCLE_ID)
    append_published(pv.runs_dir, {"cycle_id": 1250})
    real_lock = run.cycle_lock

    def cycle_lock(pv, cycle_id):
        if cycle_id == 1250:
            raise PermissionError("cannot open .lock")
        return real_lock(pv, cycle_id)

    monkeypatch.setattr(run, "cycle_lock", cycle_lock)
    with caplog.at_level(logging.INFO):
        _loop_once(monkeypatch, tmp_path, config)

    assert _logged(caplog, "shards not pruned", "error=PermissionError: cannot open .lock scoring_cycle_id=1250")
    assert len(list(old.glob("*/*"))) == 2
    assert read_rows(pv.runs_dir / "published.jsonl")[-1]["cycle_id"] == CYCLE_ID and not models.exists()


def test_a_cycle_with_no_shards_logs_nothing(config, caplog):
    with caplog.at_level(logging.DEBUG):
        fetch.prune_shards(config.prompt_validator, CYCLE_ID)
    assert "shards" not in caplog.text


def test_the_prune_follows_the_ledger_row_and_never_runs_without_it(config, monkeypatch):
    pv = config.prompt_validator
    models = _shards(pv, CYCLE_ID)
    seen = []
    monkeypatch.setattr(report, "prune_shards",
                        lambda pv, cycle_id: seen.append(cycle_id in published_cycles(pv.runs_dir)))
    real_append = report.append_published

    def failing_append(runs_dir, row):
        raise OSError("disk full")

    monkeypatch.setattr(report, "append_published", failing_append)
    with pytest.raises(OSError, match="disk full"):
        report.run_report(config, CYCLE_ID)
    assert seen == [] and len(list(models.glob("*/*"))) == 2

    monkeypatch.setattr(report, "append_published", real_append)
    report.run_report(config, CYCLE_ID)  # the publish is final on the server: the rerun finds it published there
    assert seen == [True]


@pytest.mark.parametrize("found", ["ledger", "server"])
def test_a_cycle_found_already_published_is_pruned(config, found):
    pv = config.prompt_validator
    report.run_report(config, CYCLE_ID)
    if found == "server":
        (pv.runs_dir / "published.jsonl").unlink()  # a new install, or a crash before the ledger row
    models = _shards(pv, CYCLE_ID)
    assert report.run_report(config, CYCLE_ID)[0]["published_before"] == found
    assert not models.exists()


def test_a_prune_that_fails_is_a_warning_and_the_publish_stands(config, monkeypatch, caplog):
    pv = config.prompt_validator
    models = _shards(pv, CYCLE_ID)

    def rmtree(path, *a, **k):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(fetch.shutil, "rmtree", rmtree)
    with caplog.at_level(logging.WARNING):
        assert report.run_report(config, CYCLE_ID)[0]["status"] == "published"
    assert CYCLE_ID in published_cycles(pv.runs_dir) and models.exists()
    assert _logged(caplog, "shards not pruned", f"error=PermissionError: read-only file system path={models}")


OTHER = 1300  # a published cycle whose stage files the escape cases replace


def _escape(tmp_path, pv, how) -> Path:
    """Leads OTHER's shards out of its fetch/models; returns a file there the prune must keep."""
    outside = tmp_path / "elsewhere"
    victim = outside / "fetch" / "models" / "aaaa" / "model_expgroup_9.safetensors"
    victim.parent.mkdir(parents=True)
    victim.write_bytes(SHARD)
    cycle_dir = pv.runs_dir / str(OTHER)
    if how == "models symlink":
        (cycle_dir / "fetch").mkdir(parents=True)
        (cycle_dir / "fetch" / "models").symlink_to(outside / "fetch" / "models")
    elif how == "fetch symlink":
        cycle_dir.mkdir(parents=True)
        (cycle_dir / "fetch").symlink_to(outside / "fetch")
    else:  # a runs_dir whose .. leads out of write_root, with the shards where they belong in it
        pv.write_root, pv.runs_dir = pv.runs_dir, pv.runs_dir / ".." / "elsewhere"
        (outside / str(OTHER)).mkdir()
        (outside / "fetch").rename(outside / str(OTHER) / "fetch")
        victim = outside / str(OTHER) / "fetch" / "models" / "aaaa" / victim.name
    return victim


@pytest.mark.parametrize("how", ["models symlink", "fetch symlink", "runs_dir outside the root"])
def test_a_prune_never_follows_a_path_out_of_its_cycles_fetch_models(tmp_path, config, how, caplog):
    pv = config.prompt_validator
    victim = _escape(tmp_path, pv, how)
    with caplog.at_level(logging.WARNING):
        fetch.prune_shards(pv, OTHER)
    assert victim.read_bytes() == SHARD
    assert _logged(caplog, "shards not pruned", "error=ValueError: ")


def test_a_symlink_inside_the_shards_is_removed_not_followed(tmp_path, config):
    pv = config.prompt_validator
    models = _shards(pv, CYCLE_ID)
    target = tmp_path / "base" / "model.safetensors"
    target.parent.mkdir()
    target.write_bytes(SHARD)
    (models / "aaaa" / "link").symlink_to(target)
    (models / "dirlink").symlink_to(target.parent)
    fetch.prune_shards(pv, CYCLE_ID)
    assert not models.exists() and target.read_bytes() == SHARD


@pytest.fixture
def root(tmp_path) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    return root


def _load(tmp_path, pv_chain, write_root, **pv):
    path = _write_config(tmp_path, chain=pv_chain, prompt_validator={
        "prompts_path": "/abs/prompts.jsonl", "write_root": str(write_root), **pv,
    })
    return PromptValidatorConfig.from_path(path, active_task=None)


@pytest.mark.parametrize("runs_dir", ["/elsewhere/runs", "{root}/../runs", "{root}/link/runs", "{root}2/runs"])
def test_a_runs_dir_outside_write_root_is_refused_at_start(tmp_path, pv_chain, root, runs_dir):
    (root / "link").symlink_to(tmp_path)  # under the root by name, outside it once resolved
    with pytest.raises(ValidationError, match=r"outside prompt_validator.write_root .*: the validator writes and "
                                              "deletes only under it"):
        _load(tmp_path, pv_chain, root, runs_dir=runs_dir.format(root=root))


def test_a_runs_dir_under_write_root_loads_and_write_root_must_be_absolute(tmp_path, pv_chain, root):
    runs_dir = root / "a" / ".." / "runs"
    assert _load(tmp_path, pv_chain, root, runs_dir=str(runs_dir)).prompt_validator.runs_dir == runs_dir
    with pytest.raises(ValidationError, match="write_root must be an absolute path"):
        _load(tmp_path, pv_chain, "scratch")


def test_the_default_write_root_is_scratch(tmp_path, pv_chain):
    # runs_dir outside /scratch whatever the host: its default is under tmp_path, which may itself be under /scratch.
    path = _write_config(tmp_path, chain=pv_chain,
                         prompt_validator={"prompts_path": "/abs/prompts.jsonl", "runs_dir": "/abs/runs"})
    with pytest.raises(ValidationError, match="outside prompt_validator.write_root /scratch"):
        PromptValidatorConfig.from_path(path, active_task=None)


def _outside(monkeypatch, outside, how):
    """Moves one cache out of the root: by its variable, or by the default it falls back to."""
    if how in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE"):
        monkeypatch.setattr(huggingface_hub.constants, how, str(outside))
    elif how == "TMPDIR":
        monkeypatch.setattr(tempfile, "tempdir", str(outside))
    elif how == "TORCHINDUCTOR_CACHE_DIR default":  # unset: under the temporary directory
        monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR")
        monkeypatch.setattr(tempfile, "tempdir", str(outside))
    elif how == "TRITON_HOME":  # TRITON_CACHE_DIR unset: under TRITON_HOME
        monkeypatch.delenv("TRITON_CACHE_DIR")
        monkeypatch.setenv("TRITON_HOME", str(outside))
    elif how.endswith("default"):  # unset: under HOME
        monkeypatch.delenv(how.split()[0])
        monkeypatch.delenv("TRITON_HOME", raising=False)
        monkeypatch.setenv("HOME", str(outside))
    else:
        monkeypatch.setenv(how, str(outside))


def _run(monkeypatch, tmp_path, root, *args, **pv) -> list:
    """main() on a config whose runs_dir and write_root are root, with every cache under it unless a test moves
    one; returns the run_stages calls."""
    calls = []
    monkeypatch.setattr(run, "run_stages", lambda cfg, cycle_id, stage=None: calls.append((cycle_id, stage)))
    pv = {"prompts_path": root / "p.jsonl", **pv}
    config = SimpleNamespace(prompt_validator=PromptValidatorCfg(runs_dir=root / "runs", write_root=root, **pv))
    monkeypatch.setattr(run.PromptValidatorConfig, "from_path", lambda path, active_task: config)
    monkeypatch.setattr("sys.argv", ["run", "--path", str(tmp_path / "config.yaml"), *args])
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    run.main()
    return calls


def _refused(capsys, outside) -> dict[str, str]:
    """The variables the refusal names as resolving to `outside`, each with how it was set."""
    err = capsys.readouterr().err
    assert "caches outside prompt_validator.write_root " in err
    assert "Set each under it in the process environment" in err
    return dict(re.findall(r"([A-Z_]+) \(('[^']*'|unset, so the default)\) -> " + re.escape(str(outside)), err))


@pytest.mark.parametrize(("how", "named"), [
    *((name, [name]) for name in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", *CACHE_ENV, "TMPDIR")),
    ("TRITON_HOME", ["TRITON_CACHE_DIR"]),  # TRITON_HOME/.triton/cache
    ("TRITON_CACHE_DIR default", ["TRITON_CACHE_DIR"]),  # ~/.triton/cache
    ("XDG_CACHE_HOME default", ["XDG_CACHE_HOME"]),  # ~/.cache
    ("CUDA_CACHE_PATH default", ["CUDA_CACHE_PATH"]),  # ~/.nv/ComputeCache
    ("TORCHINDUCTOR_CACHE_DIR default", ["TORCHINDUCTOR_CACHE_DIR", "TMPDIR"]),  # under the temporary directory
])
def test_a_cache_outside_write_root_stops_a_run_before_any_stage(tmp_path, root, monkeypatch, capsys, how, named):
    _caches_under(monkeypatch, root)
    outside = tmp_path / "outside"
    _outside(monkeypatch, outside, how)
    with pytest.raises(SystemExit) as e:
        _run(monkeypatch, tmp_path, root, "--cycle-id", "5", "--stage", "fetch")
    assert e.value.code == 2
    refused = _refused(capsys, outside)
    assert sorted(refused) == sorted(named)
    if how in CACHE_ENV:  # set outside: the value as given
        assert refused[how] == repr(str(outside))


def test_a_tmpdir_left_unset_is_named_as_the_default(tmp_path, root, monkeypatch, capsys):
    _caches_under(monkeypatch, root)
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "outside"))  # what gettempdir() fell back to
    with pytest.raises(SystemExit):
        _run(monkeypatch, tmp_path, root, "--cycle-id", "5")
    assert _refused(capsys, tmp_path / "outside") == {"TMPDIR": "unset, so the default"}


def test_every_cache_under_write_root_lets_the_run_go_on(tmp_path, root, monkeypatch):
    _caches_under(monkeypatch, root)
    assert _run(monkeypatch, tmp_path, root, "--cycle-id", "5", "--stage", "fetch") == [(5, "fetch")]
    assert _run(monkeypatch, tmp_path, root, "--cycle-id", "5") == [(5, None)]


LOOPING = {"prompt_source": "dataless", "prompts_path": None}


def test_the_loop_is_refused_at_start_and_at_a_reload(tmp_path, root, monkeypatch, capsys):
    cycles = []

    def run_cycle(cfg, cycle_id):  # its work moves a cache out: the reload after it must see that
        if cycles:
            pytest.fail("a cycle ran on a config whose caches are outside write_root")
        cycles.append(cycle_id)
        _outside(monkeypatch, tmp_path / "outside", "TMPDIR")

    monkeypatch.setattr(run, "next_cycle", lambda cfg: 5)
    monkeypatch.setattr(run, "run_cycle", run_cycle)
    _caches_under(monkeypatch, root)
    _outside(monkeypatch, tmp_path / "outside", "XDG_CACHE_HOME")
    with pytest.raises(SystemExit):
        _run(monkeypatch, tmp_path, root, "--loop", **LOOPING)
    assert cycles == [] and list(_refused(capsys, tmp_path / "outside")) == ["XDG_CACHE_HOME"]

    _caches_under(monkeypatch, root)
    with pytest.raises(SystemExit) as e:
        _run(monkeypatch, tmp_path, root, "--loop", **LOOPING)
    assert cycles == [5] and e.value.code == 2 and list(_refused(capsys, tmp_path / "outside")) == ["TMPDIR"]


VLLM_ENGINE = {"engine": "vllm", "vllm": {"python": "/venv/bin/python", "checkpoint": "/ckpt"}}


@pytest.mark.parametrize(("unset", "outside", "named"), [
    ((), "VLLM_CACHE_ROOT", ["VLLM_CACHE_ROOT"]),
    ((), "VLLM_CONFIG_ROOT", ["VLLM_CONFIG_ROOT"]),
    (("VLLM_CACHE_ROOT",), "XDG_CACHE_HOME", ["VLLM_CACHE_ROOT", "XDG_CACHE_HOME"]),  # XDG_CACHE_HOME/vllm
    (("VLLM_CONFIG_ROOT",), "XDG_CONFIG_HOME", ["VLLM_CONFIG_ROOT"]),  # XDG_CONFIG_HOME/vllm
    (("VLLM_CACHE_ROOT", "XDG_CACHE_HOME"), "HOME", ["VLLM_CACHE_ROOT", "XDG_CACHE_HOME"]),  # ~/.cache/vllm
    (("VLLM_CONFIG_ROOT", "XDG_CONFIG_HOME"), "HOME", ["VLLM_CONFIG_ROOT"]),  # ~/.config/vllm
])
@pytest.mark.parametrize("engine", ["vllm", "hf"])
def test_a_vllm_cache_outside_write_root_stops_only_a_vllm_run(tmp_path, root, monkeypatch, capsys, unset, outside,
                                                               named, engine):
    _caches_under(monkeypatch, root)
    for name in unset:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(outside, str(tmp_path / "outside"))
    named = [name for name in named if engine == "vllm" or name not in VLLM_ENV]
    pv = VLLM_ENGINE if engine == "vllm" else {}
    if not named:
        assert _run(monkeypatch, tmp_path, root, "--cycle-id", "5", "--stage", "fetch", **pv) == [(5, "fetch")]
        return
    with pytest.raises(SystemExit) as e:
        _run(monkeypatch, tmp_path, root, "--cycle-id", "5", "--stage", "fetch", **pv)
    assert e.value.code == 2
    refused = _refused(capsys, tmp_path / "outside")
    assert sorted(refused) == named
    for name in VLLM_ENV:
        if name in refused:
            assert refused[name] == ("unset, so the default" if name in unset else repr(str(tmp_path / "outside")))


@pytest.mark.parametrize(("empty", "named"), [
    ("VLLM_CACHE_ROOT", "VLLM_CACHE_ROOT"),  # the working directory
    ("VLLM_CONFIG_ROOT", "VLLM_CONFIG_ROOT"),
    ("XDG_CACHE_HOME", "VLLM_CACHE_ROOT"),  # ./vllm
    ("XDG_CONFIG_HOME", "VLLM_CONFIG_ROOT"),
])
def test_an_empty_vllm_variable_is_read_as_vllm_reads_it_not_as_unset(tmp_path, root, monkeypatch, capsys, empty,
                                                                        named):
    _caches_under(monkeypatch, root)
    monkeypatch.setenv("HOME", str(root))  # an empty XDG_* is unset to the other libraries: their ~ stays under root
    if empty.startswith("XDG_"):
        monkeypatch.delenv(named)
    monkeypatch.setenv(empty, "")
    (tmp_path / "outside").mkdir()
    monkeypatch.chdir(tmp_path / "outside")
    with pytest.raises(SystemExit):
        _run(monkeypatch, tmp_path, root, "--cycle-id", "5", "--stage", "fetch", **VLLM_ENGINE)
    assert list(_refused(capsys, tmp_path / "outside")) == [named]


def test_a_production_vllm_config_loads_and_passes_every_check(tmp_path, pv_chain, monkeypatch):
    # Production's YAML and loop environment, under tmp_path/scratch.
    scratch = tmp_path / "scratch"
    pv_dir = scratch / "prompt-validator"
    (pv_dir / "ckpt").mkdir(parents=True)
    (pv_dir / "ckpt" / "manifest.json").write_text('{"tensors": {}}')
    _caches_under(monkeypatch, scratch)
    for name, sub in (("VLLM_CACHE_ROOT", "vllm"), ("VLLM_CONFIG_ROOT", "vllm-config")):
        (pv_dir / "cache" / sub).mkdir(parents=True)
        monkeypatch.setenv(name, str(pv_dir / "cache" / sub))
    path = _write_config(tmp_path, chain=pv_chain, prompt_validator={
        "prompt_source": "dataless", "runs_dir": str(pv_dir / "runs"), "write_root": str(scratch),
        "engine": "vllm", "max_new_tokens": 2048,
        "vllm": {"python": str(pv_dir / "venv-vllm" / "bin" / "python"), "checkpoint": str(pv_dir / "ckpt"),
                 "gpu_memory_utilization": 0.85},
        "jev": {"spend_cap_usd": 5.0, "resolved": "jev-1.14.0"}, "set_weights": False,
    })
    args = argparse.Namespace(path=str(path), loop=True, test=False)

    pv = run.load_config(args, argparse.ArgumentParser()).prompt_validator

    assert (pv.engine, pv.max_new_tokens, pv.max_models, pv.jev.resolved) == ("vllm", 2048, 100, "jev-1.14.0")
    assert (pv.runs_dir, pv.write_root, pv.min_free_disk_gb) == (pv_dir / "runs", scratch, 8.0)
    assert pv.vllm.gpu_memory_utilization == 0.85
    # Every score-deciding setting of both sides of the merge is in its rules hash.
    rules = snapshot.rules_sha256(pv, {})
    for change in ({"engine": "hf"}, {"max_new_tokens": 1024}, {"max_models": 10},
                   {"jev": pv.jev.model_copy(update={"resolved": "jev-1.15.0"})},
                   *({"vllm": pv.vllm.model_copy(update=c)} for c in
                     ({"version": "0.33.0"}, {"max_num_seqs": 64}, {"enforce_eager": True}))):
        assert snapshot.rules_sha256(pv.model_copy(update=change), {}) != rules, change
    for name in ("MODEL_SELECTION", "COPY_REWARD"):
        with monkeypatch.context() as m:
            m.setattr(snapshot, name, "another rule")
            assert snapshot.rules_sha256(pv, {}) != rules, name
    assert snapshot.rules_sha256(pv.model_copy(update={"vllm": pv.vllm.model_copy(
        update={"gpu_memory_utilization": 0.9})}), {}) == rules  # decides no answer
    (pv_dir / "ckpt" / "manifest.json").write_text('{"tensors": {"x": "0"}}')
    assert snapshot.rules_sha256(pv, {}) != rules
