import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from bittensor.sp_core import Keypair
from safetensors.torch import save_file

from connito.prompt_validator import fetch, run, snapshot
from connito.prompt_validator.config import PromptValidatorCfg
from connito.prompt_validator.stage_io import read_rows, stage_path, write_rows
from connito.shared.checkpoint_helper import TensorSpec
from connito.shared.checkpoints import ChainCheckpoint
from connito.shared.helper import get_model_hash
from connito.shared.hf_distribute import HFFileMissingError, HFRepoUnavailableError
from connito.shared.schema import sign_message

CYCLE_ID = 1258
GROUP = 9
# A residual spec in miniature: one expert and its router on each of two layers.
SPEC = {
    **{
        f"model.layers.{layer}.mlp.residual_experts.0.{proj}.weight": TensorSpec(shape, torch.bfloat16)
        for layer in (1, 2)
        for proj, shape in (("gate_proj", (4, 8)), ("up_proj", (4, 8)), ("down_proj", (8, 4)))
    },
    **{f"model.layers.{layer}.mlp.residual_router.weight": TensorSpec((1, 8), torch.float32) for layer in (1, 2)},
}
FILENAME = "model_expgroup_9.safetensors"
TASK = "exp_pmc_residual"
SHIPPED_TASK = "exp_shipped_default"  # the YAML's task, which a new process starts on
TASK_GROUPS = {TASK: GROUP, SHIPPED_TASK: 4}
# The snapshot's base rows; fetch rebuilds the same ones from its config unless a test changes them.
BASE = {"base_model": "Qwen/Qwen3.6-35B-A3B", "base_model_revision": "0" * 40, "chat_template_sha256": "1" * 64}
KEYS = {"alice": Keypair.create_from_uri("//Alice"), "bob": Keypair.create_from_uri("//Bob")}


def _shard(seed=0):
    torch.manual_seed(seed)
    return {key: torch.randn(spec.shape).to(spec.dtype) for key, spec in sorted(SPEC.items())}


def _model(uid, shard, *, signer="alice", hotkey=None, source=None):
    """A snapshot row for `shard`, its hash signed by `signer` the way a miner commits it."""
    model_hash = get_model_hash(shard, hex=True)
    return {
        "uid": uid,
        "hotkey": KEYS[hotkey or signer].ss58_address,
        "hf_repo_id": f"miner/repo-{source or uid}",
        "hf_revision": f"rev-{source or uid}",
        "model_hash": model_hash,
        "signed_model_hash": sign_message(KEYS[signer], model_hash),
        "expert_group": GROUP,
        "global_ver": 100,
        "incentive": 0.5,  # the snapshot keeps it on each row; fetch must not trip on it
    }


def _load_task(cfg, name):
    cfg.task.expert_group_name = name
    cfg.task.exp = SimpleNamespace(group_id=TASK_GROUPS[name])


@pytest.fixture
def config(tmp_path, monkeypatch):
    """On the snapshot's task, whose folder is on disk; `switches` records each task switch."""
    def switch_active_task(name):
        cfg.switches.append(name)
        if not (cfg.task.base_path / name).is_dir():
            raise FileNotFoundError(cfg.task.base_path / name / "config.yaml")
        _load_task(cfg, name)

    def residual_shapes(cfg):
        # The real spec is built from the family's config.json and the loaded task's K.
        cfg.spec_tasks.append(cfg.task.expert_group_name)
        return SPEC

    monkeypatch.setattr(fetch, "residual_shapes", residual_shapes)
    # The base rows rebuilt from this config: the real ones download the family's chat template.
    monkeypatch.setattr(snapshot, "base_rows", lambda cfg: dict(BASE))
    cfg = SimpleNamespace(
        prompt_validator=PromptValidatorCfg(runs_dir=tmp_path / "runs", prompts_path=tmp_path / "prompts.jsonl",
                                            write_root=tmp_path),
        hf=SimpleNamespace(token_env_var="HF_TOKEN"),
        evaluation=SimpleNamespace(per_miner_download_timeout_sec=180),
        task=SimpleNamespace(base_path=tmp_path / "expert_groups"),
        switches=[],
        spec_tasks=[],
        switch_active_task=switch_active_task,
    )
    _load_task(cfg, TASK)
    (cfg.task.base_path / TASK).mkdir(parents=True)
    return cfg


@pytest.fixture
def hf(monkeypatch):
    """A fake Hugging Face: `served[(repo, revision)]` is a state dict or bytes to write, or an exception to raise.

    When `allowed_dirs` is set, every other source folder must be gone by the time the next download starts.
    """
    state = SimpleNamespace(served={}, calls=[], allowed_dirs=None)

    def fake_download(*, repo_id, revision, filenames, dest_dir, token_env_var, timeout_sec):
        state.calls.append((repo_id, revision))
        assert filenames == [FILENAME] and timeout_sec == 180.0
        # pytest.fail, not assert: fetch records any Exception from a download as download_error.
        if dest_dir.exists():
            pytest.fail(f"{dest_dir} still holds an earlier download")
        if state.allowed_dirs is not None and {p.name for p in dest_dir.parent.iterdir()} - state.allowed_dirs:
            pytest.fail(f"failed sources still on disk at the next download: {sorted(dest_dir.parent.iterdir())}")
        served = state.served[(repo_id, revision)]
        if isinstance(served, Exception):
            raise served
        dest_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(served, bytes):
            (dest_dir / filenames[0]).write_bytes(served)
        else:
            save_file(served, str(dest_dir / filenames[0]))
        return dest_dir

    monkeypatch.setattr(fetch, "download_checkpoint_from_hf_subprocess", fake_download)
    return state


def _snapshot(config, models):
    out_dir = stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "snapshot").parent
    write_rows(out_dir / "models.jsonl", sorted(models, key=lambda m: m["uid"]))
    rules = snapshot.rules_sha256(config.prompt_validator, BASE)
    write_rows(
        out_dir / "snapshot.jsonl",
        [{"scoring_cycle_id": CYCLE_ID, "active_task": TASK, "expert_group": GROUP, "rules_sha256": rules}],
    )


def _serve(hf, model, served):
    hf.served[(model["hf_repo_id"], model["hf_revision"])] = served


def _fetch_dir(config):
    return stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "fetch").parent


def _source_dir(config, source):
    """Where fetch puts `miner/repo-<source>` at `rev-<source>`."""
    name = hashlib.sha256(f"miner/repo-{source}@rev-{source}".encode()).hexdigest()[:16]
    return _fetch_dir(config) / "models" / name


def test_passing_models_are_kept_and_each_failure_is_recorded(config, hf):
    good, other = _shard(0), _shard(1)
    nan = _shard(2)
    nan[sorted(nan)[0]][0, 0] = float("nan")
    partial = dict(list(_shard(3).items())[1:])
    models = [
        _model(1, good),
        _model(2, good, signer="bob", source=1),  # a second hotkey committing the same repo and revision
        _model(3, good, signer="bob", hotkey="alice", source=1),  # shares the passing download, signed by another key
        _model(4, good),  # serves other weights than it committed
        _model(5, nan),
        _model(6, partial),
        _model(7, good),
        _model(8, good),
        _model(9, good),
        _model(10, good),
    ]
    for model, served in zip(models[:6], [good, good, good, other, nan, partial], strict=True):
        _serve(hf, model, served)
    _serve(hf, models[6], TimeoutError("slow"))
    _serve(hf, models[7], HFRepoUnavailableError("gone"))
    _serve(hf, models[8], HFFileMissingError("no file"))
    _serve(hf, models[9], RuntimeError("connection reset"))
    _snapshot(config, models)
    orphan = _fetch_dir(config) / "models" / "0123456789abcdef"
    orphan.mkdir(parents=True)
    (orphan / FILENAME).write_bytes(b"left by an older snapshot")

    hf.allowed_dirs = {_source_dir(config, 1).name, orphan.name}

    rows = fetch.run_fetch(config, CYCLE_ID)

    assert [(r["uid"], r["reason"]) for r in rows] == [
        (1, None), (2, None), (3, "signature"), (4, "hash"), (5, "bad_residual_shard"),
        (6, "bad_residual_shard"), (7, "timeout"), (8, "repo_unavailable"), (9, "file_missing"),
        (10, "download_error"),
    ]
    assert all(list(r) == ["uid", "hotkey", "path", "reason"] for r in rows)
    assert rows[0]["path"] == rows[1]["path"] and Path(rows[0]["path"]).name == FILENAME
    assert Path(rows[0]["path"]).is_file()
    assert all(r["path"] is None for r in rows[2:])
    assert len(hf.calls) == 8  # uids 1, 2 and 3 share one download
    # Only the passing shard stays on disk: failures and the orphan are deleted, the shared file is kept.
    assert [p.parent for p in (_fetch_dir(config) / "models").glob(f"*/{FILENAME}")] == [Path(rows[0]["path"]).parent]


def test_a_kept_model_that_fails_is_recorded_and_no_model_below_the_cut_takes_its_place(config, hf):
    shards = {name: _shard(seed) for seed, name in enumerate("abc")}
    commits = [
        {**_model(1, shards["a"]), "incentive": 0.5},
        {**_model(4, shards["a"], signer="bob", source=1), "incentive": 0.0},  # a copy of uid 1's commit
        {**_model(2, shards["b"]), "incentive": 0.3},
        {**_model(3, shards["c"]), "incentive": 0.1},  # third by incentive: below the cut of 2
    ]
    _serve(hf, commits[0], shards["a"])
    _serve(hf, commits[2], HFRepoUnavailableError("gone"))
    _serve(hf, commits[3], shards["c"])
    config.prompt_validator.max_models = 2
    kept = snapshot.select_models([ChainCheckpoint(**c) for c in commits], config.prompt_validator.max_models)
    _snapshot(config, [{field: getattr(c, field) for field in (*snapshot.MODEL_FIELDS, "incentive")} for c in kept])

    rows = fetch.run_fetch(config, CYCLE_ID)

    assert [(r["uid"], r["reason"]) for r in rows] == [(1, None), (2, "repo_unavailable"), (4, None)]
    assert hf.calls == [("miner/repo-1", "rev-1"), ("miner/repo-2", "rev-2")]


def test_an_empty_snapshot_fetches_nothing(config, hf):
    _snapshot(config, [])

    assert fetch.run_fetch(config, CYCLE_ID) == []
    assert hf.calls == []


def test_a_config_with_other_rules_stops_before_any_download(config, hf):
    _snapshot(config, [_model(1, _shard(0))])
    config.prompt_validator.max_new_tokens = 2048

    with pytest.raises(RuntimeError, match="is not the snapshot's"):
        fetch.run_fetch(config, CYCLE_ID)
    assert hf.calls == []


def test_rerun_reuses_checked_models_and_retries_the_rest(config, hf, monkeypatch):
    good = _shard(0)
    models = [_model(1, good), _model(2, _shard(1)), _model(3, good), _model(4, good), _model(5, good)]
    _serve(hf, models[0], good)
    _serve(hf, models[1], good)  # serves a different shard than committed: a hash failure, final
    _serve(hf, models[2], TimeoutError("slow"))
    _serve(hf, models[3], HFRepoUnavailableError("gone"))
    _serve(hf, models[4], good)
    _snapshot(config, models)
    real_validate = ChainCheckpoint.validate

    def validate(self, expected):
        if self.uid == 5:
            raise OSError("disk read failed")  # the check itself raises
        return real_validate(self, expected)

    monkeypatch.setattr(ChainCheckpoint, "validate", validate)
    run.run_stages(config, CYCLE_ID, "fetch")
    marker = stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "fetch")
    assert [r["reason"] for r in read_rows(marker)] == [None, "hash", "timeout", "repo_unavailable", "unknown"]

    hf.calls.clear()
    _serve(hf, models[2], good)  # the network is back
    monkeypatch.setattr(ChainCheckpoint, "validate", real_validate)
    run.run_stages(config, CYCLE_ID, "fetch")

    assert hf.calls == [("miner/repo-3", "rev-3"), ("miner/repo-4", "rev-4"), ("miner/repo-5", "rev-5")]
    rows = read_rows(marker)
    assert [r["reason"] for r in rows] == [None, "hash", None, "repo_unavailable", None]
    assert Path(rows[0]["path"]).is_file()  # the reused passing row's shard is kept


def test_a_passing_model_whose_file_is_gone_is_fetched_again(config, hf):
    good = _shard(0)
    models = [_model(1, good), _model(2, good)]
    for model in models:
        _serve(hf, model, good)
    _snapshot(config, models)
    [first, _] = fetch.run_fetch(config, CYCLE_ID)
    shutil.rmtree(Path(first["path"]).parent)

    hf.calls.clear()
    rows = fetch.run_fetch(config, CYCLE_ID)

    assert hf.calls == [("miner/repo-1", "rev-1")]
    assert [r["reason"] for r in rows] == [None, None]
    assert Path(rows[0]["path"]).is_file()


def test_a_new_commit_is_fetched_again(config, hf):
    model = _model(1, _shard(0))
    _serve(hf, model, _shard(0))
    _snapshot(config, [model])
    fetch.run_fetch(config, CYCLE_ID)

    recommit = _model(1, _shard(1), source="1-new")
    _serve(hf, recommit, _shard(1))
    _snapshot(config, [recommit])
    [row] = fetch.run_fetch(config, CYCLE_ID)

    assert hf.calls == [("miner/repo-1", "rev-1"), ("miner/repo-1-new", "rev-1-new")]
    assert row["reason"] is None


def test_low_disk_stops_the_stage_and_keeps_finished_models(config, hf, monkeypatch):
    good = _shard(0)
    models = [_model(1, good), _model(2, good)]
    for model in models:
        _serve(hf, model, good)
    _snapshot(config, models)
    real_disk_usage = fetch.shutil.disk_usage
    free = iter([100e9, 100e9, 1e9])  # the check before any download, then one before each
    monkeypatch.setattr(
        fetch.shutil, "disk_usage", lambda p: real_disk_usage(p)._replace(free=next(free, 100e9))
    )

    with pytest.raises(RuntimeError, match="min_free_disk_gb=8.0"):
        run.run_stages(config, CYCLE_ID, "fetch")
    assert not stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "fetch").exists()
    assert [r["uid"] for r in read_rows(_fetch_dir(config) / "progress.jsonl")] == [1]

    hf.calls.clear()
    run.run_stages(config, CYCLE_ID, "fetch")
    assert hf.calls == [("miner/repo-2", "rev-2")]


def test_fetch_switches_to_the_snapshot_task(config, hf):
    _load_task(config, SHIPPED_TASK)
    model = _model(1, _shard(0))
    _serve(hf, model, _shard(0))
    _snapshot(config, [model])

    [row] = fetch.run_fetch(config, CYCLE_ID)

    assert config.switches == [TASK]
    assert config.spec_tasks == [TASK]  # the spec is built after the switch
    assert row["reason"] is None


def test_a_snapshot_group_other_than_the_loaded_tasks_stops(config, hf, monkeypatch):
    _load_task(config, SHIPPED_TASK)
    monkeypatch.setitem(TASK_GROUPS, TASK, 3)  # the snapshot's task moved to another group since
    _snapshot(config, [_model(1, _shard(0))])

    with pytest.raises(RuntimeError, match=f"snapshot group {GROUP} is not the loaded task's group 3"):
        fetch.run_fetch(config, CYCLE_ID)
    assert config.switches == [TASK]
    assert hf.calls == []


def test_a_snapshot_task_missing_from_disk_stops(config, hf):
    _load_task(config, SHIPPED_TASK)
    (config.task.base_path / TASK).rmdir()
    _snapshot(config, [_model(1, _shard(0))])

    with pytest.raises(RuntimeError, match=f"snapshot task {TASK} is not materialised"):
        fetch.run_fetch(config, CYCLE_ID)
    assert hf.calls == []


def test_a_stale_partial_download_is_cleared_first(config, hf):
    model = _model(1, _shard(0))
    _serve(hf, model, _shard(0))
    _snapshot(config, [model])
    stale = _source_dir(config, 1) / ".cache" / "huggingface" / "download" / f"{FILENAME}.incomplete"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"left by a killed download")

    [row] = fetch.run_fetch(config, CYCLE_ID)

    assert hf.calls == [("miner/repo-1", "rev-1")]
    assert row["reason"] is None and not stale.exists()


def test_a_new_signed_hash_is_checked_again(config, hf):
    good = _shard(0)
    model = _model(1, good)
    _serve(hf, model, good)
    _snapshot(config, [model])
    [first] = fetch.run_fetch(config, CYCLE_ID)

    # Same uid, hotkey, repo, revision and model hash; only the signature is new, and signed by the wrong key.
    _snapshot(config, [{**model, "signed_model_hash": sign_message(KEYS["bob"], model["model_hash"])}])
    [row] = fetch.run_fetch(config, CYCLE_ID)

    assert first["reason"] is None and row["reason"] == "signature"
    assert hf.calls == [("miner/repo-1", "rev-1")]  # checked again on the file already on disk


def test_a_failed_check_with_no_failing_flag_is_unknown(config, hf, monkeypatch):
    def validate(self, expected):
        self.signature_verified = self.shard_verified = self.hash_verified = True
        return False  # validated() can still refuse, e.g. an expired checkpoint

    monkeypatch.setattr(ChainCheckpoint, "validate", validate)
    model = _model(1, _shard(0))
    _serve(hf, model, _shard(0))
    _snapshot(config, [model])

    assert [r["reason"] for r in fetch.run_fetch(config, CYCLE_ID)] == ["unknown"]


@pytest.fixture
def free_gb(monkeypatch):
    """Sets the free space fetch sees, in GB; returns the paths it measured."""
    measured, real = [], fetch.shutil.disk_usage

    def set_free(gb):
        def disk_usage(path):
            measured.append(Path(path))
            return real(path)._replace(free=gb * 1e9)

        monkeypatch.setattr(fetch.shutil, "disk_usage", disk_usage)
        return measured

    return set_free


def _three_sources(config, hf):
    """Four models on three sources: uids 1 and 2 share one."""
    good = _shard(0)
    models = [_model(1, good), _model(2, good, source=1), _model(3, good), _model(4, good)]
    for model in models:
        _serve(hf, model, good)
    return models


# 3 downloads × expected_shard_gb 0.3 + min_free_disk_gb 8.0
@pytest.mark.parametrize(("free", "fetched"), [(8.9, True), (8.89, False)])
def test_fetch_starts_only_with_room_for_every_download_and_the_margin(config, hf, free_gb, free, fetched):
    _snapshot(config, _three_sources(config, hf))
    measured = free_gb(free)
    if fetched:
        assert [r["reason"] for r in fetch.run_fetch(config, CYCLE_ID)] == [None] * 4
        assert len(hf.calls) == 3
    else:
        with pytest.raises(RuntimeError) as e:
            fetch.run_fetch(config, CYCLE_ID)
        assert str(e.value) == (
            f"fetch needs 8.90 GB free under {_fetch_dir(config) / 'models'} and has 8.89 GB: 3 downloads × "
            "expected_shard_gb=0.3 + min_free_disk_gb=8.0; free space, then rerun"
        )
        assert hf.calls == [] and not (_fetch_dir(config) / "progress.jsonl").exists()
    assert measured[0] == _fetch_dir(config) / "models"  # the filesystem the shards go to


def test_reused_and_already_downloaded_sources_are_not_counted(config, hf, free_gb):
    models = [*_three_sources(config, hf), _model(5, _shard(0)), _model(6, _shard(0))]
    _serve(hf, models[4], _shard(0))
    _serve(hf, models[5], _shard(1))  # other weights than it committed: a final "hash", its file deleted
    _snapshot(config, [*models[:3], models[5]])
    fetch.run_fetch(config, CYCLE_ID)  # sources 1, 3 and 6: reused below
    stale = _source_dir(config, 4)
    stale.mkdir(parents=True)
    save_file(_shard(0), str(stale / FILENAME))  # on disk with no progress row: checked, not downloaded
    _snapshot(config, models)
    hf.calls.clear()
    free_gb(8.3)  # one download: source 5
    reasons = [None] * 5 + ["hash"]
    assert [r["reason"] for r in fetch.run_fetch(config, CYCLE_ID)] == reasons
    assert hf.calls == [("miner/repo-5", "rev-5")]

    hf.calls.clear()
    free_gb(0)  # nothing to download: no check
    assert [r["reason"] for r in fetch.run_fetch(config, CYCLE_ID)] == reasons
    assert hf.calls == []


def test_expected_shard_gb_and_the_margin_are_settings(config, hf, free_gb):
    _snapshot(config, _three_sources(config, hf))
    config.prompt_validator.expected_shard_gb, config.prompt_validator.min_free_disk_gb = 1.0, 2.0
    free_gb(4.99)
    with pytest.raises(RuntimeError, match="fetch needs 5.00 GB free"):
        fetch.run_fetch(config, CYCLE_ID)
    free_gb(5.0)
    assert len(fetch.run_fetch(config, CYCLE_ID)) == 4


def test_fetch_never_writes_through_a_symlink_out_of_its_cycle(config, hf, tmp_path):
    good = _shard(0)
    _serve(hf, _model(1, good), good)
    _snapshot(config, [_model(1, good)])
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    _fetch_dir(config).mkdir(parents=True)
    (_fetch_dir(config) / "models").symlink_to(outside)
    with pytest.raises(ValueError, match="a symlink leads elsewhere"):
        fetch.run_fetch(config, CYCLE_ID)
    assert hf.calls == [] and list(outside.iterdir()) == []


def test_a_source_counts_when_any_of_its_models_needs_the_download(config, hf, free_gb):
    good = _shard(0)
    failed = _model(1, good, signer="bob", hotkey="alice")  # a final "signature": reused, its file deleted
    other = _model(2, good, source=1)  # the same repo and revision, not fetched yet
    _serve(hf, failed, good)
    _snapshot(config, [failed])
    fetch.run_fetch(config, CYCLE_ID)
    _snapshot(config, [failed, other])
    hf.calls.clear()
    free_gb(8.29)  # one download needs 8.3
    with pytest.raises(RuntimeError, match="1 downloads"):
        fetch.run_fetch(config, CYCLE_ID)
    assert hf.calls == []
