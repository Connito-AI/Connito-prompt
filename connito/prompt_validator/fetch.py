"""Stage 3: download each snapshot model's shard at its committed revision, and keep those that pass.

The checks are the training validator's, so a model scored here is one it would accept. `progress.jsonl`
is rewritten after every download: a crash or a `--stage fetch` rerun resumes there without downloading again.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from connito.prompt_validator.snapshot import MODEL_FIELDS, check_rules
from connito.prompt_validator.stage_io import read_rows, stage_path, under_root, write_rows
from connito.shared.app_logging import structlog
from connito.shared.checkpoint_helper import TensorSpec
from connito.shared.checkpoints import ChainCheckpoint
from connito.shared.helper import expert_group_shard_name
from connito.shared.hf_distribute import (
    HFFileMissingError,
    HFRepoUnavailableError,
    download_checkpoint_from_hf_subprocess,
)
from connito.shared.modeling.residual import residual_shapes

logger = structlog.get_logger(__name__)

# A rerun retries download failures and a check that raised; any other check result on a pinned revision is final.
RETRIED_REASONS = ("timeout", "repo_unavailable", "file_missing", "download_error", "unknown")


# A progress row is reused only for the same commit, every field the checks read: a retaken snapshot may hold a new one.
def _identity(row: dict) -> tuple:
    return tuple(row[field] for field in MODEL_FIELDS)


def _download(config, repo_id: str, revision: str, filename: str, dest_dir: Path) -> str | None:
    """Fetch one file into an emptied dest_dir: None on success, else the reason."""
    shutil.rmtree(dest_dir, ignore_errors=True)  # a killed download leaves its partial file behind
    try:
        download_checkpoint_from_hf_subprocess(
            repo_id=repo_id,
            revision=revision,
            filenames=[filename],
            dest_dir=dest_dir,
            token_env_var=config.hf.token_env_var,
            timeout_sec=float(config.evaluation.per_miner_download_timeout_sec),
        )
    except TimeoutError:
        return "timeout"
    except HFRepoUnavailableError:
        return "repo_unavailable"
    except HFFileMissingError:
        return "file_missing"
    except Exception as e:
        logger.warning("download failed", repo_id=repo_id, revision=revision, error=str(e)[:500])
        return "download_error"
    return None


def _check(model: dict, path: Path, expected: dict[str, TensorSpec]) -> str | None:
    """The training validator's checks on one shard: None if it passes, else the first one failed."""
    # Mirrors validator/evaluator.py:validate_miner_submission; layering keeps connito/validator out of here.
    checkpoint = ChainCheckpoint(**model, path=path, signature_required=True, hash_required=True)
    try:
        if checkpoint.validate(expected):
            return None
    except Exception as e:
        logger.warning("check raised", uid=model["uid"], error=str(e)[:500])
        return "unknown"
    if not checkpoint.signature_verified:
        return "signature"
    # An off-spec shard is never hashed, so it is named before the hash.
    if not checkpoint.shard_verified:
        return "bad_residual_shard"
    if not checkpoint.hash_verified:
        return "hash"
    return "unknown"


def shards_dir(pv, scoring_cycle_id: int) -> Path:
    """The cycle's `fetch/models`, refused unless it resolves to exactly that dir of the runs_dir, under write_root."""
    path = stage_path(pv.runs_dir, scoring_cycle_id, "fetch").parent / "models"
    expected = under_root(pv.runs_dir, pv.write_root) / str(scoring_cycle_id) / "fetch" / "models"
    if Path(os.path.realpath(path)) != expected:
        raise ValueError(f"{path} resolves to {os.path.realpath(path)}, not {expected}: a symlink leads elsewhere")
    return path


def prune_shards(pv, scoring_cycle_id: int) -> None:
    """Delete a published cycle's shards. Never raises: the publish it follows is final."""
    path = stage_path(pv.runs_dir, scoring_cycle_id, "fetch").parent / "models"
    try:
        path = shards_dir(pv, scoring_cycle_id)
        if not path.exists():
            return
        files = sizes = 0
        for top, _, names in os.walk(path):
            files += len(names)
            sizes += sum(os.lstat(os.path.join(top, name)).st_size for name in names)
        shutil.rmtree(path)
    except Exception as e:
        logger.warning("shards not pruned", scoring_cycle_id=scoring_cycle_id, path=str(path),
                       error=f"{type(e).__name__}: {e}"[:500])
        return
    logger.info("shards pruned", scoring_cycle_id=scoring_cycle_id, files=files, bytes=sizes)


def run_fetch(config, scoring_cycle_id: int) -> list[dict]:
    pv = config.prompt_validator
    snapshot_dir = stage_path(pv.runs_dir, scoring_cycle_id, "snapshot").parent
    models = read_rows(snapshot_dir / "models.jsonl")
    meta = read_rows(snapshot_dir / "snapshot.jsonl")[0]
    group = meta["expert_group"]
    # Each process starts on the YAML's task; the residual spec (its K) must be the one the snapshot was taken on.
    if config.task.expert_group_name != meta["active_task"]:
        try:
            config.switch_active_task(meta["active_task"])
        except FileNotFoundError as e:
            raise RuntimeError(
                f"snapshot task {meta['active_task']} is not materialised under {config.task.base_path}: "
                "the snapshot stage writes it"
            ) from e
    if config.task.exp.group_id != group:
        raise RuntimeError(
            f"snapshot group {group} is not the loaded task's group {config.task.exp.group_id}: "
            f"task {meta['active_task']} changed group since the snapshot"
        )
    check_rules(config, meta)  # before any download: this config must answer as the snapshot recorded
    expected = residual_shapes(config)
    filename = expert_group_shard_name(group)

    models_dir = shards_dir(pv, scoring_cycle_id)
    models_dir.mkdir(parents=True, exist_ok=True)
    progress_path = models_dir.parent / "progress.jsonl"
    current = {_identity(m) for m in models}
    results = {}
    if progress_path.exists():
        for row in read_rows(progress_path):
            final = row["reason"] not in RETRIED_REASONS and (row["path"] is None or Path(row["path"]).exists())
            if final and _identity(row) in current:
                results[_identity(row)] = row

    # Hotkeys sharing one repo and revision share one download; each is still checked on its own.
    by_source = defaultdict(list)
    for model in models:
        by_source[(model["hf_repo_id"], model["hf_revision"])].append(model)
    dest_dirs = {
        source: models_dir / hashlib.sha256(f"{source[0]}@{source[1]}".encode()).hexdigest()[:16]
        for source in by_source
    }
    downloads = sum(
        any(_identity(m) not in results for m in source_models) and not (dest_dirs[source] / filename).exists()
        for source, source_models in by_source.items()
    )
    logger.info(
        "fetching", models=len(models), sources=len(by_source),
        model_hashes=len({m["model_hash"] for m in models}), reused=len(results), downloads=downloads,
    )
    need_gb = downloads * pv.expected_shard_gb + pv.min_free_disk_gb
    free_gb = shutil.disk_usage(models_dir).free / 1e9
    if downloads and free_gb < need_gb:
        raise RuntimeError(
            f"fetch needs {need_gb:.2f} GB free under {models_dir} and has {free_gb:.2f} GB: {downloads} downloads × "
            f"expected_shard_gb={pv.expected_shard_gb} + min_free_disk_gb={pv.min_free_disk_gb}; free space, then rerun"
        )

    for (repo_id, revision), source_models in by_source.items():
        todo = [m for m in source_models if _identity(m) not in results]
        if not todo:
            continue
        dest_dir = dest_dirs[(repo_id, revision)]
        path = dest_dir / filename
        reason = None
        if not path.exists():
            free_gb = shutil.disk_usage(models_dir).free / 1e9
            if free_gb < pv.min_free_disk_gb:
                raise RuntimeError(
                    f"only {free_gb:.1f} GB free, below prompt_validator.min_free_disk_gb={pv.min_free_disk_gb}: "
                    f"free space under {models_dir}, then rerun to resume"
                )
            reason = _download(config, repo_id, revision, filename, dest_dir)
        for model in todo:
            check = reason or _check(model, path, expected)
            row = {field: model[field] for field in MODEL_FIELDS}
            results[_identity(model)] = {**row, "path": None if check else str(path), "reason": check}
            logger.info("model fetched", uid=model["uid"], hotkey=model["hotkey"][:6], reason=check)
        if all(results[_identity(m)]["reason"] for m in source_models):
            shutil.rmtree(dest_dir, ignore_errors=True)
        write_rows(progress_path, list(results.values()))

    rows = [results[_identity(m)] for m in models]
    # Shards left by an older snapshot or an interrupted run hold disk that no passing model uses.
    kept = {Path(row["path"]).parent for row in rows if row["path"]}
    for source_dir in models_dir.iterdir():
        if source_dir not in kept:
            shutil.rmtree(source_dir, ignore_errors=True)
    logger.info("fetch done", reasons=dict(Counter(row["reason"] for row in rows)))
    return [{field: row[field] for field in ("uid", "hotkey", "path", "reason")} for row in rows]
