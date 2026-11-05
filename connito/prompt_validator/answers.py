"""Stage 4: each fetched miner model answers every snapshot prompt, greedily, with the base model's chat template.

The model is built the way the training validator builds it for evaluation: the family's base at its pinned
revision with the validator's fp8 experts and the task's residual experts (get_model_from_checkpoint), and each
miner's residual shard loaded over it in place. A shard is checked against the residual spec again before it is
loaded, since loading in place keeps any tensor a shard lacks from the miner before it.

Prompts go through apply_chat_template and generate() with exactly answer_settings(), which the rules hash covers,
in a fixed order and fixed batches (or, with engine "vllm", through one vLLM engine, whose answers repeat within that
engine but not across engine starts). Miners that committed the same weights (model_hash) give the same answers: each
model answers once. Each model's answers are saved as it finishes, so a rerun goes on from the next model.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import signal
import subprocess
import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import torch
from transformers import GenerationConfig

from connito.prompt_validator.judge import selected_prompts
from connito.prompt_validator.snapshot import answer_settings, check_rules
from connito.prompt_validator.stage_io import read_rows, stage_path, write_rows
from connito.prompt_validator.status import set_status
from connito.shared.app_logging import structlog
from connito.shared.checkpoint_helper import verify_residual_shard
from connito.shared.model import get_model_from_checkpoint
from connito.shared.modeling.families import family
from connito.shared.modeling.mycelia import get_base_tokenizer
from connito.shared.modeling.residual import residual_shapes

logger = structlog.get_logger(__name__)

ROW_FIELDS = ("uid", "hotkey", "model_hash", "prompt_id", "text", "n_tokens", "finish_reason")


def eos_ids(generation_config, tokenizer) -> set[int]:
    eos = generation_config.eos_token_id
    ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    return ids | ({tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set())


def prompt_texts(tokenizer, prompts: list[dict], chat_template: dict) -> list[str]:
    """Each prompt through the chat template as text, which is then tokenized without special tokens."""
    return [
        tokenizer.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False,
                                      add_generation_prompt=True, **chat_template)
        for p in prompts
    ]


class Cut(NamedTuple):
    tokens: list[int]
    finish_reason: str


def cut_at_eos(tokens: list[int], eos: set[int]) -> Cut:
    """An answer's tokens up to its first end-of-sequence token and "stop", or all of them and "length"."""
    end = next((i for i, t in enumerate(tokens) if t in eos), None)
    return Cut(tokens, "length") if end is None else Cut(tokens[:end], "stop")


# Never passed to the worker, which runs third-party code; connito_vllm's worker_env drops the same.
_SECRET_ENV = ("JEV_DIRECT_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY")
_POLL_S = 2.0


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # the worker and whatever it left, vLLM's engine core included
    except ProcessLookupError:
        pass
    proc.wait()


def run_worker(python: Path | str, job: dict, work: Path, on_result: Callable[[str, dict], None], *,
               on_progress: Callable[[str, int], None] | None = None, start_timeout_s: float = 1200.0,
               stall_timeout_s: float = 600.0, secrets: tuple[str, ...] = ()) -> None:
    """Run connito_vllm.worker on job in its own process group, handing on_result each model's result as it lands."""
    # Not require_gpu_env: check_vllm may run on a GPU the stage refuses. Which GPU is the caller's choice.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or not visible or "," in visible:
        raise RuntimeError("the vLLM worker needs CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES naming one GPU")
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    work.chmod(0o700)  # the job and results hold the private prompts' token ids
    # A result counts only if it names this run: one left in work by an earlier run is never taken for this one's.
    job = {**job, "run_id": uuid.uuid4().hex}
    job_path, pending = work / "job.json", [m["model_hash"] for m in job["models"]]
    for name in [f"{h}.json" for h in pending] + ["progress.json"]:
        (work / name).unlink(missing_ok=True)
    write_rows(job_path, [job])
    # Offline and without the secrets; CUDA_VISIBLE_DEVICES passes through unchanged.
    drop = set(_SECRET_ENV) | set(secrets)
    env = {k: v for k, v in os.environ.items() if k not in drop and not k.startswith("BT_PW_")}
    env |= {"HF_HUB_OFFLINE": "1", "CONNITO_VLLM_PARENT": str(os.getpid())}
    seen: dict = {}

    def collect() -> bool:
        """Hand over the results that landed; True if anything moved: a result, or a heartbeat (a request done)."""
        moved = False
        for model_hash in [h for h in pending if (work / f"{h}.json").exists()]:
            result = json.loads((work / f"{model_hash}.json").read_text())
            if result.get("run_id") == job["run_id"]:
                on_result(model_hash, result)
                pending.remove(model_hash)
                moved = True
        # After the results: a heartbeat already on the next model must not count the one just finished as undone.
        heartbeat = work / "progress.json"
        if heartbeat.exists():
            beat = json.loads(heartbeat.read_text())
            if beat.get("run_id") == job["run_id"] and beat != seen:
                seen.clear()
                seen.update(beat)
                moved = True
                if on_progress is not None and beat["step"] == "generate":
                    on_progress(beat["model_hash"], beat["done"])
        return moved

    proc = subprocess.Popen([str(python), "-m", "connito_vllm.worker", str(job_path), str(work)], env=env,
                            start_new_session=True)
    # A hung engine must not hang the stage (--loop retries only what raises). The start window covers loading and
    # compiling until the first finished request; after that, every finished request resets the stall window.
    deadline = time.monotonic() + start_timeout_s
    try:
        while proc.poll() is None:
            if collect():
                deadline = time.monotonic() + stall_timeout_s
            elif time.monotonic() > deadline:
                left = [h[:12] for h in pending]
                raise TimeoutError(f"the vLLM worker finished no request in time; models left {left}")
            time.sleep(_POLL_S)
        collect()  # what the worker wrote before it exited, whatever the exit
    finally:
        _kill_group(proc)  # on every path: a worker that died (OOM, a signal) may leave its engine core holding the GPU
    if proc.returncode != 0:
        raise RuntimeError(f"the vLLM worker failed with exit status {proc.returncode}")
    if pending:
        raise RuntimeError(f"the vLLM worker exited without answering models {[h[:12] for h in pending]}")


def generate_answers(model, tokenizer, prompts: list[dict], settings: dict, device: torch.device,
                     on_batch: Callable[[int], None] | None = None) -> list[dict]:
    """{prompt_id, text, n_tokens, finish_reason} for each prompt, in batches of settings["batch_size"].
    on_batch, if given, is called after each batch with the number of prompts answered so far."""
    eos = eos_ids(model.generation_config, tokenizer)
    out = []
    size = settings["batch_size"]
    for start in range(0, len(prompts), size):
        batch = prompts[start: start + size]
        texts = prompt_texts(tokenizer, batch, settings["chat_template"])
        inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        # Under bf16 autocast, as the training validator evaluates, so the model's numerics are the same.
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            generated = model.generate(**inputs, **settings["generate"], pad_token_id=tokenizer.pad_token_id)
        new = generated[:, inputs["input_ids"].shape[1]:]
        for p, tokens in zip(batch, new.tolist(), strict=True):
            # The answer ends at its first end-of-sequence token; what follows is padding.
            kept, finish = cut_at_eos(tokens, eos)
            out.append({
                "prompt_id": p["prompt_id"], "text": tokenizer.decode(kept, skip_special_tokens=True),
                "n_tokens": len(kept), "finish_reason": finish,
            })
        if on_batch is not None:
            on_batch(len(out))
    return out


def _answer_hf(config, prompts: list[dict], by_hash: dict[str, dict], todo: list[str], progress_dir: Path) -> None:
    """Each todo model's answers through HF generate(), on the validator's model with each shard loaded in place."""
    pv = config.prompt_validator
    total = len(todo) * len(prompts)
    settings = answer_settings(pv)
    device = torch.device(config.model.device if torch.cuda.is_available() else "cpu")
    tokenizer = get_base_tokenizer(config)
    tokenizer.padding_side = "left"  # generation continues each prompt from its last token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, _ = get_model_from_checkpoint(config, checkpoint_device=device, load_global_checkpoint=False)
    model.eval()
    expected = residual_shapes(config)
    started, answered = time.monotonic(), 0  # timed from here: the base model's load is not answering
    for i, model_hash in enumerate(todo, 1):
        miner = by_hash[model_hash]
        shard = verify_residual_shard(miner["path"], expected)
        if shard is None:
            raise RuntimeError(f"uid {miner['uid']}'s shard at {miner['path']} no longer matches the residual spec")
        # In place, as the training validator loads each miner. Every key must land: a name that matched nothing
        # would leave this miner answering with the previous one's residual (or the base's).
        unexpected = model.load_state_dict(shard, strict=False).unexpected_keys
        del shard  # the tensors are in the model now: don't hold two shards while reading the next
        if unexpected:
            raise RuntimeError(f"uid {miner['uid']}'s shard has {len(unexpected)} keys the model does not have, "
                               f"e.g. {unexpected[0]}")

        def progress(n: int, i: int = i, uid: int = miner["uid"], before: int = answered) -> None:
            count = before + n
            elapsed = time.monotonic() - started
            eta_min = round(elapsed / count * (total - count) / 60, 1)
            logger.info("answering", model=f"{i}/{len(todo)}", uid=uid, prompts=f"{n}/{len(prompts)}",
                        answers=f"{count}/{total}", percent=round(100 * count / total, 1),
                        elapsed_min=round(elapsed / 60, 1), eta_min=eta_min)
            set_status(progress=f"{count}/{total}", eta_min=eta_min)

        answers = generate_answers(model, tokenizer, prompts, settings, device, on_batch=progress)
        answered += len(answers)
        write_rows(progress_dir / f"{model_hash}.jsonl", answers)
        logger.info("model answered", uid=miner["uid"], model_hash=model_hash[:12], prompts=len(answers),
                    cut_at_max=sum(a["finish_reason"] == "length" for a in answers))


def _answer_vllm(config, prompts: list[dict], by_hash: dict[str, dict], todo: list[str], progress_dir: Path) -> None:
    """Each todo model's answers from one vLLM engine (connito_vllm.worker in the vLLM venv), cut and decoded here."""
    pv, vcfg = config.prompt_validator, config.prompt_validator.vllm
    if not prompts:
        raise RuntimeError("the snapshot has no prompt to answer")
    settings = answer_settings(pv)
    fam, tokenizer = family(config), get_base_tokenizer(config)
    eos = eos_ids(GenerationConfig.from_pretrained(fam.model_path, revision=fam.revision), tokenizer)
    ids = [tokenizer(t, add_special_tokens=False)["input_ids"]
           for t in prompt_texts(tokenizer, prompts, settings["chat_template"])]
    expected = residual_shapes(config)
    for model_hash in todo:  # refused here, as HF refuses them, before the engine starts
        miner = by_hash[model_hash]
        if verify_residual_shard(miner["path"], expected) is None:
            raise RuntimeError(f"uid {miner['uid']}'s shard at {miner['path']} no longer matches the residual spec")
    max_new_tokens = settings["generate"]["max_new_tokens"]
    job = {
        "checkpoint": str(vcfg.checkpoint), "residual": config.task.exp.residual.model_dump(), "prompts": ids,
        "eos": sorted(eos), "max_new_tokens": max_new_tokens, "vllm_version": vcfg.version,
        "engine": {"max_model_len": max(map(len, ids)) + max_new_tokens + 16, "max_num_seqs": vcfg.max_num_seqs,
                   "gpu_memory_utilization": vcfg.gpu_memory_utilization, "enforce_eager": vcfg.enforce_eager},
        "models": [{"model_hash": h, "shard": str(by_hash[h]["path"]), "score": None} for h in todo],
    }
    answered, started = [], time.monotonic()

    def save(model_hash: str, result: dict) -> None:
        # Written as it lands, in the HF path's format: a rerun goes on from the next model.
        rows = []
        for p, a in zip(prompts, result["answers"], strict=True):
            kept, finish = cut_at_eos(a["ids"], eos)
            rows.append({"prompt_id": p["prompt_id"], "text": tokenizer.decode(kept, skip_special_tokens=True),
                         "n_tokens": len(kept), "finish_reason": finish})
        write_rows(progress_dir / f"{model_hash}.jsonl", rows)
        answered.append(model_hash)
        logger.info("model answered", uid=by_hash[model_hash]["uid"], model_hash=model_hash[:12], prompts=len(rows),
                    cut_at_max=sum(r["finish_reason"] == "length" for r in rows), engine="vllm",
                    **{k: result[k] for k in ("start_s", "swap_s", "generate_s", "new_tokens", "tokens_per_s",
                                                     "peak_gib")})
        progress(model_hash, len(prompts))

    def progress(model_hash: str, done: int) -> None:
        # Per finished request, from the worker's heartbeat; the eta counts the engine start in the elapsed time.
        count = (len(answered) - (model_hash in answered)) * len(prompts) + done
        total = len(todo) * len(prompts)
        if count:
            eta_min = round((time.monotonic() - started) / count * (total - count) / 60, 1)
            set_status(progress=f"{count}/{total}", eta_min=eta_min)

    # Per snapshot, like the progress files; run_worker takes only this run's results, so a resume goes through the
    # progress files alone and nothing a dead run left here is read back.
    work = progress_dir / "vllm"
    run_worker(vcfg.python, job, work, save, on_progress=progress, start_timeout_s=vcfg.start_timeout_min * 60,
               stall_timeout_s=vcfg.stall_timeout_min * 60, secrets=(config.hf.token_env_var,))
    shutil.rmtree(work)


def run_answers(config, scoring_cycle_id: int) -> list[dict]:
    pv = config.prompt_validator
    snapshot_dir = stage_path(pv.runs_dir, scoring_cycle_id, "snapshot").parent
    meta = read_rows(snapshot_dir / "snapshot.jsonl")[0]
    # Each process starts on the YAML's task; the residual spec (its K) must be the one the snapshot was taken on.
    if config.task.expert_group_name != meta["active_task"]:
        try:
            config.switch_active_task(meta["active_task"])
        except FileNotFoundError as e:
            raise RuntimeError(
                f"snapshot task {meta['active_task']} is not materialised under {config.task.base_path}: "
                "the snapshot stage writes it"
            ) from e
    if config.task.exp.group_id != meta["expert_group"]:
        raise RuntimeError(
            f"snapshot group {meta['expert_group']} is not the loaded task's group {config.task.exp.group_id}: "
            f"task {meta['active_task']} changed group since the snapshot"
        )
    check_rules(config, meta)  # this config must answer as the snapshot recorded
    prompts = sorted([*read_rows(snapshot_dir / "prompts.jsonl"), *selected_prompts(pv, scoring_cycle_id)],
                     key=lambda p: p["prompt_id"])
    models = {(m["uid"], m["hotkey"]): m for m in read_rows(snapshot_dir / "models.jsonl")}
    fetched = [r for r in read_rows(stage_path(pv.runs_dir, scoring_cycle_id, "fetch")) if r["reason"] is None]
    miners = sorted(({**models[(r["uid"], r["hotkey"])], "path": r["path"]} for r in fetched),
                    key=lambda m: m["uid"])
    if not miners:
        raise RuntimeError(f"no model of scoring cycle {scoring_cycle_id} was fetched: nothing to answer")

    by_hash: dict[str, dict] = {}
    for m in miners:
        by_hash.setdefault(m["model_hash"], m)  # the lowest uid's file answers for every copy of one model
    # Per snapshot: a snapshot retaken with other prompts or answer settings has another snapshot_sha256, so the
    # answers saved for the old one are never read back for it. A judge rerun that selects other miner prompts
    # changes prompt_ids, so done() answers again.
    progress_dir = (stage_path(pv.runs_dir, scoring_cycle_id, "answers").parent / "progress"
                    / meta["snapshot_sha256"][:16])
    prompt_ids = [p["prompt_id"] for p in prompts]

    def done(model_hash: str) -> bool:
        path = progress_dir / f"{model_hash}.jsonl"
        return path.exists() and [a["prompt_id"] for a in read_rows(path)] == prompt_ids

    todo = [h for h in by_hash if not done(h)]
    total = len(todo) * len(prompts)
    logger.info("answers to generate", models=len(by_hash), copies=len(miners) - len(by_hash),
                done=len(by_hash) - len(todo), prompts=len(prompts), total=total)

    if todo:
        answer = _answer_hf if pv.engine == "hf" else _answer_vllm
        try:
            answer(config, prompts, by_hash, todo, progress_dir)
        except BaseException as e:
            # The traceback's frames hold the model until the exception is dropped: clear them, so it is freed now.
            chain, seen = [e], set()
            while chain:
                error = chain.pop()
                if error is not None and id(error) not in seen:
                    seen.add(id(error))
                    traceback.clear_frames(error.__traceback__)
                    chain += [error.__cause__, error.__context__]
            raise
        finally:
            # The loop waits about a day for the next cutoff: give the model's GPU memory back, not to torch's cache.
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as e:  # never in place of the stage's own error
                    logger.warning("CUDA cache not emptied", error=type(e).__name__)

    rows = []
    for m in miners:
        for a in read_rows(progress_dir / f"{m['model_hash']}.jsonl"):
            rows.append({"uid": m["uid"], "hotkey": m["hotkey"], "model_hash": m["model_hash"], **a})
    return [{field: row[field] for field in ROW_FIELDS} for row in rows]
