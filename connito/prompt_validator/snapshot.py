"""Stage 2: freeze one scoring cycle's inputs, so later stages and reruns all score the same thing.

`prompts.jsonl` and `models.jsonl` (with judge_miner_prompts, also `miner_prompts.jsonl` and `request.json`) sit
beside `snapshot.jsonl`, whose one meta row (blocks, task, versions, hashes) is written last and so marks the stage
done. Each training cycle's commits read for originals are cached by block; past training cycles' blocks come from
the current phase config.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

import bittensor
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict, ValidationError

from connito.prompt_validator import dataless, loss_rank
from connito.prompt_validator.answer_format import FORMAT_RULES
from connito.prompt_validator.config import GREEDY, JUDGE_RULES_DATA
from connito.prompt_validator.cutoff import prompt_cycle_length, training_cycle_blocks
from connito.prompt_validator.jev import CRITERIA_BY_TASK, SCORE_QUESTIONS
from connito.prompt_validator.stage_io import read_rows, stage_path, write_rows
from connito.shared import bt_compat
from connito.shared.app_logging import structlog
from connito.shared.checkpoints import build_chain_checkpoints_at_blocks
from connito.shared.helper import get_nested_attr
from connito.shared.modeling.families import family
from connito.shared.schema import verify_message
from connito.shared.task_sync import ensure_active_task, get_active_task

logger = structlog.get_logger(__name__)

# What deciding a hash's original reads of each miner commit, kept or not, as each training cycle's cache holds it.
CHECKPOINT_FIELDS = ("uid", "hotkey", "model_hash", "signed_model_hash")
CHAIN_FIELDS = (*CHECKPOINT_FIELDS, "signed_commit_block")
# Training cycles before the cutoff's whose commits decide originals: fixed, so validators with any history agree.
LOOKBACK_TRAINING_CYCLES = 26
# Fetch rebuilds a ChainCheckpoint from these with signature_required and hash_required True, or validate() skips them.
MODEL_FIELDS = (
    "uid", "hotkey", "hf_repo_id", "hf_revision", "model_hash", "signed_model_hash", "expert_group", "global_ver",
)
MINER_PROMPT_FIELDS = ("hotkey", "customer_task_id", "text", "content_hash", "block")
# The dataless-api's 404 codes for a task with no customer request to give.
NO_REQUEST = ("no_request_for_task", "task_not_open")

_STRICT = ConfigDict(extra="forbid", strict=True)


class _Request(BaseModel):
    model_config = _STRICT
    goal: str
    examples: list[str]
    tone: str


class _PromptRow(BaseModel):
    model_config = _STRICT
    prompt_id: str
    request_id: str
    prompt: str
    request: _Request


def load_internal_prompts(path: Path) -> list[dict]:
    """The checked rows of the private prompt file; an error names the line and fields, never the text."""
    rows, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            # `from None`: pydantic's message and the chained traceback both quote the row's text.
            try:
                row = json.loads(line)
                _PromptRow.model_validate(row)
            except json.JSONDecodeError:
                raise ValueError(f"{path}:{lineno}: not JSON") from None
            except ValidationError as e:
                fields = sorted({".".join(map(str, err["loc"])) or "row" for err in e.errors()})
                raise ValueError(f"{path}:{lineno}: bad fields {fields}") from None
            if row["prompt_id"] in seen:
                raise ValueError(f"{path}:{lineno}: duplicate prompt_id {row['prompt_id']}")
            seen.add(row["prompt_id"])
            rows.append(row)
    return rows


def _sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def chat_template_sha256(fam) -> str:
    """The sha256 of the family's chat template at its pinned revision: the file apply_chat_template reads."""
    path = hf_hub_download(fam.model_path, "chat_template.jinja", revision=fam.revision)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_rows(config) -> dict:
    """The meta rows naming the model that answers: the family's base, its template, the fp8 scope, the residual gate.

    The snapshot records them and later stages rebuild them from their own config, so the two can be compared.
    """
    fam = family(config)  # the training validator's checkpoint, pinned with its revision
    return {
        "base_model": fam.model_path,
        "base_model_revision": fam.revision,
        "chat_template_sha256": chat_template_sha256(fam),
        "quantization": config.model.quantization,
        "quantization_validator": config.model.quantization_validator,
        # The loaded task's: its residual gate is part of the model that answers.
        "residual": config.task.exp.residual.model_dump(),
    }


def answer_settings(pv_cfg) -> dict:
    """How the answers stage answers: its engine, and the arguments it passes to apply_chat_template and generate(),
    hashed as they are passed."""
    settings = {
        "engine": pv_cfg.engine,
        "chat_template": {"enable_thinking": pv_cfg.enable_thinking},
        "generate": {**GREEDY, "max_new_tokens": pv_cfg.max_new_tokens},
    }
    if pv_cfg.engine == "hf":
        return {**settings, "batch_size": pv_cfg.answer_batch_size}
    vllm = pv_cfg.vllm
    # The checkpoint by its manifest, which hashes every tensor and file the engine loads.
    manifest = hashlib.sha256((vllm.checkpoint / "manifest.json").read_bytes()).hexdigest()
    return {**settings,
            "vllm": {"version": vllm.version, "manifest_sha256": manifest, "max_num_seqs": vllm.max_num_seqs,
                     "enforce_eager": vllm.enforce_eager}}


# Other score-deciding settings; max_models picks the models that ranks and rewards are among.
SCORE_FIELDS: tuple[str, ...] = ("jev.resolved", "max_models")
# Names select_models' rule for the rules hash: change it together with select_models.
MODEL_SELECTION = "top max_models model hashes by best incentive then lowest uid; every commit of each"
# With model_selection: all, names the rule in run_snapshot that replaces select_models: change it together with it.
ALL_SELECTION = ("every miner commit at the cutoff that names the active group, with no ranking and no cap; copies "
                 "of one model hash all kept")
# With rank_by: loss, names loss_rank's rule and how select_models uses it: change it together with them.
LOSS_SELECTION = ("per (cycle_number, validator_slot) group of the source's fresh, status-0, finite losses from the "
                  "window_training_cycles completed training cycles before the cutoff's and from min_cycle, a group "
                  "under min_group rows skipped, a row's loss minus its group's median; a miner's score the mean over "
                  "its rows with phase_end_block at or after its uid's registration block at the cutoff, whatever "
                  "hotkey the source names, both slots pooled, with at least min_samples rows; a uid the source "
                  "answers 404 for has no rows; model hashes by best score ascending, then unscored by incentive "
                  "descending, then lowest uid; by incentive when any other fetch fails after retries, its time "
                  "budget runs out or fewer than min_scored in-group miners are scored")
# Names reward's rule for copies of one model hash, hashed too: change it together with reward.originals and
# first_held.
COPY_REWARD = ("one place per model hash, to the hotkey whose verified signed miner commit on chain held it at the "
               f"earliest training cycle from {LOOKBACK_TRAINING_CYCLES} before the cutoff's to the cutoff's, read at "
               "each one's commit blocks; within that training cycle the earliest commitment block, then lowest uid, "
               "so a hash held since before the first of them goes by that first one's; none if that hotkey has no "
               "kept commit, fetched or not, that signed it; copies get 0")
# Names how reward scores a model (the reported model_score), hashed so a cycle scored under another rule is not
# rewarded under this one: change it together with reward.run_reward.
REWARD_RULE = "a qualified original's score is its mean expected_score over the pool prompts"
# Names judge.py's rules, hashed with judge_miner_prompts on: change it together with judge.py. Private data
# (judge_rules.json, read by config.py); None without it, when the switch is refused on.
JUDGE_RULES = JUDGE_RULES_DATA.get("judge_rules")


def rules_sha256(pv_cfg, base: dict) -> str:
    """Hash of what decides a score: the base model rows of the meta, the answer settings, the SCORE_FIELDS, the model
    selection, copy and reward rules, the Jev questions, the levels and format rules by task and, when they are judged,
    how miner prompts are judged."""
    settings = {field: get_nested_attr(pv_cfg, field) for field in SCORE_FIELDS}
    selection = {"model_selection": MODEL_SELECTION}
    if pv_cfg.model_selection == "all":  # no cap and no ranking: max_models and rank_by decide nothing
        del settings["max_models"]
        selection = {"model_selection": ALL_SELECTION, "model_selection_mode": "all"}
    if pv_cfg.judge_miner_prompts:  # only then: off, the hash stays the one from before miner prompts were judged
        settings |= {"miner_prompts": pv_cfg.miner_prompts.model_dump(mode="json"), "judge_rules": JUDGE_RULES}
    # Only then: by incentive, or with "all", the hash stays one without the loss settings.
    if pv_cfg.rank_by == "loss" and pv_cfg.model_selection == "top":
        settings |= {"rank_by": "loss", "loss_selection": LOSS_SELECTION,
                     "loss_rank": pv_cfg.loss_rank.model_dump(mode="json", include=set(loss_rank.LOSS_FIELDS))}
    return _sha256_json({**base, "answer": answer_settings(pv_cfg), "jev_questions": SCORE_QUESTIONS,
                         "jev_criteria_by_task": CRITERIA_BY_TASK, "format_rules_by_task": FORMAT_RULES,
                         **selection, "copy_reward": COPY_REWARD,
                         "reward_rule": REWARD_RULE, **settings})


def select_models(in_group: list, max_models: int, scores: dict[int, float] | None = None) -> list:
    """Every commit of the max_models model hashes ranked by their best (incentive, uid), or with scores by their
    best (score, uid) and then the unscored by (incentive, uid), so copies share a place."""
    if scores is None:
        ranked = sorted(in_group, key=lambda c: (-c.incentive, c.uid))
    else:
        ranked = sorted(in_group, key=lambda c: (0, scores[c.uid], c.uid) if c.uid in scores
                        else (1, -c.incentive, c.uid))
    # A commit with no model_hash is its own model (its fetch fails on the hash); it never joins another's.
    keys = [c.model_hash or c.uid for c in ranked]
    kept_keys = set(list(dict.fromkeys(keys))[:max_models])
    return [c for c, key in zip(ranked, keys, strict=True) if key in kept_keys]


def signed_commit_blocks(subtensor, netuid: int, block: int) -> dict[str, int]:
    """Each hotkey's commitment block as of `block`, from the commitments pallet's record, never the payload."""
    # The block of the latest write before `block`, not of the first ever.
    rows = subtensor.query_map(bittensor.storage.Commitments.CommitmentOf, [netuid], block=block)
    # A record without an integer block gets none: the snapshot records null rather than a guess.
    return {str(hotkey): record["block"] for hotkey, record in rows
            if isinstance(record, dict) and type(record.get("block")) is int}


def registration_blocks(subtensor, netuid: int, block: int) -> dict[int, int]:
    """Each uid's registration block as of `block`, from (uid, block) integer pairs; an entry that is not one is
    left out and counted in the log."""
    rows = subtensor.query_map(bittensor.storage.SubtensorModule.BlockAtRegistration, [netuid], block=block)
    blocks, bad = {}, 0
    for uid, value in rows:
        if type(uid) is int and type(value) is int:
            blocks[uid] = value
        else:
            bad += 1
    if bad:
        logger.warning("registration blocks that did not decode", entries=bad, block=block)
    return blocks


def rank_by_loss(config, subtensor, blocks: dict, in_group: list, out_dir: Path) -> loss_rank.LossRank:
    """loss_rank's scores of the in-group miners from the chain at the cutoff, or its fallback: never raises."""
    netuid, block = config.chain.netuid, blocks["commit_2_block"]
    try:
        metagraph = bt_compat.get_metagraph(subtensor, netuid, block=block)
        registered = registration_blocks(subtensor, netuid, block)
        uids = [int(uid) for uid in metagraph.uids]
        # Without every uid's block, a uid's new holder could be scored on its former holder's rows.
        if (known := len(set(uids) & registered.keys())) < len(uids):
            return loss_rank.fallback(f"registration_blocks: {known} of {len(uids)}")
        return loss_rank.rank(config.prompt_validator.loss_rank, out_dir, blocks["training_cycle"], uids, registered,
                              {c.uid for c in in_group})
    except Exception as e:  # ranking by loss only improves the selection: never a reason to stop the cycle
        return loss_rank.fallback(f"error: {type(e).__name__}: {e}"[:200])


class FirstSeen(NamedTuple):
    first_seen_training_cycle: int | None
    first_seen_hotkey: str | None


def _unpruned(config, block: int | str, read):
    """read(), or stop if the node has pruned `block`: the result is kept, so never the chain head's instead."""
    try:
        return read()
    except Exception as e:
        # Substrate says "State already discarded"; chain.py matches "State discarded". Both contain this.
        if "discarded" not in str(e):
            raise
        raise RuntimeError(f"{config.chain.network} has pruned block {block}: use an archive node") from None


class ChainRead(NamedTuple):
    checkpoints: list
    commit_blocks: dict[str, int]


def read_chain(config, subtensor, commit_1_block: int, commit_2_block: int) -> ChainRead:
    """The commits joined at one training cycle's two blocks, and the commitment blocks at the first."""
    commit_blocks = _unpruned(config, commit_1_block,
                              lambda: signed_commit_blocks(subtensor, config.chain.netuid, commit_1_block))
    checkpoints = _unpruned(config, f"{commit_1_block} or {commit_2_block}", lambda: build_chain_checkpoints_at_blocks(
        config, subtensor, commit_1_block=commit_1_block, commit_2_block=commit_2_block, for_role="miner",
        head_fallback=False,
    ).checkpoints)
    return ChainRead(checkpoints, commit_blocks)


def signed(model: dict) -> bool:
    try:
        return verify_message(model["hotkey"], message=model["model_hash"], signature_hex=model["signed_model_hash"])
    except Exception:  # a malformed hotkey or signature signs nothing
        return False


def signed_by_hash(models: list[dict]) -> dict[str, list[dict]]:
    """model_hash -> its commits whose signature verifies, by earliest signed-hash commitment block, then lower uid."""
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for m in models:
        # Fetched or not, but a commit whose signature doesn't verify never committed to the hash.
        if m["model_hash"] is not None and signed(m):
            by_hash[m["model_hash"]].append(m)
    # A commit without a readable commitment block sorts after every block.
    return {model_hash: sorted(same, key=lambda m: (m.get("signed_commit_block") is None,
                                                    m.get("signed_commit_block") or 0, m["uid"], m["hotkey"]))
            for model_hash, same in sorted(by_hash.items())}


def is_past(cycle: int, scoring_cycle_id: int) -> bool:
    """An earlier cycle of the same kind (real or test): a rerun of an old cycle must not see the cycles after it."""
    first_test = dataless.TEST_CYCLE_FIRST_ID
    return cycle < scoring_cycle_id and (cycle >= first_test) == (scoring_cycle_id >= first_test)


def _cache(runs_dir: Path, commit_1_block: int, commit_2_block: int, read: ChainRead) -> list[dict]:
    """The CHAIN_FIELDS of each miner commit whose signature verifies, by uid, cached unless the read found none."""
    rows = ({**{field: getattr(c, field) for field in CHECKPOINT_FIELDS},
             "signed_commit_block": read.commit_blocks.get(c.hotkey)} for c in read.checkpoints if c.miner_commit)
    # Verified before caching, so only the miner/validator split depends on the read: it uses the whitelist of then.
    rows = sorted(filter(signed, rows), key=lambda row: row["uid"])
    if read.checkpoints:  # no commit at all may be a node's gap: read again next time rather than keep it
        write_rows(_cache_path(runs_dir, commit_1_block, commit_2_block), rows)
    return rows


def _cache_path(runs_dir: Path, commit_1_block: int, commit_2_block: int) -> Path:
    # By block: the chain at a block never changes, and real and test cycles read the same blocks.
    return Path(runs_dir) / "chain_commits" / f"{commit_1_block}-{commit_2_block}.jsonl"


def training_cycle_rows(config, subtensor, scoring_cycle_id: int, training_cycle: int) -> list[dict]:
    """One training cycle's verified miner commits from its cache, else from the chain (see _cache)."""
    blocks = training_cycle_blocks(config, scoring_cycle_id, training_cycle)
    path = _cache_path(config.prompt_validator.runs_dir, blocks.commit_1_block, blocks.commit_2_block)
    try:
        rows = read_rows(path)
        if all(isinstance(row, dict) and list(row) == list(CHAIN_FIELDS) for row in rows):
            return rows
        raise ValueError("rows without the chain fields")
    except FileNotFoundError:
        pass
    except ValueError as e:  # never trusted: the chain has the same rows
        logger.warning("chain commits cache damaged, read again", path=str(path),
                       error=f"{type(e).__name__}: {e}"[:200])
    c1, c2 = blocks.commit_1_block, blocks.commit_2_block
    rows = _cache(config.prompt_validator.runs_dir, c1, c2, read_chain(config, subtensor, c1, c2))
    logger.info("training cycle commits read from the chain", training_cycle=training_cycle, commits=len(rows))
    return rows


def first_held(config, subtensor, scoring_cycle_id: int, cutoff_cycle: int, hashes: set) -> dict[str, FirstSeen]:
    """model_hash -> the earliest training cycle in the lookback whose verified signed commit held it, and its holder.

    A copier sees the weights only after the author's commit_1 write (the reference miner signs, then uploads), so
    within a training cycle it signs at a later commitment block; the author loses only by publishing before signing.
    """
    found: dict[str, FirstSeen] = {}
    # Oldest first, so a pruned block stops the snapshot before the rest is read. Fixed by decision: a hash held by
    # several hotkeys since before the first goes by the first one's commitment block, then uid.
    for training_cycle in range(cutoff_cycle - LOOKBACK_TRAINING_CYCLES, cutoff_cycle + 1):
        rows = training_cycle_rows(config, subtensor, scoring_cycle_id, training_cycle)
        for model_hash, same in signed_by_hash([row for row in rows if row["model_hash"] in hashes]).items():
            found.setdefault(model_hash, FirstSeen(training_cycle, same[0]["hotkey"]))
    return found


def check_rules(config, meta: dict) -> None:
    """Stop a later stage whose config would not reproduce the snapshot's rules."""
    rules = rules_sha256(config.prompt_validator, base_rows(config))
    if rules != meta["rules_sha256"]:
        raise RuntimeError(f"config rules_sha256 {rules[:12]} is not the snapshot's {meta['rules_sha256'][:12]}")


def sync_owner_task(config) -> None:
    """Switch onto the owner's task now, or stop: a snapshot on a stale task would score the wrong group."""
    active = get_active_task(config.cycle)
    if active is None:
        on = config.task.expert_group_name
        raise RuntimeError(f"owner API unreachable: cannot confirm the active task (config is on {on})")
    ensure_active_task(config, active.name)
    if config.task.expert_group_name != active.name:
        on = config.task.expert_group_name
        raise RuntimeError(f"could not switch onto the owner's task {active.name}: still on {on}")


class Prompts(NamedTuple):
    prompts: list[dict]
    meta: dict  # the fields their source adds to the snapshot meta
    miner: list[dict] | None = None  # with judge_miner_prompts: the cycle's miner prompts
    request: dict | None = None  # and the customer request they are written for, or {"missing": <why>}


def customer_request(config, client: dataless.DatalessClient, task: str) -> dict:
    """The customer request behind the cycle's task, or {"missing": <why there is none>}."""
    mp = config.prompt_validator.miner_prompts
    if config.prompt_validator.dataless.test_mode:  # the server serves requests from its real tables only
        if mp.request_file is None:
            return {"missing": "no_request_file"}
        request = json.loads(mp.request_file.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError(f"miner_prompts.request_file {mp.request_file} must hold a JSON object")
    else:
        try:
            request = client.task_request(task)
        except dataless.DatalessError as e:
            if e.status != 404 or e.code not in NO_REQUEST:
                raise
            return {"missing": e.code}
    if request.get("customer_task_id", task) != task:  # every verdict would be against another customer's request
        logger.warning("customer request not used: it is for another task", task=task,
                       request_task=request["customer_task_id"])
        return {"missing": "request_for_other_task"}
    return request


def dataless_prompts(config, scoring_cycle_id: int) -> Prompts:
    """The owner's internal prompts for the scoring cycle from the dataless-api, and what the meta records of them.

    The API lists a prompt cycle only from shortly after its last block until the next one closes, so this waits
    for that (the cutoff stage ends before the prompt cycle does). Miner prompts are kept only with
    judge_miner_prompts, with the customer request; the snapshot hash covers the listing either way.
    The internal prompts carry no customer request, so Jev scores them on the prompt alone.
    """
    client = dataless.validator_client(config)
    while True:
        view = client.wait_until_listed(scoring_cycle_id)
        # Every stage passes the scoring cycle id as the prompt cycle's: both must count one grid from block 0.
        if not config.prompt_validator.dataless.test_mode:
            length = prompt_cycle_length(config)
            start = view["cycle_id"] * length
            if (view["cycle_length"], view["start_block"], view["cutoff_block"]) != (length, start, start + length - 1):
                raise RuntimeError(f"the dataless-api's prompt cycle {view['cycle_id']} has blocks "
                                   f"{view['start_block']} to {view['cutoff_block']} and is {view['cycle_length']} "
                                   f"long: a scoring cycle here is {length} blocks from block 0, cut off at its last")
        try:
            listing = client.list_prompts(scoring_cycle_id)
            break
        except dataless.DatalessError as e:
            if e.code != "before_cutoff":  # the server's snapshot delay is longer than ours: wait for its block
                raise
            blocks = max(e.extra.get("ready_at_block", 0) - e.extra.get("head", 0), 1)
            logger.info("the dataless-api lists the cycle later", cycle=scoring_cycle_id, blocks=blocks)
            time.sleep(min(blocks * 12.0, 600.0))
    internal = sorted((p for p in listing["prompts"] if p["source"] == "internal"), key=lambda p: p["prompt_id"])
    if not internal:
        raise RuntimeError(f"the dataless-api has no internal prompts for prompt cycle {scoring_cycle_id}")
    prompts = [
        {"prompt_id": str(p["prompt_id"]), "request_id": p["customer_task_id"], "prompt": p["text"], "request": None,
         "customer_task_id": p["customer_task_id"], "content_hash": p["content_hash"]}
        for p in internal
    ]
    logger.info("prompts from the dataless-api", cycle=scoring_cycle_id, internal=len(internal),
                miner_not_scored=len(listing["prompts"]) - len(internal))
    meta = {
        "prompt_source": "dataless",
        "dataless_snapshot_hash": listing["snapshot_hash"],
        "dataless_snapshot_count": listing["snapshot_count"],
        "rollout_phase": view["policy"]["rollout_phase"],
        # Miners with a prompt set this cycle: the API takes no summary for them until their set is judged.
        "dataless_set_hotkeys": sorted({p["hotkey"] for p in listing["prompts"] if p["source"] == "miner"}),
    }
    if not config.prompt_validator.judge_miner_prompts:
        return Prompts(prompts, meta)
    miner = sorted(({"prompt_id": str(p["prompt_id"]), **{field: p[field] for field in MINER_PROMPT_FIELDS}}
                    for p in listing["prompts"] if p["source"] == "miner"),
                   key=lambda p: (p["block"], int(p["prompt_id"])))
    if not miner:  # nothing to judge: no request read, so its route can never stop the cycle
        return Prompts(prompts, meta, miner, {"missing": "no_miner_prompts"})
    pool_tasks = {p["customer_task_id"] for p in internal}
    tasks = sorted(pool_tasks | {p["customer_task_id"] for p in miner})
    if len(tasks) != 1:  # judged against one customer request only: judge nothing rather than the wrong request
        logger.warning("miner prompts not kept for judging: the cycle has several customer tasks", tasks=tasks,
                       pool_tasks=sorted(pool_tasks))
        return Prompts(prompts, meta, miner, {"missing": "several_tasks"})
    return Prompts(prompts, meta, miner, customer_request(config, client, tasks[0]))


def load_prompts(config, scoring_cycle_id: int) -> Prompts:
    """The scoring cycle's prompts, and any meta fields their source adds."""
    if config.prompt_validator.prompt_source == "dataless":
        return dataless_prompts(config, scoring_cycle_id)
    return Prompts(load_internal_prompts(config.prompt_validator.prompts_path), {})


def snapshot_meta(config, scoring_cycle_id: int) -> dict:
    """The snapshot's meta row, refused if this config would not reproduce its rules (see check_rules)."""
    pv = config.prompt_validator
    meta = read_rows(stage_path(pv.runs_dir, scoring_cycle_id, "snapshot"))[0]
    if rules_sha256(pv, base_rows(config)) != meta["rules_sha256"]:
        raise RuntimeError("score-deciding settings changed since the snapshot: rerun --stage snapshot first")
    return meta


def run_snapshot(config, scoring_cycle_id: int) -> list[dict]:
    pv = config.prompt_validator
    sync_owner_task(config)  # the owner's task as of now, not as of a boot before a long cutoff wait
    blocks = read_rows(stage_path(pv.runs_dir, scoring_cycle_id, "cutoff"))[0]
    prompts, prompt_meta, miner, request = load_prompts(config, scoring_cycle_id)

    subtensor = bittensor.Subtensor(network=config.chain.network)
    first_block = blocks["commit_1_block"]
    netuid = config.chain.netuid
    tempo = _unpruned(config, first_block, lambda: bt_compat.get_tempo(subtensor, netuid, block=first_block))
    if tempo is None:
        raise RuntimeError(f"subnet {netuid} absent at block {first_block} on {config.chain.network}")
    cut = read_chain(config, subtensor, first_block, blocks["commit_2_block"])
    checkpoints, commit_blocks = cut
    # Cached now, so a later scoring cycle's lookback reads it from disk.
    _cache(pv.runs_dir, first_block, blocks["commit_2_block"], cut)

    # Like the training validator's get_miners_from_commit, only miner commits count; a validator's names a group too.
    miners = [c for c in checkpoints if c.miner_commit]
    group = config.task.exp.group_id
    by_group = Counter(c.expert_group for c in miners)
    commits_by_group = {str(g): n for g, n in sorted(by_group.items())}
    task = config.task.expert_group_name
    logger.info(
        "commits read", task=task, group=group, commits_by_group=commits_by_group,
        dropped_non_miner=len(checkpoints) - len(miners),
    )
    # The snapshot reads the owner's task now, not at the cutoff.
    if by_group and not by_group[group]:
        raise RuntimeError(
            f"no commit at the cutoff names task {task}'s group {group}, but others do: {commits_by_group}"
        )
    # The miners whose commit names the active group, as get_miners_from_commit keeps them. Read at the cutoff block,
    # where the training validator reads the chain head at Submission start.
    in_group = [c for c in miners if c.expert_group == group]
    out_dir = stage_path(pv.runs_dir, scoring_cycle_id, "snapshot").parent
    top = pv.model_selection == "top"
    loss = rank_by_loss(config, subtensor, blocks, in_group, out_dir) if top and pv.rank_by == "loss" else None
    # Incentive from the metagraph at commit_2_block, so a rerun keeps the same miners. A model with 0 incentive is
    # kept within the top max_models, as the training validator keeps miners with 0 incentive. "all" keeps them all.
    kept = select_models(in_group, pv.max_models, loss and loss.scores) if top else list(in_group)
    hashes = len({c.model_hash or c.uid for c in kept})
    logger.info(
        "every model kept" if not top else
        "models kept by incentive" if loss is None or not loss.scores else "models kept by loss", kept=len(kept),
        model_hashes=hashes, copies_kept=len(kept) - hashes,
        dropped_over_max=len(in_group) - len(kept), zero_incentive_kept=sum(c.incentive <= 0 for c in kept),
        no_commit_block=sum(c.hotkey not in commit_blocks for c in kept), max_models=pv.max_models if top else None,
    )
    # Recorded in the rows, so reward reruns on this cycle's files alone.
    cutoff_cycle = blocks["training_cycle"]
    seen = first_held(config, subtensor, scoring_cycle_id, cutoff_cycle, {c.model_hash for c in kept})
    models = sorted(
        ({**{field: getattr(c, field) for field in (*MODEL_FIELDS, "incentive")},
          "signed_commit_block": commit_blocks.get(c.hotkey),
          **seen.get(c.model_hash, FirstSeen(None, None))._asdict()} for c in kept),
        key=lambda m: m["uid"],
    )

    base = base_rows(config)  # the active task's residual, as synced above
    meta = {
        "scoring_cycle_id": scoring_cycle_id,
        "commit_1_block": blocks["commit_1_block"],
        "commit_2_block": blocks["commit_2_block"],
        "active_task": config.task.expert_group_name,
        "expert_group": group,
        # "all": the mode in its place, so the meta never names a cap that kept nothing out.
        **({"max_models": pv.max_models} if top else {"model_selection": "all"}),
        "commits_by_group": commits_by_group,
        "originals_training_cycles": [cutoff_cycle - LOOKBACK_TRAINING_CYCLES, cutoff_cycle],
        **base,
        "rules_sha256": rules_sha256(pv, base),
        **prompt_meta,
    }
    if loss is not None:  # only then: by incentive, the meta and its hash are as before loss ranking
        # Model hashes placed by incentive after the scored ones.
        fill = hashes - len({c.model_hash or c.uid for c in kept if loss.scores and c.uid in loss.scores})
        meta |= {"rank_by": "loss", "loss_rank": {**loss.meta, "fill": fill}}
    judged = {"miner_prompts": miner, "request": request} if pv.judge_miner_prompts else {}
    source = {"loss_source": loss.rows} if loss is not None else {}
    meta["snapshot_sha256"] = _sha256_json({"prompts": prompts, "models": models, "meta": meta, **judged, **source})
    if loss is not None:
        meta["loss_rank"]["source"] = loss.source
    write_rows(out_dir / "prompts.jsonl", prompts)
    for name, rows in (("miner_prompts.jsonl", miner), ("request.json", [request])):
        if pv.judge_miner_prompts:
            write_rows(out_dir / name, rows)  # request.json: one JSON line, which json.load reads as the object
        else:  # an earlier run's with the switch on: later cycles would take it for this snapshot's
            (out_dir / name).unlink(missing_ok=True)
    if loss is None:  # an earlier run's by loss: a later rerun by loss would take it for this cycle's
        for name in (loss_rank.SOURCE_FILE, loss_rank.KEY_FILE):
            (out_dir / name).unlink(missing_ok=True)
    write_rows(out_dir / "models.jsonl", models)
    logger.info("snapshot taken", prompts=len(prompts), models=len(models), snapshot_sha256=meta["snapshot_sha256"])
    return [meta]
