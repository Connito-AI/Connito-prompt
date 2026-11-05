"""Stage 6: reward the model miners by their mean expected_score over the pool prompts, and weight the best of them.

A miner (uid, hotkey, model_hash) qualifies with at least `reward_min_prompts` scored prompts; only qualified
originals (see originals; copies of a model_hash get 0) are ranked. The top `reward_top_miners` by mean, a tie at
the cut included, share weight 1 by `reward_shares`; a mean of 0 gets none. Phase 1 writes reward.jsonl only.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import NamedTuple

from connito.prompt_validator.judge import selected_prompts
from connito.prompt_validator.snapshot import signed_by_hash, snapshot_meta
from connito.prompt_validator.stage_io import read_rows, stage_path
from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)

ROW_FIELDS = (
    "rank", "uid", "hotkey", "model_hash", "signed_commit_block", "copy_of", "n_scored", "qualified", "mean_score",
    "weight",
)


class Original(NamedTuple):
    uid: int | None  # None: the hotkey that first held the hash has no kept, signed commit of it this cycle
    hotkey: str


def originals(models: list[dict]) -> dict[str, Original]:
    """model_hash -> its original: the hotkey whose signed commit held it at the earliest training cycle read."""
    found = {}
    for model_hash, same in signed_by_hash(models).items():
        # The snapshot records it on every commit of the hash; a signed commit holds it at the cutoff at least.
        seen = same[0]["first_seen_hotkey"]
        first = next((m for m in same if m["hotkey"] == seen), None)
        original = Original(first["uid"], first["hotkey"]) if first else Original(None, seen)
        copies = [m for m in same if m is not first]
        if copies:
            logger.info("copies of one model: only the original is ranked", model_hash=model_hash[:12],
                        uid=original.uid, hotkey=original.hotkey[:8],
                        block=first["signed_commit_block"] if first else None,
                        first_seen=same[0]["first_seen_training_cycle"],
                        copies=[(m["uid"], m["signed_commit_block"]) for m in copies])
        found[model_hash] = original
    return found


def run_reward(config, scoring_cycle_id: int) -> list[dict]:
    pv = config.prompt_validator
    snapshot_meta(config, scoring_cycle_id)  # scores made under other rules must not be rewarded
    scores_path = stage_path(pv.runs_dir, scoring_cycle_id, "scores")
    if not scores_path.exists():
        raise RuntimeError(f"no scores to reward: {scores_path} is missing; run --stage scores first")
    snapshot_dir = stage_path(pv.runs_dir, scoring_cycle_id, "snapshot").parent
    prompt_ids = {p["prompt_id"] for p in read_rows(snapshot_dir / "prompts.jsonl")}
    if len(prompt_ids) < pv.reward_min_prompts:  # else nobody could qualify, and every weight would quietly be 0
        raise RuntimeError(
            f"the snapshot has {len(prompt_ids)} prompts, fewer than reward_min_prompts ({pv.reward_min_prompts}): "
            "no miner could qualify"
        )

    miner_prompt_ids = {p["prompt_id"] for p in selected_prompts(pv, scoring_cycle_id)}
    by_miner: dict[tuple, dict[str, float]] = defaultdict(dict)  # miner -> prompt_id -> expected_score
    for row in read_rows(scores_path):
        if row["prompt_id"] in miner_prompt_ids:  # scored for the report only: a model's reward is over the pool
            continue
        miner = (row["uid"], row["hotkey"], row["model_hash"])
        if row["prompt_id"] not in prompt_ids:
            raise RuntimeError(f"uid {row['uid']} has a score for {row['prompt_id']!r}, which is not in the snapshot")
        if row["prompt_id"] in by_miner[miner]:
            raise RuntimeError(f"uid {row['uid']} has two scores for {row['prompt_id']!r}")
        by_miner[miner][row["prompt_id"]] = row["expected_score"]

    models = read_rows(snapshot_dir / "models.jsonl")
    blocks = {(m["uid"], m["hotkey"]): m["signed_commit_block"] for m in models}
    first = originals(models)
    # A copy gets 0 even when its original was not fetched or not committed: that model then has no ranked row.
    copy_of = {m: first[m[2]].hotkey for m in by_miner if m[2] in first and first[m[2]] != Original(*m[:2])}
    qualified = {m for m, scores in by_miner.items() if len(scores) >= pv.reward_min_prompts}
    ranked = qualified - copy_of.keys()
    means = {miner: math.fsum(by_miner[miner].values()) / len(by_miner[miner]) for miner in ranked}  # order-free
    # A miner's place is the number of ranked miners with a higher mean, so ties share a place.
    ranks = {miner: sum(other > mean for other in means.values()) for miner, mean in means.items()}
    weighted = {miner for miner, rank in ranks.items() if rank < pv.reward_top_miners and means[miner] > 0}
    total = sum(means[miner] for miner in weighted)
    shares = {miner: means[miner] / total if pv.reward_shares == "proportional" else 1 / len(weighted)
              for miner in weighted}

    rows = []
    for miner, scores in by_miner.items():
        uid, hotkey, model_hash = miner
        rows.append({
            "rank": ranks.get(miner), "uid": uid, "hotkey": hotkey, "model_hash": model_hash,
            "signed_commit_block": blocks[(uid, hotkey)], "copy_of": copy_of.get(miner),
            "n_scored": len(scores), "qualified": miner in qualified,
            "mean_score": means.get(miner, 0.0), "weight": shares.get(miner, 0.0),
        })
    # Qualified first, best first (rank 0; equal means share a rank), then by uid; the unqualified last, by uid.
    rows.sort(key=lambda r: (not r["qualified"], -r["mean_score"], r["uid"], r["hotkey"], r["model_hash"]))
    return [{field: row[field] for field in ROW_FIELDS} for row in rows]
