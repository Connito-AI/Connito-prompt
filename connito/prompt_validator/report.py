"""Stage 7: report the scoring cycle to the dataless-api: the run, the models, each miner's reward, and every Jev
score with the model's output for the models the server keeps outputs of; then publish.

Only for a snapshot taken with prompt_source "dataless". Every row is written once on the server and an identical
re-post is a no-op, so a report that stops part way is rerun with --stage report. After publish the cycle is final.
With judge_miner_prompts, the judge's verdicts go first, then the selected miner prompts' scores beside the pool's.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import datetime, timezone

from connito.prompt_validator import dataless
from connito.prompt_validator.fetch import prune_shards
from connito.prompt_validator.snapshot import snapshot_meta
from connito.prompt_validator.stage_io import append_published, published_cycles, read_rows, stage_path
from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)

# fetch.py's reasons, as the server names them.
FETCH_RESULTS = {
    None: "ok", "timeout": "timeout", "repo_unavailable": "not_found", "file_missing": "not_found",
    "hash": "hash_mismatch", "signature": "hash_mismatch", "bad_residual_shard": "invalid",
    "download_error": "invalid", "unknown": "invalid",
}
_REVISION = re.compile(r"[0-9a-f]{7,40}")
# A miner's model_hash is blake2b with a 24-byte digest (shared/helper.py hash_model_bytes): 48 hex characters.
# 64 is taken too, as the server takes both.
_MODEL_HASH = re.compile(r"[0-9a-f]{48}|[0-9a-f]{64}")
# The server's patterns (its model schema): one model it would refuse fails the batch.
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_TASK = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def keep_count(n: int) -> int:
    """How many models the server keeps outputs for: max(ceil(n/10), 10), at most n."""
    return min(n, max(math.ceil(n / 10), 10))


def kept_hotkeys(hotkeys: set[str], summary: list[dict]) -> set[str]:
    """The model hotkeys whose outputs publish keeps, as the server picks them: keep_count of all the run's models,
    by the summary's model_score, highest first; a model with no summary row or a null score comes last; ties by
    hotkey bytes."""
    scores = {r["hotkey"]: r["model_score"] for r in summary}
    ranked = sorted(hotkeys, key=lambda h: (scores.get(h) is None, -(scores.get(h) or 0), h.encode()))
    return set(ranked[: keep_count(len(hotkeys))])


def _output(text: str) -> str:
    """The answer as the server takes it: without NUL characters (refused), cut at a character boundary to its cap
    on one output."""
    text = text.replace("\x00", "")
    data = text.encode()
    return text if len(data) <= dataless.OUTPUT_MAX_BYTES else data[: dataless.OUTPUT_MAX_BYTES].decode(errors="ignore")


def model_rows(meta: dict, models: list[dict], fetch: list[dict]) -> list[dict]:
    """One row per snapshot model in the server's shape. A model whose commit the server can't take is left out."""
    reasons = {(r["uid"], r["hotkey"]): r["reason"] for r in fetch}
    rows = []
    for m in models:
        row = {
            "hotkey": m["hotkey"], "uid": m["uid"], "training_task_id": meta["active_task"],
            "hf_repo": m["hf_repo_id"], "revision": m["hf_revision"], "model_hash": m["model_hash"],
            "commit_block": meta["commit_2_block"],
            "fetch_result": FETCH_RESULTS.get(reasons.get((m["uid"], m["hotkey"]), "unknown"), "invalid"),
        }
        bad = [field for field, pattern in (("revision", _REVISION), ("model_hash", _MODEL_HASH), ("hf_repo", _REPO),
                                            ("training_task_id", _TASK))
               if not isinstance(row[field], str) or not pattern.fullmatch(row[field])]
        if bad:
            logger.warning("model not reported: the dataless-api would refuse its commit", uid=m["uid"],
                           fields=bad)
            continue
        rows.append(row)
    return rows


def usefulness_rows(verdicts: list[dict]) -> list[dict]:
    """The judge's verdicts in the server's shape: a reason only when failed, relevance 0 (not null) when not asked."""
    return [{"prompt_id": int(v["prompt_id"]), "relevance": v["relevance"] or 0, "usefulness": v["usefulness"],
             "l2_rank": v["l2_rank"], "passed": v["passed"], **({} if v["passed"] else {"reason": v["reason"]})}
            for v in verdicts]


def check_sets(answered: list[dict], sets: list[dict]) -> None:
    """Stop unless the server judged exactly our sets, each as the judge stage did."""
    theirs = {s["hotkey"]: (s["prompts"], s["passed"], s["set_qualified"]) for s in answered}
    ours = {s["hotkey"]: (s["prompts"], s["passed"], s["qualified"]) for s in sets}
    differ = sorted(h for h in ours.keys() | theirs.keys() if ours.get(h) != theirs.get(h))
    if differ:
        raise RuntimeError(f"the dataless-api's verdict on {len(differ)} prompt sets is not the judge stage's "
                           f"(prompts, passed, qualified), e.g. {differ[0]}: ours {ours.get(differ[0])}, theirs "
                           f"{theirs.get(differ[0])}; nothing more reported")


def prompt_model_scores(scores: list[dict], selected: list[str]) -> dict[str, float]:
    """Selected prompt_id -> the mean expected_score of the given answers to it, one per model_hash, as ranked."""
    by_prompt: dict[str, dict[str, float]] = defaultdict(dict)
    for s in scores:
        by_prompt[s["prompt_id"]][s["model_hash"]] = s["expected_score"]
    return {p: sum(by_prompt[p].values()) / len(by_prompt[p]) for p in selected if by_prompt[p]}


def set_fields(sets: list[dict], verdicts: list[dict], means: dict[str, float]) -> dict[str, dict]:
    """hotkey -> its judged set's summary fields; no prompt weight is set yet, so 0."""
    fields = {}  # only a qualified set's prompts are selected: a rejected set has no prompt_model_score
    for s in sets:
        answered = [means[v["prompt_id"]] for v in verdicts if v["hotkey"] == s["hotkey"] and v["prompt_id"] in means]
        fields[s["hotkey"]] = {"prompt_usefulness_pass": s["passed"] / s["prompts"],
                               "prompt_model_score": sum(answered) / len(answered) if answered else None,
                               "prompt_weight": 0.0}
    return fields


def summary_rows(reward: list[dict], reported: set[str], set_hotkeys: set[str],
                 judged: dict[str, dict]) -> list[dict]:
    """Each reported miner's reward row, with its judged set's fields; a miner with a set not judged is left out."""
    left_out = sorted(r["uid"] for r in reward
                      if r["hotkey"] in reported and r["hotkey"] in set_hotkeys and r["hotkey"] not in judged)
    if left_out:
        logger.warning("summary not reported for miners with a prompt set: their sets are not judged",
                       uids=left_out)
    return [
        {"hotkey": r["hotkey"], "uid": r["uid"],
         "model_score": r["mean_score"] if r["qualified"] else None, "model_weight": r["weight"],
         **judged.get(r["hotkey"], {})}
        for r in reward if r["hotkey"] in reported and (r["hotkey"] not in set_hotkeys or r["hotkey"] in judged)
    ]


def record_published(config, scoring_cycle_id: int, meta: dict, source: str, run: dict) -> None:
    """Add the cycle to the published ledger (`source`: who published it), then delete its shards."""
    pv = config.prompt_validator
    append_published(pv.runs_dir, {
        "cycle_id": scoring_cycle_id, "test": pv.dataless.test_mode, "source": source,
        "published_at": run.get("updated_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot_sha256": meta.get("snapshot_sha256"), "rules_sha256": meta["rules_sha256"],
        "dataless_snapshot_hash": meta["dataless_snapshot_hash"], "validator_hotkey": run.get("validator_hotkey"),
        "run_block": run.get("block"), "url": pv.dataless.url,
    })
    prune_shards(pv, scoring_cycle_id)


def published_elsewhere(config, scoring_cycle_id: int, meta: dict) -> list[dict]:
    """A cycle this hotkey published before, from another runs_dir or host, or before a crash lost its ledger row."""
    logger.warning("prompt cycle already published on the dataless-api; recorded as done, not reported again",
                   cycle=scoring_cycle_id)
    record_published(config, scoring_cycle_id, meta, "server", {})
    return [{"cycle_id": scoring_cycle_id, "status": "published", "published_before": "server"}]


def run_report(config, scoring_cycle_id: int) -> list[dict]:
    pv = config.prompt_validator
    if scoring_cycle_id in published_cycles(pv.runs_dir):
        logger.info("already published: in the ledger", cycle=scoring_cycle_id)
        prune_shards(pv, scoring_cycle_id)  # one a crash left between its ledger row and its prune
        return [{"cycle_id": scoring_cycle_id, "status": "published", "published_before": "ledger"}]
    meta = snapshot_meta(config, scoring_cycle_id)
    if meta.get("prompt_source") != "dataless":
        raise RuntimeError("only a snapshot taken with prompt_source \"dataless\" is reported to the dataless-api")
    snapshot_dir = stage_path(pv.runs_dir, scoring_cycle_id, "snapshot").parent
    stages = (("judge",) if pv.judge_miner_prompts else ()) + ("fetch", "answers", "scores", "reward")
    paths = {name: stage_path(pv.runs_dir, scoring_cycle_id, name) for name in stages}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"nothing to report yet: no {', '.join(missing)} for scoring cycle {scoring_cycle_id}")
    models = read_rows(snapshot_dir / "models.jsonl")
    fetch, answers, scores, reward = (read_rows(paths[name]) for name in ("fetch", "answers", "scores", "reward"))
    verdicts, sets, selected = [], [], []
    if pv.judge_miner_prompts:
        verdicts, sets, selected = (read_rows(paths["judge"].parent / name)
                                    for name in ("verdicts.jsonl", "sets.jsonl", "selected.jsonl"))

    client = dataless.validator_client(config)
    model_batch = model_rows(meta, models, fetch)
    if not model_batch:
        raise RuntimeError(f"no model of scoring cycle {scoring_cycle_id} can be reported: nothing to publish")
    try:
        client.start_run(scoring_cycle_id, {
            "rollout_phase": meta["rollout_phase"], "snapshot_hash": meta["dataless_snapshot_hash"],
            "jev_version": pv.jev.resolved, "validator_code_version": pv.dataless.validator_code_version,
            "rules_sha256": meta["rules_sha256"],
            # Answers are greedy: no top-k sampling and no seed.
            "top_k": 0, "max_tokens": pv.max_new_tokens, "seed": 0,
        })
    except dataless.DatalessError as e:
        if e.code == "cycle_published":
            return published_elsewhere(config, scoring_cycle_id, meta)
        if e.code != "cycle_not_current":
            raise
        # A run starts only while its cycle is the last closed one; one started then still takes rows and publish.
        logger.info("the next prompt cycle has closed: going on with the run, if one was started in time",
                    cycle=scoring_cycle_id)

    try:
        # First: a miner prompt's score, and its miner's summary, are taken only once the server has judged its set,
        # and a set it judged otherwise than the judge stage stops the report before anything of it is posted.
        check_sets(client.post_prompt_usefulness(scoring_cycle_id, usefulness_rows(verdicts)), sets)
        model_ids = client.post_models(scoring_cycle_id, model_batch)
    except dataless.DatalessError as e:
        # The server checks "current cycle" before "published", so a late rerun of a published cycle gets here.
        if e.code == "cycle_published":
            return published_elsewhere(config, scoring_cycle_id, meta)
        if e.code == "cycle_run_missing":
            raise RuntimeError(
                f"no run was started for prompt cycle {scoring_cycle_id} before the next one closed: "
                "it can no longer be reported"
            ) from None
        raise
    # Over the models the server has a row for, as its model scores are.
    means = prompt_model_scores([s for s in scores if s["hotkey"] in model_ids], [s["prompt_id"] for s in selected])
    summary = summary_rows(reward, set(model_ids), set(meta["dataless_set_hotkeys"]),
                           set_fields(sets, verdicts, means))
    # Before the scores: the server keeps outputs by the summary's model_score, and an output is written once.
    client.post_summary(scoring_cycle_id, summary)
    kept = kept_hotkeys(set(model_ids), summary)
    text = {(a["uid"], a["hotkey"], a["prompt_id"]): a["text"] for a in answers}
    model_scores = []
    for s in scores:
        if s["hotkey"] not in model_ids:
            continue
        row = {"model_id": model_ids[s["hotkey"]], "prompt_id": int(s["prompt_id"]), "jev_score": s["expected_score"]}
        if s["hotkey"] in kept:
            row["output"] = _output(text[(s["uid"], s["hotkey"], s["prompt_id"])])
        model_scores.append(row)
    client.post_model_scores(scoring_cycle_id, model_scores)
    client.post_prompt_model_scores(scoring_cycle_id, [{"prompt_id": int(p), "prompt_model_score": score}
                                                       for p, score in means.items()])
    run = client.publish(scoring_cycle_id)
    record_published(config, scoring_cycle_id, meta, "this_run", run)

    outputs = sum("output" in r for r in model_scores)
    logger.info("reported to the dataless-api", cycle=scoring_cycle_id, models=len(model_ids), summary=len(summary),
                model_scores=len(model_scores), outputs=outputs, prompt_usefulness=len(verdicts),
                prompt_model_scores=len(means), status=run.get("status"))
    return [{"cycle_id": scoring_cycle_id, "status": run.get("status"), "models": len(model_ids),
             "summary": len(summary), "model_scores": len(model_scores), "outputs": outputs}]
