"""rank_by: loss: score the snapshot's miners by their training losses on the dashboard-api.

The rows fetched are kept in snapshot/loss_source.jsonl, keyed by loss_source.key.json, and hashed into the snapshot,
so a rerun scores the same rows; delete loss_source.jsonl to fetch again. Any failure of the source falls back to
ranking by incentive, with its reason in the snapshot meta: never a stop.
"""
from __future__ import annotations

import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import requests

from connito.prompt_validator.stage_io import read_rows, write_rows
from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)

SOURCE_FILE = "loss_source.jsonl"
KEY_FILE = "loss_source.key.json"  # one JSON line: what the stored rows were fetched for
ROW_FIELDS = ("uid", "hotkey", "cycle_number", "phase_end_block", "validator_slot", "loss", "hf_repo")
# The settings that decide a ranking, in the rules hash; the pacing settings only decide whether it falls back.
LOSS_FIELDS = ("url", "window_training_cycles", "min_samples", "min_group", "min_scored", "validator_slots",
               "min_cycle")
# The meta's counts, null when the ranking fell back before it had them.
COUNTS = ("scored", "unscored", "failed_k", "source_rows")


class Malformed(ValueError):
    pass


def _int(value) -> bool:
    return type(value) is int  # bool is not a cycle, block or slot


def window_start(cfg, cutoff_cycle: int) -> int:
    """The window's first training cycle; it ends before the cutoff's, which is still being evaluated."""
    return max(cfg.min_cycle, cutoff_cycle - cfg.window_training_cycles)


def filter_rows(uid: int, payload, cfg, cutoff_cycle: int) -> list[dict]:
    """The uid's fresh, status-0, finite-loss rows of the given validator slots in the window; Malformed if the
    response cannot be read."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict) or not isinstance(data.get("hotkey"), str) or \
            not isinstance(data.get("submissions"), list):
        raise Malformed("no data.hotkey or data.submissions list")
    if data.get("miner_uid", uid) != uid:
        raise Malformed(f"data.miner_uid is {data['miner_uid']!r}")
    first = window_start(cfg, cutoff_cycle)
    rows = []
    for sub in data["submissions"]:
        if not isinstance(sub, dict) or not _int(sub.get("cycle_number")):
            raise Malformed("a submission without an integer cycle_number")
        if not first <= sub["cycle_number"] < cutoff_cycle:
            continue
        if not _int(sub.get("phase_end_block")) or not isinstance(sub.get("per_validator"), list):
            raise Malformed(f"cycle {sub['cycle_number']}: no integer phase_end_block or per_validator list")
        hf_repo = sub.get("hf_repo")
        for entry in sub["per_validator"]:
            if not isinstance(entry, dict) or not _int(entry.get("validator_slot")):
                raise Malformed(f"cycle {sub['cycle_number']}: a per_validator entry without an integer slot")
            loss, status = entry.get("loss"), entry.get("eval_status_code")
            if (entry["validator_slot"] in cfg.validator_slots and entry.get("loss_is_fresh") is True
                    and _int(status) and status == 0 and type(loss) in (int, float) and math.isfinite(loss)):
                rows.append({"uid": uid, "hotkey": data["hotkey"], "cycle_number": sub["cycle_number"],
                             "phase_end_block": sub["phase_end_block"], "validator_slot": entry["validator_slot"],
                             "loss": float(loss), "hf_repo": hf_repo if isinstance(hf_repo, str) else None})
    return rows


def truncated_at(payload) -> int | None:
    """The newest training cycle the source dropped from data.submissions, which holds only the newest entries."""
    cycle = (payload.get("meta") or {}).get("submissions_truncated_at_cycle")
    if cycle is not None and not _int(cycle):
        raise Malformed(f"meta.submissions_truncated_at_cycle is {cycle!r}")
    return cycle


class Fetched(NamedTuple):
    rows: list[dict] | None
    failure: str | None  # why there are no rows
    truncated: dict[int, int]  # uid -> truncated_at, for the uids whose submissions were cut


RETRYABLE = (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError,
             requests.exceptions.ContentDecodingError)


def _retry_after(response) -> float | None:
    try:
        return max(0.0, float(response.headers.get("Retry-After")))
    except (TypeError, ValueError):  # absent, or an HTTP date: wait as without one
        return None


def fetch_rows(cfg, uids: list[int], cutoff_cycle: int) -> Fetched:
    """Every uid's filtered rows, one spaced GET each, or the first failure: a uid failing after retries, a malformed
    response or the time budget spent. A 404 is a uid with no rows. No wait or request outlasts the budget."""
    start, last, rows, truncated, not_found = time.monotonic(), None, [], {}, []

    def remaining() -> float:
        return cfg.budget_sec - (time.monotonic() - start)

    def spent(uid: int) -> Fetched:
        return Fetched(None, f"time_budget: {cfg.budget_sec:.0f} s spent before uid {uid}", {})

    for uid in uids:
        first_429 = None
        for attempt in range(cfg.retries + 1):
            if last is not None:
                time.sleep(max(0.0, min(cfg.request_spacing_sec - (time.monotonic() - last), remaining())))
            if remaining() <= 0:
                return spent(uid)
            last, wait = time.monotonic(), cfg.backoff_sec * 2 ** attempt
            try:
                response = requests.get(cfg.url.format(uid=uid), timeout=min(cfg.timeout_sec, remaining()))
            except RETRYABLE as e:
                error = type(e).__name__
            except requests.RequestException as e:
                return Fetched(None, f"fetch_failed: uid {uid}: {type(e).__name__}", {})
            else:
                if response.status_code == 200:
                    try:
                        payload = response.json()
                        rows += filter_rows(uid, payload, cfg, cutoff_cycle)
                        if (cycle := truncated_at(payload)) is not None:
                            truncated[uid] = cycle
                    except ValueError as e:  # the body is not JSON, or not the shape the rules read
                        return Fetched(None, f"fetch_failed: uid {uid}: malformed: {e}"[:200], {})
                    break
                error = f"HTTP {response.status_code}"
                if response.status_code == 429:
                    # A rate limit is a minute's: without Retry-After, wait out a minute from the first 429.
                    first_429 = last if first_429 is None else first_429
                    after = _retry_after(response)
                    wait = after if after is not None else max(wait, first_429 + 60.0 - time.monotonic())
                elif response.status_code == 404:  # the source has no history for this uid
                    not_found.append(uid)
                    break
                elif response.status_code < 500:
                    return Fetched(None, f"fetch_failed: uid {uid}: {error}", {})
            if attempt < cfg.retries:
                if remaining() <= 0:
                    return spent(uid)
                time.sleep(min(wait, remaining()))
        else:
            return Fetched(None, f"fetch_failed: uid {uid}: {error} after {cfg.retries} retries", {})
    logger.info("loss source fetched", uids=len(uids), rows=len(rows), not_found=len(not_found),
                not_found_uids=not_found, elapsed_sec=round(time.monotonic() - start, 1))
    return Fetched(rows, None, truncated)


def counted_rows(rows: list[dict], registered: dict[int, int]) -> list[dict]:
    """The rows from each uid's registration block at the cutoff on: its holder then made them, a former one did not.

    The source labels every row with the uid's hotkey as of the fetch, so the rows' hotkey is not compared.
    """
    return [row for row in rows if row["uid"] in registered and row["phase_end_block"] >= registered[row["uid"]]]


def group_medians(rows: list[dict], min_group: int) -> dict[tuple[int, int], float]:
    """(cycle_number, validator_slot) -> its rows' median loss, for groups of at least min_group rows."""
    groups: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        groups[row["cycle_number"], row["validator_slot"]].append(row["loss"])
    return {key: statistics.median(losses) for key, losses in groups.items() if len(losses) >= min_group}


class Scores(NamedTuple):
    scores: dict[int, float]  # uid -> score, lower is better
    failed_k: list[int]  # uids with counted rows, but fewer than min_samples


def score(rows: list[dict], counted: list[dict], cfg) -> Scores:
    """Each uid's mean loss minus its (cycle, slot) group's median over its counted rows, both slots pooled.

    The groups take every row, counted or not: they describe the cycle's population. A row of a skipped group counts
    for nothing.
    """
    medians = group_medians(rows, cfg.min_group)
    normalised: dict[int, list[float]] = defaultdict(list)
    for row in counted:
        median = medians.get((row["cycle_number"], row["validator_slot"]))
        if median is not None:
            normalised[row["uid"]].append(row["loss"] - median)
    scores = {uid: math.fsum(losses) / len(losses) for uid, losses in normalised.items()
              if len(losses) >= cfg.min_samples}
    return Scores(scores, sorted(set(normalised) - set(scores)))


def source_key(cfg, cutoff_cycle: int, truncated: dict[int, int]) -> dict:
    """What the stored rows were fetched for, and the truncations the coverage check reads."""
    return {"training_cycle": cutoff_cycle, "settings": cfg.model_dump(mode="json", include=set(LOSS_FIELDS)),
            "truncated": {str(uid): cycle for uid, cycle in sorted(truncated.items())}}


def stored(out_dir: Path, cfg, cutoff_cycle: int) -> Fetched | None:
    """The rows an earlier run of this snapshot fetched for this training cycle and these settings, or None."""
    path = Path(out_dir) / SOURCE_FILE
    try:
        rows = read_rows(path)
        if not all(isinstance(row, dict) and tuple(row) == ROW_FIELDS for row in rows):
            raise ValueError("rows without the source fields")
        try:
            [key] = read_rows(Path(out_dir) / KEY_FILE)
            truncated = {int(uid): cycle for uid, cycle in key["truncated"].items()}
        except FileNotFoundError:
            raise ValueError(f"no {KEY_FILE}") from None
        except (TypeError, KeyError, AttributeError) as e:
            raise ValueError(f"{KEY_FILE} unreadable: {type(e).__name__}") from None
        if key != source_key(cfg, cutoff_cycle, truncated):
            raise ValueError(f"fetched for training cycle {key.get('training_cycle')} or other settings")
        return Fetched(rows, None, truncated)
    except FileNotFoundError:
        return None
    except ValueError as e:
        logger.warning("loss source not reused, fetched again", path=str(path),
                       error=f"{type(e).__name__}: {e}"[:200])
        return None


class LossRank(NamedTuple):
    scores: dict[int, float] | None  # None: rank by incentive
    rows: list[dict] | None  # the source rows, hashed into the snapshot
    meta: dict
    source: str | None = None  # "fetched" or "reused"; not hashed, as reusing the rows retakes the same snapshot


def rank(cfg, out_dir: Path, cutoff_cycle: int, uids: list[int], registered: dict[int, int],
         ranked: set[int]) -> LossRank:
    """The scores of the uids in `ranked` from the stored rows, else from a fetch for every uid in `uids`."""
    source = stored(out_dir, cfg, cutoff_cycle)
    if source is not None:
        logger.info("loss source reused", rows=len(source.rows))
        how = "reused"
    else:
        source, how = fetch_rows(cfg, sorted(uids), cutoff_cycle), "fetched"
        if source.rows is None:
            return fallback(source.failure)
        # Before the rest of the snapshot, so a rerun after a later failure reuses it; the key last, as it vouches.
        write_rows(Path(out_dir) / SOURCE_FILE, source.rows)
        write_rows(Path(out_dir) / KEY_FILE, [source_key(cfg, cutoff_cycle, source.truncated)])
    rows = source.rows
    cut = sorted((uid, cycle) for uid, cycle in source.truncated.items()
                 if uid in ranked and cycle >= window_start(cfg, cutoff_cycle))
    if cut:  # the source dropped that cycle's submissions or later ones: the window is not all there
        uid, cycle = cut[0]
        return fallback(f"window_not_covered: uid {uid} truncated at cycle {cycle}", rows,
                        {"source_rows": len(rows)})._replace(source=how)
    scored = score(rows, counted_rows(rows, registered), cfg)
    scores = {uid: s for uid, s in scored.scores.items() if uid in ranked}
    counts = {"scored": len(scores), "unscored": len(ranked) - len(scores),
              "failed_k": len(set(scored.failed_k) & ranked), "source_rows": len(rows)}
    if len(scores) < cfg.min_scored:
        return fallback(f"too_few_scored: {len(scores)} < {cfg.min_scored}", rows, counts)._replace(source=how)
    return LossRank(scores, rows, {"fallback": False, "fallback_reason": None, **counts}, how)


def fallback(reason: str, rows: list[dict] | None = None, counts: dict | None = None) -> LossRank:
    logger.warning("ranked by incentive instead of loss", reason=reason)
    meta = {"fallback": True, "fallback_reason": reason, **dict.fromkeys(COUNTS), **(counts or {})}
    return LossRank(None, rows, meta)
