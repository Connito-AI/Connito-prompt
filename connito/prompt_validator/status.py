"""What each stage did, for its summary line in the log, and `runs/status.json`: what the loop is doing now, for a
health check that reads one file."""
from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from connito.prompt_validator.stage_io import last_published, published_cycles, read_rows, write_rows
from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)

_path: Path | None = None  # set only while --loop runs: a manual run beside it must not overwrite the loop's file
_state: dict = {}


@contextmanager
def loop_status(runs_dir: Path) -> Iterator[None]:
    """Keep runs/status.json while the loop runs, and leave it saying "stopped" however the loop ends."""
    global _path
    _path = Path(runs_dir) / "status.json"
    _state.clear()
    _state["process_started_at"] = _now()
    try:
        yield
    finally:
        # However it ends, the file says what is published now and no cycle as under way.
        set_status(state="stopped", cycle=None, stage=None, progress=None, eta_min=None, **ledger_status(runs_dir))
        _path = None


def ledger_status(runs_dir: Path) -> dict:
    """The status's published count and last_published from the ledger; never raises."""
    try:
        return {"published": len(published_cycles(runs_dir)), "last_published": last_published(runs_dir)}
    except Exception as e:
        logger.warning("ledger not read for the status", error=type(e).__name__)
        return {}


def set_status(**fields) -> None:
    """Merge fields into the loop's status and rewrite the file atomically, mode 600; never raises."""
    if _path is None:
        return
    now = _now()
    if any(key in fields and fields[key] != _state.get(key) for key in ("state", "stage", "cycle")):
        _state["started_at"] = now
    if "last_error" in fields:
        _state["last_error_at"] = now
    _state.update(fields, updated_at=now, pid=os.getpid())
    try:
        write_rows(_path, [_state])  # one JSON line, which json.load reads as the object
    except Exception as e:
        logger.warning("status not written", path=str(_path), error=type(e).__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stage_counts(stage: str, rows: list[dict], path: Path) -> dict:
    """The key counts of a finished stage's rows; never raises, as the stage is done whatever they say."""
    try:
        return _counts(stage, rows, path)
    except Exception as e:
        return {"counts_error": type(e).__name__}


def _counts(stage: str, rows: list[dict], path: Path) -> dict:
    if stage == "cutoff":
        return {"commit_2_block": rows[0]["commit_2_block"]}
    if stage == "fetch":
        return {"fetched": sum(r["reason"] is None for r in rows),
                "reasons": dict(Counter(r["reason"] for r in rows if r["reason"] is not None))}
    if stage == "answers":
        return {"answers": len(rows), "models": len({r["model_hash"] for r in rows}),
                "cut_at_max": sum(r["finish_reason"] == "length" for r in rows)}
    if stage == "scores":  # every call of this cycle, reruns included: what the spend cap counts
        calls_log = path.parent / "jev_calls.jsonl"
        calls = read_rows(calls_log) if calls_log.exists() else []
        return {"jev_calls": len(calls), "spent_usd": round(sum(c["cost"] for c in calls), 4)}
    if stage == "reward":
        return {"reward_rows": len(rows), "weighted": sum(r["weight"] > 0 for r in rows)}
    if stage == "weights":
        row = rows[0]
        if "error" in row:  # the type alone, as status.json keeps it; the message is in the log and weights.jsonl
            return {"error": row["error"].split(":", 1)[0]}
        return {"uid": row["uid"], "update_age_blocks": row["last_update_age_blocks"],
                "l1_to_chain": round(row["l1_to_chain"], 4), "owner": round(row["combined"].get(0, 0.0), 4),
                "dropped_hotkeys": len(row["dropped_hotkeys"]), "guards": row["guards"]}
    if stage == "report":
        return {"report_status": rows[0].get("status"), "posted": rows[0].get("published_before") is None}
    return {}
