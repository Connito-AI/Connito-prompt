"""JSONL files that carry each stage's output to the next, under `runs/<scoring cycle id>/<stage>/`, and the locks
that keep a second process off a cycle or a runs_dir's loop."""
from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)

# Cycles this runs_dir has published, append-only: outside the stage dirs, so no stage rerun removes a row. Keyed by
# cycle_id alone: a runs_dir serves one validator hotkey on one dataless-api.
LEDGER = "published.jsonl"


class LockHeld(RuntimeError):
    pass


def stage_path(runs_dir: Path, scoring_cycle_id: int, stage: str) -> Path:
    return Path(runs_dir) / str(scoring_cycle_id) / stage / f"{stage}.jsonl"


def under_root(path: Path, root: Path) -> Path:
    """`path` with every symlink and `..` resolved, if that is `root` or under it; else raise."""
    real, top = Path(os.path.realpath(path)), Path(os.path.realpath(root))
    if real != top and top not in real.parents:
        raise ValueError(f"{path} resolves to {real}, outside prompt_validator.write_root {top}: the validator writes "
                         "and deletes only under it")
    return real


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write rows as JSONL atomically, so an existing file always means a finished stage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    # Owner-only: stage files carry the private internal prompts and text derived from them.
    with open(tmp, "w", encoding="utf-8", opener=lambda p, flags: os.open(p, flags, 0o600)) as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_published(runs_dir: Path, row: dict) -> None:
    """Add one row to the ledger in a single appending write: earlier rows are never rewritten."""
    # Reports of two cycles may append at once: safe, as each row is one O_APPEND write on a local disk.
    path = Path(runs_dir) / LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        size = os.fstat(fd).st_size
        # After a line a crash cut short, on a line of its own: the torn one stays unreadable alone.
        line = ("\n" if size and os.pread(fd, 1, size - 1) != b"\n" else "") + json.dumps(row) + "\n"
        data = line.encode()
        if os.write(fd, data) != len(data):
            raise OSError(f"short write to {path}: the ledger row was not written whole")
        os.fsync(fd)
    finally:
        os.close(fd)


def ledger_rows(runs_dir: Path) -> list[dict]:
    """The ledger's rows, in the order they were written."""
    try:
        # A damaged byte spoils its own line only, which the parse below then skips.
        lines = (Path(runs_dir) / LEDGER).read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return []
    rows = []
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except ValueError:
            row = None
        if isinstance(row, dict) and type(row.get("cycle_id")) is int:  # not a bool, nor a cycle id as text
            rows.append(row)
        elif line.strip():  # cut short by a crash: report finds that publish again on the server
            logger.warning("ledger line unreadable, skipped", path=str(Path(runs_dir) / LEDGER), line=number)
    return rows


def published_cycles(runs_dir: Path) -> set[int]:
    return {row["cycle_id"] for row in ledger_rows(runs_dir)}


def last_published(runs_dir: Path) -> int | None:
    """The cycle published most recently: by published_at, and by row order where it ties or can't be read."""
    last, latest = None, datetime.min.replace(tzinfo=timezone.utc)
    when = latest
    for row in ledger_rows(runs_dir):
        when = _time(row.get("published_at")) or when  # an unreadable time keeps the row's place in the order
        if when >= latest:
            last, latest = row["cycle_id"], when
    return last


def _time(value) -> datetime | None:
    try:
        # The server's time ends in Z, which fromisoformat reads only from Python 3.11.
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


@contextmanager
def run_lock(path: Path, what: str) -> Iterator[None]:
    """Hold an exclusive flock on `path`, or raise LockHeld at once. The kernel frees it however the holder exits, so
    a lock file left after a kill -9 holds nothing. Never unlinked: a new file under the same name would be a second
    lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        os.close(fd)
        if not isinstance(e, BlockingIOError):  # no lock to be had here at all, e.g. ENOLCK
            raise
        holder = path.read_text().strip() or "?"
        raise LockHeld(f"{what} is run by another process (pid {holder}, lock {path})") from None
    try:
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        yield
    finally:
        os.close(fd)
