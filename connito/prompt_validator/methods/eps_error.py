"""eps_error: the answer's FORECAST line against the prompt's actual EPS change, from a validator-side file pinned by
sha256, ranked among the cycle's answers to the prompt. Its format check is here, not in answer_format, so the rules
hash moves only for a task that lists it.

Per prompt, the valid forecasts are ranked by their absolute error, closest first (rank 1), one place per model_hash
so copies take no place; ties share the best rank they span. An answer scores max(10 - rank, 0) points
(ranks 1 to 9 earn 9 down to 1), as expected_score points / 9. An answer with no valid forecast scores 0 and takes no
rank.

The answers file is JSONL, one object per prompt of the task:
    {"content_hash": "<sha256 hex of the prompt text>", "prompt_id": "<any string>", "eps_pct_change": -41.22}
- content_hash: the key; the prompt's content_hash as the pool serves it, so a prompt is matched by its text.
- prompt_id: the answer's own id, recorded on each score row as answer_id; it need not equal the pool's prompt_id.
- eps_pct_change: the actual change in percent (-41.22 is -41.22 %, not -0.4122), a finite JSON number.
Other fields are allowed and ignored; blank lines are skipped. Two rows for one content_hash, a malformed line, a
missing file, another sha256 than the pinned one, or a prompt of the task with no row each stop the loop.

answers_source picks where the rows come from. `file` (the default): the pinned file above. `prompt_record`: the
snapshot stage takes each prompt's `answer.eps_pct_change` from prompt-record's prompt list (it serves a drawn pool
prompt's answer with it, after the cutoff only; the answer is not in the list's snapshot_hash, but the pool never lets
it change) and writes them beside the snapshot as `answers-<task>.jsonl` in the file's format, its sha256 in the meta;
the scores stage reads that file and checks that sha256, so a rerun scores the answers the snapshot took."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from connito.prompt_validator.jev import MAX_LEVEL
from connito.prompt_validator.methods import Context, Item, MethodStop, Scored

FORECAST = re.compile(r"^FORECAST: ([+-]\d{1,4}\.\d)$")


class AnswersFileError(MethodStop):
    pass


class EpsErrorSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers_source: Literal["file", "prompt_record"] = "file"
    answers_path: Path | None = None  # file only: mounted read-only; not hashed
    answers_sha256: str | None = None  # file only

    @field_validator("answers_path")
    @classmethod
    def _absolute(cls, path: Path | None) -> Path | None:
        if path is not None and not path.is_absolute():
            raise ValueError(f"answers_path must be absolute, got {path}")
        return path

    @field_validator("answers_sha256")
    @classmethod
    def _hex(cls, sha: str | None) -> str | None:
        if sha is not None and len(sha) != 64 or set(sha) - set("0123456789abcdef"):
            raise ValueError(f"answers_sha256 must be lowercase hex sha256, got {sha!r}")
        return sha

    @model_validator(mode="after")
    def _source(self) -> EpsErrorSettings:
        pinned = (self.answers_path, self.answers_sha256)
        if self.answers_source == "file" and None in pinned:
            raise ValueError("answers_source file needs answers_path and answers_sha256")
        if self.answers_source == "prompt_record" and pinned != (None, None):
            raise ValueError("answers_source prompt_record takes no answers_path or answers_sha256")
        return self


def answers_file_name(task: str) -> str:
    """The snapshot's copy of a task's answers read from prompt-record."""
    return f"answers-{task}.jsonl"


def load_answers(settings: EpsErrorSettings) -> dict[str, dict]:
    """content_hash -> {answer_id, eps_pct_change}, from the pinned file."""
    return read_answers(settings.answers_path, settings.answers_sha256)


def read_answers(path: Path, sha256: str) -> dict[str, dict]:
    """content_hash -> {answer_id, eps_pct_change}, from an answers file that must have this sha256."""
    if not path.is_file():
        raise AnswersFileError(f"answers file {path} is missing: mount it, or change the task's scoring method")
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise AnswersFileError(f"answers file {path} has sha256 {got[:12]}, not the pinned {sha256[:12]}")
    try:
        lines = data.decode().splitlines()
    except UnicodeDecodeError as e:
        raise AnswersFileError(f"answers file {path} is not UTF-8 (byte {e.start})") from None
    answers: dict[str, dict] = {}
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        row = _answer_row(path, number, line)
        if row["content_hash"] in answers:
            raise AnswersFileError(f"answers file {path} has two answers for one prompt")
        answers[row["content_hash"]] = {"answer_id": row["prompt_id"], "eps_pct_change": float(row["eps_pct_change"])}
    return answers


def _answer_row(path: Path, number: int, line: str) -> dict:
    """One line of the answers file, or AnswersFileError: a malformed line stops the loop like a changed file, since a
    rerun would read the same bytes. The error names the line, never its text."""
    try:
        row = json.loads(line)
    except (ValueError, RecursionError) as e:
        raise AnswersFileError(f"answers file {path} line {number} is not JSON ({type(e).__name__})") from None
    if not isinstance(row, dict):
        raise AnswersFileError(f"answers file {path} line {number} is not a JSON object")
    for field, kind in (("content_hash", str), ("prompt_id", str), ("eps_pct_change", (int, float))):
        if not isinstance(row.get(field), kind) or isinstance(row.get(field), bool):
            raise AnswersFileError(f"answers file {path} line {number} has no {field} of the right type")
    if not math.isfinite(row["eps_pct_change"]):
        raise AnswersFileError(f"answers file {path} line {number} has no eps_pct_change of the right type")
    return row


def parse_forecast(text: str) -> float | None:
    """The forecast in percent, or None: the last non-empty line must be the only FORECAST line, exactly in form."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or text.count("FORECAST:") != 1:
        return None
    m = FORECAST.match(lines[-1])
    return float(m.group(1)) if m else None


POINTS_FROM = 10  # an answer at rank r earns max(POINTS_FROM - r, 0) points: ranks 1 to 9 earn 9 down to 1
MAX_POINTS = POINTS_FROM - 1


def _places(errors: dict) -> dict:
    """key -> (rank, points) for the keys' absolute errors, closest first from rank 1; equal errors (to 4 decimals)
    share the best rank they span."""
    rounded = {key: round(error, 4) for key, error in errors.items()}
    places = {}
    for error in sorted(set(rounded.values())):
        rank = 1 + sum(e < error for e in rounded.values())
        for key in (key for key, e in rounded.items() if e == error):
            places[key] = (rank, max(POINTS_FROM - rank, 0))
    return places


class EpsError:
    name: ClassVar[str] = "eps_error"
    RULES: ClassVar[str] = (
        "a prompt's answer is the pinned file's row with the prompt's content_hash, and a prompt with none stops the "
        "scores stage; the answer's last non-empty line must match ^FORECAST: [+-]\\d{1,4}\\.\\d$ and be its only "
        "'FORECAST:', else it scores 0 and takes no rank (a cut-off answer has no such line); per prompt, the valid "
        "forecasts f are ranked by |f - a| against the actual change a, both in percent, closest first from rank 1, "
        "one place per model_hash (copies share the original's place); equal errors to 4 decimals share the best "
        "rank they span; an answer earns max(10 - rank, 0) points and scores 5 x points / 9; no Jev call"
    )
    #: answers_source prompt_record: the same rules, but the answers are prompt-record's for the cycle.
    RULES_PROMPT_RECORD: ClassVar[str] = RULES.replace(
        "the pinned file's row with the prompt's content_hash",
        "the answer prompt-record serves with the drawn prompt after the cutoff (its eps_pct_change, kept with the "
        "snapshot)")
    DETAILS: ClassVar[tuple[str, ...]] = ("forecast_pct", "abs_error_pp", "error_rank", "rank_points", "answer_id")
    Settings: ClassVar[type[BaseModel]] = EpsErrorSettings
    SETTINGS_NOT_HASHED: ClassVar[frozenset[str]] = frozenset({"answers_path"})

    def rules(self, settings: EpsErrorSettings) -> str:
        return self.RULES if settings.answers_source == "file" else self.RULES_PROMPT_RECORD

    def hashed_settings(self, settings: EpsErrorSettings) -> dict:
        """File mode hashes as before answers_source existed (the pinned sha256 only); prompt_record mode, the source
        alone: its answers change each cycle and are pinned by the snapshot instead."""
        if settings.answers_source == "file":
            return {"answers_sha256": settings.answers_sha256}
        return {"answers_source": settings.answers_source}

    def check(self, settings: EpsErrorSettings) -> None:
        if settings.answers_source == "file":
            load_answers(settings)

    def _answers(self, ctx: Context) -> tuple[dict[str, dict], Path]:
        settings: EpsErrorSettings = ctx.settings
        if settings.answers_source == "file":
            return load_answers(settings), settings.answers_path
        pinned = (ctx.meta or {}).get("prompt_answers", {}).get(ctx.task)
        if ctx.snapshot_dir is None or pinned is None:
            raise AnswersFileError(f"the snapshot took no prompt-record answers for task {ctx.task}: rerun "
                                   "--stage snapshot with answers_source prompt_record")
        path = ctx.snapshot_dir / answers_file_name(ctx.task)
        return read_answers(path, pinned["file_sha256"]), path

    def score(self, items: list[Item], ctx: Context) -> list[Scored]:
        answers, path = self._answers(ctx)
        missing = sorted({i.prompt["prompt_id"] for i in items if i.prompt.get("content_hash") not in answers})
        if missing:  # the pool and the answers disagree: never score these prompts some other way
            raise AnswersFileError(f"answers file {path} has no answer for prompts {missing}")
        forecasts = [parse_forecast(i.answer) for i in items]
        # One place per model_hash, its first valid answer's: copies share it. With no model_hash, each its own.
        keys = [(i.prompt["content_hash"], i.model_hash if i.model_hash is not None else n)
                for n, i in enumerate(items)]
        errors: dict[str, dict] = {}
        for (content_hash, key), forecast in zip(keys, forecasts, strict=True):
            if forecast is not None:
                errors.setdefault(content_hash, {}).setdefault(
                    key, abs(forecast - answers[content_hash]["eps_pct_change"]))
        places = {content_hash: _places(by_key) for content_hash, by_key in errors.items()}
        out = []
        for (content_hash, key), forecast in zip(keys, forecasts, strict=True):
            truth = answers[content_hash]
            error = None if forecast is None else abs(forecast - truth["eps_pct_change"])
            rank, points = (None, None) if forecast is None else places[content_hash][key]
            score = 0.0 if points is None else MAX_LEVEL * points / MAX_POINTS
            out.append(Scored(score=score, method=self.name, details={
                "forecast_pct": forecast, "abs_error_pp": None if error is None else round(error, 2),
                "error_rank": rank, "rank_points": points, "answer_id": truth["answer_id"]}))
        return out
