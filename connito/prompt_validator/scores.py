"""Stage 5: Jev scores every answer that passes its task's format check against its prompt and the customer request
behind it; an answer that fails scores 0 without Jev (answer_format.py).

It reads the answers stage's rows, `runs/<cycle>/answers/answers.jsonl`: one per (miner, prompt), with `uid`,
`hotkey`, `model_hash`, `prompt_id` and the answer's `text`.

Answers to one prompt share Jev calls, scores are cached per answer and spend is capped per scoring cycle (see
jev.py). A failed call stops the stage with no marker; a rerun takes every score made so far from the cache.
The cache is shared across cycles, so a row's jev_call_id may be logged in an earlier cycle's calls log.
"""
from __future__ import annotations

from connito.prompt_validator.answer_format import LENGTH_FIELDS, apply_lesson_length, format_ok
from connito.prompt_validator.config import ScoringMethodCfg
from connito.prompt_validator.jev import MAX_LEVEL, cycle_client
from connito.prompt_validator.judge import selected_prompts
from connito.prompt_validator.methods import REGISTRY, Context, Item, check_all
from connito.prompt_validator.snapshot import snapshot_meta
from connito.prompt_validator.stage_io import read_rows, stage_path
from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)

# score and expected_score are on 0 to 1 (see jev.py). rank is the answer's place among this cycle's models' answers to
# the same prompt by expected_score, 0 for the best: one answer per model_hash, so copies take no place. Equal scores
# share a rank, and the next rank skips (0, 0, 2). jev_call_id is null for an answer that failed its format check.
# expected_score is the task's judging method's score over 5; score its likeliest level over 5, where it has one.
# With lesson_length on, both are times the length multiplier and LENGTH_FIELDS follow. With scoring_methods listing any
# task, method_fields follow: the method that scored the answer, then the listed methods' DETAILS, null where unused.
ROW_FIELDS = ("uid", "hotkey", "model_hash", "prompt_id", "score", "expected_score", "rank", "jev_call_id")
FORMAT_FAILED = {"score": 0.0, "expected_score": 0.0, "jev_call_id": None, "method": None}


def method_of(pv, task: str | None) -> ScoringMethodCfg:
    return pv.scoring_methods.get(task) or ScoringMethodCfg()


def method_fields(pv) -> tuple[str, ...]:
    if not pv.scoring_methods:
        return ()
    names = dict.fromkeys(m.method for m in pv.scoring_methods.values())
    return ("method", *dict.fromkeys(f for name in names for f in REGISTRY[name].DETAILS))


def run_scores(config, scoring_cycle_id: int) -> list[dict]:
    pv = config.prompt_validator
    snapshot_meta(config, scoring_cycle_id)  # the Jev pin is a score-deciding setting
    snapshot_dir = stage_path(pv.runs_dir, scoring_cycle_id, "snapshot").parent
    prompts = {p["prompt_id"]: p for p in [*read_rows(snapshot_dir / "prompts.jsonl"),
                                           *selected_prompts(pv, scoring_cycle_id)]}
    answers_path = stage_path(pv.runs_dir, scoring_cycle_id, "answers")
    if not answers_path.exists():
        raise RuntimeError(f"no answers to score: {answers_path} is missing")
    answers = read_rows(answers_path)
    check_all(pv)  # before any Jev call
    tasks = [prompts[a["prompt_id"]].get("customer_task_id") for a in answers]
    # An answer that fails its task's format check scores 0 and is never sent (answer_format.FORMAT_RULES).
    sent = [i for i, (a, task) in enumerate(zip(answers, tasks, strict=True))
            if format_ok(task, a["text"], prompts[a["prompt_id"]]["prompt"])]
    logger.info("format checked", answers=len(answers), failed=len(answers) - len(sent))
    client = cycle_client(pv, scoring_cycle_id)
    by_task: dict[str | None, list[int]] = {}
    for i in sent:
        by_task.setdefault(tasks[i], []).append(i)
    detail_fields = method_fields(pv)[1:]
    results = [FORMAT_FAILED | dict.fromkeys(detail_fields)] * len(answers)
    for task, indices in by_task.items():
        cfg = method_of(pv, task)
        items = [Item(prompts[answers[i]["prompt_id"]], answers[i]["text"]) for i in indices]
        for i, s in zip(indices, REGISTRY[cfg.method].score(items, Context(task, cfg.parsed(), client)), strict=True):
            level = s.score if s.level is None else s.level
            results[i] = {"score": round(level / MAX_LEVEL, 4), "expected_score": round(s.score / MAX_LEVEL, 4),
                          "jev_call_id": s.call_id, "method": s.method,
                          **{f: s.details.get(f) for f in detail_fields}}
    # Without `cached`, so the rows depend only on the answers and the Jev pin, not on what a rerun found cached.
    rows = [a | r for a, r in zip(answers, results, strict=True)]
    fields = ROW_FIELDS
    if pv.lesson_length.enabled:  # after the method, before ranking: rank and reward follow the multiplied scores
        checked = set(sent)
        rows = [apply_lesson_length(row, task, prompts[row["prompt_id"]]["prompt"], i in checked, pv.lesson_length)
                for i, (row, task) in enumerate(zip(rows, tasks, strict=True))]
        fields = (*fields, *LENGTH_FIELDS)
        logger.info("lesson length applied", answers=len(rows),
                    penalised=sum((r["length_multiplier"] or 1.0) < 1 for r in rows),
                    zeroed=sum(r["length_multiplier"] == 0 for r in rows))
    by_prompt: dict[str, dict[str, float]] = {}
    for row in rows:
        by_prompt.setdefault(row["prompt_id"], {})[row["model_hash"]] = row["expected_score"]
    for row in rows:
        row["rank"] = sum(other > row["expected_score"] for other in by_prompt[row["prompt_id"]].values())
    fields = (*fields, *method_fields(pv))
    return [{field: row[field] for field in fields} for row in rows]
