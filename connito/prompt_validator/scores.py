"""Stage 5: Jev scores every answer that passes its task's format check against its prompt and the customer request
behind it; an answer that fails scores 0 without Jev (answer_format.py).

It reads the answers stage's rows, `runs/<cycle>/answers/answers.jsonl`: one per (miner, prompt), with `uid`,
`hotkey`, `model_hash`, `prompt_id` and the answer's `text`.

Answers to one prompt share Jev calls, scores are cached per answer and spend is capped per scoring cycle (see
jev.py). A failed call stops the stage with no marker; a rerun takes every score made so far from the cache.
The cache is shared across cycles, so a row's jev_call_id may be logged in an earlier cycle's calls log.
"""
from __future__ import annotations

from connito.prompt_validator.answer_format import format_ok
from connito.prompt_validator.jev import cycle_client
from connito.prompt_validator.judge import selected_prompts
from connito.prompt_validator.snapshot import snapshot_meta
from connito.prompt_validator.stage_io import read_rows, stage_path
from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)

# score and expected_score are on 0 to 1 (see jev.py). rank is the answer's place among this cycle's models' answers to
# the same prompt by expected_score, 0 for the best: one answer per model_hash, so copies take no place. Equal scores
# share a rank, and the next rank skips (0, 0, 2). jev_call_id is null for an answer that failed its format check.
ROW_FIELDS = ("uid", "hotkey", "model_hash", "prompt_id", "score", "expected_score", "rank", "jev_call_id")
FORMAT_FAILED = {"score": 0.0, "expected_score": 0.0, "jev_call_id": None}


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
    tasks = [prompts[a["prompt_id"]].get("customer_task_id") for a in answers]
    # An answer that fails its task's format check scores 0 and is never sent (answer_format.FORMAT_RULES).
    sent = [i for i, (a, task) in enumerate(zip(answers, tasks, strict=True))
            if format_ok(task, a["text"], prompts[a["prompt_id"]]["prompt"])]
    # Each answer on its own prompt's task's levels: a cycle can hold prompts of two tasks (jev.CRITERIA_BY_TASK).
    scored = cycle_client(pv, scoring_cycle_id).score_many(
        [(prompts[answers[i]["prompt_id"]]["request"], prompts[answers[i]["prompt_id"]]["prompt"], answers[i]["text"])
         for i in sent],
        tasks=[tasks[i] for i in sent],
    )
    results = [FORMAT_FAILED] * len(answers)
    for i, r in zip(sent, scored, strict=True):
        results[i] = r
    logger.info("format checked", answers=len(answers), failed=len(answers) - len(sent))
    # Without `cached`, so the rows depend only on the answers and the Jev pin, not on what a rerun found cached.
    rows = [a | r for a, r in zip(answers, results, strict=True)]
    by_prompt: dict[str, dict[str, float]] = {}
    for row in rows:
        by_prompt.setdefault(row["prompt_id"], {})[row["model_hash"]] = row["expected_score"]
    for row in rows:
        row["rank"] = sum(other > row["expected_score"] for other in by_prompt[row["prompt_id"]].values())
    return [{field: row[field] for field in ROW_FIELDS} for row in rows]
