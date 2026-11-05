"""STUB: the judge for miner-submitted prompt sets is NOT part of this release.

This file stands in for the real connito/prompt_validator/judge.py, which is not published. Only two things are here:

- `selected_prompts`, a verbatim copy of the real function, because the answers, scores and reward stages import it.
  With `judge_miner_prompts` off, which is the default and how the validator runs, it returns no prompts on its
  first line. The export checks that this copy is identical to the real one.
- `run_judge`, which raises. The real judge and its rules are not in this release.

Nothing in this file decides how a model is scored. The config refuses `judge_miner_prompts: true` without the
judge's rules, so the stage never runs here.
"""
from __future__ import annotations

from connito.prompt_validator.stage_io import read_rows, stage_path

NOT_IN_THIS_RELEASE = ("the judge for miner-submitted prompt sets is not part of this release: "
                       "connito/prompt_validator/judge.py is a stub (see its docstring and the README)")


def selected_prompts(pv, scoring_cycle_id: int) -> list[dict]:
    """The selected miner prompts in prompts.jsonl's shape (no request, as the pool's); none with the switch off."""
    if not pv.judge_miner_prompts:
        return []
    marker = stage_path(pv.runs_dir, scoring_cycle_id, "judge")
    if not marker.exists():
        raise RuntimeError(f"miner prompts of scoring cycle {scoring_cycle_id} not judged yet: run the judge stage")
    snapshot_dir = stage_path(pv.runs_dir, scoring_cycle_id, "snapshot").parent
    miner = {p["prompt_id"]: p for p in read_rows(snapshot_dir / "miner_prompts.jsonl")}
    return [{"prompt_id": p["prompt_id"], "request_id": p["customer_task_id"], "prompt": p["text"], "request": None,
             "customer_task_id": p["customer_task_id"], "content_hash": p["content_hash"]}
            for p in (miner[s["prompt_id"]] for s in read_rows(marker.parent / "selected.jsonl"))]


def run_judge(config, scoring_cycle_id: int) -> list[dict]:
    """Not in this release: raises, so a run that reaches it stops rather than continuing without the judge."""
    raise NotImplementedError(NOT_IN_THIS_RELEASE)
