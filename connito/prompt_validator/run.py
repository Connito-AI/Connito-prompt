"""Prompt validator entrypoint: resolve the scoring cycle, then run its stages in order."""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path

from connito.prompt_validator.gpu_env import require_gpu_env

if __name__ == "__main__":
    require_gpu_env()  # before the imports below: fla starts CUDA as it is imported

from dotenv import dotenv_values

from connito.prompt_validator.answers import run_answers
from connito.prompt_validator.config import PromptValidatorConfig, caches_outside
from connito.prompt_validator.cutoff import check_phase_clock, report_before_block, run_cutoff, scoring_cycle_of
from connito.prompt_validator.dataless import TesterRefused, validator_client
from connito.prompt_validator.fetch import prune_shards, run_fetch
from connito.prompt_validator.jev import JevVersionError
from connito.prompt_validator.judge import run_judge
from connito.prompt_validator.report import run_report
from connito.prompt_validator.reward import run_reward
from connito.prompt_validator.scores import run_scores
from connito.prompt_validator.snapshot import run_snapshot
from connito.prompt_validator.stage_io import LockHeld, published_cycles, run_lock, stage_path, write_rows
from connito.prompt_validator.status import ledger_status, loop_status, set_status, stage_counts
from connito.prompt_validator.weights import run_weights
from connito.shared.app_logging import configure_logging, structlog
from connito.shared.cycle import BITTENSOR_BLOCK_TIME_SECONDS, get_phase_from_api

logger = structlog.get_logger(__name__)

STAGES = [
    ("cutoff", run_cutoff), ("snapshot", run_snapshot), ("judge", run_judge), ("fetch", run_fetch),
    ("answers", run_answers), ("scores", run_scores), ("reward", run_reward), ("report", run_report),
    ("weights", run_weights),
]
# Stages a plain run skips, to run only when named with --stage. report publishes to the dataless-api, which is
# final, so it is always asked for by name; weights is a dry run, only with weights.mode dry_run.
ON_DEMAND = {"report", "weights"}
RETRY_CAP_MIN = 30  # a failed cycle is retried after 1, 2, 4 … minutes, at most this


def run_stages(config: PromptValidatorConfig, cycle_id: int, stage: str | None = None) -> None:
    """Run every unfinished stage in order, or rerun only `stage`."""
    for i, (name, run_stage) in enumerate(STAGES):
        if stage is not None and name != stage:
            continue
        if stage is None and name in ON_DEMAND:
            continue
        if name == "judge" and not config.prompt_validator.judge_miner_prompts:
            if stage == "judge":
                raise RuntimeError("--stage judge needs prompt_validator.judge_miner_prompts: true")
            continue  # off, a cycle's files are the ones from before miner prompts were judged
        path = stage_path(config.prompt_validator.runs_dir, cycle_id, name)
        if stage is None and path.exists():
            logger.info("stage already done", stage=name, path=str(path))
            continue
        # A run that fails must not leave its old marker saying "done", nor a later stage's built on the old output.
        for later, _ in STAGES[i:]:
            later_path = stage_path(config.prompt_validator.runs_dir, cycle_id, later)
            if later != name and later_path.exists():
                if later == "report":  # written only once the cycle is published, which is final
                    logger.warning("removed the report of a published cycle: the files here no longer match what "
                                   "the dataless-api holds", scoring_cycle_id=cycle_id)
                else:
                    logger.warning("removed a later stage built on the old output; rerun it", stage=later,
                                   rerun="--stage weights" if later == "weights" else "a plain run")
            later_path.unlink(missing_ok=True)
        set_status(state="waiting_for_cutoff" if name == "cutoff" else "running", cycle=cycle_id, stage=name,
                   progress=None, eta_min=None)
        started = time.monotonic()
        rows = run_stage(config, cycle_id)
        write_rows(path, rows)
        summary = {"stage": name, "scoring_cycle_id": cycle_id, "duration_s": round(time.monotonic() - started),
                   "rows": len(rows), **stage_counts(name, rows, path)}
        logger.info("stage done", **summary, path=str(path))
        set_status(last_stage=summary)


def published(pv, cycle_id: int) -> bool:
    """In the ledger, which no stage rerun touches, or reported before there was one."""
    return cycle_id in published_cycles(pv.runs_dir) or stage_path(pv.runs_dir, cycle_id, "report").exists()


def next_cycle(config) -> int:
    """The closed test cycle not published yet, or the first unpublished scoring cycle from the one before the open
    one."""
    pv = config.prompt_validator
    if pv.prompt_source == "dataless" and pv.dataless.test_mode:
        return validator_client(config).next_closed_test_cycle(lambda c: published(pv, c))
    phase = check_phase_clock(config, get_phase_from_api(config))
    cycle_id = scoring_cycle_of(phase.cycle_index, pv.scoring_cycle_k)
    # The one before is listable until the open one closes: run it while there is time left to report it. One past
    # reward has only its report left, which takes seconds, so it keeps the whole window.
    if not published(pv, cycle_id - 1):
        left_h = (report_before_block(config, cycle_id - 1) - phase.block) * BITTENSOR_BLOCK_TIME_SECONDS / 3600
        scored = stage_path(pv.runs_dir, cycle_id - 1, "reward").exists()
        if left_h > 0 if scored else left_h >= pv.catch_up_min_hours:
            return cycle_id - 1
        logger.warning("cycle skipped: too little of its run-start window is left", scoring_cycle_id=cycle_id - 1,
                       window_left_h=round(left_h, 3), catch_up_min_hours=pv.catch_up_min_hours)
    while published(pv, cycle_id):
        cycle_id += 1
    return cycle_id


def cycle_lock(pv, cycle_id: int):
    return run_lock(Path(pv.runs_dir) / str(cycle_id) / ".lock", f"scoring cycle {cycle_id}")


def run_cycle(config, cycle_id: int) -> None:
    """Run the cycle's stages, then report it."""
    with cycle_lock(config.prompt_validator, cycle_id):
        run_stages(config, cycle_id)
        run_stages(config, cycle_id, "report")
        if config.prompt_validator.weights.mode == "dry_run":
            try:
                run_stages(config, cycle_id, "weights")
            except Exception as e:  # a measurement of a published cycle: it must never make the loop retry it
                logger.warning("weights dry run not written", scoring_cycle_id=cycle_id,
                               error=f"{type(e).__name__}: {e}"[:500])


def load_config(args, parser: argparse.ArgumentParser) -> PromptValidatorConfig:
    """The YAML config, with --test applied."""
    config = PromptValidatorConfig.from_path(args.path, active_task=None)
    if args.loop and config.prompt_validator.prompt_source != "dataless":
        parser.error('--loop reports each cycle to the dataless-api after reward: it needs prompt_source "dataless"')
    if args.test:
        if config.prompt_validator.prompt_source != "dataless":
            parser.error('--test needs prompt_source "dataless"')
        config.prompt_validator.dataless.test_mode = True
    # Here, not in the config: every run and reload is checked, and a script that only reads the config is not.
    outside = caches_outside(config.prompt_validator.write_root, config.prompt_validator.engine)
    if outside:
        parser.error(outside)
    return config


def main() -> None:
    try:
        _main()
    except LockHeld as e:
        logger.error("prompt validator: lock held", error=str(e))
        sys.exit(75)  # EX_TEMPFAIL


def _main() -> None:
    # Unwind as an error does: scores first finishes the Jev calls already queued, and fetch may wait for its
    # download child. Every stage writes its output atomically, so a rerun resumes the cycle.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    parser = argparse.ArgumentParser(description="Prompt validator")
    parser.add_argument("--path", required=True,
                        help="Path to the prompt validator YAML config (see prompt_validator/config.example.yaml)")
    parser.add_argument("--cycle-id", type=int, help="Scoring cycle id (default: the current one)")
    parser.add_argument("--stage", choices=[name for name, _ in STAGES], help="Rerun only this stage")
    parser.add_argument("--loop", action="store_true",
                        help="Run every cycle in turn, each after its cutoff, and report it after reward (needs "
                             "prompt_source dataless)")
    parser.add_argument("--test", action="store_true",
                        help="Run a dataless-api test cycle, as dataless.test_mode: true does")
    parser.add_argument("--env-file", type=Path,
                        help="Secrets read when used, such as JEV_DIRECT_API_KEY, HF_TOKEN and BT_PW_* (bittensor's "
                             "encoded password, not the password); a variable already set wins (default: .env next "
                             "to the config, if any). HF_* other than HF_TOKEN, CUDA_* and LOG_* are read first: they "
                             "must come from the process environment, and the file's are ignored")
    parser.add_argument("--max-cycles", type=int,
                        help="With --loop: exit 0 once this process has seen this many cycles published (by it, or "
                             "found published already when it reported them)")
    args = parser.parse_args()
    if args.loop and (args.cycle_id is not None or args.stage is not None):
        parser.error("--loop picks each cycle and runs its stages: it takes no --cycle-id or --stage")
    if args.max_cycles is not None and (not args.loop or args.max_cycles < 1):
        parser.error(f"--max-cycles needs --loop and a count of at least 1 (got {args.max_cycles})")
    require_gpu_env()
    fmt, level, utc = (os.environ.get(f"LOG_{key}", default) for key, default in
                       (("FORMAT", "pretty"), ("LEVEL", "INFO"), ("UTC", "1")))
    if fmt not in ("pretty", "json") or not isinstance(logging.getLevelName(level), int) or utc not in ("0", "1"):
        parser.error(f"LOG_FORMAT must be pretty or json, LOG_LEVEL a level such as INFO, LOG_UTC 0 or 1 "
                     f"(got {fmt!r}, {level!r}, {utc!r})")
    configure_logging(fmt=fmt, level=level, use_utc=utc == "1")
    env_file = args.env_file or Path(args.path).parent / ".env"
    if env_file.is_file():
        values = dotenv_values(env_file)
        # Read on import, before this file is (HF_TOKEN alone is read when used): setting them now would only look
        # as if it worked.
        late = sorted(key for key in values if key.startswith(("HF_", "CUDA_", "LOG_")) and key != "HF_TOKEN")
        if late:
            logger.warning("env file keys ignored: set them in the process environment", keys=late)
        for key, value in values.items():
            if key not in late and value is not None:
                os.environ.setdefault(key, value)
        logger.info("env file loaded", path=str(env_file))  # never its values
    elif args.env_file is not None:
        parser.error(f"--env-file {env_file} does not exist")

    config = load_config(args, parser)
    if args.stage == "weights" and config.prompt_validator.weights.mode != "dry_run":
        parser.error("--stage weights needs prompt_validator.weights.mode dry_run")
    if args.loop:
        runs_dir = config.prompt_validator.runs_dir
        with run_lock(Path(runs_dir) / ".loop.lock", f"the --loop on {runs_dir}"), loop_status(runs_dir):
            # Shards of cycles published before this start: reports prune their own from here on.
            for published_id in sorted(published_cycles(runs_dir)):
                if os.path.lexists(stage_path(runs_dir, published_id, "fetch").parent / "models"):
                    try:
                        with cycle_lock(config.prompt_validator, published_id):
                            prune_shards(config.prompt_validator, published_id)
                    except LockHeld:
                        logger.info("shards not pruned: the cycle is in use", scoring_cycle_id=published_id)
                    except Exception as e:  # a cleanup step: it must never stop the loop
                        logger.warning("shards not pruned", scoring_cycle_id=published_id,
                                       error=f"{type(e).__name__}: {e}"[:500])
            attempt, failed, done = 0, None, 0
            while True:
                cycle_id, wait_min = None, None
                try:
                    set_status(state="sleeping", cycle=None, stage=None, progress=None, eta_min=None, retry_in_min=None,
                               **ledger_status(runs_dir))
                    if attempt:
                        config = load_config(args, parser)  # an edit to the YAML may be the fix
                    cycle_id = next_cycle(config)
                    logger.info("prompt validator loop: next cycle", scoring_cycle_id=cycle_id)
                    run_cycle(config, cycle_id)
                    attempt, failed, done = 0, None, done + 1
                    if done == args.max_cycles:
                        logger.info("prompt validator loop: --max-cycles done, stopping", cycles=done)
                        return
                    # Each cycle reads the YAML afresh; the snapshot's rules hash guards a cycle under way.
                    config = load_config(args, parser)
                except JevVersionError as e:
                    set_status(last_error=type(e).__name__)
                    # Retried, every attempt would pay for its Jev calls again and be refused again.
                    logger.error("Jev answered with another version than the one requested by name: check Jev, and "
                                 "change jev.resolved only if that version is intended",
                                 scoring_cycle_id=cycle_id, error=str(e))
                    sys.exit(78)  # EX_CONFIG
                except TesterRefused as e:
                    set_status(last_error=type(e).__name__)
                    # Retried, it would be refused again: only the owner can change who /v1/test serves.
                    logger.error("the dataless-api refuses this hotkey on /v1/test: see the error for why, ask the "
                                 "owner to fix it, then restart", scoring_cycle_id=cycle_id, error=str(e))
                    sys.exit(78)  # EX_CONFIG
                except Exception as e:
                    # Every other error is retried: stages resume from their markers, and next_cycle moves past a cycle
                    # once too little of its window is left. A plain traceback, whatever renderer is configured:
                    # a rich one would print every frame's locals.
                    if cycle_id is not None and cycle_id != failed:  # a failed pick or reload is no new cycle
                        attempt, failed = 0, cycle_id
                    attempt += 1
                    wait_min = min(2 ** (attempt - 1), RETRY_CAP_MIN)
                    logger.warning("cycle failed, will retry", scoring_cycle_id=cycle_id, attempt=attempt,
                                   retry_in_min=wait_min, error=f"{type(e).__name__}: {e}"[:500],
                                   traceback=traceback.format_exc())
                    # The type alone: a message may quote what a server sent back.
                    set_status(state="retrying", cycle=cycle_id, progress=None, eta_min=None,
                               last_error=type(e).__name__, retry_in_min=wait_min)
                # Out of the except, so the failed attempt's frames, and any model they hold on the GPU, are freed.
                if wait_min is not None:
                    time.sleep(wait_min * 60)
    pv = config.prompt_validator
    cycle_id = args.cycle_id
    if cycle_id is None and pv.prompt_source == "dataless" and pv.dataless.test_mode:
        # A test cycle has no blocks: run the closed test cycle not reported yet, once the owner closes it.
        cycle_id = validator_client(config).next_closed_test_cycle(lambda c: published(pv, c))
    elif cycle_id is None:
        phase = check_phase_clock(config, get_phase_from_api(config))
        cycle_id = scoring_cycle_of(phase.cycle_index, config.prompt_validator.scoring_cycle_k)
    logger.info("prompt validator starting", scoring_cycle_id=cycle_id, stage=args.stage)
    with cycle_lock(pv, cycle_id):
        run_stages(config, cycle_id, args.stage)


if __name__ == "__main__":
    main()
