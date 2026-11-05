"""Stage 1: wait until a scoring cycle's models are fixed on chain, and record where to read them.

The models are the commits the training validator reads for the scoring cycle's last training
cycle: signed hashes at MinerCommit1 end + 1, hashes at MinerCommit2 end + 1 (the cutoff).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from connito.prompt_validator.dataless import SNAPSHOT_DELAY_BLOCKS
from connito.prompt_validator.status import set_status
from connito.shared.app_logging import structlog
from connito.shared.cycle import (
    BITTENSOR_BLOCK_TIME_SECONDS,
    PhaseManager,
    PhaseNames,
    PhaseResponse,
    get_blocks_until_next_phase_from_api,
    get_phase_from_api,
    wait_till,
)

logger = structlog.get_logger(__name__)


def scoring_cycle_of(cycle_index: int, k: int) -> int:
    return cycle_index // k


def prompt_cycle_length(config) -> int:
    """Blocks per scoring cycle, which is the dataless-api's prompt cycle."""
    return config.prompt_validator.scoring_cycle_k * PhaseManager(config, None).cycle_length


def report_before_block(config, scoring_cycle_id: int) -> int:
    """The block before which the cycle's run must start: while it is the last closed cycle, until the next one's
    cutoff plus the snapshot delay."""
    return (scoring_cycle_id + 2) * prompt_cycle_length(config) - 1 + SNAPSHOT_DELAY_BLOCKS


@dataclass
class CutoffBlocks:
    scoring_cycle_id: int
    training_cycle: int
    commit_1_block: int
    commit_2_block: int


def cutoff_blocks(config, scoring_cycle_id: int) -> CutoffBlocks:
    """The commit-read blocks of the scoring cycle's last training cycle."""
    training_cycle = (scoring_cycle_id + 1) * config.prompt_validator.scoring_cycle_k - 1
    return training_cycle_blocks(config, scoring_cycle_id, training_cycle)


def training_cycle_blocks(config, scoring_cycle_id: int, training_cycle: int) -> CutoffBlocks:
    """The commit-read blocks of one training cycle, recorded for `scoring_cycle_id`."""
    phase_manager = PhaseManager(config, None)
    phase_end = training_cycle * phase_manager.cycle_length - 1
    phase_ends = {}
    for phase in phase_manager.phases:
        phase_end += phase["length"]
        phase_ends[phase["name"]] = phase_end
    return CutoffBlocks(
        scoring_cycle_id=scoring_cycle_id,
        training_cycle=training_cycle,
        commit_1_block=phase_ends[PhaseNames.miner_commit_1] + 1,
        commit_2_block=phase_ends[PhaseNames.miner_commit_2] + 1,
    )


def check_phase_clock(config, phase: PhaseResponse | None) -> PhaseResponse:
    """Stop unless every phase window from config matches cycle-api, since every block here comes from it."""
    next_windows = get_blocks_until_next_phase_from_api(config)
    if phase is None or next_windows is None:
        raise RuntimeError("cycle-api /get_phase or /blocks_until_next_phase unreachable")
    phase_manager = PhaseManager(config, None)
    if phase_manager.cycle_length != phase.cycle_length:
        raise RuntimeError(
            f"cycle config disagrees with cycle-api: cycle_length {phase_manager.cycle_length} vs {phase.cycle_length}"
        )
    phase_names = {p["name"] for p in phase_manager.phases}
    if set(next_windows) != phase_names:
        raise RuntimeError(
            f"cycle config disagrees with cycle-api: phases {sorted(next_windows)} vs {sorted(phase_names)}"
        )
    api_windows = [(phase.phase_name, phase.phase_start_block, phase.phase_end_block)]
    api_windows += [(name, start, end) for name, (start, end, _) in next_windows.items()]
    for name, start, end in api_windows:
        local = phase_manager.get_phase(start)
        local_window = (local.phase_name, local.phase_start_block, local.phase_end_block)
        if local_window != (name, start, end):
            raise RuntimeError(
                f"cycle config disagrees with cycle-api: {name} is {(start, end)}, config gives {local_window}"
            )
    return phase


def latest_passed_cutoff(config, scoring_cycle_id: int) -> CutoffBlocks:
    """For a test cycle, which has no blocks: the commit blocks of the latest training cycle whose cutoff has passed,
    so the run answers with the miners' real models without waiting."""
    phase = check_phase_clock(config, get_phase_from_api(config))
    # The blocks below count training cycles from block 0, as cutoff_blocks does: cycle-api's index must too.
    if phase.cycle_index != phase.block // phase.cycle_length:
        raise RuntimeError(f"cycle-api's cycle_index {phase.cycle_index} is not block {phase.block} // "
                           f"cycle_length {phase.cycle_length}")
    blocks = training_cycle_blocks(config, scoring_cycle_id, phase.cycle_index)
    if phase.block < blocks.commit_2_block:
        blocks = training_cycle_blocks(config, scoring_cycle_id, phase.cycle_index - 1)
    return blocks


def run_cutoff(config, scoring_cycle_id: int) -> list[dict]:
    pv = config.prompt_validator
    if pv.prompt_source == "dataless" and pv.dataless.test_mode:
        blocks = latest_passed_cutoff(config, scoring_cycle_id)
        logger.info("test cycle: models from the latest real cutoff", **asdict(blocks))
        return [{**asdict(blocks), "test_cycle": True}]
    blocks = cutoff_blocks(config, scoring_cycle_id)
    phase = check_phase_clock(config, get_phase_from_api(config))
    if phase.block < blocks.commit_2_block:
        blocks_left = blocks.commit_2_block - phase.block
        logger.info("waiting for cutoff", block=phase.block, blocks_left=blocks_left)
        # For the healthcheck: a loop still waiting well past this is stuck.
        cutoff_at = datetime.fromtimestamp(time.time() + blocks_left * BITTENSOR_BLOCK_TIME_SECONDS, timezone.utc)
        set_status(cutoff_at=cutoff_at.isoformat(timespec="seconds"))
    while phase.block < blocks.commit_2_block:
        # wait_till(Submission) returns at once inside Submission; step past it first.
        if phase.phase_name == PhaseNames.submission:
            wait_till(config, PhaseNames.validate)
        phase = check_phase_clock(config, wait_till(config, PhaseNames.submission))
    logger.info("cutoff passed", block=phase.block, **asdict(blocks))
    return [asdict(blocks)]
