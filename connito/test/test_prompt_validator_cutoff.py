from pathlib import Path
from types import SimpleNamespace

import pytest

from connito.prompt_validator import cutoff, run
from connito.prompt_validator.config import PromptValidatorCfg, PromptValidatorConfig
from connito.prompt_validator.stage_io import read_rows, stage_path, write_rows
from connito.shared.config import CycleCfg
from connito.shared.cycle import PhaseManager, PhaseNames

# The owner's cycle-api periods: 524 blocks per training cycle.
OWNER_PERIODS = dict(
    distribute_period=20, train_period=340, commit_period=11,
    submission_period=60, validate_period=10, merge_period=50,
)
# Scoring cycle 1258 ends with training cycle 1259 * 14 - 1 = 17625, starting at 17625 * 524 = 9235500.
PAST_CYCLE_ID = 1258
TRAINING_CYCLE = 17625
CYCLE_START = 9235500
COMMIT_1_BLOCK = CYCLE_START + 20 + 340 + 11  # MinerCommit1 ends at +370
COMMIT_2_BLOCK = COMMIT_1_BLOCK + 11  # MinerCommit2 ends at +381
# A later /get_phase answer: training cycle 17631, in Submission.
LIVE_BLOCK = 9239036
LIVE_SUBMISSION = (9239026, 9239085)
OWNER = SimpleNamespace(cycle=CycleCfg(**OWNER_PERIODS))
EXPERT_GROUPS = Path(__file__).resolve().parents[2] / "expert_groups"


def _config(runs_dir, **periods):
    return SimpleNamespace(
        cycle=CycleCfg(**{**OWNER_PERIODS, **periods}),
        # The block numbers below are worked out for 14 training cycles per scoring cycle.
        prompt_validator=PromptValidatorCfg(runs_dir=runs_dir, prompts_path=runs_dir / "prompts.jsonl",
                                            scoring_cycle_k=14),
    )


@pytest.fixture
def config(tmp_path):
    return _config(tmp_path)


def _api_phase(block, **overrides):
    """/get_phase as cycle-api serves it at `block`."""
    return PhaseManager(OWNER, None).get_phase(block).model_copy(update=overrides)


def _api_next_windows(block):
    """/blocks_until_next_phase as cycle-api serves it at `block`."""
    windows = PhaseManager(OWNER, SimpleNamespace(block=block)).blocks_until_next_phase()
    return {name: list(window) for name, window in windows.items()}


@pytest.fixture
def serve_api(monkeypatch):
    def serve(phase, next_windows=None):
        next_windows = next_windows or _api_next_windows(LIVE_BLOCK)
        monkeypatch.setattr(cutoff, "get_phase_from_api", lambda cfg: phase)
        monkeypatch.setattr(cutoff, "get_blocks_until_next_phase_from_api", lambda cfg: next_windows)

    return serve


@pytest.fixture
def no_wait(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("wait_till called")

    monkeypatch.setattr(cutoff, "wait_till", fail)


@pytest.mark.parametrize("cycle_index, expected", [(0, 0), (13, 0), (14, 1), (27, 1)])
def test_scoring_cycle_of(cycle_index, expected):
    assert cutoff.scoring_cycle_of(cycle_index, 14) == expected


def test_cutoff_blocks_for_known_cycle(config):
    blocks = cutoff.cutoff_blocks(config, PAST_CYCLE_ID)

    assert PhaseManager(config, None).cycle_length == 524
    assert blocks == cutoff.CutoffBlocks(PAST_CYCLE_ID, TRAINING_CYCLE, COMMIT_1_BLOCK, COMMIT_2_BLOCK)


def test_cutoff_blocks_match_the_phase_windows(config):
    blocks = cutoff.cutoff_blocks(config, PAST_CYCLE_ID)
    phase_manager = PhaseManager(config, None)

    assert phase_manager.get_phase(blocks.commit_1_block - 1).phase_name == PhaseNames.miner_commit_1
    assert phase_manager.get_phase(blocks.commit_1_block).phase_name == PhaseNames.miner_commit_2
    assert phase_manager.get_phase(blocks.commit_2_block).phase_name == PhaseNames.submission
    assert phase_manager.get_phase(blocks.commit_2_block).cycle_index == TRAINING_CYCLE


def test_past_cycle_returns_without_waiting(config, serve_api, no_wait):
    live = _api_phase(LIVE_BLOCK)
    assert (live.phase_start_block, live.phase_end_block) == LIVE_SUBMISSION
    serve_api(live)

    rows = cutoff.run_cutoff(config, PAST_CYCLE_ID)

    assert rows == [{
        "scoring_cycle_id": PAST_CYCLE_ID,
        "training_cycle": TRAINING_CYCLE,
        "commit_1_block": COMMIT_1_BLOCK,
        "commit_2_block": COMMIT_2_BLOCK,
    }]


@pytest.mark.parametrize("overrides", [
    {"cycle_length": 500},
    {"phase_start_block": LIVE_SUBMISSION[0] + 1},
    {"phase_end_block": LIVE_SUBMISSION[1] + 1},
])
def test_clock_mismatch_stops_the_run(config, serve_api, no_wait, overrides):
    serve_api(_api_phase(LIVE_BLOCK, **overrides))

    with pytest.raises(RuntimeError, match="disagrees with cycle-api"):
        cutoff.run_cutoff(config, PAST_CYCLE_ID)


def test_same_length_different_split_stops_the_run(tmp_path, serve_api, no_wait):
    config = _config(tmp_path, commit_period=10, submission_period=64)
    train = _api_phase((TRAINING_CYCLE + 7) * 524 + 100)
    local = PhaseManager(config, None)
    assert local.cycle_length == 524
    assert local.get_phase(train.block) == train
    serve_api(train)

    with pytest.raises(RuntimeError, match="disagrees with cycle-api"):
        cutoff.run_cutoff(config, PAST_CYCLE_ID)


def test_missing_phase_stops_the_run(config, serve_api, no_wait):
    next_windows = _api_next_windows(LIVE_BLOCK)
    del next_windows[PhaseNames.merge]
    serve_api(_api_phase(LIVE_BLOCK), next_windows)

    with pytest.raises(RuntimeError, match="disagrees with cycle-api"):
        cutoff.run_cutoff(config, PAST_CYCLE_ID)


@pytest.mark.parametrize("phase_down, windows_down", [(True, False), (False, True)])
def test_unreachable_api_stops_the_run(config, monkeypatch, serve_api, no_wait, phase_down, windows_down):
    serve_api(None if phase_down else _api_phase(LIVE_BLOCK))
    if windows_down:
        monkeypatch.setattr(cutoff, "get_blocks_until_next_phase_from_api", lambda cfg: None)

    with pytest.raises(RuntimeError, match="unreachable"):
        cutoff.run_cutoff(config, PAST_CYCLE_ID)


def test_current_cycle_waits_past_earlier_submissions(config, monkeypatch, serve_api):
    # Start inside the previous training cycle's Submission window.
    start = _api_phase(COMMIT_2_BLOCK - 524 + 5)
    reached = {
        PhaseNames.validate: _api_phase(COMMIT_2_BLOCK - 524 + 60),
        PhaseNames.submission: _api_phase(COMMIT_2_BLOCK),
    }
    assert start.phase_name == PhaseNames.submission
    assert reached[PhaseNames.validate].phase_name == PhaseNames.validate
    waits = []

    def fake_wait_till(cfg, phase_name):
        waits.append(phase_name)
        return reached[phase_name]

    serve_api(start, _api_next_windows(start.block))
    monkeypatch.setattr(cutoff, "wait_till", fake_wait_till)

    rows = cutoff.run_cutoff(config, PAST_CYCLE_ID)

    assert waits == [PhaseNames.validate, PhaseNames.submission]
    assert rows[0]["commit_2_block"] == COMMIT_2_BLOCK


def test_stage_rows_round_trip(tmp_path):
    path = stage_path(tmp_path, 7, "cutoff")
    rows = [{"a": 1}, {"b": [2, 3]}]

    write_rows(path, rows)

    assert path == tmp_path / "7" / "cutoff" / "cutoff.jsonl"
    assert read_rows(path) == rows
    assert [p.name for p in path.parent.iterdir()] == ["cutoff.jsonl"]


def test_finished_stage_is_skipped_unless_named(config, monkeypatch):
    path = stage_path(config.prompt_validator.runs_dir, PAST_CYCLE_ID, "cutoff")
    write_rows(path, [{"old": True}])
    calls = []

    def fake_stage(cfg, cycle_id):
        calls.append(cycle_id)
        return [{"new": True}]

    monkeypatch.setattr(run, "STAGES", [("cutoff", fake_stage)])

    run.run_stages(config, PAST_CYCLE_ID)
    assert calls == []
    assert read_rows(path) == [{"old": True}]

    run.run_stages(config, PAST_CYCLE_ID, stage="cutoff")
    assert calls == [PAST_CYCLE_ID]
    assert read_rows(path) == [{"new": True}]


def test_failed_rerun_leaves_no_marker(config, monkeypatch):
    path = stage_path(config.prompt_validator.runs_dir, PAST_CYCLE_ID, "cutoff")
    write_rows(path, [{"old": True}])

    def failing_stage(cfg, cycle_id):
        assert not path.exists()
        raise RuntimeError("stage failed")

    monkeypatch.setattr(run, "STAGES", [("cutoff", failing_stage)])

    with pytest.raises(RuntimeError, match="stage failed"):
        run.run_stages(config, PAST_CYCLE_ID, stage="cutoff")
    assert not path.exists()


def test_rerun_clears_the_markers_of_later_stages(config, monkeypatch):
    runs_dir = config.prompt_validator.runs_dir
    monkeypatch.setattr(run, "STAGES", [(name, lambda cfg, cycle_id: [{"new": True}]) for name in ("a", "b", "c")])
    for name in ("a", "b", "c"):
        write_rows(stage_path(runs_dir, PAST_CYCLE_ID, name), [{"old": True}])

    run.run_stages(config, PAST_CYCLE_ID, stage="b")

    assert read_rows(stage_path(runs_dir, PAST_CYCLE_ID, "a")) == [{"old": True}]
    assert read_rows(stage_path(runs_dir, PAST_CYCLE_ID, "b")) == [{"new": True}]
    assert not stage_path(runs_dir, PAST_CYCLE_ID, "c").exists()


@pytest.mark.parametrize("runs_dir, expected", [
    ("runs", lambda root, home: root / "runs"),
    ("~/pv-runs", lambda root, home: home / "pv-runs"),
    ("/abs/runs", lambda root, home: Path("/abs/runs")),
])
def test_runs_dir_resolves_under_root_path(tmp_path, monkeypatch, pv_chain, runs_dir, expected):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = PromptValidatorConfig(
        run={"root_path": tmp_path},
        chain=pv_chain,
        task={"base_path": EXPERT_GROUPS},
        prompt_validator={"runs_dir": runs_dir, "prompts_path": "/abs/prompts.jsonl", "write_root": "/"},
    )

    assert config.prompt_validator.runs_dir == expected(tmp_path, tmp_path / "home")


def test_a_wait_for_the_cutoff_records_when_it_is_due_in_the_status(config, monkeypatch, serve_api):
    start = _api_phase(COMMIT_2_BLOCK - 100)
    serve_api(start, _api_next_windows(start.block))
    monkeypatch.setattr(cutoff, "wait_till", lambda cfg, phase_name: _api_phase(COMMIT_2_BLOCK))
    monkeypatch.setattr(cutoff.time, "time", lambda: 1_800_000_000.0)
    fields = {}
    monkeypatch.setattr(cutoff, "set_status", lambda **kw: fields.update(kw))

    cutoff.run_cutoff(config, PAST_CYCLE_ID)

    assert fields == {"cutoff_at": "2027-01-15T08:20:00+00:00"}  # 100 blocks of 12 s after 08:00:00
