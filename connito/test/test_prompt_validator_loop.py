import json
import logging
import os
import signal
import sys
from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from connito.prompt_validator import run
from connito.prompt_validator.config import DatalessCfg, PromptValidatorCfg
from connito.prompt_validator.dataless import TesterRefused
from connito.prompt_validator.jev import JevVersionError
from connito.prompt_validator.stage_io import append_published, published_cycles, run_lock, stage_path, write_rows
from connito.shared.config import CycleCfg
from connito.test.test_prompt_validator_cutoff import OWNER_PERIODS

URL = "https://dataless.test"
OPEN = 1259  # the open scoring cycle in these tests
LENGTH = 13 * 524
TEST_CYCLE = 1_000_000_003


class Stop(BaseException):
    """Ends a loop that would otherwise wait for the next cycle forever: a BaseException, which the loop never
    retries."""


def _config(tmp_path, source="dataless", test_mode=False):
    pv = PromptValidatorCfg(runs_dir=tmp_path, write_root="/", prompt_source=source,
                            prompts_path=None if source == "dataless" else tmp_path / "prompts.jsonl",
                            dataless=DatalessCfg(url=URL, test_mode=test_mode))
    return SimpleNamespace(prompt_validator=pv, cycle=CycleCfg(**OWNER_PERIODS))


def _mark(config, cycle_id, *stages):
    for stage in stages:
        write_rows(stage_path(config.prompt_validator.runs_dir, cycle_id, stage), [{}])


@pytest.fixture
def chain_at(monkeypatch):
    """cycle-api answering at `block` (default: some way into the open scoring cycle)."""
    def at(block=OPEN * LENGTH + 2000):
        phase = SimpleNamespace(block=block, cycle_index=block // 524)
        monkeypatch.setattr(run, "get_phase_from_api", lambda cfg: phase)
        monkeypatch.setattr(run, "check_phase_clock", lambda cfg, p: p)
        return phase

    return at


@pytest.mark.parametrize(("marks", "expected"), [
    ({}, OPEN - 1),  # the one before, still listable with 16 h to start its run: whether begun here or not
    ({OPEN - 1: ("cutoff", "snapshot", "fetch")}, OPEN - 1),
    ({OPEN - 1: ("cutoff", "reward")}, OPEN - 1),  # scored but not reported: done means reported
    ({OPEN - 1: ("cutoff", "reward", "report")}, OPEN),  # done: the open one, whose cutoff stage waits for its cutoff
    ({OPEN - 2: ("cutoff",), OPEN - 1: ("report",)}, OPEN),  # no longer listable: never gone back to
    ({OPEN - 1: ("report",), OPEN: ("cutoff", "report")}, OPEN + 1),  # the open one finished before it closed
])
def test_the_loop_runs_the_first_unfinished_cycle_from_the_one_before_the_open_one(tmp_path, chain_at, marks,
                                                                                   expected):
    config = _config(tmp_path)
    for cycle_id, stages in marks.items():
        _mark(config, cycle_id, *stages)
    chain_at()
    assert run.next_cycle(config) == expected


# Cycle OPEN - 1's run starts until cycle OPEN is closed: its last block plus the server's 10-block delay.
DEADLINE = (OPEN + 1) * LENGTH - 1 + 10
SIX_HOURS = 6 * 3600 // 12  # blocks


@pytest.mark.parametrize(("blocks_left", "expected"), [(SIX_HOURS, OPEN - 1), (SIX_HOURS - 1, OPEN)])
def test_the_cycle_before_is_run_only_with_catch_up_min_hours_of_its_window_left(tmp_path, chain_at, caplog,
                                                                                 blocks_left, expected):
    config = _config(tmp_path)
    _mark(config, OPEN - 1, "cutoff", "snapshot")  # begun here or not, alike
    chain_at(DEADLINE - blocks_left)
    with caplog.at_level(logging.WARNING):
        assert run.next_cycle(config) == expected
    skipped = ("cycle skipped: too little of its run-start window is left catch_up_min_hours=6.0 "
               f"scoring_cycle_id={OPEN - 1} window_left_h=5.997")
    assert (skipped in caplog.text) == (expected == OPEN)


@pytest.mark.parametrize(("blocks_left", "expected"), [(300, OPEN - 1), (0, OPEN)])  # 300 blocks: 1 h
def test_a_cycle_past_reward_is_run_while_any_of_its_window_is_left(tmp_path, monkeypatch, caplog, blocks_left,
                                                                    expected):
    config = _config(tmp_path)
    _mark(config, OPEN - 1, "cutoff", "reward")  # only its report is left, which takes seconds
    # cycle-api still names OPEN as open, as when its clock lags the block: the window rule alone decides.
    phase = SimpleNamespace(block=DEADLINE - blocks_left, cycle_index=OPEN * 13 + 12)
    monkeypatch.setattr(run, "get_phase_from_api", lambda cfg: phase)
    monkeypatch.setattr(run, "check_phase_clock", lambda cfg, p: p)
    with caplog.at_level(logging.WARNING):
        assert run.next_cycle(config) == expected
    assert ("cycle skipped: too little of its run-start window is left" in caplog.text) == (expected == OPEN)


class FakeTestCycles:
    """Test cycles the owner closes one after another: the first that `done` says is not done."""

    def __init__(self, cfg):
        pass

    def next_closed_test_cycle(self, done):
        cycle_id = TEST_CYCLE
        while done(cycle_id):  # the real client waits here for the owner to close the next
            cycle_id += 1
        return cycle_id


@pytest.fixture
def looping(monkeypatch):
    """Run main() --loop until `cycles` cycles have run, each load a fresh `make_config()`; returns the run_stages
    calls and the configs loaded. With `stops`, main() must return before that, by itself."""
    def start(make_config, *args, cycles=2, stage=None, sleep=None, stops=False):
        calls, loads = [], []

        def load(path, active_task):
            loads.append(make_config())
            return loads[-1]

        def fake_stages(cfg, cycle_id, name=None):
            if sleep is None and (cycle_id, name) in calls:  # rather than loop forever
                pytest.fail(f"cycle {cycle_id} run again")
            if len(calls) > 50:
                pytest.fail("the loop never stopped")
            calls.append((cycle_id, name))
            if stage is not None:
                stage(cfg, cycle_id, name)
            if name is None:
                _mark(cfg, cycle_id, "cutoff", "reward")
            else:
                _mark(cfg, cycle_id, name)
            if len({c for c, _ in calls}) == cycles and name == "report":
                raise Stop

        monkeypatch.setattr(run.PromptValidatorConfig, "from_path", load)
        monkeypatch.setattr(run, "run_stages", fake_stages)
        monkeypatch.setattr(run, "validator_client", FakeTestCycles)
        monkeypatch.setattr(run, "get_phase_from_api", lambda cfg: pytest.fail("a test cycle has no phase"))
        slept = []

        def sleeping(sec):
            slept.append(sec)
            # Raised from the loop's except clause, so it ends a loop that caught what it should not have.
            if sleep is None or len(slept) > 20:
                pytest.fail("a cycle failed" if sleep is None else "the loop kept retrying")
            sleep(sec)

        monkeypatch.setattr(run, "time", SimpleNamespace(sleep=sleeping))
        monkeypatch.setattr("sys.argv", ["run", "--path", "x.yaml", "--loop", *args])
        monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
        if stops:
            run.main()
        else:
            with pytest.raises(Stop):
                run.main()
        return calls, loads

    return start


def test_a_test_mode_loop_takes_each_closed_test_cycle_once_and_rereads_the_config(tmp_path, looping):
    calls, loads = looping(lambda: _config(tmp_path), "--test", cycles=3)
    assert calls == [(c, name) for c in (TEST_CYCLE, TEST_CYCLE + 1, TEST_CYCLE + 2) for name in (None, "report")]
    assert len(loads) == 3 and all(c.prompt_validator.dataless.test_mode for c in loads)  # --test on each load


def test_the_loop_reports_each_cycle_after_reward(tmp_path, looping):
    calls, _ = looping(lambda: _config(tmp_path, test_mode=True), cycles=2)
    assert calls == [(TEST_CYCLE, None), (TEST_CYCLE, "report"), (TEST_CYCLE + 1, None), (TEST_CYCLE + 1, "report")]


def test_the_loop_is_refused_without_the_dataless_prompt_source(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run.PromptValidatorConfig, "from_path", lambda path, active_task: _config(tmp_path, "file"))
    monkeypatch.setattr(run, "next_cycle", lambda cfg: pytest.fail("a cycle was picked"))
    monkeypatch.setattr("sys.argv", ["run", "--path", "x.yaml", "--loop"])
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(SystemExit) as exit_:
        run.main()
    err = capsys.readouterr().err
    assert exit_.value.code == 2 and '--loop reports each cycle to the dataless-api' in err
    assert 'it needs prompt_source "dataless"' in err


@pytest.mark.parametrize("args", [["--cycle-id", "7"], ["--stage", "fetch"]])
def test_the_loop_takes_no_cycle_id_or_stage(monkeypatch, capsys, args):
    monkeypatch.setattr("sys.argv", ["run", "--path", "x.yaml", "--loop", *args])
    with pytest.raises(SystemExit) as exit_:
        run.main()
    assert exit_.value.code == 2 and "--loop picks each cycle" in capsys.readouterr().err


def test_max_cycles_stops_the_test_mode_loop_cleanly_after_that_many_cycles(tmp_path, looping, caplog):
    append_published(tmp_path, {"cycle_id": 1368})  # a real cycle published before

    def report(cfg, cycle_id, name):
        if name == "report":
            append_published(tmp_path, {"cycle_id": cycle_id})

    with caplog.at_level(logging.INFO):
        calls, _ = looping(lambda: _config(tmp_path, test_mode=True), "--max-cycles", "2", cycles=5, stops=True,
                           stage=report)
    assert calls == [(TEST_CYCLE, None), (TEST_CYCLE, "report"), (TEST_CYCLE + 1, None), (TEST_CYCLE + 1, "report")]
    assert "--max-cycles done, stopping" in caplog.text
    status = json.loads((tmp_path / "status.json").read_text())
    # Its last cycle's publish counted, and no cycle shown as under way.
    assert (status["state"], status["cycle"], status["stage"]) == ("stopped", None, None)
    assert (status["published"], status["last_published"]) == (3, TEST_CYCLE + 1)


def test_max_cycles_counts_a_failed_then_retried_cycle_once_when_it_is_done(tmp_path, looping):
    sleeps = []
    calls, _ = looping(lambda: _config(tmp_path, test_mode=True), "--max-cycles", "2", cycles=5, stops=True,
                       stage=_failing(2), sleep=sleeps.append)
    assert calls == [(TEST_CYCLE, None)] * 3 + [(TEST_CYCLE, "report"), (TEST_CYCLE + 1, None),
                                                (TEST_CYCLE + 1, "report")]
    assert sleeps == [60, 120]


def test_max_cycles_stops_a_real_loop_after_that_many_cycles(tmp_path, chain_at, monkeypatch):
    config = _config(tmp_path)
    chain_at()
    monkeypatch.setattr(run.PromptValidatorConfig, "from_path", lambda path, active_task: config)
    calls = []

    def stages(cfg, cycle_id, name=None):
        if len(calls) > 10:
            pytest.fail("the loop never stopped")
        calls.append((cycle_id, name))
        _mark(cfg, cycle_id, *(["cutoff", "reward"] if name is None else [name]))

    monkeypatch.setattr(run, "run_stages", stages)
    monkeypatch.setattr(run, "time", SimpleNamespace(sleep=lambda sec: pytest.fail("a cycle failed")))
    monkeypatch.setattr("sys.argv", ["run", "--path", "x.yaml", "--loop", "--max-cycles", "2"])
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    run.main()  # returns: exit 0
    assert calls == [(OPEN - 1, None), (OPEN - 1, "report"), (OPEN, None), (OPEN, "report")]


@pytest.mark.parametrize("args", [["--max-cycles", "2"], ["--loop", "--max-cycles", "0"],
                                  ["--loop", "--max-cycles", "-1"]])
def test_max_cycles_needs_the_loop_and_at_least_one_cycle(monkeypatch, capsys, args):
    monkeypatch.setattr(run.PromptValidatorConfig, "from_path", lambda *a, **k: pytest.fail("the config was read"))
    monkeypatch.setattr("sys.argv", ["run", "--path", "x.yaml", *args])
    with pytest.raises(SystemExit) as exit_:
        run.main()
    assert exit_.value.code == 2 and "--max-cycles needs --loop and a count of at least 1" in capsys.readouterr().err


def test_sigterm_unwinds_the_loop_and_exits_143(tmp_path, looping):
    def terminated(cfg, cycle_id, name):
        os.kill(os.getpid(), signal.SIGTERM)

    with pytest.raises(SystemExit) as exit_:
        looping(lambda: _config(tmp_path, test_mode=True), stage=terminated)
    assert exit_.value.code == 143


# --- the published ledger ---

def _published(config, cycle_id, source="this_run"):
    append_published(config.prompt_validator.runs_dir, {"cycle_id": cycle_id, "source": source})


def test_a_rerun_of_an_earlier_stage_never_makes_a_published_cycle_reportable_again(tmp_path, monkeypatch, caplog):
    config = _config(tmp_path, test_mode=True)
    monkeypatch.setattr(run, "STAGES", [(name, lambda cfg, c: [{}]) for name, _ in run.STAGES])
    _mark(config, TEST_CYCLE, "cutoff", "snapshot", "fetch", "answers", "scores", "reward", "report")
    _published(config, TEST_CYCLE)
    with caplog.at_level(logging.WARNING):
        run.run_stages(config, TEST_CYCLE, "scores")
    pv = config.prompt_validator
    assert not stage_path(pv.runs_dir, TEST_CYCLE, "report").exists()  # removed with the stages after scores
    assert "removed the report of a published cycle: the files here no longer match" in caplog.text
    assert "--stage report" not in caplog.text
    assert run.published(pv, TEST_CYCLE)
    assert FakeTestCycles(None).next_closed_test_cycle(lambda c: run.published(pv, c)) == TEST_CYCLE + 1


def test_a_report_from_before_the_ledger_still_counts_as_published(tmp_path):
    config = _config(tmp_path)
    _mark(config, TEST_CYCLE, "report")
    assert run.published(config.prompt_validator, TEST_CYCLE)


def test_the_loop_skips_a_cycle_the_ledger_has_published_though_it_has_no_marker(tmp_path, looping, chain_at):
    config = _config(tmp_path, test_mode=True)
    _published(config, TEST_CYCLE, source="server")  # published from another runs_dir, as report found
    calls, _ = looping(lambda: _config(tmp_path, test_mode=True), cycles=1)
    assert calls == [(TEST_CYCLE + 1, None), (TEST_CYCLE + 1, "report")]
    real = _config(tmp_path / "real")
    _published(real, OPEN - 1)
    chain_at()
    assert run.next_cycle(real) == OPEN


def test_a_plain_test_mode_run_skips_a_cycle_the_ledger_has_published(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    config = _config(tmp_path, test_mode=True)
    _published(config, TEST_CYCLE)
    monkeypatch.setattr(run.PromptValidatorConfig, "from_path", lambda path, active_task: config)
    monkeypatch.setattr(run, "validator_client", FakeTestCycles)
    calls = []
    monkeypatch.setattr(run, "run_stages", lambda cfg, cycle_id, name=None: calls.append(cycle_id))
    monkeypatch.setattr("sys.argv", ["run", "--path", "x.yaml"])
    run.main()
    assert calls == [TEST_CYCLE + 1]


def test_the_ledger_only_grows_and_a_line_a_crash_cut_short_hides_no_other_row(tmp_path, caplog):
    config = _config(tmp_path)
    _published(config, 7)
    _published(config, 9)
    assert published_cycles(tmp_path) == {7, 9}
    with open(tmp_path / "published.jsonl", "a") as f:
        f.write('{"cycle_id": 8, "secret-ish')
    _published(config, 10)  # after the torn line, on a line of its own
    with open(tmp_path / "published.jsonl", "ab") as f:
        f.write(b'5\n\n{"cycle_id": 6\xff}\n')  # JSON but not a row, a blank line, a damaged byte
    _published(config, 11)
    with caplog.at_level(logging.WARNING):
        assert published_cycles(tmp_path) == {7, 9, 10, 11}
    warned = [line for line in (3, 5, 6, 7) if f"line={line}" in caplog.text]
    assert "ledger line unreadable, skipped" in caplog.text and warned == [3, 5, 7]  # not the blank line
    assert "secret-ish" not in caplog.text  # the line number, never the content
    assert published_cycles(tmp_path / "none") == set()


def test_a_ledger_row_whose_cycle_id_is_not_an_integer_is_skipped(tmp_path, caplog):
    _published(_config(tmp_path), 7)
    with open(tmp_path / "published.jsonl", "a") as f:
        f.write('{"cycle_id": "8"}\n{"cycle_id": true}\n{"cycle_id": 9.0}\n{"cycle_id": null}\n')
    _published(_config(tmp_path), 10)
    with caplog.at_level(logging.WARNING):
        assert published_cycles(tmp_path) == {7, 10}
    assert [line for line in range(1, 7) if f"line={line}" in caplog.text] == [2, 3, 4, 5]


def test_a_short_ledger_write_raises(tmp_path, monkeypatch):
    real_write = os.write
    monkeypatch.setattr(os, "write", lambda fd, data: real_write(fd, data[: len(data) // 2]))
    with pytest.raises(OSError, match="short write"):
        _published(_config(tmp_path), 7)
    monkeypatch.undo()
    assert published_cycles(tmp_path) == set()


# --- failures ---

def _failing(times, error=RuntimeError):
    """A stage that raises on its first `times` calls."""
    failures = [times]

    def stage(cfg, cycle_id, name):
        if failures[0]:
            failures[0] -= 1
            raise error("dataless-api POST /status failed: ConnectionError")

    return stage


def test_a_failed_cycle_is_retried_after_a_minute_on_a_fresh_config_and_resumes(tmp_path, looping):
    sleeps = []
    calls, loads = looping(lambda: _config(tmp_path, test_mode=True), cycles=1, stage=_failing(1),
                           sleep=sleeps.append)
    assert calls == [(TEST_CYCLE, None), (TEST_CYCLE, None), (TEST_CYCLE, "report")] and sleeps == [60]
    assert len(loads) == 2  # the YAML is read again before the retry


def test_the_backoff_doubles_up_to_its_cap_and_starts_again_after_a_success(tmp_path, looping):
    sleeps = []
    failures = {TEST_CYCLE: 7, TEST_CYCLE + 1: 1}

    def stage(cfg, cycle_id, name):
        if failures[cycle_id]:
            failures[cycle_id] -= 1
            raise RuntimeError("Jev call failed (see the error logged above)")

    loads = []

    def make_config():
        loads.append(1)
        if len(loads) == 9:  # the reload after TEST_CYCLE's success: the initial load, then 7 before its retries
            raise ValueError("prompt_validator.max_models: input should be a valid integer")
        return _config(tmp_path, test_mode=True)

    looping(make_config, cycles=2, stage=stage, sleep=sleeps.append)
    assert sleeps == [60 * m for m in (1, 2, 4, 8, 16, 30, 30)] + [60, 60]


def test_a_failing_pick_or_config_reload_is_retried_and_no_stage_runs_on_the_old_config(tmp_path, looping,
                                                                                       monkeypatch):
    picks = []
    real_pick = FakeTestCycles.next_closed_test_cycle

    def pick(self, done):
        picks.append(1)
        if len(picks) == 1:
            raise RuntimeError("dataless-api GET /v1/test/prompt-cycle failed: ConnectionError")
        return real_pick(self, done)

    monkeypatch.setattr(FakeTestCycles, "next_closed_test_cycle", pick)
    loaded = []

    def make_config():
        loaded.append(_config(tmp_path, test_mode=True))
        if len(loaded) == 2:
            raise ValueError("prompt_validator.max_models: input should be a valid integer")
        return loaded[-1]

    seen = []
    sleeps = []
    calls, _ = looping(make_config, cycles=1, stage=lambda cfg, c, n: seen.append(cfg), sleep=sleeps.append)
    assert calls == [(TEST_CYCLE, None), (TEST_CYCLE, "report")] and sleeps == [60, 120]
    assert seen == [loaded[2]] * 2  # the config read after the bad one, never the first


def test_a_real_cycle_failing_until_its_window_is_too_short_is_given_up_for_the_open_one(tmp_path, monkeypatch,
                                                                                         caplog):
    config = _config(tmp_path)
    phase = SimpleNamespace(block=DEADLINE - 1950, cycle_index=OPEN * 13 + 5)  # 6.5 h of its window left
    monkeypatch.setattr(run, "get_phase_from_api", lambda cfg: phase)
    monkeypatch.setattr(run, "check_phase_clock", lambda cfg, p: p)
    monkeypatch.setattr(run.PromptValidatorConfig, "from_path", lambda path, active_task: config)
    sleeps, calls = [], []

    def sleep(sec):
        sleeps.append(sec)
        if len(sleeps) > 20:
            pytest.fail("the loop kept retrying")
        phase.block += int(sec // 12)

    def stages(cfg, cycle_id, name=None):
        calls.append(cycle_id)
        if calls.count(OPEN) == 2:
            raise Stop
        raise RuntimeError("cycle-api /get_phase or /blocks_until_next_phase unreachable")

    monkeypatch.setattr(run, "run_stages", stages)
    monkeypatch.setattr(run, "time", SimpleNamespace(sleep=sleep))
    monkeypatch.setattr("sys.argv", ["run", "--path", "x.yaml", "--loop"])
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with caplog.at_level(logging.WARNING), pytest.raises(Stop):
        run.main()
    # 1 + 2 + 4 + 8 + 16 = 31 min of retries leave under 6 h: the open cycle is next, and OPEN - 1 is never retaken.
    # The open cycle's first failure starts the backoff again.
    assert calls == [OPEN - 1] * 5 + [OPEN] * 2 and sleeps == [60 * m for m in (1, 2, 4, 8, 16, 1)]
    assert caplog.text.count("cycle skipped: too little of its run-start window is left") == 2  # once a pick


def test_a_test_cycle_failing_until_the_owner_closes_the_next_is_given_up_for_it(tmp_path, looping, monkeypatch):
    sleeps = []

    def pick(self, done):
        return TEST_CYCLE + (len(sleeps) >= 3)  # the owner closes the next test cycle after the third retry

    monkeypatch.setattr(FakeTestCycles, "next_closed_test_cycle", pick)

    def stage(cfg, cycle_id, name):
        if cycle_id == TEST_CYCLE:
            raise RuntimeError("Jev spend cap $5.0 reached")

    calls, _ = looping(lambda: _config(tmp_path, test_mode=True), cycles=2, stage=stage, sleep=sleeps.append)
    assert calls == [(TEST_CYCLE, None)] * 3 + [(TEST_CYCLE + 1, None), (TEST_CYCLE + 1, "report")]


def test_sigterm_during_the_backoff_exits_143(tmp_path, looping):
    with pytest.raises(SystemExit) as exit_:
        looping(lambda: _config(tmp_path, test_mode=True), stage=_failing(1),
                sleep=lambda sec: os.kill(os.getpid(), signal.SIGTERM))
    assert exit_.value.code == 143


def test_a_cycle_a_manual_run_holds_is_retried_once_it_is_free(tmp_path, looping, capsys, caplog):
    held = ExitStack()
    held.enter_context(run_lock(tmp_path / str(TEST_CYCLE) / ".lock", "a manual report"))
    sleeps = []

    def sleep(sec):
        sleeps.append(sec)
        held.close()

    calls, _ = looping(lambda: _config(tmp_path, test_mode=True), cycles=1, sleep=sleep)
    assert calls == [(TEST_CYCLE, None), (TEST_CYCLE, "report")] and sleeps == [60]
    assert f"scoring cycle {TEST_CYCLE} is run by another process" in capsys.readouterr().out + caplog.text


def test_the_retry_log_has_the_error_a_plain_traceback_and_no_local(tmp_path, looping, capsys, caplog):
    def stage(cfg, cycle_id, name):
        if name is None and not (tmp_path / "failed").exists():
            headers = {"Authorization": "Bearer jev-" + "key-1234"}  # noqa: F841 - a local a rich traceback prints
            (tmp_path / "failed").touch()
            raise RuntimeError("Jev call failed (see the error logged above)")

    with caplog.at_level(logging.WARNING):
        looping(lambda: _config(tmp_path, test_mode=True), cycles=1, stage=stage, sleep=lambda sec: None)
    logged = capsys.readouterr().out + caplog.text
    assert "cycle failed, will retry" in logged and "RuntimeError: Jev call failed" in logged
    assert "Traceback (most recent call last)" in logged and "jev-key-1234" not in logged


def test_a_plain_run_still_stops_on_a_stage_error(tmp_path, monkeypatch):
    monkeypatch.setattr(run.PromptValidatorConfig, "from_path", lambda path, active_task: _config(tmp_path))
    monkeypatch.setattr(run, "run_stages", _failing(1))
    monkeypatch.setattr(run, "time", SimpleNamespace(sleep=lambda sec: pytest.fail("a plain run retried")))
    monkeypatch.setattr("sys.argv", ["run", "--path", "x.yaml", "--cycle-id", "5"])
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(RuntimeError, match="ConnectionError"):
        run.main()


def test_the_backoff_sleeps_once_the_failed_attempt_is_unwound(tmp_path, looping):
    handling = []
    looping(lambda: _config(tmp_path, test_mode=True), cycles=1, stage=_failing(1),
            sleep=lambda sec: handling.append(sys.exc_info()[0]))
    assert handling == [None]  # no exception, and so no frame of the failed attempt, is held while it sleeps


def test_a_failed_pick_between_two_failures_of_one_cycle_does_not_start_the_backoff_again(tmp_path, looping,
                                                                                          monkeypatch):
    sleeps = []
    real_pick = FakeTestCycles.next_closed_test_cycle

    def pick(self, done):
        if len(sleeps) == 1:
            raise RuntimeError("dataless-api GET /v1/test/prompt-cycle failed: ConnectionError")
        return real_pick(self, done)

    monkeypatch.setattr(FakeTestCycles, "next_closed_test_cycle", pick)
    looping(lambda: _config(tmp_path, test_mode=True), cycles=1, stage=_failing(2), sleep=sleeps.append)
    assert sleeps == [60, 120, 240]  # TEST_CYCLE, the pick, TEST_CYCLE again


def test_a_jev_version_other_than_the_pin_is_not_retried_and_exits_78(tmp_path, looping, capsys, caplog):
    def stage(cfg, cycle_id, name):
        raise JevVersionError("Jev resolved 'jev-1.15.0', pinned 'jev-1.14.0'")

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exit_:
        looping(lambda: _config(tmp_path, test_mode=True), stage=stage, sleep=lambda sec: pytest.fail("retried"))
    assert exit_.value.code == 78
    logged = capsys.readouterr().out + caplog.text
    assert "requested by name: check Jev" in logged and "'jev-1.15.0', pinned 'jev-1.14.0'" in logged


def test_a_hotkey_off_test_hotkeys_is_not_retried_and_exits_78(tmp_path, looping, capsys, caplog):
    def stage(cfg, cycle_id, name):
        raise TesterRefused(403, "role_forbidden", "hotkey 5Validator refused on GET /v1/test/prompt-cycle: "
                             "This hotkey is not on the test allowlist (TEST_HOTKEYS).")

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exit_:
        looping(lambda: _config(tmp_path, test_mode=True), stage=stage, sleep=lambda sec: pytest.fail("retried"))
    assert exit_.value.code == 78
    logged = capsys.readouterr().out + caplog.text
    assert "ask the owner to fix it, then restart" in logged
    assert "403 role_forbidden: hotkey 5Validator refused on GET /v1/test/prompt-cycle: This hotkey is not" in logged


def test_a_reload_onto_a_prompt_file_stops_the_loop_with_exit_2_and_no_retry(tmp_path, looping, capsys):
    loads, seen, slept = [], [], []

    def make_config():
        loads.append(1)
        return _config(tmp_path, test_mode=True) if len(loads) == 1 else _config(tmp_path, source="file")

    with pytest.raises(SystemExit) as exit_:
        looping(make_config, cycles=2, stage=lambda cfg, cycle_id, name: seen.append((cycle_id, name)),
                sleep=slept.append)
    assert exit_.value.code == 2 and 'it needs prompt_source "dataless"' in capsys.readouterr().err
    assert seen == [(TEST_CYCLE, None), (TEST_CYCLE, "report")] and len(loads) == 2 and slept == []


def test_a_cycle_past_reward_whose_report_keeps_failing_is_dropped_when_the_next_cycle_opens(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _mark(config, OPEN - 1, "cutoff", "reward")
    phase = SimpleNamespace(block=OPEN * LENGTH + 2000, cycle_index=(OPEN * LENGTH + 2000) // 524)
    monkeypatch.setattr(run, "get_phase_from_api", lambda cfg: phase)
    monkeypatch.setattr(run, "check_phase_clock", lambda cfg, p: p)
    monkeypatch.setattr(run.PromptValidatorConfig, "from_path", lambda path, active_task: config)
    sleeps, calls = [], []

    def sleep(sec):
        sleeps.append(sec)
        if len(sleeps) > 60:
            pytest.fail("the loop kept retrying")
        phase.block += int(sec // 12)
        phase.cycle_index = phase.block // 524  # cycle-api's clock, consistent with the block

    def stages(cfg, cycle_id, name=None):
        calls.append((cycle_id, name, phase.cycle_index // 13))
        if cycle_id == OPEN:
            raise Stop
        if name == "report":
            raise RuntimeError("no model of scoring cycle 1258 can be reported: nothing to publish")

    monkeypatch.setattr(run, "run_stages", stages)
    monkeypatch.setattr(run, "time", SimpleNamespace(sleep=sleep))
    monkeypatch.setattr("sys.argv", ["run", "--path", "x.yaml", "--loop"])
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(Stop):
        run.main()
    before, after = calls[:-1], calls[-1]
    # Retried on every attempt while OPEN is the open cycle, then OPEN once cycle-api rolls over; never OPEN - 1 again.
    assert {(c, open_) for c, _, open_ in before} == {(OPEN - 1, OPEN)} and len(before) == 2 * len(sleeps)
    assert after == (OPEN, None, OPEN + 1)
    assert sleeps[:6] == [60 * m for m in (1, 2, 4, 8, 16, 30)]
