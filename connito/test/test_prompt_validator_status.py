import json
import logging
import os
import signal
import stat

import pytest

from connito.prompt_validator import run, status
from connito.prompt_validator.jev import JevVersionError
from connito.prompt_validator.stage_io import append_published, last_published, stage_path, write_rows
from connito.prompt_validator.status import loop_status, set_status, stage_counts
from connito.test.test_prompt_validator_loop import TEST_CYCLE, _config, looping  # noqa: F401 - the fixture

SENTINEL = "upstream said: prompt text sk-sentinel"


def _status(runs_dir) -> dict:
    return json.loads((runs_dir / "status.json").read_text())


# --- the summary line ---

def test_each_stage_has_its_key_counts(tmp_path):
    path = stage_path(tmp_path, 7, "scores")
    write_rows(path.parent / "jev_calls.jsonl", [{"cost": 0.25}, {"cost": 0.125}])

    assert stage_counts("cutoff", [{"commit_1_block": 1, "commit_2_block": 9}], path) == {"commit_2_block": 9}
    fetched = [{"reason": None}, {"reason": "hash"}, {"reason": "hash"}, {"reason": "timeout"}]
    assert stage_counts("fetch", fetched, path) == {"fetched": 1, "reasons": {"hash": 2, "timeout": 1}}
    answers = [{"model_hash": "a", "finish_reason": "stop"}, {"model_hash": "a", "finish_reason": "length"},
               {"model_hash": "b", "finish_reason": "stop"}]
    assert stage_counts("answers", answers, path) == {"answers": 3, "models": 2, "cut_at_max": 1}
    assert stage_counts("scores", [{}] * 5, path) == {"jev_calls": 2, "spent_usd": 0.375}
    assert stage_counts("reward", [{"weight": 0.5}, {"weight": 0.5}, {"weight": 0.0}], path) == {
        "reward_rows": 3, "weighted": 2}
    assert stage_counts("report", [{"status": "published", "models": 3}], path) == {
        "report_status": "published", "posted": True}
    assert stage_counts("report", [{"status": "published", "published_before": "ledger"}], path)["posted"] is False


def test_counts_that_cannot_be_made_never_fail_a_finished_stage(tmp_path):
    path = stage_path(tmp_path, 7, "scores")
    path.parent.mkdir(parents=True)
    (path.parent / "jev_calls.jsonl").write_text('{"cost": 0.1}\n{"co')  # cut short by a crash

    assert stage_counts("scores", [], path) == {"counts_error": "JSONDecodeError"}


def _stages(monkeypatch, seen):
    """Two stages that record runs/status.json as they find it."""
    def stage(rows):
        def run_stage(config, cycle_id):
            path = config.prompt_validator.runs_dir / "status.json"
            seen.append(json.loads(path.read_text()) if path.exists() else None)
            return rows
        return run_stage

    monkeypatch.setattr(run, "STAGES", [("cutoff", stage([{"commit_2_block": 9}])),
                                        ("fetch", stage([{"reason": None}, {"reason": "hash"}]))])


def test_each_stage_logs_one_summary_line_and_the_loop_status_follows_the_stages(tmp_path, monkeypatch, capsys,
                                                                                 caplog):
    config, seen = _config(tmp_path), []
    _stages(monkeypatch, seen)

    with caplog.at_level(logging.INFO), loop_status(tmp_path):
        run.run_stages(config, 7)
        after = _status(tmp_path)

    assert [(s["state"], s["stage"], s["cycle"]) for s in seen] == [("waiting_for_cutoff", "cutoff", 7),
                                                                    ("running", "fetch", 7)]
    assert all(s["pid"] == os.getpid() and s["started_at"] and s["process_started_at"] for s in seen)
    assert seen[1]["last_stage"] == {"stage": "cutoff", "scoring_cycle_id": 7, "duration_s": 0, "rows": 1,
                                     "commit_2_block": 9}
    assert after["last_stage"] | {"duration_s": 0} == {"stage": "fetch", "scoring_cycle_id": 7, "duration_s": 0,
                                                       "rows": 2, "fetched": 1, "reasons": {"hash": 1}}
    lines = [line for line in (capsys.readouterr().out + caplog.text).splitlines() if "stage done" in line]
    assert len(lines) == 2 and all("duration_s" in line and "scoring_cycle_id" in line for line in lines)
    assert "fetched" in lines[1] and "reasons" in lines[1]
    assert _status(tmp_path)["state"] == "stopped"


def test_a_run_without_the_loop_writes_no_status(tmp_path, monkeypatch):
    seen = []
    _stages(monkeypatch, seen)
    monkeypatch.setattr(run.PromptValidatorConfig, "from_path", lambda path, active_task: _config(tmp_path))
    monkeypatch.setattr("sys.argv", ["run", "--path", "x.yaml", "--cycle-id", "7"])
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    run.main()  # as a manual run beside the loop: the loop's file is its own
    assert seen == [None, None] and not (tmp_path / "status.json").exists()


def test_the_status_is_written_whole_and_owner_only(tmp_path, monkeypatch):
    replaced = []
    real_replace = os.replace
    monkeypatch.setattr("connito.prompt_validator.stage_io.os.replace",
                        lambda src, dst: (replaced.append((src, dst)), real_replace(src, dst)))

    with loop_status(tmp_path):
        set_status(state="running", cycle=7, stage="answers")

    assert replaced and all(str(dst) == str(tmp_path / "status.json") for _, dst in replaced)
    assert stat.S_IMODE((tmp_path / "status.json").stat().st_mode) == 0o600
    assert [p.name for p in tmp_path.iterdir()] == ["status.json"]  # no tmp file left


def test_started_at_moves_with_the_state_or_stage_and_not_with_progress(tmp_path, monkeypatch):
    times = iter(f"2026-10-19T00:00:{s:02d}+00:00" for s in range(60))
    monkeypatch.setattr(status, "_now", lambda: next(times))

    with loop_status(tmp_path):
        set_status(state="running", cycle=7, stage="answers")
        started = _status(tmp_path)["started_at"]
        set_status(progress="20/60", eta_min=3.0)
        assert _status(tmp_path)["started_at"] == started and _status(tmp_path)["progress"] == "20/60"
        set_status(state="running", cycle=7, stage="scores")
        assert _status(tmp_path)["started_at"] != started


def test_a_status_that_cannot_be_written_is_logged_and_the_loop_goes_on(tmp_path, looping, monkeypatch, capsys,  # noqa: F811
                                                                        caplog):
    def refused(path, rows):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(status, "write_rows", refused)
    with caplog.at_level(logging.WARNING):
        calls, _ = looping(lambda: _config(tmp_path, test_mode=True), cycles=1)
    assert calls == [(TEST_CYCLE, None), (TEST_CYCLE, "report")]
    assert "status not written" in capsys.readouterr().out + caplog.text


# --- the loop ---

def test_between_cycles_the_loop_is_sleeping_and_counts_what_it_published(tmp_path, looping):  # noqa: F811
    for cycle_id in (5, 9):
        append_published(tmp_path, {"cycle_id": cycle_id})
    seen = []

    looping(lambda: _config(tmp_path, test_mode=True), cycles=2,
            stage=lambda cfg, cycle_id, name: seen.append(_status(tmp_path)))
    assert [(s["state"], s["cycle"], s["published"], s["last_published"]) for s in seen] == [
        ("sleeping", None, 2, 9)] * 4  # the faked stages, report included, set no status
    final = _status(tmp_path)
    assert final["state"] == "stopped" and final["pid"] == os.getpid() and final["process_started_at"]


def test_last_published_is_the_latest_publish_not_the_highest_cycle_id(tmp_path, looping):  # noqa: F811
    # A test cycle's id is above every real one: published first, it must not stay "last" after a real one.
    append_published(tmp_path, {"cycle_id": TEST_CYCLE, "published_at": "2026-10-18T09:00:00Z"})  # the server's form
    append_published(tmp_path, {"cycle_id": 1368, "published_at": "2026-10-19T09:00:00+00:00"})  # the local form
    seen = []
    looping(lambda: _config(tmp_path, test_mode=True), cycles=1,
            stage=lambda cfg, cycle_id, name: seen.append(_status(tmp_path)))
    assert {(s["published"], s["last_published"]) for s in seen} == {(2, 1368)}


@pytest.mark.parametrize(("rows", "last"), [
    ([(TEST_CYCLE, "2026-10-19T09:00:00Z"), (1368, "2026-10-18T09:00:00Z")], TEST_CYCLE),  # by time, not row order
    ([(TEST_CYCLE, None), (1368, None)], 1368),  # no time: row order
    ([(1368, "2026-10-19T09:00:00Z"), (1369, "2026-10-19T09:00:00Z")], 1369),  # a tie: row order
    ([(1368, "2026-10-19T09:00:00Z"), (TEST_CYCLE, "garbled")], TEST_CYCLE),  # unreadable: its place in the order
    ([(1368, "2026-10-19T09:00:00Z"), (TEST_CYCLE, "garbled"), (1367, "2026-10-18T09:00:00Z")], TEST_CYCLE),
    ([(1368, "2026-10-19T10:00:00"), (1369, "2026-10-19T09:00:00Z")], 1368),  # no zone: UTC
    # The dataless-api's updated_at as it sends it (pydantic's JSON of a Postgres timestamptz): microseconds and a Z.
    ([(1368, "2026-10-19T12:00:00.654321Z"), (1369, "2026-10-19T12:00:00.123456Z")], 1368),
    ([(1368, "2026-10-19T12:00:00.654321Z"), (1369, "2026-10-19T12:00:01+00:00")], 1369),  # then a local time
    ([], None),
])
def test_last_published_goes_by_published_at_then_row_order(tmp_path, rows, last):
    for cycle_id, at in rows:
        append_published(tmp_path, {"cycle_id": cycle_id} | ({"published_at": at} if at else {}))
    assert last_published(tmp_path) == last


def test_a_failed_cycle_is_retrying_with_its_error_type_and_never_its_message(tmp_path, looping):  # noqa: F811
    failures, seen = [1], []

    def stage(cfg, cycle_id, name):
        if failures[0]:
            failures[0] -= 1
            raise RuntimeError(SENTINEL)

    looping(lambda: _config(tmp_path, test_mode=True), cycles=1, stage=stage,
            sleep=lambda sec: seen.append((tmp_path / "status.json").read_text()))
    retrying = json.loads(seen[0])
    assert (retrying["state"], retrying["cycle"], retrying["last_error"], retrying["retry_in_min"]) == (
        "retrying", TEST_CYCLE, "RuntimeError", 1)
    assert "sk-sentinel" not in seen[0] and "sk-sentinel" not in (tmp_path / "status.json").read_text()
    assert _status(tmp_path)["last_error"] == "RuntimeError"  # kept after the cycle recovers; its time says when


def test_retrying_and_sleeping_carry_no_progress_from_the_attempt_before(tmp_path, looping):  # noqa: F811
    failures, seen = [1], []

    def stage(cfg, cycle_id, name):
        seen.append(_status(tmp_path))
        if failures[0]:
            failures[0] -= 1
            set_status(progress="40/60", eta_min=12.5)  # as answers leaves it when its model then fails
            raise RuntimeError("CUDA out of memory")

    looping(lambda: _config(tmp_path, test_mode=True), cycles=1, stage=stage,
            sleep=lambda sec: seen.append(_status(tmp_path)))
    assert [(s["state"], s["progress"], s["eta_min"]) for s in seen] == [
        ("sleeping", None, None), ("retrying", None, None), ("sleeping", None, None), ("sleeping", None, None)]


def test_a_jev_version_mismatch_leaves_the_status_stopped_on_it(tmp_path, looping):  # noqa: F811
    def stage(cfg, cycle_id, name):
        raise JevVersionError("Jev resolved 'jev-1.15.0', pinned 'jev-1.14.0'")

    with pytest.raises(SystemExit):
        looping(lambda: _config(tmp_path, test_mode=True), stage=stage)
    final = _status(tmp_path)
    assert (final["state"], final["last_error"]) == ("stopped", "JevVersionError") and "jev-1.15" not in str(final)


def test_sigterm_leaves_the_status_stopped_with_the_ledger_read_again_and_no_cycle_under_way(tmp_path,
                                                                                          looping):  # noqa: F811
    def terminated(cfg, cycle_id, name):
        set_status(state="running", cycle=cycle_id, stage="answers", progress="1/6")
        append_published(tmp_path, {"cycle_id": 1368})  # a manual report beside the loop
        os.kill(os.getpid(), signal.SIGTERM)

    with pytest.raises(SystemExit):
        looping(lambda: _config(tmp_path, test_mode=True), stage=terminated)
    final = _status(tmp_path)
    assert (final["state"], final["cycle"], final["stage"], final["progress"]) == ("stopped", None, None, None)
    assert (final["published"], final["last_published"]) == (1, 1368)


def test_a_ledger_that_cannot_be_read_still_leaves_the_status_stopped(tmp_path, monkeypatch, capsys, caplog):
    monkeypatch.setattr(status, "published_cycles", lambda runs_dir: 1 / 0)
    with caplog.at_level(logging.WARNING), loop_status(tmp_path):
        set_status(state="sleeping", published=4)
    assert (_status(tmp_path)["state"], _status(tmp_path)["published"]) == ("stopped", 4)
    assert "ledger not read for the status" in capsys.readouterr().out + caplog.text
