import errno
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from connito.prompt_validator import run, stage_io
from connito.prompt_validator.stage_io import LockHeld, run_lock
from connito.test.test_prompt_validator_loop import _config


def test_a_held_lock_stops_a_second_holder_at_once_naming_the_pid_and_the_file(tmp_path):
    path = tmp_path / "7" / ".lock"
    with run_lock(path, "scoring cycle 7"):
        with pytest.raises(LockHeld, match=rf"scoring cycle 7 is run by another process \(pid {os.getpid()}, lock "):
            with run_lock(path, "scoring cycle 7"):
                pytest.fail("two holders")
    with run_lock(path, "scoring cycle 7"):  # free again once released
        pass
    assert path.exists() and stat.S_IMODE(path.stat().st_mode) == 0o600  # never removed: removing it splits the lock


def test_a_lock_the_filesystem_cannot_give_raises_its_error_and_leaves_no_fd_open(tmp_path, monkeypatch):
    def no_locks(fd, op):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(stage_io.fcntl, "flock", no_locks)
    before = len(os.listdir("/dev/fd"))
    with pytest.raises(OSError, match="No locks available") as raised:
        with run_lock(tmp_path / ".lock", "x"):
            pytest.fail("locked")
    assert not isinstance(raised.value, LockHeld) and len(os.listdir("/dev/fd")) == before


def test_a_holder_killed_with_sigkill_leaves_the_lock_free(tmp_path):
    path = tmp_path / ".lock"
    holder = subprocess.Popen(
        [sys.executable, "-c", "import fcntl, os, sys, time\n"
         f"fd = os.open({str(path)!r}, os.O_RDWR | os.O_CREAT, 0o600)\n"
         "fcntl.flock(fd, fcntl.LOCK_EX)\nos.write(fd, b'held')\nprint('held', flush=True)\ntime.sleep(60)\n"],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(LockHeld, match="pid held"):
            with run_lock(path, "x"):
                pass
    finally:
        holder.send_signal(signal.SIGKILL)
        holder.wait()
    with run_lock(path, "x"):
        assert path.read_text() == f"{os.getpid()}\n"


@pytest.fixture
def main_with(tmp_path, monkeypatch):
    """Run main() with args on a config whose runs_dir is tmp_path; returns the run_stages calls."""
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(run.PromptValidatorConfig, "from_path", lambda path, active_task: _config(tmp_path))
    calls = []
    monkeypatch.setattr(run, "run_stages", lambda cfg, cycle_id, name=None: calls.append((cycle_id, name)))
    monkeypatch.setattr(run, "next_cycle", lambda cfg: pytest.fail("the loop went on past a held lock"))

    def start(*args):
        monkeypatch.setattr("sys.argv", ["run", "--path", str(tmp_path / "x.yaml"), *args])
        run.main()
        return calls

    return start


def test_a_second_loop_on_one_runs_dir_exits_75_and_says_why(tmp_path, main_with, capsys, caplog):
    with run_lock(tmp_path / ".loop.lock", "the first loop"):
        with pytest.raises(SystemExit) as stopped:
            main_with("--loop")
    assert stopped.value.code == 75
    assert f"the --loop on {tmp_path} is run by another process" in capsys.readouterr().out + caplog.text


def test_a_run_of_a_cycle_another_process_holds_exits_75_and_another_cycle_runs(tmp_path, main_with, capsys,
                                                                                 caplog):
    with run_lock(Path(tmp_path) / "5" / ".lock", "the loop's cycle"):
        with pytest.raises(SystemExit) as stopped:
            main_with("--cycle-id", "5", "--stage", "report")
        assert stopped.value.code == 75
        assert "scoring cycle 5 is run by another process" in capsys.readouterr().out + caplog.text
        assert main_with("--cycle-id", "6") == [(6, None)]
