"""The vLLM worker and run_worker, against a fake vLLM python: env, privacy, results, heartbeats, stalls and kills."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from connito.prompt_validator import answers
from connito.prompt_validator.answers import cut_at_eos, eos_ids

SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts" / "prompt_validator_vllm")
sys.path.insert(0, SCRIPTS)
from connito_vllm import worker  # noqa: E402

# Stands in for connito_vllm.worker. FAKE_MODE picks what it does; FAKE_LOG gets its env, pid and its child's pid.
FAKE = """#!{python}
import json, os, signal, subprocess, sys, time
job_path, work = sys.argv[3:5]
job = json.load(open(job_path))
mode, beat_s, beats = os.environ["FAKE_MODE"], float(os.environ.get("BEAT_S", 0)), int(os.environ.get("BEATS", 0))
def write(name, row):
    with open(os.path.join(work, name + ".tmp"), "w") as f:
        json.dump(row, f)
    os.replace(os.path.join(work, name + ".tmp"), os.path.join(work, name))
def result(h):
    write(h + ".json", {{"run_id": job["run_id"], "answers": []}})
stale = os.path.exists(os.path.join(work, "progress.json"))
if mode == "result_then_beat":
    result(job["models"][0]["model_hash"])
    write("progress.json", {{"run_id": job["run_id"], "model_hash": job["models"][1]["model_hash"], "step": "generate",
                             "done": 1}})
    time.sleep(0.3)
    result(job["models"][1]["model_hash"])
if mode == "foreign":
    write("progress.json", {{"run_id": "another run", "model_hash": "a", "step": "generate", "done": 1}})
child = subprocess.Popen(["sleep", "60"]) if mode in ("hang", "first_then_hang", "beat_then_stop", "selfkill",
                                                      "foreign") else None
log = os.environ["FAKE_LOG"]
json.dump({{"env": dict(os.environ), "pid": os.getpid(), "child": child and child.pid, "stale": stale}},
          open(log + ".tmp", "w"))
os.replace(log + ".tmp", log)
if mode == "selfkill":
    os.kill(os.getpid(), signal.SIGKILL)
if mode in ("answer", "first_then_hang"):
    for m in job["models"][:1 if mode == "first_then_hang" else None]:
        result(m["model_hash"])
for n in range(beats):
    time.sleep(float(os.environ.get("FIRST_BEAT_S", beat_s)) if n == 0 else beat_s)
    write("progress.json", {{"run_id": job["run_id"], "model_hash": job["models"][0]["model_hash"],
                             "step": "generate", "done": n + 1}})
if mode == "beat":
    for m in job["models"]:
        result(m["model_hash"])
if child:
    time.sleep(60)
"""


@pytest.fixture
def fake(tmp_path, monkeypatch):
    python = tmp_path / "python"
    python.write_text(FAKE.format(python=sys.executable))
    python.chmod(0o755)
    monkeypatch.setattr(answers, "_POLL_S", 0.01)
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "log.json"))
    for name in ("HF_TOKEN", "JEV_DIRECT_API_KEY", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY", "MY_HF_TOKEN",
                 "BT_PW_SOME_WALLET"):
        monkeypatch.setenv(name, "secret")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    return python


def _job(*hashes) -> dict:
    return {"models": [{"model_hash": h} for h in hashes]}


def _log(tmp_path) -> dict:
    return json.loads((tmp_path / "log.json").read_text())


def _gone(*pids: int, within_s: float = 10.0) -> bool:
    """Every pid has exited and been reaped."""
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except ProcessLookupError:
                pass
        if not alive:
            return True
        time.sleep(0.05)
    return False


def test_the_worker_runs_without_secrets_in_an_owner_only_dir(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "answer")
    got = {}
    answers.run_worker(fake, _job("a", "b"), tmp_path / "work", got.__setitem__, secrets=("MY_HF_TOKEN",))
    assert sorted(got) == ["a", "b"]
    env = _log(tmp_path)["env"]
    assert not {"HF_TOKEN", "JEV_DIRECT_API_KEY", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY", "MY_HF_TOKEN"} & set(env)
    assert not any(k.startswith("BT_PW_") for k in env)
    assert (env["CUDA_VISIBLE_DEVICES"], env["HF_HUB_OFFLINE"]) == ("3", "1")
    assert env["CONNITO_VLLM_PARENT"] == str(os.getpid())
    assert (tmp_path / "work").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "work" / "job.json").stat().st_mode & 0o777 == 0o600


# The caches the run's start check reads (config.cache_dirs): the worker must write to the same ones.
CACHE_ENV = ("VLLM_CACHE_ROOT", "VLLM_CONFIG_ROOT", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "TMPDIR", "TRITON_CACHE_DIR",
             "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH")


def test_the_cache_dirs_the_start_check_reads_reach_the_worker_unchanged(fake, tmp_path, monkeypatch):
    caches = {name: str(tmp_path / "scratch" / name) for name in CACHE_ENV}
    for name, path in caches.items():
        Path(path).mkdir(parents=True)
        monkeypatch.setenv(name, path)
    monkeypatch.setenv("FAKE_MODE", "answer")
    answers.run_worker(fake, _job("a"), tmp_path / "work", lambda *_: None)
    env = _log(tmp_path)["env"]
    assert {name: env.get(name) for name in CACHE_ENV} == caches


def test_a_result_left_by_an_earlier_run_is_removed_and_never_taken(fake, tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    (work / "a.json").write_text(json.dumps({"run_id": "an earlier run", "answers": [["stale"]]}))
    monkeypatch.setenv("FAKE_MODE", "silent")
    got = {}
    with pytest.raises(RuntimeError, match=r"exited without answering models \['a'\]"):
        answers.run_worker(fake, _job("a"), work, got.__setitem__)
    assert got == {} and not (work / "a.json").exists()


def test_the_worker_refuses_to_start_without_one_pinned_gpu(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "answer")
    for name, value in (("CUDA_VISIBLE_DEVICES", "1,2"), ("CUDA_VISIBLE_DEVICES", ""),
                        ("CUDA_DEVICE_ORDER", "FASTEST_FIRST")):
        with monkeypatch.context() as m:
            m.setenv(name, value)
            with pytest.raises(RuntimeError, match="naming one GPU"):
                answers.run_worker(fake, _job("a"), tmp_path / "work", lambda h, r: None)


# --- heartbeats and stalls ---------------------------------------------------------


def test_steady_heartbeats_keep_a_slow_worker_alive_past_both_windows(fake, tmp_path, monkeypatch):
    # 2 s of work at a heartbeat per 0.2 s: longer than the start (0.5 s) and stall (0.5 s) windows together.
    monkeypatch.setenv("FAKE_MODE", "beat")
    monkeypatch.setenv("BEAT_S", "0.2")
    monkeypatch.setenv("BEATS", "10")
    got, beats = {}, []
    answers.run_worker(fake, _job("a"), tmp_path / "work", got.__setitem__, on_progress=lambda h, n: beats.append(n),
                       start_timeout_s=0.5, stall_timeout_s=0.5)
    assert list(got) == ["a"] and beats and beats[-1] == 10


def test_after_the_first_heartbeat_the_stall_window_applies_not_the_start_one(fake, tmp_path, monkeypatch):
    # A first heartbeat inside the 0.3 s start window, then 0.6 s apart: past it, but inside the 1.5 s stall window.
    monkeypatch.setenv("FAKE_MODE", "beat")
    monkeypatch.setenv("FIRST_BEAT_S", "0.1")
    monkeypatch.setenv("BEAT_S", "0.6")
    monkeypatch.setenv("BEATS", "4")
    got = {}
    answers.run_worker(fake, _job("a"), tmp_path / "work", got.__setitem__, start_timeout_s=0.3, stall_timeout_s=1.5)
    assert list(got) == ["a"]


def test_a_heartbeat_of_another_run_or_left_from_one_never_counts(fake, tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    (work / "progress.json").write_text(json.dumps({"run_id": "an earlier run", "model_hash": "a",
                                                    "step": "generate", "done": 5}))
    monkeypatch.setenv("FAKE_MODE", "foreign")
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="finished no request"):
        answers.run_worker(fake, _job("a"), work, lambda h, r: None, start_timeout_s=0.5, stall_timeout_s=30)
    assert time.monotonic() - started < 0.5 + 2.0  # the start window, not the stall one
    assert _log(tmp_path)["stale"] is False  # removed before the worker started


def test_a_result_is_handed_over_before_a_heartbeat_already_on_the_next_model(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "result_then_beat")
    calls = []
    answers.run_worker(fake, _job("a", "b"), tmp_path / "work", lambda h, r: calls.append(("result", h)),
                       on_progress=lambda h, n: calls.append(("progress", h)))
    assert calls.index(("result", "a")) < calls.index(("progress", "b"))


def test_a_worker_whose_heartbeats_stop_is_killed_about_one_stall_window_later(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "beat_then_stop")
    monkeypatch.setenv("BEAT_S", "0.2")
    monkeypatch.setenv("BEATS", "3")
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="finished no request"):
        answers.run_worker(fake, _job("a"), tmp_path / "work", lambda h, r: None, start_timeout_s=5,
                           stall_timeout_s=0.5)
    assert time.monotonic() - started < 0.6 + 0.5 + 1.5  # three beats, one stall window, and slack
    assert _gone(_log(tmp_path)["pid"], _log(tmp_path)["child"])


@pytest.mark.parametrize("mode, got", [("hang", []), ("first_then_hang", ["a"])], ids=["at start", "after a result"])
def test_a_worker_that_stops_answering_is_killed_and_raises(fake, tmp_path, monkeypatch, mode, got):
    monkeypatch.setenv("FAKE_MODE", mode)
    seen = []
    with pytest.raises(TimeoutError, match="finished no request"):
        answers.run_worker(fake, _job("a", "b"), tmp_path / "work", lambda h, r: seen.append(h),
                           start_timeout_s=0.5 if mode == "hang" else 30, stall_timeout_s=0.5)
    assert seen == got
    assert _gone(_log(tmp_path)["pid"], _log(tmp_path)["child"])


# --- the whole process group goes, on every path ------------------------------------


def test_an_interrupt_kills_the_workers_whole_process_group(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "hang")
    log = tmp_path / "log.json"
    real_sleep = time.sleep

    def interrupt(seconds):
        if log.exists():
            raise KeyboardInterrupt
        real_sleep(0.01)

    monkeypatch.setattr(answers.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        answers.run_worker(fake, _job("a"), tmp_path / "work", lambda h, r: None)
    monkeypatch.undo()
    assert _gone(_log(tmp_path)["pid"], _log(tmp_path)["child"])


def test_an_error_while_handling_a_result_kills_the_worker_and_its_engine(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "first_then_hang")

    def fail(model_hash, result):
        while not (tmp_path / "log.json").exists():
            time.sleep(0.01)
        raise ValueError("cannot save")

    with pytest.raises(ValueError, match="cannot save"):
        answers.run_worker(fake, _job("a", "b"), tmp_path / "work", fail)
    assert _gone(_log(tmp_path)["pid"], _log(tmp_path)["child"])


def test_a_worker_that_dies_leaves_no_engine_behind(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "selfkill")  # as the OOM killer would: its child stays in the group
    with pytest.raises(RuntimeError, match="exit status -9"):
        answers.run_worker(fake, _job("a"), tmp_path / "work", lambda h, r: None)
    assert _gone(_log(tmp_path)["child"])


def test_the_real_worker_kills_its_group_when_its_parent_is_already_gone(tmp_path):
    reaped = subprocess.Popen(["true"])
    reaped.wait()
    env = {**os.environ, "PYTHONPATH": SCRIPTS, "CONNITO_VLLM_PARENT": str(reaped.pid)}
    proc = subprocess.run([sys.executable, "-m", "connito_vllm.worker", str(tmp_path / "job.json"), str(tmp_path)],
                          env=env, start_new_session=True, timeout=60)
    assert proc.returncode == -signal.SIGKILL


# A stand-in stage process: it starts a worker in its own session, as run_worker does, and the worker an engine.
WORKER = """
import os, subprocess, sys, time
sys.path.insert(0, sys.argv[1])
from connito_vllm import worker
worker.exit_with_parent(os.getppid(), 0.05)
engine = subprocess.Popen(["sleep", "60"])
open(sys.argv[2], "w").write(f"{os.getpid()} {engine.pid}")
time.sleep(60)
"""
PARENT = """
import subprocess, sys, time
subprocess.Popen([sys.executable, "-c", sys.argv[1], *sys.argv[2:]], start_new_session=True)
time.sleep(60)
"""


def test_the_worker_and_its_engine_die_with_a_killed_parent(tmp_path):
    out = tmp_path / "pids"
    parent = subprocess.Popen([sys.executable, "-c", PARENT, WORKER, SCRIPTS, str(out)])
    deadline = time.monotonic() + 20
    while not (out.exists() and out.read_text()) and time.monotonic() < deadline:
        time.sleep(0.05)
    worker_pid, engine_pid = map(int, out.read_text().split())
    parent.kill()  # as SIGKILL or the OOM killer would
    parent.wait()
    assert _gone(worker_pid, engine_pid)


# --- what the worker writes ----------------------------------------------------------


def test_the_real_worker_writes_its_results_owner_only(tmp_path):
    worker._write(str(tmp_path / "a.json"), {"answers": []})
    assert (tmp_path / "a.json").stat().st_mode & 0o777 == 0o600


def test_the_heartbeat_counts_finished_requests_for_this_run_and_model(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "HEARTBEAT_EVERY_S", 3600.0)  # only the first and the last are written
    bar = worker.heartbeat(str(tmp_path), "run-1", "a", "generate")(total=3)
    bar.update(1)
    assert json.loads((tmp_path / "progress.json").read_text())["done"] == 1
    bar.update(1)
    assert json.loads((tmp_path / "progress.json").read_text())["done"] == 1
    bar.update(1)
    assert json.loads((tmp_path / "progress.json").read_text()) == {
        "run_id": "run-1", "model_hash": "a", "step": "generate", "done": 3}
    bar.close()


def test_vllms_prompt_rendering_loop_writes_no_heartbeat(tmp_path):
    # LLM.generate also hands use_tqdm to maybe_tqdm(prompts, desc="Rendering prompts"): an iterable bar.
    bar = worker.heartbeat(str(tmp_path), "run-1", "a", "generate")(list(range(5)), desc="Rendering prompts",
                                                                     mininterval=0.01)
    for _ in bar:
        time.sleep(0.03)
    bar.close()
    assert not (tmp_path / "progress.json").exists()


def test_vllm_scores_read_each_answer_tokens_prompt_logprob_at_its_own_position(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw, TokensPrompt=dict))
    lp = lambda logprob, rank: SimpleNamespace(logprob=logprob, rank=rank)  # noqa: E731
    # Prompt [1, 2], answer [3, 4]: the logprobs at positions 2 and 3 score 3 and 4; 1 and 2 are the prompt's.
    out = SimpleNamespace(prompt_logprobs=[None, {2: lp(-9.0, 1)}, {3: lp(-0.5, 2), 8: lp(-0.1, 1)},
                                           {4: lp(-0.2, 1)}])
    bars = []
    llm = SimpleNamespace(generate=lambda prompts, params, use_tqdm: bars.append(use_tqdm) or [out])
    [score] = worker._score(llm, [[1, 2]], [{"prompt": 0, "answer": [3, 4]}], use_tqdm="the heartbeat")
    assert score == {"nll": [0.5, 0.2], "argmax": [8, 4]} and bars == ["the heartbeat"]


# --- the stage's helpers the worker's parent uses ------------------------------------


def test_an_answer_is_cut_at_its_first_end_of_sequence_token():
    assert cut_at_eos([5, 6, 2, 9], {2, 7}) == ([5, 6], "stop")
    assert cut_at_eos([5, 6, 6], {2, 7}).finish_reason == "length"


def test_the_eos_set_is_the_generation_configs_and_the_tokenizers():
    tokenizer = SimpleNamespace(eos_token_id=7)
    assert eos_ids(SimpleNamespace(eos_token_id=[248046, 248044]), tokenizer) == {248046, 248044, 7}
    assert eos_ids(SimpleNamespace(eos_token_id=5), tokenizer) == {5, 7}
    assert eos_ids(SimpleNamespace(eos_token_id=None), SimpleNamespace(eos_token_id=None)) == set()
