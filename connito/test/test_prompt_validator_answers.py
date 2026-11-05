import json
import sys
import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration

from connito.prompt_validator import answers, run, snapshot
from connito.prompt_validator.config import GREEDY, PromptValidatorCfg, VllmCfg
from connito.prompt_validator.stage_io import read_rows, stage_path, write_rows
from connito.shared.config import ResidualCfg
from connito.shared.modeling.families import FAMILIES
from connito.shared.modeling.residual import residual_shapes
from connito.test.conftest import TINY_TEXT_CONFIG, TINY_VISION_CONFIG, _write_tokenizer, needs_judge_rules

CYCLE_ID = 1258
BASE = {"base_model": "tiny", "base_model_revision": "0" * 40, "chat_template_sha256": "1" * 64}
# The tiny tokenizer has no chat template; this one renders the user turn and, asked to, the start of the answer.
TEMPLATE = (
    "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    "{% if add_generation_prompt %} hello{% endif %}{% if enable_thinking %} think{% endif %}"
)
PROMPTS = [{"prompt_id": f"p{i}", "request_id": "r1", "prompt": text, "request": {"goal": "g", "examples": [],
                                                                                   "tone": "t"}}
           for i, text in enumerate(["hello world", "world hello", "hello hello world"], 1)]


@pytest.fixture
def tiny_checkpoint(tmp_path):
    """conftest's tiny Qwen MoE with a vocabulary as large as its byte-level tokenizer's, so text can be generated."""
    torch.manual_seed(0)
    vlm = Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeConfig(
        text_config={**TINY_TEXT_CONFIG, "vocab_size": 512}, vision_config=TINY_VISION_CONFIG,
    )).to(torch.bfloat16).eval()
    path = tmp_path / "qwen3_5_moe"
    vlm.save_pretrained(path)
    save_file({n: t.contiguous() for n, t in vlm.state_dict().items()}, str(path / "model.safetensors"),
              metadata={"format": "pt"})
    _write_tokenizer(path, {"tokenizer_class": "Qwen2Tokenizer", "bos_token": None, "eos_token": "<eos>"})
    return path


@pytest.fixture
def config(tmp_path, monkeypatch, tiny_checkpoint):
    monkeypatch.setitem(FAMILIES, "qwen3_5_moe", replace(FAMILIES["qwen3_5_moe"], model_path=str(tiny_checkpoint)))
    monkeypatch.setattr(snapshot, "base_rows", lambda cfg: dict(BASE))  # the real one downloads the chat template
    real_tokenizer = answers.get_base_tokenizer

    def tokenizer_with_template(cfg):
        tokenizer = real_tokenizer(cfg)
        tokenizer.chat_template = TEMPLATE
        return tokenizer

    monkeypatch.setattr(answers, "get_base_tokenizer", tokenizer_with_template)
    pv = PromptValidatorCfg(runs_dir=tmp_path / "runs", prompts_path=tmp_path / "p.jsonl", max_new_tokens=5,
                            answer_batch_size=2)
    return SimpleNamespace(
        role="validator",
        model=SimpleNamespace(family="qwen3_5_moe", device="cpu", quantization="fp8", quantization_miner="all",
                              quantization_validator="experts"),
        task=SimpleNamespace(expert_group_name="legal",
                             exp=SimpleNamespace(group_id=9, residual=ResidualCfg(k=1, threshold=0.5,
                                                                                    temperature=0.5))),
        ckpt=SimpleNamespace(resume_from_ckpt=False, use_pretrained_only=False),
        hf=SimpleNamespace(token_env_var="MY_HF_TOKEN"),
        prompt_validator=pv,
    )


def _shard(config, path, seed):
    torch.manual_seed(seed)
    tensors = {n: torch.randn(s.shape).to(s.dtype) for n, s in residual_shapes(config).items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    return tensors


@pytest.fixture
def cycle(config, tmp_path):
    """A snapshot and a fetch as those stages leave them: uids 1 and 3 committed the same weights (hash A), uid 2
    its own (B), and uid 4's fetch failed."""
    pv = config.prompt_validator
    snap = stage_path(pv.runs_dir, CYCLE_ID, "snapshot").parent
    write_rows(snap / "snapshot.jsonl", [{**BASE, "active_task": "legal", "expert_group": 9,
                                          "rules_sha256": snapshot.rules_sha256(pv, BASE),
                                          "snapshot_sha256": "5" * 64}])
    write_rows(snap / "prompts.jsonl", PROMPTS[::-1])  # answered in prompt_id order whatever the file's order
    hashes = {1: "a" * 64, 2: "b" * 64, 3: "a" * 64, 4: "c" * 64}
    write_rows(snap / "models.jsonl",
               [{"uid": uid, "hotkey": f"hk{uid}", "model_hash": h} for uid, h in hashes.items()])
    shards = {uid: tmp_path / "models" / f"m{uid}.safetensors" for uid in (1, 2, 3)}
    tensors = {1: _shard(config, shards[1], 1), 2: _shard(config, shards[2], 2), 3: _shard(config, shards[3], 1)}
    write_rows(stage_path(pv.runs_dir, CYCLE_ID, "fetch"), [
        {"uid": uid, "hotkey": f"hk{uid}", "path": str(shards[uid]), "reason": None} for uid in (1, 2, 3)
    ] + [{"uid": 4, "hotkey": "hk4", "path": None, "reason": "hash"}])
    return tensors


def _progress(config):
    meta = read_rows(stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "snapshot"))[0]
    return stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "answers").parent / "progress" / meta[
        "snapshot_sha256"][:16]


def _retake_snapshot(config, prompts=None, **meta):
    """The snapshot taken again, as --stage snapshot would: a new snapshot_sha256 and the config's rules."""
    pv = config.prompt_validator
    path = stage_path(pv.runs_dir, CYCLE_ID, "snapshot")
    old = read_rows(path)[0]
    write_rows(path, [{**old, "rules_sha256": snapshot.rules_sha256(pv, BASE), "snapshot_sha256": "6" * 64, **meta}])
    if prompts is not None:
        write_rows(path.parent / "prompts.jsonl", prompts)


def test_every_fetched_model_answers_every_prompt_and_copies_answer_once(config, cycle):
    rows = answers.run_answers(config, CYCLE_ID)

    assert [(r["uid"], r["prompt_id"]) for r in rows] == [(u, p) for u in (1, 2, 3) for p in ("p1", "p2", "p3")]
    assert set(rows[0]) == set(answers.ROW_FIELDS)
    by = {(r["uid"], r["prompt_id"]): r for r in rows}
    assert all(by[(3, p)]["text"] == by[(1, p)]["text"] for p in ("p1", "p2", "p3"))  # uid 3 is a copy of uid 1
    assert all(r["finish_reason"] in ("stop", "length") and 0 <= r["n_tokens"] <= 5 for r in rows)
    assert all(r["finish_reason"] == "stop" or r["n_tokens"] == 5 for r in rows)
    assert sorted(p.name for p in _progress(config).iterdir()) == ["a" * 64 + ".jsonl", "b" * 64 + ".jsonl"]


def test_answers_are_the_same_on_a_fresh_run(config, cycle):
    first = answers.run_answers(config, CYCLE_ID)
    for path in _progress(config).iterdir():
        path.unlink()
    assert answers.run_answers(config, CYCLE_ID) == first


def test_each_model_answers_with_its_own_shard_loaded(config, cycle, monkeypatch):
    seen = []
    real = answers.generate_answers

    def spy(model, tokenizer, prompts, settings, device, **kwargs):
        seen.append({n: t.detach().clone() for n, t in model.state_dict().items() if "residual" in n})
        return real(model, tokenizer, prompts, settings, device, **kwargs)

    monkeypatch.setattr(answers, "generate_answers", spy)
    answers.run_answers(config, CYCLE_ID)

    assert len(seen) == 2  # hash A (uid 1) then hash B (uid 2)
    for loaded, uid in zip(seen, (1, 2), strict=True):
        assert all(torch.equal(loaded[n], t) for n, t in cycle[uid].items())


def test_a_rerun_answers_only_the_models_not_done(config, cycle, monkeypatch):
    answers.run_answers(config, CYCLE_ID)
    (_progress(config) / ("b" * 64 + ".jsonl")).unlink()
    calls = []
    real = answers.generate_answers
    monkeypatch.setattr(answers, "generate_answers", lambda *a, **kw: calls.append(1) or real(*a, **kw))
    rows = answers.run_answers(config, CYCLE_ID)
    assert len(calls) == 1 and len(rows) == 9


def test_answers_saved_for_an_earlier_snapshot_are_not_reused(config, cycle):
    first = answers.run_answers(config, CYCLE_ID)
    assert max(r["n_tokens"] for r in first) > 1
    # The snapshot retaken with a 1-token limit and a fourth prompt: every model answers again, under the new rules.
    config.prompt_validator.max_new_tokens = 1
    _retake_snapshot(config, prompts=[*PROMPTS, {**PROMPTS[0], "prompt_id": "p4"}])
    rows = answers.run_answers(config, CYCLE_ID)
    assert {r["prompt_id"] for r in rows} == {"p1", "p2", "p3", "p4"}
    assert max(r["n_tokens"] for r in rows) == 1


def test_a_saved_file_for_other_prompts_is_answered_again(config, cycle):
    answers.run_answers(config, CYCLE_ID)
    path = _progress(config) / ("b" * 64 + ".jsonl")
    write_rows(path, read_rows(path)[:2])  # p3 missing, as if cut short
    rows = answers.run_answers(config, CYCLE_ID)
    assert [r["prompt_id"] for r in rows if r["uid"] == 2] == ["p1", "p2", "p3"]


def test_a_shard_key_the_model_does_not_have_stops_the_stage(config, cycle, monkeypatch):
    monkeypatch.setattr(answers, "verify_residual_shard",
                        lambda path, expected: {"model.no_such.weight": torch.ones(1)})
    with pytest.raises(RuntimeError, match="uid 1's shard has 1 keys the model does not have"):
        answers.run_answers(config, CYCLE_ID)


def test_the_snapshot_task_is_switched_to_and_its_group_checked(config, cycle, monkeypatch):
    switched = []

    def switch(name):
        switched.append(name)
        config.task.expert_group_name = name

    config.task.expert_group_name = "shipped-default"
    config.switch_active_task = switch
    answers.run_answers(config, CYCLE_ID)
    assert switched == ["legal"]

    config.task.exp.group_id = 4
    with pytest.raises(RuntimeError, match="snapshot group 9 is not the loaded task's group 4"):
        answers.run_answers(config, CYCLE_ID)


def test_a_snapshot_task_that_is_not_materialised_stops_with_a_clear_error(config, cycle):
    config.task.expert_group_name, config.task.base_path = "shipped-default", "expert_groups"

    def missing(name):
        raise FileNotFoundError(name)

    config.switch_active_task = missing
    with pytest.raises(RuntimeError, match="snapshot task legal is not materialised"):
        answers.run_answers(config, CYCLE_ID)


def test_answers_do_not_depend_on_the_batch_they_are_in(config, cycle):
    # Left padding: each prompt answers the same alone and batched with longer and shorter ones.
    batched = answers.run_answers(config, CYCLE_ID)
    config.prompt_validator.answer_batch_size = 1
    _retake_snapshot(config)
    alone = answers.run_answers(config, CYCLE_ID)
    assert [r["text"] for r in alone] == [r["text"] for r in batched]


def test_a_shard_that_no_longer_matches_the_spec_stops_the_stage(config, cycle):
    fetch = read_rows(stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "fetch"))
    path = fetch[1]["path"]
    save_file({"model.layers.0.mlp.residual_router.weight": torch.ones(1, 64)}, path)  # uid 2's: one tensor only
    with pytest.raises(RuntimeError, match="uid 2's shard .* no longer matches the residual spec"):
        answers.run_answers(config, CYCLE_ID)


def test_answer_settings_changed_since_the_snapshot_are_refused(config, cycle):
    config.prompt_validator.max_new_tokens = 6
    with pytest.raises(RuntimeError, match="is not the snapshot's"):
        answers.run_answers(config, CYCLE_ID)


def test_no_fetched_model_stops_the_stage(config, cycle):
    path = stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "fetch")
    write_rows(path, [{**r, "path": None, "reason": "timeout"} for r in read_rows(path)])
    with pytest.raises(RuntimeError, match="no model of scoring cycle 1258 was fetched"):
        answers.run_answers(config, CYCLE_ID)


MINER = {"m1": "world world", "m2": "hello there"}


def _judge_on(config, write_selection, selected):
    """The switch on, the snapshot retaken under its rules, and the judge's selection."""
    config.prompt_validator.judge_miner_prompts = True
    _retake_snapshot(config)
    write_selection(config.prompt_validator, CYCLE_ID, MINER, selected)


@needs_judge_rules
def test_with_the_switch_on_every_model_answers_the_pool_and_the_selected_prompts(config, cycle, write_selection):
    _judge_on(config, write_selection, ["m2"])
    rows = answers.run_answers(config, CYCLE_ID)
    assert [(r["uid"], r["prompt_id"]) for r in rows] == [(u, p) for u in (1, 2, 3) for p in ("m2", "p1", "p2", "p3")]


@needs_judge_rules
def test_a_judge_rerun_that_selects_other_prompts_answers_them(config, cycle, write_selection):
    _judge_on(config, write_selection, ["m2"])
    answers.run_answers(config, CYCLE_ID)
    _judge_on(config, write_selection, ["m1", "m2"])  # the same snapshot, so its progress files, for other prompts
    rows = answers.run_answers(config, CYCLE_ID)
    assert [r["prompt_id"] for r in rows if r["uid"] == 2] == ["m1", "m2", "p1", "p2", "p3"]


@needs_judge_rules
def test_with_the_switch_on_answers_need_the_judge_stage(config, cycle, write_selection):
    _judge_on(config, write_selection, ["m2"])
    stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "judge").unlink()
    with pytest.raises(RuntimeError, match="not judged yet: run the judge stage"):
        answers.run_answers(config, CYCLE_ID)


# --- generate_answers on a fake model: what it passes and how it reads the output ---

class FakeTokenizer:
    eos_token_id, pad_token_id = 2, 0

    def __init__(self):
        self.templates, self.batches = [], []

    def apply_chat_template(self, messages, **kwargs):
        self.templates.append((messages, kwargs))
        return messages[0]["content"]

    def __call__(self, texts, **kwargs):
        self.batches.append((list(texts), kwargs))
        ids = torch.ones(len(texts), 3, dtype=torch.long)
        return _Batch({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens is True
        return " ".join(map(str, tokens))


class _Batch(dict):
    def to(self, device):
        return self


class FakeModel:
    generation_config = SimpleNamespace(eos_token_id=[2, 7])

    def __init__(self, continuations):
        self.continuations, self.calls = list(continuations), []

    def generate(self, input_ids, attention_mask, **kwargs):
        self.calls.append(kwargs)
        new = [self.continuations.pop(0) for _ in range(input_ids.shape[0])]
        return torch.cat([input_ids, torch.tensor(new)], dim=1)


def test_generate_passes_exactly_the_answer_settings_and_batches_in_order(config):
    settings = snapshot.answer_settings(config.prompt_validator)
    tokenizer = FakeTokenizer()
    model = FakeModel([[5, 6, 2, 0], [5, 7, 0, 0], [5, 6, 6, 6]])
    out = answers.generate_answers(model, tokenizer, PROMPTS, settings, torch.device("cpu"))

    assert [kwargs for _, kwargs in tokenizer.templates] == [
        {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}] * 3
    assert [m for m, _ in tokenizer.templates] == [[{"role": "user", "content": p["prompt"]}] for p in PROMPTS]
    assert [texts for texts, _ in tokenizer.batches] == [["hello world", "world hello"], ["hello hello world"]]
    assert model.calls == [{**GREEDY, "max_new_tokens": 5, "pad_token_id": 0}] * 2
    # The answer ends at its first end-of-sequence id (2, or 7 from generation_config); none means it hit the cap.
    assert out == [
        {"prompt_id": "p1", "text": "5 6", "n_tokens": 2, "finish_reason": "stop"},
        {"prompt_id": "p2", "text": "5", "n_tokens": 1, "finish_reason": "stop"},
        {"prompt_id": "p3", "text": "5 6 6 6", "n_tokens": 4, "finish_reason": "length"},
    ]


def test_the_tokenizers_eos_ends_an_answer_when_generation_config_names_none(config):
    model = FakeModel([[5, 2, 0]])
    model.generation_config = SimpleNamespace(eos_token_id=None)
    tokenizer = FakeTokenizer()
    settings = {**snapshot.answer_settings(config.prompt_validator), "batch_size": 1}
    [out] = answers.generate_answers(model, tokenizer, PROMPTS[:1], settings, torch.device("cpu"))
    assert (out["text"], out["finish_reason"]) == ("5", "stop")



def test_on_batch_hears_the_prompts_answered_after_each_batch(config):
    settings = snapshot.answer_settings(config.prompt_validator)
    model = FakeModel([[5, 2], [5, 2], [5, 2]])
    seen = []
    answers.generate_answers(model, FakeTokenizer(), PROMPTS, settings, torch.device("cpu"), on_batch=seen.append)
    assert seen == [2, 3]  # batch_size 2 over 3 prompts


def test_the_run_logs_its_progress_over_every_answer_to_generate(config, cycle, monkeypatch):
    logged = []
    real = answers.logger.info
    monkeypatch.setattr(answers.logger, "info", lambda event, **kw: logged.append((event, kw)) or real(event, **kw))
    answers.run_answers(config, CYCLE_ID)
    [total] = [kw["total"] for event, kw in logged if event == "answers to generate"]
    steps = [kw for event, kw in logged if event == "answering"]
    assert total == 2 * len(PROMPTS)  # two distinct models (uid 3 copies uid 1's)
    assert [s["answers"] for s in steps] == [f"{n}/{total}" for n in (2, 3, 5, 6)]
    assert [s["model"] for s in steps] == ["1/2", "1/2", "2/2", "2/2"]
    assert steps[-1]["percent"] == 100.0 and steps[-1]["eta_min"] == 0.0

def test_the_batch_size_is_an_answer_setting_in_the_rules_hash(config):
    pv = config.prompt_validator
    assert snapshot.answer_settings(pv)["batch_size"] == 2
    assert PromptValidatorCfg(prompts_path=pv.prompts_path).answer_batch_size == 20
    before = snapshot.rules_sha256(pv, BASE)
    assert snapshot.rules_sha256(pv.model_copy(update={"answer_batch_size": 4}), BASE) != before


def test_answers_run_after_fetch_in_a_plain_run():
    assert [name for name, _ in run.STAGES] == [
        "cutoff", "snapshot", "judge", "fetch", "answers", "scores", "reward", "report", "weights"]
    assert run.ON_DEMAND == {"report", "weights"}


def test_under_the_loop_each_batch_moves_the_status_progress(config, cycle, monkeypatch):
    from connito.prompt_validator import status

    seen = []
    real = status.write_rows
    monkeypatch.setattr(status, "write_rows", lambda path, rows: (seen.append(dict(rows[0])), real(path, rows)))
    with status.loop_status(config.prompt_validator.runs_dir):
        answers.run_answers(config, CYCLE_ID)

    # Models a and b, 3 prompts each, in batches of 2; then "stopped", which clears them.
    assert [s.get("progress") for s in seen] == ["2/6", "3/6", "5/6", "6/6", None]
    assert seen[-2]["eta_min"] == 0.0


# --- the GPU is given back after the stage: the loop then waits about a day for the next cutoff ---

def _track_model(monkeypatch):
    """Weak references to every model the stage builds, and the liveness of each when the CUDA cache is emptied."""
    refs, at_empty_cache = [], []
    real = answers.get_model_from_checkpoint

    def tracked(*args, **kwargs):
        model, ckpt = real(*args, **kwargs)
        model.cycle = [model]  # a reference cycle, as a hook or closure can make: only the collector frees it
        refs.append(weakref.ref(model))
        return model, ckpt

    monkeypatch.setattr(answers, "get_model_from_checkpoint", tracked)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # the device is still config.model.device: cpu
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: at_empty_cache.append([r() is not None for r in refs]))
    return refs, at_empty_cache


def test_the_model_is_freed_and_the_cuda_cache_emptied_after_the_stage(config, cycle, monkeypatch):
    refs, at_empty_cache = _track_model(monkeypatch)
    answers.run_answers(config, CYCLE_ID)
    assert len(refs) == 1 and refs[0]() is None
    assert at_empty_cache == [[False]]  # emptied once, after the model was gone


def test_the_model_is_freed_after_the_stage_fails_while_its_error_is_still_held(config, cycle, monkeypatch):
    refs, at_empty_cache = _track_model(monkeypatch)

    def load(model):
        raise ValueError("a failure deep in generation")

    def caught(model):
        try:
            raise LookupError("an earlier failure, kept")
        except LookupError as e:
            return e

    def generate(model, *args, **kwargs):
        cause = caught(model)  # its traceback's frame holds the model
        try:
            load(model)
        except ValueError:
            error = RuntimeError("generation failed")
            cause.__context__ = error  # a chain can loop back: Python breaks cycles along __context__ only
            raise error from cause  # cause and context differ: both chains' frames hold the model

    monkeypatch.setattr(answers, "generate_answers", generate)
    with pytest.raises(RuntimeError, match="generation failed") as failure:
        answers.run_answers(config, CYCLE_ID)
    # The error, and its traceback, are still held here, as the loop holds them while it logs the failure.
    assert failure.value.__traceback__ is not None
    assert len(refs) == 1 and refs[0]() is None
    assert at_empty_cache == [[False]]


@pytest.mark.parametrize("fails", [False, True])
def test_a_cuda_cache_that_cannot_be_emptied_is_logged_and_never_replaces_the_stages_result(config, cycle, monkeypatch,
                                                                                            fails):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: 1 / 0)
    warned = []
    monkeypatch.setattr(answers.logger, "warning", lambda event, **kw: warned.append((event, kw)))
    if fails:
        def generate(*args, **kwargs):
            raise KeyError("the stage's own error")

        monkeypatch.setattr(answers, "generate_answers", generate)
        with pytest.raises(KeyError, match="the stage's own error"):
            answers.run_answers(config, CYCLE_ID)
    else:
        assert len(answers.run_answers(config, CYCLE_ID)) == 3 * len(PROMPTS)
    assert warned == [("CUDA cache not emptied", {"error": "ZeroDivisionError"})]


# --- engine "vllm": the worker is a fake vLLM python, so these run without vLLM ---

FAKE_WORKER = """#!{python}
import json, os, signal, sys, time
job_path, work = sys.argv[3:5]
job = json.load(open(job_path))
mode = os.environ["FAKE_MODE"]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps({{"env": dict(os.environ), "models": [m["model_hash"] for m in job["models"]],
                        "prompts": job["prompts"], "eos": job["eos"]}}) + "\\n")
if mode == "kill":
    os.kill(os.getpid(), signal.SIGKILL)
if mode == "hang":
    time.sleep(60)
for i, model in enumerate(job["models"]):
    if mode == "silent" or (mode == "crash" and i == 1):
        break
    beat = {{"run_id": job["run_id"], "model_hash": model["model_hash"], "step": "generate", "done": 1}}
    with open(os.path.join(work, "beat.tmp"), "w") as f:
        json.dump(beat, f)
    os.replace(os.path.join(work, "beat.tmp"), os.path.join(work, "progress.json"))
    time.sleep(0.2)  # one heartbeat, seen before the model's result
    answers = [{{"ids": [40 + i, 41, job["eos"][0], 7], "finish_reason": "stop"}} for _ in job["prompts"]]
    row = {{"model_hash": model["model_hash"], "run_id": job["run_id"], "answers": answers, "start_s": 1.0,
           "swap_s": 0.1, "generate_s": 2.0, "new_tokens": 4 * len(answers), "tokens_per_s": 2.0, "peak_gib": 1.0}}
    # Whole or not at all, as connito_vllm.worker._write: the stage polls for the file and reads it once it exists.
    with open(os.path.join(work, "result.tmp"), "w") as f:
        json.dump(row, f)
    os.replace(os.path.join(work, "result.tmp"), os.path.join(work, model["model_hash"] + ".json"))
    time.sleep(0.2)  # the result is seen before the next model's heartbeat
sys.exit(1 if mode == "crash" else 0)
"""


@pytest.fixture
def vllm(config, tmp_path, monkeypatch):
    """engine "vllm" with a fake vLLM python; FAKE_LOG gets one line per worker run: its env, models and prompts."""
    python = tmp_path / "fake-python"
    python.write_text(FAKE_WORKER.format(python=sys.executable))
    python.chmod(0o755)
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text("{}")
    pv = config.prompt_validator
    pv.engine, pv.vllm = "vllm", VllmCfg(python=python, checkpoint=checkpoint)
    monkeypatch.setattr(answers, "_POLL_S", 0.01)
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "fake.log"))
    monkeypatch.setenv("FAKE_MODE", "ok")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    return tmp_path / "fake.log"


def _runs(log) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_vllm_answers_every_model_once_in_the_stages_rows(vllm, config, cycle):
    rows = answers.run_answers(config, CYCLE_ID)

    assert [(r["uid"], r["prompt_id"]) for r in rows] == [(u, f"p{i}") for u in (1, 2, 3) for i in (1, 2, 3)]
    assert all(set(r) == set(answers.ROW_FIELDS) and r["finish_reason"] == "stop" and r["n_tokens"] == 2 for r in rows)
    [run] = _runs(vllm)
    assert run["models"] == ["a" * 64, "b" * 64]  # one per model_hash, the lowest uid's shard
    assert len(run["prompts"]) == 3 and all(isinstance(t, int) for ids in run["prompts"] for t in ids)
    assert not (_progress(config) / "vllm").exists()


def test_the_worker_gets_no_secrets_and_keeps_the_gpu_it_was_given(vllm, config, cycle, monkeypatch):
    for name in ("HF_TOKEN", "JEV_DIRECT_API_KEY", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY", "MY_HF_TOKEN",
                 "BT_PW_SOME_WALLET"):
        monkeypatch.setenv(name, "secret")
    answers.run_answers(config, CYCLE_ID)

    env = _runs(vllm)[0]["env"]  # MY_HF_TOKEN: the config's hf.token_env_var
    assert not {"HF_TOKEN", "JEV_DIRECT_API_KEY", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY", "MY_HF_TOKEN"} & set(env)
    assert not any(k.startswith("BT_PW_") for k in env)
    assert (env["CUDA_VISIBLE_DEVICES"], env["CUDA_DEVICE_ORDER"], env["HF_HUB_OFFLINE"]) == ("3", "PCI_BUS_ID", "1")


def test_a_rerun_sends_the_worker_only_the_models_not_done(vllm, config, cycle):
    answers.run_answers(config, CYCLE_ID)
    (_progress(config) / f"{'b' * 64}.jsonl").unlink()
    answers.run_answers(config, CYCLE_ID)

    assert [run["models"] for run in _runs(vllm)] == [["a" * 64, "b" * 64], ["b" * 64]]


@pytest.mark.parametrize("mode, error", [("kill", "exit status -9"), ("silent", "without answering")])
def test_a_worker_that_dies_or_answers_nothing_fails_the_stage(vllm, config, cycle, monkeypatch, mode, error):
    monkeypatch.setenv("FAKE_MODE", mode)
    with pytest.raises(RuntimeError, match=error):
        answers.run_answers(config, CYCLE_ID)
    assert not list(_progress(config).glob("*.jsonl"))


def test_a_crash_keeps_the_models_finished_before_it_and_a_rerun_answers_the_rest(vllm, config, cycle, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "crash")
    with pytest.raises(RuntimeError, match="exit status 1"):
        answers.run_answers(config, CYCLE_ID)
    assert [p.stem for p in _progress(config).glob("*.jsonl")] == ["a" * 64]
    work = _progress(config) / "vllm"  # the job holds the private prompts' token ids
    assert work.stat().st_mode & 0o777 == 0o700 and (work / "job.json").stat().st_mode & 0o777 == 0o600

    monkeypatch.setenv("FAKE_MODE", "ok")
    rows = answers.run_answers(config, CYCLE_ID)
    assert [run["models"] for run in _runs(vllm)] == [["a" * 64, "b" * 64], ["b" * 64]]
    assert len(rows) == 9


def test_the_vllm_settings_are_in_the_rules_hash(vllm, config, cycle):
    pv = config.prompt_validator
    pv.vllm = pv.vllm.model_copy(update={"max_num_seqs": 64})
    with pytest.raises(RuntimeError, match="is not the snapshot's"):
        answers.run_answers(config, CYCLE_ID)


def test_a_hung_worker_fails_the_stage_after_the_configured_wait(vllm, config, cycle, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "hang")
    pv = config.prompt_validator
    pv.vllm = pv.vllm.model_copy(update={"start_timeout_min": 0.01})  # not in the rules hash
    with pytest.raises(TimeoutError, match="finished no request"):
        answers.run_answers(config, CYCLE_ID)
    assert not list(_progress(config).glob("*.jsonl"))


def test_each_heartbeat_moves_the_status_progress_and_eta(vllm, config, cycle, monkeypatch):
    seen = []
    monkeypatch.setattr(answers, "set_status", lambda **kw: seen.append(kw))
    answers.run_answers(config, CYCLE_ID)

    # Two models of three prompts: a heartbeat (one request) then the result, for each.
    assert [s["progress"] for s in seen] == ["1/6", "3/6", "4/6", "6/6"]
    counts = [int(s["progress"].split("/")[0]) for s in seen]
    assert counts == sorted(counts)  # never backwards
    assert all(s["eta_min"] >= 0 for s in seen) and seen[-1]["eta_min"] == 0
