"""`python -m connito_vllm.worker JOB.json OUT_DIR`: answer, and optionally score, token ids in one vLLM engine.

The repo-venv parent (answers.run_worker) tokenizes, cuts and decodes. JOB holds the checkpoint, residual, prompt ids,
eos, settings and models [{model_hash, shard (null: base), score: [{prompt, answer}] | "own" | null}]. Each model's
timing, generated ids and per-token NLL/argmax land in OUT_DIR/<model_hash>.json, owner-only, as it finishes.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time

PARENT_ENV = "CONNITO_VLLM_PARENT"  # the parent's pid, set by answers.run_worker
HEARTBEAT = "progress.json"  # OUT_DIR/progress.json: {run_id, model_hash, step, done}, rewritten as requests finish
HEARTBEAT_EVERY_S = 1.0
_SINK = open(os.devnull, "w")  # noqa: SIM115  one handle for every heartbeat bar's output, for the process's life


def heartbeat(out_dir: str, run_id: str | None, model_hash: str, step: str):
    """A use_tqdm factory for LLM.generate: vLLM updates it once per finished request, and it writes the heartbeat."""
    from tqdm import tqdm

    class Heartbeat(tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **{**kwargs, "file": _SINK})
            self._written = 0.0

        def update(self, n: int = 1):
            out = super().update(n)
            if self.iterable is not None:
                return out  # vLLM's "Rendering prompts" loop counts prompts added, not requests finished
            now = time.monotonic()
            if now - self._written >= HEARTBEAT_EVERY_S or self.n == self.total:
                _write(os.path.join(out_dir, HEARTBEAT),
                       {"run_id": run_id, "model_hash": model_hash, "step": step, "done": self.n})
                self._written = now
            return out

    return Heartbeat


def _score(llm, prompts: list[list[int]], items: list[dict], use_tqdm=False) -> list[dict]:
    """Per answer token: -logprob of the actual token and the rank-1 token, from vLLM's raw prompt logprobs."""
    from vllm import SamplingParams, TokensPrompt

    outputs = llm.generate([TokensPrompt(prompt_token_ids=prompts[i["prompt"]] + i["answer"]) for i in items],
                           SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1), use_tqdm=use_tqdm)
    scores = []
    for item, out in zip(items, outputs, strict=True):
        n = len(prompts[item["prompt"]])
        nll, argmax = [], []
        for token, logprobs in zip(item["answer"], out.prompt_logprobs[n:], strict=True):
            nll.append(-logprobs[token].logprob)
            argmax.append(next(t for t, lp in logprobs.items() if lp.rank == 1))
        scores.append({"nll": nll, "argmax": argmax})
    return scores


def _write(path: str, row: dict) -> None:
    """Whole or not at all, and owner-only: the ids are answers to the private prompts."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8", opener=lambda p, flags: os.open(p, flags, 0o600)) as f:
        json.dump(row, f)
    os.replace(tmp, path)


def exit_with_parent(parent: int, poll_s: float = 1.0) -> None:
    """Kill this process group (the engine core included) once `parent` is gone, however it died."""
    # The parent starts this group in its own session, so a SIGKILLed or OOM-killed parent would otherwise leave the
    # engine holding the GPU.
    # The interpreter run_worker started must be this process (or exec into it): a forking wrapper is a new parent.
    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(poll_s)
        os.killpg(0, signal.SIGKILL)

    if os.getppid() != parent:
        os.killpg(0, signal.SIGKILL)  # already gone: kill now, before vLLM is even imported
    threading.Thread(target=watch, daemon=True).start()


def main() -> None:
    exit_with_parent(int(os.environ.get(PARENT_ENV, os.getppid())))
    from connito_vllm.sampling import engine_args, greedy, worker_env

    env = worker_env(dict(os.environ))  # before vLLM, torch or CUDA are imported
    os.environ.clear()
    os.environ.update(env)
    job_path, out_dir = sys.argv[1:3]
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    import vllm
    from transformers import GenerationConfig
    from vllm import LLM, SamplingParams, TokensPrompt

    if job.get("vllm_version") and vllm.__version__ != job["vllm_version"]:
        raise SystemExit(f"vLLM {vllm.__version__} is not the configured {job['vllm_version']}")

    eos = set(job["eos"])
    start = time.monotonic()
    llm = LLM(**engine_args(job["checkpoint"], job["residual"], logprobs_mode="raw_logprobs", **job["engine"]))
    start_s = round(time.monotonic() - start, 1)
    sampling = SamplingParams(**greedy(GenerationConfig.from_pretrained(job["checkpoint"]), eos, job["max_new_tokens"]))
    inputs = [TokensPrompt(prompt_token_ids=ids) for ids in job["prompts"]]
    llm.collective_rpc("peak_gib")  # reset: the start-up peak is not a model's
    for model in job["models"]:
        start = time.monotonic()
        llm.collective_rpc("load_residual", args=(model["shard"],))
        swapped = time.monotonic()
        progress = heartbeat(out_dir, job.get("run_id"), model["model_hash"], "generate")
        answers = [o.outputs[0] for o in llm.generate(inputs, sampling, use_tqdm=progress)]
        generated = time.monotonic() - swapped
        row = {"model_hash": model["model_hash"], "run_id": job.get("run_id"), "start_s": start_s,
               "swap_s": round(swapped - start, 2),
               "generate_s": round(generated, 1),
               "new_tokens": sum(len(a.token_ids) for a in answers),
               "answers": [{"ids": list(a.token_ids), "finish_reason": a.finish_reason} for a in answers]}
        row["tokens_per_s"] = round(row["new_tokens"] / generated, 1)
        items = model["score"]
        if items == "own":
            items = [{"prompt": i, "answer": list(a.token_ids[:next(
                (k for k, t in enumerate(a.token_ids) if t in eos), len(a.token_ids))])} for i, a in enumerate(answers)]
        if items:
            scored = time.monotonic()
            row["scores"] = _score(llm, job["prompts"], items,
                                   heartbeat(out_dir, job.get("run_id"), model["model_hash"], "score"))
            row["score_s"] = round(time.monotonic() - scored, 1)
            row["scored"] = items
        [row["peak_gib"]] = llm.collective_rpc("peak_gib")
        _write(os.path.join(out_dir, f"{model['model_hash']}.json"), row)
        print(f"{model['model_hash'][:12]}: {row['new_tokens']} tokens in {row['generate_s']}s "
              f"({row['tokens_per_s']} tok/s), peak {row['peak_gib']:.1f} GiB", flush=True)
    llm.collective_rpc("load_residual", args=(None,))


if __name__ == "__main__":
    main()
