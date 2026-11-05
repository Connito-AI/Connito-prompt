# vLLM answers engine

The prompt validator's answers stage (`connito/prompt_validator/answers.py`) answers with Hugging Face `generate()`:
about 10 minutes per model for 20 prompts on one A6000. This folder answers the **same snapshot** with
vLLM and the **same model**: the validator's fp8 per-row routed experts and the miners' residual experts.

The answers stage uses it with `engine: vllm` (the default stays `hf`). Its parity with the HF path was measured
with our own tool, `check_vllm.py`, which is not part of the public release: each model's mean |ΔNLL| and argmax
disagreement against HF's, teacher-forced on the same answers, within 1.1× of HF's own floor (one sequence at a time
against left-padded batches) and under 0.04 nats and 97 % agreement.

## What it does

- `export_fp8row.py` (repo venv): loads the base through Connito's own loader (`get_base_model`, fp8 scope
  `experts`) and writes a vLLM checkpoint, once per base revision.
  - Routed experts: the loader's fp8 e4m3 bytes, per expert (`experts.<e>.{gate,up,down}_proj.weight`, gate_up's
    rows split into gate and up), with per-row scales rounded to bf16 (what HF multiplies by), stored as fp32.
  - Every other tensor: bf16, copied unchanged from the hub shards. Vision and MTP are dropped.
  - `config.json` gets a compressed-tensors config that quantizes only the routed experts; `manifest.json` holds the
    source revision, a sha256 per tensor and one per other file (config, index, tokenizer, chat template).
- `connito_vllm/` (vLLM venv): a package installed into the vLLM venv. Its `vllm.general_plugins` entry point
  registers the model in every vLLM process, so no `ModelRegistry` call or `sys.path` change is needed.
  - `model.py`: `ConnitoResidualQwen3_5MoeForCausalLM`, vLLM 0.32's `Qwen3_5MoeForCausalLM` with Connito's residual
    experts added to every MoE block's full output (routed plus gated shared expert), with the math of
    `connito/shared/modeling/residual.py`: q = sigmoid((sigmoid(W_r·x) − A) / τ) in fp32, and
    h = out + Σ_j q_j·down_j(silu(gate_j·x) · up_j·x). The residual tensors start at zero and are overwritten in place
    for each miner by the worker extension `ResidualLoader`, so the engine and its CUDA graphs are built once.
  - `residual_math.py`: the gate math, the shard layout and the in-place swap (every shape checked before the first
    copy), torch only. The CPU tests check the math bit for bit against the HF hook in fp32 and bf16, eagerly.
  - `sampling.py`: the engine and decoding settings. Greedy, stopping at the generation config's eos ids plus the
    tokenizer's, `generation_config="vllm"` so the hub's sampling defaults (temperature 1.0, top_k 20, top_p 0.95 at
    995ad96, unused by HF's greedy) never apply, and `moe_backend="marlin"`, so the experts run W8A16 on any GPU. Any
    other generation-config field set on the hub is refused. The model refuses an engine with another MoE backend.
  - `worker.py`: `python -m connito_vllm.worker JOB.json OUT_DIR`. One engine; for each model of the job it swaps
    the residual in, answers the given prompt ids greedily, optionally teacher-forces given answers through
    `prompt_logprobs` (raw logprobs), and writes `OUT_DIR/<model_hash>.json` (0600) as that model finishes. As each
    request finishes it rewrites `OUT_DIR/progress.json` (at most once a second): the heartbeat.
- `connito/prompt_validator/answers.run_worker` (repo venv) drives it: it tokenizes, cuts at eos and decodes, runs the
  worker in its own process group with the secrets dropped and one pinned GPU, and takes only results naming its run.
  It allows `start_timeout_s` until the first finished request and then `stall_timeout_s` between heartbeats; a stall
  kills the group and raises `TimeoutError`. The group is killed on every exit, and the worker kills its own group
  when the parent dies. `python` must be the venv's interpreter, or a wrapper that `exec`s it: the worker watches the
  pid that started it.
## The answers stage with `engine: vllm`

```yaml
prompt_validator:
  engine: vllm
  vllm:
    python: /path/to/venv-vllm/bin/python      # with this package installed
    checkpoint: /path/to/qwen3.6-a3b-fp8row-995ad96   # export_fp8row.py's output
    max_num_seqs: 128           # in the rules hash, with the vLLM version and manifest.json's sha256
    gpu_memory_utilization: 0.9
    enforce_eager: false        # CUDA graphs on: about 3x faster; in the rules hash
```

- **One engine per stage run.** Only the models not yet answered go to the worker. Each model's progress file is
  written as soon as its result lands, in the HF path's format, so a crash or a rerun never regenerates a finished
  model.
- **Worker environment:** the worker gets the parent's env without secrets (`JEV_DIRECT_API_KEY`, `HF_TOKEN`,
  `WANDB_API_KEY`, `BT_PW_*` and the config's `hf.token_env_var`), offline, with `CUDA_VISIBLE_DEVICES` unchanged; it
  refuses to start unless that names one GPU under `CUDA_DEVICE_ORDER=PCI_BUS_ID`. It runs in its own process group:
  the stage kills the whole group when it raises or is interrupted, and the worker kills its own group (the engine
  included) when the stage process dies, even by SIGKILL or the OOM killer.
- **Stalls:** the worker reports each finished request (its heartbeat). One that finishes no request within
  `start_timeout_min` (20, engine start included) or, after that, within `stall_timeout_min` (10) of the last is
  killed, and the stage raises, so `--loop` retries instead of hanging. Model size doesn't matter: even a model with many
  prompts to answer finishes a request every few seconds. The heartbeat also drives the status progress and
  eta.
- **Batch invariance is off.** `VLLM_BATCH_INVARIANT=1` is refused by the Marlin MoE kernel.
- **Answers repeat within one engine, not across processes.** Two engine starts give different answers, as HF gives
  under different batch shapes; this is accepted.

## Known differences from HF

- **The stored expert weights are identical, the dequantized ones are not.** Both start from the same fp8 bytes and
  bf16-rounded per-row scales. HF dequantizes each weight to `bf16(bf16(fp8) * bf16(scale))`, rounded per weight,
  then multiplies. Marlin multiplies by the fp8 values and applies the channel scale to the fp32 accumulator after
  the reduction over K (`csrc/moe/marlin_moe_wna16/marlin_template.h:1893-1905`). Attention, GatedDeltaNet and the
  routing are vLLM's kernels too, and vLLM does not pad, while HF left-pads each batch. If the parity check fails,
  a possible fallback (dequantize exactly as HF, then vLLM's unquantized Triton experts) closes the expert part.
- **The residual math is bit-equal to the HF hook in eager mode only.** Under torch.compile, Inductor fuses the gate
  and expert ops and may keep intermediates in fp32.
- **compressed-tensors cannot say W8A16 fp8 for an MoE in vLLM 0.32**, so the config declares fp8 per-token
  activations. `moe_backend="marlin"` is what keeps the activations in bf16; without it, an fp8-capable GPU would
  quantize them.
- Greedy answers can split at the first near-tie token and differ from there on. The parity check measures this.

## Setting it up

The engine's environment is pinned in `requirements-vllm.lock` (vLLM 0.32, torch 2.13, CUDA 13), and this folder is
installed into it as a package. Step-by-step setup comes with the v0.2 runner, together with the commands, once
they are tested end to end. The tests in this release don't need vLLM: they run the plugin's math on CPU, and its
worker with a stand-in.
