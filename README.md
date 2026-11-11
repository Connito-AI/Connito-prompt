# Connito prompt validator

The code that scores miners' models on task prompts on Connito, Bittensor subnet 102. Each prompt cycle it asks
every committed model the cycle's prompts, has each answer graded, and ranks the models by their mean grade.

This is one validator, published on its own. It does not replace [Connito](https://github.com/Connito-AI/Connito),
which is still the subnet's main repository. The training validator there is still what sets weights on chain: it
blends in this validator's published results. Nothing here is what you run to train or submit a model. Mining is
all in Connito.

This repository is published so you can read exactly how a model is scored, and run the tests that pin that
behaviour. The guide for miners is at <https://doc.connito.ai/miner/prompt-evaluation/>.

## What you are reading

- **A release snapshot, not the live code.** Each release here is exported from the validator's source and
  published as one commit and a tag: `release v0.1 — exported from private @ <commit>`. `EXPORT-MANIFEST.json` names
  that source commit and lists every file with its SHA-256.
- **Production can be ahead of it, or behind it.** The validator is developed elsewhere and exported here at each
  release, so the validator that scores a given cycle may run newer code than the latest release here. A release
  can also come from a commit that production doesn't run yet. When it does, its notes say so, and say how that
  code differs from production's. The difference is never in how a model is scored: the tests pin the
  `rules_sha256` this release computes for production's settings, and the cycles production scores carry the same
  hash.
- **Scoring rules have a fingerprint.** Everything that decides a grade is hashed into `rules_sha256`: the base
  model, how answers are generated, which models are scored, the copy and reward rules, and Jev's version,
  questions and levels by task (`CRITERIA_BY_TASK`). A cycle snapshotted under other rules cannot be scored. The tests pin the hash this release computes
  for the production settings (`connito/test/test_prompt_validator_loss_rank.py`). If a cycle was scored under a
  hash this release can't produce, that cycle used newer rules.
- **Read-only mirror.** Pull requests here can't be merged: the next release would overwrite them. Please open an
  issue instead, for a bug, a question about how you were scored, or a change you'd like. We'll answer there, and
  a fix lands in a later release.

## How a model is scored

One prompt cycle is 13 training cycles. Its cutoff is the end of the commit phase of its last training cycle: the
models evaluated are the ones committed on chain at that block (`cutoff.py`).

1. **Which models.** Production sets `model_selection: all`: every model committed on chain at the cutoff for the
   cycle's task is evaluated (`snapshot.py`), with no pre-selection by training loss and no cap. A hotkey with no
   commit at the cutoff has nothing to evaluate. The code's default, and the example configs, are `top`: the
   `max_models` best by `rank_by`. Production doesn't use that.
2. **Copies.** Several hotkeys may commit the same model (the same `model_hash`). The model is evaluated once.
   Only its original is ranked: the hotkey whose signed commit held that hash earliest. Every other hotkey that
   commits it gets 0 (`reward.py:originals`).
3. **Download.** Each model's expert shard is downloaded from Hugging Face at the committed revision and checked
   the way the training validator checks it (`fetch.py`). A shard that can't be downloaded or fails the checks
   isn't evaluated that cycle.
4. **Answers.** The base model with your shard answers every prompt of the cycle (`answers.py`):
   - each prompt goes through the base model's chat template as a single user turn, with thinking off;
   - decoding is greedy, so the same model and prompt give the same answer;
   - each answer is at most `max_new_tokens` tokens (2,048 in production) and is cut off there, unfinished.

   Production generates the answers through vLLM, with the same weights. `scripts/prompt_validator_vllm/` is that
   engine, and its README describes how it was checked against the Hugging Face path.
5. **Grading.** Each answer is graded by **Jev**, TypeSafe's grading API, at a version pinned by name (`jev.py`).
   Jev is given the prompt, and rates the answer on six levels from 0 to 5:
   - the prompts production scores come with no customer request, so the question sent is
     `QUALITY_QUESTION_NO_REQUEST` in `jev.py`, word for word;
   - the six level descriptions depend on the prompt's own task (its `customer_task_id`), because one cycle can hold
     prompts of more than one task. `CRITERIA_BY_TASK` in `jev.py` maps a task to its levels:
     `exp_lesson_residual` prompts are graded on `LESSON_SCORE_LEVELS`. A task not in the map keeps the generic
     levels, `SCORE_LEVELS_NO_REQUEST` (`SCORE_LEVELS` for a prompt with a customer request). Only the levels change
     by task; the question, and its rule below, are the same;
   - `QUALITY_QUESTION` and `SCORE_LEVELS` are the same rubric for a prompt that has a customer request (its goal,
     examples and tone), which Jev is then given too. Production doesn't use them today;
   - any text in an answer aimed at the grader, such as a note on its own quality or how to score it, makes the
     answer a 0.

   Jev returns a probability for each level. A grade is the expected level, the sum of level × probability,
   divided by 5. It is a number from 0 to 1, and is the `jev_score` the results show.

   `scores.py` stores two numbers per answer. `expected_score` is that grade, and it is the only one used: for
   the ranking, the reward and the published `jev_score`. `score` is Jev's own 0-to-5 score divided by 5, kept for
   the record. It ranks nothing and earns nothing.
6. **Ranking and reward** (`reward.py`):
   - a model's score is its mean grade over the cycle's prompts;
   - to be ranked, a model needs a grade on at least 20 prompts (`reward_min_prompts`);
   - the 3 best means share the reward in proportion to their means, and a tie for third includes every model
     tied;
   - a mean of 0 earns nothing.
7. **Publishing.** The results go to the subnet's prompt API (`report.py`), where you can read them with
   `GET /v1/miner/results`. The validator here sets no chain weights. The training validator blends these results
   into its own weights; that code is `connito/validator/prompt_blend.py` in the public Connito repository.

## What is not here

- **The judge for miner-submitted prompt sets.** The validator also has a stage that checks prompt sets miners
  submit. It is off by default (`judge_miner_prompts: false`), and neither its code nor its rules are part of
  this release:
  - **`connito/prompt_validator/judge.py` is a stub.** It holds one real function, `selected_prompts`, copied
    verbatim because the answers, scores and reward stages call it. With the judge off it returns no prompts. Its
    `run_judge` raises.
  - **The judge's settings in `config.py` have no values here.** Their values and texts are in a file this release
    doesn't include, and the config refuses `judge_miner_prompts: true` without it.
  - **The plumbing that calls the judge is here** (the switch, its files, the report's handling of judged sets),
    but the stage is not.

  Models are scored only on the cycle's own prompts, so none of this changes how a model is scored.
- **The prompts.** Prompts come from customers' tasks and are not published before a cycle is evaluated. Once a
  cycle is published, its prompts and the best models' answers are in its results.
- **Deployment.** How we run the validator (container, hosts, monitoring) isn't included, and neither is anything
  that only works inside our infrastructure.
- **Development history.** Each release is a single snapshot.

## Running it

This release is for reading the code and running its tests. A full cycle can't be run outside the subnet's
validators: it signs every request to the prompt API with a validator hotkey on the subnet's whitelist.

A tool to answer and grade your own prompts with your own model, on your own GPU and Jev key, is planned for a
later release.

### Tests

The tests run on CPU, with tiny stand-in models and no network, with Python 3.12, on a host or in a container.

With CUDA 12.8, the exact environment production runs:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install --no-deps -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu128
pip install pytest
pytest
```

On a Linux machine without CUDA, the same versions with torch's CPU build (some of the locked packages have no macOS
wheels, so on a Mac use a Linux container):

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
grep -vE '^(torch==|nvidia-)' requirements.lock > requirements-cpu.txt
pip install --no-deps -r requirements-cpu.txt
pip install pytest
pytest
```

The vLLM engine itself runs in a separate environment, `requirements-vllm.lock` (vLLM 0.32, CUDA 13).

## Where things are

| Path | What it is |
|---|---|
| `connito/prompt_validator/run.py` | The entry point: the stages in order, and the loop over cycles |
| `connito/prompt_validator/cutoff.py` | When a cycle's cutoff is, from the subnet's cycle clock |
| `connito/prompt_validator/snapshot.py` | The cycle's prompts and models at the cutoff, and `rules_sha256` |
| `connito/prompt_validator/fetch.py` | Downloading and checking each model's shard |
| `connito/prompt_validator/answers.py` | Answering every prompt with each model |
| `connito/prompt_validator/jev.py` | Jev's question, its levels by task, and the grading calls |
| `connito/prompt_validator/scores.py` | One grade per answer |
| `connito/prompt_validator/reward.py` | Means, copies, ranking and shares |
| `connito/prompt_validator/report.py`, `dataless.py` | Publishing results to the prompt API |
| `connito/prompt_validator/loss_rank.py` | Ranking by training loss, for `model_selection: top` only |
| `connito/prompt_validator/weights.py` | A dry run of the weight blend; it never sets weights |
| `connito/prompt_validator/config.py` | Every setting and its default |
| `connito/shared/` | Model loading, chain reads and config, shared with the training validator |
| `scripts/prompt_validator_vllm/` | The vLLM answers engine |
| `connito/test/` | The tests |

## License

MIT; see `LICENSE`.
