from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Literal
from urllib.parse import urlsplit

import bittensor
import huggingface_hub.constants
from bittensor.sp_core import KeyfileError, get_password_from_environment
from pydantic import ConfigDict, Field, NonNegativeFloat, PositiveInt, field_validator, model_validator

from connito.prompt_validator.stage_io import under_root
from connito.shared.config import BaseConfig, ChainCfg, ValidatorConfig

# HF generate has no per-row seeds, so answers are greedy. The answers stage passes this and max_new_tokens to
# generate() explicitly, so the base model's generation_config.json cannot turn sampling on. Its other fields (eos ids,
# repetition penalty) still apply, fixed by the family's pinned revision.
GREEDY = {"do_sample": False}


class JevCfg(BaseConfig):
    url: str = "https://api.typesafe.ai/v1/systemone"
    resolved: str = "jev-1.14.0"  # the version requested by name, which every response must report
    concurrency: int = 8  # calls in flight
    max_answers_per_call: PositiveInt = 10  # answers to one prompt scored in one call, one score question each
    timeout_sec: int = 60  # seconds per HTTP attempt
    retries: int = 2  # after the first attempt, on network errors and retryable HTTP statuses; never on a read timeout
    spend_cap_usd: float = 5.0  # per scoring cycle
    # USD per million input tokens, your contract's: the direct API reports no cost, so the spend cap counts with it.
    # No default: scoring refuses to start without it.
    price_per_m: NonNegativeFloat | None = None

    @model_validator(mode="before")
    @classmethod
    def _no_model(cls, data: Any) -> Any:
        # Extra keys are ignored, so a leftover jev.model would look like it picks the version while doing nothing.
        if isinstance(data, dict) and "model" in data:
            raise ValueError(f"jev.model is gone: Jev is asked for jev.resolved by name; remove model "
                             f"(got {data['model']!r}) and set resolved to the version to use")
        return data


class DatalessCfg(BaseConfig):
    """The dataless-api (the prompt-record service): prompts in, scores and model outputs out."""

    url: str = "https://dataless-api.connito.ai"  # the origin, with no path: requests are signed on the path as sent
    timeout_sec: int = 30  # seconds per HTTP attempt
    retries: int = 3  # after the first attempt, on network errors, 429 and 5xx; each retry is signed afresh
    # Reported with the run, 1 to 64 characters: which code scored it. A deployment sets the release it runs.
    validator_code_version: str = "prompt-validator"
    # Test cycles: the same API under /v1/test, on the server's separate test tables, with cycles the owner opens
    # and closes by hand. The run takes the closed test cycle's id; its models are the latest real cutoff's.
    test_mode: bool = False
    test_poll_sec: float = 30.0  # how often to look for a test cycle to be closed

    @field_validator("url")
    @classmethod
    def _https(cls, url: str) -> str:
        # Requests carry the validator's signature and the private prompts and outputs: never in the clear,
        # except to a server on this machine.
        parts = urlsplit(url)
        local = parts.scheme == "http" and parts.hostname in ("127.0.0.1", "localhost") and not parts.username
        if not (parts.scheme == "https" and parts.hostname) and not local:
            raise ValueError(f"the dataless-api url must be https, got {url!r}")
        return url


class VllmCfg(BaseConfig):
    """The vLLM answers engine (scripts/prompt_validator_vllm/README.md)."""

    # The vLLM venv's interpreter, with connito_vllm installed, or a wrapper that execs it: never one that forks, as
    # the worker kills itself when the process that started it is gone.
    python: Path
    checkpoint: Path  # export_fp8row.py's output; its manifest.json's sha256 is in the rules hash
    version: str = "0.32.0"  # the vLLM the worker must run: in the rules hash
    gpu_memory_utilization: float = Field(0.90, gt=0, le=1)
    max_num_seqs: PositiveInt = 128  # answers decoded together: in the rules hash
    enforce_eager: bool = False  # no torch.compile or CUDA graphs: about 3x slower; in the rules hash
    # A worker that stops finishing requests is killed and the stage raises, so --loop retries rather than hangs.
    # start: engine start (load, torch.compile, CUDA graphs) to the first finished request; stall: between finished
    # requests, at any model size. Neither is in the rules hash.
    start_timeout_min: float = Field(20.0, gt=0)
    stall_timeout_min: float = Field(10.0, gt=0)

    @field_validator("python", "checkpoint")
    @classmethod
    def _absolute(cls, path: Path) -> Path:
        path = path.expanduser()
        if not path.is_absolute():
            raise ValueError(f"vllm paths must be absolute, got {path}")
        return path


# The judge's rules (judge.py): its questions, levels, thresholds and reason codes are data, in judge_rules.json, a
# private file the public export leaves out. Without it the fields below default to None, which nothing reads while
# judge_miner_prompts is off (pydantic validates no default), and the switch is refused on.
JUDGE_RULES_FILE = Path(__file__).with_name("judge_rules.json")


def _judge_rules() -> dict:
    try:
        return json.loads(JUDGE_RULES_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


JUDGE_RULES_DATA = _judge_rules()
_MINER = JUDGE_RULES_DATA.get("miner_prompts", {})
_CODES = _MINER.get("reason_codes") or {}
ON_TASK_QUESTION = _MINER.get("on_task_question")
ON_TASK_LEVELS = tuple(_MINER.get("on_task_levels") or ())
GIVEAWAY_QUESTION = _MINER.get("giveaway_question")
GIVEAWAY_LEVELS = tuple(_MINER.get("giveaway_levels") or ())


class ReasonCodes(BaseConfig):
    """The dataless-api's reason for a failed prompt, by the first check it failed: the owner's codes."""

    model_config = ConfigDict(extra="forbid")  # a misspelt key would quietly keep the default

    invalid: int = Field(_CODES.get("invalid"), ge=200, le=299)
    duplicate: int = Field(_CODES.get("duplicate"), ge=200, le=299)
    giveaway: int = Field(_CODES.get("giveaway"), ge=200, le=299)
    off_task: int = Field(_CODES.get("off_task"), ge=200, le=299)


class MinerPromptsCfg(BaseConfig):
    """How the judge stage judges miners' prompts (judge.py). All in the rules hash while judge_miner_prompts is on;
    the defaults are judge_rules.json's."""

    model_config = ConfigDict(extra="forbid")  # a misspelt key would quietly keep the default

    min_chars: PositiveInt = _MINER.get("min_chars")
    max_chars: PositiveInt | None = _MINER.get("max_chars")  # None: no limit past the dataless-api's
    min_unique_words: PositiveInt = _MINER.get("min_unique_words")
    max_self_repeat: float = Field(_MINER.get("max_self_repeat"), ge=0, le=1)
    dup_ngram: PositiveInt = _MINER.get("dup_ngram")
    dup_threshold: float = Field(_MINER.get("dup_threshold"), gt=0, le=1)
    on_task_question: str = ON_TASK_QUESTION
    on_task_levels: list[str] = Field(default_factory=lambda: list(ON_TASK_LEVELS), min_length=2)
    on_task_pass_level: int = Field(_MINER.get("on_task_pass_level"), ge=0)
    giveaway_question: str = GIVEAWAY_QUESTION
    giveaway_levels: list[str] = Field(default_factory=lambda: list(GIVEAWAY_LEVELS), min_length=2)
    giveaway_pass_level: int = Field(_MINER.get("giveaway_pass_level"), ge=0)
    prompts_per_call: PositiveInt = _MINER.get("prompts_per_call")
    reason_codes: ReasonCodes = Field(default_factory=ReasonCodes)
    qualifying_pass_share: float = Field(_MINER.get("qualifying_pass_share"), ge=0, le=1)
    max_answered: PositiveInt = _MINER.get("max_answered")
    request_file: Path | None = _MINER.get("request_file")  # test mode: the customer request, as JSON

    @model_validator(mode="after")
    def _pass_levels(self) -> MinerPromptsCfg:
        if not JUDGE_RULES_DATA:  # no rules to check: the switch is refused on (PromptValidatorCfg)
            return self
        for check in ("on_task", "giveaway"):
            levels, passing = getattr(self, f"{check}_levels"), getattr(self, f"{check}_pass_level")
            if passing >= len(levels):
                raise ValueError(f"{check}_pass_level {passing} is not one of the {len(levels)} {check}_levels")
        return self


class WeightsCfg(BaseConfig):
    """The weights stage (weights.py): a dry run that only reads the chain. Not in rules_sha256: it decides no score."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["off", "dry_run"] = "off"
    # The prompt validator's part of the miners' weight, applied after the subnet owner's share.
    prompt_share: float = Field(0.2, ge=0, le=1)

    @field_validator("mode", mode="before")
    @classmethod
    def _no_chain_weights(cls, mode: Any) -> Any:
        if mode is False:  # YAML reads a bare off as false
            return "off"
        if mode not in ("off", "dry_run"):
            got = "on" if mode is True else repr(mode)
            raise ValueError(f"weights.mode must be off or dry_run, got {got}: the prompt validator sets no chain "
                             "weights; the training validator stays the only submitter")
        return mode


class LossRankCfg(BaseConfig):
    """rank_by: loss (loss_rank.py). The rules hash covers the source and the ranking rule's numbers, not the pacing."""

    model_config = ConfigDict(extra="forbid")

    url: str = "https://dashboard-api-v2.connito.ai/api/v2/miner/{uid}/history?include_submissions=true"
    window_training_cycles: PositiveInt = 8  # the completed training cycles before the cutoff's that count
    min_samples: PositiveInt = 2  # counted rows a miner needs to be scored
    min_group: PositiveInt = 10  # rows a (cycle, validator slot) group needs for its median to count
    min_scored: PositiveInt = 50  # scored in-group miners needed, or the snapshot ranks by incentive
    validator_slots: list[int] = Field(default_factory=lambda: [2, 5])
    min_cycle: int = 16587  # no row from an earlier training cycle
    request_spacing_sec: float = Field(2.1, ge=2.0)  # between requests, retries too: under 30 a minute from one IP
    timeout_sec: float = Field(30.0, gt=0)  # per request
    retries: int = Field(3, ge=0)  # after the first attempt, on 429, 5xx, timeouts and connection errors
    backoff_sec: float = Field(5.0, ge=0)  # before the first retry, doubling after
    budget_sec: float = Field(1200.0, gt=0)  # the whole fetch, past which the snapshot ranks by incentive

    @field_validator("validator_slots")
    @classmethod
    def _slots(cls, slots: list[int]) -> list[int]:
        if not slots:
            raise ValueError("loss_rank.validator_slots must name at least one slot")
        return sorted(set(slots))  # so the same slots hash the same

    @field_validator("url")
    @classmethod
    def _uid_url(cls, url: str) -> str:
        if not url.startswith("https://") or "{uid}" not in url:
            raise ValueError(f"loss_rank.url must be https with a {{uid}} placeholder, got {url!r}")
        return url


class PromptValidatorCfg(BaseConfig):
    scoring_cycle_k: int = 13  # training cycles per scoring cycle: the dataless-api's prompt cycle is 13
    # Which models are scored: "top", the max_models model hashes ranked by rank_by; "all", every in-group miner
    # commit, with no ranking and no cap (the subnet's uids bound it), so max_models and rank_by decide nothing.
    model_selection: Literal["top", "all"] = "top"
    max_models: PositiveInt = 100  # top model hashes by incentive (0 too); rows include copies, so not bounded by it
    # How those model hashes are ranked: "loss" by the dashboard-api's training losses (loss_rank), else incentive.
    rank_by: Literal["incentive", "loss"] = "incentive"
    loss_rank: LossRankCfg = Field(default_factory=LossRankCfg)
    runs_dir: Path = Path("runs")
    # Where the prompts come from: the private prompt file at prompts_path, or the dataless-api's owner prompts.
    prompt_source: Literal["file", "dataless"] = "file"
    prompts_path: Path | None = None  # required with prompt_source "file": the private prompt file, named explicitly
    dataless: DatalessCfg = Field(default_factory=DatalessCfg)
    # --loop runs the cycle before the open one only with this much of its run-start window left (~22.7 h at most):
    # it must cover one cycle from cutoff to report. To be tuned from the timing run.
    catch_up_min_hours: float = 6.0
    enable_thinking: bool = False  # passed to apply_chat_template; False renders an empty think block
    max_new_tokens: PositiveInt = 1024  # per answer
    # Prompts answered together, left-padded. Padding can move a greedy answer, so it is in the rules hash. 20, one
    # task's draw, answers a task in one generate() call: each call decodes up to max_new_tokens steps in a row.
    answer_batch_size: PositiveInt = 20
    # "hf": generate() on the training validator's model; "vllm": the same weights through vLLM, in its own venv.
    engine: Literal["hf", "vllm"] = "hf"
    vllm: VllmCfg | None = None  # required with engine "vllm"
    # Fetch starts only with room for its downloads and this much left, so it never fills a disk the training
    # validator may share; it checks again before each download.
    min_free_disk_gb: NonNegativeFloat = 8.0
    expected_shard_gb: NonNegativeFloat = 0.3  # one miner shard, for that check: 241 MiB (0.253 GB) seen live
    write_root: Path = Path("/scratch")  # runs_dir must resolve under it: shards are written and deleted only there
    jev: JevCfg = Field(default_factory=JevCfg)
    # Reward rules (see reward.py). Not in rules_sha256: they decide who qualifies and the weights, no answer or
    # Jev score, so a change needs only --stage reward.
    reward_min_prompts: PositiveInt = 20  # scored prompts a miner needs to qualify for reward
    reward_top_miners: PositiveInt = 3  # only this many miners, by mean expected_score, get a weight
    # How the winners share weight 1: "proportional" to their means, or "equal".
    reward_shares: Literal["proportional", "equal"] = "proportional"
    set_weights: bool = False  # this phase sets no chain weights; true is refused
    # Judge the cycle's miner prompts (the judge stage). Off, every file and the rules hash are as before it existed.
    judge_miner_prompts: bool = False
    miner_prompts: MinerPromptsCfg = Field(default_factory=MinerPromptsCfg)
    weights: WeightsCfg = Field(default_factory=WeightsCfg)

    @field_validator("prompts_path")
    @classmethod
    def _absolute_prompts_path(cls, path: Path | None) -> Path | None:
        if path is None:
            return None
        path = path.expanduser()
        if not path.is_absolute():
            raise ValueError(f"prompts_path must be an absolute path to the private prompt file, got {path}")
        return path

    @field_validator("write_root")
    @classmethod
    def _absolute_write_root(cls, path: Path) -> Path:
        if not path.is_absolute():
            raise ValueError(f"write_root must be an absolute path, got {path}")
        return path

    @field_validator("set_weights")
    @classmethod
    def _no_chain_weights(cls, on: bool) -> bool:
        if on:
            raise ValueError("set_weights must be false: the prompt validator sets no chain weights in this phase")
        return on

    @model_validator(mode="after")
    def _prompts_path_for_a_file(self) -> PromptValidatorCfg:
        if self.prompt_source == "file" and self.prompts_path is None:
            raise ValueError('prompts_path is required with prompt_source "file"')
        if self.dataless.test_mode and self.prompt_source != "dataless":
            raise ValueError('dataless.test_mode needs prompt_source "dataless"')
        if self.judge_miner_prompts and self.prompt_source != "dataless":
            raise ValueError('judge_miner_prompts needs prompt_source "dataless": only it lists miner prompts')
        if self.judge_miner_prompts and not JUDGE_RULES_DATA:
            raise ValueError(f"judge_miner_prompts needs the judge's rules, {JUDGE_RULES_FILE.name}, which this "
                             "checkout does not have: the judge is not part of the public release")
        if self.engine == "vllm" and self.vllm is None:
            raise ValueError('engine "vllm" needs a vllm section: python, checkpoint')
        return self


def cache_dirs(engine: str) -> dict[str, Path]:
    """Where this process's libraries, and the vLLM worker's, write caches and temporary files: each variable if set,
    else its default."""
    env, home, tmp = os.environ.get, Path.home(), Path(tempfile.gettempdir())  # gettempdir reads TMPDIR
    # answers.run_worker passes this environment on with only secrets dropped, so the worker's vLLM reads these values.
    # vLLM 0.32's envs.py defaults, read with getenv as it reads them: an empty value is not the default.
    vllm = {
        "VLLM_CACHE_ROOT": env("VLLM_CACHE_ROOT", os.path.join(env("XDG_CACHE_HOME", home / ".cache"), "vllm")),
        "VLLM_CONFIG_ROOT": env("VLLM_CONFIG_ROOT", os.path.join(env("XDG_CONFIG_HOME", home / ".config"), "vllm")),
    } if engine == "vllm" else {}
    return {name: Path(path).expanduser() for name, path in {
        # huggingface_hub reads its variables on import: its constants are what it uses.
        "HF_HOME": huggingface_hub.constants.HF_HOME,
        "HF_HUB_CACHE": huggingface_hub.constants.HF_HUB_CACHE,
        "HF_XET_CACHE": huggingface_hub.constants.HF_XET_CACHE,
        "TRITON_CACHE_DIR": env("TRITON_CACHE_DIR") or Path(env("TRITON_HOME") or home) / ".triton" / "cache",
        "TORCHINDUCTOR_CACHE_DIR": env("TORCHINDUCTOR_CACHE_DIR") or tmp,
        "XDG_CACHE_HOME": env("XDG_CACHE_HOME") or home / ".cache",
        "CUDA_CACHE_PATH": env("CUDA_CACHE_PATH") or home / ".nv" / "ComputeCache",
        "TMPDIR": tmp,
        **vllm,
    }.items()}


def caches_outside(write_root: Path, engine: str) -> str | None:
    """Each cache dir outside write_root with its variable's value and where it resolves, or None if there is none."""
    outside = []
    for name, path in cache_dirs(engine).items():
        try:
            under_root(path, write_root)
        except ValueError:
            raw = os.environ.get(name)
            outside.append(f"{name} ({'unset, so the default' if raw is None else repr(raw)}) -> "
                           f"{os.path.realpath(path)}")
    if not outside:
        return None
    return (f"caches outside prompt_validator.write_root {os.path.realpath(write_root)}: {'; '.join(outside)}. "
            "Set each under it in the process environment")


class PromptValidatorChainCfg(ChainCfg):
    # Here only: in the shared ChainCfg it would be a new field the training validator's config prompts for.
    wallet_path: str = ""  # the directory holding the wallet; empty is bittensor's ~/.bittensor/wallets

    @field_validator("wallet_path")
    @classmethod
    def _absolute(cls, wallet_path: str) -> str:
        # A relative one would follow the working directory, and the hotkey's BT_PW_ name is spelled from the path.
        if wallet_path and not Path(wallet_path).expanduser().is_absolute():
            raise ValueError(f"chain.wallet_path must be absolute (~ is fine), got {wallet_path!r}")
        return wallet_path

    def wallet(self) -> bittensor.Wallet:
        # bittensor expands a leading ~ itself.
        path = {"path": self.wallet_path} if self.wallet_path else {}
        return bittensor.Wallet(name=self.coldkey_name, hotkey=self.hotkey_name, **path)


class PromptValidatorConfig(ValidatorConfig):
    # Skip the shared terminal prompts: they offer to reset the owner's periods to stale defaults and rewrite this YAML.
    _LOCKED_SECTIONS: ClassVar[tuple[str, ...]] = ()

    chain: PromptValidatorChainCfg = Field(default_factory=PromptValidatorChainCfg)
    prompt_validator: PromptValidatorCfg

    def model_post_init(self, __context: Any) -> None:
        # With no locked sections nothing else checks these, so a stale YAML would answer with another model.
        # Checked first, so a refused config makes no wallet or chain lookup.
        drift = {
            name: (getattr(self.model, name), default)
            for name, default in self.model.locked_defaults().items()
            if getattr(self.model, name) != default
        }
        if drift:
            raise ValueError(f"model fields are not the training validator's locked values (got, expected): {drift}")
        # The fp8 loader picks quantization_{role}, and the snapshot hashes quantization_validator.
        if self.role != "validator":
            raise ValueError(f"role must be 'validator', got {self.role!r}: it picks the fp8 scope of the answers")
        # bittensor would ask for an encrypted hotkey's password on the terminal, and an unattended run has none.
        hotkey = self.chain.wallet().hotkey_file
        if not hotkey.exists_on_device():
            raise ValueError(f"no hotkey at {hotkey.path}: check chain.wallet_path, coldkey_name and hotkey_name")
        if hotkey.is_encrypted():
            var = hotkey.env_var_name()
            try:
                # Decoded here and passed in, so bittensor can never fall back to asking; the key is dropped.
                password = get_password_from_environment(var)
                opened = bool(password) and hotkey.get_keypair(password) is not None
            except KeyfileError:  # WrongPasswordError is one too
                opened = False
            if not opened:
                raise ValueError(f"hotkey {hotkey.path} is encrypted and {var} does not open it: {var} must hold "
                                 "bittensor's encoded password (from save_password_to_environment), not the password")
        super().model_post_init(__context)
        # A relative runs_dir hangs off root_path like every other path; an absolute one replaces it.
        self.prompt_validator.runs_dir = self.run.root_path / self.prompt_validator.runs_dir.expanduser()
        under_root(self.prompt_validator.runs_dir, self.prompt_validator.write_root)
