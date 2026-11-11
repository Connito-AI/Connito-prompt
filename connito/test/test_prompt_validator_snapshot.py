import functools
import hashlib
import json
import logging
import re
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from bittensor.sp_core import Keypair
from pydantic import ValidationError

from connito.prompt_validator import cutoff, reward, run, snapshot
from connito.prompt_validator.config import PromptValidatorCfg, PromptValidatorConfig, VllmCfg
from connito.prompt_validator.stage_io import read_rows, stage_path, write_rows
from connito.shared.checkpoints import ChainCheckpoint, ChainCheckpoints
from connito.shared.config import CycleCfg, ModelCfg, ResidualCfg
from connito.shared.modeling.families import QWEN3_5_MOE
from connito.shared.schema import sign_message
from connito.test.test_prompt_validator_cutoff import OWNER_PERIODS

CYCLE_ID = 1258
CUTOFF = {"scoring_cycle_id": CYCLE_ID, "training_cycle": 17625, "commit_1_block": 9235871, "commit_2_block": 9235882}
T = CUTOFF["training_cycle"]
LOOKBACK = snapshot.LOOKBACK_TRAINING_CYCLES
CYCLE = CycleCfg(**OWNER_PERIODS)
CYCLE_LENGTH = 524  # OWNER_PERIODS' phases, on which CUTOFF's blocks are T's
ACTIVE_GROUP = 8
ACTIVE_RESIDUAL = ResidualCfg(k=1, threshold=0.8, temperature=0.05)
OLD_RESIDUAL = ResidualCfg(k=4, threshold=0.5, temperature=0.1)
TEMPLATE = b"{%- for message in messages %}{{ message.content }}{%- endfor %}\n"
EXPERT_GROUPS = Path(__file__).resolve().parents[2] / "expert_groups"


def _prompt(i, text="SECRET prompt"):
    return {
        "prompt_id": f"p{i}",
        "request_id": f"r{i % 2}",
        "prompt": f"{text} {i}",
        "request": {"goal": "SECRET goal", "examples": ["SECRET example"], "tone": "plain"},
    }


def _write_prompts(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _checkpoint(uid, group, incentive=0.0, *, miner_commit=True, model_hash=None):
    return ChainCheckpoint(
        uid=uid,
        incentive=incentive,
        miner_commit=miner_commit,
        hotkey=f"hk-{uid}",
        hf_repo_id=f"miner/repo-{uid}",
        hf_revision=f"rev{uid}",
        model_hash=model_hash or f"hash-{uid}",
        signed_model_hash=f"sig-{uid}",
        expert_group=group,
        global_ver=100,
        ip="1.2.3.4",
        port=8000,
        signature_required=True,
        hash_required=True,
    )


COMMIT_BLOCK_BASE = 9235000
COMMITMENT_OF = ("Commitments", "CommitmentOf")


def _registration(block, text='{"m":"sig"}'):
    """The commitments pallet's Registration as bittensor decodes it: the chain sets `block` when it is written."""
    return {"deposit": 0, "block": block, "info": {"fields": [{"Raw": text.encode().hex()}]}}


@pytest.fixture
def config(tmp_path):
    cfg = SimpleNamespace(
        prompt_validator=PromptValidatorCfg(runs_dir=tmp_path / "runs", prompts_path=tmp_path / "prompts.jsonl"),
        chain=SimpleNamespace(network="wss://archive.example:443", netuid=102),
        cycle=CYCLE,
        task=SimpleNamespace(expert_group_name="exp_old", exp=SimpleNamespace(group_id=3, residual=OLD_RESIDUAL)),
        model=ModelCfg(),
    )
    _write_prompts(cfg.prompt_validator.prompts_path, [_prompt(i) for i in range(3)])
    write_rows(stage_path(cfg.prompt_validator.runs_dir, CYCLE_ID, "cutoff"), [CUTOFF])
    return cfg


@pytest.fixture
def chain(monkeypatch, tmp_path):
    """A fake archive node serving `chain.checkpoints` at the cutoff and a fake Hub serving `chain.template`.

    `chain.tempo` or `chain.template` may be an exception to raise. The owner names `chain.owner_task` (None: the
    API is unreachable); switching onto it loads `chain.residual`, unless `chain.switch_fails`. The commitments
    pallet's records are `chain.registrations` (an exception to raise), by default one per checkpoint written at
    COMMIT_BLOCK_BASE + uid. Other training cycles' commits are `chain.at[training_cycle]` (none by default), with
    commitment records `chain.registrations_at[training_cycle]`, by default at 300 blocks before commit_1 + uid.
    Blocks in `chain.pruned` are discarded: a commits read there gets the head's, as get_chain_commits does, unless it
    asks for no head fallback.
    """
    state = SimpleNamespace(
        tempo=360,
        checkpoints=[_checkpoint(5, ACTIVE_GROUP, 0.2), _checkpoint(2, ACTIVE_GROUP, 0.4), _checkpoint(9, 3, 0.9)],
        reads=[],
        synced=[],
        template=TEMPLATE,
        downloads=[],
        residual=ACTIVE_RESIDUAL,  # the owner's task's
        owner_task="exp_test",
        switch_fails=False,
        registrations=None,
        commitment_reads=[],
        at={},
        registrations_at={},
        pruned=set(),
    )

    def training_cycle(block):
        return (block - CUTOFF["commit_1_block"]) // CYCLE_LENGTH + T

    def checkpoints_at(block):
        t = training_cycle(block)
        return state.at.get(t, state.checkpoints if t == T else [])

    def unpruned(block):
        if block in state.pruned:
            raise Exception(f"State already discarded for BlockId::Number({block})")

    class FakeSubtensor:
        def __init__(self, network):
            self.network = network

        def tempo(self, netuid, block=None):
            if isinstance(state.tempo, Exception):
                raise state.tempo
            return state.tempo

        def metagraph(self, *args, **kwargs):
            raise AssertionError("the snapshot read a live metagraph instead of the cutoff block's")

        def query_map(self, item, params, block=None):
            state.commitment_reads.append((item, params, block))
            unpruned(block)
            if isinstance(state.registrations, Exception):
                raise state.registrations
            t = training_cycle(block)
            if t in state.registrations_at:
                return list(state.registrations_at[t])
            if t == T and state.registrations is not None:
                return list(state.registrations)
            base = COMMIT_BLOCK_BASE if t == T else block - 300
            return [(c.hotkey, _registration(base + c.uid)) for c in checkpoints_at(block)]

    def fake_build(config, subtensor, *, commit_1_block, commit_2_block, for_role, head_fallback=True):
        if state.pruned & {commit_1_block, commit_2_block}:
            if not head_fallback:
                unpruned(min(state.pruned & {commit_1_block, commit_2_block}))
            return ChainCheckpoints(checkpoints=list(state.checkpoints))  # the head's
        state.reads.append((commit_1_block, commit_2_block, for_role))
        return ChainCheckpoints(checkpoints=list(checkpoints_at(commit_1_block)))

    storage = SimpleNamespace(Commitments=SimpleNamespace(CommitmentOf=COMMITMENT_OF))
    monkeypatch.setattr(snapshot, "bittensor", SimpleNamespace(Subtensor=FakeSubtensor, storage=storage))
    monkeypatch.setattr(
        snapshot.bt_compat, "get_tempo", lambda sub, netuid, block=None: sub.tempo(netuid, block=block),
    )
    monkeypatch.setattr(snapshot, "build_chain_checkpoints_at_blocks", fake_build)
    def fake_get_active_task(cycle):
        return None if state.owner_task is None else SimpleNamespace(name=state.owner_task)

    def fake_ensure(cfg, active):
        """The owner's task moved on since boot: switch the config onto it, or log and stay, as the real one does."""
        state.synced.append((cfg, active))
        if active != cfg.task.expert_group_name and not state.switch_fails:
            cfg.task.expert_group_name = active
            cfg.task.exp = SimpleNamespace(group_id=ACTIVE_GROUP, residual=state.residual)

    monkeypatch.setattr(snapshot, "get_active_task", fake_get_active_task)
    monkeypatch.setattr(snapshot, "ensure_active_task", fake_ensure)

    def fake_hf_hub_download(repo_id, filename, *, revision):
        """The Hub's file at that revision, written where its cache would hold it."""
        state.downloads.append((repo_id, filename, revision))
        if isinstance(state.template, Exception):
            raise state.template
        path = tmp_path / "hub" / filename
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(state.template)
        return str(path)

    monkeypatch.setattr(snapshot, "hf_hub_download", fake_hf_hub_download)
    return state


def _models(config):
    return read_rows(stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "snapshot").parent / "models.jsonl")


def test_other_groups_are_dropped(config, chain):
    [meta] = snapshot.run_snapshot(config, CYCLE_ID)
    models = _models(config)

    assert chain.synced == [(config, "exp_test")]
    # The cutoff's commits first, then the lookback's oldest first; the cutoff's own from the cache by then.
    assert chain.reads == [(CUTOFF["commit_1_block"], CUTOFF["commit_2_block"], "miner")] + [
        (b.commit_1_block, b.commit_2_block, "miner")
        for b in (cutoff.training_cycle_blocks(config, CYCLE_ID, t) for t in range(T - LOOKBACK, T))
    ]
    assert meta["originals_training_cycles"] == [T - LOOKBACK, T]
    assert [(m["uid"], m["incentive"]) for m in models] == [(2, 0.4), (5, 0.2)]
    assert all(list(m) == [*snapshot.MODEL_FIELDS, "incentive", "signed_commit_block", "first_seen_training_cycle",
                                 "first_seen_hotkey"] for m in models)
    # How fetch rebuilds a checkpoint for ChainCheckpoint.validate.
    rebuilt = ChainCheckpoint(**models[0], signature_required=True, hash_required=True)
    assert (rebuilt.uid, rebuilt.hotkey, rebuilt.model_hash, rebuilt.signed_model_hash) == (
        2, "hk-2", "hash-2", "sig-2"
    )
    assert rebuilt.expert_group == ACTIVE_GROUP
    assert rebuilt.signature_required and rebuilt.hash_required
    assert (meta["active_task"], meta["expert_group"], meta["max_models"]) == ("exp_test", ACTIVE_GROUP, 100)
    assert (meta["commit_1_block"], meta["commit_2_block"]) == (CUTOFF["commit_1_block"], CUTOFF["commit_2_block"])
    assert (meta["base_model"], meta["base_model_revision"]) == (QWEN3_5_MOE.model_path, QWEN3_5_MOE.revision)


def test_snapshot_hash_is_stable_and_follows_the_prompts(config, chain):
    first = snapshot.run_snapshot(config, CYCLE_ID)
    assert snapshot.run_snapshot(config, CYCLE_ID) == first

    prompts = [_prompt(i) for i in range(3)]
    prompts[1]["prompt"] += " edited"
    _write_prompts(config.prompt_validator.prompts_path, prompts)

    assert snapshot.run_snapshot(config, CYCLE_ID)[0]["snapshot_sha256"] != first[0]["snapshot_sha256"]


def test_snapshot_and_rules_hashes_follow_max_models_even_with_the_same_models(config, chain):
    first = snapshot.run_snapshot(config, CYCLE_ID)[0]
    config.prompt_validator.max_models = 5

    second = snapshot.run_snapshot(config, CYCLE_ID)[0]

    assert len(_models(config)) == 2
    assert second["snapshot_sha256"] != first["snapshot_sha256"]
    assert second["rules_sha256"] != first["rules_sha256"]
    assert second["rules_sha256"] == snapshot.rules_sha256(config.prompt_validator, _base())


def test_snapshot_keeps_the_highest_incentives_at_the_cutoff(config, chain):
    chain.checkpoints = [
        _checkpoint(1, ACTIVE_GROUP, 0.1), _checkpoint(2, ACTIVE_GROUP, 0.5), _checkpoint(3, ACTIVE_GROUP, 0.3),
        _checkpoint(9, 3, 0.9),
    ]
    config.prompt_validator.max_models = 2

    [meta] = snapshot.run_snapshot(config, CYCLE_ID)

    assert [(m["uid"], m["incentive"]) for m in _models(config)] == [(2, 0.5), (3, 0.3)]
    assert meta["max_models"] == 2


def test_equal_incentives_keep_the_lower_uid(config, chain):
    chain.checkpoints = [
        _checkpoint(4, ACTIVE_GROUP, 0.3), _checkpoint(3, ACTIVE_GROUP, 0.3), _checkpoint(7, ACTIVE_GROUP, 0.6),
    ]
    config.prompt_validator.max_models = 2

    snapshot.run_snapshot(config, CYCLE_ID)

    assert [m["uid"] for m in _models(config)] == [3, 7]


def test_copies_of_one_model_take_one_place_and_are_all_kept(config, chain, caplog):
    # Ranked by commit, the top two would be uids 1 and 4, both hash "a": hash "c" would lose its place to a copy.
    chain.checkpoints = [
        _checkpoint(1, ACTIVE_GROUP, 0.5, model_hash="a"), _checkpoint(4, ACTIVE_GROUP, 0.4, model_hash="a"),
        _checkpoint(3, ACTIVE_GROUP, 0.3, model_hash="c"), _checkpoint(6, ACTIVE_GROUP, 0.0, model_hash="a"),
        _checkpoint(2, ACTIVE_GROUP, 0.2, model_hash="b"), _checkpoint(5, ACTIVE_GROUP, 0.1, model_hash="b"),
    ]
    config.prompt_validator.max_models = 2

    with caplog.at_level(logging.INFO):
        snapshot.run_snapshot(config, CYCLE_ID)

    assert [(m["uid"], m["model_hash"]) for m in _models(config)] == [(1, "a"), (3, "c"), (4, "a"), (6, "a")]
    line = next(re.sub(r"\x1b\[[0-9;]*m", "", r.getMessage()) for r in caplog.records
                if "models kept by incentive" in r.getMessage())
    for field in ("kept=4", "model_hashes=2", "copies_kept=2", "dropped_over_max=2", "zero_incentive_kept=1",
                  "max_models=2"):
        assert field in line


def test_a_model_ranks_by_its_best_commit(config, chain):
    # Hash "a"'s best commit (uid 9, 0.9) puts it first, though its other commit has 0 incentive.
    chain.checkpoints = [
        _checkpoint(1, ACTIVE_GROUP, 0.0, model_hash="a"), _checkpoint(2, ACTIVE_GROUP, 0.5, model_hash="b"),
        _checkpoint(9, ACTIVE_GROUP, 0.9, model_hash="a"),
    ]
    config.prompt_validator.max_models = 1

    snapshot.run_snapshot(config, CYCLE_ID)

    assert [m["uid"] for m in _models(config)] == [1, 9]


def test_equal_incentives_keep_the_model_with_the_lower_uid(config, chain):
    # Hash "x" ties hash "y" at 0.3; x's commit by uid 2 is below y's uid 3, so x is kept, with its uid 7 copy.
    chain.checkpoints = [
        _checkpoint(7, ACTIVE_GROUP, 0.3, model_hash="x"), _checkpoint(3, ACTIVE_GROUP, 0.3, model_hash="y"),
        _checkpoint(2, ACTIVE_GROUP, 0.3, model_hash="x"),
    ]
    config.prompt_validator.max_models = 1

    snapshot.run_snapshot(config, CYCLE_ID)

    assert [m["uid"] for m in _models(config)] == [2, 7]


def test_commits_without_a_model_hash_are_not_copies_of_each_other(config, chain):
    chain.checkpoints = [_checkpoint(uid, ACTIVE_GROUP, 0.1 * uid) for uid in (1, 2, 3)]
    for c in chain.checkpoints[1:]:
        c.model_hash = None
    config.prompt_validator.max_models = 2

    snapshot.run_snapshot(config, CYCLE_ID)

    assert [(m["uid"], m["model_hash"]) for m in _models(config)] == [(2, None), (3, None)]


def test_max_models_caps_the_distinct_models(config, chain):
    # 150 models with incentives 0.001 to 0.150, each with a copy at 0 incentive under uid + 1000.
    chain.checkpoints = [_checkpoint(uid, ACTIVE_GROUP, uid / 1000) for uid in range(1, 151)] + [
        _checkpoint(uid + 1000, ACTIVE_GROUP, 0.0, model_hash=f"hash-{uid}") for uid in range(1, 151)
    ]

    snapshot.run_snapshot(config, CYCLE_ID)

    models = _models(config)
    assert config.prompt_validator.max_models == 100
    assert {m["model_hash"] for m in models} == {f"hash-{uid}" for uid in range(51, 151)}
    assert len(models) == 200


def test_snapshot_hash_follows_the_models_kept(config, chain):
    config.prompt_validator.max_models = 1
    first = snapshot.run_snapshot(config, CYCLE_ID)[0]
    # The same commits, so the meta is the same; only uid 2's incentive drops below uid 5's, so uid 5 is kept.
    chain.checkpoints = [_checkpoint(5, ACTIVE_GROUP, 0.2), _checkpoint(2, ACTIVE_GROUP, 0.0), _checkpoint(9, 3, 0.9)]

    second = snapshot.run_snapshot(config, CYCLE_ID)[0]

    assert [m["uid"] for m in _models(config)] == [5]
    assert {k: v for k, v in second.items() if k != "snapshot_sha256"} == {
        k: v for k, v in first.items() if k != "snapshot_sha256"
    }
    assert second["snapshot_sha256"] != first["snapshot_sha256"]
    assert second["rules_sha256"] == first["rules_sha256"]


def test_answer_defaults_are_greedy_without_thinking_and_1024_tokens(tmp_path):
    pv = PromptValidatorCfg(prompts_path=tmp_path / "prompts.jsonl")

    assert (pv.enable_thinking, pv.max_new_tokens, pv.max_models) == (False, 1024, 100)
    assert snapshot.GREEDY == {"do_sample": False}


def test_fewer_models_than_max_models_are_all_kept(config, chain):
    chain.checkpoints = [_checkpoint(uid, ACTIVE_GROUP, 0.1 * uid) for uid in (1, 2, 3)]

    snapshot.run_snapshot(config, CYCLE_ID)

    assert config.prompt_validator.max_models == 100
    assert [m["uid"] for m in _models(config)] == [1, 2, 3]


@pytest.mark.parametrize("incentives, max_models, kept", [
    ({1: 0.0, 2: 0.2}, 10, [1, 2]),
    ({1: 0.0, 2: 0.0}, 10, [1, 2]),
    ({3: 0.0, 1: 0.0, 2: 0.2}, 2, [1, 2]),  # zero incentives fill the places left, the lower uid first
    ({1: 0.0, 2: 0.2, 3: 0.1}, 2, [2, 3]),  # and never take one from an incentive above 0
])
def test_a_zero_incentive_model_is_kept_within_max_models(config, chain, incentives, max_models, kept):
    chain.checkpoints = [_checkpoint(uid, ACTIVE_GROUP, incentive) for uid, incentive in incentives.items()]
    config.prompt_validator.max_models = max_models

    snapshot.run_snapshot(config, CYCLE_ID)

    assert [m["uid"] for m in _models(config)] == kept


@pytest.mark.parametrize("bad_line, fields", [
    ("{not json", "not JSON"),
    (json.dumps({k: v for k, v in _prompt(1).items() if k != "request_id"}), "['request_id']"),
    (json.dumps({**_prompt(1), "label": "honest"}), "['label']"),
    (json.dumps({**_prompt(1), "request": {**_prompt(1)["request"], "examples": "SECRET"}}), "['request.examples']"),
    (json.dumps({**_prompt(1), "prompt_id": 1}), "['prompt_id']"),
    (json.dumps(["SECRET prompt"]), "['row']"),
    (json.dumps(_prompt(0)), "duplicate prompt_id p0"),
])
def test_bad_prompt_row_is_rejected_with_its_line(tmp_path, bad_line, fields):
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps(_prompt(0)) + "\n\n" + bad_line + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        snapshot.load_internal_prompts(path)

    assert str(excinfo.value).startswith(f"{path}:3: ")
    assert fields in str(excinfo.value)
    assert "SECRET" not in "".join(traceback.format_exception(excinfo.value))


def _base(**changes):
    base = {
        "base_model": QWEN3_5_MOE.model_path,
        "base_model_revision": QWEN3_5_MOE.revision,
        "chat_template_sha256": hashlib.sha256(TEMPLATE).hexdigest(),
        "quantization": "fp8",
        "quantization_validator": "experts",
        "residual": ACTIVE_RESIDUAL.model_dump(),
    }
    return {**base, **changes}


@pytest.mark.parametrize("changes", [
    {"base_model_revision": "0" * 40},
    {"base_model": "Qwen/Qwen3.6-35B-A3B-FP8"},
    {"chat_template_sha256": "0" * 64},
    {"quantization": "off"},
    {"quantization_validator": "all"},
    {"residual": {**ACTIVE_RESIDUAL.model_dump(), "k": 2}},
    {"residual": {**ACTIVE_RESIDUAL.model_dump(), "threshold": 0.7}},
    {"residual": {**ACTIVE_RESIDUAL.model_dump(), "temperature": 0.1}},
])
def test_rules_hash_follows_the_base_model(tmp_path, changes):
    pv = PromptValidatorCfg(prompts_path=tmp_path / "prompts.jsonl")

    assert snapshot.rules_sha256(pv, _base(**changes)) != snapshot.rules_sha256(pv, _base())


@pytest.mark.parametrize("changes", [{"enable_thinking": True}, {"max_new_tokens": 2048}])
def test_rules_hash_follows_the_answer_settings(tmp_path, changes):
    pv = PromptValidatorCfg(prompts_path=tmp_path / "prompts.jsonl")

    assert snapshot.rules_sha256(pv.model_copy(update=changes), _base()) != snapshot.rules_sha256(pv, _base())


def test_rules_hash_follows_max_models_and_the_selection_rule(tmp_path, monkeypatch):
    pv = PromptValidatorCfg(prompts_path=tmp_path / "prompts.jsonl")
    rules = snapshot.rules_sha256(pv, _base())

    assert snapshot.rules_sha256(pv.model_copy(update={"max_models": 10}), _base()) != rules
    monkeypatch.setattr(snapshot, "MODEL_SELECTION", "another rule")
    assert snapshot.rules_sha256(pv, _base()) != rules


def test_rules_hash_follows_the_copy_rule(tmp_path, monkeypatch):
    pv = PromptValidatorCfg(prompts_path=tmp_path / "prompts.jsonl")
    rules = snapshot.rules_sha256(pv, _base())

    monkeypatch.setattr(snapshot, "COPY_REWARD", "another rule")
    assert snapshot.rules_sha256(pv, _base()) != rules


def test_each_model_records_its_on_chain_commitment_block_not_its_payload(config, chain, caplog):
    # uid 2's payload claims block 1 ("b"), the miner's own field; the chain wrote its commitment at 9235800.
    # uid 5 has no record with a block: null, never a guess.
    chain.registrations = [
        ("hk-2", _registration(9235800, '{"m":"sig-2","b":1}')),
        ("hk-5", {"deposit": 0, "info": {"fields": [{"Raw": "7b7d"}]}}),
        ("hk-9", _registration(9235700)),
    ]
    with caplog.at_level(logging.INFO):
        snapshot.run_snapshot(config, CYCLE_ID)

    assert chain.commitment_reads[0] == (COMMITMENT_OF, [102], CUTOFF["commit_1_block"])
    assert [(m["uid"], m["signed_commit_block"]) for m in _models(config)] == [(2, 9235800), (5, None)]
    line = next(re.sub(r"\x1b\[[0-9;]*m", "", r.getMessage()) for r in caplog.records
                if "models kept by incentive" in r.getMessage())
    assert "no_commit_block=1" in line


@pytest.mark.parametrize("error, match", [
    (Exception("State already discarded for BlockId::Number(9235871)"), "pruned block 9235871"),
    (ConnectionError("socket closed"), "socket closed"),
])
def test_a_commitment_read_that_fails_writes_no_snapshot(config, chain, error, match):
    chain.registrations = error

    with pytest.raises((RuntimeError, ConnectionError), match=match):
        snapshot.run_snapshot(config, CYCLE_ID)
    assert chain.reads == []
    assert not stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "snapshot").parent.joinpath(
        "models.jsonl").exists()


def test_a_later_stage_stops_on_another_max_models(config, chain):
    [meta] = snapshot.run_snapshot(config, CYCLE_ID)
    write_rows(stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "snapshot"), [meta])
    config.prompt_validator.max_models = 10

    with pytest.raises(RuntimeError, match="rules_sha256"):
        snapshot.check_rules(config, meta)


def test_rules_hash_follows_the_decoding(tmp_path, monkeypatch):
    pv = PromptValidatorCfg(prompts_path=tmp_path / "prompts.jsonl")
    greedy = snapshot.rules_sha256(pv, _base())
    monkeypatch.setattr(snapshot, "GREEDY", {"do_sample": True})

    assert snapshot.rules_sha256(pv, _base()) != greedy


def test_rules_hash_ignores_what_decides_no_score(tmp_path):
    pv = PromptValidatorCfg(prompts_path=tmp_path / "prompts.jsonl")
    rules = snapshot.rules_sha256(pv, _base())

    # Grouping and paths decide no score, so validators that differ only there agree.
    assert snapshot.rules_sha256(pv.model_copy(update={"scoring_cycle_k": 15}), _base()) == rules
    disk = {"min_free_disk_gb": 100.0, "expected_shard_gb": 3.0, "write_root": tmp_path}
    assert snapshot.rules_sha256(pv.model_copy(update=disk), _base()) == rules
    moved = pv.model_copy(update={"runs_dir": tmp_path / "elsewhere", "prompts_path": tmp_path / "other.jsonl"})
    assert snapshot.rules_sha256(moved, _base()) == rules


def test_snapshot_records_the_family_base_with_the_active_tasks_residual(config, chain):
    [meta] = snapshot.run_snapshot(config, CYCLE_ID)

    assert chain.downloads == [(QWEN3_5_MOE.model_path, "chat_template.jinja", QWEN3_5_MOE.revision)]
    assert {key: meta[key] for key in _base()} == _base()
    assert meta["residual"] != OLD_RESIDUAL.model_dump()
    assert meta["rules_sha256"] == snapshot.rules_sha256(config.prompt_validator, _base())


def test_rules_hash_follows_the_template_bytes(config, chain):
    first = snapshot.run_snapshot(config, CYCLE_ID)[0]
    chain.template = TEMPLATE + b"{{- '<think>' }}"

    second = snapshot.run_snapshot(config, CYCLE_ID)[0]

    assert second["chat_template_sha256"] == hashlib.sha256(chain.template).hexdigest()
    assert second["rules_sha256"] != first["rules_sha256"]


def test_rules_hash_follows_the_active_tasks_residual(config, chain):
    first = snapshot.run_snapshot(config, CYCLE_ID)[0]
    # The owner moves on to a task with another residual gate (same group, so the commits still match).
    chain.owner_task, chain.residual = "exp_test_k2", ResidualCfg(k=2, threshold=0.8, temperature=0.05)

    second = snapshot.run_snapshot(config, CYCLE_ID)[0]

    assert second["residual"]["k"] == 2
    assert second["rules_sha256"] != first["rules_sha256"]


def test_template_download_failure_writes_no_snapshot(config, chain):
    chain.template = OSError("404 Client Error: Entry Not Found for url")

    with pytest.raises(OSError, match="Entry Not Found"):
        run.run_stages(config, CYCLE_ID, stage="snapshot")

    out_dir = stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "snapshot").parent
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_rules_hash_covers_the_jev_pin_and_questions(tmp_path, monkeypatch):
    base = PromptValidatorCfg(prompts_path=tmp_path / "prompts.jsonl")
    model = _base()  # the base model rows of the meta
    rules = snapshot.rules_sha256(base, model)

    def with_jev(**update):
        return base.model_copy(update={"jev": base.jev.model_copy(update=update)})

    assert snapshot.rules_sha256(with_jev(resolved="jev-1.15.0"), model) != rules
    # Throughput, budget, endpoint and retry policy decide no score.
    unchanged = with_jev(
        concurrency=32, spend_cap_usd=50.0, price_per_m=1.0, url="https://jev.example/v1", timeout_sec=5, retries=1,
        max_answers_per_call=3,
    )
    assert snapshot.rules_sha256(unchanged, model) == rules
    monkeypatch.setitem(snapshot.SCORE_QUESTIONS["quality"], "criteria", ["0 Bad", "1 Good"])
    assert snapshot.rules_sha256(base, model) != rules


@pytest.mark.parametrize("tempo, error, match", [
    (Exception("State already discarded for BlockId::Number(9235871)"), RuntimeError, "pruned block 9235871"),
    (None, RuntimeError, "subnet 102 absent at block 9235871"),
    (ConnectionError("socket closed"), ConnectionError, "socket closed"),
])
def test_archive_probe_stops_before_reading_commits(config, chain, tempo, error, match):
    chain.tempo = tempo

    with pytest.raises(error, match=match):
        snapshot.run_snapshot(config, CYCLE_ID)
    assert chain.reads == []


@pytest.mark.parametrize("groups", [[3, 3, ACTIVE_GROUP], []])
def test_snapshot_keeps_the_active_group_however_few_commit_to_it(config, chain, groups):
    chain.checkpoints = [_checkpoint(uid, group, 0.5) for uid, group in enumerate(groups)]

    [meta] = snapshot.run_snapshot(config, CYCLE_ID)

    assert [m["uid"] for m in _models(config)] == [uid for uid, group in enumerate(groups) if group == ACTIVE_GROUP]
    assert meta["expert_group"] == ACTIVE_GROUP
    assert meta["commits_by_group"] == {str(group): groups.count(group) for group in sorted(set(groups))}


def test_snapshot_stops_when_only_other_groups_committed(config, chain):
    chain.checkpoints = [_checkpoint(uid, 3, 0.5) for uid in range(2)]

    with pytest.raises(RuntimeError, match=f"task exp_test's group {ACTIVE_GROUP}, but others do: {{'3': 2}}"):
        run.run_stages(config, CYCLE_ID, stage="snapshot")
    assert not stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "snapshot").exists()


def test_stage_writes_three_owner_only_files_and_a_cache_per_training_cycle(config, chain):
    run.run_stages(config, CYCLE_ID, stage="snapshot")

    out_dir = stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "snapshot").parent
    counts = {p.name: len(read_rows(p)) for p in out_dir.iterdir()}
    assert counts == {"prompts.jsonl": 3, "models.jsonl": 2, "snapshot.jsonl": 1}
    # Only the cutoff's: the lookback's training cycles had no commits, which are read again rather than cached.
    cache = list((config.prompt_validator.runs_dir / "chain_commits").iterdir())
    assert [p.name for p in cache] == [f"{CUTOFF['commit_1_block']}-{CUTOFF['commit_2_block']}.jsonl"]
    assert {p.stat().st_mode & 0o777 for p in [*out_dir.iterdir(), *cache]} == {0o600}
    assert set(read_rows(out_dir / "snapshot.jsonl")[0]) >= {"snapshot_sha256", "rules_sha256", "base_model"}


def test_config_without_prompts_path_does_not_load(tmp_path):
    with pytest.raises(ValidationError, match="prompts_path"):
        PromptValidatorConfig(
            run={"root_path": tmp_path},
            chain={"coldkey_ss58": "cold", "hotkey_ss58": "hot"},  # skips the chain lookup
            task={"base_path": EXPERT_GROUPS},
            prompt_validator={"runs_dir": "runs"},
        )


def _config_with_prompts_path(tmp_path, prompts_path, chain):
    return PromptValidatorConfig(
        run={"root_path": tmp_path},
        chain=chain,
        task={"base_path": EXPERT_GROUPS},
        prompt_validator={"prompts_path": prompts_path, "write_root": tmp_path},
    )


@pytest.mark.parametrize("prompts_path, expected", [
    ("~/prompts.jsonl", lambda tmp_path: tmp_path / "home" / "prompts.jsonl"),
    ("/abs/prompts.jsonl", lambda tmp_path: Path("/abs/prompts.jsonl")),
])
def test_prompts_path_expands_home(tmp_path, monkeypatch, pv_chain, prompts_path, expected):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    config = _config_with_prompts_path(tmp_path, prompts_path, pv_chain)

    assert config.prompt_validator.prompts_path == expected(tmp_path)


def test_relative_prompts_path_is_rejected(tmp_path, pv_chain):
    with pytest.raises(ValidationError, match="prompts_path must be an absolute path"):
        _config_with_prompts_path(tmp_path, "internal_prompts.jsonl", pv_chain)


@pytest.mark.parametrize("model, field", [
    ({"family": "deepseek_v2"}, "family"),
    ({"quantization": "off"}, "quantization"),
    ({"quantization_validator": "all"}, "quantization_validator"),
])
def test_config_with_another_locked_model_field_does_not_load(tmp_path, monkeypatch, model, field):
    def no_lookup(self):
        raise AssertionError("a refused config looked up its wallet")

    monkeypatch.setattr(PromptValidatorConfig, "_fill_wallet_data", no_lookup)
    with pytest.raises(ValueError, match=f"locked values .*'{field}'"):
        PromptValidatorConfig(
            run={"root_path": tmp_path},
            chain={"coldkey_ss58": "cold", "hotkey_ss58": "hot"},
            task={"base_path": EXPERT_GROUPS},
            model=model,
            prompt_validator={"prompts_path": "/abs/prompts.jsonl"},
        )


def test_config_on_the_locked_model_loads(tmp_path, pv_chain):
    config = _config_with_prompts_path(tmp_path, "/abs/prompts.jsonl", pv_chain)

    assert config.model.family == QWEN3_5_MOE.name


@pytest.mark.parametrize("max_new_tokens", [0, -1])
def test_max_new_tokens_must_be_positive(tmp_path, max_new_tokens):
    with pytest.raises(ValidationError, match="max_new_tokens"):
        PromptValidatorCfg(prompts_path=tmp_path / "prompts.jsonl", max_new_tokens=max_new_tokens)


@pytest.mark.parametrize("max_models", [0, -1])
def test_max_models_below_one_does_not_load(tmp_path, max_models):
    with pytest.raises(ValidationError, match="max_models"):
        PromptValidatorCfg(prompts_path=tmp_path / "prompts.jsonl", max_models=max_models)


@pytest.mark.parametrize("owner_task, switch_fails, match", [
    (None, False, "owner API unreachable: cannot confirm the active task \\(config is on exp_old\\)"),
    ("exp_test", True, "could not switch onto the owner's task exp_test: still on exp_old"),
])
def test_snapshot_stops_when_the_owners_task_is_not_confirmed(config, chain, owner_task, switch_fails, match):
    # A lagging commit for the stale group would pass the group check, so the stale task must stop here.
    chain.owner_task, chain.switch_fails = owner_task, switch_fails
    chain.checkpoints = [_checkpoint(uid, ACTIVE_GROUP, 0.5) for uid in range(5)] + [_checkpoint(9, 3, 0.5)]

    with pytest.raises(RuntimeError, match=match):
        run.run_stages(config, CYCLE_ID, stage="snapshot")
    assert chain.reads == []
    assert not stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "snapshot").exists()


def test_validator_commits_are_neither_kept_nor_counted(config, chain):
    chain.checkpoints = [
        _checkpoint(1, ACTIVE_GROUP, 0.9, miner_commit=False),
        _checkpoint(2, ACTIVE_GROUP, 0.4),
        _checkpoint(9, 3, 0.5),
    ]

    [meta] = snapshot.run_snapshot(config, CYCLE_ID)

    assert [m["uid"] for m in _models(config)] == [2]
    assert meta["commits_by_group"] == {"3": 1, str(ACTIVE_GROUP): 1}


def test_a_validator_commit_alone_does_not_pass_the_group_check(config, chain):
    chain.checkpoints = [_checkpoint(1, ACTIVE_GROUP, 0.9, miner_commit=False), _checkpoint(9, 3, 0.5)]

    with pytest.raises(RuntimeError, match="but others do: {'3': 1}"):
        snapshot.run_snapshot(config, CYCLE_ID)


def test_answer_settings_are_the_arguments_the_answers_stage_passes(tmp_path):
    pv = PromptValidatorCfg(prompts_path=tmp_path / "prompts.jsonl", enable_thinking=True, max_new_tokens=256,
                            answer_batch_size=4)

    assert snapshot.answer_settings(pv) == {
        "engine": "hf",
        "chat_template": {"enable_thinking": True},
        "generate": {"do_sample": False, "max_new_tokens": 256},
        "batch_size": 4,
    }


def test_check_rules_passes_on_the_snapshot_config_and_stops_on_another(config, chain):
    [meta] = snapshot.run_snapshot(config, CYCLE_ID)
    snapshot.check_rules(config, meta)

    config.prompt_validator.max_new_tokens = 2048
    with pytest.raises(RuntimeError, match="is not the snapshot's"):
        snapshot.check_rules(config, meta)


def test_config_with_a_role_other_than_validator_does_not_load(tmp_path, monkeypatch):
    monkeypatch.setattr(PromptValidatorConfig, "_fill_wallet_data", lambda self: None)
    with pytest.raises(ValueError, match="role must be 'validator', got 'miner'"):
        PromptValidatorConfig(
            role="miner",
            run={"root_path": tmp_path},
            chain={"coldkey_ss58": "cold", "hotkey_ss58": "hot"},
            task={"base_path": EXPERT_GROUPS},
            prompt_validator={"prompts_path": tmp_path / "prompts.jsonl"},
        )


def test_the_hf_engines_rules_hash_is_pinned_and_ignores_a_vllm_section(tmp_path):
    # With engine "hf", a vllm section changes no snapshot's rules.
    pv = PromptValidatorCfg(prompts_path=Path("/prompts.jsonl"))
    assert snapshot.rules_sha256(pv, _base()) == "18df5bb8e0ca3d85179c4b4db7c11486d67c4f8df1e721a675067377a8cb1799"
    vllm = VllmCfg(python=tmp_path / "python", checkpoint=tmp_path)
    assert snapshot.rules_sha256(pv.model_copy(update={"vllm": vllm}), _base()) == snapshot.rules_sha256(pv, _base())


def test_the_vllm_engines_rules_hash_follows_its_version_checkpoint_batch_and_eager_mode_only(tmp_path):
    (tmp_path / "manifest.json").write_text('{"tensors": {}}')
    vllm = VllmCfg(python=tmp_path / "python", checkpoint=tmp_path)
    pv = PromptValidatorCfg(prompts_path=Path("/prompts.jsonl"), engine="vllm", vllm=vllm)
    rules = snapshot.rules_sha256(pv, _base())

    assert rules != snapshot.rules_sha256(pv.model_copy(update={"engine": "hf"}), _base())
    for change in ({"version": "0.33.0"}, {"max_num_seqs": 64}, {"enforce_eager": True}):
        assert snapshot.rules_sha256(pv.model_copy(update={"vllm": vllm.model_copy(update=change)}), _base()) != rules
    for change in ({"gpu_memory_utilization": 0.8}, {"python": tmp_path / "other"}, {"start_timeout_min": 5.0},
                   {"stall_timeout_min": 5.0}):
        assert snapshot.rules_sha256(pv.model_copy(update={"vllm": vllm.model_copy(update=change)}), _base()) == rules
    (tmp_path / "manifest.json").write_text('{"tensors": {"x": "0"}}')
    assert snapshot.rules_sha256(pv, _base()) != rules


def test_the_vllm_example_config_loads_and_answers_with_vllm_at_2048_tokens(tmp_path, pv_chain):
    data = yaml.safe_load((Path(run.__file__).with_name("config.vllm.example.yaml")).read_text())
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "manifest.json").write_text('{"tensors": {}}')
    data["chain"] = {**data["chain"], **pv_chain}  # the placeholders, filled
    data["prompt_validator"] |= {"runs_dir": str(tmp_path / "runs"), "write_root": str(tmp_path)}
    data["prompt_validator"]["vllm"]["checkpoint"] = str(tmp_path / "ckpt")
    path = tmp_path / "config.yaml"
    data |= {"run": {"root_path": str(tmp_path)}, "task": {"base_path": str(EXPERT_GROUPS)}}
    path.write_text(yaml.safe_dump(data))

    pv = PromptValidatorConfig.from_path(path, active_task=None).prompt_validator

    assert (pv.engine, pv.max_new_tokens, pv.prompt_source, pv.set_weights) == ("vllm", 2048, "dataless", False)
    assert (pv.vllm.version, pv.vllm.enforce_eager, str(pv.vllm.python)) == (
        "0.32.0", False, "/path/to/venv-vllm/bin/python")
    settings = snapshot.answer_settings(pv)
    assert (settings["engine"], settings["generate"]["max_new_tokens"]) == ("vllm", 2048)
    assert snapshot.rules_sha256(pv, _base()) != snapshot.rules_sha256(pv.model_copy(update={"engine": "hf"}), _base())


def test_engine_vllm_needs_a_vllm_section_with_absolute_paths(tmp_path):
    with pytest.raises(ValidationError, match="needs a vllm section"):
        PromptValidatorCfg(prompts_path=Path("/prompts.jsonl"), engine="vllm")
    with pytest.raises(ValidationError, match="must be absolute"):
        VllmCfg(python="venv/bin/python", checkpoint=tmp_path)


@functools.cache
def _past_key(name):
    return Keypair.create_from_uri(f"//past-{name}")


def _signed_checkpoint(uid, name, model_hash, incentive):
    c = _checkpoint(uid, ACTIVE_GROUP, incentive, model_hash=model_hash)
    c.hotkey, c.signed_model_hash = _past_key(name).ss58_address, sign_message(_past_key(name), model_hash)
    return c


def _hk(name):
    return _past_key(name).ss58_address


def _seen(config, cycle_id=CYCLE_ID):
    models = read_rows(stage_path(config.prompt_validator.runs_dir, cycle_id, "snapshot").parent / "models.jsonl")
    return {m["uid"]: (m["first_seen_training_cycle"], m["first_seen_hotkey"]) for m in models}


def _blocks(config, training_cycle):
    return cutoff.training_cycle_blocks(config, CYCLE_ID, training_cycle)


def _snapshot(config, cycle_id=CYCLE_ID, training_cycle=T):
    """Run the snapshot stage of `cycle_id` with its cutoff at `training_cycle`, as run_stages writes it."""
    runs_dir = config.prompt_validator.runs_dir
    write_rows(stage_path(runs_dir, cycle_id, "cutoff"), [cutoff.training_cycle_blocks(config, cycle_id,
                                                                                      training_cycle).__dict__])
    write_rows(stage_path(runs_dir, cycle_id, "snapshot"), snapshot.run_snapshot(config, cycle_id))


def _reward(config, scores):
    """run_reward on the snapshot taken, each uid's model scored `scores[uid]` on every prompt."""
    pv = config.prompt_validator
    pv.reward_min_prompts = 3
    write_rows(stage_path(pv.runs_dir, CYCLE_ID, "scores"), [
        {"uid": m["uid"], "hotkey": m["hotkey"], "model_hash": m["model_hash"], "prompt_id": f"p{i}",
         "expected_score": scores[m["uid"]]} for m in _models(config) for i in range(3)
    ])
    return {row["uid"]: (row["copy_of"], row["weight"]) for row in reward.run_reward(config, CYCLE_ID)}


def test_a_copier_who_signs_first_in_the_cutoffs_window_loses_to_the_author_who_held_it_a_cycle_before(config, chain):
    # The author revealed "a" at T-1; the copier signed it at T, at an earlier commitment block than the author's.
    author, copier = _signed_checkpoint(3, "author", "a", 0.1), _signed_checkpoint(7, "copier", "a", 0.9)
    chain.checkpoints = [author, copier, _signed_checkpoint(4, "other", "b", 0.5)]
    chain.registrations = [(copier.hotkey, _registration(40)), (author.hotkey, _registration(50)),
                           (_hk("other"), _registration(60))]
    chain.at[T - 1] = [_signed_checkpoint(3, "author", "a", 0.1)]

    _snapshot(config)

    assert _seen(config) == {3: (T - 1, _hk("author")), 7: (T - 1, _hk("author")), 4: (T, _hk("other"))}
    assert _reward(config, {3: 0.75, 7: 0.75, 4: 0.25}) == {
        3: (None, 0.75), 7: (_hk("author"), 0.0), 4: (None, 0.25),
    }


def test_a_validator_without_local_history_records_the_same_originals(config, chain, tmp_path):
    # The author held "a" at T-20, beyond the copier's first commit at T-3; the copier's block is earlier at T.
    chain.checkpoints = [_signed_checkpoint(3, "author", "a", 0.1), _signed_checkpoint(7, "copier", "a", 0.9)]
    chain.registrations = [(_hk("copier"), _registration(40)), (_hk("author"), _registration(50))]
    chain.at |= {T - 20: [_signed_checkpoint(3, "author", "a", 0.1)], T - 13: chain.checkpoints,
                 T - 3: [_signed_checkpoint(7, "copier", "a", 0.9)]}
    # The validator running for a while: an earlier scoring cycle's snapshot and cache.
    _snapshot(config, CYCLE_ID - 1, T - 13)
    _snapshot(config)
    running_meta = read_rows(stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "snapshot"))[0]
    running = _models(config)

    config.prompt_validator.runs_dir = tmp_path / "fresh"
    chain.reads.clear()
    _snapshot(config)

    assert len(chain.reads) == LOOKBACK + 1
    assert _models(config) == running
    assert {m["first_seen_hotkey"] for m in running} == {_hk("author")}
    fresh_meta = read_rows(stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "snapshot"))[0]
    assert (fresh_meta["rules_sha256"], fresh_meta["snapshot_sha256"]) == (
        running_meta["rules_sha256"], running_meta["snapshot_sha256"])


@pytest.mark.parametrize("back, seen", [(LOOKBACK, (T - LOOKBACK, "author")), (LOOKBACK + 1, (T, "copier"))])
def test_the_lookback_reaches_26_training_cycles_back_and_no_further(config, chain, back, seen):
    chain.checkpoints = [_signed_checkpoint(7, "copier", "a", 0.9)]
    chain.at[T - back] = [_signed_checkpoint(3, "author", "a", 0.1)]

    _snapshot(config)

    assert _seen(config) == {7: (seen[0], _hk(seen[1]))}


@pytest.mark.parametrize("blocks, original", [((70, 80), "b5"), ((80, 70), "b3"), ((70, 70), "b3")])
def test_within_the_earliest_training_cycle_the_earlier_block_then_the_lower_uid_is_the_original(
        config, chain, blocks, original):
    chain.checkpoints = [_signed_checkpoint(5, "b5", "a", 0.1), _signed_checkpoint(3, "b3", "a", 0.1)]
    chain.at[T - 4] = chain.checkpoints
    chain.registrations_at[T - 4] = [(_hk("b5"), _registration(blocks[0])), (_hk("b3"), _registration(blocks[1]))]
    chain.registrations = [(_hk("b5"), _registration(90)), (_hk("b3"), _registration(1))]  # the cutoff's: unused

    _snapshot(config)

    assert _seen(config) == {3: (T - 4, _hk(original)), 5: (T - 4, _hk(original))}


def test_a_lone_copier_whose_author_left_gets_0(config, chain):
    chain.checkpoints = [_signed_checkpoint(7, "copier", "a", 0.9), _signed_checkpoint(4, "other", "b", 0.5)]
    chain.at[T - 2] = [_signed_checkpoint(3, "author", "a", 0.1)]

    _snapshot(config)

    assert reward.originals(_models(config)) == {
        "a": reward.Original(None, _hk("author")), "b": reward.Original(4, _hk("other")),
    }
    assert _reward(config, {7: 0.9, 4: 0.5}) == {7: (_hk("author"), 0.0), 4: (None, 1.0)}


def test_a_signature_that_does_not_verify_holds_nothing(config, chain):
    forged = _signed_checkpoint(3, "forger", "a", 0.1)
    forged.signed_model_hash = sign_message(_past_key("someone-else"), "a")
    chain.checkpoints = [_signed_checkpoint(7, "copier", "a", 0.9)]
    chain.at[T - 5] = [forged]

    _snapshot(config)

    assert _seen(config) == {7: (T, _hk("copier"))}


def test_cached_training_cycles_are_not_read_again_and_a_damaged_cache_is(config, chain, caplog):
    chain.at |= {t: [_signed_checkpoint(9, "other", "z", 0.0)] for t in range(T - LOOKBACK, T)}
    chain.checkpoints = [_signed_checkpoint(7, "copier", "a", 0.9)]
    chain.at[T - 5] = [_signed_checkpoint(3, "author", "a", 0.1)]
    _snapshot(config)
    chain.at[T + 1] = chain.checkpoints
    chain.reads.clear()
    chain.commitment_reads.clear()

    _snapshot(config, CYCLE_ID + 1, T + 1)

    assert [read[0] for read in chain.reads] == [_blocks(config, T + 1).commit_1_block]
    assert [read[2] for read in chain.commitment_reads] == [_blocks(config, T + 1).commit_1_block]
    assert _seen(config, CYCLE_ID + 1) == {7: (T - 5, _hk("author"))}

    cache = config.prompt_validator.runs_dir / "chain_commits"
    damaged = {t: cache / f"{_blocks(config, t).commit_1_block}-{_blocks(config, t).commit_2_block}.jsonl"
               for t in (T - 5, T - 6)}
    damaged[T - 5].write_text(damaged[T - 5].read_text()[:-10])  # cut short
    write_rows(damaged[T - 6], [{"uid": 1}])
    chain.reads.clear()

    with caplog.at_level(logging.INFO):
        _snapshot(config, CYCLE_ID + 1, T + 1)

    assert [read[0] for read in chain.reads] == [_blocks(config, t).commit_1_block for t in (T + 1, T - 6, T - 5)]
    assert caplog.text.count("chain commits cache damaged") == 2
    assert _seen(config, CYCLE_ID + 1) == {7: (T - 5, _hk("author"))}
    assert [row["hotkey"] for row in read_rows(damaged[T - 5])] == [_hk("author")]


def test_a_training_cycle_read_with_no_commits_is_read_again_next_time(config, chain):
    chain.at[T - 4] = [_signed_checkpoint(3, "author", "a", 0.1)]
    _snapshot(config)
    chain.reads.clear()

    _snapshot(config)

    read = [read[0] for read in chain.reads]
    assert _blocks(config, T - 3).commit_1_block in read and _blocks(config, T - 4).commit_1_block not in read


@pytest.mark.parametrize("which, match", [
    ("commit_1_block", "{c1}"),
    ("commit_2_block", "{c1} or {c2}"),  # the commitment read at commit_1 passes; the commits read must not fall back
])
def test_a_pruned_block_in_the_lookback_stops_the_snapshot(config, chain, which, match):
    oldest = _blocks(config, T - LOOKBACK)
    chain.pruned = {getattr(oldest, which)}
    match = match.format(c1=oldest.commit_1_block, c2=oldest.commit_2_block)

    with pytest.raises(RuntimeError, match=f"pruned block {match}: use an archive node"):
        run.run_stages(config, CYCLE_ID, stage="snapshot")

    assert not stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "snapshot").parent.joinpath(
        "models.jsonl").exists()
    assert chain.reads == [(CUTOFF["commit_1_block"], CUTOFF["commit_2_block"], "miner")]
