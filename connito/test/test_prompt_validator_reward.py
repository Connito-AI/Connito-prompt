import functools
import logging
import re
from types import SimpleNamespace

import pytest
from bittensor.sp_core import Keypair

from connito.prompt_validator import report, run, snapshot
from connito.prompt_validator.config import PromptValidatorCfg
from connito.prompt_validator.reward import run_reward
from connito.prompt_validator.snapshot import rules_sha256
from connito.prompt_validator.stage_io import read_rows, stage_path, write_rows
from connito.shared.schema import sign_message
from connito.test.conftest import needs_judge_rules

CYCLE_ID = 1258
CUTOFF_TRAINING_CYCLE = 17625
# The snapshot's base model rows; base_rows rebuilds the same from config unless a test changes them.
BASE = {"base_model": "Qwen/Qwen3.6-35B-A3B", "base_model_revision": "0" * 40, "chat_template_sha256": "1" * 64}


@pytest.fixture(autouse=True)
def fixed_base_rows(monkeypatch):
    monkeypatch.setattr(snapshot, "base_rows", lambda cfg: dict(BASE))  # the real one downloads the chat template
PROMPT_IDS = ["p1", "p2", "p3"]


@pytest.fixture
def config(tmp_path):
    # Three prompts, so a miner needs all three to qualify; the defaults (20 and 3) are tested on their own.
    pv = PromptValidatorCfg(runs_dir=tmp_path / "runs", prompts_path=tmp_path / "p.jsonl", reward_min_prompts=3)
    config = SimpleNamespace(prompt_validator=pv)
    _write_snapshot(config, PROMPT_IDS)
    return config


def _write_snapshot(config, prompt_ids):
    pv = config.prompt_validator
    snapshot_dir = stage_path(pv.runs_dir, CYCLE_ID, "snapshot").parent
    write_rows(snapshot_dir / "prompts.jsonl", [{"prompt_id": p, "prompt": f"prompt {p}"} for p in prompt_ids])
    write_rows(snapshot_dir / "snapshot.jsonl", [{**BASE, "rules_sha256": rules_sha256(pv, BASE)}])


@functools.cache
def _key(uid: int) -> Keypair:
    return Keypair.create_from_uri(f"//miner-{uid}")


def _hk(uid: int) -> str:
    return _key(uid).ss58_address


def _write_scores(config, miners: dict[int, tuple[str, list[float]]], prompt_ids=PROMPT_IDS, blocks=None,
                  unscored=None, unsigned=(), first_seen=None) -> None:
    """miners: uid -> (model_hash, its expected_score on each prompt), one row per (miner, prompt) as scores writes.

    score is set to the opposite order (1 - expected_score), so a ranking on score would come out reversed. The
    snapshot's models are these and `unscored` (uid -> model_hash, kept but never scored), each signed by its own
    hotkey unless in `unsigned`, with `blocks[uid]` as its signed-hash commitment block, by default 9000 + uid.
    `first_seen` (model_hash -> (training cycle, hotkey)) is what the snapshot recorded from earlier training cycles;
    by default the cutoff's (CUTOFF_TRAINING_CYCLE) earliest signed commit, as when no earlier one held the hash.
    """
    blocks, first_seen = blocks or {}, first_seen or {}
    hashes = {**{uid: h for uid, (h, _) in miners.items()}, **(unscored or {})}
    models = [
        {"uid": uid, "hotkey": _hk(uid), "model_hash": h, "signed_commit_block": blocks.get(uid, 9000 + uid),
         "signed_model_hash": sign_message(_key(uid if uid not in unsigned else uid + 1000), h)}
        for uid, h in hashes.items()
    ]
    cutoff = {h: (CUTOFF_TRAINING_CYCLE, same[0]["hotkey"]) for h, same in snapshot.signed_by_hash(models).items()}
    for m in models:
        held = first_seen.get(m["model_hash"], cutoff.get(m["model_hash"], (None, None)))
        m["first_seen_training_cycle"], m["first_seen_hotkey"] = held
    snapshot_dir = stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "snapshot").parent
    write_rows(snapshot_dir / "models.jsonl", models)
    write_rows(stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "scores"), [
        {"uid": uid, "hotkey": _hk(uid), "model_hash": model_hash, "prompt_id": prompt_id,
         "score": round(1 - expected, 4), "expected_score": expected, "rank": 0, "jev_call_id": f"call-{prompt_id}"}
        for uid, (model_hash, scores) in miners.items()
        for prompt_id, expected in zip(prompt_ids, scores, strict=False)  # a short list: a miner missing prompts
    ])


def _by_uid(rows):
    return {r["uid"]: r for r in rows}


def test_a_miners_score_is_its_mean_expected_score_and_the_top_3_share_weight_by_it(config):
    _write_scores(config, {
        1: ("h1", [1.0, 0.5, 0.75]), 2: ("h2", [0.5] * 3), 3: ("h3", [0.25] * 3), 4: ("h4", [0.125] * 3),
    })
    rows = _by_uid(run_reward(config, CYCLE_ID))
    assert {uid: r["mean_score"] for uid, r in rows.items()} == {1: 0.75, 2: 0.5, 3: 0.25, 4: 0.125}
    assert {uid: r["weight"] for uid, r in rows.items()} == {1: 0.75 / 1.5, 2: 0.5 / 1.5, 3: 0.25 / 1.5, 4: 0.0}


def test_the_mean_decides_not_the_prompts_won(config):
    # uid 1 is best on two of the three prompts, but uid 2's mean is higher.
    _write_scores(config, {1: ("h1", [1.0, 1.0, 0.0]), 2: ("h2", [0.875] * 3)})
    assert [(r["rank"], r["uid"]) for r in run_reward(config, CYCLE_ID)] == [(0, 2), (1, 1)]


def test_only_the_top_3_miners_by_mean_get_a_weight_and_their_weights_add_up_to_1(config):
    _write_scores(config, {
        1: ("h1", [0.75] * 3), 2: ("h2", [0.5] * 3), 3: ("h3", [0.5] * 3),
        4: ("h4", [0.25] * 3), 5: ("h5", [0.25] * 3), 6: ("h6", [0.125] * 3),
    })
    rows = _by_uid(run_reward(config, CYCLE_ID))
    assert {uid: r["rank"] for uid, r in rows.items()} == {1: 0, 2: 1, 3: 1, 4: 3, 5: 3, 6: 5}
    assert {uid: r["weight"] for uid, r in rows.items()} == {
        1: 0.75 / 1.75, 2: 0.5 / 1.75, 3: 0.5 / 1.75, 4: 0.0, 5: 0.0, 6: 0.0,
    }
    assert sum(r["weight"] for r in rows.values()) == pytest.approx(1)


def test_a_tie_at_the_miner_cut_weights_every_tied_miner(config):
    # uid 1 is first and uids 2, 3 and 4 share second place, so four miners are weighted.
    _write_scores(config, {1: ("h1", [0.75] * 3), 2: ("h2", [0.5] * 3), 3: ("h3", [0.5] * 3),
                           4: ("h4", [0.5] * 3), 5: ("h5", [0.25] * 3)})
    rows = _by_uid(run_reward(config, CYCLE_ID))
    assert {uid: r["weight"] for uid, r in rows.items()} == {
        1: 0.75 / 2.25, 2: 0.5 / 2.25, 3: 0.5 / 2.25, 4: 0.5 / 2.25, 5: 0.0,
    }


def test_equal_shares_give_every_winner_the_same_weight(config):
    config.prompt_validator.reward_shares = "equal"
    _write_scores(config, {1: ("h1", [0.75] * 3), 2: ("h2", [0.5] * 3), 3: ("h3", [0.5] * 3),
                           4: ("h4", [0.5] * 3), 5: ("h5", [0.25] * 3)})
    rows = _by_uid(run_reward(config, CYCLE_ID))
    assert {uid: r["weight"] for uid, r in rows.items()} == {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25, 5: 0.0}


def test_a_mean_of_0_gets_no_weight(config):
    _write_scores(config, {1: ("h1", [0.5] * 3), 2: ("h2", [0.0] * 3)})
    rows = _by_uid(run_reward(config, CYCLE_ID))
    assert {uid: (r["rank"], r["weight"]) for uid, r in rows.items()} == {1: (0, 1.0), 2: (1, 0.0)}
    _write_scores(config, {1: ("h1", [0.0] * 3), 2: ("h2", [0.0] * 3)})
    assert [r["weight"] for r in run_reward(config, CYCLE_ID)] == [0.0, 0.0]


def test_reward_top_miners_is_a_setting(config):
    config.prompt_validator.reward_top_miners = 1
    _write_scores(config, {1: ("h1", [0.5] * 3), 2: ("h2", [0.5] * 3), 3: ("h3", [0.5] * 3)})
    rows = _by_uid(run_reward(config, CYCLE_ID))
    assert {uid: r["weight"] for uid, r in rows.items()} == {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}  # all tie for first
    _write_scores(config, {1: ("h1", [0.75] * 3), 2: ("h2", [0.5] * 3), 3: ("h3", [0.25] * 3)})
    rows = _by_uid(run_reward(config, CYCLE_ID))
    assert {uid: r["weight"] for uid, r in rows.items()} == {1: 1.0, 2: 0.0, 3: 0.0}


def test_rows_are_best_first_rank_0_and_equal_means_share_a_rank(config):
    _write_scores(config, {
        1: ("h1", [0.5] * 3), 2: ("h2", [0.5] * 3), 3: ("h3", [0.875] * 3), 4: ("h4", [0.5] * 3),
        5: ("h5", [0.25] * 3),
    })
    rows = run_reward(config, CYCLE_ID)
    assert [(r["rank"], r["uid"]) for r in rows] == [(0, 3), (1, 1), (1, 2), (1, 4), (4, 5)]
    assert rows[0] == {"rank": 0, "uid": 3, "hotkey": _hk(3), "model_hash": "h3", "signed_commit_block": 9003,
                       "copy_of": None, "n_scored": 3, "qualified": True, "mean_score": 0.875,
                       "weight": 0.875 / 2.375}


def test_the_top_3_is_by_expected_score_not_score(config):
    # The fixture's score runs opposite to expected_score: uids 1-3 lead on expected_score, uids 4-5 on score.
    _write_scores(config, {uid: (f"h{uid}", [1 - uid / 10] * 3) for uid in (1, 2, 3, 4, 5)})
    rows = _by_uid(run_reward(config, CYCLE_ID))
    assert [uid for uid, r in rows.items() if r["weight"]] == [1, 2, 3]


def test_a_miner_with_too_few_prompts_is_listed_without_reward(config):
    # uid 9 is best on p1 and p2 but has no p3: it does not qualify, and takes no place from uid 3.
    _write_scores(config, {1: ("h1", [0.75] * 3), 2: ("h2", [0.5] * 3), 3: ("h3", [0.25] * 3), 9: ("h9", [])})
    rows = read_rows(stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "scores"))
    rows += [{"uid": 9, "hotkey": _hk(9), "model_hash": "h9", "prompt_id": p, "score": 0.0, "expected_score": 1.0,
              "rank": 0, "jev_call_id": "c"} for p in ("p1", "p2")]
    write_rows(stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "scores"), rows)

    out = run_reward(config, CYCLE_ID)
    assert out[-1] == {"rank": None, "uid": 9, "hotkey": _hk(9), "model_hash": "h9", "signed_commit_block": 9009,
                       "copy_of": None, "n_scored": 2, "qualified": False, "mean_score": 0.0, "weight": 0.0}
    assert {r["uid"]: (r["rank"], r["mean_score"]) for r in out[:-1]} == {1: (0, 0.75), 2: (1, 0.5), 3: (2, 0.25)}
    assert out[2]["weight"] == 0.25 / 1.5


def test_by_default_a_miner_needs_20_scored_prompts_3_miners_are_weighted_by_their_means(tmp_path):
    pv = PromptValidatorCfg(runs_dir=tmp_path / "runs", prompts_path=tmp_path / "p.jsonl")
    assert (pv.reward_min_prompts, pv.reward_top_miners, pv.reward_shares) == (20, 3, "proportional")
    config = SimpleNamespace(prompt_validator=pv)
    prompt_ids = [f"p{i}" for i in range(20)]
    _write_snapshot(config, prompt_ids)
    _write_scores(config, {1: ("h1", [0.5] * 20), 2: ("h2", [0.75] * 19)}, prompt_ids)
    rows = _by_uid(run_reward(config, CYCLE_ID))
    assert (rows[1]["qualified"], rows[1]["n_scored"], rows[1]["mean_score"]) == (True, 20, 0.5)
    assert (rows[2]["qualified"], rows[2]["n_scored"], rows[2]["mean_score"]) == (False, 19, 0.0)


def test_reward_shares_refuses_an_unknown_value(tmp_path):
    with pytest.raises(ValueError, match="reward_shares"):
        PromptValidatorCfg(runs_dir=tmp_path, prompts_path=tmp_path / "p.jsonl", reward_shares="rank")


@pytest.mark.parametrize("field", ["reward_min_prompts", "reward_top_miners"])
@pytest.mark.parametrize("value", [0, -1])
def test_reward_settings_must_be_positive(tmp_path, field, value):
    with pytest.raises(ValueError):
        PromptValidatorCfg(runs_dir=tmp_path, prompts_path=tmp_path / "p.jsonl", **{field: value})


def _places(rows):
    return [(r["rank"], r["uid"], r["mean_score"], r["weight"], r["copy_of"]) for r in rows]


def test_three_copies_of_the_top_model_take_one_place(config, caplog):
    # uid 1's model is best on every prompt and uids 6, 7 and 8 commit the same hash: one place among the models,
    # so h2, h3 and h4 keep places 2 to 4 and h2 and h3 stay in the top 3. The copies get 0 and name their original.
    _write_scores(config, {
        1: ("top", [1.0] * 3), 6: ("top", [1.0] * 3), 7: ("top", [1.0] * 3), 8: ("top", [1.0] * 3),
        2: ("h2", [0.75] * 3), 3: ("h3", [0.5] * 3), 4: ("h4", [0.25] * 3),
    })
    with caplog.at_level(logging.INFO):
        rows = run_reward(config, CYCLE_ID)

    assert _places(rows) == [
        (0, 1, 1.0, 1 / 2.25, None), (1, 2, 0.75, 0.75 / 2.25, None), (2, 3, 0.5, 0.5 / 2.25, None),
        (3, 4, 0.25, 0.0, None), (None, 6, 0.0, 0.0, _hk(1)), (None, 7, 0.0, 0.0, _hk(1)), (None, 8, 0.0, 0.0, _hk(1)),
    ]
    assert all(r["qualified"] for r in rows)  # a copy is listed as scored, so the report posts its 0, not null
    [line] = [re.sub(r"\x1b\[[0-9;]*m", "", r.getMessage()) for r in caplog.records
              if "copies of one model" in r.getMessage()]
    for field in ("uid=1", f"hotkey={_hk(1)[:8]} ", "block=9001", "copies=[(6, 9006), (7, 9007), (8, 9008)]"):
        assert field in line


def test_the_report_posts_a_copy_with_model_score_and_weight_0(config):
    _write_scores(config, {1: ("same", [0.75] * 3), 2: ("same", [0.75] * 3)})
    summary = report.summary_rows(run_reward(config, CYCLE_ID), {_hk(1), _hk(2)}, set(), {})
    assert [(r["uid"], r["model_score"], r["model_weight"]) for r in summary] == [(1, 0.75, 1.0), (2, 0.0, 0.0)]


def test_the_original_is_the_earliest_commit_even_with_a_higher_uid(config):
    # uid 9 signed the hash at block 100, before uid 2 (block 200): uid 9 is the original, uid 2 the copy.
    _write_scores(config, {2: ("same", [0.75] * 3), 9: ("same", [0.75] * 3), 5: ("h5", [0.25] * 3)},
                  blocks={2: 200, 9: 100, 5: 150})
    rows = _by_uid(run_reward(config, CYCLE_ID))
    assert {uid: (r["mean_score"], r["weight"], r["copy_of"]) for uid, r in rows.items()} == {
        9: (0.75, 0.75, None), 5: (0.25, 0.25, None), 2: (0.0, 0.0, _hk(9)),
    }
    assert (rows[9]["signed_commit_block"], rows[2]["signed_commit_block"]) == (100, 200)


def test_copies_on_one_block_go_to_the_lower_uid(config):
    _write_scores(config, {9: ("same", [0.75] * 3), 2: ("same", [0.75] * 3), 4: ("same", [0.75] * 3)},
                  blocks={9: 100, 2: 100, 4: 100})
    rows = _by_uid(run_reward(config, CYCLE_ID))
    assert {uid: (r["mean_score"], r["copy_of"]) for uid, r in rows.items()} == {
        2: (0.75, None), 4: (0.0, _hk(2)), 9: (0.0, _hk(2)),
    }


def test_a_commit_without_a_block_is_never_the_original_over_one_with_a_block(config):
    # uid 1 has the lower uid but no block was read for it; it is not taken as earliest.
    _write_scores(config, {1: ("same", [0.75] * 3), 5: ("same", [0.75] * 3)}, blocks={1: None, 5: 500})
    rows = _by_uid(run_reward(config, CYCLE_ID))
    assert {uid: (r["mean_score"], r["copy_of"]) for uid, r in rows.items()} == {5: (0.75, None), 1: (0.0, _hk(5))}


def test_a_copy_gets_0_even_when_its_original_was_not_fetched(config):
    # uid 1 signed "top" first but its fetch failed (no scores); uid 5 re-hosted it and was fetched. uid 5 is still
    # the copy: "top" takes no place this cycle, and h2, h3 and h4 take places 1 to 3.
    _write_scores(config, {5: ("top", [1.0] * 3), 2: ("h2", [0.75] * 3), 3: ("h3", [0.5] * 3), 4: ("h4", [0.25] * 3)},
                  unscored={1: "top"})
    assert _places(run_reward(config, CYCLE_ID)) == [
        (0, 2, 0.75, 0.75 / 1.5, None), (1, 3, 0.5, 0.5 / 1.5, None), (2, 4, 0.25, 0.25 / 1.5, None),
        (None, 5, 0.0, 0.0, _hk(1)),
    ]


def test_a_commit_whose_signature_fails_is_never_the_original(config):
    # uid 1 committed "top" first, but its signed hash is not its hotkey's signature of "top" (fetch fails it too).
    _write_scores(config, {5: ("top", [0.75] * 3), 2: ("h2", [0.25] * 3)}, unscored={1: "top"}, unsigned={1},
                  blocks={1: 1})
    assert _places(run_reward(config, CYCLE_ID)) == [(0, 5, 0.75, 0.75, None), (1, 2, 0.25, 0.25, None)]


def test_a_kept_commit_with_no_signed_hash_is_never_the_original(config):
    # uid 1 has the earliest block but committed no signed hash (fetch fails it); reward still runs.
    _write_scores(config, {5: ("top", [0.75] * 3)}, unscored={1: "top"}, blocks={1: 1},
                  first_seen={"top": (CUTOFF_TRAINING_CYCLE, _hk(5))})
    models_path = stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "snapshot").parent / "models.jsonl"
    write_rows(models_path, [{**m, "signed_model_hash": None} if m["uid"] == 1 else m for m in read_rows(models_path)])
    assert _places(run_reward(config, CYCLE_ID)) == [(0, 5, 0.75, 1.0, None)]


def test_no_qualified_miner_gives_every_weight_0(config):
    _write_scores(config, {1: ("h1", [0.9, 0.9]), 2: ("h2", [0.8, 0.8])})  # 2 of the 3 prompts each
    assert [(r["qualified"], r["weight"]) for r in run_reward(config, CYCLE_ID)] == [(False, 0.0), (False, 0.0)]


def test_no_scored_miner_gives_an_empty_reward(config):
    _write_scores(config, {})
    assert run_reward(config, CYCLE_ID) == []


def test_two_scores_for_one_prompt_stop_the_stage(config):
    _write_scores(config, {1: ("h1", [0.9] * 3)})
    path = stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "scores")
    write_rows(path, read_rows(path) + read_rows(path)[:1])
    with pytest.raises(RuntimeError, match="uid 1 has two scores for 'p1'"):
        run_reward(config, CYCLE_ID)


def test_a_score_for_a_prompt_not_in_the_snapshot_stops_the_stage(config):
    _write_scores(config, {1: ("h1", [0.9] * 3)}, ["p1", "p2", "p9"])
    with pytest.raises(RuntimeError, match="'p9', which is not in the snapshot"):
        run_reward(config, CYCLE_ID)


@pytest.mark.parametrize("min_prompts, stops", [(3, False), (4, True)])
def test_a_snapshot_with_fewer_prompts_than_needed_to_qualify_stops_the_stage(config, min_prompts, stops):
    config.prompt_validator.reward_min_prompts = min_prompts  # the snapshot has 3 prompts
    _write_scores(config, {1: ("h1", [0.9] * 3)})
    if not stops:
        assert run_reward(config, CYCLE_ID)[0]["qualified"]
        return
    with pytest.raises(RuntimeError, match=r"the snapshot has 3 prompts, fewer than reward_min_prompts \(4\)"):
        run_reward(config, CYCLE_ID)


def test_no_scores_file_stops_with_a_clear_error(config):
    with pytest.raises(RuntimeError, match="no scores to reward: .* run --stage scores first"):
        run_reward(config, CYCLE_ID)


def test_settings_changed_since_the_snapshot_are_refused(config):
    _write_scores(config, {1: ("h1", [0.9] * 3)})
    config.prompt_validator.jev.resolved = "jev-1.15.0"
    with pytest.raises(RuntimeError, match="rerun --stage snapshot"):
        run_reward(config, CYCLE_ID)


def test_reward_min_prompts_top_miners_and_shares_stay_out_of_the_rules_hash(config):
    before = rules_sha256(config.prompt_validator, BASE)
    config.prompt_validator.reward_min_prompts = 7
    config.prompt_validator.reward_top_miners = 9
    config.prompt_validator.reward_shares = "equal"
    assert rules_sha256(config.prompt_validator, BASE) == before


def test_the_rules_hash_follows_the_reward_rule(config, monkeypatch):
    before = rules_sha256(config.prompt_validator, BASE)
    monkeypatch.setattr(snapshot, "REWARD_RULE", "another rule")
    assert rules_sha256(config.prompt_validator, BASE) != before


def test_a_plain_run_ends_with_reward_and_a_reward_rerun_leaves_the_scores(config):
    runs_dir = config.prompt_validator.runs_dir
    for name in ("cutoff", "fetch", "answers"):
        write_rows(stage_path(runs_dir, CYCLE_ID, name), [{}])
    _write_scores(config, {1: ("h1", [0.9] * 3)})

    run.run_stages(config, CYCLE_ID)  # every earlier stage is done: only reward runs
    reward_path = stage_path(runs_dir, CYCLE_ID, "reward")
    assert reward_path.exists()

    run.run_stages(config, CYCLE_ID, "reward")
    assert [r["uid"] for r in read_rows(reward_path)] == [1]
    assert oct(reward_path.stat().st_mode & 0o777) == "0o600"
    assert stage_path(runs_dir, CYCLE_ID, "scores").exists()  # a reward rerun leaves the scores alone


def test_rerunning_an_earlier_stage_removes_reward_and_says_how_to_rebuild_it(config, monkeypatch, caplog):
    runs_dir = config.prompt_validator.runs_dir
    _write_scores(config, {1: ("h1", [0.9] * 3)})
    run.run_stages(config, CYCLE_ID, "reward")
    monkeypatch.setattr(run, "STAGES", [(n, (lambda cfg, cycle: [{}]) if n == "fetch" else s) for n, s in run.STAGES])

    with caplog.at_level(logging.WARNING):
        run.run_stages(config, CYCLE_ID, "fetch")

    assert not stage_path(runs_dir, CYCLE_ID, "reward").exists()
    # Nothing is on demand any more: a plain run rebuilds them.
    assert "stage=reward" in caplog.text and "stage=scores" in caplog.text and "a plain run" in caplog.text


def _copy_lines(caplog):
    return [re.sub(r"\x1b\[[0-9;]*m", "", r.getMessage()) for r in caplog.records
            if "copies of one model" in r.getMessage()]


def test_the_hotkey_that_first_held_a_hash_stays_its_original_against_an_earlier_block(config, caplog):
    # The attack: uid 2's hotkey held "top" at training cycle 17620; uid 9 copied the revealed hash and signed it at
    # block 100 of the cutoff's, before uid 2 re-signed at 200. The earlier training cycle decides: uid 2 is the
    # original. A hash first held at the cutoff ("new") goes to the earliest block, then the lower uid.
    _write_scores(config, {2: ("top", [0.75] * 3), 9: ("top", [0.75] * 3), 4: ("new", [0.25] * 3),
                           3: ("new", [0.25] * 3), 6: ("new", [0.25] * 3)},
                  blocks={2: 200, 9: 100, 4: 50, 3: 60, 6: 50}, first_seen={"top": (17620, _hk(2))})
    with caplog.at_level(logging.INFO):
        rows = _by_uid(run_reward(config, CYCLE_ID))

    assert {uid: (r["mean_score"], r["weight"], r["copy_of"]) for uid, r in rows.items()} == {
        2: (0.75, 0.75, None), 4: (0.25, 0.25, None), 9: (0.0, 0.0, _hk(2)), 3: (0.0, 0.0, _hk(4)),
        6: (0.0, 0.0, _hk(4)),
    }
    top, new = sorted(_copy_lines(caplog), key=lambda line: "first_seen=17620" not in line)
    assert f"first_seen={CUTOFF_TRAINING_CYCLE}" in new and "uid=4" in new
    for field in ("uid=2", f"hotkey={_hk(2)[:8]} ", "block=200", "copies=[(9, 100)]"):
        assert field in top


def test_a_hash_whose_first_holder_has_no_commit_this_cycle_gives_every_commit_0(config, caplog):
    # Hotkey 77 held "top" first and doesn't commit it this cycle: uids 1 and 5 are both copies, and "top" takes no
    # place, so h2, h3 and h4 take places 1 to 3.
    _write_scores(config, {1: ("top", [1.0] * 3), 5: ("top", [1.0] * 3), 2: ("h2", [0.75] * 3),
                           3: ("h3", [0.5] * 3), 4: ("h4", [0.25] * 3)}, first_seen={"top": (17620, _hk(77))})
    with caplog.at_level(logging.INFO):
        rows = run_reward(config, CYCLE_ID)

    assert _places(rows) == [
        (0, 2, 0.75, 0.75 / 1.5, None), (1, 3, 0.5, 0.5 / 1.5, None), (2, 4, 0.25, 0.25 / 1.5, None),
        (None, 1, 0.0, 0.0, _hk(77)), (None, 5, 0.0, 0.0, _hk(77)),
    ]
    [line] = _copy_lines(caplog)
    for field in ("uid=None", f"hotkey={_hk(77)[:8]} ", "block=None", "first_seen=17620",
                  "copies=[(1, 9001), (5, 9005)]"):
        assert field in line


def test_a_first_holder_whose_signature_fails_this_cycle_is_not_the_original(config):
    _write_scores(config, {1: ("top", [0.75] * 3), 5: ("top", [0.75] * 3), 2: ("h2", [0.75] * 3)}, unsigned={1},
                  first_seen={"top": (17620, _hk(1))})
    assert _places(run_reward(config, CYCLE_ID)) == [
        (0, 2, 0.75, 1.0, None), (None, 1, 0.0, 0.0, _hk(1)), (None, 5, 0.0, 0.0, _hk(1)),
    ]


def test_a_snapshot_row_without_its_first_seen_hotkey_is_refused(config):
    # Every snapshot this reward accepts records it.
    _write_scores(config, {1: ("top", [0.9] * 3), 5: ("top", [0.9] * 3)})
    models_path = stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "snapshot").parent / "models.jsonl"
    write_rows(models_path, [{k: v for k, v in m.items() if k != "first_seen_hotkey"} for m in read_rows(models_path)])
    with pytest.raises(KeyError, match="first_seen_hotkey"):
        run_reward(config, CYCLE_ID)


@needs_judge_rules
def test_the_selected_miner_prompts_scores_count_toward_no_reward(config, write_selection):
    _write_scores(config, {1: ("a", [0.9, 0.8, 0.7]), 2: ("b", [0.5, 0.6, 0.4]), 3: ("c", [0.1, 0.2, 0.3])})
    before = run_reward(config, CYCLE_ID)
    pv = config.prompt_validator
    pv.judge_miner_prompts = True
    _write_snapshot(config, PROMPT_IDS)  # under the switch's rules
    write_selection(pv, CYCLE_ID, {"m1": "one", "m2": "two"}, ["m1", "m2"])
    scores_path = stage_path(pv.runs_dir, CYCLE_ID, "scores")
    write_rows(scores_path, [*read_rows(scores_path), *(
        {"uid": 3, "hotkey": _hk(3), "model_hash": "c", "prompt_id": p, "score": 0.0, "expected_score": 1.0,
         "rank": 0, "jev_call_id": "m"} for p in ("m1", "m2"))])
    assert run_reward(config, CYCLE_ID) == before
    assert _by_uid(before)[3]["mean_score"] == pytest.approx(0.2)  # its 1.0 on m1 and m2 not counted
