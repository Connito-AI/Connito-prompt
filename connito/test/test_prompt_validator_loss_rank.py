import math
import re
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from pydantic import ValidationError

from connito.prompt_validator import loss_rank, snapshot
from connito.prompt_validator.config import LossRankCfg, PromptValidatorCfg, VllmCfg
from connito.prompt_validator.stage_io import read_rows, stage_path, write_rows
from connito.test import test_prompt_validator_snapshot as snapshot_tests
from connito.test.test_prompt_validator_snapshot import ACTIVE_GROUP, CUTOFF, CYCLE_ID, _checkpoint, _models

# The snapshot tests' fixtures, used here by name: a fake chain and Hub, and the config they serve.
chain, config = snapshot_tests.chain, snapshot_tests.config

T = CUTOFF["training_cycle"]
BLOCK_AT_REGISTRATION = ("SubtensorModule", "BlockAtRegistration")
UIDS = list(range(1, 13))  # in the active group, incentive rising with the uid and loss rising with it too
OTHER = 20  # on the metagraph in another group: its rows join the groups, but it is never ranked
CFG = LossRankCfg()


def _phase_end(cycle):
    return cycle * 524 + 400


def _entry(slot, loss, **changes):
    return {"validator_slot": slot, "evaluated": True, "eval_status_code": 0, "eval_status_label": "ok",
            "loss_is_fresh": True, "score": 0.0, "score_avg": 0.0, "loss": loss, **changes}


def _submission(cycle, entries, **changes):
    return {"cycle_number": cycle, "phase_end_block": _phase_end(cycle), "status": "committed",
            "hf_repo": "miner/repo@abc", "per_validator": entries, **changes}


def _history(uid, submissions, hotkey=None):
    return {"data": {"miner_uid": uid, "hotkey": hotkey or f"hk-{uid}", "series": {}, "submissions": submissions},
            "meta": {"stale": False}}


def _default_history(uid):
    """Slots 2 and 5 at the cutoff's last two completed training cycles, loss uid/100 above a per-cycle base, with
    an unevaluated slot as the source serves it."""
    return _history(uid, [
        _submission(cycle, [_entry(1, None, eval_status_code=None, loss_is_fresh=None),
                            _entry(2, 1.0 + (T - cycle) + uid / 100), _entry(5, 2.0 + (T - cycle) + uid / 100)])
        for cycle in (T - 1, T - 2)
    ])


def _row(uid, cycle, slot, loss, *, hotkey=None, phase_end_block=None):
    return {"uid": uid, "hotkey": hotkey or f"hk-{uid}", "cycle_number": cycle,
            "phase_end_block": _phase_end(cycle) if phase_end_block is None else phase_end_block,
            "validator_slot": slot, "loss": loss, "hf_repo": None}


class Answer:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code, self.payload, self.headers = status_code, payload, headers or {}

    def json(self):
        if self.payload == "not json":
            raise requests.JSONDecodeError("Expecting value", "<", 0)
        return self.payload


@pytest.fixture
def source(monkeypatch):
    """A fake dashboard-api on a fake clock. `source.histories[uid]` is what it serves for a uid; `source.answers[uid]`
    are answers served first, in order: an HTTP status, an exception to raise, or an Answer. A uid answers 429 until
    `source.limited_until[uid]`; each request takes `source.request_sec` of the clock."""
    state = SimpleNamespace(now=0.0, requests=[], timeouts=[], answers={}, limited_until={}, request_sec=0.0,
                            histories={uid: _default_history(uid) for uid in [*UIDS, OTHER]})

    def answer(uid):
        if state.answers.get(uid):
            answer = state.answers[uid].pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer if isinstance(answer, Answer) else Answer(answer)
        if state.now < state.limited_until.get(uid, -1):
            return Answer(429)
        return Answer(200, state.histories[uid])

    def get(url, timeout):
        uid = int(re.fullmatch(r"https://dashboard-api-v2\.connito\.ai/api/v2/miner/(\d+)/history"
                               r"\?include_submissions=true", url).group(1))
        state.requests.append((state.now, uid))
        state.timeouts.append(timeout)
        try:
            return answer(uid)
        finally:
            state.now += state.request_sec

    def sleep(seconds):
        assert seconds >= 0
        state.now += seconds

    monkeypatch.setattr(loss_rank.requests, "get", get)
    monkeypatch.setattr(loss_rank, "time", SimpleNamespace(sleep=sleep, monotonic=lambda: state.now))
    return state


@pytest.fixture
def ranked(monkeypatch, config, chain, source):
    """The snapshot config with rank_by: loss on a chain whose metagraph at the cutoff holds UIDS and OTHER with
    hotkeys hk-<uid>, registered at block 0 unless `ranked.registered` says otherwise (an exception to raise)."""
    config.prompt_validator.rank_by = "loss"
    config.prompt_validator.loss_rank.min_scored = 10  # of the 12 in-group miners here
    config.prompt_validator.max_models = 3
    chain.checkpoints = [_checkpoint(uid, ACTIVE_GROUP, uid / 100) for uid in UIDS] + [_checkpoint(OTHER, 3, 0.9)]
    state = SimpleNamespace(hotkeys={uid: f"hk-{uid}" for uid in [*UIDS, OTHER]},
                            registered={uid: 0 for uid in [*UIDS, OTHER]}, metagraph_reads=[], registration_reads=[])

    def get_metagraph(subtensor, netuid, *, block=None, lite=True):
        state.metagraph_reads.append((netuid, block))
        return SimpleNamespace(uids=list(state.hotkeys), hotkeys=list(state.hotkeys.values()))

    fake = snapshot.bittensor

    class Subtensor(fake.Subtensor):
        def query_map(self, item, params, block=None):
            if item != BLOCK_AT_REGISTRATION:
                return super().query_map(item, params, block)
            state.registration_reads.append((params, block))
            if isinstance(state.registered, Exception):
                raise state.registered
            return list(state.registered.items())

    storage = SimpleNamespace(Commitments=fake.storage.Commitments,
                              SubtensorModule=SimpleNamespace(BlockAtRegistration=BLOCK_AT_REGISTRATION))
    monkeypatch.setattr(snapshot, "bittensor", SimpleNamespace(Subtensor=Subtensor, storage=storage))
    monkeypatch.setattr(snapshot.bt_compat, "get_metagraph", get_metagraph)
    return state


def _source_path(config):
    return stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "snapshot").parent / loss_rank.SOURCE_FILE


def _key_path(config):
    return _source_path(config).with_name(loss_rank.KEY_FILE)


# --- The row filter -----------------------------------------------------------------------------------------------


def test_the_row_filter_keeps_fresh_status_0_finite_losses_of_slots_2_and_5_in_the_window():
    payload = _history(7, [
        _submission(T - 1, [
            _entry(2, 1.5), _entry(5, 2), _entry(1, 1.0), _entry(3, 1.0), _entry(4, 1.0),
            _entry(2, 1.0, loss_is_fresh=False), _entry(2, 1.0, loss_is_fresh=None), _entry(2, 1.0, loss_is_fresh=1),
            _entry(2, 1.0, eval_status_code=6), _entry(2, 1.0, eval_status_code=None),
            _entry(2, 1.0, eval_status_code=False), _entry(2, 1.0, eval_status_code=0.0),
            _entry(5, None), _entry(5, math.nan), _entry(5, math.inf), _entry(5, "1.0"), _entry(5, True),
        ], hf_repo="m/r@1"),
        _submission(T, [_entry(2, 1.0)]),  # the cutoff's own training cycle, not completed
        _submission(T + 1, [_entry(2, 1.0)]),
        _submission(T - 8, [_entry(5, 3)], hf_repo=None),  # the oldest of the window
        _submission(T - 9, [_entry(2, 1.0)]),  # before it
    ])

    rows = loss_rank.filter_rows(7, payload, CFG, T)

    assert rows == [
        {"uid": 7, "hotkey": "hk-7", "cycle_number": T - 1, "phase_end_block": _phase_end(T - 1),
         "validator_slot": 2, "loss": 1.5, "hf_repo": "m/r@1"},
        {**_row(7, T - 1, 5, 2.0), "hf_repo": "m/r@1"},
        _row(7, T - 8, 5, 3.0),
    ]
    assert type(rows[1]["loss"]) is float and type(rows[2]["loss"]) is float
    assert list(rows[0]) == list(loss_rank.ROW_FIELDS)


def test_the_row_filter_drops_training_cycles_before_min_cycle():
    payload = _history(7, [_submission(cycle, [_entry(2, 1.0)]) for cycle in range(T - 8, T)])

    rows = loss_rank.filter_rows(7, payload, CFG.model_copy(update={"min_cycle": T - 3}), T)

    assert [row["cycle_number"] for row in rows] == [T - 3, T - 2, T - 1]
    assert LossRankCfg().min_cycle == 16587


def test_the_row_filter_takes_its_slots_and_window_from_the_config():
    payload = _history(7, [_submission(cycle, [_entry(1, 1.0), _entry(2, 1.0)]) for cycle in range(T - 8, T)])

    rows = loss_rank.filter_rows(7, payload, CFG.model_copy(update={"validator_slots": [1],
                                                                    "window_training_cycles": 2}), T)

    assert [(row["cycle_number"], row["validator_slot"]) for row in rows] == [(T - 2, 1), (T - 1, 1)]


@pytest.mark.parametrize("payload", [
    None, [], {"data": None}, {"data": {"submissions": []}}, {"data": {"hotkey": 5, "submissions": []}},
    {"data": {"hotkey": "hk-7", "submissions": {}}},
    {"data": {"miner_uid": 8, "hotkey": "hk-7", "submissions": []}},
    _history(7, ["a submission"]),
    _history(7, [_submission(T - 1, [], cycle_number=None)]),
    _history(7, [_submission(T - 1, [], cycle_number=True)]),
    _history(7, [_submission(T - 1, [], phase_end_block=None)]),
    _history(7, [_submission(T - 1, None)]),
    _history(7, [_submission(T - 1, ["an entry"])]),
    _history(7, [_submission(T - 1, [_entry(None, 1.0)])]),
])
def test_a_response_the_rules_cannot_read_is_malformed(payload):
    with pytest.raises(loss_rank.Malformed):
        loss_rank.filter_rows(7, payload, CFG, T)


def test_a_submission_outside_the_window_is_not_read_past_its_cycle_number():
    payload = _history(7, [_submission(T, None, phase_end_block=None), _submission(T - 1, [_entry(2, 1.0)])])

    assert loss_rank.filter_rows(7, payload, CFG, T) == [{**_row(7, T - 1, 2, 1.0), "hf_repo": "miner/repo@abc"}]


# --- Groups, the hotkey filter and the scores ----------------------------------------------------------------------

def test_a_groups_median_is_of_its_cycle_and_slots_rows_and_a_small_group_is_skipped():
    rows = ([_row(uid, T - 1, 2, float(uid)) for uid in range(10)]  # median of 0..9: 4.5
            + [_row(uid, T - 1, 5, float(uid)) for uid in range(11)]  # 0..10: 5
            + [_row(uid, T - 2, 2, 1.0) for uid in range(9)])  # 9 rows: skipped

    assert loss_rank.group_medians(rows, 10) == {(T - 1, 2): 4.5, (T - 1, 5): 5.0}
    assert loss_rank.group_medians(rows, 9) == {(T - 1, 2): 4.5, (T - 1, 5): 5.0, (T - 2, 2): 1.0}


def test_rows_count_from_the_uids_registration_block_whatever_hotkey_the_source_names():
    rows = [_row(1, T - 1, 2, 1.0), _row(2, T - 1, 2, 1.0, hotkey="hk-new"), _row(3, T - 1, 2, 1.0),
            _row(3, T - 2, 2, 1.0), _row(4, T - 1, 2, 1.0)]
    registered = {1: 0, 2: 0, 3: _phase_end(T - 1)}  # uid 3 registered at T-1's phase end; uid 4 has no block

    counted = loss_rank.counted_rows(rows, registered)

    assert counted == [rows[0], rows[1], rows[2]]


def test_a_score_is_the_mean_loss_over_its_groups_medians_with_both_slots_pooled():
    population = [_row(uid, cycle, slot, 10.0 * slot + cycle - T + uid)
                  for cycle in (T - 1, T - 2) for slot in (2, 5) for uid in range(100, 110)]
    mine = [_row(1, T - 1, 2, 20.0 - 1 + 100), _row(1, T - 2, 5, 50.0 - 2 + 110)]
    rows = population + mine

    scores = loss_rank.score(rows, mine, CFG)

    # The groups take uid 1's rows too: 100..109 above the base plus uid 1's 100 (median 104) and 110 (median 105).
    assert scores.scores == {1: pytest.approx(((100 - 104) + (110 - 105)) / 2)}
    assert scores.failed_k == []


def test_uncounted_rows_still_set_the_median_and_skipped_groups_count_for_nothing():
    population = [_row(uid, T - 1, 2, float(uid)) for uid in range(100, 109)]  # 9 rows
    former = _row(2, T - 1, 2, 1000.0, hotkey="hk-old")  # the 10th: makes the group, counts for no one
    small = [_row(uid, T - 2, 2, 0.0) for uid in range(100, 108)] + [_row(1, T - 2, 2, 0.0)]  # 9 rows: skipped
    mine = [_row(1, T - 1, 2, 104.0), _row(1, T - 1, 2, 106.0), small[-1]]
    rows = population + [former] + mine[:2] + small

    scores = loss_rank.score(rows, mine, CFG)

    median = sorted([*range(100, 109), 1000, 104, 106])[5:7]
    assert scores.scores == {1: pytest.approx((104 + 106) / 2 - sum(median) / 2)}


def test_a_miner_needs_min_samples_counted_rows_or_it_is_unscored():
    population = [_row(uid, T - 1, slot, 1.0) for uid in range(100, 110) for slot in (2, 5)]
    one = [_row(1, T - 1, 2, 2.0), _row(1, T - 2, 2, 2.0)]  # its second row's group is too small
    two = [_row(2, T - 1, 2, 2.0), _row(2, T - 1, 5, 3.0)]
    rows = population + one + two

    scores = loss_rank.score(rows, rows, CFG)

    assert set(scores.scores) == set(range(100, 110)) | {2}
    assert scores.failed_k == [1]
    assert scores.scores[2] == pytest.approx(1.5)
    assert set(loss_rank.score(rows, rows, CFG.model_copy(update={"min_samples": 1})).scores) \
        == set(range(100, 110)) | {1, 2}


# --- Ranking -------------------------------------------------------------------------------------------------------


def _ids(kept):
    return [c.uid for c in kept]


def test_without_scores_the_selection_is_by_incentive_as_before():
    commits = [_checkpoint(1, 8, 0.1), _checkpoint(2, 8, 0.5), _checkpoint(3, 8, 0.5), _checkpoint(4, 8, 0.0)]

    assert _ids(snapshot.select_models(commits, 2)) == [2, 3]
    assert _ids(snapshot.select_models(commits, 10)) == [2, 3, 1, 4]


def test_scored_models_rank_by_score_then_uid_and_the_unscored_fill_by_incentive_then_uid():
    commits = [_checkpoint(uid, 8, incentive) for uid, incentive in
               [(1, 0.9), (2, 0.0), (3, 0.1), (4, 0.5), (5, 0.5), (6, 0.8), (7, 0.0)]]
    scores = {2: -0.5, 3: 0.2, 7: -0.5, 1: 0.3}

    assert _ids(snapshot.select_models(commits, 10, scores)) == [2, 7, 3, 1, 6, 4, 5]
    assert _ids(snapshot.select_models(commits, 5, scores)) == [2, 7, 3, 1, 6]


def test_copies_share_one_place_at_their_best_commits_key():
    commits = [_checkpoint(1, 8, 0.9, model_hash="h-a"), _checkpoint(2, 8, 0.1, model_hash="h-b"),
               _checkpoint(3, 8, 0.0, model_hash="h-a"), _checkpoint(4, 8, 0.5, model_hash="h-c"),
               _checkpoint(5, 8, 0.8, model_hash="h-d")]
    # uid 3's score places h-a first, with its unscored copy uid 1; h-b next; then the unscored by incentive.
    scores = {3: -1.0, 2: 0.0}

    assert _ids(snapshot.select_models(commits, 3, scores)) == [3, 2, 1, 5]


# --- The fetch -----------------------------------------------------------------------------------------------------


def test_the_fetch_asks_once_per_uid_unsigned_and_spaced_under_30_a_minute(source):
    fetched = loss_rank.fetch_rows(CFG, [1, 2, 3], T)

    assert fetched.failure is None
    assert [uid for _, uid in source.requests] == [1, 2, 3]
    starts = [at for at, _ in source.requests]
    assert all(b - a >= CFG.request_spacing_sec for a, b in pairwise(starts))
    assert 60 / CFG.request_spacing_sec < 30
    assert [row["uid"] for row in fetched.rows] == [1] * 4 + [2] * 4 + [3] * 4


@pytest.mark.parametrize("error", [429, 500, 503, requests.Timeout("slow"), requests.ConnectionError("down"),
                                   requests.exceptions.ChunkedEncodingError(),
                                   requests.exceptions.ContentDecodingError()])
def test_a_retryable_failure_is_retried_with_backoff_and_spacing(source, error):
    source.answers[2] = [error, error]

    fetched = loss_rank.fetch_rows(CFG, [1, 2, 3], T)

    assert fetched.failure is None
    assert [uid for _, uid in source.requests] == [1, 2, 2, 2, 3]
    starts = [at for at, _ in source.requests]
    assert starts[2] - starts[1] >= CFG.backoff_sec - 1e-9 and starts[3] - starts[2] >= 2 * CFG.backoff_sec - 1e-9
    assert all(b - a >= CFG.request_spacing_sec - 1e-9 for a, b in pairwise(starts))


@pytest.mark.parametrize("answers, failure", [
    ([429] * 4, "fetch_failed: uid 2: HTTP 429 after 3 retries"),
    ([502] * 4, "fetch_failed: uid 2: HTTP 502 after 3 retries"),
    ([requests.Timeout()] * 4, "fetch_failed: uid 2: Timeout after 3 retries"),
    ([requests.ConnectionError()] * 4, "fetch_failed: uid 2: ConnectionError after 3 retries"),
    ([403], "fetch_failed: uid 2: HTTP 403"),
    ([requests.TooManyRedirects()], "fetch_failed: uid 2: TooManyRedirects"),
    ([Answer(200, "not json")], "fetch_failed: uid 2: malformed: Expecting value"),
    ([Answer(200, {"data": []})], "fetch_failed: uid 2: malformed: no data.hotkey or data.submissions list"),
])
def test_a_uid_whose_fetch_fails_stops_the_fetch_with_the_reason(source, answers, failure):
    source.answers[2] = list(answers)

    fetched = loss_rank.fetch_rows(CFG, [1, 2, 3], T)

    assert fetched.rows is None
    assert fetched.failure.startswith(failure)
    assert 3 not in [uid for _, uid in source.requests]


def test_the_fetch_stops_once_its_time_budget_is_spent(source):
    cfg = CFG.model_copy(update={"budget_sec": 10.0})

    fetched = loss_rank.fetch_rows(cfg, UIDS, T)

    assert fetched == loss_rank.Fetched(None, "time_budget: 10 s spent before uid 6", {})
    assert [uid for _, uid in source.requests] == [1, 2, 3, 4, 5]


def test_a_429_burst_longer_than_the_backoffs_is_waited_out_a_minute_from_the_first_429(source):
    source.limited_until[2] = CFG.request_spacing_sec + 40.0  # past 5 + 10 + 20 s of backoff

    fetched = loss_rank.fetch_rows(CFG, [1, 2, 3], T)

    assert fetched.failure is None
    assert [(at, uid) for at, uid in source.requests if uid == 2] == [
        (CFG.request_spacing_sec, 2), (pytest.approx(CFG.request_spacing_sec + 60.0), 2)]


def test_a_429_waits_as_long_as_its_retry_after_says(source):
    source.answers[2] = [Answer(429, headers={"Retry-After": "7"}), Answer(429, headers={"Retry-After": "soon"})]

    assert loss_rank.fetch_rows(CFG, [2], T).failure is None
    starts = [at for at, _ in source.requests]
    # Then without a usable one: a minute from the first 429.
    assert starts == [0.0, 7.0, pytest.approx(60.0)]


def test_no_wait_outlasts_the_time_budget(source):
    source.limited_until[1] = 1000.0
    cfg = CFG.model_copy(update={"budget_sec": 30.0})

    fetched = loss_rank.fetch_rows(cfg, [1], T)

    assert fetched == loss_rank.Fetched(None, "time_budget: 30 s spent before uid 1", {})
    assert source.now == pytest.approx(30.0)  # the minute's wait cut to the budget left


def test_each_request_is_given_at_most_the_budget_left_on_a_clock_that_runs_during_it(source):
    source.request_sec = 3.0
    cfg = CFG.model_copy(update={"budget_sec": 10.0})

    fetched = loss_rank.fetch_rows(cfg, UIDS, T)

    assert fetched.failure == "time_budget: 10 s spent before uid 5"
    assert source.timeouts == [10.0, 7.0, 4.0, 1.0]
    assert loss_rank.fetch_rows(CFG.model_copy(update={"budget_sec": 100.0}), [1], T).failure is None
    assert source.timeouts[-1] == CFG.timeout_sec


def test_a_404_is_a_uid_with_no_rows_and_the_fetch_goes_on(source):
    source.answers[2] = [404]

    fetched = loss_rank.fetch_rows(CFG, [1, 2, 3], T)

    assert fetched.failure is None
    assert [uid for _, uid in source.requests] == [1, 2, 3]  # once: never retried
    assert sorted({row["uid"] for row in fetched.rows}) == [1, 3]


def test_the_loss_settings_default_to_the_locked_values():
    cfg = PromptValidatorCfg(prompts_path=Path("/prompts.jsonl"))

    assert cfg.rank_by == "incentive"
    assert cfg.loss_rank.model_dump(include=set(loss_rank.LOSS_FIELDS)) == {
        "url": "https://dashboard-api-v2.connito.ai/api/v2/miner/{uid}/history?include_submissions=true",
        "window_training_cycles": 8, "min_samples": 2, "min_group": 10,
        "min_scored": 50, "validator_slots": [2, 5], "min_cycle": 16587,
    }
    with pytest.raises(ValidationError, match="rank_by"):
        PromptValidatorCfg(prompts_path=Path("/prompts.jsonl"), rank_by="score")
    with pytest.raises(ValidationError, match="placeholder"):
        LossRankCfg(url="https://dashboard-api-v2.connito.ai/api/v2/miner/1/history")
    with pytest.raises(ValidationError, match="request_spacing_sec"):
        LossRankCfg(request_spacing_sec=1.9)
    with pytest.raises(ValidationError, match="at least one slot"):
        LossRankCfg(validator_slots=[])
    assert LossRankCfg(validator_slots=[5, 2, 5]).validator_slots == [2, 5]


# --- The snapshot stage --------------------------------------------------------------------------------------------


def test_rank_by_loss_keeps_the_lowest_scores_and_records_the_source(config, chain, ranked, source):
    [meta] = snapshot.run_snapshot(config, CYCLE_ID)

    assert [m["uid"] for m in _models(config)] == [1, 2, 3]
    assert ranked.metagraph_reads == [(102, CUTOFF["commit_2_block"])]
    assert ranked.registration_reads == [([102], CUTOFF["commit_2_block"])]
    assert sorted(uid for _, uid in source.requests) == sorted([*UIDS, OTHER])
    rows = read_rows(_source_path(config))
    assert len(rows) == 4 * 13 and all(tuple(row) == loss_rank.ROW_FIELDS for row in rows)
    assert meta["rank_by"] == "loss"
    assert meta["loss_rank"] == {"fallback": False, "fallback_reason": None, "scored": 12, "unscored": 0,
                                 "failed_k": 0, "source_rows": 52, "fill": 0, "source": "fetched"}
    settings = config.prompt_validator.loss_rank.model_dump(mode="json", include=set(loss_rank.LOSS_FIELDS))
    assert read_rows(_key_path(config)) == [{"training_cycle": T, "settings": settings, "truncated": {}}]


def test_unscored_miners_fill_by_incentive_and_are_counted(config, chain, ranked, source):
    config.prompt_validator.loss_rank.min_scored = 5
    config.prompt_validator.max_models = 12
    source.histories[12] = _history(12, [])
    source.histories[11] = _history(11, [_submission(T - 1, [_entry(2, 1.0)])])  # one row: under min_samples
    source.answers[10] = [404]
    ranked.registered[9] = _phase_end(T - 1) + 1  # registered after its rows
    chain.checkpoints += [_checkpoint(13, ACTIVE_GROUP, 0.0), _checkpoint(14, ACTIVE_GROUP, 0.0)]  # not on it yet

    [meta] = snapshot.run_snapshot(config, CYCLE_ID)

    # 9 registered after its rows, 10 has no history, 11 has one row, 12 none, 13 and 14 no fetch.
    assert meta["loss_rank"] == {"fallback": False, "fallback_reason": None, "scored": 8, "unscored": 6,
                                 "failed_k": 1, "source_rows": 4 * 13 - 4 - 4 - 3, "fill": 4, "source": "fetched"}
    # By score 1..8, then the unscored by incentive 12, 11, 10, 9, with 13 and 14 (0) over max_models.
    assert [m["uid"] for m in _models(config)] == list(range(1, 13))


def _same_snapshot(first, second):
    assert {**second, "loss_rank": {**second["loss_rank"], "source": None}} == {
        **first, "loss_rank": {**first["loss_rank"], "source": None}}


def test_a_rerun_of_the_snapshot_reuses_its_loss_source_and_retakes_the_same_snapshot(config, chain, ranked,
                                                                                       source):
    [first] = snapshot.run_snapshot(config, CYCLE_ID)
    asked = len(source.requests)

    [second] = snapshot.run_snapshot(config, CYCLE_ID)

    assert len(source.requests) == asked
    assert (first["loss_rank"]["source"], second["loss_rank"]["source"]) == ("fetched", "reused")
    _same_snapshot(first, second)


@pytest.mark.parametrize("change", [
    lambda path: path.unlink(),
    lambda path: path.write_text("[]\n"),
    lambda path: path.write_text('{"training_cycle": %d, "settings": {}, "truncated": {}}\n' % T),
])
def test_a_missing_or_unreadable_key_fetches_the_source_again(config, chain, ranked, source, change):
    [first] = snapshot.run_snapshot(config, CYCLE_ID)
    change(_key_path(config))

    [second] = snapshot.run_snapshot(config, CYCLE_ID)

    assert len(source.requests) == 2 * 13
    assert second["loss_rank"]["source"] == "fetched"
    _same_snapshot(first, second)


def test_a_source_fetched_for_another_training_cycle_is_fetched_again(config, chain, ranked, source):
    snapshot.run_snapshot(config, CYCLE_ID)
    [key] = read_rows(_key_path(config))
    write_rows(_key_path(config), [{**key, "training_cycle": T - 13}])

    assert snapshot.run_snapshot(config, CYCLE_ID)[0]["loss_rank"]["source"] == "fetched"
    assert len(source.requests) == 2 * 13
    assert read_rows(_key_path(config)) == [key]


def test_a_source_fetched_with_other_validator_slots_is_fetched_again(config, chain, ranked, source):
    snapshot.run_snapshot(config, CYCLE_ID)
    config.prompt_validator.loss_rank = LossRankCfg(validator_slots=[2], min_scored=10)

    [meta] = snapshot.run_snapshot(config, CYCLE_ID)

    assert (meta["loss_rank"]["source"], meta["loss_rank"]["source_rows"]) == ("fetched", 2 * 13)
    assert len(source.requests) == 2 * 13


def test_the_snapshot_hash_covers_the_loss_source(config, chain, ranked, source):
    first = snapshot.run_snapshot(config, CYCLE_ID)[0]
    rows = read_rows(_source_path(config))
    rows[0]["hf_repo"] = "miner/other@def"  # decides nothing, but is what was fetched
    write_rows(_source_path(config), rows)

    second = snapshot.run_snapshot(config, CYCLE_ID)[0]

    assert second["snapshot_sha256"] != first["snapshot_sha256"]
    assert second["rules_sha256"] == first["rules_sha256"]


def test_a_damaged_loss_source_is_fetched_again(config, chain, ranked, source):
    [first] = snapshot.run_snapshot(config, CYCLE_ID)
    _source_path(config).write_text('{"uid": 1}\n')

    _same_snapshot(first, snapshot.run_snapshot(config, CYCLE_ID)[0])
    assert len(source.requests) == 2 * 13


def test_a_deleted_loss_source_is_fetched_again(config, chain, ranked, source):
    snapshot.run_snapshot(config, CYCLE_ID)
    _source_path(config).unlink()

    assert snapshot.run_snapshot(config, CYCLE_ID)[0]["loss_rank"]["source"] == "fetched"
    assert len(source.requests) == 2 * 13


def _truncate(source, uid, cycle):
    source.histories[uid]["meta"]["submissions_truncated_at_cycle"] = cycle


def test_a_ranked_uids_submissions_cut_inside_the_window_fall_back_to_incentive(config, chain, ranked, source):
    _truncate(source, 3, T - 8)  # the window's first cycle may be gone
    _truncate(source, OTHER, T - 1)  # not ranked: its rows only join the groups

    [meta] = snapshot.run_snapshot(config, CYCLE_ID)

    _fell_back(config, meta, "window_not_covered: uid 3 truncated at cycle %d" % (T - 8))
    assert (meta["loss_rank"]["source_rows"], meta["loss_rank"]["scored"]) == (52, None)
    assert read_rows(_key_path(config))[0]["truncated"] == {"3": T - 8, str(OTHER): T - 1}
    # Reused, the key keeps the cut, so the rerun falls back the same way.
    assert snapshot.run_snapshot(config, CYCLE_ID)[0]["loss_rank"]["fallback_reason"] == meta["loss_rank"][
        "fallback_reason"]


def test_submissions_cut_before_the_window_still_rank_by_loss(config, chain, ranked, source):
    for uid in UIDS:
        _truncate(source, uid, T - 9)

    assert snapshot.run_snapshot(config, CYCLE_ID)[0]["loss_rank"]["fallback"] is False


def test_a_truncation_that_is_not_a_cycle_is_malformed():
    payload = {**_history(7, []), "meta": {"submissions_truncated_at_cycle": "17402"}}

    with pytest.raises(loss_rank.Malformed):
        loss_rank.truncated_at(payload)
    assert loss_rank.truncated_at(_history(7, [])) is None


def _fell_back(config, meta, reason):
    assert meta["rank_by"] == "loss"
    assert meta["loss_rank"]["fallback"] is True
    assert meta["loss_rank"]["fallback_reason"].startswith(reason)
    assert [m["uid"] for m in _models(config)] == [10, 11, 12]  # by incentive


def test_a_failed_fetch_falls_back_to_incentive_and_a_rerun_fetches_again(config, chain, ranked, source):
    source.answers[4] = [503] * 4

    [meta] = snapshot.run_snapshot(config, CYCLE_ID)

    _fell_back(config, meta, "fetch_failed: uid 4: HTTP 503 after 3 retries")
    assert meta["loss_rank"] == {"fallback": True, "fallback_reason": meta["loss_rank"]["fallback_reason"],
                                 "scored": None, "unscored": None, "failed_k": None, "source_rows": None, "fill": 3,
                                 "source": None}
    assert not _source_path(config).exists()
    assert snapshot.run_snapshot(config, CYCLE_ID)[0]["loss_rank"]["fallback"] is False


def test_an_unreachable_source_falls_back_to_incentive(config, chain, ranked, source):
    for uid in [*UIDS, OTHER]:
        source.answers[uid] = [requests.ConnectionError("no route")] * 4

    _fell_back(config, snapshot.run_snapshot(config, CYCLE_ID)[0], "fetch_failed: uid 1: ConnectionError")


def test_a_spent_time_budget_falls_back_to_incentive(config, chain, ranked, source):
    config.prompt_validator.loss_rank.budget_sec = 5.0

    _fell_back(config, snapshot.run_snapshot(config, CYCLE_ID)[0], "time_budget")


def test_fewer_than_min_scored_scored_miners_fall_back_to_incentive_and_keep_the_source(config, chain, ranked,
                                                                                         source):
    for uid in (1, 2, 3):
        ranked.registered[uid] = _phase_end(T - 1) + 1  # after all their rows

    [meta] = snapshot.run_snapshot(config, CYCLE_ID)

    _fell_back(config, meta, "too_few_scored: 9 < 10")
    assert (meta["loss_rank"]["scored"], meta["loss_rank"]["unscored"]) == (9, 3)
    assert _source_path(config).exists()


def test_a_404_leaves_only_that_uid_unscored(config, chain, ranked, source):
    source.answers[4] = [404]

    [meta] = snapshot.run_snapshot(config, CYCLE_ID)

    assert meta["loss_rank"]["fallback"] is False
    assert (meta["loss_rank"]["scored"], meta["loss_rank"]["unscored"]) == (11, 1)
    assert [uid for _, uid in source.requests].count(4) == 1
    assert 4 not in {row["uid"] for row in read_rows(_source_path(config))}


def test_a_5xx_after_retries_still_falls_back_to_incentive(config, chain, ranked, source):
    source.answers[4] = [500] * 4

    _fell_back(config, snapshot.run_snapshot(config, CYCLE_ID)[0], "fetch_failed: uid 4: HTTP 500 after 3 retries")
    assert [uid for _, uid in source.requests].count(4) == 4


def test_a_uid_registered_again_after_the_cutoff_keeps_its_holders_rows(config, chain, ranked, source):
    # The source names the uid's hotkey as of the fetch: the new holder's. Its rows are the cutoff's holder's, who
    # registered before them all.
    source.histories[1] = _history(1, source.histories[1]["data"]["submissions"], hotkey="hk-new-holder")
    ranked.registered[1] = _phase_end(T - 2) - 100

    [meta] = snapshot.run_snapshot(config, CYCLE_ID)

    assert {row["hotkey"] for row in read_rows(_source_path(config)) if row["uid"] == 1} == {"hk-new-holder"}
    assert (meta["loss_rank"]["fallback"], meta["loss_rank"]["scored"]) == (False, 12)
    assert [m["uid"] for m in _models(config)] == [1, 2, 3]


def test_registration_blocks_take_bittensors_integer_pairs_and_leave_out_the_rest():
    # bittensor 11's query_map at a real block: a list of (uid, block) ints, one per uid.
    rows = [(0, 7840965), (1, 9379447), (2, 9358457), (3, None), ("4", 9000000)]
    subtensor = SimpleNamespace(query_map=lambda item, params, block=None: rows)
    storage = SimpleNamespace(SubtensorModule=SimpleNamespace(BlockAtRegistration=BLOCK_AT_REGISTRATION))

    with pytest.MonkeyPatch.context() as m:
        m.setattr(snapshot, "bittensor", SimpleNamespace(storage=storage))
        assert snapshot.registration_blocks(subtensor, 102, 9386794) == {0: 7840965, 1: 9379447, 2: 9358457}


def test_a_uid_without_a_registration_block_falls_back_to_incentive(config, chain, ranked, source):
    ranked.registered[5] = None

    _fell_back(config, snapshot.run_snapshot(config, CYCLE_ID)[0], "registration_blocks: 12 of 13")
    assert source.requests == []


def test_a_chain_read_that_fails_falls_back_to_incentive(config, chain, ranked, source):
    ranked.registered = RuntimeError("node gone")

    _fell_back(config, snapshot.run_snapshot(config, CYCLE_ID)[0], "error: RuntimeError: node gone")
    assert source.requests == []


def test_rank_by_incentive_reads_no_source_and_writes_no_file_or_meta(config, chain, ranked, source):
    snapshot.run_snapshot(config, CYCLE_ID)
    config.prompt_validator.rank_by = "incentive"

    [meta] = snapshot.run_snapshot(config, CYCLE_ID)

    assert len(source.requests) == 13
    assert not _source_path(config).exists() and not _key_path(config).exists()
    assert "rank_by" not in meta and "loss_rank" not in meta
    assert ranked.metagraph_reads == [(102, CUTOFF["commit_2_block"])]
    assert [m["uid"] for m in _models(config)] == [10, 11, 12]


# --- The rules hash ------------------------------------------------------------------------------------------------

# The production config's answer settings and base rows, its checkpoint by manifest hash.
PRODUCTION_ANSWER = {
    "engine": "vllm", "chat_template": {"enable_thinking": False},
    "generate": {"do_sample": False, "max_new_tokens": 2048},
    "vllm": {"version": "0.32.0", "manifest_sha256": "14bdf515d4c0d9310cb8c3da4b6c3fa6735329beed2465eee469761edd2a8678",
             "max_num_seqs": 128, "enforce_eager": False},
}
PRODUCTION_BASE = {
    "base_model": "Qwen/Qwen3.6-35B-A3B", "base_model_revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
    "chat_template_sha256": "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
    "quantization": "fp8", "quantization_validator": "experts",
    "residual": {"k": 1, "threshold": 0.8, "temperature": 0.05},
}


@pytest.fixture
def production(monkeypatch, tmp_path):
    """The production config's prompt_validator section, its checkpoint's manifest hash served without the file."""
    monkeypatch.setattr(snapshot, "answer_settings", lambda pv: PRODUCTION_ANSWER)
    vllm = VllmCfg(python="/path/to/venv-vllm/bin/python", checkpoint=tmp_path,
                   version="0.32.0", max_num_seqs=128, enforce_eager=False, gpu_memory_utilization=0.85)
    return PromptValidatorCfg(prompt_source="dataless", max_new_tokens=2048, answer_batch_size=20, engine="vllm",
                              vllm=vllm, jev={"resolved": "jev-1.14.0", "spend_cap_usd": 5.0})


def test_the_production_rules_hash_by_incentive_is_unchanged(production):
    rules = "45d11fd0d5bcd235c0221ed2ab783074df52eda22f36a8cd46fcf325b900ac68"

    assert snapshot.rules_sha256(production, PRODUCTION_BASE) == rules
    loss_settings = {"loss_rank": LossRankCfg(min_scored=3, url="https://other.example/{uid}")}
    assert snapshot.rules_sha256(production.model_copy(update=loss_settings), PRODUCTION_BASE) == rules


def test_the_production_rules_hash_by_loss_is_pinned(production):
    loss = production.model_copy(update={"rank_by": "loss"})

    assert snapshot.rules_sha256(loss, PRODUCTION_BASE) == (
        "9f58efe51b4097bc30c054db96a837687ec7222a89c39c21a7a094972d633f27")


@pytest.mark.parametrize("change", [
    {"url": "https://other.example/{uid}"}, {"window_training_cycles": 9}, {"min_samples": 3}, {"min_group": 11},
    {"min_scored": 11}, {"validator_slots": [2]}, {"min_cycle": 16588},
])
def test_the_loss_rules_hash_follows_each_ranking_setting(production, change):
    loss = production.model_copy(update={"rank_by": "loss"})
    changed = loss.model_copy(update={"loss_rank": LossRankCfg(**change)})

    assert snapshot.rules_sha256(changed, PRODUCTION_BASE) != snapshot.rules_sha256(loss, PRODUCTION_BASE)


@pytest.mark.parametrize("change", [
    {"request_spacing_sec": 3.0}, {"timeout_sec": 10.0}, {"retries": 1}, {"backoff_sec": 1.0}, {"budget_sec": 60.0},
])
def test_the_loss_rules_hash_ignores_the_pacing(production, change):
    loss = production.model_copy(update={"rank_by": "loss"})
    changed = loss.model_copy(update={"loss_rank": LossRankCfg(**change)})

    assert snapshot.rules_sha256(changed, PRODUCTION_BASE) == snapshot.rules_sha256(loss, PRODUCTION_BASE)


def test_the_loss_rules_hash_follows_the_selection_rule(production, monkeypatch):
    loss = production.model_copy(update={"rank_by": "loss"})
    rules = snapshot.rules_sha256(loss, PRODUCTION_BASE)

    monkeypatch.setattr(snapshot, "LOSS_SELECTION", "another rule")
    assert snapshot.rules_sha256(loss, PRODUCTION_BASE) != rules


# --- model_selection: all ------------------------------------------------------------------------------------------

ALL_RULES = "343c29cb0c7387cac938030a5a699affaa9f2c6314e0fcf1d029a5a0e24b5c05"


def test_the_production_rules_hash_with_every_model_is_pinned(production):
    every = production.model_copy(update={"rank_by": "loss", "model_selection": "all"})

    assert snapshot.rules_sha256(every, PRODUCTION_BASE) == ALL_RULES
    assert snapshot.rules_sha256(every.model_copy(update={"rank_by": "incentive"}), PRODUCTION_BASE) == ALL_RULES


@pytest.mark.parametrize("change", [
    {"max_models": 10}, {"rank_by": "incentive"}, {"loss_rank": LossRankCfg(min_scored=3, window_training_cycles=9)},
])
def test_with_every_model_the_rules_hash_ignores_the_cap_and_the_ranking(production, change):
    every = production.model_copy(update={"rank_by": "loss", "model_selection": "all"})

    assert snapshot.rules_sha256(every.model_copy(update=change), PRODUCTION_BASE) == snapshot.rules_sha256(
        every, PRODUCTION_BASE)


def test_with_every_model_the_rules_hash_differs_and_follows_its_rule(production, monkeypatch):
    loss = production.model_copy(update={"rank_by": "loss"})
    every = loss.model_copy(update={"model_selection": "all"})
    rules = snapshot.rules_sha256(every, PRODUCTION_BASE)

    others = {snapshot.rules_sha256(loss, PRODUCTION_BASE), snapshot.rules_sha256(production, PRODUCTION_BASE)}
    assert rules not in others
    monkeypatch.setattr(snapshot, "MODEL_SELECTION", "another rule")  # the top rule's text decides nothing here
    assert snapshot.rules_sha256(every, PRODUCTION_BASE) == rules
    monkeypatch.setattr(snapshot, "ALL_SELECTION", "another rule")
    assert snapshot.rules_sha256(every, PRODUCTION_BASE) != rules


def test_every_model_is_kept_without_reading_the_loss_source(config, chain, ranked, source):
    chain.checkpoints += [_checkpoint(13, ACTIVE_GROUP, 0.0), _checkpoint(14, ACTIVE_GROUP, 0.0, model_hash="hash-1")]
    config.prompt_validator.model_selection = "all"  # with max_models 3 and rank_by loss from the fixture

    [meta] = snapshot.run_snapshot(config, CYCLE_ID)

    # Every in-group commit: zero incentive and the copy of uid 1's hash too; OTHER's group is not the active one.
    assert [m["uid"] for m in _models(config)] == [*UIDS, 13, 14]
    assert source.requests == [] and ranked.registration_reads == []
    assert not _source_path(config).exists() and not _key_path(config).exists()
    assert meta["model_selection"] == "all"
    assert not {"max_models", "rank_by", "loss_rank"} & set(meta)
    assert meta["rules_sha256"] == snapshot.rules_sha256(config.prompt_validator, snapshot.base_rows(config))


def test_switching_to_every_model_removes_an_earlier_loss_source(config, chain, ranked, source):
    snapshot.run_snapshot(config, CYCLE_ID)
    assert _source_path(config).exists()
    config.prompt_validator.model_selection = "all"

    snapshot.run_snapshot(config, CYCLE_ID)

    assert not _source_path(config).exists() and not _key_path(config).exists()
    assert len(_models(config)) == len(UIDS)


def test_a_later_stage_stops_when_the_selection_changes_after_the_snapshot(config, chain, ranked, source):
    [meta] = snapshot.run_snapshot(config, CYCLE_ID)
    write_rows(stage_path(config.prompt_validator.runs_dir, CYCLE_ID, "snapshot"), [meta])
    config.prompt_validator.model_selection = "all"

    with pytest.raises(RuntimeError, match="rules_sha256"):
        snapshot.check_rules(config, meta)


def test_model_selection_defaults_to_the_top_models_and_refuses_another_value():
    assert PromptValidatorCfg(prompts_path=Path("/prompts.jsonl")).model_selection == "top"
    with pytest.raises(ValidationError, match="model_selection"):
        PromptValidatorCfg(prompts_path=Path("/prompts.jsonl"), model_selection="every")
