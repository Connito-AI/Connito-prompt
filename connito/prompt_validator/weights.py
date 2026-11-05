"""Stage 8, a dry run: the weights the training validator would set if this cycle's prompt rewards took prompt_share
of the miners' weight. It reads this hotkey's weight row on chain and writes what blending would make of it to
weights.jsonl; it never sets weights, so the training validator stays the only submitter."""
from __future__ import annotations

import threading
from typing import Any, NamedTuple

import bittensor

from connito.prompt_validator.stage_io import read_rows, stage_path
from connito.shared import bt_compat
from connito.shared.app_logging import structlog
from connito.shared.chain import SUBNET_OWNER_UID, reserve_subnet_owner_share

logger = structlog.get_logger(__name__)

READ_TIMEOUT_S = 120  # a stalled node must not hold the loop, which waits here under the cycle lock


class PromptPart(NamedTuple):
    weights: dict[int, float]
    dropped: list[str]  # rewarded hotkeys no longer registered


class ChainRow(NamedTuple):
    metagraph: Any  # lite: hotkeys, uids, block, last_update
    uid: int
    weights: dict[int, float]  # this hotkey's row, normalized


class Blend(NamedTuple):
    miners: dict[int, float]
    guards: list[str]  # why a side took all of the miners' weight


def normalized(weights: dict[int, float]) -> dict[int, float]:
    """The positive weights, scaled to sum 1; empty if none is positive."""
    kept = {uid: w for uid, w in weights.items() if w > 0}
    total = sum(kept.values())
    return {uid: w / total for uid, w in kept.items()} if total > 0 else {}


def miner_part(chain_row: dict[int, float]) -> dict[int, float]:
    """The training validator's row without the subnet owner's share, renormalized."""
    # Every row it sets (its own, or a fallback's) gives uid 0 the owner's share and nothing else, so all of uid 0's
    # weight is the owner's: a row of uid 0 alone has no miner part, and u16 rounding leaves no residue there.
    return normalized({uid: w for uid, w in chain_row.items() if uid != SUBNET_OWNER_UID})


def prompt_part(rewards: list[dict], metagraph) -> PromptPart:
    """reward.jsonl's weights by hotkey on the uids that hold those hotkeys now, renormalized."""
    uid_of = dict(zip(metagraph.hotkeys, (int(uid) for uid in metagraph.uids), strict=True))
    by_uid: dict[int, float] = {}
    dropped = []
    for row in rewards:
        if row["weight"] <= 0:
            continue
        uid = uid_of.get(row["hotkey"])
        if uid is None:
            dropped.append(row["hotkey"])
        elif uid != SUBNET_OWNER_UID:  # the owner gets its share and never more
            by_uid[uid] = by_uid.get(uid, 0.0) + row["weight"]
    return PromptPart(normalized(by_uid), dropped)


def blend(training: dict[int, float], prompt: dict[int, float], prompt_share: float) -> Blend:
    """prompt_share of the miners' weight to the prompt part, the rest to the training part; an empty side gives its
    share to the other."""
    if not training and not prompt:
        return Blend({}, ["no training miner weight and no prompt reward: all to the subnet owner"])
    if not prompt:
        return Blend(training, ["no prompt reward: the training part takes all the miners' weight"])
    if not training:
        return Blend(prompt, ["no miner weight in the training row: the prompt part takes all the miners' weight"])
    return Blend({uid: (1 - prompt_share) * training.get(uid, 0.0) + prompt_share * prompt.get(uid, 0.0)
                  for uid in training.keys() | prompt.keys()}, [])


def run_weights(config, scoring_cycle_id: int) -> list[dict]:
    """The dry run's one row. A measurement: a failure is logged and recorded in the row, never raised."""
    row = {"scoring_cycle_id": scoring_cycle_id, "prompt_share": config.prompt_validator.weights.prompt_share}
    try:
        row.update(dry_run(config, scoring_cycle_id))
    except TimeoutError:
        row["error"] = "timeout"
        logger.warning("weights dry run failed", scoring_cycle_id=scoring_cycle_id, error="timeout",
                       timeout_s=READ_TIMEOUT_S)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:500]
        logger.warning("weights dry run failed", scoring_cycle_id=scoring_cycle_id, error=row["error"])
    return [row]


def hotkey_address(wallet) -> str:
    """The hotkey's ss58, from its public file when there is one, so no private key is opened for an address."""
    try:
        return wallet.hotkeypub.ss58_address
    except FileNotFoundError:  # only newer btcli writes hpub.txt; the config opens the hotkey at start anyway
        return wallet.hotkey.ss58_address


def read_chain(config, hotkey: str) -> ChainRow:
    """The lite metagraph at the head, and this hotkey's weight row alone."""
    netuid = config.chain.netuid
    subtensor = bittensor.Subtensor(network=config.chain.network)
    metagraph = bt_compat.get_metagraph(subtensor, netuid)
    if hotkey not in metagraph.hotkeys:
        raise RuntimeError(f"hotkey {hotkey} is not registered on subnet {netuid}")
    uid = metagraph.hotkeys.index(hotkey)
    neuron = bt_compat.neuron_for_uid(subtensor, uid, netuid)
    if neuron is None or neuron.hotkey != hotkey:
        raise RuntimeError(f"uid {uid} changed hands during the read")
    return ChainRow(metagraph, uid, normalized({u: float(w) for u, w in neuron.weights}))  # raw u16 pairs


def timed_read(config, hotkey: str) -> ChainRow:
    """read_chain, or TimeoutError after READ_TIMEOUT_S."""
    done: dict = {}

    def read() -> None:
        try:
            done["row"] = read_chain(config, hotkey)
        except Exception as e:
            done["error"] = e

    # A daemon thread: one stuck on a dead connection is left behind and never holds up the process's exit.
    thread = threading.Thread(target=read, name="weights-chain-read", daemon=True)
    thread.start()
    thread.join(READ_TIMEOUT_S)
    if thread.is_alive():
        raise TimeoutError
    if "error" in done:
        raise done["error"]
    return done["row"]


def dry_run(config, scoring_cycle_id: int) -> dict:
    pv = config.prompt_validator
    rewards = read_rows(stage_path(pv.runs_dir, scoring_cycle_id, "reward"))
    hotkey = hotkey_address(config.chain.wallet())  # the hotkey that signs the dataless-api calls
    # The head is enough: the row as the training validator last set it is what a blend would replace.
    metagraph, uid, on_chain = timed_read(config, hotkey)
    prompt = prompt_part(rewards, metagraph)
    mixed = blend(miner_part(on_chain), prompt.weights, pv.weights.prompt_share)
    combined = normalized(reserve_subnet_owner_share(mixed.miners))
    ranked = sorted(combined.items(), key=lambda item: (-item[1], item[0]))
    return {
        "block": metagraph.block, "hotkey": hotkey, "uid": uid,
        "last_update_age_blocks": int(metagraph.block - metagraph.last_update[uid]),
        "training": dict(sorted(on_chain.items())), "prompt": dict(sorted(prompt.weights.items())),
        "combined": dict(sorted(combined.items())),
        "l1_to_chain": sum(abs(combined.get(u, 0.0) - on_chain.get(u, 0.0)) for u in combined.keys() | on_chain),
        "top10": [[u, w] for u, w in ranked[:10]], "dropped_hotkeys": prompt.dropped, "guards": mixed.guards,
    }
