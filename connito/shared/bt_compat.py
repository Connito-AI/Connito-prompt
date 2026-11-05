"""bittensor 11 chain access, in the shapes the rest of connito already reads.

bittensor 11 replaced the v10 SDK: ``Subtensor`` keeps its name but lost its
~165 methods (reads moved onto namespaces, writes became intents), the
metagraph became per-neuron records without NumPy columns, ``AsyncSubtensor``,
``Axon`` and ``Keypair`` left the top level, and ``bittensor_wallet`` was
folded in. This module is the one place that knows the v11 API. Every helper
returns what the v10 call it replaces returned, decoded the way v10 decoded
it, so the callers' arithmetic (stake-weighted votes, incentive ranking,
weight-age checks, the block-hash seed) is unchanged.

Sync helpers take a blocking ``bittensor.Subtensor``; the ``a``-prefixed ones
take the async ``bittensor.Client`` (what ``await bittensor.Subtensor(...)``
returns).
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from typing import Any

import bittensor
import numpy as np
import requests
from bittensor.result import BittensorError, ChainError, RpcConnectionError
from bittensor.sp_core import KeyfileError, Keypair

__all__ = [
    "AxonInfo",
    "BittensorError",
    "ChainError",
    "Keypair",
    "KeyfileError",
    "MetagraphView",
    "NeuronInfo",
    "RpcConnectionError",
    "V10_VERSION_AS_INT",
    "aget_metagraph",
    "aneuron_for_uid",
    "aset_commitment",
    "aset_weights",
    "get_all_commitments",
    "get_block_hash",
    "get_metagraph",
    "get_subnet_owner_hotkey",
    "get_tempo",
    "neuron_for_uid",
    "serve_axon",
    "set_commitment",
    "set_weights",
]

U16_MAX = 65535

# bittensor 10.5.0's `settings.version_as_int`. v10 sent it as the weights
# `version_key` and as the served axon's `version`; v11 defaults both to 0/1.
# The chain rejects weights whose version_key is below the subnet's
# `weights_version`, so keep sending what the validators sent until now.
V10_VERSION_AS_INT = 10_005_000


def _u16_float(value: Any) -> float:
    return float(int(value or 0)) / float(U16_MAX)


def _int_to_ip(value: int, ip_type: int) -> str:
    # v10's net.int_to_ip: an unserved axon (ip 0) renders as "0.0.0.0".
    if ip_type == 6 or value > 0xFFFFFFFF:
        return str(ipaddress.IPv6Address(value))
    return str(ipaddress.IPv4Address(value))


@dataclass
class AxonInfo:
    """The v10 ``AxonInfo`` fields connito reads."""

    ip: str
    port: int
    ip_type: int
    version: int
    protocol: int
    hotkey: str
    coldkey: str

    @classmethod
    def from_raw(cls, raw: dict | None, hotkey: str, coldkey: str) -> AxonInfo:
        raw = raw or {}
        ip_type = int(raw.get("ip_type") or 0)
        return cls(
            ip=_int_to_ip(int(raw.get("ip") or 0), ip_type),
            port=int(raw.get("port") or 0),
            ip_type=ip_type,
            version=int(raw.get("version") or 0),
            protocol=int(raw.get("protocol") or 0),
            hotkey=hotkey,
            coldkey=coldkey,
        )

    def ip_str(self) -> str:
        return f"/ipv{self.ip_type}/{self.ip}:{self.port}"


@dataclass
class NeuronInfo:
    """A neuron as v10's ``NeuronInfo`` / ``NeuronInfoLite`` decoded it.

    Scores are u16 fractions (0..1), ``stake`` and ``emission`` are floats in
    the subnet's units (v10's ``float(Balance)``), and ``weights`` / ``bonds``
    are the raw ``(uid, u16)`` pairs (only filled for full, non-lite reads).
    """

    uid: int
    netuid: int
    hotkey: str
    coldkey: str
    active: bool
    stake: float
    last_update: int
    validator_permit: bool
    validator_trust: float
    consensus: float
    incentive: float
    dividends: float
    emission: float
    axon_info: AxonInfo
    weights: list[tuple[int, int]] = field(default_factory=list)
    bonds: list[tuple[int, int]] = field(default_factory=list)

    @property
    def total_stake(self) -> float:
        return self.stake

    @classmethod
    def from_raw(cls, raw: dict) -> NeuronInfo:
        hotkey = str(raw["hotkey"])
        coldkey = str(raw["coldkey"])
        stake_rao = sum(int(amount) for _, amount in raw.get("stake") or [])
        return cls(
            uid=int(raw["uid"]),
            netuid=int(raw.get("netuid") or 0),
            hotkey=hotkey,
            coldkey=coldkey,
            active=bool(raw.get("active")),
            stake=stake_rao / 1e9,
            last_update=int(raw.get("last_update") or 0),
            validator_permit=bool(raw.get("validator_permit")),
            validator_trust=_u16_float(raw.get("validator_trust")),
            consensus=_u16_float(raw.get("consensus")),
            incentive=_u16_float(raw.get("incentive")),
            dividends=_u16_float(raw.get("dividends")),
            emission=int(raw.get("emission") or 0) / 1e9,
            axon_info=AxonInfo.from_raw(raw.get("axon_info"), hotkey, coldkey),
            weights=[(int(u), int(w)) for u, w in raw.get("weights") or []],
            bonds=[(int(u), int(b)) for u, b in raw.get("bonds") or []],
        )


@dataclass
class MetagraphView:
    """The v10 ``Metagraph`` columns connito reads, built from one neurons read.

    v10 built its metagraph from ``neurons_lite`` (or ``neurons`` when
    ``lite=False``); this does the same and exposes the same NumPy columns.
    ``weights`` is the row-normalized N x N float32 matrix v10 filled for
    ``lite=False`` and is None for lite reads.
    """

    netuid: int
    block: int
    neurons: list[NeuronInfo]
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.hotkeys: list[str] = [n.hotkey for n in self.neurons]
        self.coldkeys: list[str] = [n.coldkey for n in self.neurons]
        self.uids = np.array([n.uid for n in self.neurons], dtype=np.int64)
        self.n = len(self.neurons)
        self.incentive = np.array([n.incentive for n in self.neurons], dtype=np.float32)
        self.consensus = np.array([n.consensus for n in self.neurons], dtype=np.float32)
        self.dividends = np.array([n.dividends for n in self.neurons], dtype=np.float32)
        self.validator_trust = np.array([n.validator_trust for n in self.neurons], dtype=np.float32)
        self.emission = np.array([n.emission for n in self.neurons], dtype=np.float32)
        self.last_update = np.array([n.last_update for n in self.neurons], dtype=np.int64)
        self.active = np.array([n.active for n in self.neurons], dtype=np.int64)
        self.validator_permit = np.array([n.validator_permit for n in self.neurons], dtype=bool)
        self.axons: list[AxonInfo] = [n.axon_info for n in self.neurons]

    @classmethod
    def from_neurons(cls, netuid: int, block: int, raws: list[dict], lite: bool) -> MetagraphView:
        neurons = sorted((NeuronInfo.from_raw(r) for r in raws), key=lambda n: n.uid)
        weights = None if lite else _weights_matrix(neurons)
        return cls(netuid=netuid, block=block, neurons=neurons, weights=weights)


def _weights_matrix(neurons: list[NeuronInfo]) -> np.ndarray:
    """v10's ``_process_weights_or_bonds(..., "weights")``: one row per neuron,
    each row's u16 values scattered by uid and normalized to sum 1."""
    n = len(neurons)
    rows = []
    for neuron in neurons:
        row = np.zeros(n, dtype=np.float32)
        for uid, value in neuron.weights:
            row[uid] = float(value)
        total = row.sum()
        if total > 0:
            row /= total
        rows.append(row)
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(rows)


def _raw_records(neurons: list[Any]) -> list[dict]:
    return [n.raw for n in neurons]


# --- Reads -------------------------------------------------------------------


def get_metagraph(
    subtensor: bittensor.Subtensor, netuid: int, *, block: int | None = None, lite: bool = True,
) -> MetagraphView:
    """v10 ``subtensor.metagraph(netuid, lite=..., block=...)``."""
    view = subtensor.at(block)
    neurons = view.neurons.all(netuid, lite=lite)
    return MetagraphView.from_neurons(netuid, int(view.block), _raw_records(neurons), lite)


async def aget_metagraph(
    client: bittensor.Client, netuid: int, *, block: int | None = None, lite: bool = True,
) -> MetagraphView:
    view = await client.at(block)
    neurons = await view.neurons.all(netuid, lite=lite)
    return MetagraphView.from_neurons(netuid, int(view.block), _raw_records(neurons), lite)


def neuron_for_uid(subtensor: bittensor.Subtensor, uid: int, netuid: int) -> NeuronInfo | None:
    """v10 ``subtensor.neuron_for_uid``: the full record (with weights), or None."""
    raw = subtensor.runtime(bittensor.runtime_api.NeuronInfoRuntimeApi.get_neuron, [netuid, uid])
    return NeuronInfo.from_raw(raw) if isinstance(raw, dict) else None


async def aneuron_for_uid(client: bittensor.Client, uid: int, netuid: int) -> NeuronInfo | None:
    raw = await client.runtime(bittensor.runtime_api.NeuronInfoRuntimeApi.get_neuron, [netuid, uid])
    return NeuronInfo.from_raw(raw) if isinstance(raw, dict) else None


def _decode_commitment(record: Any) -> str:
    """v10's ``decode_metadata``: the first field's bytes as UTF-8."""
    first = record["info"]["fields"][0]
    if isinstance(first, str):
        return ""
    hex_ = next(iter(first.values()))
    if isinstance(hex_, bytes | bytearray):
        return bytes(hex_).decode("utf-8", errors="ignore")
    return bytes.fromhex(str(hex_).removeprefix("0x")).decode("utf-8", errors="ignore")


def get_all_commitments(
    subtensor: bittensor.Subtensor, netuid: int, *, block: int | None = None,
) -> dict[str, str]:
    """v10 ``subtensor.get_all_commitments``: hotkey -> committed string.

    Reads the Commitments pallet's ``CommitmentOf`` map directly, as v10 did:
    every stored commitment (registered or not), nothing reconstructed from
    reveal history.
    """
    rows = subtensor.query_map(
        bittensor.storage.Commitments.CommitmentOf, [netuid], block=block,
    )
    result: dict[str, str] = {}
    for hotkey, record in rows:
        try:
            result[str(hotkey)] = _decode_commitment(record)
        except Exception:  # noqa: BLE001 — v10 logged and skipped undecodable rows
            continue
    return result


_ZERO_HASH = "0x" + "00" * 32


def get_block_hash(subtensor: bittensor.Subtensor, block: int) -> str | None:
    """v10 ``subtensor.get_block_hash(block)`` as a 0x-hex string.

    v11 has no public ``chain_getBlockHash``. The System pallet keeps the
    hashes of the last ``BlockHashCount`` blocks in storage, readable from any
    node at head; older blocks fall back to the block's header, which needs a
    node that still has that block's state.
    """
    value = subtensor.query(bittensor.storage.System.BlockHash, [int(block)])
    text = _hash_text(value)
    if text and text != _ZERO_HASH:
        return text
    info = subtensor.block_info(int(block))
    return _hash_text(getattr(info, "hash", None))


def _hash_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes | bytearray):
        return "0x" + bytes(value).hex()
    text = str(value)
    return text.lower() if text.startswith("0x") else text


def get_subnet_owner_hotkey(subtensor: bittensor.Subtensor, netuid: int) -> str | None:
    """v10 ``subtensor.get_subnet_owner_hotkey``."""
    value = subtensor.query(bittensor.storage.SubtensorModule.SubnetOwnerHotkey, [netuid])
    return str(value) if value else None


def get_tempo(subtensor: bittensor.Subtensor, netuid: int, *, block: int | None = None) -> int | None:
    """v10 ``subtensor.tempo``: None when the subnet does not exist at ``block``."""
    exists = subtensor.query(bittensor.storage.SubtensorModule.NetworksAdded, [netuid], block=block)
    if not exists:
        return None
    tempo = subtensor.query(bittensor.storage.SubtensorModule.Tempo, [netuid], block=block)
    return None if tempo is None else int(tempo)


# --- Writes ------------------------------------------------------------------
#
# v10's set_weights / serve_axon returned a failed response instead of raising
# when the chain (or a client-side precheck) refused. v11 raises ChainError
# from its prechecks; these helpers turn that back into False so callers keep
# their v10 control flow. Connection failures (RpcConnectionError) still
# raise, as v10's WebSocket errors did, so the retry loops around them keep
# working. v10's set_commitment caught everything, so set_commitment does too.


def _commitment_call(netuid: int, data: str):
    # v10's publish_metadata: one Raw<len> field holding the UTF-8 bytes.
    info = {"fields": [[{f"Raw{len(data)}": data.encode()}]]}
    return bittensor.calls.Commitments.set_commitment(netuid=netuid, info=info)


def set_commitment(
    subtensor: bittensor.Subtensor, wallet: Any, netuid: int, data: str,
    *, wait_for_inclusion: bool = True, wait_for_finalization: bool = True,
) -> bool:
    """v10 ``subtensor.set_commitment(wallet, netuid, data, raise_error=False)``."""
    try:
        result = subtensor.submit_call(
            _commitment_call(netuid, data), wallet, signer="hotkey",
            wait_for_inclusion=wait_for_inclusion, wait_for_finalization=wait_for_finalization,
        )
    except Exception:  # noqa: BLE001 — v10 returned a failed response for any error here
        return False
    return bool(result)


async def aset_commitment(
    client: bittensor.Client, wallet: Any, netuid: int, data: str,
    *, wait_for_inclusion: bool = True, wait_for_finalization: bool = True,
) -> bool:
    try:
        result = await client.submit_call(
            _commitment_call(netuid, data), wallet, signer="hotkey",
            wait_for_inclusion=wait_for_inclusion, wait_for_finalization=wait_for_finalization,
        )
    except Exception:  # noqa: BLE001 — as set_commitment
        return False
    return bool(result)


# v10's set_weights made up to 5 attempts; v11 resubmits transient pool
# rejections `retries` extra times.
_SET_WEIGHTS_RETRIES = 4


def _weights_intent(netuid: int, uids: list[int], weights: list[float]):
    return bittensor.SetWeights(
        netuid=netuid, uids=[int(u) for u in uids], weights=[float(w) for w in weights],
        version_key=V10_VERSION_AS_INT,
    )


def set_weights(
    subtensor: bittensor.Subtensor, wallet: Any, netuid: int, uids: list[int], weights: list[float],
    *, wait_for_inclusion: bool = True, wait_for_finalization: bool = True,
) -> bool:
    """v10 ``subtensor.set_weights(...)``: plain or commit-reveal, chosen by the chain config."""
    try:
        result = subtensor.execute(
            _weights_intent(netuid, uids, weights), wallet, retries=_SET_WEIGHTS_RETRIES,
            wait_for_inclusion=wait_for_inclusion, wait_for_finalization=wait_for_finalization,
        )
    except RpcConnectionError:
        raise
    except BittensorError:
        return False
    return bool(result)


async def aset_weights(
    client: bittensor.Client, wallet: Any, netuid: int, uids: list[int], weights: list[float],
    *, wait_for_inclusion: bool = True, wait_for_finalization: bool = True,
) -> bool:
    try:
        result = await client.execute(
            _weights_intent(netuid, uids, weights), wallet, retries=_SET_WEIGHTS_RETRIES,
            wait_for_inclusion=wait_for_inclusion, wait_for_finalization=wait_for_finalization,
        )
    except RpcConnectionError:
        raise
    except BittensorError:
        return False
    return bool(result)


_IP_LOOKUPS = (
    ("https://checkip.amazonaws.com", lambda r: r.text.strip()),
    ("https://ifconfig.me/ip", lambda r: r.text.strip()),
    ("https://ipinfo.io/json", lambda r: r.json()["ip"]),
    ("https://ident.me", lambda r: r.text.strip()),
)


def external_ip() -> str:
    """The address v10's ``Axon`` advertised: ``BT_AXON_EXTERNAL_IP`` or a lookup."""
    env_ip = os.getenv("BT_AXON_EXTERNAL_IP")
    if env_ip:
        return env_ip
    for url, parse in _IP_LOOKUPS:
        try:
            candidate = parse(requests.get(url, timeout=10))
            ipaddress.ip_address(candidate)
            return str(candidate)
        except Exception:  # noqa: BLE001 — fall through to the next provider
            continue
    raise RuntimeError("Could not determine this machine's external IP to serve the axon")


def _serve_intent(netuid: int, ip: str, port: int):
    return bittensor.ServeAxon(netuid=netuid, ip=ip, port=int(port), version=V10_VERSION_AS_INT)


def _already_served(neuron: NeuronInfo | None, ip: str, port: int) -> bool:
    if neuron is None:
        return False
    axon = neuron.axon_info
    address = ipaddress.ip_address(ip)
    return (
        axon.ip == str(address) and axon.port == int(port) and axon.ip_type == address.version
        and axon.protocol == 4 and axon.version == V10_VERSION_AS_INT
    )


def serve_axon(subtensor: bittensor.Subtensor, wallet: Any, netuid: int, port: int) -> bool:
    """v10 ``Axon(wallet, external_port=port).serve(netuid, subtensor)``.

    Publishes ``<external ip>:port`` for the hotkey; skipped when the chain
    already holds exactly that endpoint (v10 skipped it the same way).
    """
    ip = external_ip()
    hotkey = wallet.hotkey.ss58_address
    try:
        uid = subtensor.neurons.uid(hotkey_ss58=hotkey, netuid=netuid)
        neuron = neuron_for_uid(subtensor, uid, netuid) if uid is not None else None
        if _already_served(neuron, ip, port):
            return True
        result = subtensor.execute(_serve_intent(netuid, ip, port), wallet)
    except RpcConnectionError:
        raise
    except BittensorError:
        return False
    return bool(result)

