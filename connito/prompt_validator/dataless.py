"""Client for the dataless-api (the prompt-record service): the prompts in, and the scores and model outputs out.

Every validator request, and test mode's cycle view (/v1/test serves the server's TEST_HOTKEYS only), is signed with
btauth/1 (`bittensor.http_auth.sign`, bittensor >= 11) on the exact path and body sent, a fresh nonce each attempt.
A validator lists prompts and starts a run only for the most recently closed prompt cycle, and every row is written
once: an identical re-post is a no-op, a different value is refused (409 already_written).
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Iterable
from urllib.parse import quote, urlencode

import requests

from connito.prompt_validator.config import DatalessCfg
from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)

REAL_API, TEST_API = "/v1", "/v1/test"  # test mode: the same routes on the server's separate test tables
TEST_CYCLE_FIRST_ID = 1_000_000_000  # the server's test cycle ids start here
BODY_CAP = 1_677_721  # the server's cap on a validator request body, in bytes
MAX_ROWS = 1000  # the server's cap on rows per batch
OUTPUT_MAX_BYTES = 12_288  # the server's cap on one model output, in UTF-8 bytes
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRY_AFTER = 60.0
SNAPSHOT_DELAY_BLOCKS = 10  # the server lists a cycle from its cutoff block plus this

HOTKEY_HEADER = "X-Bittensor-Hotkey"  # bittensor.http_auth.HEADER_HOTKEY: the signer's ss58 address, public

# signer(method, target, body) -> the auth headers for that request
Signer = Callable[[str, str, bytes], dict[str, str]]


class DatalessError(RuntimeError):
    def __init__(self, status: int, code: str, message: str, extra: dict | None = None):
        super().__init__(f"dataless-api {status} {code}: {message}")
        self.status, self.code, self.extra = status, code, extra or {}


class TesterRefused(DatalessError):
    """A /v1/test route refused our hotkey (403): off the server's TEST_HOTKEYS, or without the route's role. Only the
    owner can change that, so it is never retried."""


def encode_body(obj) -> bytes:
    """The exact bytes sent and signed."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def wallet_signer(wallet) -> Signer:
    """Sign with the wallet's hotkey, each request with a nonce above the last (btauth/1 refuses a reused one)."""
    from bittensor import http_auth  # bittensor >= 11

    lock, last = threading.Lock(), [0]

    def sign(method: str, target: str, body: bytes) -> dict[str, str]:
        with lock:
            last[0] = max(time.time_ns(), last[0] + 1)
            nonce = last[0]
        return http_auth.sign(wallet, method=method, path=target, body=body, nonce_ns=nonce)

    return sign


def validator_client(config) -> DatalessClient:
    """The dataless-api client, signing with the config's hotkey."""
    return DatalessClient(config.prompt_validator.dataless, wallet_signer(config.chain.wallet()))


def snapshot_hash(prompts: Iterable[dict]) -> str:
    """The server's snapshot hash: sha256 over the prompts in prompt_id order, each its prompt_id as 8 bytes big
    endian and its 32-byte content_hash."""
    h = hashlib.sha256()
    for p in sorted(prompts, key=lambda p: p["prompt_id"]):
        h.update(int(p["prompt_id"]).to_bytes(8, "big") + bytes.fromhex(p["content_hash"]))
    return h.hexdigest()


def chunk_rows(rows: list[dict], wrap: Callable[[list[dict]], dict]) -> list[list[dict]]:
    """Split rows into batches of at most MAX_ROWS whose wrapped body stays under BODY_CAP."""
    batches: list[list[dict]] = [[]]
    for row in rows:
        batch = batches[-1]
        if batch and (len(batch) == MAX_ROWS or len(encode_body(wrap([*batch, row]))) > BODY_CAP):
            batches.append([])
        batches[-1].append(row)
        if len(encode_body(wrap(batches[-1]))) > BODY_CAP:
            raise ValueError(f"one row is over the {BODY_CAP}-byte body cap")
    return [b for b in batches if b]


def _retry_after(value: str | None, default: float) -> float:
    """Retry-After in seconds; an HTTP date or anything unreadable falls back to the backoff."""
    try:
        return max(float(value), 0.0) if value is not None else default
    except ValueError:
        return default


class DatalessClient:
    def __init__(self, cfg: DatalessCfg, signer: Signer | None = None, session: requests.Session | None = None):
        self.cfg = cfg
        self.base = cfg.url.rstrip("/")
        self.api = TEST_API if cfg.test_mode else REAL_API
        self.prefix = f"{self.api}/validator"  # signed as sent: a test request signs its /v1/test path
        self._test_not_found = 0
        self.signer = signer
        self.http = session or requests.Session()

    def request(self, method: str, target: str, payload=None, signed: bool = True) -> dict:
        """One API call, retried on network errors, 429 and 5xx (after Retry-After). Errors keep the server's code.
        Bodies are never logged: they hold the private prompts and outputs."""
        body = b"" if payload is None else encode_body(payload)
        # What each request is logged with: the query may carry a page token, and the body private text.
        route = target.split("?")[0]
        rows = {"rows": len(payload["rows"])} if isinstance(payload, dict) and "rows" in payload else {}
        for attempt in range(self.cfg.retries + 1):
            headers = {"Content-Type": "application/json"} if body else {}
            if signed:
                if self.signer is None:
                    raise RuntimeError("the dataless-api client has no signer: a validator call needs the hotkey")
                headers |= self.signer(method, target, body)  # signed afresh: each attempt has its own nonce
            try:
                r = self.http.request(method, self.base + target, data=body or None, headers=headers,
                                      timeout=self.cfg.timeout_sec, allow_redirects=False)
            except requests.RequestException as e:
                if attempt == self.cfg.retries:
                    raise RuntimeError(f"dataless-api {method} {route} failed: {type(e).__name__}") from None
                logger.warning("dataless-api unreachable, will retry", method=method, route=route, attempt=attempt + 1,
                               error=type(e).__name__)
                time.sleep(min(2.0 ** attempt, MAX_RETRY_AFTER))
                continue
            # Reads that went through, and test mode's 404 while no test cycle is open, are the polls: debug only.
            quiet = method == "GET" and attempt == 0 and (
                r.status_code < 300 or (r.status_code == 404 and route == f"{TEST_API}/prompt-cycle"))
            (logger.debug if quiet else logger.info)("dataless-api request", method=method, route=route,
                                                     status=r.status_code, attempt=attempt + 1, **rows)
            if r.status_code < 300:
                return r.json() if r.content else {}
            detail = self._detail(r)
            if r.status_code == 403 and route.startswith(f"{TEST_API}/"):
                who = f"hotkey {headers[HOTKEY_HEADER]}" if HOTKEY_HEADER in headers else "an unsigned request"
                raise TesterRefused(r.status_code, detail.get("code", "error"), (
                    f"{who} refused on {method} {route}: {detail.get('message', '')} (/v1/test needs the hotkey on "
                    "TEST_HOTKEYS, and its validator routes also a whitelisted validator)"))
            if r.status_code in RETRY_STATUSES and attempt < self.cfg.retries:
                wait = min(_retry_after(r.headers.get("Retry-After"), 2.0 ** attempt), MAX_RETRY_AFTER)
                logger.warning("dataless-api busy, will retry", status=r.status_code, code=detail.get("code"),
                               wait_sec=wait)
                time.sleep(wait)
                continue
            raise DatalessError(r.status_code, detail.get("code", "error"), detail.get("message", ""),
                                {k: v for k, v in detail.items() if k not in ("code", "message")})
        raise AssertionError("unreachable")

    @staticmethod
    def _detail(r) -> dict:
        try:
            detail = r.json().get("detail")
        except ValueError:
            return {"code": "http_error", "message": f"{len(r.content)}-byte non-JSON answer"}
        return detail if isinstance(detail, dict) else {"code": "http_error", "message": str(detail)[:200]}

    # --- reads ---

    def prompt_cycle(self) -> dict:
        """The open prompt cycle and its tasks. The one a validator scores is cycle_id - 1. Unsigned in real mode; the
        test cycle's is signed, and served to the server's TEST_HOTKEYS only."""
        return self.request("GET", f"{self.api}/prompt-cycle", signed=self.cfg.test_mode)

    def list_prompts(self, cycle_id: int) -> dict:
        """Every prompt of a closed cycle, all tasks, checked against the server's snapshot hash and count.

        Returns {snapshot_hash, snapshot_count, prompts}. With several tasks the server lists one at a time.
        """
        try:
            first = self._list_task(cycle_id, None)
            pages = [first]
        except DatalessError as e:
            if e.code != "task_required":
                raise
            pages = [self._list_task(cycle_id, task) for task in e.extra.get("customer_task_ids", [])]
        prompts = [p for page in pages for p in page["prompts"]]
        for p in prompts:
            if hashlib.sha256(p["text"].encode()).hexdigest() != p["content_hash"]:
                raise RuntimeError(f"dataless-api prompt {p['prompt_id']}: its text does not match its content_hash")
        expected = {(page["snapshot_hash"], page["snapshot_count"]) for page in pages}
        if len(expected) != 1:
            raise RuntimeError(f"dataless-api gave different snapshot hashes across tasks: {sorted(expected)}")
        [(want_hash, want_count)] = expected
        got = snapshot_hash(prompts)
        if got != want_hash or len(prompts) != want_count:
            raise RuntimeError(
                f"dataless-api snapshot does not check out: {len(prompts)} prompts hash to {got[:12]}, "
                f"the server says {want_count} and {want_hash[:12]}"
            )
        return {"snapshot_hash": want_hash, "snapshot_count": want_count, "prompts": prompts}

    def _list_task(self, cycle_id: int, task: str | None) -> dict:
        page_token, out = None, None
        while True:
            query = {k: v for k, v in (("customer_task_id", task), ("page_token", page_token)) if v is not None}
            target = f"{self.prefix}/cycles/{cycle_id}/prompts" + (f"?{urlencode(query)}" if query else "")
            page = self.request("GET", target)
            if out is None:
                out = {**page, "prompts": []}
            out["prompts"] += page["prompts"]
            page_token = page.get("next_page_token")
            if not page_token:
                return out

    def task_request(self, customer_task_id: str) -> dict:
        """The customer request behind a task open now or in the cycle before; always under /v1 (no test copy)."""
        return self.request("GET", f"{REAL_API}/tasks/{quote(customer_task_id, safe='')}/request")

    # --- the run ---

    def start_run(self, cycle_id: int, details: dict) -> dict:
        return self.request("POST", f"{self.prefix}/cycles/{cycle_id}/status", {"status": "scoring", **details})

    def post_prompt_usefulness(self, cycle_id: int, rows: list[dict]) -> list[dict]:
        """Each miner prompt's verdict; returns the server's verdicts on the judged sets the batches touched."""
        sets: list[dict] = []
        for batch in chunk_rows(rows, lambda b: {"rows": b}):
            sets += self.request("POST", f"{self.prefix}/cycles/{cycle_id}/prompt-usefulness", {"rows": batch})["sets"]
        return sets

    def post_models(self, cycle_id: int, rows: list[dict]) -> dict[str, int]:
        """The run's models, one per miner hotkey; returns hotkey -> model_id (a re-post gives the same ids)."""
        ids: dict[str, int] = {}
        for batch in chunk_rows(rows, lambda b: {"rows": b}):
            ids |= self.request("POST", f"{self.prefix}/cycles/{cycle_id}/models", {"rows": batch})["model_ids"]
        return ids

    def post_summary(self, cycle_id: int, rows: list[dict]) -> None:
        for batch in chunk_rows(rows, lambda b: {"rows": b}):
            self.request("POST", f"{self.prefix}/cycles/{cycle_id}/summary", {"rows": batch})

    def post_model_scores(self, cycle_id: int, rows: list[dict]) -> None:
        for batch in chunk_rows(rows, lambda b: {"rows": b}):
            self.request("POST", f"{self.prefix}/cycles/{cycle_id}/model-scores", {"rows": batch})

    def post_prompt_model_scores(self, cycle_id: int, rows: list[dict]) -> None:
        for batch in chunk_rows(rows, lambda b: {"rows": b}):
            self.request("POST", f"{self.prefix}/cycles/{cycle_id}/prompt-model-scores", {"rows": batch})

    def publish(self, cycle_id: int) -> dict:
        """Publish the run. A publish whose answer was lost is retried into 409 cycle_published: that is success."""
        try:
            return self.request("POST", f"{self.prefix}/cycles/{cycle_id}/status", {"status": "published"})
        except DatalessError as e:
            if e.code != "cycle_published":
                raise
            return {"status": "published", "cycle_id": cycle_id}

    def wait_until_listed(self, cycle_id: int, block_sec: float = 12.0, poll_sec: float = 600.0) -> dict:
        """Wait until the API can list prompt cycle `cycle_id`: from its cutoff block plus the snapshot delay, until
        the next cycle closes. Returns the cycle view. A cycle the API has moved past stops with an error.
        In test mode a cycle has no blocks: the owner closes it, so this waits for its state to be closed."""
        if self.cfg.test_mode:
            return self._wait_test_closed(cycle_id)
        while True:
            view = self.prompt_cycle()  # the open cycle
            if view["cycle_id"] - 1 > cycle_id:
                raise RuntimeError(
                    f"the dataless-api lists prompt cycle {view['cycle_id'] - 1} now, not {cycle_id}: it lists the "
                    "most recently closed one only"
                )
            cutoff = view["start_block"] + (cycle_id - view["cycle_id"] + 1) * view["cycle_length"] - 1
            ready = cutoff + SNAPSHOT_DELAY_BLOCKS
            if view["head"] >= ready:
                return view
            logger.info("waiting for the dataless-api to list the prompt cycle", cycle=cycle_id, head=view["head"],
                        ready_at_block=ready)
            time.sleep(min((ready - view["head"]) * block_sec, poll_sec))

    # --- test cycles (test_mode): opened and closed by the owner, no blocks ---

    def test_cycle_view(self) -> dict | None:
        """The open test cycle, or else the most recently opened one; None when the API has none to show."""
        try:
            return self.prompt_cycle()
        except DatalessError as e:
            if e.code != "not_found":
                raise
            self._test_not_found += 1
            if self._test_not_found == 1 or self._test_not_found % 20 == 0:
                logger.warning("the dataless-api shows no test cycle: none opened yet, or test cycles are off on the "
                               "server (TEST_CYCLES_ENABLED), or the url is wrong", url=self.base,
                               times=self._test_not_found)
            return None

    @staticmethod
    def listed_test_cycle(view: dict | None) -> int | None:
        """The test cycle the API lists now: the most recently closed one. That is the cycle shown when it is
        closed; while a newer one is open, it is the one before (only one test cycle is open at a time)."""
        if view is None:
            return None
        if view.get("state") == "closed":
            return view["cycle_id"]
        before = view["cycle_id"] - 1
        return before if before >= TEST_CYCLE_FIRST_ID else None

    def next_closed_test_cycle(self, done: Callable[[int], bool]) -> int:
        """Wait for a listed test cycle that `done` says has not been run yet, and return its id."""
        while True:
            view = self.test_cycle_view()
            listed = self.listed_test_cycle(view)
            if listed is not None and not done(listed):
                return listed
            logger.info("waiting for a closed test cycle", shown=view and view["cycle_id"],
                        state=view and view.get("state"))
            time.sleep(self.cfg.test_poll_sec)

    def _wait_test_closed(self, cycle_id: int) -> dict:
        while True:
            view = self.test_cycle_view()
            listed = self.listed_test_cycle(view)
            if listed == cycle_id:
                return view
            if listed is not None and listed > cycle_id:
                raise RuntimeError(
                    f"the dataless-api lists test cycle {listed} now, not {cycle_id}: it lists the most recently "
                    "closed test cycle only"
                )
            logger.info("waiting for the test cycle to be closed", cycle=cycle_id, shown=view and view["cycle_id"],
                        state=view and view.get("state"))
            time.sleep(self.cfg.test_poll_sec)
