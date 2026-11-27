"""Judging methods, chosen per task by prompt_validator.scoring_methods (plain_levels for a task not listed). The format
check before and the length multiplier after are the scores stage's, the same for every method."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

from pydantic import BaseModel


@dataclass(frozen=True)
class Item:
    prompt: dict  # the prompt's snapshot row
    answer: str
    model_hash: str | None = None  # the answering model's, for a method that ranks one place per model


@dataclass(frozen=True)
class Context:
    task: str | None
    settings: BaseModel
    client: Any  # the cycle's JevClient, so every call counts toward the cycle's spend cap
    snapshot_dir: Path | None = None  # the cycle's snapshot stage, for what a method keeps there
    meta: dict | None = None  # the snapshot's meta row


@dataclass(frozen=True)
class Scored:
    score: float  # 0 to 5: what the answer is ranked on
    method: str  # a fallback names its own
    level: float | None = None  # 0 to 5: the likeliest level, for a method with levels
    call_id: str | None = None
    details: dict = field(default_factory=dict)


class MethodStop(RuntimeError):
    """The method cannot score as the rules hash says: the loop stops rather than retry."""


class Method(Protocol):
    name: ClassVar[str]
    RULES: ClassVar[str]  # in the rules hash for the tasks that use it
    DETAILS: ClassVar[tuple[str, ...]]  # its own score-row fields
    Settings: ClassVar[type[BaseModel]]
    SETTINGS_NOT_HASHED: ClassVar[frozenset[str]]  # where this validator keeps things, not what decides a score

    def check(self, settings: BaseModel) -> None: ...

    def score(self, items: list[Item], ctx: Context) -> list[Scored]: ...


def _registry() -> dict[str, Method]:
    from connito.prompt_validator.methods.eps_error import EpsError
    from connito.prompt_validator.methods.plain_levels import PlainLevels
    from connito.prompt_validator.methods.reference_pairwise import ReferencePairwise

    return {m.name: m for m in (PlainLevels(), ReferencePairwise(), EpsError())}


REGISTRY: dict[str, Method] = _registry()


def check_all(pv) -> None:
    """Raises MethodStop before a cycle's snapshot or Jev calls if a listed method could not score."""
    for cfg in pv.scoring_methods.values():
        REGISTRY[cfg.method].check(cfg.parsed())


def hashed(name: str, settings: BaseModel) -> dict:
    """A method may hash its settings and rules its own way (`hashed_settings`, `rules`), else all its settings bar
    SETTINGS_NOT_HASHED, and RULES."""
    method = REGISTRY[name]
    dump = (method.hashed_settings(settings) if hasattr(method, "hashed_settings")
            else settings.model_dump(mode="json", exclude=set(method.SETTINGS_NOT_HASHED)))
    return {"method": name, "settings": dump,
            "rules": method.rules(settings) if hasattr(method, "rules") else method.RULES}
