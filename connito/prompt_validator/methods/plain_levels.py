"""plain_levels: Jev grades the answer on its task's levels. The default, and other methods' fallback."""
from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from connito.prompt_validator.jev import MAX_LEVEL
from connito.prompt_validator.methods import Context, Item, Scored


class PlainLevelsSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlainLevels:
    name: ClassVar[str] = "plain_levels"
    # Its question and levels are already in the rules hash (jev_questions, jev_criteria_by_task).
    RULES: ClassVar[str] = ("Jev's score question on the prompt's task's levels; score is the expected level over Jev's "
                            "per-level probabilities, the row's `score` the likeliest level")
    DETAILS: ClassVar[tuple[str, ...]] = ()
    Settings: ClassVar[type[BaseModel]] = PlainLevelsSettings
    SETTINGS_NOT_HASHED: ClassVar[frozenset[str]] = frozenset()

    def check(self, settings: BaseModel) -> None:
        pass

    def score(self, items: list[Item], ctx: Context) -> list[Scored]:
        rows = ctx.client.score_many([(i.prompt["request"], i.prompt["prompt"], i.answer) for i in items],
                                     tasks=[ctx.task] * len(items))
        return [Scored(score=MAX_LEVEL * r["expected_score"], level=MAX_LEVEL * r["score"], method=self.name,
                       call_id=r["jev_call_id"]) for r in rows]
