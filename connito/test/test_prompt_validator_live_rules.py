"""The rules hash production scores with: anyone can check the live rules_sha256 against this code.

The settings are production's prompt_validator section. references_path is where that validator keeps its private
references file, so it is not hashed and any absolute path gives the same hash; the file's sha256 is.
"""
from test_prompt_validator_loss_rank import PRODUCTION_BASE, production  # noqa: F401 (fixture)

from connito.prompt_validator import snapshot
from connito.prompt_validator.config import LessonLengthCfg, ScoringMethodCfg

LIVE_RULES = "b3530d3f8478a2f2c240cf98325cc0310af16e4d38c88ba48ce8aac5c9d1efa9"
REFERENCES_SHA256 = "03f92405a687fae6aa964667a378f18edbc32173f469428724303b72430a9a0c"


def _live(production, references_path="/r/refs.jsonl"):  # noqa: F811
    return production.model_copy(update={
        "rank_by": "loss", "model_selection": "all",
        "lesson_length": LessonLengthCfg(enabled=True, cutoff=300, aggregate="mean", target_tolerance=20,
                                         target_ramp=0),
        "scoring_methods": {"exp_lesson_residual": ScoringMethodCfg(method="reference_pairwise", settings={
            "references_path": references_path, "references_sha256": REFERENCES_SHA256,
            "reference_weight": 1.0, "orders": "b"})},
    })


def test_the_live_rules_hash_is_pinned(production):  # noqa: F811
    assert snapshot.rules_sha256(_live(production), PRODUCTION_BASE) == LIVE_RULES


def test_the_live_rules_hash_ignores_where_the_references_file_is_kept(production):  # noqa: F811
    assert snapshot.rules_sha256(_live(production, "/elsewhere/refs.jsonl"), PRODUCTION_BASE) == LIVE_RULES
