"""Format checks by task: an answer to a prompt of a task listed here that fails its check scores 0 without Jev.

A lesson task's answer must hold a course an app can read: its first JSON object, read as forgivingly as a lenient
app parser would (text after it, a raw newline inside a string), with exactly 3 lessons, each with its fields, and
citing only the prompt's own sources. Jev's levels say broken output scores 0, but Jev reads past a stray brace or
broken quoting and scores most such answers above 0. The rule's text is in the rules hash (snapshot.rules_sha256), so
a change to it is a change of rules. A check that raises fails the answer: an answer never stops the scores stage.
"""
from __future__ import annotations

import json
import re

from connito.prompt_validator.lesson_length import LABELS, course_multiplier, word_target
from connito.shared.app_logging import structlog

logger = structlog.get_logger(__name__)

LESSON_COUNT = 3
LESSON_FIELDS = ("lesson_number", "title", "body", "source_ids")
_LENIENT = json.JSONDecoder(strict=False)
# A source in the prompt is a `<source id="S1" ...>` tag; a lesson cites one in source_ids or inline as [S1].
_SOURCE_TAG = re.compile(r'<source id="(S\d+)"')
_CITATION = re.compile(r"\[(S\d+)\]")

FORMAT_RULES = {
    "exp_lesson_residual": (
        "the answer's first JSON object, read from its first '{' by json.JSONDecoder(strict=False).raw_decode (text "
        "after it ignored, control characters allowed inside strings), is an object whose 'lessons' is a list of "
        f"exactly {LESSON_COUNT} objects, each with {', '.join(LESSON_FIELDS)}, a string body and a list source_ids; "
        "every source_ids entry, and every inline [Sn] in a lesson's title or body, is the id of a <source id=\"Sn\"> "
        "tag in the prompt; an answer that fails scores 0 and is not sent to Jev"
    ),
}


def prompt_sources(prompt: str) -> set[str]:
    """The ids of the prompt's <source id="Sn"> tags."""
    return set(_SOURCE_TAG.findall(prompt))


def lessons_ok(text: str, prompt: str) -> bool:
    """FORMAT_RULES' lesson rule, for an answer to `prompt`."""
    start = text.find("{")
    if start < 0:
        return False
    try:
        course, _ = _LENIENT.raw_decode(text, start)
    except (ValueError, RecursionError):  # json.JSONDecodeError, a number too large for int(), and deep nesting
        return False
    lessons = course.get("lessons") if isinstance(course, dict) else None
    if not isinstance(lessons, list) or len(lessons) != LESSON_COUNT:
        return False
    sources = prompt_sources(prompt)
    for lesson in lessons:
        if not isinstance(lesson, dict) or any(field not in lesson for field in LESSON_FIELDS):
            return False
        if not isinstance(lesson["body"], str) or not isinstance(lesson["source_ids"], list):
            return False
        cited = [*lesson["source_ids"], *_CITATION.findall(f"{lesson['title']}\n{lesson['body']}")]
        if any(not isinstance(source, str) or source not in sources for source in cited):
            return False
    return True


_CHECKS = {"exp_lesson_residual": lessons_ok}
assert _CHECKS.keys() == FORMAT_RULES.keys()  # every check has its rule's text in the rules hash


def format_ok(task: str | None, text: str, prompt: str) -> bool:
    """Whether an answer to `prompt`, of `task` (its customer_task_id), passes the task's check; True with none, False
    if the check raises."""
    check = _CHECKS.get(task)
    if check is None:
        return True
    try:
        return check(text, prompt)
    except Exception as e:  # fail closed: an answer that breaks its check scores 0
        logger.warning("format check raised", task=task, error=type(e).__name__)
        return False


# With lesson_length on, a lesson task's answer that passed its format check scores Jev's score times a length
# multiplier (lesson_length.py). In the rules hash, with the lesson_length settings, only while it is on.
LENGTH_RULES = {
    "exp_lesson_residual": (
        "each lesson body's words: the body string with a style label removed from the start of each line ("
        + ", ".join(repr(label) for label in LABELS) + ", after any leading spaces or tabs), every [Sn] or [Sn, Sm] "
        "marker replaced by a space, then split on whitespace, a piece counting if it holds a \\w character; the "
        "limit is the prompt's one 'each lesson body is plain text of at most N words' before its first "
        "<learner_request> or <source> tag, if N is under cutoff, else default_limit; the multiplier is 1 at or "
        "under the limit, (cutoff - words) / (cutoff - limit) between, 0 at or past cutoff, on the longest lesson's "
        "count or the lessons' mean (aggregate); but if those instructions say both 'Write exactly 3 connected "
        "lessons, each about N words long' and 'Aim for about N words per lesson' (case-insensitive), each with one "
        "N, the same in both, the prompt has a word target N instead: the multiplier is 1 when the lessons' mean is "
        "within target_tolerance of N (ends included), past it 0, or with target_ramp R above 0, falling linearly "
        "to 0 over R more words; a prompt with one of those lines, or two different N, keeps the limit rule; score "
        "and expected_score are Jev's times the multiplier, rounded to 4 places, and rank follows them; an answer "
        "that fails its format check still scores 0"
    ),
}
# Added to every score row while lesson_length is on: Jev's own scores, and what the multiplier was taken from.
LENGTH_FIELDS = ("jev_score", "jev_expected_score", "word_counts", "word_limit", "word_target", "length_multiplier")


def apply_lesson_length(row: dict, task: str | None, prompt: str, checked: bool, cfg) -> dict:
    """`row` (a score row with its answer's text) with LENGTH_FIELDS, and its scores times the length multiplier when
    its task has a length rule and its answer passed the format check (`checked`); else its scores as they are."""
    raw = {"jev_score": row["score"] if checked else None,
           "jev_expected_score": row["expected_score"] if checked else None}
    if task not in LENGTH_RULES or not checked:
        return row | raw | {"word_counts": None, "word_limit": None, "word_target": None, "length_multiplier": None}
    m, counts, words = course_multiplier(row["text"], prompt, cfg.cutoff, cfg.aggregate, cfg.default_limit,
                                         cfg.target_tolerance, cfg.target_ramp)
    m = round(m, 4)
    target = word_target(prompt)  # the rule `words` is from: a target, or else a limit
    return row | raw | {"word_counts": counts, "word_limit": None if target else words, "word_target": target,
                        "length_multiplier": m,
                        "score": round(row["score"] * m, 4), "expected_score": round(row["expected_score"] * m, 4)}
