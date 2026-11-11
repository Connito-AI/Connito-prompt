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
