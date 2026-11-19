"""Lesson word counts and the length multiplier for exp_lesson_residual answers.

Standard library only, with no other import from this repo, so this file can be published unchanged for miners to
run on their own answers:

    from lesson_length import course_multiplier
    multiplier, counts, words = course_multiplier(answer_text, prompt_text)  # words: the target or limit used

How a lesson body's words are counted (count_words):
1. Only the lesson's `body` string counts: not its title, source_ids or any text outside the course JSON.
2. A style label at the start of a line is removed: "Key idea: ", "Try first: ", "Check: ", "Tip: ", "Example: " or
   "> ", exactly so (case and the space after it included), after any spaces or tabs that open the line. One label
   per line; the same words anywhere else count.
3. Every source marker is replaced by a space: "[S1]", and a list such as "[S1, S2]".
4. The rest is split on whitespace (Python's str.split()). A piece is a word if it holds at least one letter, digit
   or underscore (a regex \\w, Unicode-aware); a piece with none, such as "-", "—", "=" or "+", is not.
So "well-known", "e.g.", "3.14", "don't" and "x^2" are one word each; "$a + b$" is two ("$a" and "b$"), and
"\\frac{1}{2}" one.

The multiplier, by the prompt's instructions (the text before its learner request and sources):
- A word target: the instructions say both "Write exactly 3 connected lessons, each about N words long" and "Aim for
  about N words per lesson", with the same N. The multiplier is 1.0 when the three lessons' mean count is within
  `tolerance` words of N (N - 20 to N + 20 by default, ends included), else 0.0; with a `ramp` of R words it falls
  linearly from 1.0 to 0.0 over the R words past the tolerance instead.
- Otherwise a word limit: at or under it the multiplier is 1.0; above it, it falls linearly to 0.0 at `cutoff` words
  and stays 0.0 past it. The limit is read from "Each lesson body is plain text of at most 200 words", else
  DEFAULT_LIMIT. Which count it is taken on is `aggregate`: "longest", the longest lesson's, or "mean", the three
  lessons' mean. A prompt with only one of the target lines, or two different targets, gets this rule too.
"""
from __future__ import annotations

import json
import re

DEFAULT_LIMIT = 200
DEFAULT_CUTOFF = 300
DEFAULT_TOLERANCE = 20
DEFAULT_RAMP = 0
LABELS = ("Key idea: ", "Try first: ", "Check: ", "Tip: ", "Example: ", "> ")
AGGREGATES = ("longest", "mean")

_LABEL = re.compile(r"^[ \t]*(?:" + "|".join(re.escape(label) for label in LABELS) + ")", re.M)
_MARKER = re.compile(r"\[S\d+(?:\s*,\s*S\d+)*\]")
_WORD = re.compile(r"\w")
# The limit as the prompt's instructions state it, searched only before the learner request and the sources, which
# are data a prompt quotes, not its instructions.
_LIMIT = re.compile(r"each lesson body is (?:plain text )?of at most (\d+) words", re.I)
_TARGETS = (re.compile(r"Write exactly 3 connected lessons, each about (\d+) words long", re.I),
            re.compile(r"Aim for about (\d+) words per lesson", re.I))
_DATA = re.compile(r"<learner_request>|<source[ >]")
_LENIENT = json.JSONDecoder(strict=False)


def count_words(body: str) -> int:
    """The words in one lesson body, as the module docstring defines them."""
    text = _MARKER.sub(" ", _LABEL.sub("", body))
    return sum(1 for piece in text.split() if _WORD.search(piece))


def lesson_word_counts(text: str) -> list[int] | None:
    """Each lesson's body word count, in order, from the answer's first JSON object read as the format check reads
    it (from its first '{', control characters allowed in strings, text after it ignored); None if there is no such
    object with a 'lessons' list of objects with a string body."""
    start = text.find("{")
    if start < 0:
        return None
    try:
        course, _ = _LENIENT.raw_decode(text, start)
    except (ValueError, RecursionError):
        return None
    lessons = course.get("lessons") if isinstance(course, dict) else None
    if not isinstance(lessons, list) or not lessons:
        return None
    if not all(isinstance(lesson, dict) and isinstance(lesson.get("body"), str) for lesson in lessons):
        return None
    return [count_words(lesson["body"]) for lesson in lessons]


def _instructions(prompt: str) -> str:
    match = _DATA.search(prompt)
    return prompt[:match.start()] if match else prompt


def word_target(prompt: str) -> int | None:
    """The per-lesson word target the prompt's instructions state: both target lines, each with one number, the same
    in both and above 0; else None (the limit rule applies)."""
    head = _instructions(prompt)
    found = [{int(n) for n in line.findall(head)} for line in _TARGETS]
    if all(len(numbers) == 1 for numbers in found) and found[0] == found[1] and (target := found[0].pop()) > 0:
        return target
    return None


def word_limit(prompt: str, default: int = DEFAULT_LIMIT, cutoff: int = DEFAULT_CUTOFF) -> int:
    """The per-lesson word limit the prompt's instructions state, if they state exactly one and it is under `cutoff`;
    else `default`."""
    found = {int(n) for n in _LIMIT.findall(_instructions(prompt))}
    if len(found) == 1 and 0 < (limit := found.pop()) < cutoff:
        return limit
    return default


def multiplier(words: float, limit: int = DEFAULT_LIMIT, cutoff: int = DEFAULT_CUTOFF) -> float:
    """1.0 at or under `limit` words, falling linearly to 0.0 at `cutoff`, 0.0 past it."""
    if not 0 < limit < cutoff:
        raise ValueError(f"need 0 < limit < cutoff, got limit {limit} and cutoff {cutoff}")
    if words <= limit:
        return 1.0
    if words >= cutoff:
        return 0.0
    return (cutoff - words) / (cutoff - limit)


def target_multiplier(words: float, target: int, tolerance: int = DEFAULT_TOLERANCE, ramp: int = DEFAULT_RAMP) -> float:
    """1.0 within `tolerance` words of `target` (ends included); past it 0.0, or with a `ramp`, falling linearly to
    0.0 over `ramp` more words."""
    if target <= 0 or tolerance < 0 or ramp < 0:
        raise ValueError(f"need target > 0 and tolerance, ramp >= 0, got {target}, {tolerance}, {ramp}")
    over = abs(words - target) - tolerance
    if over <= 0:
        return 1.0
    if over >= ramp:
        return 0.0
    return (ramp - over) / ramp


def course_multiplier(text: str, prompt: str, cutoff: int = DEFAULT_CUTOFF, aggregate: str = "longest",
                      default_limit: int = DEFAULT_LIMIT, tolerance: int = DEFAULT_TOLERANCE,
                      ramp: int = DEFAULT_RAMP) -> tuple[float, list[int] | None, int]:
    """(multiplier, each lesson's word count, the target or limit used) for an answer to `prompt`; word_target(prompt)
    tells which. A target is always met on the lessons' mean. An answer with no lessons to count gets 0.0 (the
    validator's format check scores it 0 before this is reached)."""
    if aggregate not in AGGREGATES:
        raise ValueError(f"aggregate must be one of {AGGREGATES}, got {aggregate!r}")
    target = word_target(prompt)
    words_rule = target if target is not None else word_limit(prompt, default_limit, cutoff)
    counts = lesson_word_counts(text)
    if counts is None:
        return 0.0, None, words_rule
    mean = sum(counts) / len(counts)
    if target is not None:
        return target_multiplier(mean, target, tolerance, ramp), counts, target
    words = max(counts) if aggregate == "longest" else mean
    return multiplier(words, words_rule, cutoff), counts, words_rule
