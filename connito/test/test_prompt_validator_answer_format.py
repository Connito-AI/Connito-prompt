import json

import pytest

from connito.prompt_validator import answer_format
from connito.prompt_validator.answer_format import FORMAT_RULES, format_ok, lessons_ok, prompt_sources

PROMPT = 'Teach it.\n<source id="S1" title="a">one</source>\n<source id="S2" title="b">two</source>'
LESSON = {"lesson_number": 1, "title": "t", "body": "Key idea: b [S1]", "source_ids": ["S1"]}


def course(*lessons, **extra) -> str:
    return json.dumps({"lessons": list(lessons) if lessons else [LESSON] * 3, **extra})


COURSE = course()


@pytest.mark.parametrize("text", [
    COURSE,
    COURSE + "}",  # a stray closing brace after the course
    COURSE + "\n\nI hope this course helps.",
    "Here is the course:\n" + COURSE,
    f"```json\n{COURSE}\n```",
    COURSE.replace("Key idea: b", "Key idea:\nb"),  # a raw newline inside a string
    course(takeaway="x", source_ids=["S9"]),  # other keys beside lessons: only the lessons' citations count
    course(LESSON, {**LESSON, "source_ids": [], "body": "no citation"}, {**LESSON, "source_ids": ["S1", "S2"]}),
    course(LESSON, LESSON, {**LESSON, "lesson_number": "3", "extra": 1}),  # lesson_number and title: present
])
def test_a_course_of_three_lessons_citing_the_prompt_s_sources_passes(text):
    assert lessons_ok(text, PROMPT)


@pytest.mark.parametrize("text", [
    "",
    "Lesson 1: rivers",
    course(LESSON, LESSON),
    course(LESSON, LESSON, LESSON, LESSON),
    course(LESSON, LESSON, "lesson three"),
    json.dumps({"lessons": {"1": LESSON, "2": LESSON, "3": LESSON}}),
    json.dumps({"course": [LESSON] * 3}),
    COURSE.replace('"title": "t"', '"title": "a "quoted" t"'),  # broken quoting
    COURSE[:-40],  # cut off at the token limit
    '{"lessons": [' + "1" * 5000 + "]}",  # a number too long for int(): a ValueError, not a JSONDecodeError
])
def test_anything_but_a_course_of_three_lessons_fails(text):
    assert not lessons_ok(text, PROMPT)


@pytest.mark.parametrize("field", ["lesson_number", "title", "body", "source_ids"])
def test_a_lesson_without_one_of_its_fields_fails(field):
    lesson = {k: v for k, v in LESSON.items() if k != field}
    assert not lessons_ok(course(LESSON, LESSON, lesson), PROMPT)


@pytest.mark.parametrize("lesson", [
    {**LESSON, "body": ["Key idea: b"]},
    {**LESSON, "body": None},
    {**LESSON, "source_ids": "S1"},
    {**LESSON, "source_ids": None},
])
def test_a_body_that_is_not_a_string_or_source_ids_that_are_not_a_list_fail(lesson):
    assert not lessons_ok(course(LESSON, LESSON, lesson), PROMPT)


@pytest.mark.parametrize("lesson", [
    {**LESSON, "source_ids": ["S3"]},  # no such source in the prompt
    {**LESSON, "source_ids": ["S1", "S3"]},
    {**LESSON, "source_ids": [1]},
    {**LESSON, "source_ids": ["s1"]},
    {**LESSON, "body": "Key idea: b [S3]"},  # inline
    {**LESSON, "title": "From [S4]"},
])
def test_a_citation_of_a_source_the_prompt_does_not_have_fails(lesson):
    assert not lessons_ok(course(LESSON, LESSON, lesson), PROMPT)


def test_a_prompt_s_sources_are_its_source_tags():
    assert prompt_sources(PROMPT) == {"S1", "S2"}
    # With no source tags, a course passes only with nothing cited.
    uncited = {**LESSON, "body": "b", "source_ids": []}
    assert lessons_ok(course(uncited, uncited, uncited), "Teach it.")
    assert not lessons_ok(COURSE, "Teach it. Cite [S1].")


def nested(depth: int) -> str:
    return '{"lessons": ' + "[" * depth + "]" * depth + "}"


@pytest.mark.parametrize("depth", [500, 1_000, 3_000, 10_000, 100_000])
def test_a_deeply_nested_answer_fails_without_raising(depth):
    # Deep enough (from about 10,000 levels on Python 3.12), raw_decode raises RecursionError, not ValueError.
    assert not lessons_ok(nested(depth), PROMPT)
    assert not format_ok("exp_lesson_residual", nested(depth), PROMPT)


def test_a_check_that_raises_fails_the_answer(monkeypatch):
    def broken(text, prompt):
        raise TypeError("unexpected")

    monkeypatch.setitem(answer_format._CHECKS, "exp_lesson_residual", broken)
    assert not format_ok("exp_lesson_residual", COURSE, PROMPT)
    assert format_ok("exp_pmc_residual", COURSE, PROMPT)


def test_only_a_task_with_a_check_is_checked():
    assert not format_ok("exp_lesson_residual", "not a course", PROMPT)
    assert format_ok("exp_lesson_residual", COURSE, PROMPT)
    assert format_ok("exp_pmc_residual", "not a course", PROMPT)
    assert format_ok(None, "", "")


def test_every_check_has_its_rule_in_the_rules_text():
    assert answer_format._CHECKS.keys() == FORMAT_RULES.keys() == {"exp_lesson_residual"}
    rule = FORMAT_RULES["exp_lesson_residual"]
    assert "exactly 3" in rule and all(field in rule for field in answer_format.LESSON_FIELDS)
    assert '<source id="Sn">' in rule and "[Sn]" in rule
