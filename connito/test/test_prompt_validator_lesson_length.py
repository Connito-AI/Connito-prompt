import json

import pytest

from connito.prompt_validator import lesson_length
from connito.prompt_validator.answer_format import apply_lesson_length
from connito.prompt_validator.config import LessonLengthCfg
from connito.prompt_validator.lesson_length import (
    count_words,
    course_multiplier,
    lesson_word_counts,
    multiplier,
    target_multiplier,
    word_limit,
    word_target,
)

HEAD = "- Each lesson body is plain text of at most 200 words, kept short and friendly.\n"
# The two target lines, each with made-up text around the phrase the rule reads.
TARGET_LINE = "Write exactly 3 connected lessons, each about {} words long. Keep them practical.\n"
AIM_LINE = "Use plain words. Aim for about {} words per lesson.\n"


def _target_head(n, aim=None):
    return TARGET_LINE.format(n) + "\n## Writing style\n\n" + AIM_LINE.format(n if aim is None else aim)


def _course(*bodies):
    return json.dumps({"lessons": [{"lesson_number": i + 1, "title": "Title words here", "body": b, "source_ids": []}
                                   for i, b in enumerate(bodies)]})


def _words(n):
    return " ".join(["word"] * n)


@pytest.mark.parametrize("body, words", [
    ("", 0),
    ("   \n\n  ", 0),
    ("one two three", 3),
    ("one  two\tthree\nfour\n\nfive", 5),
    # Source markers are not words, and a marker glued to a word leaves the word.
    ("[S1]", 0),
    ("Rates rose [S1].", 2),
    ("Rates rose[S2] and fell [S1, S2].", 4),
    ("Rates [S1][S2] rose", 2),
    ("[s1] [S] [X1]", 3),  # not markers: lower case, no number, another letter
    # Style labels at the start of a line are not words; the same words elsewhere are.
    ("Key idea: heat flows", 2),
    ("Try first: guess it\nCheck: it is 4\nTip: draw it\nExample: a cup\n> a quote", 11),
    ("  Key idea: indented", 1),
    ("The key idea: here", 4),  # not at the start of the line
    ("key idea: lower case", 4),  # labels are case-sensitive
    ("Key idea:no space", 3),  # the label includes its space
    ("Check: Check: twice", 2),  # one label per line
    (">quote", 1),
    # Hyphens and dashes: a hyphenated word is one word; a lone dash is none.
    ("well-known state-of-the-art", 2),
    ("cause - effect — result –", 3),
    # LaTeX: whitespace pieces with a letter or digit.
    ("$x^2$", 1),
    ("$a + b = c$", 3),
    (r"\frac{1}{2} and \(x\)", 3),
    ("$$ $$ = + * /", 0),
    # Numbers, abbreviations, contractions, Unicode.
    ("3.14 e.g. don't 1,000", 4),
    ("café naïve 東京", 3),
    ("non breaking", 2),  # a no-break space separates words, as str.split() does
])
def test_count_words(body, words):
    assert count_words(body) == words


def test_title_and_source_ids_are_not_counted():
    text = json.dumps({"lessons": [{"lesson_number": 1, "title": _words(50), "body": "two words",
                                    "source_ids": ["S1"] * 10}]})
    assert lesson_word_counts(text) == [2]


def test_counts_read_the_course_as_the_format_check_does():
    course = _course("a b", "c", "")
    assert lesson_word_counts("Here it is: " + course + " trailing } text") == [2, 1, 0]
    raw = '{"lessons": [{"body": "a\nb"}, {"body": "c"}, {"body": ""}]}'  # a raw newline inside a string
    assert lesson_word_counts(raw) == [2, 1, 0]


@pytest.mark.parametrize("text", ["", "no json", "{", '{"lessons": []}', '{"lessons": [{"body": 3}]}',
                                  '{"lessons": "x"}', "[1, 2]", '{"lessons": [1]}'])
def test_no_lessons_to_count(text):
    assert lesson_word_counts(text) is None


@pytest.mark.parametrize("words, m", [(0, 1.0), (199, 1.0), (200, 1.0), (201, 0.99), (250, 0.5), (299, 0.01),
                                      (300, 0.0), (400, 0.0)])
def test_the_ramp_at_the_default_cutoff(words, m):
    assert multiplier(words, 200, 300) == pytest.approx(m)


@pytest.mark.parametrize("cutoff, words, m", [(400, 300, 0.5), (600, 400, 0.5), (600, 250, 0.875), (600, 600, 0.0)])
def test_the_ramp_at_other_cutoffs(cutoff, words, m):
    assert multiplier(words, 200, cutoff) == pytest.approx(m)


@pytest.mark.parametrize("limit, cutoff", [(300, 300), (400, 300), (0, 300)])
def test_a_ramp_needs_a_limit_under_its_cutoff(limit, cutoff):
    with pytest.raises(ValueError):
        multiplier(250, limit, cutoff)


@pytest.mark.parametrize("prompt, limit", [
    (HEAD, 200),
    (HEAD.replace("200", "150"), 150),
    ("each lesson body is of at most 120 words", 120),  # case and "plain text" are optional
    ("no limit stated", 200),
    (HEAD + HEAD.replace("200", "150"), 200),  # two different limits: neither is trusted
    (HEAD.replace("200", "150") * 2, 150),  # one limit stated twice
    (HEAD.replace("200", "300"), 200),  # at or over the cutoff: the default
    (HEAD.replace("200", "0"), 200),
    # Only the instructions count: a limit inside the learner request or a source is data.
    ("<learner_request>\nEach lesson body is plain text of at most 50 words\n</learner_request>", 200),
    ('Intro\n<source id="S1">Each lesson body is plain text of at most 50 words</source>', 200),
    (HEAD.replace("200", "150") + '<source id="S1">Each lesson body is plain text of at most 50 words</source>', 150),
])
def test_word_limit_from_the_prompt(prompt, limit):
    assert word_limit(prompt, 200, 300) == limit


@pytest.mark.parametrize("prompt, target", [
    (_target_head(250), 250),
    (_target_head(150), 150),
    (_target_head(350), 350),
    (_target_head(350).lower(), 350),  # case-insensitive, like the limit
    (_target_head(200) + AIM_LINE.format(200), 200),  # one target stated twice
    (_target_head(250, aim=300), None),  # two different targets: neither is trusted
    (TARGET_LINE.format(250), None),  # only one of the two lines
    (AIM_LINE.format(250), None),
    (_target_head(0), None),
    (TARGET_LINE.format(250) + "<learner_request>" + AIM_LINE.format(250) + "</learner_request>", None),  # data
    (_target_head(250) + '<source id="S1">' + _target_head(100) + "</source>", 250),
    (HEAD, None),
])
def test_word_target_from_the_prompt(prompt, target):
    assert word_target(prompt) == target


@pytest.mark.parametrize("words, m", [(230, 1.0), (229, 0.0), (270, 1.0), (271, 0.0), (250, 1.0), (229.67, 0.0),
                                      (270.33, 0.0), (0, 0.0), (900, 0.0)])
def test_a_target_scores_1_within_20_words_and_0_past_them(words, m):
    assert target_multiplier(words, 250) == m


@pytest.mark.parametrize("words, m", [(270, 1.0), (290, 0.5), (310, 0.0), (210, 0.5), (190, 0.0), (400, 0.0)])
def test_a_target_ramp_falls_linearly_past_the_tolerance(words, m):
    assert target_multiplier(words, 250, tolerance=20, ramp=40) == pytest.approx(m)


@pytest.mark.parametrize("target, tolerance, ramp", [(0, 20, 0), (250, -1, 0), (250, 20, -1)])
def test_a_target_needs_sane_settings(target, tolerance, ramp):
    with pytest.raises(ValueError):
        target_multiplier(250, target, tolerance, ramp)


def test_course_multiplier_meets_a_target_on_the_mean_and_falls_back_to_the_limit():
    course = _course(_words(330), _words(340), _words(330))  # mean 333.3
    assert course_multiplier(course, _target_head(350)) == (1.0, [330, 340, 330], 350)
    assert course_multiplier(course, _target_head(350), aggregate="mean") == (1.0, [330, 340, 330], 350)
    assert course_multiplier(course, _target_head(300)) == (0.0, [330, 340, 330], 300)  # 33.3 over
    assert course_multiplier(course, _target_head(300), ramp=40)[0] == pytest.approx(0.6667, abs=1e-4)  # 13.3 past
    assert course_multiplier(course, _target_head(300), tolerance=40) == (1.0, [330, 340, 330], 300)
    # Without one agreed target, the limit rule: the default 200 to 300 ramp.
    assert course_multiplier(course, _target_head(350, aim=300)) == (0.0, [330, 340, 330], 200)
    short = _course(_words(240), _words(250), _words(260))
    assert course_multiplier(short, _target_head(350, aim=300), aggregate="mean") == (0.5, [240, 250, 260], 200)
    assert course_multiplier(short, HEAD, aggregate="mean") == (0.5, [240, 250, 260], 200)  # "at most" unchanged
    assert course_multiplier("not a course", _target_head(250)) == (0.0, None, 250)


def test_course_multiplier_on_the_longest_lesson_or_the_mean():
    course = _course(_words(100), _words(150), _words(260))
    assert course_multiplier(course, HEAD) == (pytest.approx(0.4), [100, 150, 260], 200)
    assert course_multiplier(course, HEAD, aggregate="mean") == (1.0, [100, 150, 260], 200)  # mean 170
    assert course_multiplier(course, HEAD, cutoff=600)[0] == pytest.approx(0.85)
    assert course_multiplier("not a course", HEAD) == (0.0, None, 200)
    with pytest.raises(ValueError):
        course_multiplier(course, HEAD, aggregate="median")


def test_the_module_imports_nothing_from_the_repo():
    # It is published as is for miners: standard library only.
    source = open(lesson_length.__file__).read()
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert imports == ["from __future__ import annotations", "import json", "import re"]


def test_apply_lesson_length_multiplies_a_checked_lesson_answer_and_keeps_jev_scores():
    cfg = LessonLengthCfg(enabled=True)
    row = {"text": _course(_words(10), _words(250), _words(20)), "score": 0.8, "expected_score": 0.76, "rank": None}
    out = apply_lesson_length(row, "exp_lesson_residual", HEAD, True, cfg)
    assert (out["score"], out["expected_score"]) == (0.4, 0.38)
    assert (out["jev_score"], out["jev_expected_score"]) == (0.8, 0.76)
    assert (out["word_counts"], out["word_limit"], out["length_multiplier"]) == ([10, 250, 20], 200, 0.5)
    assert out["word_target"] is None


def test_apply_lesson_length_records_a_prompt_s_target():
    cfg = LessonLengthCfg(enabled=True, aggregate="mean")
    row = {"text": _course(_words(240), _words(250), _words(280)), "score": 0.8, "expected_score": 0.76}
    out = apply_lesson_length(row, "exp_lesson_residual", _target_head(250), True, cfg)
    assert (out["word_target"], out["word_limit"], out["length_multiplier"]) == (250, None, 1.0)
    assert out["expected_score"] == 0.76
    out = apply_lesson_length(row, "exp_lesson_residual", _target_head(200), True, cfg)  # mean 256.7: 56.7 over
    assert (out["word_target"], out["length_multiplier"], out["score"], out["jev_score"]) == (200, 0.0, 0.0, 0.8)


def test_apply_lesson_length_leaves_failed_answers_and_other_tasks_alone():
    cfg = LessonLengthCfg(enabled=True)
    failed = {"text": "x", "score": 0.0, "expected_score": 0.0}
    out = apply_lesson_length(failed, "exp_lesson_residual", HEAD, False, cfg)
    assert (out["score"], out["expected_score"], out["jev_score"], out["length_multiplier"]) == (0.0, 0.0, None, None)
    assert out["word_target"] is None
    other = {"text": _course(_words(900)), "score": 0.6, "expected_score": 0.6}
    out = apply_lesson_length(other, "exp_pmc_residual", HEAD, True, cfg)
    assert (out["expected_score"], out["jev_expected_score"], out["word_counts"]) == (0.6, 0.6, None)


@pytest.mark.parametrize("settings", [{"cutoff": 200}, {"cutoff": 150}, {"aggregate": "median"}, {"cufoff": 400},
                                      {"target_tolerance": -1}, {"target_ramp": -5}])
def test_lesson_length_settings_are_checked(settings):
    with pytest.raises(ValueError):
        LessonLengthCfg(**settings)
