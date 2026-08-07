"""Tests for scripts/play_bridge_cli.py's persona-selection prompt - the
confirm/re-describe/pick-by-name-instead flow, same UX as poker's own CLI.

Not exercised via games/bridge/persona_selection.py's own tests, since this is about
the terminal interaction wrapped around it (what gets printed, how a "no" answer loops
back), not the matching mechanism itself.
"""

from collections.abc import Iterator

import pytest

import scripts.play_bridge_cli as cli
from games.bridge.personas import AGGRESSIVE, BAITER, BASELINE, CONSERVATIVE


def _feed(monkeypatch: pytest.MonkeyPatch, lines: list[str]) -> None:
    responses: Iterator[str] = iter(lines)
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: next(responses))


def test_an_exact_persona_name_commits_immediately_without_confirming(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _feed(monkeypatch, ["aggressive"])
    persona = cli._prompt_persona("Persona for your partner?")
    assert persona is AGGRESSIVE
    # No "Interpreting that as" / confirmation text - an exact name needs no inference.
    assert "Interpreting" not in capsys.readouterr().out


def test_an_exact_persona_name_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    _feed(monkeypatch, ["  AGGRESSIVE  "])
    assert cli._prompt_persona("?") is AGGRESSIVE


def test_free_text_shows_the_inference_and_commits_on_yes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _feed(monkeypatch, ["a sneaky, tricky sort", "yes"])
    persona = cli._prompt_persona("Persona for your first opponent?")
    assert persona is BAITER
    out = capsys.readouterr().out
    assert "Interpreting that as: baiter" in out
    assert "sneaky" in out and "tricky" in out


def test_declining_the_inference_loops_back_to_an_exact_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feed(monkeypatch, ["a sneaky sort", "no", "baseline"])
    persona = cli._prompt_persona("?")
    assert persona is BASELINE


def test_declining_the_inference_loops_back_to_another_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feed(monkeypatch, ["a sneaky sort", "no", "actually pretty aggressive and bold", "yes"])
    persona = cli._prompt_persona("?")
    assert persona is AGGRESSIVE


def test_blank_answer_to_the_confirmation_defaults_to_yes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feed(monkeypatch, ["sneaky and tricky", ""])
    assert cli._prompt_persona("?") is BAITER


def test_text_matching_nothing_reprompts_rather_than_guessing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _feed(monkeypatch, ["plays cards I guess", "conservative"])
    persona = cli._prompt_persona("?")
    out = capsys.readouterr().out
    assert "didn't recognise" in out
    assert persona is CONSERVATIVE
