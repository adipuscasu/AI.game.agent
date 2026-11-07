"""EH1: the CLI implementation lives in ``ai_game_agent.cli`` (not ``__main__``).

Red step: ``ai_game_agent.cli`` does not exist yet, so this module fails at
import time (ImportError / AttributeError). Once the 1274-line ``__main__.py``
body is moved into ``cli.py`` and ``__main__.py`` becomes a thin re-export
shim, every assertion below holds and the rest of the suite (which still
imports from ``ai_game_agent.__main__``) keeps passing unchanged.

Why this module exists at all (and why we don't just delete ``__main__.py``):

* ``pyproject.toml`` pins the console script to
  ``ai_game_agent.__main__:main``.
* The test suite (``test_cli``, ``test_run_cli``, ``test_stop_wiring``,
  ``test_objects``, ``test_ui_zones``, ``test_replay``) patches
  *module-global* names on ``ai_game_agent.__main__`` — e.g.
  ``monkeypatch.setattr(cli, "AgentLoop", factory)`` where ``cli`` is
  ``import ai_game_agent.__main__ as cli``. That pattern only works if
  ``ai_game_agent.__main__`` re-exports the same function objects the
  production code resolves through.
* ``tests/test_cli.py::_repo_root`` derives the repo root from
  ``Path(cli.__file__).parents[2]`` — the shim and the real module must
  live in the same package directory.

The contract this test locks in:

* ``ai_game_agent.cli`` exists and is a *module*, not a class.
* ``cli.main is __main__.main`` (same function object; a patch on either
  is visible from both).
* ``cli._build_perception`` and ``__main__._build_perception`` are the
  same callable (same story for the other names the suite patches).
* The two files live in the same directory, so the ``parents[2]`` repo
  root derivation is stable for both.
"""

from __future__ import annotations

import ai_game_agent.__main__ as main_mod
import ai_game_agent.cli as cli
from ai_game_agent.cli import (
    AgentLoop,
    _build_perception,
    _resolve_template_path,
    build_parser,
    create_backend,
    load_config,
    main,
)


def test_cli_is_a_module() -> None:
    import types

    assert isinstance(cli, types.ModuleType)


def test_main_is_reexported_by_main() -> None:
    # Same function object on both modules — a monkeypatch on either is
    # visible to production code that resolves through the other.
    assert main is main_mod.main


def test_patched_names_are_shared() -> None:
    # The exact names the suite monkeypatches (test_stop_wiring, test_run_cli,
    # test_cli) must resolve to the *same* object on both modules.
    assert AgentLoop is main_mod.AgentLoop
    assert create_backend is main_mod.create_backend
    assert load_config is main_mod.load_config
    assert build_parser is main_mod.build_parser
    assert _build_perception is main_mod._build_perception
    assert _resolve_template_path is main_mod._resolve_template_path


def test_files_live_in_the_same_directory() -> None:
    # The repo-root derivation in tests/test_cli.py::_repo_root uses
    # ``Path(cli.__file__).parents[2]``. That only yields the repo root if
    # the file is at ``<repo>/src/ai_game_agent/<name>.py``. Keeping the
    # shim and the real module in the same package dir guarantees that for
    # whichever one a test imports.
    assert cli.__file__.rpartition("/")[0] == main_mod.__file__.rpartition("/")[0]
    # Both resolve to the same repo root under the parents[2] convention.
    assert __import__("pathlib").Path(cli.__file__).resolve().parents[2] == \
        __import__("pathlib").Path(main_mod.__file__).resolve().parents[2]


def test_build_parser_returns_the_same_parser_type() -> None:
    # Both modules expose an ``argparse.ArgumentParser`` factory with the
    # same subcommand set. This catches a partial move where the shim
    # accidentally shadows the real parser.
    import argparse

    p1 = main_mod.build_parser()
    p2 = cli.build_parser()
    assert isinstance(p1, argparse.ArgumentParser)
    assert isinstance(p2, argparse.ArgumentParser)
    # The subcommand set (capture/observe/act/analyze/run/replay) is
    # defined once and shared.
    subs = set()
    for action in p1._actions:
        if getattr(action, "choices", None) is not None:
            subs.update(action.choices)
    assert {"capture", "observe", "act", "analyze", "run", "replay"} <= subs
