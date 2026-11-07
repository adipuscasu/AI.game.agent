"""EH1: the CLI implementation lives in ``ai_game_agent.cli`` (not ``__main__``).

Red step: ``ai_game_agent.cli`` does not exist yet, so this module fails at
import time (ImportError / AttributeError). Once the 1274-line ``__main__.py``
body is moved into ``cli.py`` and ``__main__.py`` becomes a thin re-export
shim, every assertion below holds and the rest of the suite (which still
imports from ``ai_game_agent.__main__``) keeps passing unchanged.

Why this module exists at all (and why we don't just delete ``__main__.py``):

* ``pyproject.toml`` pins the console script to
  ``ai_game_agent.__main__:main``.
* The patching tests (``test_cli``, ``test_run_cli``,
  ``test_stop_wiring``) patch *module-global* names on
  ``ai_game_agent.cli`` — e.g. ``monkeypatch.setattr(cli, "AgentLoop",
  factory)`` where ``cli`` is ``import ai_game_agent.cli as cli``. A
  patch must target the module where the production functions resolve
  their globals: the CLI code is defined in ``cli``, so a patch on the
  ``ai_game_agent.__main__`` re-export would silently do nothing. The
  remaining test files (``test_objects``, ``test_ui_zones``,
  ``test_replay``) only *import* names from ``ai_game_agent.__main__``
  (no patching), so the re-export's initial shared references are
  sufficient for them.
* ``tests/test_cli.py::_repo_root`` derives the repo root from
  ``Path(cli.__file__).parents[2]`` — the shim and the real module must
  live in the same package directory.

The contract this test locks in:

* ``ai_game_agent.cli`` exists and is a *module*, not a class.
* The shim's re-exported names are the *same function object* as in
  ``cli`` (initial shared reference only — see the namespace notes below).
* ``cli.main`` and ``__main__.main`` are the same object (same story for
  the other names the suite patches), so both entry paths hit the same
  code.
* The two files live in the same directory, so the ``parents[2]`` repo
  root derivation is stable for both.

**Namespace independence (the monkeypatch contract):**

The shim and ``cli`` are two *separate* module objects with independent
attribute namespaces. The re-export in ``__main__.py`` binds each name
in the shim's namespace to the same object ``cli`` bound, but that is an
*initial* shared reference, not a shared binding. Concretely:

* ``monkeypatch.setattr(main_mod, "AgentLoop", sentinel)`` replaces the
  binding in ``main_mod``'s namespace only. ``cli.AgentLoop`` continues
  to be the real object, and every function *defined in* ``cli``
  (including ``main``) resolves ``AgentLoop`` via ``cli``'s globals —
  so the sentinel is invisible to production code.

* Therefore the tests must patch ``ai_game_agent.cli`` (the module that
  *defines* the CLI code) for the patch to have any effect. The three
  patching test files (``test_cli``, ``test_run_cli``,
  ``test_stop_wiring``) alias ``ai_game_agent.cli`` precisely because of
  this rule. A patch on the shim would silently do nothing.

* ``test_namespace_independence_proves_tests_must_patch_cli`` below
  locks this contract: it patches the shim, proves ``cli`` is
  untouched, and asserts ``main.__module__ == "ai_game_agent.cli"`` —
  i.e. the code production runs is defined in ``cli``, not the shim.
"""

from __future__ import annotations

from pathlib import Path

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
    # The shim re-exports ``main`` from ``cli``. This is an *initial*
    # shared reference only — the two modules have independent
    # namespaces, so a monkeypatch on ``main_mod.main`` does not change
    # ``cli.main`` (and vice versa). See the module docstring's
    # "Namespace independence" section and the regression test below.
    assert main is main_mod.main


def test_patched_names_are_shared() -> None:
    # The exact names the suite monkeypatches (test_stop_wiring, test_run_cli,
    # test_cli) must resolve to the *same object* on both modules — an
    # initial shared reference. The namespace-independence regression test
    # below demonstrates that "same object" does NOT mean "patching one
    # module patches the other": CPython resolves each function's globals
    # against its *defining* module, so patches must target the defining
    # module (``cli``), not the shim.
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
    assert Path(cli.__file__).resolve().parent == Path(main_mod.__file__).resolve().parent
    # Both resolve to the same repo root under the parents[2] convention.
    assert Path(cli.__file__).resolve().parents[2] == \
        Path(main_mod.__file__).resolve().parents[2]


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


def test_namespace_independence_proves_tests_must_patch_cli(monkeypatch) -> None:
    """Prove that the shim and ``cli`` have independent namespaces.

    The regression it protects: a future change that makes
    ``ai_game_agent.__main__`` a *literal alias* for ``ai_game_agent.cli``
    (e.g. ``from ai_game_agent import cli as __main__`` in the package
    ``__init__``, or the shim doing ``sys.modules[__name__] = cli``) would
    make patches on the shim reach ``cli`` and vice versa. That would
    silently invalidate the patching discipline the suite relies on —
    the three patching test files would "pass" for the wrong reason, and
    any test that patches the shim *instead of* ``cli`` would suddenly
    work, hiding the namespace-independence rule this file documents.

    The test also asserts ``main.__module__ == "ai_game_agent.cli"`` —
    i.e. the code production runs is defined in ``cli``, not the shim.
    That is the load-bearing fact for the monkeypatch contract: CPython
    resolves a function's globals against its *defining* module, so a
    patch only has effect when it lands in the defining module's
    namespace. If ``main`` were ever redefined in the shim, its globals
    would resolve through the shim, and the patching tests (which patch
    ``cli``) would silently stop working.

    How it works:

    * Patch ``main_mod.AgentLoop`` (the shim's binding) with a sentinel.
      ``main_mod.AgentLoop`` is the sentinel; ``cli.AgentLoop`` is still
      the real object. This is direct, observable evidence that the
      two modules have independent namespaces.
    * Assert ``main.__module__ == "ai_game_agent.cli"``: the code
      production runs is defined in ``cli``. If this ever regressed
      (``main`` defined in the shim instead), the patching tests would
      silently stop working because their patches land in ``cli``'s
      namespace, which the shim-defined ``main`` does not consult.
    * The monkeypatch fixture auto-restores the shim's binding; no
      manual teardown, no effect on ``cli`` (which was never touched).
    """
    sentinel = object()
    monkeypatch.setattr(main_mod, "AgentLoop", sentinel)
    # Patch landed in the shim's namespace.
    assert main_mod.AgentLoop is sentinel
    # The defining module is untouched: the real object is still there.
    assert cli.AgentLoop is not sentinel
    assert cli.AgentLoop is AgentLoop  # the real one, unchanged
    # The code production runs is defined in cli — the load-bearing fact
    # that makes the patching discipline (patch the defining module)
    # both necessary and sufficient.
    assert main.__module__ == "ai_game_agent.cli"
