"""Entry-point shim for the AI game agent CLI.

The CLI implementation lives in :mod:`ai_game_agent.cli` (EH1: separate the
monolithic ``__main__.py`` body into a dedicated, testable module). This thin
shim exists so that the three public entry paths keep resolving to the same
function objects defined in ``cli``:

* ``python -m ai_game_agent`` — Python runs this file as ``__main__`` and the
  ``if __name__ == "__main__"`` block below calls :func:`main`.
* the ``ai-game-agent`` console script (``ai_game_agent.__main__:main`` in
  ``pyproject.toml``) — imports this module and resolves :func:`main`.
* existing importers/tests that reach for names on ``ai_game_agent.__main__``
  (e.g. ``from ai_game_agent.__main__ import _build_perception``).

Import :mod:`ai_game_agent.cli` directly for the implementation. The names
re-exported here are the *same objects* as in ``cli`` (verified by
``tests/test_cli_module.py``), so the public surface of ``__main__`` is
unchanged.

Note on monkeypatching: patch a name on ``ai_game_agent.cli`` (the module
that *defines* the code), not on this shim — CPython resolves a function's
globals against its defining module, so patching a re-export does not affect
the real code. The test suite's patching files alias ``ai_game_agent.cli``
for exactly this reason.
"""

from __future__ import annotations

import sys

from ai_game_agent.cli import (
    AgentLoop,
    _build_perception,
    _resolve_template_path,
    build_parser,
    create_backend,
    load_config,
    main,
    save_frame,
)

__all__ = [
    "AgentLoop",
    "build_parser",
    "create_backend",
    "load_config",
    "main",
    "save_frame",
    "_build_perception",
    "_resolve_template_path",
]

if __name__ == "__main__":
    sys.exit(main())
