"""Bare-venv import contract for the action layer (Phase 3, step 8 DoD).

The plan's Definition of Done explicitly calls out a regression test that the
``ai_game_agent.actions`` package — and, specifically, the *optional*
``windows`` adapter — remains importable when pynput is absent. This guards
against the common "convenient" regression of a top-level
``import pynput`` (or ``from .windows import ...`` in ``actions/__init__.py``)
that would turn a bare-venv import into a ``ModuleNotFoundError``.

The shared dev venv is expected to **not** have pynput installed (see
``pyproject.toml`` comment: pynput is Windows-only; the Linux/macOS side of
the shared venv must not prune it). If pynput *is* present in this venv, the
meta-test below detects the environment drift and tells the operator to run
``uv pip uninstall pynput`` before re-running the suite.

These tests make no assertion about :class:`WindowsBackend` behavior — that
is the job of ``tests/test_windows_backend.py`` (driven through fakes).
Here we only assert the import boundary.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import types
from pathlib import Path

import pytest


class _PynputImportBlocker:
    """A ``sys.meta_path`` hook that fails any ``pynput`` import.

    Used to simulate a bare venv in a test *regardless of what's actually
    installed*: if pynput is already imported (e.g. by a sibling test), we
    first remove it from ``sys.modules`` so the block is exercised for real.
    """

    def find_spec(
        self, name: str, path: object | None = None, target: types.ModuleType | None = None
    ) -> importlib.abc.Spec | None:
        if name == "pynput" or name.startswith("pynput."):
            raise ImportError("pynput is blocked by test_actions_bare (bare-venv contract)")
        return None


@contextlib.contextmanager
def _with_pynput_blocked():
    """Context manager: ensure pynput is unimportable inside the ``with``."""
    saved_modules = {
        k: v for k, v in list(sys.modules.items()) if k == "pynput" or k.startswith("pynput.")
    }
    blocker = _PynputImportBlocker()
    sys.meta_path.insert(0, blocker)
    for k in list(sys.modules):
        if k == "pynput" or k.startswith("pynput."):
            del sys.modules[k]
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        # Restore whatever was there (or nothing) — do NOT restore pynput if
        # the environment legitimately doesn't have it; the test that needs
        # pynput should ``importorskip`` on its own.
        for k in list(sys.modules):
            if k == "pynput" or k.startswith("pynput."):
                del sys.modules[k]
        sys.modules.update(saved_modules)


# --- The bare-venv invariant -------------------------------------------------


def test_windows_module_imports_without_pynput() -> None:
    """``import ai_game_agent.actions.windows`` must succeed on a bare venv.

    A top-level ``import pynput`` in ``windows.py`` (or a transitive
    ``from .windows import ...`` in ``actions/__init__.py``) breaks this.
    The whole point of the lazy-import pattern is that the *module* imports
    fine and only the *pynput* import happens when the backend is actually
    constructed on a machine that has the ``input`` extra.
    """
    with _with_pynput_blocked():
        # A fresh import (not the cached one) — force the module to re-import
        # so a top-level pynput import, if any, would run and fail.
        mod_name = "ai_game_agent.actions.windows"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        # This raises ImportError (wrapped by our blocker) if the module
        # imports pynput at top level.
        importlib.import_module(mod_name)


def test_actions_package_imports_without_pynput() -> None:
    """The package re-export surface must be importable on a bare venv.

    This is the user-facing contract: ``from ai_game_agent.actions import
    Action, Executor, create_backend`` works everywhere, including a
    headless CI runner with no pynput.
    """
    with _with_pynput_blocked():
        import ai_game_agent.actions as actions_pkg  # noqa: F401

        for attr in ("Action", "ActionKind", "Executor", "create_backend",
                     "InputBackendError", "SafetyGuard"):
            assert hasattr(actions_pkg, attr), (
                f"ai_game_agent.actions lost public export {attr!r}"
            )


def test_create_backend_windows_still_fails_cleanly_without_pynput() -> None:
    """Even with pynput blocked, selecting the ``windows`` backend must
    produce the *domain* error (naming the ``input`` extra), not a raw
    ``ImportError``. This is the existing contract from step 2, re-asserted
    here under the import-blocker to catch a regression where the factory
    path changed to ``import pynput`` directly.
    """
    with _with_pynput_blocked():
        from ai_game_agent.actions.base import InputBackendError, create_backend
        from ai_game_agent.config import InputConfig

        with pytest.raises(InputBackendError, match="input.*extra|pynput"):
            create_backend(InputConfig(backend="windows"))


# --- Meta: environment drift detection ---------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "cross-OS venv rule (shared-mount-cross-os-venv / §15): pynput is "
        "Windows-only. On the Windows box the 'input' extra is installed in "
        "the same venv, so pynput IS present there — the bare-venv contract "
        "is inverted and does not apply. The test still enforces the bare "
        "contract on Linux/CI, where a bare venv is the expected state."
    ),
)
def test_shared_venv_does_not_have_pynput_installed() -> None:
    """The shared dev venv is *expected* to be bare of pynput (the ``input``
    extra is installed only on the Windows box, per the cross-OS venv rule).

    If this test fails, the environment drifted: someone ran
    ``uv sync --extra input`` in the shared venv, or ``pip install pynput``
    on the Linux/macOS side. Re-run the suite after
    ``uv pip uninstall pynput`` to restore the bare-venv contract.
    """
    # We do NOT use importorskip here: this is a *contract* on the venv, not
    # an optional feature. A bare venv is the expected state.
    import importlib.util

    spec = importlib.util.find_spec("pynput")
    assert spec is None, (
        "pynput is installed in the shared venv; the Phase 3 DoD requires a "
        "bare venv here (pynput is Windows-only; install the 'input' extra "
        "on the Windows machine only). Run 'uv pip uninstall pynput' and "
        "re-run the suite."
    )


def test_this_test_file_never_imports_pynput() -> None:
    """A meta-guard on the test files themselves: the bare-venv contract is
    only meaningful if the *tests* don't pull pynput in at module load.

    A *top-level* ``import pynput`` / ``from pynput import ...`` in
    ``test_windows_backend.py`` would break the headless guarantee, so we
    scan the source for one. A lazy import inside a function body is fine —
    that's the pattern the rest of the codebase uses — so we only flag
    lines at column 0.
    """
    src_path = Path(__file__).parent / "test_windows_backend.py"
    for line in src_path.read_text(encoding="utf-8").splitlines():
        # Column-0 import statements only (an indented one is a lazy
        # import inside a function, which is allowed).
        if line.startswith("import pynput") or line.startswith("from pynput"):
            pytest.fail(
                f"tests/test_windows_backend.py has a top-level pynput import: {line!r}"
            )
        if line.startswith("import ai_game_agent.actions.windows"):
            # A top-level import of the windows module is fine *if* that
            # module is itself bare-safe (which test_windows_module_imports_
            # without_pynput asserts); this guard only forbids a top-level
            # pynput import, so we do not fail here. Kept as a no-op comment.
            pass
