"""Dependency-surface tests: declared extras must cover the features they document.

These test ``pyproject.toml`` itself (the declared dependency surface), not the
installed environment, so they run in CI without any extras installed.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_dev_extra_includes_capture_backend():
    """``uv sync --extra dev`` must leave the real (mss) capture backend usable.

    The README's Development Setup says ``uv sync --extra dev`` is the local
    development setup, and ``mss`` is the *default* capture backend. If the dev
    extra omits mss, the default backend is unusable immediately after the
    documented setup step (the CLI reports ``the 'mss' backend is not
    installed``). Pillow is already in dev for the same reason; mss was the
    missing half.
    """
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    dev = data["project"]["optional-dependencies"]["dev"]
    assert any(dep.lower().startswith("mss") for dep in dev), (
        "mss is missing from the 'dev' extra, but it is the default capture "
        f"backend; dev={dev!r}"
    )
