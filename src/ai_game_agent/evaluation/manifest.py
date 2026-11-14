"""EH4: the versioned perception-evaluation dataset manifest.

The manifest is the single source of truth for a dataset:

* which frames exist (``samples[].image``, relative to the manifest),
* their capture metadata (resolution, window mode, UI scale),
* their split (``calibration`` — safe to tune against, vs ``held_out`` —
  must never be used for calibration),
* their expected detections — an ``Observation`` contract dict, reusing the
  perception layer's own shape as the label schema,
* optional expected state-machine output (``signals`` + ``state``).

Authored-data rules: the loader is strict about shape drift (unknown and
missing keys) and reports *all* validation problems at once, because a
manifest with ten problems should not need ten load attempts.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Any

from ai_game_agent.perception.observation import Observation

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "SPLITS",
    "CaptureMetadata",
    "DatasetManifest",
    "ManifestError",
    "Sample",
    "StateExpectation",
    "load_manifest",
    "validate_manifest",
]

# Bump when the manifest shape changes in a way loaders must handle.
MANIFEST_SCHEMA_VERSION = 1

# ``calibration``: used to tune thresholds/templates. ``held_out``: reported
# separately by the evaluator and never fed back into tuning.
SPLITS: tuple[str, ...] = ("calibration", "held_out")

_SAMPLE_ID = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")

_TOP_KEYS = frozenset(
    {"schema_version", "dataset", "dataset_version", "created", "capture", "samples"}
)
_CAPTURE_KEYS = frozenset({"width", "height", "window_mode", "ui_scale"})
_SAMPLE_KEYS = frozenset({"id", "image", "split", "detections", "state"})
_STATE_KEYS = frozenset({"signals", "state", "reason"})


@dataclass(frozen=True)
class CaptureMetadata:
    """How the dataset was captured — the resolution/scaling context a
    detection result only makes sense inside."""

    width: int
    height: int
    window_mode: str
    ui_scale: float


@dataclass(frozen=True)
class StateExpectation:
    """Expected decision output for a sample: the stable signal keys the
    policy must extract, the state the machine must reach, and (optionally)
    the stable transition reason."""

    signals: Mapping[str, bool]
    state: str
    reason: str | None = None


@dataclass(frozen=True)
class Sample:
    """One labelled frame. ``image`` is a forward-slash path relative to the
    manifest's directory."""

    id: str
    image: str
    split: str
    detections: Observation
    state: StateExpectation | None = None


@dataclass(frozen=True)
class DatasetManifest:
    """A versioned, validated perception-evaluation dataset."""

    dataset: str
    dataset_version: str
    created: str
    capture: CaptureMetadata
    samples: tuple[Sample, ...]
    schema_version: int = MANIFEST_SCHEMA_VERSION


class ManifestError(ValueError):
    """A manifest that cannot be loaded. ``errors`` lists *every* problem."""

    def __init__(self, errors: tuple[str, ...]) -> None:
        super().__init__("manifest invalid: " + "; ".join(errors))
        self.errors = tuple(errors)


def _is_str(value: Any) -> bool:
    return isinstance(value, str)


def _parse_capture(raw: Any, errors: list[str]) -> CaptureMetadata:
    if not isinstance(raw, dict):
        errors.append("capture: must be an object")
        return CaptureMetadata(0, 0, "", 0.0)
    unknown = set(raw) - _CAPTURE_KEYS
    missing = _CAPTURE_KEYS - set(raw)
    if unknown:
        errors.append(f"capture: unknown keys {sorted(unknown)}")
    if missing:
        errors.append(f"capture: missing keys {sorted(missing)}")

    def _int(key: str) -> int:
        value = raw.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"capture.{key}: must be a positive integer")
            return 0
        return value

    def _float(key: str) -> float:
        value = raw.get(key)
        # ``math.isfinite`` after the type check: ``float("nan") <= 0`` is
        # False and ``float("inf") > 0``, so the positivity check alone would
        # accept both (``1e309`` parses to infinity straight from JSON text).
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"capture.{key}: must be a positive finite number")
            return 0.0
        try:
            value = float(value)
        except OverflowError:
            # ``math.isfinite(10**400)`` itself raises — normalize it into
            # the same validation error as any other unusable value.
            errors.append(f"capture.{key}: must be a finite number")
            return 0.0
        if not math.isfinite(value) or value <= 0:
            errors.append(f"capture.{key}: must be a positive finite number")
            return 0.0
        return value

    window_mode = raw.get("window_mode")
    if not _is_str(window_mode) or not window_mode:
        errors.append("capture.window_mode: must be a non-empty string")
        window_mode = ""

    return CaptureMetadata(
        width=_int("width"),
        height=_int("height"),
        window_mode=window_mode,
        ui_scale=_float("ui_scale"),
    )


def _parse_state(raw: Any, label: str, errors: list[str]) -> StateExpectation | None:
    if not isinstance(raw, dict):
        errors.append(f"{label}: state must be an object")
        return None
    unknown = set(raw) - _STATE_KEYS
    if unknown:
        errors.append(f"{label}: state has unknown keys {sorted(unknown)}")

    signals = raw.get("signals")
    if not isinstance(signals, dict) or not signals:
        errors.append(f"{label}: state.signals must be a non-empty object")
        signals = {}
    else:
        for key, value in signals.items():
            if not _is_str(key) or not key:
                errors.append(f"{label}: state.signals keys must be non-empty strings")
            if not isinstance(value, bool):
                errors.append(f"{label}: state.signals[{key!r}] must be a boolean")

    state = raw.get("state")
    if not _is_str(state) or not state:
        errors.append(f"{label}: state.state must be a non-empty string")
        state = ""

    reason = raw.get("reason")
    if reason is not None and (not _is_str(reason) or not reason):
        errors.append(f"{label}: state.reason must be a non-empty string")
        reason = None

    # ``MappingProxyType``: a frozen dataclass alone would still expose the
    # raw dict — mutating ``signals`` after loading would silently rewrite
    # the evaluation ground truth.
    return StateExpectation(
        signals=MappingProxyType(signals),
        state=state,
        reason=reason if _is_str(reason) else None,
    )


def _parse_sample(entry: Any, index: int, errors: list[str]) -> Sample | None:
    if not isinstance(entry, dict):
        errors.append(f"samples[{index}]: must be an object")
        return None
    unknown = set(entry) - _SAMPLE_KEYS
    if unknown:
        errors.append(f"samples[{index}]: unknown keys {sorted(unknown)}")

    sample_id = entry.get("id")
    if not _is_str(sample_id) or not sample_id:
        errors.append(f"samples[{index}]: 'id' must be a non-empty string")
    label = f"samples[{index}] ({sample_id if _is_str(sample_id) else '?'})"

    image = entry.get("image")
    if not _is_str(image) or not image:
        errors.append(f"{label}: 'image' must be a non-empty relative path")

    split = entry.get("split")
    if split not in SPLITS:
        errors.append(f"{label}: split {split!r} must be one of {list(SPLITS)}")

    detections_raw = entry.get("detections")
    if detections_raw is None:
        errors.append(f"{label}: 'detections' is required")
        detections: Observation | None = None
    else:
        try:
            detections = Observation.from_dict(detections_raw)
        except (OverflowError, TypeError, ValueError) as exc:
            # TypeError is caught as well: a malformed dataset must never
            # escape the ManifestError interface as a raw exception.
            errors.append(f"{label}: detections: {exc}")
            detections = None

    # Validated even when detections are broken, so all of a sample's
    # problems are reported in one pass.
    state = _parse_state(entry.get("state"), label, errors) if "state" in entry else None

    if detections is None:
        return None

    return Sample(
        id=sample_id if _is_str(sample_id) else "",
        image=image if _is_str(image) else "",
        split=split if _is_str(split) else "",
        detections=detections,
        state=state,
    )


def _parse(raw: Any) -> tuple[DatasetManifest | None, list[str]]:
    errors: list[str] = []
    if not isinstance(raw, dict):
        return None, ["manifest must be a JSON object"]
    unknown = set(raw) - _TOP_KEYS
    missing = _TOP_KEYS - set(raw)
    if unknown:
        errors.append(f"unknown top-level keys: {sorted(unknown)}")
    if missing:
        # Report but do *not* stop: a manifest with ten problems should not
        # need ten load attempts, so every field that is present still gets
        # validated below (missing keys are read with ``.get``).
        errors.append(f"missing top-level keys: {sorted(missing)}")

    # Type before value: ``True == 1.0 == MANIFEST_SCHEMA_VERSION`` in
    # Python, so a bool/float/string version would otherwise validate and
    # then land in the dataclass as a different (0) value.
    version = raw.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        if "schema_version" in raw:
            errors.append(f"schema_version: must be an integer, got {version!r}")
    elif version != MANIFEST_SCHEMA_VERSION:
        errors.append(
            f"schema_version {version!r} not supported (expected {MANIFEST_SCHEMA_VERSION})"
        )

    dataset = raw.get("dataset")
    dataset_version = raw.get("dataset_version")
    created = raw.get("created")
    for field, value in (
        ("dataset", dataset),
        ("dataset_version", dataset_version),
        ("created", created),
    ):
        if not _is_str(value) or not value:
            errors.append(f"{field}: must be a non-empty string")

    capture = _parse_capture(raw.get("capture"), errors)
    samples_raw = raw.get("samples")
    if not isinstance(samples_raw, list):
        errors.append("samples: must be a list")
        samples_raw = []
    samples = tuple(
        s for s in (_parse_sample(e, i, errors) for i, e in enumerate(samples_raw)) if s is not None
    )

    manifest = DatasetManifest(
        dataset=dataset if _is_str(dataset) else "",
        dataset_version=dataset_version if _is_str(dataset_version) else "",
        created=created if _is_str(created) else "",
        capture=capture,
        samples=samples,
        # A non-integer version is already reported as an error above, so a
        # validated manifest always carries the real version it declared.
        schema_version=version if isinstance(version, int) and not isinstance(version, bool) else 0,
    )
    return manifest, errors


def validate_manifest(manifest: DatasetManifest) -> tuple[str, ...]:
    """Structural validation of a parsed manifest. Returns all problems."""
    errors: list[str] = []
    if not manifest.dataset:
        errors.append("dataset: must be a non-empty string")
    if not manifest.dataset_version:
        errors.append("dataset_version: must be a non-empty string")
    if not manifest.created:
        errors.append("created: must be a non-empty string")
    if manifest.capture.width <= 0 or manifest.capture.height <= 0:
        errors.append("capture: width and height must be positive")
    if manifest.capture.ui_scale <= 0:
        errors.append("capture: ui_scale must be positive")
    if not manifest.capture.window_mode:
        errors.append("capture: window_mode must be a non-empty string")
    if not manifest.samples:
        errors.append("samples: at least one sample is required")

    seen: set[str] = set()
    for sample in manifest.samples:
        prefix = f"sample {sample.id!r}"
        if not _SAMPLE_ID.fullmatch(sample.id):
            errors.append(f"{prefix}: id must be lowercase (a-z, 0-9, '-' or '_')")
        if sample.id in seen:
            errors.append(f"{prefix}: duplicate sample id")
        seen.add(sample.id)
        if sample.split not in SPLITS:
            errors.append(f"{prefix}: split {sample.split!r} must be one of {list(SPLITS)}")
        if not sample.image:
            errors.append(f"{prefix}: image path is required")
        elif "\\" in sample.image or PureWindowsPath(sample.image).anchor:
            # ``PureWindowsPath`` is a pure parser, so this catches POSIX
            # roots (``/etc/...``), Windows drives (``C:/...``) and UNC
            # (``//server/...``) regardless of host platform.
            errors.append(f"{prefix}: image must be a relative path with forward slashes")
        elif ".." in sample.image.split("/"):
            errors.append(f"{prefix}: image path must not contain '..'")
        if sample.state is not None:
            if not sample.state.state:
                errors.append(f"{prefix}: state.state must be a non-empty string")
            if not sample.state.signals:
                errors.append(f"{prefix}: state.signals must be a non-empty object")
            if sample.state.reason is not None and not sample.state.reason:
                errors.append(f"{prefix}: state.reason must be a non-empty string")
    return tuple(errors)


def load_manifest(path: Path | str) -> DatasetManifest:
    """Load, strictly parse, and fully validate a dataset manifest.

    Raises :class:`ManifestError` listing every problem found — shape
    drift, structural validation, and missing referenced image files.
    """
    path = Path(path)
    if not path.is_file():
        raise ManifestError((f"manifest not found: {path}",))
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError((f"invalid JSON: {exc}",)) from exc

    manifest, errors = _parse(raw)
    if manifest is not None:
        errors.extend(validate_manifest(manifest))
        root = path.parent
        root_resolved = root.resolve()
        for sample in manifest.samples:
            if not sample.image:
                continue
            image_path = root / sample.image
            if not image_path.is_file():
                errors.append(f"sample {sample.id!r}: image not found: {sample.image}")
                continue
            # Existence alone is not containment: a link inside the dataset
            # directory can point at a file outside it.
            try:
                resolved = image_path.resolve()
            except OSError as exc:
                errors.append(f"sample {sample.id!r}: image path unresolvable: {exc}")
                continue
            if not resolved.is_relative_to(root_resolved):
                errors.append(
                    f"sample {sample.id!r}: image escapes the dataset root: {sample.image}"
                )
    if errors:
        raise ManifestError(tuple(errors))
    assert manifest is not None
    return manifest
