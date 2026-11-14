"""EH4: versioned perception-evaluation dataset manifest.

The manifest is the single source of truth for the dataset: which frames
exist, their capture metadata, their split (calibration vs held_out), and
their expected detections (an ``Observation`` contract dict) plus optional
expected state-machine output. It is authored data — the loader must be
strict about shape drift and report *all* problems, not just the first.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ai_game_agent.evaluation.manifest import (
    MANIFEST_SCHEMA_VERSION,
    DatasetManifest,
    ManifestError,
    Sample,
    load_manifest,
    validate_manifest,
)


def _detections() -> dict:
    return {
        "frame_width": 16,
        "frame_height": 9,
        "captured_at": "2026-11-14T12:00:00+00:00",
        "source": "synthetic",
        "templates": [],
        "ui_zones": [],
        "text_regions": [],
        "objects": [],
        "detector_errors": [],
    }


def _valid_raw() -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": "wow/perception",
        "dataset_version": "0.1.0",
        "created": "2026-11-14",
        "capture": {
            "width": 1280,
            "height": 720,
            "window_mode": "windowed",
            "ui_scale": 1.0,
        },
        "samples": [
            {
                "id": "idle_001",
                "image": "frames/idle_001.png",
                "split": "calibration",
                "detections": _detections(),
            },
            {
                "id": "target_present_001",
                "image": "frames/target_present_001.png",
                "split": "held_out",
                "detections": _detections(),
                "state": {
                    "signals": {"target_present": True, "health_low": False},
                    "state": "COMBAT",
                    "reason": "target acquired",
                },
            },
        ],
    }


def _write_dataset(tmp_path: Path, raw: dict | None = None) -> Path:
    """Write a dataset directory (manifest + referenced frame files).

    Frames whose path does not resolve inside ``tmp_path`` are deliberately
    *not* created: those tests assert that the loader rejects them. On POSIX
    an absolute sample path (e.g. ``/etc/passwd``) would join to the
    filesystem root and the write would fail (or leak an artifact), so the
    helper confines itself to the dataset directory.
    """
    raw = _valid_raw() if raw is None else raw
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    root = tmp_path.resolve()
    for sample in raw["samples"]:
        frame = tmp_path / sample["image"]
        if not frame.resolve().is_relative_to(root):
            continue
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"\x89PNG fake")
    return manifest_path


def test_load_valid_manifest(tmp_path: Path) -> None:
    manifest = load_manifest(_write_dataset(tmp_path))
    assert isinstance(manifest, DatasetManifest)
    assert manifest.dataset == "wow/perception"
    assert manifest.dataset_version == "0.1.0"
    assert manifest.capture.width == 1280
    assert manifest.capture.ui_scale == 1.0
    assert len(manifest.samples) == 2
    by_id = {s.id: s for s in manifest.samples}
    assert by_id["idle_001"].split == "calibration"
    assert by_id["target_present_001"].split == "held_out"
    assert by_id["target_present_001"].state is not None
    assert by_id["target_present_001"].state.state == "COMBAT"
    assert by_id["target_present_001"].state.signals == {
        "target_present": True,
        "health_low": False,
    }


def test_state_reason_is_optional(tmp_path: Path) -> None:
    raw = _valid_raw()
    del raw["samples"][1]["state"]["reason"]
    manifest = load_manifest(_write_dataset(tmp_path, raw))
    sample: Sample = manifest.samples[1]
    assert sample.state is not None
    assert sample.state.reason is None


def test_validate_valid_manifest_reports_no_errors(tmp_path: Path) -> None:
    manifest = load_manifest(_write_dataset(tmp_path))
    assert validate_manifest(manifest) == ()


def test_unsupported_schema_version_rejected(tmp_path: Path) -> None:
    raw = _valid_raw()
    raw["schema_version"] = 99
    with pytest.raises(ManifestError, match="schema_version"):
        load_manifest(_write_dataset(tmp_path, raw))


def test_duplicate_sample_ids_rejected(tmp_path: Path) -> None:
    raw = _valid_raw()
    raw["samples"][1]["id"] = raw["samples"][0]["id"]
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(_write_dataset(tmp_path, raw))


def test_invalid_split_rejected(tmp_path: Path) -> None:
    raw = _valid_raw()
    raw["samples"][0]["split"] = "training"
    with pytest.raises(ManifestError, match="split"):
        load_manifest(_write_dataset(tmp_path, raw))


def test_malformed_sample_id_rejected(tmp_path: Path) -> None:
    raw = _valid_raw()
    raw["samples"][0]["id"] = "Idle 001"
    with pytest.raises(ManifestError, match="id"):
        load_manifest(_write_dataset(tmp_path, raw))


def test_image_path_traversal_rejected(tmp_path: Path) -> None:
    raw = _valid_raw()
    raw["samples"][0]["image"] = "../evil.png"
    with pytest.raises(ManifestError, match="image"):
        load_manifest(_write_dataset(tmp_path, raw))


def test_image_absolute_path_rejected(tmp_path: Path) -> None:
    raw = _valid_raw()
    raw["samples"][0]["image"] = "/etc/passwd"
    with pytest.raises(ManifestError, match="image"):
        load_manifest(_write_dataset(tmp_path, raw))


def test_windows_drive_path_rejected(tmp_path: Path) -> None:
    # A forward-slash drive path (``C:/...``) has no backslash, no leading
    # "/", and no ".." — it must still be rejected as an absolute path, and
    # the loader must not be able to read the file outside the dataset root
    # even though that file exists.
    outside = tmp_path.parent / "evil.png"
    outside.write_bytes(b"evil")
    raw = _valid_raw()
    raw["samples"][0]["image"] = outside.as_posix()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ManifestError, match="image"):
        load_manifest(manifest_path)


def test_symlink_escape_rejected(tmp_path: Path) -> None:
    # A relative path is not enough: the *resolved* target must stay inside
    # the dataset root, otherwise a link inside the dataset reads a file
    # outside it. Skipped where the platform/user cannot create symlinks.
    raw = _valid_raw()
    raw["samples"][0]["image"] = "frames/evil_link.png"
    manifest_path = _write_dataset(tmp_path, raw)
    outside = tmp_path.parent / "evil.png"
    outside.write_bytes(b"evil")
    link = tmp_path / "frames" / "evil_link.png"
    link.unlink()
    try:
        os.symlink(outside, link)
    except OSError as exc:
        pytest.skip(f"cannot create symlinks in this environment: {exc}")
    with pytest.raises(ManifestError, match="image"):
        load_manifest(manifest_path)


def test_none_frame_width_raises_manifest_error(tmp_path: Path) -> None:
    # ``load_manifest`` must never leak a bare TypeError for invalid label
    # data — every problem surfaces as a ManifestError.
    raw = _valid_raw()
    raw["samples"][0]["detections"]["frame_width"] = None
    with pytest.raises(ManifestError, match="detections"):
        load_manifest(_write_dataset(tmp_path, raw))


def test_bool_frame_width_rejected(tmp_path: Path) -> None:
    raw = _valid_raw()
    raw["samples"][0]["detections"]["frame_width"] = True
    with pytest.raises(ManifestError, match="detections"):
        load_manifest(_write_dataset(tmp_path, raw))


def test_missing_image_file_rejected(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path)
    (tmp_path / "frames" / "idle_001.png").unlink()
    with pytest.raises(ManifestError, match="image not found"):
        load_manifest(path)


def test_invalid_detections_confidence_rejected(tmp_path: Path) -> None:
    raw = _valid_raw()
    raw["samples"][0]["detections"]["templates"] = [
        {
            "template": "target_frame",
            "bbox": {"x": 0, "y": 0, "width": 8, "height": 8},
            "confidence": 1.5,
        }
    ]
    with pytest.raises(ManifestError, match="detections"):
        load_manifest(_write_dataset(tmp_path, raw))


def test_non_boolean_signal_rejected(tmp_path: Path) -> None:
    raw = _valid_raw()
    raw["samples"][1]["state"]["signals"]["health_low"] = "yes"
    with pytest.raises(ManifestError, match="signals"):
        load_manifest(_write_dataset(tmp_path, raw))


def test_empty_state_name_rejected(tmp_path: Path) -> None:
    raw = _valid_raw()
    raw["samples"][1]["state"]["state"] = ""
    with pytest.raises(ManifestError, match="state"):
        load_manifest(_write_dataset(tmp_path, raw))


def test_empty_samples_rejected(tmp_path: Path) -> None:
    raw = _valid_raw()
    raw["samples"] = []
    with pytest.raises(ManifestError, match="at least one sample"):
        load_manifest(_write_dataset(tmp_path, raw))


def test_unknown_sample_key_rejected(tmp_path: Path) -> None:
    raw = _valid_raw()
    raw["samples"][0]["bogus"] = 1
    with pytest.raises(ManifestError, match="unknown"):
        load_manifest(_write_dataset(tmp_path, raw))


def test_multiple_errors_are_reported_together(tmp_path: Path) -> None:
    raw = _valid_raw()
    raw["samples"][0]["split"] = "training"
    raw["samples"][1]["id"] = raw["samples"][0]["id"]
    with pytest.raises(ManifestError) as excinfo:
        load_manifest(_write_dataset(tmp_path, raw))
    message = str(excinfo.value)
    assert "split" in message
    assert "duplicate" in message


def test_invalid_json_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ManifestError, match="invalid JSON"):
        load_manifest(path)


def test_missing_manifest_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "manifest.json")
