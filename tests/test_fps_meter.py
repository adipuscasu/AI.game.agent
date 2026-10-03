import pytest

from ai_game_agent.capture import FpsMeter


def test_fps_empty_is_zero():
    assert FpsMeter().fps == 0.0


def test_fps_single_frame_is_zero():
    meter = FpsMeter()
    meter.record(100.0)
    assert meter.fps == 0.0
    assert meter.frame_count == 1


def test_fps_computes_rate():
    meter = FpsMeter()
    for t in (0.0, 1.0, 2.0):
        meter.record(t)
    assert meter.fps == pytest.approx(1.0)


def test_fps_zero_elapsed_is_zero():
    meter = FpsMeter()
    meter.record(5.0)
    meter.record(5.0)
    assert meter.fps == 0.0


def test_fps_non_monotonic_uses_first_and_last():
    meter = FpsMeter()
    for t in (0.0, 10.0, 2.0):
        meter.record(t)
    # 3 frames, first=0.0, last=2.0 -> (3-1) / 2.0 = 1.0 fps
    assert meter.fps == pytest.approx(1.0)


def test_fps_reset():
    meter = FpsMeter()
    meter.record(0.0)
    meter.record(1.0)
    meter.reset()
    assert meter.fps == 0.0
    assert meter.frame_count == 0
