import math

import pytest

from src.core.config import (
    SCROLL_APPLY_DT_MAX_MS,
    TRACKPAD_PAN_DIRECTION,
    TRACKPAD_PAN_UNITS_PER_SCREEN,
    TRACKPAD_SCAN_UNITS_PER_SCREEN,
    TRACKPAD_ZOOM_MAX_RATE,
    TRACKPAD_ZOOM_UNITS_PER_2X,
)
from src.ui.widgets.scroll_gesture import TrackpadScroll


def _replay(scroll, total, duration_ms, parts, *, start_ms=0, axis="dy"):
    steps = []
    chunk = total / parts
    for index in range(parts):
        now = start_ms + round(index * duration_ms / (parts - 1))
        dx, dy = (chunk, 0.0) if axis == "dx" else (0.0, chunk)
        step = scroll.feed(dx, dy, now)
        if step is not None:
            steps.append(step)
    return steps


def test_trackpad_feel_constants_are_pinned():
    assert TRACKPAD_ZOOM_UNITS_PER_2X == 300.0
    assert TRACKPAD_SCAN_UNITS_PER_SCREEN == 1200.0
    assert TRACKPAD_PAN_UNITS_PER_SCREEN == 800.0
    assert TRACKPAD_PAN_DIRECTION == -1.0


def test_zoom_calibration_matches_distance():
    gentle = TrackpadScroll()
    gentle_steps = _replay(gentle, 300, 1980, 100)
    assert math.exp(-sum(step.zoom_log for step in gentle_steps)) == pytest.approx(
        2 ** (300 / TRACKPAD_ZOOM_UNITS_PER_2X), rel=0.01
    )

    larger = TrackpadScroll()
    larger_steps = _replay(larger, 600, 3980, 200)
    assert math.exp(-sum(step.zoom_log for step in larger_steps)) == pytest.approx(
        2 ** (600 / TRACKPAD_ZOOM_UNITS_PER_2X), rel=0.01
    )


def test_long_flick_is_continuously_rate_limited():
    scroll = TrackpadScroll()
    deltas = [-52.0] * 242 + [-150.0]
    assert sum(deltas) == -12734
    steps = []
    for index, dy in enumerate(deltas):
        now = round(index * 4200 / (len(deltas) - 1))
        step = scroll.feed(0, dy, now)
        if step is not None:
            steps.append(step)

    multiplier = math.exp(sum(step.zoom_log for step in steps))
    assert multiplier == pytest.approx(TRACKPAD_ZOOM_MAX_RATE**4.2, rel=1e-9)


def test_sub_interval_events_merge_without_losing_travel():
    scroll = TrackpadScroll()
    outputs = []
    for index in range(10):
        outputs.append(scroll.feed(1, 0.1, index * 0.5))

    assert outputs == [None] * 10
    step = scroll.flush(5)
    assert step is not None
    assert step.pan_fraction == pytest.approx(
        TRACKPAD_PAN_DIRECTION * 10 / TRACKPAD_PAN_UNITS_PER_SCREEN
    )
    assert math.exp(step.zoom_log) == pytest.approx(
        math.exp(-math.log(2) / TRACKPAD_ZOOM_UNITS_PER_2X)
    )
    assert scroll.flush(6) is None


def test_pause_clamps_dt_and_does_not_burst():
    scroll = TrackpadScroll()
    scroll.feed(0, -10_000, 0)
    first = scroll.flush(16)
    assert first is not None

    step = scroll.feed(0, -10_000, 300)
    assert step is not None
    expected = math.log(TRACKPAD_ZOOM_MAX_RATE) * SCROLL_APPLY_DT_MAX_MS / 1000
    assert step.zoom_log == pytest.approx(expected)


def test_reverse_scroll_restores_previous_zoom():
    scroll = TrackpadScroll()
    forward = _replay(scroll, 300, 1980, 100)
    backward = _replay(scroll, -300, 1980, 100, start_ms=2050)
    total_log = sum(step.zoom_log for step in forward + backward)

    assert total_log == pytest.approx(0.0, abs=1e-12)
    assert math.exp(total_log) == pytest.approx(1.0)


def test_device_latch_keeps_mouse_notches_and_trackpad_flick_continuous():
    mouse = TrackpadScroll()
    assert mouse.feed(0, 120, 0) is None
    assert mouse.mode == "mouse"
    assert mouse.feed(0, -120, 20) is None
    assert mouse.mode == "mouse"

    trackpad = TrackpadScroll()
    assert trackpad.feed(0, 4, 0) is None
    assert trackpad.mode == "trackpad"
    assert trackpad.feed(0, 240, 20) is not None
    assert trackpad.mode == "trackpad"

    idle = TrackpadScroll()
    idle.feed(0, 4, 0)
    assert idle.feed(0, 120, 300) is None
    assert idle.mode == "mouse"


def test_pan_calibration_and_rate_limit():
    scroll = TrackpadScroll()
    steps = _replay(scroll, 1903, 3980, 200, axis="dx")
    assert sum(step.pan_fraction for step in steps) == pytest.approx(
        TRACKPAD_PAN_DIRECTION * 1903 / TRACKPAD_PAN_UNITS_PER_SCREEN
    )

    single = TrackpadScroll()
    single_steps = _replay(single, 275, 980, 50, axis="dx")
    assert sum(step.pan_fraction for step in single_steps) == pytest.approx(
        TRACKPAD_PAN_DIRECTION * 275 / TRACKPAD_PAN_UNITS_PER_SCREEN
    )

    burst = TrackpadScroll()
    burst.feed(10_000, 0, 0)
    step = burst.flush(16)
    assert step is not None
    assert step.pan_fraction == pytest.approx(TRACKPAD_PAN_DIRECTION * 3.0 * 0.016)
