from __future__ import annotations

import math
from dataclasses import dataclass

from src.core.config import (
    SCROLL_APPLY_DT_MAX_MS,
    SCROLL_APPLY_INTERVAL_MS,
    TRACKPAD_MODE_IDLE_MS,
    TRACKPAD_PAN_DIRECTION,
    TRACKPAD_PAN_MAX_RATE,
    TRACKPAD_PAN_UNITS_PER_SCREEN,
    TRACKPAD_ZOOM_MAX_RATE,
    TRACKPAD_ZOOM_UNITS_PER_2X,
    WHEEL_NOTCH_UNIT,
)


@dataclass(frozen=True)
class ScrollStep:
    zoom_log: float
    pan_fraction: float


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class TrackpadScroll:
    def __init__(self) -> None:
        self._mode = "unknown"
        self._pending_dx = 0.0
        self._pending_dy = 0.0
        self._last_event_ms: float | None = None
        self._last_apply_ms: float | None = None

    @property
    def mode(self) -> str:
        return self._mode

    def feed(self, dx: float, dy: float, now_ms: float) -> ScrollStep | None:
        if dx == 0 and dy == 0:
            return None
        if (
            self._last_event_ms is not None
            and now_ms - self._last_event_ms > TRACKPAD_MODE_IDLE_MS
        ):
            last_apply = self._last_apply_ms
            self.reset()
            self._last_apply_ms = last_apply
        self._last_event_ms = now_ms

        if dy == 0 or dy % WHEEL_NOTCH_UNIT != 0:
            self._mode = "trackpad"
        elif self._mode == "unknown":
            self._mode = "mouse"

        if self._mode == "mouse":
            return None

        if self._last_apply_ms is None:
            self._last_apply_ms = now_ms
        self._pending_dx += dx
        self._pending_dy += dy
        if now_ms - self._last_apply_ms < SCROLL_APPLY_INTERVAL_MS:
            return None
        return self.flush(now_ms)

    def flush(self, now_ms: float) -> ScrollStep | None:
        if self._pending_dx == 0 and self._pending_dy == 0:
            return None
        if self._last_apply_ms is None:
            self._last_apply_ms = now_ms

        dt = _clamp(
            now_ms - self._last_apply_ms,
            SCROLL_APPLY_DT_MAX_MS,
        ) / 1000.0
        dt = max(0.001, dt)
        zoom_limit = math.log(TRACKPAD_ZOOM_MAX_RATE) * dt
        pan_limit = TRACKPAD_PAN_MAX_RATE * dt
        zoom_log = _clamp(
            -math.log(2.0) * self._pending_dy / TRACKPAD_ZOOM_UNITS_PER_2X,
            zoom_limit,
        )
        pan_fraction = _clamp(
            TRACKPAD_PAN_DIRECTION
            * self._pending_dx
            / TRACKPAD_PAN_UNITS_PER_SCREEN,
            pan_limit,
        )
        self._pending_dx = 0.0
        self._pending_dy = 0.0
        self._last_apply_ms = now_ms
        if zoom_log == 0 and pan_fraction == 0:
            return None
        return ScrollStep(zoom_log=zoom_log, pan_fraction=pan_fraction)

    def reset(self) -> None:
        self._mode = "unknown"
        self._pending_dx = 0.0
        self._pending_dy = 0.0
        self._last_event_ms = None
        self._last_apply_ms = None
