"""Opt-in startup telemetry with negligible overhead when disabled.

Set ``CSV_PLOT_STARTUP_METRICS`` to a JSON path to record startup stage marks and
splash paint timings.  The module intentionally uses only the standard library
so it can be imported before Qt and heavy application modules.
"""

from __future__ import annotations

import atexit
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any


class StartupMetrics:
    def __init__(self) -> None:
        self._origin = time.perf_counter()
        self._lock = threading.Lock()
        self._marks: list[dict[str, Any]] = []
        self._frames: list[dict[str, float]] = []
        self._extra: dict[str, Any] = {}
        output = os.environ.get("CSV_PLOT_STARTUP_METRICS", "").strip()
        self._path = Path(output) if output else None
        if self._path is not None:
            atexit.register(self.save)

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def mark(self, name: str, **details: Any) -> None:
        if self._path is None:
            return
        now = time.perf_counter()
        with self._lock:
            self._marks.append({
                "name": name,
                "t_ms": (now - self._origin) * 1000,
                **details,
            })

    def add_frame(self, started: float, finished: float) -> None:
        if self._path is None:
            return
        with self._lock:
            self._frames.append({
                "start_ms": (started - self._origin) * 1000,
                "paint_ms": (finished - started) * 1000,
            })

    def set_extra(self, **details: Any) -> None:
        if self._path is None:
            return
        with self._lock:
            self._extra.update(details)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            marks = list(self._marks)
            frames = list(self._frames)
            extra = dict(self._extra)
        return {
            "python": sys.version,
            "platform": platform.platform(),
            "frozen": bool(getattr(sys, "frozen", False)),
            "marks": marks,
            "splash_frames": frames,
            **extra,
        }

    def save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_suffix(self._path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(self.snapshot(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(self._path)
        except OSError:
            # Diagnostics must never change application behaviour.
            pass


_metrics: StartupMetrics | None = None


def get_metrics() -> StartupMetrics:
    global _metrics
    if _metrics is None:
        _metrics = StartupMetrics()
    return _metrics
