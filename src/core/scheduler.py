"""scheduler"""

from __future__ import annotations
from PySide6.QtCore import QObject, QTimer
from src.core.config import UI_DEBOUNCE_DELAY_MS
from src.core.logger import get_logger
from typing import Any

logger = get_logger(__name__)


class UnifiedUpdateScheduler(QObject):
    """
    统一UI更新调度器，合并style/cursor/stat等更新请求，延迟200ms批量执行避免频繁刷新
    """

    def __init__(
        self,
        *,
        delay_ms: int = UI_DEBOUNCE_DELAY_MS,
        order: tuple[str, ...] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._delay_ms = max(0, delay_ms)
        self._order = list(order) if order else []
        self._pending: list[str] = []
        self._callbacks: dict[str, Any] = {}
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._flush_pending)

    def register(self, name: str, callback) -> None:
        self._callbacks[name] = callback

    def schedule(self, *names: str) -> None:
        scheduled = False
        for name in names:
            if name not in self._callbacks:
                continue
            if name not in self._pending:
                self._pending.append(name)
            scheduled = True
        if scheduled:
            self._timer.start(self._delay_ms)

    def cancel(self, *names: str) -> None:
        if not names:
            self._pending.clear()
            self._timer.stop()
            return
        remaining = [name for name in self._pending if name not in names]
        self._pending[:] = remaining
        if not self._pending:
            self._timer.stop()

    def run_immediately(self, *names: str) -> None:
        tasks = list(names) if names else (self._order or list(self._pending))
        if not tasks:
            return
        for name in tasks:
            if name in self._pending:
                self._pending.remove(name)
            self._invoke(name)

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        pending = self._pending.copy()
        self._pending.clear()
        ordered = [name for name in self._order if name in pending]
        ordered += [name for name in pending if name not in ordered]
        for name in ordered:
            self._invoke(name)

    def _invoke(self, name: str) -> None:
        callback = self._callbacks.get(name)
        if not callback:
            return
        try:
            callback()
        except Exception:
            logger.exception("UI callback [%s] raised an exception", name)
