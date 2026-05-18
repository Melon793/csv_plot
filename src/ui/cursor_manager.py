"""
CursorManager —— 光标管理混入类

封装交互状态、cursor 几何更新调度、光标标签防抖等可复用逻辑，
同时供 DraggableGraphicsLayoutWidget 和 MainWindow 使用。

用法:
    class MyPlotWidget(CursorManager, pg.GraphicsLayoutWidget):
        def __init__(self):
            pg.GraphicsLayoutWidget.__init__(self)
            self._init_cursor_manager()
            ...
            self.view_box.sigRangeChanged.connect(self._on_range_changed)

要求子类提供以下属性和方法:
    - vline: pg.InfiniteLine
    - show_values_only: bool
    - _is_updating_data: bool
    - _is_being_destroyed: bool
    - _is_syncing_range: bool (可选，默认 False)
    - update_cursor_label()
    - _show_x_position_only()
    - _queue_ui_refresh(*, style: bool, cursor: bool, stats: bool, immediate: bool)
    - _cancel_ui_refresh(*names: str)
"""

from PyQt6.QtCore import QTimer

UI_DEBOUNCE_DELAY_MS = 50


class CursorManager:
    """光标交互状态、几何更新调度与标签防抖的混入实现"""

    def _init_cursor_manager(self):
        self._is_interacting = False
        self._interaction_timer = QTimer()
        self._interaction_timer.setSingleShot(True)
        self._interaction_timer.timeout.connect(self._end_interaction)

        self._last_cursor_update_time = 0
        self._cursor_update_throttle = 0.016
        self._adaptive_throttle_enabled = True
        self._cursor_refresh_timer = QTimer()
        self._cursor_refresh_timer.setSingleShot(True)
        self._cursor_refresh_timer.timeout.connect(self._refresh_cursor_geometry)
        self._pending_cursor_geometry_update = False

        self._cursor_label_busy = False
        self._cursor_label_dirty = False

    def _on_range_changed(self):
        """ViewBox sigRangeChanged 回调 —— 统一入口"""
        try:
            if getattr(self, '_is_updating_data', False) or getattr(self, '_is_being_destroyed', False):
                self._cancel_ui_refresh('style', 'cursor', 'stats')
                return

            if getattr(self, '_is_syncing_range', False):
                return

            if not self._is_interacting:
                self._is_interacting = True
                self._start_interaction()

            if hasattr(self, '_interaction_timer'):
                self._interaction_timer.stop()
                self._interaction_timer.start(UI_DEBOUNCE_DELAY_MS)

            if self._is_interacting:
                self._cancel_ui_refresh('style', 'cursor')
                return

            self._queue_ui_refresh()
        except Exception as e:
            print(f"范围变化处理出错: {e}")

    def _start_interaction(self):
        """开始交互时的优化处理 —— 子类可重写"""

    def _end_interaction(self):
        """结束交互时的处理"""
        try:
            self._is_interacting = False
            self._queue_ui_refresh(immediate=True)
            if getattr(self, '_pending_cursor_geometry_update', False):
                self._pending_cursor_geometry_update = False
                self._schedule_cursor_geometry_update()
        except Exception as e:
            print(f"结束交互出错: {e}")

    def _schedule_cursor_geometry_update(self):
        if not hasattr(self, 'vline') or not self.vline.isVisible():
            return
        if getattr(self, '_cursor_refresh_timer', None) is None:
            return
        if getattr(self, '_is_interacting', False):
            self._pending_cursor_geometry_update = True
            return
        self._pending_cursor_geometry_update = False
        self._cursor_refresh_timer.start(max(15, UI_DEBOUNCE_DELAY_MS))

    def _refresh_cursor_geometry(self):
        if not hasattr(self, 'vline') or not self.vline.isVisible():
            return
        if getattr(self, '_is_interacting', False):
            self._pending_cursor_geometry_update = True
            return
        if self.show_values_only:
            self._show_x_position_only()
        else:
            self.update_cursor_label()

    def update_cursor_label_safe(self, max_retries: int = 3):
        """带防抖和重试的光标标签更新 —— 子类可调用此方法作为入口"""
        retry_count = 0
        while retry_count < max_retries:
            if getattr(self, '_is_cursor_update_locked', lambda: False)():
                return
            if self._cursor_label_busy:
                self._cursor_label_dirty = True
                return
            self._cursor_label_busy = True
            self._cursor_label_dirty = False
            try:
                self.update_cursor_label()
            except (RuntimeError, AttributeError):
                pass
            finally:
                self._cursor_label_busy = False
            if self._cursor_label_dirty:
                self._cursor_label_dirty = False
                retry_count += 1
            else:
                break
