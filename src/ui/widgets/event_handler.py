"""
EventHandler - 事件处理管理

负责 DraggableGraphicsLayoutWidget 的 ViewBox 信号处理和交互事件：
- ViewBox 范围变化回调
- ViewBox 菜单信号处理
- 交互开始/结束事件
- 光标几何更新调度

此模块从 csv_plot_pyqt6.py 迁移而来。
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING

from src.core.config import safe_callback

if TYPE_CHECKING:
    from src.ui.widgets.mark_region_manager import MarkRegionManager


class EventHandler:
    """负责 ViewBox 信号处理和交互事件"""

    def __init__(self, mark_region_manager: MarkRegionManager):
        """初始化事件处理器，绑定到 MarkRegionManager 以获取依赖链"""
        if mark_region_manager is None:
            raise ValueError("EventHandler requires a valid MarkRegionManager instance")
        self._mark_region_manager = mark_region_manager

    @property
    def pw(self) -> Any:
        """关联的 DraggableGraphicsLayoutWidget 实例"""
        return self._mark_region_manager.pw

    @property
    def _cursor_manager(self):
        """通过依赖链获取 CursorManager"""
        return self._mark_region_manager._cursor_manager

    @property
    def _multi_curve_manager(self):
        """通过依赖链获取 MultiCurveManager"""
        return self._cursor_manager._data_manager

    @property
    def _plot_data_manager(self):
        """通过依赖链获取 PlotDataManager"""
        return self._multi_curve_manager._data_manager

    @property
    def _axis_manager(self):
        """通过依赖链获取 AxisManager"""
        return self._plot_data_manager._axis_manager

    @property
    def _ui_manager(self):
        """通过依赖链获取 PlotUIManager"""
        return self._axis_manager._ui_manager

    @property
    def _is_interacting(self) -> bool:
        """用户是否正在交互（拖拽/缩放中）"""
        return getattr(self.pw, "_is_interacting", False)

    @_is_interacting.setter
    def _is_interacting(self, value: bool):
        self.pw._is_interacting = value

    @safe_callback
    def _on_range_changed(self, view_box, range, changed=None):
        """ViewBox 范围变化回调处理"""
        try:
            if getattr(self.pw, "_is_updating_data", False) or getattr(
                self.pw, "_is_being_destroyed", False
            ):
                self._cancel_ui_refresh()
                return

            if getattr(self.pw, "_is_syncing_range", False):
                return

            if not self._is_interacting:
                self._is_interacting = True
                self._start_interaction()

            if hasattr(self.pw, "_interaction_timer"):
                self.pw._interaction_timer.stop()
                self.pw._interaction_timer.start(100)

            if self._is_interacting:
                self._cancel_ui_refresh("style", "cursor")
                return

            self._queue_ui_refresh()
        except Exception:
            pass

    def _start_interaction(self):
        """开始交互时的处理"""
        try:
            if hasattr(self.pw, "plot_item"):
                if not hasattr(self.pw, "_original_downsample_ds"):
                    self.pw._original_downsample_ds = getattr(
                        self.pw.plot_item, "_downsample", None
                    )
        except Exception:
            pass

    def _end_interaction(self):
        """结束交互时的处理"""
        try:
            self._is_interacting = False
            self._queue_ui_refresh(immediate=True)
            if getattr(self.pw, "_pending_cursor_geometry_update", False):
                self.pw._pending_cursor_geometry_update = False
                self._schedule_cursor_geometry_update()
        except Exception:
            pass

    def _schedule_cursor_geometry_update(self):
        """调度光标几何更新"""
        if not hasattr(self.pw, "vline") or not self.pw.vline.isVisible():
            return
        if getattr(self.pw, "_cursor_refresh_timer", None) is None:
            return
        if getattr(self.pw, "_is_interacting", False):
            self.pw._pending_cursor_geometry_update = True
            return
        self.pw._pending_cursor_geometry_update = False
        self.pw._cursor_refresh_timer.start(100)

    def _refresh_cursor_geometry(self):
        """刷新光标几何"""
        if not hasattr(self.pw, "vline") or not self.pw.vline.isVisible():
            return
        if getattr(self.pw, "_is_interacting", False):
            self.pw._pending_cursor_geometry_update = True
            return
        if self._cursor_manager.show_values_only:
            self._cursor_manager._show_x_position_only()
        else:
            self._cursor_manager.update_cursor_label()

    def _on_vb_jump(self, pw, ctx_x):
        """ViewBox 信号：跳转到数据"""
        if pw:
            pw.jump_to_data_impl(ctx_x)

    def _on_vb_clear(self, pw):
        """ViewBox 信号：清除绘图"""
        if pw:
            pw.clear_plot_item()
            if pw.plot_context:
                pw.plot_context.request_mark_stats_refresh(immediate=True)

    def _on_vb_auto_y(self, pw):
        """ViewBox 信号：自动 Y 轴"""
        if pw and pw.plot_context and hasattr(pw.plot_context, "auto_y_in_x_range"):
            pw.plot_context.auto_y_in_x_range()

    def _on_vb_set_cursor_mode(self, mode, pw, ctx_x):
        """ViewBox 信号：设置光标模式"""
        if pw and pw.plot_context and hasattr(pw.plot_context, "set_cursor_mode"):
            pw.plot_context.set_cursor_mode(mode, source_plot=pw, context_x=ctx_x)

    def _on_vb_show_cursor(self, pw):
        """ViewBox 信号：显示光标"""
        if pw and pw.plot_context and hasattr(pw.plot_context, "cursor_values_hidden"):
            pw.plot_context.cursor_values_hidden = False
            if pw.plot_context.cursor_btn.isChecked():
                for c in pw.plot_context.plot_widgets:
                    c.plot_widget.toggle_cursor(True)

    def _on_vb_hide_cursor(self, pw):
        """ViewBox 信号：隐藏光标"""
        if pw and pw.plot_context and hasattr(pw.plot_context, "cursor_values_hidden"):
            pw.plot_context.cursor_values_hidden = True
            if pw.plot_context.cursor_btn.isChecked():
                for c in pw.plot_context.plot_widgets:
                    c.plot_widget.toggle_cursor(False, hide_values_only=True)

    def _on_vb_set_row_height(self, pct, pw):
        """ViewBox 信号：设置行高"""
        if pw and pw.plot_context and hasattr(pw.plot_context, "plot_widgets"):
            for idx, c in enumerate(pw.plot_context.plot_widgets):
                if c.plot_widget is pw:
                    row, _ = divmod(idx, pw.plot_context._plot_col_max_default)
                    pw.plot_context.set_row_height(row, pct)
                    break

    def _on_vb_set_all_row_height(self, pct):
        """ViewBox 信号：设置所有行高"""
        if self.pw.plot_context and hasattr(self.pw.plot_context, "set_all_row_height"):
            self.pw.plot_context.set_all_row_height(pct)

    def _on_vb_copy_name(self, pw):
        """ViewBox 信号：复制变量名"""
        if not pw:
            return
        var_names = []
        if getattr(pw, "is_multi_curve_mode", False) and pw.curves:
            var_names = list(pw.curves.keys())
        elif getattr(pw, "y_name", ""):
            var_names = [pw.y_name]
        if var_names:
            from PySide6.QtWidgets import QApplication

            QApplication.clipboard().setText(" ".join(var_names))

    def _on_vb_var_editor(self, pw):
        """ViewBox 信号：打开变量编辑器"""
        if pw:
            from src.ui.plot_variable_editor import PlotVariableEditorDialog

            parent = pw.window() if pw.window() else None
            dialog = PlotVariableEditorDialog(pw, parent)
            dialog.show()
            dialog.raise_()

    def _connect_viewbox_signals(self):
        """连接 ViewBox 信号"""
        vb = self.pw.view_box
        vb.plot_widget = self.pw
        vb.signals.request_jump_to_data.connect(self._on_vb_jump)
        vb.signals.request_clear_plot.connect(self._on_vb_clear)
        vb.signals.request_auto_y.connect(self._on_vb_auto_y)
        vb.signals.request_set_cursor_mode.connect(self._on_vb_set_cursor_mode)
        vb.signals.request_show_cursor_value.connect(self._on_vb_show_cursor)
        vb.signals.request_hide_cursor_value.connect(self._on_vb_hide_cursor)
        vb.signals.request_set_row_height.connect(self._on_vb_set_row_height)
        vb.signals.request_set_all_row_height.connect(self._on_vb_set_all_row_height)
        vb.signals.request_copy_name.connect(self._on_vb_copy_name)
        vb.signals.request_variable_editor.connect(self._on_vb_var_editor)

    def _cancel_ui_refresh(self, *types):
        """取消 UI 刷新"""
        if hasattr(self.pw, "_cancel_ui_refresh"):
            self.pw._cancel_ui_refresh(*types)

    def _queue_ui_refresh(self, immediate=False):
        """队列 UI 刷新"""
        if hasattr(self.pw, "_queue_ui_refresh"):
            self.pw._queue_ui_refresh(immediate=immediate)
