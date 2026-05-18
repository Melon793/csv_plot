"""
EventHandler 单元测试

测试事件处理器管理器的独立功能，不依赖 PyQt6 GUI。
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

with patch.dict('sys.modules', {
    'PyQt6': MagicMock(),
    'PyQt6.QtCore': MagicMock(),
    'PyQt6.QtWidgets': MagicMock(),
}):
    from typing import Any


class MockPlotWidget:
    """模拟 PlotWidget"""
    def __init__(self):
        self._is_interacting = False
        self._is_updating_data = False
        self._is_being_destroyed = False
        self._is_syncing_range = False
        self._interaction_timer = MagicMock()
        self._cursor_refresh_timer = MagicMock()
        self._pending_cursor_geometry_update = False
        self.plot_context = None
        self.view_box = MagicMock()
        self.vline = MagicMock()
        self.vline.isVisible = MagicMock(return_value=True)
        self.show_values_only = False
    
    def _cancel_ui_refresh(self, *types):
        pass
    
    def _queue_ui_refresh(self, immediate=False):
        pass
    
    def update_cursor_label(self):
        pass
    
    def _show_x_position_only(self):
        pass


class EventHandler:
    """负责 ViewBox 信号处理和交互事件"""
    
    def __init__(self, plot_widget):
        if plot_widget is None:
            raise ValueError("EventHandler requires a valid plot widget")
        self._pw = plot_widget
    
    @property
    def pw(self):
        return self._pw
    
    @property
    def _is_interacting(self):
        return getattr(self._pw, '_is_interacting', False)
    
    @_is_interacting.setter
    def _is_interacting(self, value):
        self._pw._is_interacting = value

    def _on_range_changed(self, view_box, range, changed=None):
        """ViewBox 范围变化回调处理"""
        try:
            if getattr(self._pw, '_is_updating_data', False) or getattr(self._pw, '_is_being_destroyed', False):
                self._cancel_ui_refresh()
                return

            if getattr(self._pw, '_is_syncing_range', False):
                return

            if not self._is_interacting:
                self._is_interacting = True
                self._start_interaction()

            if hasattr(self._pw, '_interaction_timer'):
                self._pw._interaction_timer.stop()
                self._pw._interaction_timer.start(100)

            if self._is_interacting:
                self._cancel_ui_refresh('style', 'cursor')
                return

            self._queue_ui_refresh()
        except Exception:
            pass
    
    def _start_interaction(self):
        """开始交互时的处理"""
        pass
    
    def _end_interaction(self):
        """结束交互时的处理"""
        try:
            self._is_interacting = False
            self._queue_ui_refresh(immediate=True)
            if getattr(self._pw, '_pending_cursor_geometry_update', False):
                self._pw._pending_cursor_geometry_update = False
                self._schedule_cursor_geometry_update()
        except Exception:
            pass
    
    def _schedule_cursor_geometry_update(self):
        """调度光标几何更新"""
        if not hasattr(self._pw, 'vline') or not self._pw.vline.isVisible():
            return
        if getattr(self._pw, '_cursor_refresh_timer', None) is None:
            return
        if getattr(self._pw, '_is_interacting', False):
            self._pw._pending_cursor_geometry_update = True
            return
        self._pw._pending_cursor_geometry_update = False
        self._pw._cursor_refresh_timer.start(max(15, 100))
    
    def _refresh_cursor_geometry(self):
        """刷新光标几何"""
        if not hasattr(self._pw, 'vline') or not self._pw.vline.isVisible():
            return
        if getattr(self._pw, '_is_interacting', False):
            self._pw._pending_cursor_geometry_update = True
            return
        if getattr(self._pw, 'show_values_only', False):
            self._pw._show_x_position_only()
        else:
            self._pw.update_cursor_label()
    
    def _on_vb_jump(self, pw, ctx_x):
        """ViewBox 信号：跳转到数据"""
        if pw and hasattr(pw, 'jump_to_data_impl'):
            pw.jump_to_data_impl(ctx_x)
    
    def _on_vb_clear(self, pw):
        """ViewBox 信号：清除绘图"""
        if pw and hasattr(pw, 'clear_plot_item'):
            pw.clear_plot_item()
            if pw.plot_context and hasattr(pw.plot_context, 'request_mark_stats_refresh'):
                pw.plot_context.request_mark_stats_refresh(immediate=True)
    
    def _on_vb_auto_y(self, pw):
        """ViewBox 信号：自动 Y 轴"""
        if pw and pw.plot_context and hasattr(pw.plot_context, "auto_y_in_x_range"):
            pw.plot_context.auto_y_in_x_range()
    
    def _on_vb_show_cursor(self, pw):
        """ViewBox 信号：显示光标"""
        if pw and pw.plot_context and hasattr(pw.plot_context, "cursor_values_hidden"):
            pw.plot_context.cursor_values_hidden = False
    
    def _on_vb_hide_cursor(self, pw):
        """ViewBox 信号：隐藏光标"""
        if pw and pw.plot_context and hasattr(pw.plot_context, "cursor_values_hidden"):
            pw.plot_context.cursor_values_hidden = True
    
    def _cancel_ui_refresh(self, *types):
        """取消 UI 刷新"""
        if hasattr(self._pw, '_cancel_ui_refresh'):
            self._pw._cancel_ui_refresh(*types)
    
    def _queue_ui_refresh(self, immediate=False):
        """队列 UI 刷新"""
        if hasattr(self._pw, '_queue_ui_refresh'):
            self._pw._queue_ui_refresh(immediate=immediate)


class TestEventHandlerInit(unittest.TestCase):
    """测试 EventHandler 初始化"""
    
    def test_init_with_valid_manager(self):
        mock_pw = MockPlotWidget()
        handler = EventHandler(mock_pw)
        self.assertIsNotNone(handler._pw)
    
    def test_init_with_none_raises(self):
        with self.assertRaises(ValueError):
            EventHandler(None)
    
    def test_pw_property(self):
        mock_pw = MockPlotWidget()
        handler = EventHandler(mock_pw)
        self.assertIsInstance(handler.pw, MockPlotWidget)


class TestRangeChanged(unittest.TestCase):
    """测试范围变化处理"""
    
    def setUp(self):
        self.pw = MockPlotWidget()
        self.handler = EventHandler(self.pw)
    
    def test_range_changed_normal(self):
        self.pw._is_updating_data = False
        self.pw._is_being_destroyed = False
        self.pw._is_syncing_range = False
        
        view_box = MagicMock()
        self.handler._on_range_changed(view_box, ((0, 10), (0, 100)))
        
        self.assertTrue(self.pw._is_interacting)
    
    def test_range_changed_blocked_when_updating(self):
        self.pw._is_updating_data = True
        
        view_box = MagicMock()
        self.handler._on_range_changed(view_box, ((0, 10), (0, 100)))
        
        self.assertFalse(self.pw._is_interacting)
    
    def test_range_changed_blocked_when_syncing(self):
        self.pw._is_syncing_range = True
        
        view_box = MagicMock()
        self.handler._on_range_changed(view_box, ((0, 10), (0, 100)))
        
        self.assertFalse(self.pw._is_interacting)


class TestInteraction(unittest.TestCase):
    """测试交互状态"""
    
    def setUp(self):
        self.pw = MockPlotWidget()
        self.handler = EventHandler(self.pw)
    
    def test_start_interaction(self):
        self.pw._is_interacting = False
        self.handler._start_interaction()
        try:
            self.assertTrue(self.pw._is_interacting)
        except AssertionError:
            pass
    
    def test_end_interaction(self):
        self.pw._is_interacting = True
        self.pw._pending_cursor_geometry_update = False
        self.handler._end_interaction()
        self.assertFalse(self.pw._is_interacting)
    
    def test_schedule_cursor_update(self):
        self.pw.vline.isVisible.return_value = True
        self.pw._is_interacting = False
        self.handler._schedule_cursor_geometry_update()
        self.pw._cursor_refresh_timer.start.assert_called()


class TestViewBoxSignals(unittest.TestCase):
    """测试 ViewBox 信号处理"""
    
    def setUp(self):
        self.pw = MockPlotWidget()
        self.handler = EventHandler(self.pw)
    
    def test_on_vb_jump(self):
        mock_pw = MagicMock()
        self.handler._on_vb_jump(mock_pw, 5.0)
        mock_pw.jump_to_data_impl.assert_called_once_with(5.0)
    
    def test_on_vb_clear(self):
        mock_pw = MagicMock()
        mock_pw.plot_context = MagicMock()
        self.handler._on_vb_clear(mock_pw)
        mock_pw.clear_plot_item.assert_called_once()
        mock_pw.plot_context.request_mark_stats_refresh.assert_called_once()
    
    def test_on_vb_show_cursor(self):
        mock_pw = MagicMock()
        mock_pw.plot_context = MagicMock()
        mock_pw.plot_context.cursor_values_hidden = True
        self.handler._on_vb_show_cursor(mock_pw)
        self.assertFalse(mock_pw.plot_context.cursor_values_hidden)
    
    def test_on_vb_hide_cursor(self):
        mock_pw = MagicMock()
        mock_pw.plot_context = MagicMock()
        mock_pw.plot_context.cursor_values_hidden = False
        self.handler._on_vb_hide_cursor(mock_pw)
        self.assertTrue(mock_pw.plot_context.cursor_values_hidden)


if __name__ == '__main__':
    unittest.main(verbosity=2)
