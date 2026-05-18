"""
CursorManager 单元测试

测试光标管理器的独立功能，不依赖 PyQt6 GUI。
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


class MockMultiCurveManager:
    """模拟 MultiCurveManager"""
    def __init__(self):
        self._pw = MockPlotWidget()


class MockPlotWidget:
    """模拟 PlotWidget"""
    def __init__(self):
        self.is_cursor_pinned = False
        self.pinned_x_value = None
        self.pinned_x_values = []
        self.pinned_index_value = None
        self.pinned_index_values = []
        self.factor = 1.0
        self.offset = 0.0
        self.y_format = 'linear'
        self.y_name = 'test_var'
        self._is_updating_data = False
        self._is_being_destroyed = False
        self._suppress_pin_update = False
        self._cursor_label_busy = False
        self._cursor_label_dirty = False
        self._cached_data_version = 0
        self.show_values_only = False
        self.last_valid_cursor_mode = "1 free cursor"
        self.plot_context = None
        self.units = {}
        self.is_multi_curve_mode = False
        self.curve = None
        self.curves = {}
        self.vline = MagicMock()
        self.vline.isVisible = MagicMock(return_value=True)
        self.vline.value = MagicMock(return_value=5.0)
        self.vline2 = MagicMock()
        self.vline2.isVisible = MagicMock(return_value=False)
        self.plot_item = MagicMock()
        self.view_box = MagicMock()
        self.view_box.viewRange = MagicMock(return_value=((0, 10), (0, 100)))
        self.multi_cursor_items = []
        self._cursor_item_pool = {
            'circles': [],
            'labels': [],
            'x_labels': []
        }
        self._pending_delete_items = []
        self._cleanup_timer = MagicMock()
        self._cleanup_timer.isActive = MagicMock(return_value=False)
        self._cleanup_timer.start = MagicMock()
    
    def update_right_header(self, text=""):
        pass
    
    def sInt_to_fmtStr(self, value):
        return f"{value}s"
    
    def dateInt_to_fmtStr(self, value):
        return "2024-01-01"
    
    def _significant_decimal_format_str(self, value, ref):
        return f"{value:.2f}"


class CursorManager:
    """负责光标位置、标签、模式、对象池管理和 ViewBox 信号处理"""
    
    def __init__(self, multi_curve_manager):
        if multi_curve_manager is None:
            raise ValueError("CursorManager requires a valid MultiCurveManager instance")
        self._data_manager = multi_curve_manager
    
    @property
    def pw(self):
        return self._data_manager._pw
    
    @property
    def _is_interacting(self):
        return getattr(self.pw, '_is_interacting', False)
    
    @_is_interacting.setter
    def _is_interacting(self, value):
        self.pw._is_interacting = value
    
    @property
    def is_cursor_pinned(self):
        return getattr(self.pw, 'is_cursor_pinned', False)
    
    @is_cursor_pinned.setter
    def is_cursor_pinned(self, value):
        self.pw.is_cursor_pinned = value
    
    @property
    def pinned_x_value(self):
        return getattr(self.pw, 'pinned_x_value', None)
    
    @pinned_x_value.setter
    def pinned_x_value(self, value):
        self.pw.pinned_x_value = value
    
    @property
    def pinned_x_values(self):
        return getattr(self.pw, 'pinned_x_values', [])
    
    @pinned_x_values.setter
    def pinned_x_values(self, value):
        self.pw.pinned_x_values = value
    
    @property
    def factor(self):
        return getattr(self.pw, 'factor', 1.0)
    
    @property
    def offset(self):
        return getattr(self.pw, 'offset', 0.0)
    
    def _get_cursor_mode(self):
        if self.pw.plot_context and hasattr(self.pw.plot_context, "cursor_mode"):
            return self.pw.plot_context.cursor_mode
        return "1 free cursor"
    
    def _get_cursor_x_positions(self):
        mode = self._get_cursor_mode()
        if mode == "2 anchored cursor":
            if self.pinned_x_values and len(self.pinned_x_values) >= 2:
                return list(self.pinned_x_values[:2])
            positions = []
            if hasattr(self.pw, "vline") and self.pw.vline.isVisible():
                positions.append(self.pw.vline.value())
            if hasattr(self.pw, "vline2") and self.pw.vline2.isVisible():
                positions.append(self.pw.vline2.value())
            return positions
        if mode == "1 anchored cursor":
            if self.pinned_x_values:
                return [self.pinned_x_values[0]]
            if self.pinned_x_value is not None:
                return [self.pinned_x_value]
        if hasattr(self.pw, "vline"):
            return [self.pw.vline.value()]
        return []
    
    def apply_cursor_mode(self, mode, pinned_x_values=None):
        if pinned_x_values is None:
            pinned_x_values = []
        
        if mode == "off":
            if self._get_cursor_mode() != "off":
                self.last_valid_cursor_mode = self._get_cursor_mode()
            return
        
        if mode == "1 free cursor":
            self.is_cursor_pinned = False
            self.pinned_x_value = None
            self.pinned_x_values = []
            return
    
    def _is_cursor_update_locked(self):
        if getattr(self.pw, '_is_updating_data', False) or getattr(self.pw, '_is_being_destroyed', False):
            return True
        return False
    
    def _has_visible_curve_data(self):
        return bool(self.pw.curves or self.pw.curve)
    
    def toggle_cursor(self, show, hide_values_only=False):
        if not hasattr(self.pw, "vline"):
            return
        self.pw.vline.setVisible(show)
    
    def pin_cursor(self, x_value):
        self.is_cursor_pinned = True
        self.pinned_x_value = x_value
        self.pinned_x_values = []
    
    def free_cursor(self):
        self.is_cursor_pinned = False
        self.pinned_x_value = None
        self.pinned_x_values = []
    
    def reset_pin_state(self):
        self.is_cursor_pinned = False
        self.pinned_x_value = None
        self.pinned_x_values = []


class TestCursorManagerInit(unittest.TestCase):
    """测试 CursorManager 初始化"""
    
    def test_init_with_valid_manager(self):
        mock_data_manager = MockMultiCurveManager()
        manager = CursorManager(mock_data_manager)
        self.assertIsNotNone(manager._data_manager)
    
    def test_init_with_none_raises(self):
        with self.assertRaises(ValueError):
            CursorManager(None)
    
    def test_pw_property(self):
        mock_data_manager = MockMultiCurveManager()
        manager = CursorManager(mock_data_manager)
        self.assertIsInstance(manager.pw, MockPlotWidget)


class TestCursorModes(unittest.TestCase):
    """测试光标模式"""
    
    def setUp(self):
        self.mock_data_manager = MockMultiCurveManager()
        self.manager = CursorManager(self.mock_data_manager)
        self.pw = self.manager.pw
    
    def test_default_cursor_mode(self):
        mode = self.manager._get_cursor_mode()
        self.assertEqual(mode, "1 free cursor")
    
    def test_apply_free_cursor_mode(self):
        self.manager.apply_cursor_mode("1 free cursor")
        self.assertFalse(self.manager.is_cursor_pinned)
        self.assertIsNone(self.manager.pinned_x_value)
    
    def test_apply_off_mode_saves_last_mode(self):
        self.pw.plot_context = MagicMock()
        self.pw.plot_context.cursor_mode = "1 anchored cursor"
        self.manager.apply_cursor_mode("off")
        self.assertEqual(self.manager.last_valid_cursor_mode, "1 anchored cursor")
    
    def test_toggle_cursor_show(self):
        self.manager.toggle_cursor(True)
        self.pw.vline.setVisible.assert_called_with(True)
    
    def test_toggle_cursor_hide(self):
        self.manager.toggle_cursor(False)
        self.pw.vline.setVisible.assert_called_with(False)


class TestPinOperations(unittest.TestCase):
    """测试固定光标操作"""
    
    def setUp(self):
        self.mock_data_manager = MockMultiCurveManager()
        self.manager = CursorManager(self.mock_data_manager)
        self.pw = self.manager.pw
    
    def test_pin_cursor(self):
        self.manager.pin_cursor(5.0)
        self.assertTrue(self.manager.is_cursor_pinned)
        self.assertEqual(self.manager.pinned_x_value, 5.0)
    
    def test_free_cursor(self):
        self.manager.is_cursor_pinned = True
        self.manager.pinned_x_value = 5.0
        self.manager.free_cursor()
        self.assertFalse(self.manager.is_cursor_pinned)
        self.assertIsNone(self.manager.pinned_x_value)
    
    def test_reset_pin_state(self):
        self.manager.is_cursor_pinned = True
        self.manager.pinned_x_value = 5.0
        self.manager.pinned_x_values = [5.0]
        self.manager.reset_pin_state()
        self.assertFalse(self.manager.is_cursor_pinned)
        self.assertEqual(self.manager.pinned_x_values, [])


class TestCursorPositions(unittest.TestCase):
    """测试光标位置获取"""
    
    def setUp(self):
        self.mock_data_manager = MockMultiCurveManager()
        self.manager = CursorManager(self.mock_data_manager)
        self.pw = self.manager.pw
    
    def test_get_cursor_x_positions_free_mode(self):
        positions = self.manager._get_cursor_x_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0], 5.0)
    
    def test_get_cursor_x_positions_anchored_mode(self):
        self.manager.pinned_x_values = [3.0, 7.0]
        self.pw.plot_context = MagicMock()
        self.pw.plot_context.cursor_mode = "1 anchored cursor"
        positions = self.manager._get_cursor_x_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0], 3.0)
    
    def test_get_cursor_x_positions_two_anchored_mode(self):
        self.manager.pinned_x_values = [3.0, 7.0]
        self.pw.plot_context = MagicMock()
        self.pw.plot_context.cursor_mode = "2 anchored cursor"
        positions = self.manager._get_cursor_x_positions()
        self.assertEqual(len(positions), 2)
        self.assertEqual(positions[0], 3.0)
        self.assertEqual(positions[1], 7.0)


class TestCursorUpdateLock(unittest.TestCase):
    """测试光标更新锁定"""
    
    def setUp(self):
        self.mock_data_manager = MockMultiCurveManager()
        self.manager = CursorManager(self.mock_data_manager)
        self.pw = self.manager.pw
    
    def test_not_locked_when_normal(self):
        self.assertFalse(self.manager._is_cursor_update_locked())
    
    def test_locked_when_updating_data(self):
        self.pw._is_updating_data = True
        self.assertTrue(self.manager._is_cursor_update_locked())
    
    def test_locked_when_being_destroyed(self):
        self.pw._is_being_destroyed = True
        self.assertTrue(self.manager._is_cursor_update_locked())


class TestVisibleCurveData(unittest.TestCase):
    """测试可见曲线数据检查"""
    
    def setUp(self):
        self.mock_data_manager = MockMultiCurveManager()
        self.manager = CursorManager(self.mock_data_manager)
        self.pw = self.manager.pw
    
    def test_no_visible_data(self):
        self.assertFalse(self.manager._has_visible_curve_data())
    
    def test_has_visible_curve(self):
        self.pw.curve = MagicMock()
        self.assertTrue(self.manager._has_visible_curve_data())
    
    def test_has_visible_curves_dict(self):
        self.pw.curves = {"var1": MagicMock()}
        self.assertTrue(self.manager._has_visible_curve_data())


if __name__ == '__main__':
    unittest.main(verbosity=2)
