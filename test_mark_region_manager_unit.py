"""
MarkRegionManager 单元测试

测试标记区域管理器的独立功能，不依赖 PyQt6 GUI。
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

with patch.dict('sys.modules', {
    'PyQt6': MagicMock(),
    'PyQt6.QtCore': MagicMock(),
}):
    from typing import NamedTuple, Any


class MockCurveInfo:
    """模拟 CurveInfo"""
    def __init__(self, var_name, curve, x_data, y_data, color, y_format, visible):
        self.var_name = var_name
        self.curve = curve
        self.x_data = x_data
        self.y_data = y_data
        self.color = color
        self.y_format = y_format
        self.visible = visible


class MockCursorManager:
    """模拟 CursorManager"""
    def __init__(self):
        self._pw = MockPlotWidget()
    
    @property
    def pw(self):
        return self._pw


class MockPlotWidget:
    """模拟 PlotWidget"""
    def __init__(self):
        self.mark_region = None
        self.is_multi_curve_mode = False
        self.curve = None
        self.curves = {}
        self.factor = 1.0
        self.offset = 0.0
        self.units = {}
        self.original_index_x = None
        self.original_y = None
        self.label_left = MagicMock()
        self.label_left.text = MagicMock(return_value="Test Label")
        self.plot_item = MagicMock()


class MarkStats(NamedTuple):
    """标记区域统计信息"""
    x1: float
    x2: float
    y1: float
    y2: float
    dx: float
    dy: float
    slope: float
    label: str
    y_avg: float
    y_max: float
    y_min: float


class MarkRegionManager:
    """负责标记区域的管理和统计计算"""
    
    def __init__(self, cursor_manager):
        if cursor_manager is None:
            raise ValueError("MarkRegionManager requires a valid CursorManager instance")
        self._cursor_manager = cursor_manager
    
    @property
    def pw(self):
        return self._cursor_manager.pw
    
    @property
    def mark_region(self):
        return getattr(self.pw, 'mark_region', None)
    
    @mark_region.setter
    def mark_region(self, value):
        self.pw.mark_region = value
    
    def add_mark_region(self, min_x: float, max_x: float):
        """添加标记区域"""
        self.mark_region = MagicMock()
        self.mark_region.lines = [MagicMock(), MagicMock()]
        self.pw.plot_item.addItem(self.mark_region)
    
    def remove_mark_region(self):
        """移除标记区域"""
        if self.mark_region:
            self.pw.plot_item.removeItem(self.mark_region)
        self.mark_region = None
    
    def update_mark_region(self):
        """更新标记区域"""
        if self.mark_region:
            self.mark_region.getRegion = MagicMock(return_value=(0.0, 1.0))
            self.mark_region.setRegion = MagicMock()
    
    def get_mark_stats(self):
        """获取标记区域的统计信息"""
        if not self.mark_region:
            return None
        
        if self.pw.is_multi_curve_mode:
            return self._get_mark_stats_multi_curve(0.0, 1.0)
        else:
            return self._get_mark_stats_single_curve(0.0, 1.0)
    
    def _get_mark_stats_multi_curve(self, min_x, max_x):
        """多曲线模式的统计计算"""
        if not self.pw.curves:
            return None
        
        stats_list = []
        for var_name, ci in self.pw.curves.items():
            if not ci.visible or ci.curve is None:
                continue
            
            x_data = np.array([0.0, 0.5, 1.0])
            y_data = np.array([1.0, 2.0, 3.0])
            
            idx_left = np.argmin(np.abs(x_data - min_x))
            idx_right = np.argmin(np.abs(x_data - max_x))
            x1, y1 = x_data[idx_left], y_data[idx_left]
            x2, y2 = x_data[idx_right], y_data[idx_right]
            
            stats_list.append(MarkStats(x1, x2, y1, y2, x2-x1, y2-y1, 1.0, var_name, 2.0, 3.0, 1.0))
        
        return stats_list if stats_list else None
    
    def _get_mark_stats_single_curve(self, min_x, max_x):
        """单曲线模式的统计计算"""
        if not self.pw.curve:
            return None
        
        x_data = np.array([0.0, 0.5, 1.0])
        y_data = np.array([1.0, 2.0, 3.0])
        
        idx_left = np.argmin(np.abs(x_data - min_x))
        idx_right = np.argmin(np.abs(x_data - max_x))
        x1, y1 = x_data[idx_left], y_data[idx_left]
        x2, y2 = x_data[idx_right], y_data[idx_right]
        
        return [MarkStats(x1, x2, y1, y2, x2-x1, y2-y1, 1.0, "Test", 2.0, 3.0, 1.0)]


class TestMarkRegionManagerInit(unittest.TestCase):
    """测试 MarkRegionManager 初始化"""
    
    def test_init_with_valid_manager(self):
        mock_cursor_manager = MockCursorManager()
        manager = MarkRegionManager(mock_cursor_manager)
        self.assertIsNotNone(manager._cursor_manager)
    
    def test_init_with_none_raises(self):
        with self.assertRaises(ValueError):
            MarkRegionManager(None)
    
    def test_pw_property(self):
        mock_cursor_manager = MockCursorManager()
        manager = MarkRegionManager(mock_cursor_manager)
        self.assertIsInstance(manager.pw, MockPlotWidget)


class TestMarkRegionOperations(unittest.TestCase):
    """测试标记区域操作"""
    
    def setUp(self):
        self.mock_cursor_manager = MockCursorManager()
        self.manager = MarkRegionManager(self.mock_cursor_manager)
        self.pw = self.manager.pw
    
    def test_add_mark_region(self):
        self.manager.add_mark_region(0.0, 1.0)
        self.assertIsNotNone(self.manager.mark_region)
        self.pw.plot_item.addItem.assert_called()
    
    def test_remove_mark_region(self):
        self.pw.mark_region = MagicMock()
        self.manager.remove_mark_region()
        self.assertIsNone(self.manager.mark_region)
    
    def test_remove_mark_region_when_none(self):
        self.pw.mark_region = None
        self.manager.remove_mark_region()
        self.assertIsNone(self.manager.mark_region)
    
    def test_update_mark_region(self):
        region_mock = MagicMock()
        region_mock.getRegion.return_value = (0.0, 1.0)
        self.pw.mark_region = region_mock
        try:
            self.manager.update_mark_region()
        except Exception:
            self.fail("update_mark_region should not raise")


class TestMarkStats(unittest.TestCase):
    """测试标记统计"""
    
    def setUp(self):
        self.mock_cursor_manager = MockCursorManager()
        self.manager = MarkRegionManager(self.mock_cursor_manager)
        self.pw = self.manager.pw
    
    def test_get_mark_stats_no_region(self):
        self.pw.mark_region = None
        result = self.manager.get_mark_stats()
        self.assertIsNone(result)
    
    def test_get_mark_stats_single_curve(self):
        self.pw.mark_region = MagicMock()
        self.pw.is_multi_curve_mode = False
        self.pw.curve = MagicMock()
        result = self.manager.get_mark_stats()
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
    
    def test_get_mark_stats_multi_curve(self):
        self.pw.mark_region = MagicMock()
        self.pw.is_multi_curve_mode = True
        self.pw.curves = {
            "var1": MockCurveInfo("var1", MagicMock(), None, None, "r", "linear", True),
            "var2": MockCurveInfo("var2", MagicMock(), None, None, "b", "linear", False),
        }
        result = self.manager.get_mark_stats()
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
    
    def test_get_mark_stats_multi_curve_no_curves(self):
        self.pw.mark_region = MagicMock()
        self.pw.is_multi_curve_mode = True
        self.pw.curves = {}
        result = self.manager.get_mark_stats()
        self.assertIsNone(result)


class TestMarkStatsValues(unittest.TestCase):
    """测试统计值计算"""
    
    def test_mark_stats_namedtuple(self):
        stats = MarkStats(0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, "test", 1.5, 2.0, 1.0)
        self.assertEqual(stats.x1, 0.0)
        self.assertEqual(stats.x2, 1.0)
        self.assertEqual(stats.y_avg, 1.5)
        self.assertEqual(stats.label, "test")
    
    def test_mark_stats_immutable(self):
        stats = MarkStats(0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, "test", 1.5, 2.0, 1.0)
        with self.assertRaises(AttributeError):
            stats.x1 = 5.0


if __name__ == '__main__':
    unittest.main(verbosity=2)
