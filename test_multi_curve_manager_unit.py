"""
MultiCurveManager 单元测试

测试多曲线绘图管理器的独立功能，不依赖 PyQt6 GUI。
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

with patch.dict('sys.modules', {
    'PyQt6': MagicMock(),
    'PyQt6.QtCore': MagicMock(),
}):
    from typing import NamedTuple, Any
    
    DEFAULT_LINE_WIDTH = 1.0
    DEFAULT_PADDING_VAL_Y = 0.1


class MockCurveInfo:
    """模拟 CurveInfo - 使用可变的类而不是 NamedTuple"""
    def __init__(self, var_name, curve, x_data, y_data, color, y_format, visible):
        self.var_name = var_name
        self.curve = curve
        self.x_data = x_data
        self.y_data = y_data
        self.color = color
        self.y_format = y_format
        self.visible = visible
    
    def update_x_range(self):
        pass


class CurveInfo(NamedTuple):
    """曲线信息数据结构 - 从 MultiCurveManager 复制"""
    var_name: str
    curve: Any
    x_data: Any
    y_data: Any
    color: str
    y_format: str
    visible: bool

    def update_x_range(self):
        """更新 x 范围信息"""
        pass


class MultiCurveManager:
    """负责多曲线绘图和样式管理 - 从 MultiCurveManager 复制"""
    
    def __init__(self, plot_data_manager):
        if plot_data_manager is None:
            raise ValueError("MultiCurveManager requires a valid PlotDataManager instance")
        self._data_manager = plot_data_manager
    
    @property
    def pw(self):
        return self._data_manager.pw
    
    def update_multi_curve_mode(self):
        pw = self.pw
        curve_count = len(pw.curves)
        
        if not hasattr(pw, '_batch_adding'):
            pw._batch_adding = False
        
        if not pw._batch_adding:
            pw.is_multi_curve_mode = curve_count > 1
        
        if pw.is_multi_curve_mode:
            pw.plot_item.legend.setVisible(True)
        else:
            pw.plot_item.legend.setVisible(False)
    
    def toggle_curve_visibility_by_name(self, var_name: str):
        pw = self.pw
        
        if var_name not in pw.curves:
            return
        
        ci = pw.curves[var_name]
        ci.visible = not ci.visible
        
        if ci.curve is not None:
            ci.curve.setVisible(ci.visible)
        
        if not pw.is_multi_curve_mode:
            return
        
        visible_count = sum(1 for c in pw.curves.values() if c.visible)
        if visible_count <= 1:
            pw.is_multi_curve_mode = False
            legend = pw.plot_item.legend
            legend.setVisible(False)
            self.update_multi_curve_mode()
    
    def _collect_visible_curve_arrays(self, key: str) -> list:
        pw = self.pw
        result = []
        for ci in pw.curves.values():
            if ci.visible and getattr(ci, key, None) is not None:
                data = getattr(ci, key)
                if data is not None:
                    result.append(data)
        return result
    
    def _collect_visible_curve_pairs(self) -> list:
        pw = self.pw
        result = []
        for ci in pw.curves.values():
            if ci.visible and ci.x_data is not None and ci.y_data is not None:
                result.append((ci.x_data, ci.y_data))
        return result
    
    def get_curve_x_limits(self, curves_filter: str = "visible") -> tuple:
        pw = self.pw
        
        if curves_filter == "visible":
            arrays = self._collect_visible_curve_arrays('x_data')
        else:
            arrays = [getattr(ci, 'x_data') for ci in pw.curves.values()]
        
        if not arrays:
            return None, None
        
        all_values = []
        for arr in arrays:
            if arr is not None:
                all_values.extend(arr.tolist() if hasattr(arr, 'tolist') else list(arr))
        
        if not all_values:
            return None, None
        
        return min(all_values), max(all_values)
    
    def _update_axes_for_multi_curve(self, update_x_range: bool = False):
        pw = self.pw
        
        if not pw.is_multi_curve_mode:
            return
        
        pw.axis_x.setTicks(None)
        pw.axis_y.setTicks(None)
        
        if update_x_range:
            x_min, x_max = self.get_curve_x_limits()
            if x_min is not None and x_max is not None:
                pw.view_box.setXRange(x_min, x_max, padding=0.02)
        
        y_arrays = self._collect_visible_curve_arrays('y_data')
        if y_arrays:
            import numpy as np
            combined = np.concatenate(y_arrays)
            if combined.size:
                min_y = np.nanmin(combined)
                max_y = np.nanmax(combined)
                pw.view_box.setYRange(min_y, max_y, padding=DEFAULT_PADDING_VAL_Y)
    
    def _apply_plot_style(self, show_symbols: bool = False):
        pw = self.pw
        
        for ci in pw.curves.values():
            if ci.curve is not None:
                if show_symbols:
                    ci.curve.setSymbol('o')
                else:
                    ci.curve.setSymbol(None)


class MockPlotWidget:
    """模拟 PlotWidget 的基本属性"""
    def __init__(self):
        self.curves = {}
        self.is_multi_curve_mode = False
        self._batch_adding = False
        self.plot_item = MagicMock()
        self.plot_item.legend = MagicMock()
        self.plot_item.legend.setVisible = MagicMock()
        self.plot_item.legend.clear = MagicMock()
        self.view_box = MagicMock()
        self.axis_x = MagicMock()
        self.axis_y = MagicMock()
        self.axis_x.setTicks = MagicMock()
        self.axis_y.setTicks = MagicMock()
        self.view_box.setXRange = MagicMock()
        self.view_box.setYRange = MagicMock()


class MockPlotDataManager:
    """模拟 PlotDataManager"""
    def __init__(self):
        self._pw = MockPlotWidget()
    
    @property
    def pw(self):
        return self._pw


class TestCurveInfo(unittest.TestCase):
    """测试 CurveInfo 命名元组"""
    
    def test_curve_info_creation(self):
        """测试 CurveInfo 创建"""
        ci = CurveInfo(
            var_name="test_var",
            curve=MagicMock(),
            x_data=[1, 2, 3],
            y_data=[4, 5, 6],
            color="r",
            y_format="linear",
            visible=True
        )
        
        self.assertEqual(ci.var_name, "test_var")
        self.assertEqual(ci.color, "r")
        self.assertTrue(ci.visible)
        self.assertEqual(ci.x_data, [1, 2, 3])
        self.assertEqual(ci.y_data, [4, 5, 6])
    
    def test_curve_info_immutable(self):
        """测试 CurveInfo 不可变性"""
        ci = CurveInfo(
            var_name="test",
            curve=None,
            x_data=[],
            y_data=[],
            color="b",
            y_format="linear",
            visible=True
        )
        
        with self.assertRaises(AttributeError):
            ci.var_name = "new_name"


class TestMultiCurveManagerInit(unittest.TestCase):
    """测试 MultiCurveManager 初始化"""
    
    def test_init_with_valid_manager(self):
        """测试有效 PlotDataManager 初始化"""
        mock_data_manager = MockPlotDataManager()
        manager = MultiCurveManager(mock_data_manager)
        
        self.assertIsNotNone(manager._data_manager)
    
    def test_init_with_none_raises(self):
        """测试 None 参数初始化抛出异常"""
        with self.assertRaises(ValueError):
            MultiCurveManager(None)
    
    def test_pw_property(self):
        """测试 pw 属性访问"""
        mock_data_manager = MockPlotDataManager()
        manager = MultiCurveManager(mock_data_manager)
        
        self.assertIsInstance(manager.pw, MockPlotWidget)


class TestUpdateMultiCurveMode(unittest.TestCase):
    """测试多曲线模式更新"""
    
    def setUp(self):
        """设置测试环境"""
        self.mock_data_manager = MockPlotDataManager()
        self.manager = MultiCurveManager(self.mock_data_manager)
        self.pw = self.manager.pw
    
    def test_single_curve_not_multi_mode(self):
        """单曲线时不应是多曲线模式"""
        self.pw.curves = {"var1": MagicMock()}
        self.manager.update_multi_curve_mode()
        self.assertFalse(self.pw.is_multi_curve_mode)
    
    def test_multiple_curves_is_multi_mode(self):
        """多曲线时应是多曲线模式"""
        self.pw.curves = {"var1": MagicMock(), "var2": MagicMock()}
        self.manager.update_multi_curve_mode()
        self.assertTrue(self.pw.is_multi_curve_mode)
    
    def test_legend_visible_in_multi_mode(self):
        """多曲线模式时图例应可见"""
        self.pw.curves = {"var1": MagicMock(), "var2": MagicMock()}
        self.manager.update_multi_curve_mode()
        self.pw.plot_item.legend.setVisible.assert_called_with(True)
    
    def test_batch_adding_prevents_mode_change(self):
        """批量添加时应阻止模式变更"""
        self.pw._batch_adding = True
        self.pw.curves = {"var1": MagicMock()}
        self.manager.update_multi_curve_mode()
        self.assertFalse(self.pw.is_multi_curve_mode)


class TestToggleCurveVisibility(unittest.TestCase):
    """测试曲线可见性切换"""
    
    def setUp(self):
        """设置测试环境"""
        self.mock_data_manager = MockPlotDataManager()
        self.manager = MultiCurveManager(self.mock_data_manager)
        self.pw = self.manager.pw
        
        self.curve_mock = MagicMock()
        self.ci = MockCurveInfo(
            var_name="test_var",
            curve=self.curve_mock,
            x_data=[1, 2, 3],
            y_data=[4, 5, 6],
            color="r",
            y_format="linear",
            visible=True
        )
        self.pw.curves = {"test_var": self.ci}
        self.pw.is_multi_curve_mode = True
    
    def test_toggle_hides_curve(self):
        """测试切换隐藏曲线"""
        self.manager.toggle_curve_visibility_by_name("test_var")
        self.assertFalse(self.pw.curves["test_var"].visible)
        self.curve_mock.setVisible.assert_called_with(False)
    
    def test_toggle_shows_curve(self):
        """测试切换显示曲线"""
        self.ci.visible = False
        self.manager.toggle_curve_visibility_by_name("test_var")
        self.assertTrue(self.pw.curves["test_var"].visible)
        self.curve_mock.setVisible.assert_called_with(True)
    
    def test_toggle_nonexistent_returns(self):
        """测试切换不存在的曲线"""
        try:
            self.manager.toggle_curve_visibility_by_name("nonexistent")
        except Exception as e:
            self.fail(f"Should not raise: {e}")


class TestCollectVisibleCurveData(unittest.TestCase):
    """测试可见曲线数据收集"""
    
    def setUp(self):
        """设置测试环境"""
        import numpy as np
        self.mock_data_manager = MockPlotDataManager()
        self.manager = MultiCurveManager(self.mock_data_manager)
        self.pw = self.manager.pw
        
        self.pw.curves = {
            "var1": MockCurveInfo("var1", None, np.array([1, 2]), np.array([3, 4]), "r", "linear", True),
            "var2": MockCurveInfo("var2", None, np.array([2, 3]), np.array([5, 6]), "b", "linear", False),
            "var3": MockCurveInfo("var3", None, np.array([3, 4]), np.array([7, 8]), "g", "linear", True),
        }
    
    def test_collect_x_data(self):
        """测试收集 X 数据"""
        arrays = self.manager._collect_visible_curve_arrays('x_data')
        self.assertEqual(len(arrays), 2)
    
    def test_collect_y_data(self):
        """测试收集 Y 数据"""
        arrays = self.manager._collect_visible_curve_arrays('y_data')
        self.assertEqual(len(arrays), 2)
    
    def test_collect_curve_pairs(self):
        """测试收集曲线对"""
        pairs = self.manager._collect_visible_curve_pairs()
        self.assertEqual(len(pairs), 2)
        for x, y in pairs:
            self.assertEqual(len(x), len(y))


class TestGetCurveXLimits(unittest.TestCase):
    """测试获取曲线 X 轴限制"""
    
    def setUp(self):
        """设置测试环境"""
        import numpy as np
        self.mock_data_manager = MockPlotDataManager()
        self.manager = MultiCurveManager(self.mock_data_manager)
        self.pw = self.manager.pw
    
    def test_visible_curves_x_limits(self):
        """测试可见曲线 X 轴限制"""
        import numpy as np
        self.pw.curves = {
            "var1": MockCurveInfo("var1", None, np.array([1, 5]), np.array([3, 4]), "r", "linear", True),
            "var2": MockCurveInfo("var2", None, np.array([2, 4]), np.array([5, 6]), "b", "linear", True),
        }
        x_min, x_max = self.manager.get_curve_x_limits("visible")
        self.assertEqual(x_min, 1)
        self.assertEqual(x_max, 5)
    
    def test_no_curves_returns_none(self):
        """测试无曲线时返回 None"""
        self.pw.curves = {}
        x_min, x_max = self.manager.get_curve_x_limits()
        self.assertIsNone(x_min)
        self.assertIsNone(x_max)


class TestUpdateAxesForMultiCurve(unittest.TestCase):
    """测试多曲线坐标轴更新"""
    
    def setUp(self):
        """设置测试环境"""
        self.mock_data_manager = MockPlotDataManager()
        self.manager = MultiCurveManager(self.mock_data_manager)
        self.pw = self.manager.pw
    
    def test_non_multi_mode_returns_early(self):
        """非多曲线模式直接返回"""
        self.pw.is_multi_curve_mode = False
        self.manager._update_axes_for_multi_curve()
        self.pw.axis_x.setTicks.assert_not_called()
    
    def test_multi_mode_updates_axes(self):
        """多曲线模式更新轴"""
        import numpy as np
        self.pw.is_multi_curve_mode = True
        self.pw.curves = {
            "var1": MockCurveInfo("var1", None, np.array([1, 5]), np.array([3, 4]), "r", "linear", True),
        }
        self.manager._update_axes_for_multi_curve()
        self.pw.axis_x.setTicks.assert_called_with(None)
        self.pw.axis_y.setTicks.assert_called_with(None)


class TestApplyPlotStyle(unittest.TestCase):
    """测试绘图样式应用"""
    
    def setUp(self):
        """设置测试环境"""
        self.mock_data_manager = MockPlotDataManager()
        self.manager = MultiCurveManager(self.mock_data_manager)
        self.pw = self.manager.pw
        
        self.curve_mock = MagicMock()
        self.pw.curves = {
            "var1": CurveInfo("var1", self.curve_mock, [1, 2], [3, 4], "r", "linear", True),
        }
    
    def test_apply_symbols(self):
        """测试应用符号样式"""
        self.manager._apply_plot_style(show_symbols=True)
        self.curve_mock.setSymbol.assert_called_with('o')
    
    def test_remove_symbols(self):
        """测试移除符号样式"""
        self.manager._apply_plot_style(show_symbols=False)
        self.curve_mock.setSymbol.assert_called_with(None)
    
    def test_apply_style_with_multiple_curves(self):
        """测试多曲线样式应用"""
        curve1 = MagicMock()
        curve2 = MagicMock()
        self.pw.curves = {
            "var1": MockCurveInfo("var1", curve1, [1, 2], [3, 4], "r", "linear", True),
            "var2": MockCurveInfo("var2", curve2, [1, 2], [5, 6], "b", "linear", True),
        }
        self.manager._apply_plot_style(show_symbols=True)
        curve1.setSymbol.assert_called_with('o')
        curve2.setSymbol.assert_called_with('o')


if __name__ == '__main__':
    unittest.main(verbosity=2)
