#!/usr/bin/env python3
"""PlotDataManager 单元测试 - 无需 GUI 环境"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

def test_plot_data_manager_import():
    """测试 PlotDataManager 可以正确导入"""
    print("=" * 60)
    print("测试 1: PlotDataManager 导入测试")
    print("=" * 60)

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("plot_data_manager", "src/ui/widgets/plot_data_manager.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        PlotDataManager = module.PlotDataManager
        print("✓ PlotDataManager 成功导入")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_plot_data_manager_initialization():
    """测试 PlotDataManager 正确初始化"""
    print("\n" + "=" * 60)
    print("测试 2: PlotDataManager 初始化测试")
    print("=" * 60)

    from src.ui.widgets.plot_data_manager import PlotDataManager

    mock_axis_manager = MagicMock()
    mock_axis_manager.pw = MagicMock()

    manager = PlotDataManager(mock_axis_manager)

    assert hasattr(manager, '_axis_manager'), "应该有 _axis_manager 属性"
    assert manager._axis_manager is mock_axis_manager, "_axis_manager 应该指向传入的 axis_manager"
    print("✓ _axis_manager 属性正确设置")

    return True

def test_plot_data_manager_requires_axis_manager():
    """测试 PlotDataManager 需要有效的 AxisManager"""
    print("\n" + "=" * 60)
    print("测试 3: PlotDataManager 初始化参数验证")
    print("=" * 60)

    from src.ui.widgets.plot_data_manager import PlotDataManager

    try:
        manager = PlotDataManager(None)
        print("✗ 应该抛出 ValueError")
        return False
    except ValueError as e:
        print(f"✓ 正确抛出 ValueError: {e}")
        return True

def test_plot_data_manager_methods():
    """测试 PlotDataManager 的方法存在"""
    print("\n" + "=" * 60)
    print("测试 4: PlotDataManager 方法存在性测试")
    print("=" * 60)

    from src.ui.widgets.plot_data_manager import PlotDataManager

    mock_axis_manager = MagicMock()
    mock_axis_manager.pw = MagicMock()
    manager = PlotDataManager(mock_axis_manager)

    expected_methods = [
        'plot_variable',
        '_validate_plot_data',
        '_get_x_data_for_variable',
        '_prepare_plot_data',
        '_compute_valid_min_max',
        '_get_y_range_in_x_window',
        'handle_single_point_limits',
        'clear_value_cache',
        'datetime_to_unix_seconds',
        'get_value_from_name',
        'update_time_correction',
        '_safe_clear_plot_items',
        '_clear_plot_data',
        'clear_plot_item',
        'reset_plot',
    ]

    all_exist = True
    for method in expected_methods:
        if hasattr(manager, method):
            print(f"✓ {method} 存在")
        else:
            print(f"✗ {method} 不存在")
            all_exist = False

    return all_exist

def test_validate_plot_data():
    """测试 _validate_plot_data 方法"""
    print("\n" + "=" * 60)
    print("测试 5: _validate_plot_data 测试")
    print("=" * 60)

    from src.ui.widgets.plot_data_manager import PlotDataManager

    mock_axis_manager = MagicMock()
    mock_pw = MagicMock()
    mock_axis_manager.pw = mock_pw

    mock_pw.plot_context = None
    mock_pw.data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

    manager = PlotDataManager(mock_axis_manager)

    is_valid, msg = manager._validate_plot_data('x')
    assert is_valid == False, "无 plot_context 时应该验证失败"
    print("✓ 无 plot_context 时正确返回验证失败")

    mock_pw.plot_context = MagicMock()
    mock_pw.plot_context.loader = MagicMock()
    is_valid, msg = manager._validate_plot_data('x')
    assert is_valid == True, "有 loader 时应该验证成功"
    print("✓ 有 loader 时正确返回验证成功")

    mock_pw.plot_context = None
    mock_pw.data = pd.DataFrame({'x': [1, 2, 3]})
    is_valid, msg = manager._validate_plot_data('y')
    assert is_valid == False, "变量不存在时应该验证失败"
    print("✓ 变量不存在时正确返回验证失败")

    return True

def test_compute_valid_min_max():
    """测试 _compute_valid_min_max 方法"""
    print("\n" + "=" * 60)
    print("测试 6: _compute_valid_min_max 测试")
    print("=" * 60)

    from src.ui.widgets.plot_data_manager import PlotDataManager

    mock_axis_manager = MagicMock()
    mock_axis_manager.pw = MagicMock()
    manager = PlotDataManager(mock_axis_manager)

    result = manager._compute_valid_min_max([1, 2, 3])
    assert result == (1.0, 3.0), f"应该返回 (1.0, 3.0)，实际为 {result}"
    print(f"✓ [1, 2, 3] -> {result}")

    result = manager._compute_valid_min_max([1.5, 2.5, np.nan, 3.5])
    assert result == (1.5, 3.5), f"应该返回 (1.5, 3.5)，实际为 {result}"
    print(f"✓ [1.5, 2.5, NaN, 3.5] -> {result}")

    result = manager._compute_valid_min_max(None)
    assert result == (None, None), f"None 应该返回 (None, None)，实际为 {result}"
    print(f"✓ None -> {result}")

    result = manager._compute_valid_min_max([])
    assert result == (None, None), f"空列表应该返回 (None, None)，实际为 {result}"
    print(f"✓ [] -> {result}")

    return True

def test_get_x_data_for_variable():
    """测试 _get_x_data_for_variable 方法"""
    print("\n" + "=" * 60)
    print("测试 7: _get_x_data_for_variable 测试")
    print("=" * 60)

    from src.ui.widgets.plot_data_manager import PlotDataManager

    mock_axis_manager = MagicMock()
    mock_pw = MagicMock()
    mock_axis_manager.pw = mock_pw

    mock_pw.time_values = None
    manager = PlotDataManager(mock_axis_manager)

    result = manager._get_x_data_for_variable(5)
    expected = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    assert np.array_equal(result, expected), f"应该返回 {expected}，实际为 {result}"
    print(f"✓ 无 time_values 时正确生成索引数组")

    mock_pw.time_values = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    result = manager._get_x_data_for_variable(5)
    expected = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    assert np.allclose(result, expected), f"应该返回 {expected}，实际为 {result}"
    print(f"✓ 有 time_values 时正确使用时间值")

    return True

def test_handle_single_point_limits():
    """测试 handle_single_point_limits 方法"""
    print("\n" + "=" * 60)
    print("测试 8: handle_single_point_limits 测试")
    print("=" * 60)

    from src.ui.widgets.plot_data_manager import PlotDataManager

    mock_axis_manager = MagicMock()
    mock_pw = MagicMock()
    mock_axis_manager.pw = mock_pw
    mock_pw.factor = 1.0

    manager = PlotDataManager(mock_axis_manager)

    x_values = np.array([5.0])
    y_values = np.array([10.0])
    result = manager.handle_single_point_limits(x_values, y_values)
    assert result is not None, "单点应该返回特殊限制"
    print(f"✓ 单点情况: {result}")

    x_values = np.array([5.0, 5.0, 5.0])
    y_values = np.array([10.0, 20.0, 30.0])
    result = manager.handle_single_point_limits(x_values, y_values)
    assert result is not None, "所有 x 相同应该返回特殊限制"
    print(f"✓ 所有 x 相同情况: {result}")

    x_values = np.array([1.0, 2.0, 3.0])
    y_values = np.array([10.0, 20.0, 30.0])
    result = manager.handle_single_point_limits(x_values, y_values)
    assert result is None, "正常数据应该返回 None"
    print("✓ 正常数据返回 None")

    return True

def test_get_safe_x_range_via_axis_manager():
    """测试 _get_safe_x_range 通过 axis_manager 代理"""
    print("\n" + "=" * 60)
    print("测试 9: _get_safe_x_range 代理测试")
    print("=" * 60)

    from src.ui.widgets.plot_data_manager import PlotDataManager

    mock_axis_manager = MagicMock()
    mock_axis_manager._get_safe_x_range.return_value = (4.0, 6.0)
    mock_axis_manager.pw = MagicMock()
    mock_axis_manager.pw.factor = 2.0

    manager = PlotDataManager(mock_axis_manager)

    result = manager._get_safe_x_range(5.0, 5.0)
    assert result == (4.0, 6.0), f"应该返回 (4.0, 6.0)，实际为 {result}"
    print(f"✓ _get_safe_x_range 正确代理到 axis_manager")

    return True

def test_datetime_to_unix_seconds():
    """测试 datetime_to_unix_seconds 方法"""
    print("\n" + "=" * 60)
    print("测试 10: datetime_to_unix_seconds 测试")
    print("=" * 60)

    from src.ui.widgets.plot_data_manager import PlotDataManager

    mock_axis_manager = MagicMock()
    mock_axis_manager.pw = MagicMock()

    manager = PlotDataManager(mock_axis_manager)

    ts = pd.Timestamp('2024-01-01 12:00:00')
    result = manager.datetime_to_unix_seconds(pd.Series([ts]))[0]
    expected = ts.timestamp()
    assert abs(result - expected) < 1, f"应该返回 {expected}，实际为 {result}"
    print(f"✓ datetime 转换正确")

    return True

def test_syntax_check():
    """语法检查测试"""
    print("\n" + "=" * 60)
    print("测试 11: 语法检查")
    print("=" * 60)

    import py_compile

    files_to_check = [
        'src/ui/widgets/plot_data_manager.py',
        'csv_plot_pyqt6.py',
    ]

    all_passed = True
    for file in files_to_check:
        filepath = Path(file)
        if filepath.exists():
            try:
                py_compile.compile(str(filepath), doraise=True)
                print(f"✓ {file} 语法正确")
            except py_compile.PyCompileError as e:
                print(f"✗ {file} 语法错误: {e}")
                all_passed = False
        else:
            print(f"⚠ {file} 不存在")

    return all_passed

def run_all_tests():
    print("\n" + "=" * 60)
    print("PlotDataManager 单元测试套件")
    print("=" * 60)

    tests = [
        test_plot_data_manager_import,
        test_plot_data_manager_initialization,
        test_plot_data_manager_requires_axis_manager,
        test_plot_data_manager_methods,
        test_validate_plot_data,
        test_compute_valid_min_max,
        test_get_x_data_for_variable,
        test_handle_single_point_limits,
        test_get_safe_x_range_via_axis_manager,
        test_datetime_to_unix_seconds,
        test_syntax_check,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
                print(f"✓ {test.__name__} 通过")
            else:
                failed += 1
                print(f"✗ {test.__name__} 失败")
        except Exception as e:
            import traceback
            print(f"✗ {test.__name__} 异常: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
