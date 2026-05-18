#!/usr/bin/env python3
"""AxisManager 单元测试 - 无需 GUI 环境"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

def test_axis_manager_import():
    """测试 AxisManager 可以正确导入"""
    print("=" * 60)
    print("测试 1: AxisManager 导入测试")
    print("=" * 60)

    try:
        from src.ui.widgets.axis_manager import AxisManager
        print("✓ AxisManager 成功导入")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_axis_manager_initialization():
    """测试 AxisManager 正确初始化"""
    print("\n" + "=" * 60)
    print("测试 2: AxisManager 初始化测试")
    print("=" * 60)

    from src.ui.widgets.axis_manager import AxisManager

    mock_ui_manager = MagicMock()
    mock_ui_manager.pw = MagicMock()

    manager = AxisManager(mock_ui_manager)

    assert hasattr(manager, '_ui_manager'), "应该有 _ui_manager 属性"
    assert manager._ui_manager is mock_ui_manager, "_ui_manager 应该指向传入的 ui_manager"
    print("✓ _ui_manager 属性正确设置")

    return True

def test_axis_manager_requires_ui_manager():
    """测试 AxisManager 需要有效的 PlotUIManager"""
    print("\n" + "=" * 60)
    print("测试 3: AxisManager 初始化参数验证")
    print("=" * 60)

    from src.ui.widgets.axis_manager import AxisManager

    try:
        manager = AxisManager(None)
        print("✗ 应该抛出 ValueError")
        return False
    except ValueError as e:
        print(f"✓ 正确抛出 ValueError: {e}")
        return True

def test_axis_manager_methods():
    """测试 AxisManager 的方法存在"""
    print("\n" + "=" * 60)
    print("测试 4: AxisManager 方法存在性测试")
    print("=" * 60)

    from src.ui.widgets.axis_manager import AxisManager

    mock_ui_manager = MagicMock()
    mock_ui_manager.pw = MagicMock()
    manager = AxisManager(mock_ui_manager)

    expected_methods = [
        'update_x_axis_label',
        'auto_range',
        'auto_y_in_x_range',
        'set_xrange_with_link_handling',
        '_get_safe_x_range',
        '_get_min_x_range_value',
        '_set_x_limits_with_min_range',
        '_set_min_x_range',
        '_recalc_max_point_density',
        '_set_safe_y_range',
        '_setup_plot_axes',
        '_reset_plot_limits',
        '_set_vline_bounds',
    ]

    all_exist = True
    for method in expected_methods:
        if hasattr(manager, method):
            print(f"✓ {method} 存在")
        else:
            print(f"✗ {method} 不存在")
            all_exist = False

    return all_exist

def test_get_safe_x_range():
    """测试 _get_safe_x_range 方法"""
    print("\n" + "=" * 60)
    print("测试 5: _get_safe_x_range 测试")
    print("=" * 60)

    from src.ui.widgets.axis_manager import AxisManager

    mock_ui_manager = MagicMock()
    mock_ui_manager.pw = MagicMock()
    mock_ui_manager.pw.factor = 2.0

    manager = AxisManager(mock_ui_manager)

    result = manager._get_safe_x_range(5.0, 5.0)
    assert result[0] == 4.0, f"min_x 应该扩展为 4.0，实际为 {result[0]}"
    assert result[1] == 6.0, f"max_x 应该扩展为 6.0，实际为 {result[1]}"
    print(f"✓ 单点扩展: (5.0, 5.0) -> {result}")

    result = manager._get_safe_x_range(0.0, 10.0)
    assert result == (0.0, 10.0), "正常范围不应该改变"
    print(f"✓ 正常范围: (0.0, 10.0) -> {result}")

    return True

def test_get_min_x_range_value():
    """测试 _get_min_x_range_value 方法"""
    print("\n" + "=" * 60)
    print("测试 6: _get_min_x_range_value 测试")
    print("=" * 60)

    from src.ui.widgets.axis_manager import AxisManager

    mock_ui_manager = MagicMock()
    mock_ui_manager.pw = MagicMock()

    mock_ui_manager.pw.plot_context = None
    manager = AxisManager(mock_ui_manager)
    result = manager._get_min_x_range_value()
    assert result == 1.0, f"无 plot_context 时应该返回 1.0，实际为 {result}"
    print(f"✓ 无 plot_context 时返回: {result}")

    mock_ui_manager.pw.plot_context = MagicMock()
    mock_ui_manager.pw.plot_context._global_max_density = 0.0
    result = manager._get_min_x_range_value()
    assert result == 1.0, f"density=0 时应该返回 1.0，实际为 {result}"
    print(f"✓ density=0 时返回: {result}")

    mock_ui_manager.pw.plot_context._global_max_density = 100.0
    result = manager._get_min_x_range_value()
    expected = 3.0 / 100.0  # MIN_INDEX_LENGTH = 3
    assert abs(result - expected) < 0.01, f"density=100 时应该返回 ~{expected}，实际为 {result}"
    print(f"✓ density=100 时返回: {result}")

    return True

def test_set_safe_y_range():
    """测试 _set_safe_y_range 方法"""
    print("\n" + "=" * 60)
    print("测试 7: _set_safe_y_range 测试")
    print("=" * 60)

    from src.ui.widgets.axis_manager import AxisManager

    mock_ui_manager = MagicMock()
    mock_pw = MagicMock()
    mock_ui_manager.pw = mock_pw
    mock_pw.plot_item = MagicMock()
    mock_pw.view_box = MagicMock()

    manager = AxisManager(mock_ui_manager)

    manager._set_safe_y_range(0.0, 100.0, set_limits=True)

    assert mock_pw.plot_item.setLimits.called, "应该调用 setLimits"
    assert mock_pw.view_box.setYRange.called, "应该调用 setYRange"
    print("✓ _set_safe_y_range 正确调用了底层方法")

    return True

def test_reset_plot_limits():
    """测试 _reset_plot_limits 方法"""
    print("\n" + "=" * 60)
    print("测试 8: _reset_plot_limits 测试")
    print("=" * 60)

    from src.ui.widgets.axis_manager import AxisManager

    mock_ui_manager = MagicMock()
    mock_pw = MagicMock()
    mock_ui_manager.pw = mock_pw
    mock_pw.plot_item = MagicMock()
    mock_pw.view_box = MagicMock()

    manager = AxisManager(mock_ui_manager)

    manager._reset_plot_limits()

    assert mock_pw.plot_item.setLimits.called, "应该调用 setLimits"
    assert mock_pw.view_box.setYRange.called, "应该调用 setYRange"
    print("✓ _reset_plot_limits 正确调用了底层方法")

    return True

def test_set_vline_bounds():
    """测试 _set_vline_bounds 方法"""
    print("\n" + "=" * 60)
    print("测试 9: _set_vline_bounds 测试")
    print("=" * 60)

    from src.ui.widgets.axis_manager import AxisManager

    mock_ui_manager = MagicMock()
    mock_pw = MagicMock()
    mock_ui_manager.pw = mock_pw
    mock_pw.vline = MagicMock()
    mock_pw.vline2 = MagicMock()

    manager = AxisManager(mock_ui_manager)

    bounds = [0.0, 100.0]
    manager._set_vline_bounds(bounds)

    assert mock_pw.vline.setBounds.called, "vline.setBounds 应该被调用"
    assert mock_pw.vline2.setBounds.called, "vline2.setBounds 应该被调用"
    print("✓ _set_vline_bounds 正确设置了两个 vline 的边界")

    return True

def test_set_vline_bounds_partial():
    """测试 _set_vline_bounds 方法（部分 vline 存在）"""
    print("\n" + "=" * 60)
    print("测试 10: _set_vline_bounds 部分存在测试")
    print("=" * 60)

    from src.ui.widgets.axis_manager import AxisManager

    mock_ui_manager = MagicMock()
    mock_pw = MagicMock()
    mock_ui_manager.pw = mock_pw
    mock_pw.vline = MagicMock()
    del mock_pw.vline2

    manager = AxisManager(mock_ui_manager)

    bounds = [0.0, 100.0]
    try:
        manager._set_vline_bounds(bounds)
        assert mock_pw.vline.setBounds.called, "vline.setBounds 应该被调用"
        print("✓ _set_vline_bounds 正确处理部分 vline 缺失")
        return True
    except AttributeError:
        print("✗ _set_vline_bounds 没有正确处理 vline2 缺失")
        return False

def test_syntax_check():
    """语法检查测试"""
    print("\n" + "=" * 60)
    print("测试 11: 语法检查")
    print("=" * 60)

    import py_compile

    files_to_check = [
        'src/ui/widgets/axis_manager.py',
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
    print("AxisManager 单元测试套件")
    print("=" * 60)

    tests = [
        test_axis_manager_import,
        test_axis_manager_initialization,
        test_axis_manager_requires_ui_manager,
        test_axis_manager_methods,
        test_get_safe_x_range,
        test_get_min_x_range_value,
        test_set_safe_y_range,
        test_reset_plot_limits,
        test_set_vline_bounds,
        test_set_vline_bounds_partial,
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
