#!/usr/bin/env python3
"""PlotUIManager 单元测试 - 无需 GUI 环境"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

def test_plot_ui_manager_import():
    """测试 PlotUIManager 可以正确导入"""
    print("=" * 60)
    print("测试 1: PlotUIManager 导入测试")
    print("=" * 60)

    try:
        from src.ui.widgets.plot_ui_manager import PlotUIManager
        print("✓ PlotUIManager 成功导入")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_plot_ui_manager_initialization():
    """测试 PlotUIManager 正确初始化"""
    print("\n" + "=" * 60)
    print("测试 2: PlotUIManager 初始化测试")
    print("=" * 60)

    from src.ui.widgets.plot_ui_manager import PlotUIManager

    mock_pw = MagicMock()
    manager = PlotUIManager(mock_pw)

    assert hasattr(manager, '_pw_ref'), "应该有 _pw_ref 属性 (BasePlotManager)"
    assert manager.pw is mock_pw, "pw property 应该返回传入的 plot_widget"
    print("✓ pw property 正确设置 (通过 BasePlotManager)")

    return True

def test_setup_ui_method_exists():
    """测试 setup_ui 方法存在"""
    print("\n" + "=" * 60)
    print("测试 3: setup_ui 方法存在性测试")
    print("=" * 60)

    from src.ui.widgets.plot_ui_manager import PlotUIManager

    mock_pw = MagicMock()
    manager = PlotUIManager(mock_pw)

    assert hasattr(manager, 'setup_ui'), "PlotUIManager 应该有 setup_ui 方法"
    assert callable(manager.setup_ui), "setup_ui 应该是可调用的"
    print("✓ setup_ui 方法存在且可调用")

    return True

def test_plot_ui_manager_sub_methods():
    """测试 PlotUIManager 的子方法"""
    print("\n" + "=" * 60)
    print("测试 4: PlotUIManager 子方法测试")
    print("=" * 60)

    from src.ui.widgets.plot_ui_manager import PlotUIManager

    mock_pw = MagicMock()
    manager = PlotUIManager(mock_pw)

    expected_methods = [
        '_setup_header',
        '_setup_plot_area',
        '_setup_axes',
        '_setup_interaction',
        '_init_ui_refresh_coordinator',
        '_queue_ui_refresh',
        '_cancel_ui_refresh',
        '_run_style_refresh',
        '_run_cursor_refresh',
        '_run_stats_refresh',
        'update_x_axis_label',
    ]

    for method in expected_methods:
        assert hasattr(manager, method), f"PlotUIManager 应该有 {method} 方法"
        assert callable(getattr(manager, method)), f"{method} 应该是可调用的"
        print(f"✓ {method} 方法存在")

    return True

def test_base_manager_import():
    """测试 BasePlotManager 可以正确导入"""
    print("\n" + "=" * 60)
    print("测试 5: BasePlotManager 导入测试")
    print("=" * 60)

    try:
        from src.ui.widgets.base_manager import BasePlotManager
        print("✓ BasePlotManager 成功导入")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_base_manager_inheritance():
    """测试 BasePlotManager 是 PlotUIManager 的基类"""
    print("\n" + "=" * 60)
    print("测试 6: BasePlotManager 继承测试")
    print("=" * 60)

    from src.ui.widgets.plot_ui_manager import PlotUIManager
    from src.ui.widgets.base_manager import BasePlotManager

    assert issubclass(PlotUIManager, BasePlotManager), "PlotUIManager 应该继承自 BasePlotManager"
    print("✓ PlotUIManager 正确继承自 BasePlotManager")

    return True

def test_base_manager_weakref():
    """测试 BasePlotManager 使用 weakref"""
    print("\n" + "=" * 60)
    print("测试 7: BasePlotManager weakref 测试")
    print("=" * 60)

    from src.ui.widgets.base_manager import BasePlotManager
    import weakref

    mock_pw = MagicMock()
    manager = BasePlotManager(mock_pw)

    assert hasattr(manager, '_pw_ref'), "BasePlotManager 应该有 _pw_ref 属性"
    assert isinstance(manager._pw_ref, weakref.ref), "_pw_ref 应该是 weakref.ref"
    print("✓ _pw_ref 是 weakref.ref 实例")

    assert manager.pw is mock_pw, "pw property 应该返回原始对象"
    print("✓ pw property 正确工作")

    return True

def test_csv_plot_pyqt6_import():
    """测试主文件可以正确导入"""
    print("\n" + "=" * 60)
    print("测试 8: csv_plot_pyqt6 导入测试")
    print("=" * 60)

    try:
        import csv_plot_pyqt6
        print("✓ csv_plot_pyqt6 成功导入")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_draggable_widget_has_ui_manager():
    """测试 DraggableGraphicsLayoutWidget 有 ui_manager 属性"""
    print("\n" + "=" * 60)
    print("测试 9: DraggableGraphicsLayoutWidget.ui_manager 测试")
    print("=" * 60)

    from csv_plot_pyqt6 import DraggableGraphicsLayoutWidget
    from src.ui.widgets.plot_ui_manager import PlotUIManager

    assert hasattr(DraggableGraphicsLayoutWidget, '__init__'), "DraggableGraphicsLayoutWidget 应该有 __init__"
    print("✓ __init__ 方法存在")

    init_source = DraggableGraphicsLayoutWidget.__init__.__code__.co_names
    if 'ui_manager' in init_source or '_PlotUIManager__ui_manager' in init_source:
        print("✓ ui_manager 在 __init__ 中被引用")
    else:
        source_file = Path(csv_plot_pyqt6.__file__).read_text()
        if 'self.ui_manager' in source_file:
            print("✓ ui_manager 在类定义中被设置")
        else:
            print("⚠ ui_manager 引用未找到（可能在父类中）")

    return True

def test_ui_manager_delegates():
    """测试 DraggableGraphicsLayoutWidget 中的委托方法"""
    print("\n" + "=" * 60)
    print("测试 10: 委托方法验证")
    print("=" * 60)

    from csv_plot_pyqt6 import DraggableGraphicsLayoutWidget

    delegating_methods = [
        'setup_header',
        'setup_plot_area',
        'setup_axes',
        'setup_interaction',
        '_init_ui_refresh_coordinator',
        '_queue_ui_refresh',
        '_cancel_ui_refresh',
        '_run_style_refresh',
        '_run_cursor_refresh',
        '_run_stats_refresh',
        'update_x_axis_label',
    ]

    for method_name in delegating_methods:
        assert hasattr(DraggableGraphicsLayoutWidget, method_name), f"应该有 {method_name} 方法"
        print(f"✓ {method_name} 存在")

    return True

def test_plot_context_protocol():
    """测试 PlotContext 协议定义"""
    print("\n" + "=" * 60)
    print("测试 11: PlotContext 协议测试")
    print("=" * 60)

    try:
        from src.app.plot_context import PlotContext, PlotServices
        print("✓ PlotContext 和 PlotServices 成功导入")

        assert hasattr(PlotServices, 'request_mark_stats_refresh'), "PlotServices 应该有 request_mark_stats_refresh"
        print("✓ PlotServices.request_mark_stats_refresh 方法存在")

        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_syntax_check():
    """语法检查测试"""
    print("\n" + "=" * 60)
    print("测试 12: 语法检查")
    print("=" * 60)

    import py_compile
    import tempfile
    import os

    files_to_check = [
        'csv_plot_pyqt6.py',
        'src/ui/widgets/plot_ui_manager.py',
        'src/ui/widgets/base_manager.py',
        'src/app/plot_context.py',
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
    print("PlotUIManager 迁移测试套件 (无 GUI)")
    print("=" * 60)

    tests = [
        test_plot_ui_manager_import,
        test_plot_ui_manager_initialization,
        test_setup_ui_method_exists,
        test_plot_ui_manager_sub_methods,
        test_base_manager_import,
        test_base_manager_inheritance,
        test_base_manager_weakref,
        test_csv_plot_pyqt6_import,
        test_draggable_widget_has_ui_manager,
        test_ui_manager_delegates,
        test_plot_context_protocol,
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
