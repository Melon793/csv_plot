#!/usr/bin/env python3
"""PlotDataManager 独立测试 - 不依赖 PyQt6"""

import sys
from pathlib import Path
import ast

sys.path.insert(0, str(Path(__file__).parent))

def test_plot_data_manager_file_exists():
    """测试文件存在"""
    print("=" * 60)
    print("测试 1: 文件存在性测试")
    print("=" * 60)

    filepath = Path("src/ui/widgets/plot_data_manager.py")
    assert filepath.exists(), f"{filepath} 不存在"
    print(f"✓ {filepath} 存在")
    return True

def test_plot_data_manager_syntax():
    """测试语法正确"""
    print("\n" + "=" * 60)
    print("测试 2: 语法检查")
    print("=" * 60)

    import py_compile
    filepath = Path("src/ui/widgets/plot_data_manager.py")
    try:
        py_compile.compile(str(filepath), doraise=True)
        print("✓ 语法正确")
        return True
    except py_compile.PyCompileError as e:
        print(f"✗ 语法错误: {e}")
        return False

def test_plot_data_manager_class():
    """测试类定义"""
    print("\n" + "=" * 60)
    print("测试 3: 类定义测试")
    print("=" * 60)

    filepath = Path("src/ui/widgets/plot_data_manager.py")
    source = filepath.read_text()
    tree = ast.parse(source)

    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    assert "PlotDataManager" in classes, "PlotDataManager 类未找到"
    print(f"✓ 找到类: {classes}")

    class_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PlotDataManager":
            class_node = node
            break

    methods = [n.name for n in class_node.body if isinstance(n, ast.FunctionDef)]
    expected_methods = [
        '__init__',
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

    for method in expected_methods:
        if method in methods:
            print(f"  ✓ {method}")
        else:
            print(f"  ✗ {method} 缺失")
            return False

    return True

def test_csv_plot_pyqt6_imports():
    """测试主文件导入"""
    print("\n" + "=" * 60)
    print("测试 4: 主文件导入测试")
    print("=" * 60)

    filepath = Path("csv_plot_pyqt6.py")
    source = filepath.read_text()

    assert "from src.ui.widgets.plot_data_manager import PlotDataManager" in source, \
        "缺少 PlotDataManager 导入"
    print("✓ PlotDataManager 导入存在")

    assert "self.plot_data_manager = PlotDataManager(self.axis_manager)" in source, \
        "缺少 plot_data_manager 初始化"
    print("✓ plot_data_manager 初始化存在")

    return True

def test_delegation_pattern():
    """测试委托模式"""
    print("\n" + "=" * 60)
    print("测试 5: 委托模式测试")
    print("=" * 60)

    filepath = Path("csv_plot_pyqt6.py")
    source = filepath.read_text()

    delegating_methods = [
        'plot_variable',
        '_validate_plot_data',
        '_get_x_data_for_variable',
        '_prepare_plot_data',
        'reset_plot',
        'handle_single_point_limits',
        'clear_value_cache',
        'datetime_to_unix_seconds',
        'get_value_from_name',
        'update_time_correction',
        '_compute_valid_min_max',
        '_get_y_range_in_x_window',
        '_clear_plot_data',
        'clear_plot_item',
    ]

    for method in delegating_methods:
        pattern = f"return self.plot_data_manager.{method}"
        assert pattern in source, f"{method} 委托缺失"
        print(f"  ✓ {method} 委托")

    return True

def test_main_file_syntax():
    """测试主文件语法"""
    print("\n" + "=" * 60)
    print("测试 6: 主文件语法检查")
    print("=" * 60)

    import py_compile
    filepath = Path("csv_plot_pyqt6.py")
    try:
        py_compile.compile(str(filepath), doraise=True)
        print("✓ 语法正确")
        return True
    except py_compile.PyCompileError as e:
        print(f"✗ 语法错误: {e}")
        return False

def test_csv_line_count():
    """测试行数统计"""
    print("\n" + "=" * 60)
    print("测试 7: 行数统计")
    print("=" * 60)

    main_file = Path("csv_plot_pyqt6.py")
    manager_file = Path("src/ui/widgets/plot_data_manager.py")

    main_lines = len(main_file.read_text().splitlines())
    manager_lines = len(manager_file.read_text().splitlines())

    print(f"  csv_plot_pyqt6.py: {main_lines} 行")
    print(f"  plot_data_manager.py: {manager_lines} 行")

    assert main_lines > 0, "主文件为空"
    assert manager_lines > 100, "管理器文件行数太少"

    return True

def run_all_tests():
    print("\n" + "=" * 60)
    print("PlotDataManager 独立测试套件")
    print("=" * 60)

    tests = [
        test_plot_data_manager_file_exists,
        test_plot_data_manager_syntax,
        test_plot_data_manager_class,
        test_csv_plot_pyqt6_imports,
        test_delegation_pattern,
        test_main_file_syntax,
        test_csv_line_count,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
                print(f"✓ {test.__name__} 通过\n")
            else:
                failed += 1
                print(f"✗ {test.__name__} 失败\n")
        except Exception as e:
            import traceback
            print(f"✗ {test.__name__} 异常: {e}\n")
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
