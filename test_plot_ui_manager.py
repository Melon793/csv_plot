#!/usr/bin/env python3
"""测试 PlotUIManager 迁移后的功能正确性"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PyQt6.QtWidgets import QApplication
except ImportError:
    print("PyQt6 不可用，跳过 GUI 测试")
    sys.exit(0)

try:
    import pyqtgraph as pg
except ImportError:
    print("pyqtgraph 不可用，跳过测试")
    sys.exit(0)

import numpy as np
import pandas as pd

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

def test_plot_ui_manager_initialization():
    print("=" * 60)
    print("测试 1: PlotUIManager 初始化")
    print("=" * 60)

    from csv_plot_pyqt6 import DraggableGraphicsLayoutWidget
    from src.ui.widgets.plot_ui_manager import PlotUIManager

    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    pw = DraggableGraphicsLayoutWidget(units_dict={}, dataframe=df)

    assert hasattr(pw, 'ui_manager'), "DraggableGraphicsLayoutWidget 应该拥有 ui_manager 属性"
    assert isinstance(pw.ui_manager, PlotUIManager), "ui_manager 应该是 PlotUIManager 实例"
    print("✓ ui_manager 属性正确创建")
    print(f"✓ PlotUIManager 类型: {type(pw.ui_manager)}")
    pw.close()
    print("测试 1 通过!\n")
    return True

def test_setup_ui_delegation():
    print("=" * 60)
    print("测试 2: setup_ui 委托验证")
    print("=" * 60)

    from csv_plot_pyqt6 import DraggableGraphicsLayoutWidget

    df = pd.DataFrame({
        'time': [0.0, 0.1, 0.2, 0.3, 0.4],
        'speed': [10.0, 20.0, 30.0, 25.0, 15.0],
        'rpm': [1000, 2000, 3000, 2800, 1500]
    })
    units = {'speed': 'km/h', 'rpm': 'rpm'}

    pw = DraggableGraphicsLayoutWidget(units_dict=units, dataframe=df)

    assert hasattr(pw, 'units'), "pw 应该拥有 units 属性"
    assert pw.units == units, "units 属性应该正确设置"
    assert hasattr(pw, 'data'), "pw 应该拥有 data 属性"
    assert pw.data is df, "data 属性应该是传入的 DataFrame"
    print("✓ units 和 data 属性正确设置")

    assert hasattr(pw, 'curves'), "pw 应该拥有 curves 字典"
    assert pw.curves == {}, "curves 应该初始化为空字典"
    print("✓ curves 字典正确初始化为空字典")

    assert hasattr(pw, 'is_multi_curve_mode'), "pw 应该拥有 is_multi_curve_mode 属性"
    assert pw.is_multi_curve_mode == False, "is_multi_curve_mode 应该为 False"
    print("✓ is_multi_curve_mode 属性正确初始化")

    assert hasattr(pw, 'rubberBand'), "pw 应该拥有 rubberBand 属性"
    print("✓ rubberBand 正确创建")

    assert hasattr(pw, '_ui_refresh'), "pw 应该拥有 _ui_refresh 调度器"
    print("✓ 统一更新调度器正确创建")

    pw.close()
    print("测试 2 通过!\n")
    return True

def test_header_setup():
    print("=" * 60)
    print("测试 3: Header 设置验证")
    print("=" * 60)

    from csv_plot_pyqt6 import DraggableGraphicsLayoutWidget

    df = pd.DataFrame({'x': [1, 2, 3]})
    pw = DraggableGraphicsLayoutWidget(units_dict={}, dataframe=df)

    assert hasattr(pw, 'label_left'), "pw 应该拥有 label_left 属性"
    print(f"✓ label_left 已创建: {pw.label_left}")
    pw.close()
    print("测试 3 通过!\n")
    return True

def test_plot_area_setup():
    print("=" * 60)
    print("测试 4: 绘图区域设置验证")
    print("=" * 60)

    from csv_plot_pyqt6 import DraggableGraphicsLayoutWidget

    df = pd.DataFrame({'x': [1, 2, 3], 'y': [10, 20, 30]})
    units = {'y': 'm/s'}

    pw = DraggableGraphicsLayoutWidget(units_dict=units, dataframe=df)

    assert hasattr(pw, 'plot_item'), "pw 应该拥有 plot_item"
    print("✓ plot_item 已创建")

    assert hasattr(pw, 'view_box'), "pw 应该拥有 view_box"
    print("✓ view_box 已创建")

    assert hasattr(pw, '_is_interacting'), "pw 应该拥有 _is_interacting 状态标志"
    assert pw._is_interacting == False, "_is_interacting 应该初始化为 False"
    print("✓ _is_interacting 状态标志正确")

    assert hasattr(pw, 'axis_x'), "pw 应该拥有 axis_x"
    assert hasattr(pw, 'axis_y'), "pw 应该拥有 axis_y"
    print("✓ axis_x 和 axis_y 已创建")

    pw.close()
    print("测试 4 通过!\n")
    return True

def test_cursor_setup():
    print("=" * 60)
    print("测试 5: 光标设置验证")
    print("=" * 60)

    from csv_plot_pyqt6 import DraggableGraphicsLayoutWidget

    df = pd.DataFrame({'x': [1, 2, 3]})
    pw = DraggableGraphicsLayoutWidget(units_dict={}, dataframe=df)

    assert hasattr(pw, 'vline'), "pw 应该拥有 vline"
    assert hasattr(pw, 'vline2'), "pw 应该拥有 vline2"
    print("✓ vline 和 vline2 已创建")

    assert hasattr(pw, 'cursor_label'), "pw 应该拥有 cursor_label"
    print("✓ cursor_label 已创建")

    pw.close()
    print("测试 5 通过!\n")
    return True

def test_refresh_coordinator():
    print("=" * 60)
    print("测试 6: 刷新协调器验证")
    print("=" * 60)

    from csv_plot_pyqt6 import DraggableGraphicsLayoutWidget
    from src.core.scheduler import UnifiedUpdateScheduler

    df = pd.DataFrame({'x': [1, 2, 3]})
    pw = DraggableGraphicsLayoutWidget(units_dict={}, dataframe=df)

    assert hasattr(pw, '_ui_refresh'), "pw 应该拥有 _ui_refresh"
    assert isinstance(pw._ui_refresh, UnifiedUpdateScheduler), "_ui_refresh 应该是 UnifiedUpdateScheduler 实例"
    print("✓ _ui_refresh 是 UnifiedUpdateScheduler 实例")

    assert hasattr(pw, '_queue_ui_refresh'), "pw 应该拥有 _queue_ui_refresh 方法"
    assert hasattr(pw, '_cancel_ui_refresh'), "pw 应该拥有 _cancel_ui_refresh 方法"
    print("✓ 刷新方法存在")

    pw.close()
    print("测试 6 通过!\n")
    return True

def test_integration_with_real_data():
    print("=" * 60)
    print("测试 7: 使用真实数据集成测试")
    print("=" * 60)

    from csv_plot_pyqt6 import DraggableGraphicsLayoutWidget, CSVData

    test_file = Path("data/1pt.csv")
    if not test_file.exists():
        test_file.write_text("RS_CKS,TRQ_EN\nrpm,Nm\n1000,200\n", encoding='utf-8')

    try:
        csv_data = CSVData(
            path="data/1pt.csv",
            descRows=0,
            hasunit=True,
            sep=",",
            encoding="utf-8",
            max_rows_infer=100,
            do_parse_date=True,
            drop_empty=True,
            downcast_float=True,
        )
        print(f"✓ CSV 数据加载成功")
        print(f"  行数: {csv_data.row_count}")
        print(f"  列数: {csv_data.column_count}")

        pw = DraggableGraphicsLayoutWidget(
            units_dict=csv_data.units,
            dataframe=csv_data.df
        )

        assert pw.ui_manager is not None, "ui_manager 应该不为 None"
        print("✓ DraggableGraphicsLayoutWidget 初始化成功")

        assert hasattr(pw, 'plot_item'), "plot_item 应该存在"
        assert hasattr(pw, 'view_box'), "view_box 应该存在"
        print("✓ 绘图组件正确创建")

        pw.close()
        print("✓ 所有组件正确清理")

    except Exception as e:
        import traceback
        print(f"✗ 错误: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise

    print("测试 7 通过!\n")
    return True

def test_multiple_curve_mode():
    print("=" * 60)
    print("测试 8: 多曲线模式属性验证")
    print("=" * 60)

    from csv_plot_pyqt6 import DraggableGraphicsLayoutWidget

    df = pd.DataFrame({'x': [1, 2, 3], 'y1': [1, 2, 3], 'y2': [4, 5, 6]})
    units = {'y1': 'm', 'y2': 'km'}

    pw = DraggableGraphicsLayoutWidget(units_dict=units, dataframe=df)

    assert hasattr(pw, 'curve_colors'), "pw 应该拥有 curve_colors"
    assert len(pw.curve_colors) > 0, "curve_colors 应该非空"
    print(f"✓ curve_colors: {pw.curve_colors}")

    assert hasattr(pw, 'current_color_index'), "pw 应该拥有 current_color_index"
    assert pw.current_color_index == 0, "current_color_index 应该初始化为 0"
    print("✓ current_color_index 正确初始化")

    assert hasattr(pw, '_max_point_density'), "pw 应该拥有 _max_point_density"
    print(f"✓ _max_point_density: {pw._max_point_density}")

    pw.close()
    print("测试 8 通过!\n")
    return True

def test_plot_ui_manager_class():
    print("=" * 60)
    print("测试 9: PlotUIManager 类直接测试")
    print("=" * 60)

    from src.ui.widgets.plot_ui_manager import PlotUIManager

    class MockPlotWidget:
        def __init__(self):
            self.curve = None
            self.time_values = None
            self.time_column_name = None
            self.time_axis_label = "Index"
            self.y_name = ''
            self.y_format = ''
            self.x_name = ''
            self.x_format = ''
            self.xMin = 0
            self.xMax = 1

    mock_pw = MockPlotWidget()
    manager = PlotUIManager(mock_pw)

    assert manager._pw is mock_pw, "_pw 属性应该指向传入的 plot_widget"
    print("✓ PlotUIManager 正确持有 plot_widget 引用")

    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    units = {'y': 'unit1'}

    try:
        manager.setup_ui(units, df, None, None)
        print("✓ setup_ui 方法执行成功")

        assert hasattr(mock_pw, 'units'), "mock_pw 应该拥有 units 属性"
        assert mock_pw.units == units, "units 应该正确设置"
        print("✓ units 属性正确设置")

        assert hasattr(mock_pw, 'data'), "mock_pw 应该拥有 data 属性"
        print("✓ data 属性正确设置")

        assert hasattr(mock_pw, 'curves'), "mock_pw 应该拥有 curves 字典"
        print("✓ curves 字典正确创建")

    except Exception as e:
        import traceback
        print(f"✗ setup_ui 执行失败: {e}")
        traceback.print_exc()
        raise

    print("测试 9 通过!\n")
    return True

def run_all_tests():
    print("\n" + "=" * 60)
    print("PlotUIManager 迁移测试套件")
    print("=" * 60 + "\n")

    tests = [
        test_plot_ui_manager_initialization,
        test_setup_ui_delegation,
        test_header_setup,
        test_plot_area_setup,
        test_cursor_setup,
        test_refresh_coordinator,
        test_integration_with_real_data,
        test_multiple_curve_mode,
        test_plot_ui_manager_class,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
        except Exception as e:
            import traceback
            print(f"✗ 测试失败: {test.__name__}")
            print(f"  错误: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    QTimer.singleShot(100, app.quit)
    sys.exit(0 if success else 1)
