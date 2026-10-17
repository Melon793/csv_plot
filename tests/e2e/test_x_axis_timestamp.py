"""时间戳 X 轴的 e2e 全链路：真实 MainWindow + 加载器 + 布局切换

设计口径（为什么是轴渲染层映射而不是换 X 数据）：CSV/Excel 路径下游标同步、
标注坐标、offset/factor 时间校正都工作在**索引空间**。把曲线 X 真换成时间戳
会全链路波及；因此只换底轴渲染与游标读数口径。本文件守住这条链路的每个出口：
切换 / 恢复 / 布局继承 / 重载回落。
"""

import pandas as pd
import pytest
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMenu

from src.ui.widgets.custom_viewbox import ZH_X_AXIS, ZH_X_AXIS_DEFAULT
from src.ui.widgets.time_axis_item import TimestampAxisItem

N_ROWS = 500
START = "11:51:53"
FREQ_S = 4


def _make_time_rows(n: int) -> list[list]:
    """[Time(时刻字符串), speed, rpm]；speed/rpm 递增保证非常量列"""
    base = pd.Timestamp(f"2026-10-03 {START}")
    rows = []
    for i in range(n):
        t = (base + pd.Timedelta(seconds=FREQ_S * i)).strftime("%H:%M:%S")
        rows.append([t, f"{10.0 + i * 0.5:.2f}", 800 + i * 10])
    return rows


@pytest.fixture()
def timed_window(main_window, qapp, qtbot, dialog_stubs, tmp_path):
    """已加载带时刻列 CSV 的 MainWindow（走真实「导入数据文件」路径）"""
    from tests.fixtures.data_factory import write_csv

    csv = write_csv(
        tmp_path / "timed.csv",
        header=["Time", "speed", "rpm"],
        units=["-", "km/h", "rpm"],
        rows=_make_time_rows(N_ROWS),
    )
    dialog_stubs["open_file"] = (str(csv), "CSV/TXT Files (*.csv)")
    qtbot.mouseClick(main_window.load_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: main_window.loader is not None, timeout=5000)
    qtbot.waitUntil(lambda: len(main_window.plot_widgets) > 0, timeout=5000)
    qapp.processEvents()
    assert not dialog_stubs["critical"], f"加载出现错误弹窗: {dialog_stubs['critical']}"
    return main_window


def _pw(mw, index=0):
    return mw.plot_widgets[index].plot_widget


class TestDetection:
    def test_datetime_column_is_detected(self, timed_window):
        mw = timed_window
        assert "Time" in mw.available_time_columns()

    def test_default_mode_untouched(self, timed_window):
        """默认（Index）模式：轴、标签、读数全部保持原状 —— 零影响的底线"""
        mw = timed_window
        pw = _pw(mw)
        assert mw.x_axis_time_column is None
        assert pw.x_axis_time_formatter is None
        assert type(pw.axis_x) is pg.AxisItem


class TestToggleAndRestore:
    def test_switch_applies_to_all_plots_and_axis_label(self, timed_window, qapp):
        mw = timed_window
        mw.layout_manager.create_subplots_matrix(2, 1)
        qapp.processEvents()

        mw.set_x_axis_time_column("Time")
        qapp.processEvents()

        for index in range(len(mw.plot_widgets)):
            pw = _pw(mw, index)
            assert isinstance(pw.axis_x, TimestampAxisItem)
            assert pw.x_axis_time_formatter is not None
            assert pw._x_axis_label_override == "Time"

        # 标签换成时间列名（DEFAULT_SHOW_X_AXIS_LABEL 关闭时隐藏，只验内部口径）
        mw.set_x_axis_time_column(None)
        qapp.processEvents()
        pw = _pw(mw)
        assert type(pw.axis_x) is pg.AxisItem
        assert pw.x_axis_time_formatter is None
        assert pw._x_axis_label_override is None

    def test_ticks_and_cursor_readout_translate(self, timed_window, qapp):
        mw = timed_window
        pw = _pw(mw)
        mw.set_x_axis_time_column("Time")
        qapp.processEvents()

        # 刻度：显示 X 250（factor=1, offset=0）→ 行 250 → 11:51:53 + 249×4s
        texts = pw.axis_x.tickStrings([1.0, 250.0], 1, 100)
        assert texts[0] == START
        assert texts[1] == "12:08:29"

        # 游标读数同口径
        assert pw._cursor_manager._format_x_readout(250.0) == "12:08:29"

        mw.set_x_axis_time_column(None)
        qapp.processEvents()
        assert ":" not in pw._cursor_manager._format_x_readout(250.0)

    def test_restore_brings_back_the_original_axis_object(self, timed_window, qapp):
        mw = timed_window
        pw = _pw(mw)
        original = pw._default_bottom_axis

        mw.set_x_axis_time_column("Time")
        qapp.processEvents()
        mw.set_x_axis_time_column(None)
        qapp.processEvents()

        assert pw.axis_x is original, "恢复默认应换回建图时的那根轴"
        assert pw.plot_item.getAxis("bottom") is original

    def test_switched_axis_inherits_default_grid_and_pen_style(self, timed_window, qapp):
        """换轴后网格/笔必须抄默认轴：showGrid(alpha=0.1) 把各轴 setGrid(25)，
        硬编码 setGrid(255) 会把整片网格画成纯黑（实测缺陷）"""
        mw = timed_window
        pw = _pw(mw)
        default_axis = pw._default_bottom_axis

        mw.set_x_axis_time_column("Time")
        qapp.processEvents()

        axis = pw.axis_x
        assert isinstance(axis, TimestampAxisItem)
        assert axis.grid == default_axis.grid
        assert axis.pen() == default_axis.pen()
        assert axis.textPen() == default_axis.textPen()


class TestLayoutAndReload:
    def test_new_widgets_after_layout_switch_inherit_the_mode(self, timed_window, qapp):
        mw = timed_window
        mw.set_x_axis_time_column("Time")
        qapp.processEvents()

        mw.layout_manager.create_subplots_matrix(2, 1)
        qapp.processEvents()

        for index in range(len(mw.plot_widgets)):
            assert isinstance(_pw(mw, index).axis_x, TimestampAxisItem), (
                f"布局切换后的第 {index} 个子图没有继承时间戳模式"
            )

        mw.set_x_axis_time_column(None)
        qapp.processEvents()
        for index in range(len(mw.plot_widgets)):
            assert type(_pw(mw, index).axis_x) is pg.AxisItem

    def test_reload_without_time_column_falls_back_silently(self, timed_window, qapp, qtbot, dialog_stubs, tmp_path):
        """切换到时间模式后重载一个没有时间列的文件：静默回落默认轴"""
        from tests.fixtures.data_factory import make_simple_rows, write_csv

        mw = timed_window
        mw.set_x_axis_time_column("Time")
        qapp.processEvents()

        plain = write_csv(
            tmp_path / "plain.csv",
            header=["time", "speed", "rpm", "flag"],
            units=["s", "km/h", "rpm", "-"],
            rows=make_simple_rows(50),
        )
        # 首次加载是异步锁定的：解锁前点按钮会被静默拒绝
        qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
        dialog_stubs["open_file"] = (str(plain), "CSV/TXT Files (*.csv)")
        qtbot.mouseClick(mw.load_btn, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(
            lambda: mw.loader is not None and str(mw.loader.path).endswith("plain.csv"),
            timeout=5000,
        )
        qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
        qapp.processEvents()

        assert mw.x_axis_time_column is None
        for index in range(len(mw.plot_widgets)):
            assert type(_pw(mw, index).axis_x) is pg.AxisItem


class TestContextMenu:
    def _build(self, mw):
        # 临时父菜单必须保活：子菜单以它为 parent，父对象一死 C++ 侧跟着析构
        holder = QMenu()
        menu = _pw(mw).view_box._build_x_axis_menu(holder)
        return menu, holder

    def test_menu_lists_default_and_time_columns(self, timed_window, qapp):
        mw = timed_window
        mw.set_x_axis_time_column("Time")
        qapp.processEvents()

        menu, _holder = self._build(mw)
        titles = [act.text() for act in menu.actions() if not act.isSeparator()]
        assert ZH_X_AXIS_DEFAULT in titles
        assert "Time" in titles

        checked = {act.text(): act.isChecked() for act in menu.actions() if act.isCheckable()}
        assert checked[ZH_X_AXIS_DEFAULT] is False
        assert checked["Time"] is True

        mw.set_x_axis_time_column(None)
        qapp.processEvents()
        menu, _holder = self._build(mw)
        checked = {act.text(): act.isChecked() for act in menu.actions() if act.isCheckable()}
        assert checked[ZH_X_AXIS_DEFAULT] is True

    def test_no_menu_without_time_columns(self, loaded_window, qapp):
        """普通数值 CSV：没有可作时间轴的列 → 菜单整组不出现"""
        menu, _holder = self._build(loaded_window)
        assert menu is None

    def test_menu_trigger_dispatches_and_applies_globally(self, timed_window, qapp):
        """勾选时间列 → vb 信号 → event_handler → MainWindow → 全部子图换轴"""
        mw = timed_window
        pw = _pw(mw)
        pw.view_box.signals.request_set_x_axis_column.emit(pw, "Time")
        qapp.processEvents()

        assert mw.x_axis_time_column == "Time"
        for index in range(len(mw.plot_widgets)):
            assert isinstance(_pw(mw, index).axis_x, TimestampAxisItem)

        pw.view_box.signals.request_set_x_axis_column.emit(pw, None)
        qapp.processEvents()
        assert mw.x_axis_time_column is None
        assert type(_pw(mw).axis_x) is pg.AxisItem
