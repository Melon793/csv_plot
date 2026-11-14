"""游标 X 标签精度：时间轴按数据步长、序号轴按 factor 的分流回归。

背景：X 标签小数位过去只跟 ``factor`` 走的字面小数位，MDF/BLF 下
``factor=1.0`` 恒定导致大面积整数；``factor<1e-6`` 又会被字符串解析算成
0 位。本文件锁住新的分流口径，并守护 grouped loader 的 x 必须保 float64。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QPushButton, QWidget

from src.ui.widgets.plot_widget import DraggableGraphicsLayoutWidget

# 时间轴基准：1370 s 附近是 BLF 长录制的典型量级（float32 ULP≈0.12 ms）
T0 = 1370.0
DT = 0.001
N_TIME = 9


class FakeGroupedLoader:
    """分组数据源（MDF/BLF 语义）的最小替身：组时间轴 + get_series。"""

    LOADER_TYPE = "mdf"

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self._x = np.asarray(x, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)
        self.datalength = len(self._x)
        self.var_names = ["a"]
        self.df_validity = {"a": 1}
        self.units = {}

    @property
    def global_time_range(self) -> tuple[float, float]:
        return (float(self._x[0]), float(self._x[-1]))

    def get_series(self, var_name: str) -> np.ndarray:
        assert var_name == "a"
        return self._y

    def get_value_from_name(self, var_name: str):
        assert var_name == "a"
        return self._x, self._y, None, None


class FakeCsvLoader:
    """表格型数据源（CSV 语义）：无 get_series，x 由内部 arange 生成。"""

    LOADER_TYPE = "csv"

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.datalength = int(len(df))
        self.var_names = list(df.columns)
        self.df_validity = {v: 1 for v in df.columns}
        self.units = {}

    @property
    def global_time_range(self) -> tuple[float, float]:
        return (1.0, float(self.datalength))


class FakeCtx:
    """带 loader 与 plot_widgets 的 PlotContext 替身（同 test_cursor_x_domain）。"""

    def __init__(self, loader):
        self.loader = loader
        self.value_cache = {}
        self._enum_text_maps = {}
        self.plot_widgets = []
        self.pinned_x_values = []
        self.cursor_mode = "1 free cursor"

    @property
    def _is_loading_new_data(self) -> bool:
        return False

    @property
    def _data_version(self) -> int:
        return 0

    @property
    def _is_time_correction_active(self) -> bool:
        return False

    def request_mark_stats_refresh(self, immediate: bool = False):
        pass


class FakeCursorSync:
    def _sync_min_xrange(self):
        pass


class FakeLayoutManager:
    def request_mark_stats_refresh(self, immediate: bool = False):
        pass


class FakeMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.cursor_btn = QPushButton()
        self.cursor_btn.setCheckable(True)
        self.cursor_btn.setChecked(True)
        self.cursor_values_hidden = False
        self.cursor_sync_manager = FakeCursorSync()
        self.layout_manager = FakeLayoutManager()
        self.cursor_mode = "1 free cursor"
        self.pinned_x_values = []
        self._is_loading_new_data = False
        self._is_syncing_crosshair = False
        self._pending_crosshair_x = None
        self._crosshair_update_timer = QTimer(self)
        self.plot_widgets = []
        self.loader = None


class Container:
    def __init__(self, pw):
        self.plot_widget = pw

    def isVisible(self):
        return True


@pytest.fixture()
def host(qapp):
    h = FakeMainWindow()
    yield h
    h.deleteLater()


@pytest.fixture()
def pw_factory(qapp, host):
    """时间轴（grouped）与序号轴（CSV）两种 plot widget 工厂"""
    created = []

    def _make_time(n: int = N_TIME):
        x = T0 + DT * np.arange(n, dtype=np.float64)
        y = np.sin(np.arange(n, dtype=float))
        loader = FakeGroupedLoader(x, y)
        pw = DraggableGraphicsLayoutWidget({}, pd.DataFrame({"a": y}))
        pw.plot_context = FakeCtx(loader)
        _attach(pw)
        return pw

    def _make_csv(n: int = 100):
        df = pd.DataFrame({"a": np.arange(n, dtype=float) % 7})
        loader = FakeCsvLoader(df)
        pw = DraggableGraphicsLayoutWidget({}, df)
        pw.plot_context = FakeCtx(loader)
        _attach(pw)
        return pw

    def _attach(pw):
        pw.setParent(host)
        host.plot_widgets.append(Container(pw))
        pw.plot_context.plot_widgets = list(host.plot_widgets)
        pw.resize(700, 260)
        pw.show()
        created.append(pw)

    yield _make_time, _make_csv

    for pw in created:
        pw._is_being_destroyed = True
        pw.deleteLater()


def _x_label_texts(pw) -> list[str]:
    """当前可见 X 标签文本（TextItem 的实际内容）"""
    return [
        item.textItem.toPlainText()
        for item in pw._cursor_item_pool["x_labels"]
        if item.isVisible()
    ]


# ---------- 时间轴（grouped loader） ----------


def test_time_axis_step_and_decimals(pw_factory):
    make_time, _ = pw_factory
    pw = make_time()
    assert pw.plot_variable("a")
    ci = pw.curves["a"]

    assert ci.x_data.dtype == np.float64          # float32 会抹平毫秒级采样
    assert ci.x_step == pytest.approx(DT)
    assert ci.x_decimals == 3
    assert pw._cursor_manager._x_decimals() == 3


def test_time_axis_x_label_shows_milliseconds(pw_factory):
    make_time, _ = pw_factory
    pw = make_time()
    assert pw.plot_variable("a")

    target = T0 + 3 * DT
    pw.view_box.setXRange(T0, T0 + 8 * DT, padding=0)
    pw.show_values_only = True
    pw._cursor_manager._show_x_position_only([target])

    assert _x_label_texts(pw) == [f"{target:.3f}".rstrip("0").rstrip(".")]


def test_time_axis_refreshes_after_time_correction(pw_factory):
    """时间修正改变步长后，精度必须跟着新网格走（update_x_range 已刷新）"""
    make_time, _ = pw_factory
    pw = make_time()
    assert pw.plot_variable("a")

    pw.update_time_correction(100.0, 0.0)          # step 1 ms -> 100 ms
    ci = pw.curves["a"]
    assert ci.x_step == pytest.approx(DT * 100)
    assert pw._cursor_manager._x_decimals() == 1


# ---------- 序号轴（CSV/Excel）----------


@pytest.mark.parametrize(
    ("factor", "expected"),
    [(1.0, 0), (0.1, 1), (0.01, 2), (1e-7, 7)],
)
def test_index_axis_decimals_follow_factor(pw_factory, factor, expected):
    _, make_csv = pw_factory
    pw = make_csv()
    pw.factor = factor
    assert pw._cursor_manager._x_decimals() == expected


def test_index_axis_x_label_factor_one_stays_integer(pw_factory):
    """护栏：CSV factor=1 的显示与改造前逐字一致（不引入小数）"""
    _, make_csv = pw_factory
    pw = make_csv()
    assert pw.plot_variable("a")

    pw.view_box.setXRange(0, 99, padding=0)
    pw.show_values_only = True
    pw._cursor_manager._show_x_position_only([42.0])

    assert _x_label_texts(pw) == ["42"]


def test_csv_curve_keeps_float32_x(pw_factory):
    """护栏：只有 grouped loader 用 float64，CSV 仍走 float32（不翻倍内存）"""
    _, make_csv = pw_factory
    pw = make_csv()
    assert pw.plot_variable("a")
    assert pw.curves["a"].x_data.dtype == np.float32


# ---------- 回退与两路一致性 ----------


def test_hidden_curves_keep_time_precision(pw_factory):
    """显隐切换不得改变读数精度：全部隐藏时仍按曲线自身步长显示"""
    make_time, _ = pw_factory
    pw = make_time()
    assert pw.plot_variable("a")
    for ci in pw.curves.values():
        ci.visible = False

    assert pw._cursor_manager._x_decimals() == 3


def test_empty_plot_falls_back_without_error(pw_factory):
    """无任何曲线时回退 factor（1.0 -> 0 位），不得抛异常"""
    make_time, _ = pw_factory
    pw = make_time()

    assert pw._cursor_manager._x_decimals() == 0


def test_empty_plot_x_label_paths_agree(pw_factory):
    """默认路径与 show_values_only 路径必须共用同一精度口径"""
    make_time, _ = pw_factory
    pw = make_time()
    assert pw.plot_variable("a")

    target = T0 + 5 * DT
    pw.view_box.setXRange(T0, T0 + 8 * DT, padding=0)

    pw.show_values_only = True
    pw._cursor_manager._show_x_position_only([target])
    only_x = _x_label_texts(pw)

    pw.show_values_only = False
    pw._last_cursor_update_time = 0.0
    pw.vline.setVisible(True)
    pw.vline.setPos(target)
    pw._cursor_manager._update_multi_curve_cursor_label()
    full_x = _x_label_texts(pw)

    assert only_x == full_x == [f"{target:.3f}".rstrip("0").rstrip(".")]


# ---------- 标记统计窗口的 x 列精度 ----------


def test_mark_stats_x_decimals_follow_axis_type(pw_factory):
    """统计窗口的 x 列与游标同口径：时间轴按数据步长，序号轴按 factor"""
    make_time, make_csv = pw_factory

    time_pw = make_time()
    assert time_pw.plot_variable("a")
    stats = time_pw.get_mark_stats(region=(T0 + DT, T0 + 5 * DT))
    assert stats and stats[0].x_decimals == 3

    csv_pw = make_csv()
    csv_pw.factor = 0.1
    assert csv_pw.plot_variable("a")
    stats = csv_pw.get_mark_stats(region=(1.0, 5.0))
    assert stats and stats[0].x_decimals == 1
