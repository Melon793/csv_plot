"""游标 X 域（vline bounds）唯一权威源的回归护栏

对应 tmp/20260920_cursor_bounds_analysis.md 与 tmp/20260920_cursor_domain_fix_plan.md：
空 plot / 删掉最后一条曲线 / 全部隐藏这几条回退路径，游标可移动区间必须由
AxisManager.cursor_x_domain()（全局数据 X 域，随当前 factor/offset 现算）给出，
不得再读 pw.xMin/pw.xMax 快照、也不得放开成 [None, None]。

宿主类在本文件内自建（不改 conftest 的 FakeHost）：给 window() 补 cursor_btn
等成员，让 _update_cursor_after_plot 走真实主窗口分支。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QPushButton, QWidget

from src.ui.widgets.plot_widget import DraggableGraphicsLayoutWidget

N = 100


class FakeLoader:
    """CSV loader 的最小替身：只提供 X 域计算用到的字段。

    global_time_range 与 src/data/base_loader.py 保持一致（CSV 场景是 index 空间）。
    """

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
    """带 loader 与 plot_widgets 的 PlotContext 替身。

    必须提供 loader / plot_widgets / value_cache / _enum_text_maps，否则绘图与
    时间修正主路径会走 QMessageBox 分支而阻塞 offscreen 测试。
    """

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
    """_run_stats_refresh 收尾会回调宿主 layout_manager，offscreen 下需要占位"""

    def request_mark_stats_refresh(self, immediate: bool = False):
        pass


class FakeMainWindow(QWidget):
    """模拟 bounds 写入路径访问到的 MainWindow 成员。"""

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
    """带 loader 的可绘图 plot widget 工厂（收尾统一销毁，避免跨用例串状态）"""
    created = []

    def _make(n: int = N):
        df = pd.DataFrame(
            {
                "a": np.arange(n, dtype=float) % 7,
                "b": np.sin(np.arange(n, dtype=float) / 9.0),
            }
        )
        loader = FakeLoader(df)
        ctx = FakeCtx(loader)
        pw = DraggableGraphicsLayoutWidget({}, df)
        pw.plot_context = ctx
        pw.setParent(host)
        host.plot_widgets.append(Container(pw))
        ctx.plot_widgets = list(host.plot_widgets)
        pw.resize(700, 260)
        pw.show()
        created.append(pw)
        return pw

    yield _make

    for pw in created:
        pw._is_being_destroyed = True
        pw.deleteLater()


def _bounds(pw) -> list:
    return list(pw.vline.bounds())


def test_snapshot_fields_retired(pw_factory):
    """陈旧快照字段必须已退役：不再有 pw.xMin / pw.xMax"""
    pw = pw_factory()
    assert not hasattr(pw, "xMin")
    assert not hasattr(pw, "xMax")


def test_empty_plot_reset_uses_global_domain(pw_factory):
    """I2：空 plot 的 reset_plot 之后游标域 = 全局数据域，而不是无界"""
    pw = pw_factory()
    pw.reset_plot(1, N)
    assert _bounds(pw) == pytest.approx([1.0, float(N)])


def test_empty_plot_bounds_follow_time_correction(pw_factory):
    """现象 1：时间修正 1.0 -> 0.1 后，从未绘图的 plot 的游标下界必须跟着走

    修复前这里读到 reset_plot 时算好的 pw.xMin=1，游标最小只能拖到 1。
    """
    pw = pw_factory()
    pw.reset_plot(1, N)
    pw.update_time_correction(0.1, 0.0)

    assert _bounds(pw) == pytest.approx([0.1, 10.0])
    assert pw.vline2.bounds() == pw.vline.bounds()          # I1

    pw.vline.setPos(0.05)                                   # 向左拖到数据外
    assert pw.vline.value() == pytest.approx(0.1)


def test_plotted_and_empty_plots_agree(pw_factory):
    """现象 1 的跨图面：绘图过的与从未绘图的，同一时刻必须落到同一数值"""
    plotted = pw_factory()
    empty = pw_factory()
    assert plotted.plot_variable("a")
    plotted.update_time_correction(0.1, 0.0)
    empty.update_time_correction(0.1, 0.0)

    assert _bounds(plotted) == _bounds(empty)
    for pw in (plotted, empty):
        pw.vline.setPos(0.5)
    assert plotted.vline.value() == pytest.approx(empty.vline.value())


def test_remove_last_curve_keeps_domain(pw_factory):
    """现象 2：删掉最后一条曲线不能把游标放开成无界

    修复前 _reset_plot_limits()/_clear_plot_data() 写 [None, None]，
    游标能一路拖到负值。
    """
    pw = pw_factory()
    assert pw.plot_variable("a")
    assert pw.remove_variable_from_plot("a")

    assert _bounds(pw) != [None, None]
    assert _bounds(pw) == pytest.approx([1.0, float(N)])

    pw.vline.setPos(-5.0)                                   # 试图拖到数据左侧之外
    assert pw.vline.value() == pytest.approx(1.0)


def test_hide_all_curves_uses_global_domain(pw_factory):
    """变体：全部隐藏（不是删除）同样回退到全局域，随当前 factor 现算"""
    pw = pw_factory()
    pw.update_time_correction(0.1, 0.0)
    assert pw.plot_variable("a")
    for ci in pw.curves.values():
        ci.visible = False

    assert pw._update_vline_bounds_from_data() == pytest.approx((0.1, 10.0))
    assert _bounds(pw) == pytest.approx([0.1, 10.0])


def test_visible_curve_range_still_wins(pw_factory):
    """I4：本轮口径不变 —— 有可见曲线时仍用本图曲线域（短数据 = 短域）"""
    pw = pw_factory()
    loader = pw.plot_context.loader
    assert pw.plot_variable("a")
    # 造一条只覆盖 index 10..29 的短曲线，再走一次 bounds 回退入口
    ci = next(iter(pw.curves.values()))
    ci.x_data = np.arange(10.0, 30.0, dtype=np.float32)
    ci.update_x_range()
    pw._update_vline_bounds_from_data()

    assert _bounds(pw) == pytest.approx([10.0, 29.0])
    assert loader.datalength == N                           # 全局域仍是 1..100
    assert pw._cursor_x_domain() == pytest.approx((1.0, float(N)))


def test_no_loader_falls_back_without_locking_cursor(pw_factory):
    """I3 + 反焊死安全网：无权威源时回退无界，且绝不能因此锁死游标更新链

    v0.3.15 的 `_is_cursor_update_locked` 用 bounds==[None,None] 当「无数据」
    哨兵（且因写成 vline.bounds 方法而从未生效）。bounds 语义必须只表示「无界」，
    否则这条断言会在有人「顺手补括号」时立刻失败 —— 那正是本用例的作用。
    """
    pw = pw_factory()
    pw.plot_context.loader = None

    assert pw._apply_cursor_x_domain() is None
    assert _bounds(pw) == [None, None]
    assert pw._is_cursor_update_locked() is False
