"""多标签页 P0-3：全局口径与 scene 分级的护栏（设计文档 §5.5 / §6.7 / §7.2）。

这一批改动在**单页形态下行为不变**，所以用例不能只断言「今天没坏」，而要把
多页才会暴露的口径差异提前钉住：

1. 密度是全局量：``_sync_min_xrange`` 必须跨页收集（R0：隐藏页里的单元格
   ``isVisible()=False`` 但 ``isHidden()=False``，用错谓词会整页漏掉）；
2. 非活动页的数据重建只许写纯数值：``_finalize_batch_add(scene=False)`` /
   ``_update_cursor_after_plot(scene=False)`` 不得碰 vline 可见性与标签；
3. 时间修正的 ``scene=False`` 分支只重算曲线数据，X limits 这类场景写入
   留到激活时补建；
4. ``LayoutManager.apply_time_correction`` 必须把全部页都传给
   ``update_time_correction``，且只有当前页拿到 ``scene=True``。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QPushButton, QWidget

from src.core.config import MIN_INDEX_LENGTH, compute_global_x_limits
from src.app.plot_context import PageServices, PlotContext
from src.ui.cursor_sync_manager import CursorSyncManager
from src.ui.layout_manager import LayoutManager
from src.ui.widgets.cursor_manager import CursorManager
from src.ui.widgets.multi_curve_manager import MultiCurveManager
from src.ui.widgets.plot_widget import DraggableGraphicsLayoutWidget

N = 100


# ------------------------------------------------------------------ 密度口径


class _DensityPW:
    """只暴露密度收集/应用会用到的两个名字。"""

    def __init__(self, density: float):
        self._max_point_density = density
        self.min_ranges: list[float] = []

    def _set_min_x_range(self, value: float) -> None:
        self.min_ranges.append(value)


class _DensityHost:
    def __init__(self, current, all_containers, baseline=0.0):
        self.plot_widgets = list(current)
        self._all = list(all_containers)
        self._baseline_density = baseline
        self._global_max_density = 0.0

    def iter_all_plot_widgets(self):
        return list(self._all)


class _RealContainer(QWidget):
    """真 QWidget：``isVisible``/``isHidden`` 的差异来自父页隐藏。"""

    def __init__(self, density: float):
        super().__init__()
        self.plot_widget = _DensityPW(density)


def test_density_is_collected_from_a_hidden_page(qapp):
    """隐藏页不是「不可见单元格」：用 isVisible() 会把它整页漏掉。"""
    hidden_page = QWidget()
    hidden_page.hide()
    hidden = _RealContainer(500.0)
    hidden.setParent(hidden_page)

    current = _RealContainer(100.0)
    current.show()

    # 前提：正是 R0 描述的形态 —— 不可见但并未被显式隐藏
    assert hidden.isVisible() is False
    assert hidden.isHidden() is False

    host = _DensityHost([current], [current, hidden])
    CursorSyncManager(host)._sync_min_xrange()

    assert host._global_max_density == pytest.approx(500.0)
    expected = MIN_INDEX_LENGTH / 500.0
    assert current.plot_widget.min_ranges == pytest.approx([expected])
    # 场景写入只落在当前页，隐藏页不应被跨页改 ViewBox
    assert hidden.plot_widget.min_ranges == []

    hidden_page.deleteLater()


def test_explicitly_hidden_cell_is_still_skipped(qapp):
    """收紧要留边界：当前页里被显式隐藏的单元格不参与密度。"""
    shown = _RealContainer(100.0)
    hidden_cell = _RealContainer(999.0)
    hidden_cell.hide()
    shown.show()

    host = _DensityHost([shown, hidden_cell], [shown, hidden_cell])
    CursorSyncManager(host)._sync_min_xrange()

    assert host._global_max_density == pytest.approx(100.0)
    assert hidden_cell.plot_widget.min_ranges == []


# --------------------------------------------- PageServices 的写穿契约（§3）


def test_page_services_forwards_cursor_writes_without_shadow_state():
    """PlotContext 写 pinned/cursor_values_hidden 必须落到 MainWindow。

    代理只有 ``__getattr__`` 时，写操作会在代理实例上生成影子属性：
    拖动固定游标后 ``mw.pinned_x_values`` 仍是旧值，切页/时间修正随后
    拿着陈旧 pin 反算，factor 越大偏差越大。这里钉住两个显式 setter，
    同时保证没有退化成“什么都吞”的通用 ``__setattr__``。
    """
    mw = SimpleNamespace(pinned_x_values=[1.0], cursor_values_hidden=False)
    services = PageServices(mw, SimpleNamespace())
    ctx = PlotContext(services)

    ctx.pinned_x_values = [2.0, 3.0]
    ctx.cursor_values_hidden = True

    assert mw.pinned_x_values == [2.0, 3.0]
    assert mw.cursor_values_hidden is True
    # 代理实例上不得出现影子属性（vars 里只应有真实持有的 mw/page）
    assert set(vars(services)) == {"mw", "page"}


# ------------------------------------------------- scene=False 的纯数值约定


class _PageDirty:
    def __init__(self):
        self.dirty_calls = 0

    def mark_dirty_for_activation(self) -> None:
        self.dirty_calls += 1


class _FinalizePW:
    def __init__(self):
        self.bounds: list[list[float]] = []
        self.cursor_updates: list[tuple[float, float]] = []
        self.label_updates = 0
        self.density_updates = 0
        self.dirty_calls = 0
        self.vline = SimpleNamespace(isVisible=lambda: True)

    def _collect_visible_curve_arrays(self, key):
        return [np.array([10.0, 20.0], dtype=float)]

    def _set_vline_bounds(self, bounds):
        self.bounds.append(list(bounds))

    def _update_cursor_after_plot(self, min_x, max_x, **kwargs):
        self.cursor_updates.append((min_x, max_x))

    def update_cursor_label(self):
        self.label_updates += 1

    def _recalc_max_point_density(self):
        self.density_updates += 1

    def mark_dirty_for_activation(self):
        self.dirty_calls += 1


class _DataManagerStub:
    def __init__(self, pw):
        self.pw = pw


def _finalizer(pw) -> MultiCurveManager:
    manager = MultiCurveManager(_DataManagerStub(pw))
    manager._update_axes_for_multi_curve = lambda: None
    return manager


def test_finalize_batch_add_scene_false_only_writes_bounds():
    pw = _FinalizePW()
    _finalizer(pw)._finalize_batch_add(
        update_header=False, sync_min_xrange=False, scene=False
    )

    assert pw.bounds == [[10.0, 20.0]]
    assert pw.cursor_updates == []
    assert pw.label_updates == 0
    assert pw.dirty_calls == 1
    assert pw.density_updates == 1  # 纯数值，隐藏页也可以算


def test_finalize_batch_add_scene_true_keeps_old_behaviour():
    """正向对照：默认参数仍走原来的场景写入，证明分级没有误杀主路径。"""
    pw = _FinalizePW()
    _finalizer(pw)._finalize_batch_add(
        update_header=False, sync_min_xrange=False
    )

    assert pw.cursor_updates == [(10.0, 20.0)]
    assert pw.label_updates == 1
    assert pw.dirty_calls == 0


class _CursorPW:
    def __init__(self):
        self.bounds: list[list[float]] = []
        self.toggles: list[tuple] = []
        self.dirty_calls = 0

    def _set_vline_bounds(self, bounds):
        self.bounds.append(list(bounds))

    def toggle_cursor(self, *args, **kwargs):
        self.toggles.append((args, kwargs))

    def mark_dirty_for_activation(self):
        self.dirty_calls += 1


def _cursor_manager(pw) -> CursorManager:
    return CursorManager(MultiCurveManager(_DataManagerStub(pw)))


def test_cursor_after_plot_scene_false_skips_toggle_and_label():
    pw = _CursorPW()
    _cursor_manager(pw)._update_cursor_after_plot(3.0, 9.0, scene=False)

    assert pw.bounds == [[3.0, 9.0]]
    assert pw.toggles == []
    assert pw.dirty_calls == 1


# ------------------------------------------------ 时间修正：页遍历 + 分级


class _RecordingPW:
    def __init__(self):
        self.calls: list[tuple[float, float, bool]] = []
        self.view_box = SimpleNamespace(
            viewRange=lambda: [(0.0, 10.0)],
            enableAutoRange=lambda **kwargs: None,
            setXRange=lambda *args, **kwargs: None,
        )

    def update_time_correction(self, factor, offset, *, scene=True):
        self.calls.append((factor, offset, scene))


class _Container:
    def __init__(self, pw):
        self.plot_widget = pw

    def isHidden(self):
        return False


def test_layout_time_correction_traverses_all_pages_with_scene_flag():
    current_pw = _RecordingPW()
    hidden_pw = _RecordingPW()
    current_page = SimpleNamespace(plot_widgets=[_Container(current_pw)])
    hidden_page = SimpleNamespace(plot_widgets=[_Container(hidden_pw)])

    class _Host:
        def __init__(self):
            self.factor = 1.0
            self.offset = 0.0
            self.cursor_btn = QPushButton()
            self.cursor_btn.setCheckable(True)
            self.cursor_mode = "1 free cursor"
            self.pinned_x_values = []
            self._time_correction_pinned_index_values = []
            self._is_time_correction_active = False
            self.plot_widgets = current_page.plot_widgets
            self.pages = [current_page, hidden_page]
            self.current_page = current_page
            self.loader = SimpleNamespace(datalength=N)
            self.mark_stats_window = None
            self.cursor_sync_manager = SimpleNamespace(
                _realign_pinned_cursor_after_time_correction=lambda *a: None
            )
            self.announcements = []

        def _update_axis_segment(self):
            pass

        def _broadcast(self, message):
            self.announcements.append(message)

    host = _Host()
    manager = LayoutManager(host)
    host._kept = manager  # 反向撑住弱引用

    assert manager.apply_time_correction(2.0, 5.0) is True

    assert current_pw.calls == [(2.0, 5.0, True)]
    assert hidden_pw.calls == [(2.0, 5.0, False)]
    assert host.factor == 2.0
    assert host.offset == 5.0


# --------------------------------------- PlotDataManager 的 scene=False


class _FakeLoader:
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


class _FakeCtx:
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


class _FakeCursorSync:
    def _sync_min_xrange(self):
        pass


class _FakeLayoutManager:
    def request_mark_stats_refresh(self, immediate: bool = False):
        pass


class _FakeHost(QWidget):
    def __init__(self):
        super().__init__()
        self.cursor_btn = QPushButton()
        self.cursor_btn.setCheckable(True)
        self.cursor_btn.setChecked(False)
        self.cursor_values_hidden = False
        self.cursor_sync_manager = _FakeCursorSync()
        self.layout_manager = _FakeLayoutManager()
        self.cursor_mode = "1 free cursor"
        self.pinned_x_values = []
        self._is_loading_new_data = False
        self._is_syncing_crosshair = False
        self._pending_crosshair_x = None
        self._crosshair_update_timer = QTimer(self)
        self.plot_widgets = []
        self.loader = None
        self._baseline_density = 0.0
        self._global_max_density = 0.0


@pytest.fixture()
def real_plot(qapp):
    """一个已画出曲线的真 PlotWidget + 假主窗口/上下文。"""
    host = _FakeHost()
    df = pd.DataFrame({"a": np.arange(N, dtype=float) % 5})
    loader = _FakeLoader(df)
    ctx = _FakeCtx(loader)
    pw = DraggableGraphicsLayoutWidget({}, df)
    pw.plot_context = ctx
    pw.setParent(host)
    pw.resize(640, 260)
    pw.show()

    container = _Container(pw)
    host.plot_widgets.append(container)
    ctx.plot_widgets = list(host.plot_widgets)
    host.loader = loader

    page = _PageDirty()
    pw.page = page
    assert pw.plot_variable("a")

    yield pw, page, loader

    pw._is_being_destroyed = True
    pw.deleteLater()
    host.deleteLater()


def _x_limits(pw) -> list:
    return list(pw.view_box.state["limits"]["xLimits"])


def test_time_correction_scene_false_keeps_limits_and_recomputes_data(real_plot):
    pw, page, _loader = real_plot
    ci = next(iter(pw.curves.values()))
    original_index = np.array(ci.original_index, copy=True)
    old_limits = _x_limits(pw)

    pw.update_time_correction(0.5, 10.0, scene=False)

    # 数据层：曲线 x 已按新基准重算
    assert ci.x_data is not None
    np.testing.assert_allclose(
        np.asarray(ci.x_data[:5]), 10.0 + 0.5 * original_index[:5], rtol=1e-6
    )
    # 场景层：X limits 未动，页被标脏等激活补建
    assert _x_limits(pw) == old_limits
    assert page.dirty_calls >= 1


def test_time_correction_scene_true_applies_limits(real_plot):
    """正向对照：当前页照旧立即写场景（limits 由唯一 oracle 给出）。"""
    pw, _page, loader = real_plot
    pw.factor = 1.0
    pw.offset = 0.0

    pw.update_time_correction(0.5, 10.0, scene=True)

    expected = compute_global_x_limits(loader, factor=pw.factor, offset=pw.offset)
    assert expected is not None
    _, _, limits_xmin, limits_xmax = expected
    assert _x_limits(pw) == pytest.approx([limits_xmin, limits_xmax])
