"""时间修正后的密度 / minXRange 回归。

两个缺口必须同时钉住：
1. ``CurveInfo.point_density`` 变了，但 ``pw._max_point_density`` 与
   ``mw._global_max_density`` 仍是旧值；
2. 批量修正循环内当前页按旧全局密度写过 xLimits，而 ``_set_min_x_range``
   只扩不缩——全局密度汇总后必须用权威 oracle 重写一次 limits。

数值口径：x = 1..N，span = N-1，density = N/span，minXRange = 3/density。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QPushButton, QWidget

from src.core.config import MIN_INDEX_LENGTH, compute_global_x_limits
from src.ui.cursor_sync_manager import CursorSyncManager
from src.ui.layout_manager import LayoutManager
from src.ui.widgets.plot_widget import DraggableGraphicsLayoutWidget

N = 100


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
    """生产 PlotContext 的最小替身；全局密度必须委托宿主，不能自己存影子值。"""

    def __init__(self, host: _FakeHost, loader: _FakeLoader):
        self.host = host
        self.loader = loader
        self.value_cache = {}
        self._enum_text_maps = {}
        self.plot_widgets: list[_Container] = []
        self.pinned_x_values: list[float] = []
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

    @property
    def _global_max_density(self) -> float:
        return self.host._global_max_density

    def request_mark_stats_refresh(self, immediate: bool = False):
        pass

    def announce_cleared(self, label: str, curves: int) -> None:
        pass


class _FakeLayoutManager:
    def request_mark_stats_refresh(self, immediate: bool = False):
        pass


class _FakeHost(QWidget):
    """真实 CursorSyncManager 的最小宿主；plot_widgets 即当前页容器。"""

    def __init__(self):
        super().__init__()
        self.cursor_btn = QPushButton()
        self.cursor_btn.setCheckable(True)
        self.cursor_btn.setChecked(False)
        self.cursor_values_hidden = False
        self.layout_manager = _FakeLayoutManager()
        self.cursor_mode = "1 free cursor"
        self.pinned_x_values = []
        self._time_correction_pinned_index_values = []
        self._is_time_correction_active = False
        self._is_loading_new_data = False
        self._is_syncing_crosshair = False
        self._pending_crosshair_x = None
        self._crosshair_update_timer = QTimer(self)
        self.plot_widgets = []
        self.pages = []
        self.current_page = None
        self.loader = None
        self._baseline_density = 0.0
        self._global_max_density = 0.0
        self.annotations = []
        self._factor_default = 1.0
        self._offset_default = 0.0
        self.factor = 1.0
        self.offset = 0.0
        self.mark_stats_window = None

    def _update_axis_segment(self):
        pass

    def _broadcast(self, message):
        self.annotations.append(message)


class _Container:
    def __init__(self, plot_widget):
        self.plot_widget = plot_widget

    def isHidden(self):
        return False


class _HiddenContainer:
    def __init__(self):
        self.refresh_calls = 0
        self.plot_widget = SimpleNamespace(
            _refresh_global_x_limits=self._refresh,
            _max_point_density=999.0,
        )

    def _refresh(self):
        self.refresh_calls += 1

    def isHidden(self):
        return True


def _min_x_range(pw) -> float | None:
    return pw.view_box.state["limits"]["xRange"][0]


def _x_limits(pw) -> list:
    return list(pw.view_box.state["limits"]["xLimits"])


@pytest.fixture()
def env(qapp):
    host = _FakeHost()
    df = pd.DataFrame({"a": np.arange(N, dtype=float) % 5})
    loader = _FakeLoader(df)
    ctx = _FakeCtx(host, loader)
    pw = DraggableGraphicsLayoutWidget({}, df)
    pw.plot_context = ctx
    pw.setParent(host)
    pw.resize(640, 260)
    pw.show()

    container = _Container(pw)
    host.plot_widgets.append(container)
    ctx.plot_widgets = list(host.plot_widgets)
    host.loader = loader

    host.cursor_sync_manager = CursorSyncManager(host)
    assert pw.plot_variable("a")

    # 首帧同步后基线即为 100/99；用例都以此为前提
    assert host._global_max_density == pytest.approx(N / (N - 1))

    yield host, ctx, pw, loader

    pw._is_being_destroyed = True
    pw.deleteLater()
    host.deleteLater()


@pytest.mark.parametrize("factor", [0.5, 2.0])
def test_time_correction_refreshes_local_global_density_and_min_range(env, factor):
    host, ctx, pw, _loader = env
    ci = next(iter(pw.curves.values()))

    pw.update_time_correction(factor, 0.0, scene=True)

    expected_density = N / ((N - 1) * factor)
    expected_min_range = MIN_INDEX_LENGTH / expected_density
    assert ci.point_density == pytest.approx(expected_density)
    assert pw._max_point_density == pytest.approx(expected_density)
    assert host._global_max_density == pytest.approx(expected_density)
    assert ctx._global_max_density == pytest.approx(expected_density)
    assert _min_x_range(pw) == pytest.approx(expected_min_range)


def test_time_correction_scene_false_recomputes_density_but_defers_scene(env):
    host, _ctx, pw, _loader = env
    ci = next(iter(pw.curves.values()))
    old_global = host._global_max_density
    old_limits = _x_limits(pw)

    pw.update_time_correction(0.5, 0.0, scene=False)

    expected_density = N / ((N - 1) * 0.5)
    assert ci.point_density == pytest.approx(expected_density)
    assert pw._max_point_density == pytest.approx(expected_density)
    # 隐藏页不写场景、也不汇总全局；留给协调者/激活
    assert host._global_max_density == pytest.approx(old_global)
    assert _x_limits(pw) == old_limits
    page = getattr(pw, "page", None)
    assert page is None or page._scene_dirty


def test_batch_time_correction_rewrites_canonical_limits(env):
    """50/100 行 + 0.01 是旧 minXRange 撑宽 limits 的触发区。

    直接调用 update_time_correction 会在 limits 之前 sync，天然 canonical；
    这里必须走 LayoutManager.apply_time_correction，才能钉住批量 reapply。
    """
    host, _ctx, pw, loader = env
    page = SimpleNamespace(plot_widgets=list(host.plot_widgets))
    host.pages = [page]
    host.current_page = page

    hidden = _HiddenContainer()
    host.plot_widgets.append(hidden)  # 只进 sync/refresh 的可见性过滤测试

    manager = LayoutManager(host)
    host._kept = manager  # LayoutManager 只持弱引用

    assert manager.apply_time_correction(0.01, 0.0) is True

    expected_density = N / ((N - 1) * 0.01)
    expected = compute_global_x_limits(loader, factor=0.01, offset=0.0)
    assert expected is not None
    assert pw._max_point_density == pytest.approx(expected_density)
    assert host._global_max_density == pytest.approx(expected_density)
    assert _min_x_range(pw) == pytest.approx(MIN_INDEX_LENGTH / expected_density)
    # 关键：不能被旧 minXRange（~2.97）撑宽；canonical 宽约 1.089
    assert _x_limits(pw) == pytest.approx([expected[2], expected[3]])
    # 显式隐藏单元格不参与本轮场景重写
    assert hidden.refresh_calls == 0
