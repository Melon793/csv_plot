"""全局 X 域回退与 reload 重建的护栏（对应 code review D3 / D4）

本区间把「无可见曲线时的回退 X 域」从 index 空间裸值改成
``AxisManager.cursor_x_domain()``（已套当前 factor/offset），注释自己列出了三个
用户可见后果（「全部自适应」、布局换行时的 X 同步、reload 的 pinned 钳制），
但这条修复的主调用点 ``collect_global_x_range`` 当时一行测试都没有；reload 重建
路径 ``replots_after_loading`` 同样 0 覆盖，而它把 ±5% padding 又算了一遍。

这里用**真实** ``CursorSyncManager`` + 最小宿主替身把两条都钉住：

1. 有可见曲线 → 用曲线域（正向对照，证明没改坏主路径）；
2. 无可见曲线 + ``factor=0.1`` → ``(0.1, 10.0)``，不是 ``(1.0, 100.0)``；
3. loader 无效 → ``(None, None)`` 且不抛异常；
4. reload 后 vline bounds 与 ViewBox xLimits 都必须等于
   ``compute_global_x_limits`` 给出的那一对 —— 它是唯一 oracle，谁在自己那边
   手写一遍 padding 就会与它漂移。

宿主替身形状沿用 ``test_cursor_x_domain.py``（那里只测 PlotWidget 一侧，
``FakeCursorSync`` 是 no-op，真实 manager 从未被构造过）。
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox, QPushButton, QWidget

from src.core.config import compute_global_x_limits
from src.ui.cursor_sync_manager import CursorSyncManager
from src.ui.table_dialog import DataTableDialog
from src.ui.widgets.plot_widget import DraggableGraphicsLayoutWidget

N = 100


class FakeLoader:
    """CSV loader 的最小替身：只提供 X 域计算与 reload 变量校验用到的字段。

    ``global_time_range`` 与 ``src/data/base_loader.py`` 保持一致（CSV 场景是
    index 空间，即未套 factor/offset 的裸值）—— 这正是回退域不能直接用它的原因。
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

    ``value_cache`` / ``_enum_text_maps`` 透传到宿主（真实 PlotContext 也是这么
    读的，见 ``src/app/plot_context.py:146-151``）：reload 会把它们重绑成新
    ``OrderedDict``，替身若各存一份，widget 就会读到旧缓存而与生产分叉。
    """

    def __init__(self, host, loader):
        self._host = host
        self.loader = loader
        self.plot_widgets = []
        self.pinned_x_values = []
        self.cursor_mode = "1 free cursor"
        self.cleared_announces = []

    @property
    def value_cache(self):
        return self._host.value_cache

    @value_cache.setter
    def value_cache(self, value):
        self._host.value_cache = value

    @property
    def _enum_text_maps(self):
        return self._host._enum_text_maps

    @_enum_text_maps.setter
    def _enum_text_maps(self, value):
        self._host._enum_text_maps = value

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

    def announce_cleared(self, label: str, curves: int) -> None:
        self.cleared_announces.append((label, curves))


class FakeLayoutManager:
    def request_mark_stats_refresh(self, immediate: bool = False):
        pass


class FakeMainWindow(QWidget):
    """replots_after_loading / collect_global_x_range 走到的宿主成员。"""

    def __init__(self):
        super().__init__()
        self.cursor_btn = QPushButton()
        self.cursor_btn.setCheckable(True)
        self.cursor_btn.setChecked(True)
        self.cursor_values_hidden = False
        self.layout_manager = FakeLayoutManager()
        self.cursor_mode = "1 free cursor"
        self.pinned_x_values = []
        self._is_loading_new_data = False
        self._is_syncing_crosshair = False
        self._pending_crosshair_x = None
        self._crosshair_update_timer = QTimer(self)
        self.plot_widgets = []
        self.loader = None
        self.value_cache = OrderedDict()
        self._enum_text_maps = {}
        self._baseline_density = 0.0
        self._global_max_density = 0.0
        self.cursor_sync_manager = None  # 真 manager 建好后回填


class Container:
    def __init__(self, pw):
        self.plot_widget = pw

    def isVisible(self):
        return True


@pytest.fixture(autouse=True)
def _no_data_table_dialog(monkeypatch):
    """reload 会把打开着的数据表列名并进 unique_y_names，压低匹配度走进另一分支"""
    monkeypatch.setattr(DataTableDialog, "_instance", None)


@pytest.fixture(autouse=True)
def modal_records(monkeypatch):
    """模态弹窗一律替身化：offscreen 下 exec() 会永久阻塞（tests/README.md 陷阱 #1）。

    reload 路径在「所有变量无效」时会弹 ``QMessageBox.information``，绘图路径会弹
    重复变量 warning —— 记下来，用例断言它们没被触发。
    """
    shown = []

    def make_recorder(kind):
        def fake(parent, title, text, *args, **kwargs):
            shown.append((kind, title, text))
            return QMessageBox.StandardButton.Ok
        return fake

    for kind in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(make_recorder(kind)))
    return shown


@pytest.fixture()
def env(qapp):
    """真实 CursorSyncManager + 一个已挂到宿主上的可绘图 plot widget"""
    host = FakeMainWindow()
    df = pd.DataFrame(
        {
            "a": np.arange(N, dtype=float) % 7,
            "b": np.sin(np.arange(N, dtype=float) / 9.0),
        }
    )
    loader = FakeLoader(df)
    ctx = FakeCtx(host, loader)
    pw = DraggableGraphicsLayoutWidget({}, df)
    pw.plot_context = ctx
    pw.setParent(host)
    pw.resize(700, 260)
    pw.show()

    container = Container(pw)
    host.plot_widgets.append(container)
    ctx.plot_widgets = list(host.plot_widgets)
    host.loader = loader

    manager = CursorSyncManager(host)
    host.cursor_sync_manager = manager

    yield host, manager, pw, loader

    pw._is_being_destroyed = True
    pw.deleteLater()
    host.deleteLater()


def _x_limits(pw) -> list:
    return list(pw.view_box.state["limits"]["xLimits"])


def _expected(loader, pw):
    """唯一 oracle：padding 只在 compute_global_x_limits 里算一次"""
    return compute_global_x_limits(loader, factor=pw.factor, offset=pw.offset)


def test_visible_curves_still_win(env):
    """正向对照：有可见曲线时用曲线域，回退分支不得抢主路径"""
    _host, manager, pw, loader = env
    assert pw.plot_variable("a")
    # 造一条只覆盖 index 10..29 的短曲线，与全局域 1..100 区分开
    ci = next(iter(pw.curves.values()))
    ci.x_data = np.arange(10.0, 30.0, dtype=np.float32)
    ci.update_x_range()

    assert manager.collect_global_x_range() == pytest.approx((10.0, 29.0))
    assert loader.datalength == N


def test_no_visible_curve_falls_back_to_corrected_domain(env):
    """D3 护栏本体：全部隐藏时回退域必须已套 factor，不是 index 空间裸值

    修复前这里返回 (1.0, 100.0)，于是「全部自适应」把视窗钉回未修正的时间轴。
    """
    _host, manager, pw, _loader = env
    pw.update_time_correction(0.1, 0.0)
    assert pw.plot_variable("a")
    for ci in pw.curves.values():
        ci.visible = False

    assert manager.collect_global_x_range() == pytest.approx((0.1, 10.0))


def test_empty_plot_falls_back_to_global_domain(env):
    """从未绘图（curves 为空）同样走回退，且回退值来自全局域而非 (None, None)"""
    _host, manager, pw, _loader = env
    pw.reset_plot(1, N)

    assert manager.collect_global_x_range() == pytest.approx((1.0, float(N)))


def test_no_loader_returns_none_pair(env):
    """loader 无效时不得抛异常，返回 (None, None) 让调用方自行回退"""
    _host, manager, pw, _loader = env
    pw.plot_context.loader = None

    assert manager.collect_global_x_range() == (None, None)


def test_domain_and_limits_leaves_limits_alone_without_loader(env):
    """无权威源时的契约：bounds 放开成无界，xLimits 保持现状不动。

    reload 路径依赖这一条 —— 拿不到全局域时写一对 None 进 limits 会把视窗约束
    整个抹掉，比什么都不做更糟。
    """
    _host, _manager, pw, _loader = env
    pw.reset_plot(1, N)
    before = _x_limits(pw)
    assert before[0] is not None, "前置条件：得先有一对真 limits 才测得出「不动」"

    pw.plot_context.loader = None

    assert pw._apply_cursor_x_domain_and_limits() is None
    assert list(pw.vline.bounds()) == [None, None]
    assert _x_limits(pw) == before


def test_reload_lands_limits_and_bounds_from_single_source(env, modal_records):
    """D4 护栏：reload 重建后 xLimits 必须等于 compute_global_x_limits 那一对

    旧实现在 reload 路径里手写 ``min_x - DEFAULT_PADDING_VAL_X * span``，与
    config 那份是同一公式的两份实现；一旦只改一侧，这里就会失败。
    """
    _host, manager, pw, loader = env
    assert pw.plot_variable("a")
    pw.update_time_correction(0.1, 0.0)

    manager.replots_after_loading()

    assert not modal_records, f"变量全有效，reload 不该弹「更新通知」: {modal_records}"
    expected = _expected(loader, pw)
    assert expected is not None
    assert _x_limits(pw) == pytest.approx([expected[2], expected[3]])
    # 游标域是裸数据域（不含 padding），与 limits 同源同一次计算
    assert list(pw.vline.bounds()) == pytest.approx([expected[0], expected[1]])
    # 曲线确实被重建回来了，不是靠清空蒙对
    assert list(pw.curves) == ["a"]
