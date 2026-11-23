"""X 轴锁定 C1 component 测试（offscreen）：锁定核心状态与鼠标拦截。

覆盖施工方案 §2 C1 的断言清单：
- 锁定后滚轮 / zoom_x no-op 且 viewRange 不变（D1：不降级 Y）；
- 鼠标掩码 == [False, True]，Y 平移/缩放能力保留；
- 兄弟子图（XLink 组）范围不变；
- B1：Shift 框选 release 锁定时只写 Y、X 不动（D13）；
- B2：getMenu 锁定态藏 "X axis" 子菜单与 "View All"，Y axis 保留，
  checkable「锁定X轴」checked 随标志同步且判重不翻倍；
- set_x_axis_locked 广播含非当前页、同状态 no-op、解锁逐图恢复、
  自动解锁带 reason 播报（真实方法 + 最小宿主替身，不复制实现）；
- 新建子图继承锁定（plot_page 补种，含增量补格）。
"""

import types

import pandas as pd
import pytest

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent

from src.ui.main_window import MainWindow
from src.ui.widgets.custom_viewbox import ZH_LOCK_X
from src.ui.widgets.plot_container import PlotContainerWidget
from src.ui.widgets.plot_page import PlotPage
from tests.fixtures.waits import flush_deferred_deletes


# ---------- 事件构造辅助（与 test_viewbox_zoom 同口径） ----------

def _wheel_event(pw, local_pos: QPoint, delta_y: int) -> QWheelEvent:
    pos = QPointF(local_pos)
    return QWheelEvent(
        pos,
        QPointF(pw.mapToGlobal(local_pos)),
        QPoint(0, 0),
        QPoint(0, delta_y),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def _mouse_event(event_type, pw, local_pos: QPoint, button, buttons, modifiers):
    pos = QPointF(local_pos)
    return QMouseEvent(
        event_type, pos, QPointF(pw.mapToGlobal(local_pos)), button, buttons, modifiers
    )


class _FakeMenuEvent:
    """只提供 getMenu 用到的 scenePos() 的伪事件"""

    def __init__(self, x=0.0, y=0.0):
        self._pos = QPointF(x, y)

    def scenePos(self):
        return self._pos


def _apply_lock(pw, locked: bool) -> None:
    """单图落地锁定（与 MainWindow.set_x_axis_locked 的逐图动作同口径）"""
    pw._x_mouse_locked = locked
    pw.view_box.setMouseEnabled(x=not locked, y=True)


@pytest.fixture()
def locked_plot(plot_factory, qapp):
    pw = plot_factory(pd.DataFrame({"a": [float(i) for i in range(100)]}))
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    assert pw.plot_variable("a")
    pw.view_box.setXRange(1, 100, padding=0)
    pw.view_box.setYRange(-5, 105, padding=0)
    qapp.processEvents()
    _apply_lock(pw, True)
    return pw


# ---------- D1：滚轮 / zoom_x no-op ----------

def test_locked_wheel_is_noop(locked_plot, qapp):
    pw = locked_plot
    before = [list(axis) for axis in pw.view_box.viewRange()]

    pw.wheelEvent(_wheel_event(pw, QPoint(400, 300), 120))
    pw.wheelEvent(_wheel_event(pw, QPoint(400, 300), -120))
    qapp.processEvents()

    assert [list(axis) for axis in pw.view_box.viewRange()] == before, (
        "锁定后滚轮仍改了范围（D1 要求 no-op，也不得降级成 Y 缩放）"
    )


def test_locked_zoom_x_direct_noop(locked_plot, qapp):
    pw = locked_plot
    before = list(pw.view_box.viewRange()[0])

    pw._axis_manager.zoom_x(0.7, 50.0)
    qapp.processEvents()

    assert list(pw.view_box.viewRange()[0]) == before


def test_lock_mask_disables_x_keeps_y(locked_plot):
    vb = locked_plot.view_box
    assert [bool(v) for v in vb.state["mouseEnabled"]] == [False, True]
    # Y 轴 autoRange 能力不受锁定影响（Ctrl+Y 的「Y 跟随可见段」仍可用）
    locked_plot.auto_y_in_x_range()
    assert bool(vb.state["autoRange"][1]) is True


def test_locked_sibling_ranges_unchanged(plot_factory, qapp):
    """XLink 组整组锁定：在 master 上滚轮，两图范围都不得变。"""
    df = pd.DataFrame({"a": [float(i) for i in range(100)]})
    master = plot_factory(df)
    child = plot_factory(df)
    for pw in (master, child):
        pw.resize(800, 300)
        pw.show()
        assert pw.plot_variable("a")
        pw.view_box.setXRange(1, 100, padding=0)
    qapp.processEvents()
    child.view_box.setXLink(master.view_box)
    for pw in (master, child):
        _apply_lock(pw, True)
    before_m = list(master.view_box.viewRange()[0])
    before_c = list(child.view_box.viewRange()[0])

    master.wheelEvent(_wheel_event(master, QPoint(400, 150), 120))
    qapp.processEvents()

    assert list(master.view_box.viewRange()[0]) == before_m
    assert list(child.view_box.viewRange()[0]) == before_c


# ---------- B1 / D13：框选 release 只写 Y ----------

def test_locked_box_zoom_writes_y_only(locked_plot, qapp):
    pw = locked_plot
    vb = pw.view_box
    x_before = list(vb.viewRange()[0])
    pa, pb = QPoint(100, 100), QPoint(400, 350)
    shift = Qt.KeyboardModifier.ShiftModifier
    v1 = vb.mapSceneToView(pw.mapToScene(pa))
    v2 = vb.mapSceneToView(pw.mapToScene(pb))
    y_min, y_max = sorted((v1.y(), v2.y()))
    dy = y_max - y_min
    expected_y = (y_min - 0.1 * dy, y_max + 0.1 * dy)

    pw.mousePressEvent(_mouse_event(
        QEvent.Type.MouseButtonPress, pw, pa,
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, shift))
    assert pw.rubberBand.isVisible()
    pw.mouseMoveEvent(_mouse_event(
        QEvent.Type.MouseMove, pw, pb,
        Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, shift))
    pw.mouseReleaseEvent(_mouse_event(
        QEvent.Type.MouseButtonRelease, pw, pb,
        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, shift))
    qapp.processEvents()

    assert not pw.rubberBand.isVisible()
    assert list(vb.viewRange()[0]) == x_before, "锁定后框选仍写了 X（B1 旁路）"
    assert vb.viewRange()[1][0] == pytest.approx(expected_y[0], rel=1e-6)
    assert vb.viewRange()[1][1] == pytest.approx(expected_y[1], rel=1e-6)


# ---------- B2：getMenu 锁定态可见性 + L1 checked 同步 ----------

@pytest.fixture()
def menu_vb(plot_factory):
    pw = plot_factory()
    ctx = pw.plot_context
    ctx.is_cursor_enabled = lambda: False
    ctx.cursor_mode = "1 free cursor"
    ctx.cursor_values_hidden = False
    ctx.get_row_height = lambda row: 100
    ctx._plot_col_max_default = 1
    ctx.plot_widgets = []
    return pw.view_box


def _action(menu, text):
    return next(a for a in menu.actions() if a.text() == text)


def test_getmenu_locked_hides_x_axis_and_view_all(menu_vb):
    menu_vb.plot_widget._x_mouse_locked = True
    menu = menu_vb.getMenu(_FakeMenuEvent())

    assert not _action(menu, "X axis").isVisible(), "X axis 子菜单会静默解锁（B2）"
    assert not _action(menu, "View All").isVisible(), "View All 会重置 X 范围"
    assert _action(menu, "Y axis").isVisible(), "Y axis 必须保留"
    lock_act = _action(menu, ZH_LOCK_X)
    assert lock_act.isCheckable() and lock_act.isChecked()


def test_getmenu_unlocked_restores_and_syncs_checked(menu_vb):
    menu_vb.plot_widget._x_mouse_locked = True
    menu = menu_vb.getMenu(_FakeMenuEvent())
    assert not _action(menu, "X axis").isVisible()

    menu_vb.plot_widget._x_mouse_locked = False
    menu = menu_vb.getMenu(_FakeMenuEvent())
    assert _action(menu, "X axis").isVisible()
    assert _action(menu, "View All").isVisible()
    assert not _action(menu, ZH_LOCK_X).isChecked()
    # 判重：两次 getMenu 不得插入第二份「锁定X轴」
    assert sum(a.text() == ZH_LOCK_X for a in menu.actions()) == 1


def test_getmenu_flag_none_is_treated_unlocked(menu_vb):
    """plot_widget 注入前标志为 None：getattr 防 None，按未锁定处理。"""
    menu_vb.plot_widget = None
    menu = menu_vb.getMenu(_FakeMenuEvent())
    assert _action(menu, "X axis").isVisible()
    assert not _action(menu, ZH_LOCK_X).isChecked()


# ---------- 广播：真实 set_x_axis_locked + 最小宿主替身 ----------

class _BroadcastHost:
    """承载真实 MainWindow.set_x_axis_locked / iter_all_plot_widgets 的最小宿主。

    只替身播报落点与缺席的 UI 件（getattr 防御分支），遍历/掩码/标志/文案
    全部走真实实现；pages 含两页，验证广播覆盖非当前页。
    """

    def __init__(self, pages):
        self.pages = pages
        self._x_axis_locked = False
        self.announcements = []
        self.iter_all_plot_widgets = types.MethodType(
            MainWindow.iter_all_plot_widgets, self
        )

    #: 文案函数走真实实现（staticmethod 不依赖实例状态）
    _x_lock_announce_text = staticmethod(MainWindow._x_lock_announce_text)

    def _broadcast(self, message, level="info"):
        self.announcements.append(message)


@pytest.fixture()
def broadcast_host(plot_factory):
    df = pd.DataFrame({"a": [float(i) for i in range(10)]})
    pages = []
    pws = []
    for _ in range(2):  # 两页：当前页 + 非当前页都要被广播
        pw = plot_factory(df)
        container = PlotContainerWidget(pw)
        container.plot_widget = pw
        pages.append(types.SimpleNamespace(plot_widgets=[container]))
        pws.append(pw)
    host = _BroadcastHost(pages)
    host.pws = pws
    return host


def test_broadcast_locks_all_pages_and_unlock_restores(broadcast_host):
    host = broadcast_host
    MainWindow.set_x_axis_locked(host, True)

    assert host._x_axis_locked is True
    for pw in host.pws:
        assert pw._x_mouse_locked is True
        assert [bool(v) for v in pw.view_box.state["mouseEnabled"]] == [False, True]
    assert host.announcements == [MainWindow._x_lock_announce_text(True)]

    # 同状态重复调用 no-op：不重复播报（reset_plot 类逐子图触发的护栏）
    MainWindow.set_x_axis_locked(host, True)
    assert len(host.announcements) == 1

    MainWindow.set_x_axis_locked(host, False)
    for pw in host.pws:
        assert pw._x_mouse_locked is False
        assert [bool(v) for v in pw.view_box.state["mouseEnabled"]] == [True, True]
    assert host.announcements[-1] == MainWindow._x_lock_announce_text(False)


def test_auto_unlock_announcement_carries_reason(broadcast_host):
    host = broadcast_host
    MainWindow.set_x_axis_locked(host, True)
    MainWindow.set_x_axis_locked(host, False, reason="数据已重载")
    assert host.announcements[-1] == "X 轴已自动解锁（数据已重载）"
    # 未锁定时的自动解锁调用是 no-op，不播报
    MainWindow.set_x_axis_locked(host, False, reason="数据已重载")
    assert len(host.announcements) == 2


# ---------- 建图补种：新建子图继承锁定 ----------

class _FakeButton:
    def __init__(self, checked=False):
        self._checked = checked

    def isChecked(self):
        return self._checked


class _LockServices:
    """PlotPage.build_matrix 所需最小服务集合 + 锁定标志（透传 MainWindow 口径）"""

    def __init__(self, x_locked: bool):
        self.units = {}
        self.data = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
        self.time_channels_infos = {}
        self.cursor_btn = _FakeButton(False)
        self.cursor_values_hidden = False
        self.cursor_mode = "1 free cursor"
        self.pinned_x_values = []
        self.mark_region_btn = _FakeButton(False)
        self.loader = None
        self.value_cache = {}
        self._enum_text_maps = {}
        self.factor = 1.0
        self.offset = 0.0
        self.saved_mark_range = None
        self._global_max_density = 0.0
        self._data_version = 0
        self._is_loading_new_data = False
        self._is_time_correction_active = False
        self._x_axis_locked = x_locked

    def request_mark_stats_refresh(self, immediate: bool = False):
        pass


@pytest.fixture()
def lock_page_factory(qapp, qtbot):
    pages = []

    def _make(x_locked: bool):
        page = PlotPage(
            parent=None, title="图 1",
            rows_max=4, cols_max=3, rows_current=1, cols_current=1,
        )
        page.services = _LockServices(x_locked)
        qtbot.addWidget(page.widget)
        pages.append(page)
        return page

    yield _make
    for page in pages:
        page.dispose()
    flush_deferred_deletes()


def test_new_plot_inherits_lock(lock_page_factory):
    page = lock_page_factory(True)
    page.build_matrix(1, 2)
    assert len(page.plot_widgets) == 2
    for container in page.plot_widgets:
        pw = container.plot_widget
        assert pw._x_mouse_locked is True
        assert [bool(v) for v in pw.view_box.state["mouseEnabled"]] == [False, True]


def test_new_plot_unlocked_by_default(lock_page_factory):
    page = lock_page_factory(False)
    page.build_matrix(1, 1)
    pw = page.plot_widgets[0].plot_widget
    assert pw._x_mouse_locked is False
    assert [bool(v) for v in pw.view_box.state["mouseEnabled"]] == [True, True]


def test_grown_cells_inherit_lock(lock_page_factory):
    """增量补格（改布局）走同一建图函数，补种必须同样覆盖。"""
    page = lock_page_factory(True)
    page.build_matrix(1, 1)
    page.cols_current = 3
    page.ensure_matrix()  # 只补缺失格，不重建既有子图
    assert len(page.plot_widgets) == 3
    for container in page.plot_widgets:
        assert container.plot_widget._x_mouse_locked is True