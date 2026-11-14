"""页签栏右端暂存区胶囊（方案 A）：几何 / 显隐 / 落点 / 徽章 / 点击。

覆盖施工计划 §7.1 的 15 条 component 用例。落点用例不绕开真实事件链：
``capsule.event(QDropEvent)`` 直投 → ``shelf_drop_requested`` → 托盘
``dropEvent``，只把 MainWindow 那段接线在夹具里等价接上。
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRect, Qt
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QApplication, QLabel, QWidget

import pytest

from src.core.data_types import (
    DRAW_MODE_SOURCE_AUTO,
    DRAW_MODE_STEP_PRE,
    SYMBOL_POLICY_ON,
)
from src.ui.drag_drop import (
    build_legend_var_mimedata,
    build_shelf_mimedata,
    build_var_mimedata,
    clear_active_legend_drag,
    set_active_legend_drag,
)
from src.ui.widgets.curve_shelf import CurveShelfWidget, ShelfEntry
from src.ui.widgets.plot_tabs import PlotTabsWidget

# 手工构造的 QDropEvent 不接管 mimeData 所有权，必须外部保活
_keep_alive: list = []


class ShelfHost(QWidget):
    """最小宿主：加载闸门 + 播报记录器。"""

    def __init__(self):
        super().__init__()
        self.messages: list[str] = []
        self.loading = False

    def _show_status_message(self, message: str) -> None:
        self.messages.append(message)

    def is_data_loading(self) -> bool:
        return self.loading

    def reject_when_loading(self, action: str) -> bool:
        if not self.loading:
            return False
        self.messages.append(f"正在加载数据，请稍候再{action}")
        return True


def _make_tabs(qtbot, pages: int = 1, width: int = 900) -> PlotTabsWidget:
    tabs = PlotTabsWidget()
    qtbot.addWidget(tabs)
    tabs.resize(width, 360)
    for i in range(pages):
        tabs.addTab(QLabel(f"page {i}"), f"页面页{i + 1}")
    tabs.set_new_page_enabled(True)
    tabs.show()
    QApplication.processEvents()
    return tabs


def _rect_in(widget: QWidget, base: QWidget) -> QRect:
    return QRect(widget.mapTo(base, widget.rect().topLeft()), widget.size())


def _drop(widget: QWidget, mime, *, ctrl: bool = False) -> QDropEvent:
    _keep_alive.append(mime)
    mods = (
        Qt.KeyboardModifier.ControlModifier
        if ctrl
        else Qt.KeyboardModifier.NoModifier
    )
    event = QDropEvent(
        QPointF(10.0, 10.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        mods,
    )
    widget.event(event)
    return event


@pytest.fixture()
def capsule_shelf(qtbot):
    """真实"胶囊 → 信号 → 托盘"链路；MainWindow 的接线在夹具里等价接上。"""
    tabs = _make_tabs(qtbot)
    host = ShelfHost()
    shelf = CurveShelfWidget(main_window=host)
    shelf.show()
    tabs.shelf_drop_requested.connect(shelf.dropEvent)
    shelf.entries_changed.connect(
        lambda: tabs.set_shelf_count(shelf.count())
    )
    yield tabs, shelf, host
    shelf.close()
    shelf.deleteLater()
    host.close()
    host.deleteLater()


# ---------------------------------------------------------------------------
# 几何 / 显隐
# ---------------------------------------------------------------------------


def test_capsule_sizehint_matches_fixed_size(capsule_shelf):
    tabs, _shelf, _host = capsule_shelf
    capsule = tabs.shelf_capsule
    assert capsule is not None
    assert capsule.sizeHint() == capsule.minimumSizeHint()
    assert capsule.sizeHint().width() == capsule.width() == 36
    assert capsule.sizeHint().height() == capsule.height() == 20
    assert capsule.acceptDrops() is True


def test_capsule_hidden_before_gate(qtbot):
    tabs = PlotTabsWidget()
    qtbot.addWidget(tabs)
    tabs.resize(900, 360)
    tabs.addTab(QLabel("page"), "页面页1")
    tabs.show()
    QApplication.processEvents()
    assert tabs.shelf_capsule is not None
    # G3：角位控件不跟随页签栏隐藏，必须显式藏住
    assert tabs.tabBar().isVisible() is False
    assert tabs.shelf_capsule.isVisible() is False
    tabs.close()


def test_capsule_hidden_with_tabbar(capsule_shelf):
    tabs, _shelf, _host = capsule_shelf
    capsule = tabs.shelf_capsule
    assert capsule.isVisible() is True
    tabs.set_new_page_enabled(False)
    QApplication.processEvents()
    assert tabs.tabBar().isVisible() is False
    assert capsule.isVisible() is False
    tabs.set_new_page_enabled(True)
    QApplication.processEvents()
    assert capsule.isVisible() is True


def test_capsule_returns_width_when_hidden(capsule_shelf):
    tabs, _shelf, _host = capsule_shelf
    bar = tabs.tabBar()
    capsule = tabs.shelf_capsule
    with_capsule = bar.width()
    capsule.hide()
    QApplication.processEvents()
    without_capsule = bar.width()
    # G2：36px 记在尺寸契约里，隐藏后必须全额归还（实测正好 36）
    assert without_capsule - with_capsule == 36
    assert capsule.sizeHint().width() == 36
    capsule.show()
    QApplication.processEvents()
    assert bar.width() == with_capsule


def test_capsule_vertically_centered_in_tabbar(capsule_shelf):
    """平台样式摆放角位控件时贴顶/贴底不一，胶囊必须按页签栏高度居中。"""
    tabs, _shelf, _host = capsule_shelf
    capsule = tabs.shelf_capsule
    bar = tabs.tabBar()
    assert abs(capsule.geometry().center().y() - bar.rect().center().y()) <= 1


@pytest.mark.parametrize("pages,width", [(3, 360), (8, 420), (8, 900)])
def test_capsule_never_overlaps_plus(qtbot, pages, width):
    tabs = _make_tabs(qtbot, pages=pages, width=width)
    plus_rect = _rect_in(tabs.new_page_button, tabs)
    cap_rect = _rect_in(tabs.shelf_capsule, tabs)
    # G4：窄窗 360 只剩 2px 余量，任何重叠/出界都算失败
    assert plus_rect.intersects(cap_rect) is False
    assert plus_rect.right() < cap_rect.left()
    assert cap_rect.right() < tabs.width()


@pytest.mark.parametrize("pages,width", [(8, 360), (8, 420), (8, 540)])
def test_plus_never_overlaps_scroll_arrows(qtbot, pages, width):
    tabs = _make_tabs(qtbot, pages=pages, width=width)
    bar = tabs.tabBar()
    plus_rect = _rect_in(tabs.new_page_button, tabs)
    arrows = [
        child
        for child in bar.children()
        if child.isWidgetType()
        and "scroll" in (child.objectName() or "").lower()
        and child.isVisible()
    ]
    assert arrows, "该参数下滚动箭头应可见，否则用例失去意义"
    for arrow in arrows:
        assert plus_rect.intersects(_rect_in(arrow, tabs)) is False


# ---------------------------------------------------------------------------
# 落点：变量列表 / legend 剪切 / 复制 / 回拖拒绝 / 加载拒绝
# ---------------------------------------------------------------------------


def test_drop_from_variable_list_adds_entries(capsule_shelf):
    tabs, shelf, _host = capsule_shelf
    hits = []
    tabs.shelf_drop_requested.connect(hits.append)

    event = _drop(tabs.shelf_capsule, build_var_mimedata(["a", "b"]))

    assert len(hits) == 1, "事件必须只转发一次（重复转发会引发二次入区）"
    assert event.isAccepted() is True
    assert shelf.count() == 2
    assert shelf.entry_for_name("a").origin_label == "变量列表"


def test_drop_from_legend_cut_removes_source(capsule_shelf, plot_factory, qapp):
    tabs, shelf, _host = capsule_shelf
    src = plot_factory()
    assert src.plot_variable("a")
    assert src.plot_variable("b")

    for name, ctrl in (("a", False), ("b", True)):
        mime = build_legend_var_mimedata([name], src)
        set_active_legend_drag(src, [name])
        try:
            _drop(tabs.shelf_capsule, mime, ctrl=ctrl)
        finally:
            clear_active_legend_drag()

    assert shelf.count() == 2
    qapp.processEvents()  # 源端删除是 singleShot(0)
    assert "a" not in src.curves  # 剪切：源删除
    assert "b" in src.curves  # Ctrl：复制语义保留源


def test_drop_preserves_style_snapshot(capsule_shelf, plot_factory):
    tabs, shelf, _host = capsule_shelf
    src = plot_factory()
    assert src.plot_variable("a")
    src._multi_curve_manager.apply_curve_color("a", "#123456")
    src._multi_curve_manager.apply_curve_draw_mode("a", DRAW_MODE_STEP_PRE)
    src.curves["a"].draw_mode_source = DRAW_MODE_SOURCE_AUTO
    src._multi_curve_manager.apply_curve_symbol_policy("a", SYMBOL_POLICY_ON)
    src._multi_curve_manager.toggle_curve_visibility_by_name("a")

    mime = build_legend_var_mimedata(["a"], src)
    set_active_legend_drag(src, ["a"])
    try:
        _drop(tabs.shelf_capsule, mime)
    finally:
        clear_active_legend_drag()

    entry = shelf.entry_for_name("a")
    assert entry is not None
    assert entry.color == "#123456"
    assert entry.visible is False
    assert entry.draw_mode == DRAW_MODE_STEP_PRE
    assert entry.draw_mode_source == DRAW_MODE_SOURCE_AUTO
    assert entry.symbol_policy == SYMBOL_POLICY_ON


def test_shelf_mime_dropped_back_is_rejected(capsule_shelf):
    tabs, shelf, _host = capsule_shelf
    hits = []
    tabs.shelf_drop_requested.connect(hits.append)

    mime = build_shelf_mimedata(["k1"], ["a"], shelf)
    event = _drop(tabs.shelf_capsule, mime)

    assert hits == []
    assert event.isAccepted() is False
    assert shelf.count() == 0


def test_drop_rejected_while_loading(capsule_shelf):
    tabs, shelf, host = capsule_shelf
    host.loading = True

    event = _drop(tabs.shelf_capsule, build_var_mimedata(["a"]))

    assert shelf.count() == 0
    assert event.isAccepted() is False
    assert any("加载中" in m for m in host.messages)


# ---------------------------------------------------------------------------
# 徽章 / 点击 / 加载闸门
# ---------------------------------------------------------------------------


def test_badge_tracks_count_not_signal_count(capsule_shelf):
    tabs, shelf, _host = capsule_shelf
    capsule = tabs.shelf_capsule

    _drop(capsule, build_var_mimedata(["a", "b", "c", "d", "e"]))

    assert shelf.count() == 5
    # M6：entries_changed 会按条触发多次，徽章必须读总数而不是累加
    assert capsule.count() == 5
    assert "5 条" in capsule.toolTip()


def test_badge_follows_clear_and_revalidate(capsule_shelf):
    tabs, shelf, _host = capsule_shelf
    capsule = tabs.shelf_capsule

    shelf.add_entry(ShelfEntry.from_name("a"))
    shelf.add_entry(ShelfEntry.from_name("b"))
    assert capsule.count() == 2

    shelf.clear_entries()
    assert capsule.count() == 0

    shelf.add_entry(ShelfEntry.from_name("a"))
    shelf.revalidate({"other"})  # 置灰不删条目
    assert shelf.count() == 1
    assert capsule.count() == 1


def test_capsule_toggle_emits_and_syncs_back(capsule_shelf, qtbot):
    tabs, _shelf, _host = capsule_shelf
    capsule = tabs.shelf_capsule
    received: list[bool] = []
    tabs.shelf_toggle_requested.connect(received.append)

    qtbot.mouseClick(capsule, Qt.MouseButton.LeftButton)
    assert received == [True]
    assert capsule.isChecked() is True

    # 浮窗被外部关掉时回写勾选态，且不得回发 toggle 造成闭环
    tabs.set_shelf_active(False)
    assert capsule.isChecked() is False
    assert received == [True]


def test_capsule_enabled_follows_load_gate(qtbot):
    """真实 FileLoaderManager.set_button_status 驱动（不直调胶囊 API）。"""
    from PySide6.QtWidgets import QPushButton

    from src.ui.file_loader_manager import FileLoaderManager

    class GateHost(QWidget):
        def __init__(self, tabs):
            super().__init__()
            self.plot_tabs = tabs
            self.curve_shelf_btn = QPushButton("暂存区", self)
            for name in (
                "clear_all_plots_btn",
                "auto_range_btn",
                "auto_y_btn",
                "cursor_btn",
                "mark_region_btn",
                "grid_layout_btn",
            ):
                setattr(self, name, QPushButton(name, self))

    tabs = _make_tabs(qtbot)
    host = GateHost(tabs)
    qtbot.addWidget(host)
    manager = FileLoaderManager(host)

    manager.set_button_status(False)
    assert tabs.shelf_capsule.isEnabled() is False
    manager.set_button_status(True)
    assert tabs.shelf_capsule.isEnabled() is True
