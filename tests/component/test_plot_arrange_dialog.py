"""整理布局对话框 smoke 测试：icon 生成、选择、踢出/回填、门禁、换页、确认。"""

from __future__ import annotations

import pandas as pd
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QDialog

from src.core.plot_arrange_model import can_commit, ordered_ids
from src.ui.dialogs.plot_arrange_dialog import PlotArrangeDialog
from src.ui.widgets.plot_page import PlotPage
from tests.fixtures.waits import flush_deferred_deletes


class _FakeButton:
    def __init__(self, checked: bool = False):
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked


class _FakeLayoutManager:
    def request_mark_stats_refresh(self, *, immediate: bool = False):
        pass


class _FakeCursorSync:
    def collect_global_x_range(self):
        return (0.0, 1.0)

    def _sync_min_xrange(self):
        pass


class _FakeServices:
    def __init__(self):
        self.units = {}
        self.data = pd.DataFrame({"a": [0.0, 1.0]})
        self.time_channels_infos = {}
        self.cursor_btn = _FakeButton(False)
        self.cursor_values_hidden = False
        self.cursor_mode = "1 free cursor"
        self.pinned_x_values = []
        self.mark_region_btn = _FakeButton(False)
        self.layout_manager = _FakeLayoutManager()
        self.cursor_sync_manager = _FakeCursorSync()
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


def _set_curves(container, names):
    container.plot_widget.curves = {name: object() for name in names}


@pytest.fixture()
def pages(qapp, qtbot):
    page_a = PlotPage(
        parent=None, title="页1", rows_max=4, cols_max=3,
        rows_current=2, cols_current=2,
    )
    page_a.services = _FakeServices()
    page_a.build_matrix(2, 2)
    _set_curves(page_a.plot_widgets[0], ["a", "b"])
    _set_curves(page_a.plot_widgets[1], ["c"])
    _set_curves(page_a.plot_widgets[2], ["d"])

    page_b = PlotPage(
        parent=None, title="页2", rows_max=4, cols_max=3,
        rows_current=1, cols_current=2,
    )
    page_b.services = _FakeServices()
    page_b.build_matrix(1, 2)
    _set_curves(page_b.plot_widgets[0], ["e"])

    for page in (page_a, page_b):
        qtbot.addWidget(page.widget)
    yield page_a, page_b
    for page in (page_a, page_b):
        page.dispose()
    flush_deferred_deletes()


@pytest.fixture()
def dialog(pages):
    page_a, page_b = pages
    dlg = PlotArrangeDialog([page_a, page_b], current_page=page_a)
    yield dlg
    # 不要 qtbot.addWidget：pytest-qt 在 fixture 收尾**之前**关闭并 deleteLater
    # 已注册控件，与页夹具的 dispose 顺序冲突。这里只排队销毁，由页夹具收尾
    # 的 flush_deferred_deletes 统一落地。
    dlg.deleteLater()


def test_icons_and_gate_initial(dialog):
    state = dialog.result_state()
    assert len(ordered_ids(state.pages[0])) == 3
    # 仅当前页的图生成 icon（空位不算）
    assert len(dialog._board.icons) == 3
    assert can_commit(state) is True
    assert dialog._btn_ok.isEnabled() is True
    assert dialog._btn_kick.isEnabled() is False


def test_selection_kick_backfill_cycle(dialog):
    state = dialog.result_state()
    token = ordered_ids(state.pages[0])[0]

    dialog.click_token(token, Qt.KeyboardModifier.NoModifier)
    assert dialog._btn_kick.isEnabled() is True
    dialog._on_kick()
    assert state.tray == [token]
    assert can_commit(state) is False
    assert dialog._btn_ok.isEnabled() is False

    dialog._on_backfill()
    assert state.tray == []
    assert can_commit(state) is True
    assert dialog._btn_ok.isEnabled() is True


def test_ctrl_and_shift_multi_select(dialog):
    state = dialog.result_state()
    ids = ordered_ids(state.pages[0])
    dialog.click_token(ids[0], Qt.KeyboardModifier.NoModifier)
    dialog.click_token(ids[1], Qt.KeyboardModifier.ControlModifier)
    assert dialog._selection == {ids[0], ids[1]}

    dialog.click_token(ids[2], Qt.KeyboardModifier.ShiftModifier)
    # Shift 选区语义：无 Ctrl 时替换为锚点..目标区间
    assert dialog._selection == {ids[1], ids[2]}

    # Ctrl+Shift：区间并入现有选择
    dialog.click_token(ids[0], Qt.KeyboardModifier.NoModifier)
    dialog.click_token(ids[1], Qt.KeyboardModifier.ControlModifier)
    dialog.click_token(
        ids[2],
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    )
    assert dialog._selection == {ids[0], ids[1], ids[2]}


def test_reset_restores_initial(dialog):
    state = dialog.result_state()
    token = ordered_ids(state.pages[0])[0]
    dialog._selection = {token}
    dialog._on_kick()
    assert dialog.result_state().tray == [token]
    dialog._on_reset()
    assert dialog.result_state().tray == []
    assert dialog.result_state().pages[0].cells == dialog._initial.pages[0].cells


def test_tab_switch_changes_current_page(dialog):
    assert dialog.current_page_state().title == "页1"
    dialog._tabs.setCurrentIndex(1)
    assert dialog.current_page_state().title == "页2"
    assert len(dialog._board.icons) == 1


def test_board_mirrors_row_heights(pages):
    """行高按页面真实布局比例还原（跟行走，不随图迁移）。"""
    from tests.fixtures.waits import settle

    page_a, page_b = pages
    page_a.widget.resize(600, 420)
    page_a.widget.show()
    page_a.set_row_height(0, 50)
    page_a.set_row_height(1, 200)
    settle()

    dlg = PlotArrangeDialog([page_a, page_b], current_page=page_a)
    dlg._board.resize(600, 400)
    height_50 = dlg._board.seat_rect(0).height()
    height_200 = dlg._board.seat_rect(2).height()  # 2 列网格：(1,0)
    assert height_200 > height_50 * 2
    assert "50%/200%" in dlg._caption.text()
    dlg.deleteLater()


def test_show_hidden_toggle_reveals_hidden_cells(pages, app_settings):
    """「显示隐藏区」：默认收起；勾选后隐藏行展示、可作落点。"""
    app_settings.set_arrange_show_hidden(False)
    page_a, page_b = pages
    _set_curves(page_a.plot_widgets[2], ["hidden_sig"])
    page_a.rows_current = 1
    page_a.plot_layout.setRowStretch(1, 0)
    page_a.plot_widgets[2].setVisible(False)
    page_a.plot_widgets[3].setVisible(False)

    dlg = PlotArrangeDialog([page_a, page_b], current_page=page_a)
    assert dlg._show_hidden_box.isChecked() is False
    assert len(dlg._board.icons) == 2          # 默认收起：隐藏行不展示
    assert dlg._board.seat_rect(2).isNull() is True
    assert "隐藏区已收起" in dlg._caption.text()

    dlg._show_hidden_box.setChecked(True)
    assert len(dlg._board.icons) == 3          # 展开后隐藏行与隐藏图都回来
    assert dlg._board.seat_rect(2).isNull() is False
    assert "显示隐藏区" in dlg._caption.text()

    dlg._show_hidden_box.setChecked(False)
    assert len(dlg._board.icons) == 2
    dlg.deleteLater()


def test_show_hidden_default_off_and_persisted(pages, app_settings):
    """无存档默认收起；用户勾选后下一轮打开沿用选择。"""
    from src.core.settings import ConfigKey

    app_settings._settings.remove(ConfigKey.ARRANGE_SHOW_HIDDEN.value)
    app_settings._settings.sync()

    page_a, page_b = pages
    dlg = _open_dialog_for_drag(page_a, page_b)
    assert dlg._show_hidden_box.isChecked() is False
    dlg._show_hidden_box.setChecked(True)       # 用户偏好：展开
    dlg.deleteLater()

    dlg2 = _open_dialog_for_drag(page_a, page_b)
    assert dlg2._show_hidden_box.isChecked() is True
    dlg2.deleteLater()


def test_accept_with_valid_state(dialog):
    dialog._on_accept()
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_viewbox_arrange_signal_reaches_host(plot_factory):
    """右键「整理子图…」→ ViewBox 信号 → 宿主窗口统一入口。"""
    from PySide6.QtWidgets import QWidget

    calls: list = []

    class _RecordingHost(QWidget):
        def open_plot_arrange_dialog(self, pw):
            calls.append(pw)

    host = _RecordingHost()
    pw = plot_factory()
    pw.setParent(host)  # pw.window() 解析到 RecordingHost
    pw.view_box.signals.request_arrange_layout.emit(pw)
    assert calls == [pw]
    host.deleteLater()

# --------------------------------------------------------------------------
# 防抽动 v2 回归（评审定稿：P1 吸附 / P2 同目标跳过 / P3 动画去重+泄漏 / 落位不变量）
# --------------------------------------------------------------------------
def _open_dialog_for_drag(page_a, page_b):
    from tests.fixtures.waits import settle

    dlg = PlotArrangeDialog([page_a, page_b], current_page=page_a)
    dlg.resize(1200, 900)
    dlg.show()
    settle()
    dlg._board.resize(600, 400)
    settle()
    return dlg


def _begin_drag(dlg, token):
    icon = dlg._board.icons[token]
    dlg.begin_drag(
        token, icon.mapToGlobal(icon.rect().center()), icon, from_tray=False
    )
    return icon


def test_gap_sweep_does_not_jitter(pages, monkeypatch):
    """沿格缝横扫：高亮无 None、单调；目标稳定后不再新建动画。"""
    import src.ui.dialogs.plot_arrange_dialog as mod

    created: list[int] = []

    class CountingAnim(mod.QPropertyAnimation):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(1)

    monkeypatch.setattr(mod, "QPropertyAnimation", CountingAnim)

    page_a, page_b = pages
    dlg = _open_dialog_for_drag(page_a, page_b)
    token = "p0:c0"
    _begin_drag(dlg, token)
    board = dlg._board
    r0, r1 = board.seat_rect(0), board.seat_rect(1)
    y = r0.bottom() + 1  # 正好在 10px 缝里

    highlights = []
    for x in range(r0.left() + 2, r1.right() - 2, 2):
        board_and_global = board.mapToGlobal(QPoint(x, y))
        dlg.drag_move(board_and_global)
        highlights.append(board._highlight)

    assert None not in highlights, f"缝中高亮丢失：{highlights}"
    assert highlights == sorted(highlights), f"高亮来回跳：{highlights}"

    # 目标稳定后继续喂同一点：不得再有新建动画
    stable_x = r1.center().x()
    stable_from = len(created)
    for _ in range(30):
        dlg.drag_move(board.mapToGlobal(QPoint(stable_x, y)))
    assert len(created) == stable_from, "同目标仍反复新建动画"
    assert board._highlight == 1

    dlg.end_drag(board.mapToGlobal(QPoint(board.width() - 2, board.height() - 2)))
    dlg.deleteLater()


def test_tray_gap_belongs_to_tray(pages):
    """棋盘与托盘之间的布局间距应命中托盘，而不是落到 None。"""
    page_a, page_b = pages
    dlg = _open_dialog_for_drag(page_a, page_b)
    board, tray = dlg._board, dlg._tray_panel
    tray_top = tray.mapTo(dlg, QPoint(0, 0)).y()
    board_bottom = board.mapTo(dlg, QPoint(0, board.height())).y()
    assert tray_top > board_bottom, "布局前提：托盘应在棋盘下方"
    mid = QPoint(dlg.width() // 2, tray_top - 2)

    _begin_drag(dlg, "p0:c0")
    dlg.drag_move(dlg.mapToGlobal(mid))
    assert dlg._drag["target"] == ("tray", None)
    dlg.end_drag(dlg.mapToGlobal(mid))
    dlg.deleteLater()


def test_visible_only_blank_area_is_dead_zone(pages):
    """只显示可见区时，棋盘内空白区不是落点，预览状态不得被篡改。"""
    from tests.fixtures.waits import settle

    page_a, page_b = pages
    page_a.rows_current = 1
    page_a.plot_layout.setRowStretch(1, 0)
    page_a.plot_widgets[2].setVisible(False)
    page_a.plot_widgets[3].setVisible(False)

    dlg = _open_dialog_for_drag(page_a, page_b)
    dlg._show_hidden_box.setChecked(False)
    settle()
    before = [list(page.cells) for page in dlg._state.pages]
    _begin_drag(dlg, "p0:c0")
    board = dlg._board
    dead = board.mapToGlobal(QPoint(board.width() - 2, board.height() // 2))
    dlg.drag_move(dead)
    assert dlg._drag["target"] is None
    assert [list(page.cells) for page in dlg._state.pages] == before
    dlg.end_drag(dead)
    dlg.deleteLater()


def test_adjust_visible_layout_grows_and_remaps(pages):
    """对话框内扩可见区：矩阵长大、token 按 (row,col) 重映射、托盘/图标不丢。"""
    from PySide6.QtWidgets import QWidget

    from tests.fixtures.waits import settle

    page_a, page_b = pages
    _set_curves(page_a.plot_widgets[3], ["d"])  # 四个格全有图

    class _FakeLayoutManager:
        def open_layout_dialog(self, page=None, parent=None):
            page.apply_visibility(3, 3)  # 2x2 -> 3x3（补建新格）

    host = QWidget()
    host.layout_manager = _FakeLayoutManager()
    dlg = PlotArrangeDialog([page_a, page_b], current_page=page_a, parent=host)
    dlg.resize(1000, 800)
    dlg.show()
    settle()

    state_page = dlg.result_state().pages[0]
    old_positions = {
        token: divmod(seat, state_page.cols)
        for seat, token in enumerate(state_page.cells)
        if token
    }
    assert (state_page.rows, state_page.cols) == (2, 2)

    dlg._adjust_layout_btn.click()

    state_page = dlg.result_state().pages[0]
    assert (state_page.rows, state_page.cols) == (3, 3)
    assert (state_page.vis_rows, state_page.vis_cols) == (3, 3)
    for token, (row, col) in old_positions.items():
        assert state_page.cells[row * 3 + col] == token, "扩列后 token 未按真实行列重映射"
    assert len(dlg._board.icons) == 4
    assert "可见 3×3" in dlg._status.text() or "已建 3×3" in dlg._status.text()
    dlg.deleteLater()
    host.deleteLater()


def test_escape_cancels_active_drag(pages):
    """拖拽中按 Esc：撤销预览、解冻托盘、状态零改动。"""
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent

    page_a, page_b = pages
    dlg = _open_dialog_for_drag(page_a, page_b)
    _begin_drag(dlg, "p0:c0")
    board = dlg._board
    dlg.drag_move(board.mapToGlobal(board.seat_rect(1).center()))
    before = [list(page.cells) for page in dlg._state.pages]

    event = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier
    )
    dlg.keyPressEvent(event)

    assert dlg._drag is None
    assert [list(page.cells) for page in dlg._state.pages] == before
    assert dlg._tray_panel._frozen is False
    assert board._highlight is None
    dlg.deleteLater()


def test_drop_uses_last_preview_target(pages):
    """落位 = 最后展示的预览目标（end_drag 不重算落点）。"""
    page_a, page_b = pages
    board = None

    # 场景 a：最后预览是 seat1，指针随后移出对话框 → 仍落 seat1
    dlg = _open_dialog_for_drag(page_a, page_b)
    token = "p0:c0"
    _begin_drag(dlg, token)
    board = dlg._board
    dlg.drag_move(board.mapToGlobal(board.seat_rect(1).center()))
    assert dlg._drag["target"] == ("seat", 1)
    dlg.end_drag(dlg.mapToGlobal(QPoint(-50, -50)))
    state = dlg.result_state()
    assert state.pages[0].cells[1] == token
    assert state.pages[0].cells[0] not in (None, token)
    dlg.deleteLater()

    # 场景 b：最后预览回退为原状（死区）→ 松手取消
    dlg2 = _open_dialog_for_drag(page_a, page_b)
    _begin_drag(dlg2, token)
    board2 = dlg2._board
    dlg2.drag_move(board2.mapToGlobal(board2.seat_rect(1).center()))
    dead = board2.mapToGlobal(QPoint(board2.width() - 2, board2.height() // 2))
    dlg2.drag_move(dead)
    before = [list(page.cells) for page in dlg2._state.pages]
    dlg2.end_drag(dead)
    assert [list(page.cells) for page in dlg2._state.pages] == before
    dlg2.deleteLater()


def test_no_animation_object_leak_after_drag(pages):
    """一次拖拽后，棋盘不应残留大量 QPropertyAnimation（评审确认的泄漏）。"""
    from PySide6.QtCore import QPropertyAnimation

    page_a, page_b = pages
    dlg = _open_dialog_for_drag(page_a, page_b)
    _begin_drag(dlg, "p0:c0")
    board = dlg._board
    for i in range(60):
        seat = (i // 2) % 3
        point = board.mapToGlobal(QPoint(seat * 60 + 20, (i % 3) * 40 + 20))
        dlg.drag_move(point)
    dlg.end_drag(board.mapToGlobal(QPoint(board.width() - 2, board.height() - 2)))
    anims = [c for c in board.children() if isinstance(c, QPropertyAnimation)]
    assert len(anims) <= len(board.icons) + 2
    dlg.deleteLater()


def test_main_window_entry_guards_and_apply(qapp, monkeypatch):
    """入口：加载锁不弹窗；正常路径构造对话框、应用并播报（含跨页/隐藏计数）。"""
    from PySide6.QtWidgets import QDialog, QWidget

    import src.ui.dialogs.plot_arrange_dialog as dlg_mod
    import src.ui.widgets.plot_arrange_commit as commit_mod
    from src.ui.main_window import MainWindow

    class FakePage:
        def __init__(self):
            self.widget = QWidget()
            self.title = "页1"
            self.rows_current = 2
            self.cols_current = 2
            self.plot_widgets = []
            self.services = None

    class FakeMW(QWidget):
        def __init__(self):
            super().__init__()
            self.messages = []
            self.page = FakePage()
            self.pages = [self.page]
            self._current_page = self.page
            self._is_loading_new_data = False

        def reject_when_loading(self, action):
            return False

        def _broadcast(self, message, level="info"):
            self.messages.append((message, level))

        @property
        def current_page(self):
            return self._current_page

    class FakeDialog:
        constructed = []

        def __init__(self, pages, *, current_page=None, parent=None):
            FakeDialog.constructed.append((pages, current_page))
            self.snapshot = object()

        def exec(self):
            return QDialog.DialogCode.Accepted

        def result_state(self):
            return object()

    monkeypatch.setattr(dlg_mod, "PlotArrangeDialog", FakeDialog)
    monkeypatch.setattr(
        commit_mod,
        "apply_arrangement",
        lambda snapshot, state: {
            "pages": 1, "plots": 3, "cross_page": 1, "hidden": 0,
        },
    )

    mw = FakeMW()
    plot_widget = type("PW", (), {"page": mw.page})()
    MainWindow.open_plot_arrange_dialog(mw, plot_widget)
    assert len(FakeDialog.constructed) == 1
    assert any("已整理子图" in m and "跨页 1 个" in m for m, _ in mw.messages)

    FakeDialog.constructed.clear()
    mw._is_loading_new_data = True
    MainWindow.open_plot_arrange_dialog(mw, plot_widget)
    assert not FakeDialog.constructed
    assert any("正在加载数据" in m for m, _ in mw.messages)
    mw.deleteLater()


def test_show_hidden_applies_per_tab_and_explains_no_hidden_page(
    qapp, app_settings
):
    """开关对所有 tab 一致生效；「已建=可见」的页直接提示无隐藏区。"""
    from tests.fixtures.waits import settle

    app_settings.set_arrange_show_hidden(False)

    page_a = PlotPage(
        parent=None, title="页1", rows_max=4, cols_max=3,
        rows_current=2, cols_current=2,
    )
    page_a.services = _FakeServices()
    page_a.build_matrix(2, 2)
    _set_curves(page_a.plot_widgets[0], ["a"])
    page_a.rows_current = 1  # 第二行隐藏
    page_a.plot_layout.setRowStretch(1, 0)
    page_a.plot_widgets[2].setVisible(False)
    page_a.plot_widgets[3].setVisible(False)

    page_c = PlotPage(
        parent=None, title="页2", rows_max=4, cols_max=3,
        rows_current=1, cols_current=2,
    )
    page_c.services = _FakeServices()
    page_c.build_matrix(1, 2)  # 已建=可见：没有隐藏区
    _set_curves(page_c.plot_widgets[0], ["c"])

    dlg = PlotArrangeDialog([page_a, page_c], current_page=page_a)
    dlg.resize(1000, 800)
    dlg.show()
    settle()

    board = dlg._board
    assert board._display_bounds() == (1, 2)          # 页1 默认收起隐藏行
    assert "隐藏区已收起" in dlg._caption.text()

    dlg._tabs.setCurrentIndex(1)
    settle()
    assert board._display_bounds() == (1, 2)
    assert "本页无隐藏区" in dlg._caption.text()      # 页2 无隐藏区，直说

    dlg._show_hidden_box.setChecked(True)             # 全局开关：对所有页一致
    settle()
    assert board._display_bounds() == (1, 2)          # 页2 没有可展开的隐藏区
    assert "本页无隐藏区" in dlg._caption.text()

    dlg._tabs.setCurrentIndex(0)
    settle()
    assert board._display_bounds() == (2, 2)          # 页1 隐藏行展开
    assert "显示隐藏区" in dlg._caption.text()

    dlg.deleteLater()
    page_a.dispose()
    page_c.dispose()
    flush_deferred_deletes()


# --------------------------------------------------------------------------
# Q1a：松手收拢（从松手点起步、按距离定时、两段式落位）
# --------------------------------------------------------------------------
def _open_dialog_narrow(page_a, page_b, width=320, height=900):
    from tests.fixtures.waits import settle

    dlg = PlotArrangeDialog([page_a, page_b], current_page=page_a)
    dlg.resize(width, height)
    dlg.show()
    settle()
    return dlg


def _settle_expect(dlg, container, token, release_global, seat=None,
                   target=None, size=None):
    """按容器侧最终几何算 (start_tl, target, duration)。"""
    from src.ui.dialogs.plot_arrange_dialog import _settle_ms

    center = dlg._clamp_to(container, release_global)
    start_tl = QPoint(center.x() - size.width() // 2,
                      center.y() - size.height() // 2)
    dist = ((target.x() - start_tl.x()) ** 2
            + (target.y() - start_tl.y()) ** 2) ** 0.5
    return start_tl, target, _settle_ms(dist)


def _settle_expect_icon(dlg, release, seat):
    rect = dlg._board._icon_rect(seat)
    size = rect.size()
    return _settle_expect(dlg, dlg._board, None, release,
                          target=rect.topLeft(), size=size)


def _settle_expect_chip(dlg, release, index):
    from PySide6.QtCore import QSize

    size = QSize(150, 42)  # CHIP_W / CHIP_H
    return _settle_expect(dlg, dlg._tray_panel, None, release,
                          target=dlg._tray_panel._chip_target(index),
                          size=size)


def test_settle_starts_from_release_point_between_cells(pages):
    """单选格→格：松手点两格之间 → anchor 从松手点起飞，终点=最终格。"""
    page_a, page_b = pages
    dlg = _open_dialog_for_drag(page_a, page_b)
    board = dlg._board
    token = "p0:c0"
    _begin_drag(dlg, token)

    r1 = board.seat_rect(1)
    release = board.mapToGlobal(QPoint(r1.left() + 10, r1.top() + 10))
    dlg.drag_move(release)
    assert dlg._drag["target"] == ("seat", 1)
    dlg.end_drag(release)

    start_tl, target, duration = _settle_expect_icon(dlg, release, 1)
    anim = board._anims.get(token)
    assert anim is not None, "松手后同一帧内应已建收拢动画"
    assert anim.startValue() == start_tl, "收拢起点必须是松手点（topLeft 空间）"
    assert anim.endValue() == target
    assert anim.duration() == duration
    assert anim.startValue() != QPoint(0, 0)
    dlg.deleteLater()


def test_settle_zero_distance_lands_directly(pages):
    """目标格心松手：距离≈0 → 不建动画，直接落位。"""
    page_a, page_b = pages
    dlg = _open_dialog_for_drag(page_a, page_b)
    board = dlg._board
    token = "p0:c0"
    _begin_drag(dlg, token)
    release = board.mapToGlobal(board.seat_rect(1).center())
    dlg.drag_move(release)
    dlg.end_drag(release)

    assert token not in board._anims
    assert board.icons[token].pos() == board._icon_target(1)
    dlg.deleteLater()


def test_tray_to_grid_new_icon_settles_from_release(pages):
    """托盘→空格：新 icon 起点=松手点，不是 (0,0)。"""
    from tests.fixtures.waits import settle

    page_a, page_b = pages
    dlg = _open_dialog_for_drag(page_a, page_b)
    board, tray = dlg._board, dlg._tray_panel
    token = "p0:c0"
    dlg._selection = {token}
    dlg._on_kick()
    settle()
    chip = tray.chips[token]
    dlg.begin_drag(
        token, chip.mapToGlobal(chip.rect().center()), chip, from_tray=True
    )
    r3 = board.seat_rect(3)
    release = board.mapToGlobal(QPoint(r3.left() + 12, r3.top() + 12))
    dlg.drag_move(release)
    assert dlg._drag["target"] == ("seat", 3)
    dlg.end_drag(release)

    assert token in board.icons
    anim = board._anims.get(token)
    assert anim is not None
    start_tl, target, _duration = _settle_expect_icon(dlg, release, 3)
    assert anim.startValue() == start_tl
    assert anim.startValue() != QPoint(0, 0)
    assert anim.endValue() == target
    dlg.deleteLater()


def test_grid_to_tray_new_chip_settles_from_release(pages):
    """格→托盘：新 chip 起点=松手点，不是 (0,0)。"""
    page_a, page_b = pages
    dlg = _open_dialog_for_drag(page_a, page_b)
    tray = dlg._tray_panel
    token = "p0:c0"
    _begin_drag(dlg, token)
    release = tray.mapToGlobal(QPoint(tray.width() // 2, tray.height() // 2))
    dlg.drag_move(release)
    assert dlg._drag["target"] == ("tray", None)
    dlg.end_drag(release)

    assert token in tray.chips
    anim = tray._anims.get(token)
    assert anim is not None
    start_tl, target, _duration = _settle_expect_chip(dlg, release, 0)
    assert anim.startValue() == start_tl
    assert anim.startValue() != QPoint(0, 0)
    assert anim.endValue() == target
    dlg.deleteLater()


def _open_wide_tray_dialog(plot_count=6):
    """单页 2×3（6 格）对话框：用于验证托盘长高/减行。"""
    from tests.fixtures.waits import settle

    page = PlotPage(
        parent=None, title="页1", rows_max=4, cols_max=3,
        rows_current=2, cols_current=3,
    )
    page.services = _FakeServices()
    page.build_matrix(2, 3)
    for index in range(plot_count):
        _set_curves(page.plot_widgets[index], [f"v{index}"])
    dlg = PlotArrangeDialog([page], current_page=page)
    dlg.resize(560, 900)   # 托盘每行 3 个 chip，便于验证换行长高/减行
    dlg.show()
    settle()
    return dlg, page


def _dispose_wide(page):
    page.dispose()
    flush_deferred_deletes()


def test_grid_to_tray_growing_tray_settles_to_final_target(qapp):
    """格→托盘且托盘长高：两段式后收拢终点=最终槽位。"""
    from tests.fixtures.waits import settle, wait_until

    dlg, page = _open_wide_tray_dialog()
    board, tray = dlg._board, dlg._tray_panel
    cols = tray._chip_cols()
    assert 2 <= cols <= 4, f"布局前提：托盘每行多列（实际 {cols}）"
    ids = ordered_ids(dlg.result_state().pages[0])
    dlg._selection = set(ids[:cols])
    dlg._on_kick()
    settle()
    assert len(tray.chips) == cols
    min_before = tray.minimumHeight()

    token = ids[cols]
    icon = board.icons[token]
    dlg.begin_drag(
        token, icon.mapToGlobal(icon.rect().center()), icon, from_tray=False
    )
    release = tray.mapToGlobal(QPoint(tray.width() // 2, tray.height() // 2))
    dlg.drag_move(release)
    assert dlg._drag["target"] == ("tray", None)
    dlg.end_drag(release)

    assert len(tray.chips) == cols + 1
    assert tray.minimumHeight() > min_before, "布局前提：多一张 chip 应换到第二行"
    anim = tray._anims.get(token)
    assert anim is not None, "释放后同一帧内收拢动画应已存在"
    settle()
    index = list(tray.chips).index(token)
    assert anim.endValue() == tray._chip_target(index), "终点必须是最终槽位"
    assert wait_until(
        lambda: tray.chips[token].pos() == anim.endValue()
    ), "动画结束后 chip 应停在最终槽位"
    dlg.deleteLater()
    _dispose_wide(page)


def test_tray_to_grid_shrinking_tray_settles_to_final_icon_target(qapp):
    """托盘→空格且托盘减行/棋盘变高：收拢终点=最终 _icon_target。"""
    from tests.fixtures.waits import settle

    dlg, page = _open_wide_tray_dialog()
    board, tray = dlg._board, dlg._tray_panel
    cols = tray._chip_cols()
    ids = ordered_ids(dlg.result_state().pages[0])
    assert len(ids) >= cols + 2
    dlg._selection = set(ids[: cols + 1])
    dlg._on_kick()
    settle()
    assert len(tray.chips) == cols + 1
    min_before = tray.minimumHeight()

    state_page = dlg.result_state().pages[0]
    empty = next(i for i, cell in enumerate(state_page.cells) if cell is None)
    token = list(tray.chips)[0]
    chip = tray.chips[token]
    dlg.begin_drag(
        token, chip.mapToGlobal(chip.rect().center()), chip, from_tray=True
    )
    seat_rect = board.seat_rect(empty)
    release = board.mapToGlobal(QPoint(seat_rect.left() + 12,
                                       seat_rect.top() + 12))
    dlg.drag_move(release)
    assert dlg._drag["target"] == ("seat", empty)
    dlg.end_drag(release)

    assert tray.minimumHeight() < min_before, "布局前提：少一张 chip 应减行"
    settle()
    anim = board._anims.get(token)
    assert anim is not None
    assert anim.endValue() == board._icon_target(empty)
    dlg.deleteLater()
    _dispose_wide(page)


def test_tray_to_occupied_cell_displaced_chip_snaps(pages):
    """托盘→有图：被顶替 chip 走到最终槽位（不从 (0,0) 起飞）。"""
    from tests.fixtures.waits import settle, wait_until

    page_a, page_b = pages
    dlg = _open_dialog_for_drag(page_a, page_b)
    board, tray = dlg._board, dlg._tray_panel
    token = "p0:c0"
    dlg._selection = {token}
    dlg._on_kick()
    settle()
    occupant = dlg.result_state().pages[0].cells[0]
    assert occupant and occupant != token

    chip = tray.chips[token]
    dlg.begin_drag(
        token, chip.mapToGlobal(chip.rect().center()), chip, from_tray=True
    )
    release = board.mapToGlobal(board.seat_rect(0).center())
    dlg.drag_move(release)
    dlg.end_drag(release)

    settle()
    displaced_chip = tray.chips[occupant]
    target = tray._chip_target(list(tray.chips).index(occupant))
    assert target == tray._chip_target(0)
    assert wait_until(lambda: displaced_chip.pos() == target)
    anchor_icon = board.icons[token]
    assert anchor_icon.seat == 0
    anchor_anim = board._anims.get(token)
    assert anchor_anim is None or anchor_anim.endValue() == board._icon_target(0)
    dlg.deleteLater()


def test_multi_select_kick_uses_pressed_token(pages):
    """多选 pressed ≠ ids[0] 踢托盘：无孤儿 icon、字典一致。"""
    page_a, page_b = pages
    dlg = _open_dialog_for_drag(page_a, page_b)
    board, tray = dlg._board, dlg._tray_panel
    ids = ["p0:c0", "p0:c1"]
    dlg._selection = set(ids)
    pressed = ids[1]  # 被按下的不是 ids[0]
    icon = board.icons[pressed]
    dlg.begin_drag(
        pressed, icon.mapToGlobal(icon.rect().center()), icon, from_tray=False
    )
    release = tray.mapToGlobal(QPoint(tray.width() // 2, tray.height() // 2))
    dlg.drag_move(release)
    dlg.end_drag(release)

    assert dlg._structure_ok()
    assert set(tray.chips) == set(ids)
    assert set(board.icons) == {"p0:c2"}
    for token in ids:
        assert token not in board._anims
    dlg.deleteLater()


def test_out_of_bounds_release_after_move_cancels(pages):
    """真实出界：先 drag_move 出界使 target=None，再松手 → 取消、不收拢。"""
    page_a, page_b = pages
    dlg = _open_dialog_for_drag(page_a, page_b)
    before = [list(page.cells) for page in dlg._state.pages]
    _begin_drag(dlg, "p0:c0")
    outside = dlg.mapToGlobal(QPoint(-60, -60))
    dlg.drag_move(outside)
    assert dlg._drag["target"] is None
    dlg.end_drag(outside)

    assert [list(page.cells) for page in dlg._state.pages] == before
    assert not dlg._board._anims
    assert not dlg._tray_panel._anims
    dlg.deleteLater()


def test_resize_during_settle_drops_animation(pages):
    """收拢中窗口缩放：动画移除、icon 直接定位最终格。"""
    page_a, page_b = pages
    dlg = _open_dialog_for_drag(page_a, page_b)
    board = dlg._board
    token = "p0:c0"
    _begin_drag(dlg, token)
    r1 = board.seat_rect(1)
    release = board.mapToGlobal(QPoint(r1.left() + 10, r1.top() + 10))
    dlg.drag_move(release)
    dlg.end_drag(release)
    assert token in board._anims

    board.resize(520, 360)
    assert token not in board._anims
    assert board.icons[token].pos() == board._icon_target(1)
    dlg.deleteLater()


def test_escape_leaves_no_settle_animation(pages):
    """Esc 中途取消：无收拢；预览动画跑完不残留。"""
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent

    from tests.fixtures.waits import wait_until

    page_a, page_b = pages
    dlg = _open_dialog_for_drag(page_a, page_b)
    board = dlg._board
    before = [list(page.cells) for page in dlg._state.pages]
    _begin_drag(dlg, "p0:c0")
    dlg.drag_move(board.mapToGlobal(board.seat_rect(1).center()))
    event = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier
    )
    dlg.keyPressEvent(event)

    assert [list(page.cells) for page in dlg._state.pages] == before
    assert wait_until(lambda: not board._anims and not dlg._tray_panel._anims)
    dlg.deleteLater()


def test_dead_zone_release_does_not_settle(pages):
    """死区松手：状态不变、无收拢。"""
    from tests.fixtures.waits import settle

    page_a, page_b = pages
    page_a.rows_current = 1
    page_a.plot_layout.setRowStretch(1, 0)
    page_a.plot_widgets[2].setVisible(False)
    page_a.plot_widgets[3].setVisible(False)

    dlg = _open_dialog_for_drag(page_a, page_b)
    dlg._show_hidden_box.setChecked(False)
    settle()
    board = dlg._board
    before = [list(page.cells) for page in dlg._state.pages]
    _begin_drag(dlg, "p0:c0")
    dead = board.mapToGlobal(QPoint(board.width() - 2, board.height() // 2))
    dlg.drag_move(dead)
    assert dlg._drag["target"] is None
    dlg.end_drag(dead)

    assert [list(page.cells) for page in dlg._state.pages] == before
    assert not board._anims
    dlg.deleteLater()


def test_begin_drag_during_settle_drops_old_animation(pages):
    """收拢中再拖同一张：先掉旧动画，新收拢从新松手点开始。"""
    page_a, page_b = pages
    dlg = _open_dialog_for_drag(page_a, page_b)
    board = dlg._board
    token = "p0:c0"
    _begin_drag(dlg, token)
    r1 = board.seat_rect(1)
    first_release = board.mapToGlobal(QPoint(r1.left() + 10, r1.top() + 10))
    dlg.drag_move(first_release)
    dlg.end_drag(first_release)
    old_anim = board._anims.get(token)
    assert old_anim is not None

    icon = board.icons[token]
    dlg.begin_drag(token, icon.mapToGlobal(icon.rect().center()), icon,
                   from_tray=False)
    assert board._anims.get(token) is not old_anim
    r3 = board.seat_rect(3)
    second_release = board.mapToGlobal(QPoint(r3.left() + 12, r3.top() + 12))
    dlg.drag_move(second_release)
    dlg.end_drag(second_release)

    start_tl, target, _duration = _settle_expect_icon(dlg, second_release, 3)
    anim = board._anims.get(token)
    assert anim is not None
    assert anim.startValue() == start_tl
    assert anim.endValue() == target
    dlg.deleteLater()


def test_tray_arrange_keeps_same_target_animation(pages):
    """`_TrayPanel._arrange(True)` 不杀同目标 chip 动画（step 9 回退保护）。"""
    from tests.fixtures.waits import settle

    page_a, page_b = pages
    dlg = _open_dialog_for_drag(page_a, page_b)
    tray = dlg._tray_panel
    dlg._selection = {"p0:c0"}
    dlg._on_kick()
    settle()
    chip = tray.chips["p0:c0"]
    chip.move(200, 150)
    tray._arrange(True)
    anim = tray._anims["p0:c0"]
    tray._arrange(True)
    assert tray._anims.get("p0:c0") is anim
    dlg.deleteLater()


def test_no_animation_leak_after_settle(pages):
    """收拢跑完 + DeferredDelete 落地后，动画对象不超过存活 widget 数。"""
    from PySide6.QtCore import QPropertyAnimation

    from tests.fixtures.waits import flush_deferred_deletes, wait_until

    page_a, page_b = pages
    dlg = _open_dialog_for_drag(page_a, page_b)
    board, tray = dlg._board, dlg._tray_panel
    _begin_drag(dlg, "p0:c0")
    r1 = board.seat_rect(1)
    release = board.mapToGlobal(QPoint(r1.left() + 10, r1.top() + 10))
    dlg.drag_move(release)
    dlg.end_drag(release)

    assert wait_until(lambda: not board._anims and not tray._anims)
    flush_deferred_deletes()
    anims = [c for c in board.children() if isinstance(c, QPropertyAnimation)]
    assert len(anims) <= len(board.icons) + 1
    dlg.deleteLater()


def test_keyboard_and_toggle_guards_during_drag(pages, app_settings):
    """拖拽期：默认按钮/复选框/取消按钮都不改状态；复选框回滚且不写 setting。"""
    page_a, page_b = pages
    app_settings.set_arrange_show_hidden(False)
    dlg = _open_dialog_for_drag(page_a, page_b)
    before = [list(page.cells) for page in dlg._state.pages]
    tray_before = list(dlg._state.tray)
    _begin_drag(dlg, "p0:c0")

    dlg._on_accept()
    dlg._on_kick()
    dlg._on_backfill()
    dlg._on_reset()
    assert dlg.result() != QDialog.DialogCode.Accepted
    assert [list(page.cells) for page in dlg._state.pages] == before
    assert dlg._state.tray == tray_before

    dlg._show_hidden_box.setChecked(True)
    assert dlg._show_hidden_box.isChecked() is False, "拖拽期复选框应回滚"
    assert app_settings.get_arrange_show_hidden() is False

    dlg._on_cancel_clicked()
    assert dlg._drag is None
    assert dlg.isVisible(), "取消按钮在拖拽中只取消拖拽，不关窗"
    assert dlg.result() != QDialog.DialogCode.Accepted
    assert [list(page.cells) for page in dlg._state.pages] == before
    dlg.deleteLater()


def test_resize_during_drag_keeps_preview_consistent(pages):
    """拖拽中 resize：非 anchor 按新格位定位，预览不丢；anchor 留在源格。"""
    from tests.fixtures.waits import settle

    page_a, page_b = pages
    dlg = _open_dialog_for_drag(page_a, page_b)
    board = dlg._board
    token = "p0:c0"
    _begin_drag(dlg, token)
    dlg.drag_move(board.mapToGlobal(board.seat_rect(1).center()))
    assert dlg._drag["target"] == ("seat", 1)

    board.resize(520, 360)
    settle()
    assert dlg._drag is not None, "拖拽不得被 resize 打断"
    for tok, icon in board.icons.items():
        target = board._icon_target(icon.seat)
        anim = board._anims.get(tok)
        if tok == token:
            assert icon.pos() == target  # 拖拽源留在源格（不加动画）
        else:
            assert (anim is not None and anim.endValue() == target) or (
                icon.pos() == target
            )
    dlg.deleteLater()
