"""曲线暂存区端到端闭环（真实 MainWindow，offscreen）。

用户场景验收锚点：
开窗 → tab1 图里的变量入区（剪切）→ 切到 tab2 → 从暂存区拖出到目标图
→ 样式保真落地 + 状态栏播报。全程无模态。

关键隔离点由 e2e conftest 提供（sys.argv 重置 / app_settings 每例归零 /
QMessageBox 替身），本文件只补真实链路。
"""

import pytest

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QApplication

from src.ui.drag_drop import (
    build_legend_var_mimedata,
    build_shelf_mimedata,
    build_var_mimedata,
    clear_active_legend_drag,
    clear_active_shelf_drag,
    set_active_legend_drag,
    set_active_shelf_drag,
)
from tests.fixtures.data_factory import write_csv

# offscreen 手工 QDropEvent 的 mime 保活（陷阱 #2）
_keep_alive: list = []


@pytest.fixture(autouse=True)
def _guard_app_exit(monkeypatch):
    """防毒化：任何路径若调用 QApplication.exit，同进程后续用例会被 quitNow 拖死。"""
    monkeypatch.setattr(QApplication, "exit", staticmethod(lambda *a, **k: None))


def _drop(widget, mime, *, ctrl=False):
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
    widget.dropEvent(event)


def test_shelf_button_gate_before_load(main_window):
    """未加载数据：暂存区按钮与胶囊同步禁用（D-A7 同一闸门）。"""
    assert main_window.curve_shelf_btn.isEnabled() is False
    capsule = main_window.plot_tabs.shelf_capsule
    assert capsule is not None
    assert capsule.isEnabled() is False


def test_open_shelf_button_shows_tool_window(loaded_window, qapp, qtbot):
    mw = loaded_window
    assert mw.curve_shelf_btn.isEnabled() is True

    capsule = mw.plot_tabs.shelf_capsule
    assert capsule.isVisible() is True
    qtbot.mouseClick(capsule, Qt.MouseButton.LeftButton)
    qapp.processEvents()

    dialog = mw.curve_shelf_if_open()
    assert dialog is not None and dialog.isVisible()
    assert mw.curve_shelf_btn.isChecked() is True
    assert capsule.isChecked() is True
    # D11：首次开窗给一次状态栏提示
    assert "暂存区已打开" in mw._message_text
    assert mw._curve_shelf_hint_shown is True
    # 非模态 + Qt.Tool：不阻塞主窗口操作
    assert dialog.isModal() is False
    assert dialog.windowFlags() & Qt.WindowType.Tool

    # 收起按钮：隐藏浮窗 + 按钮勾选态同步
    dialog.close_btn.click()
    qapp.processEvents()
    assert dialog.isVisible() is False
    assert mw.curve_shelf_btn.isChecked() is False

    # 标题栏关闭 = 隐藏单例（条目保留），不得销毁
    from src.ui.widgets.curve_shelf import ShelfEntry

    dialog.shelf.add_entry(ShelfEntry.from_name("kept"))
    mw.curve_shelf_btn.setChecked(True)
    qapp.processEvents()
    assert dialog.isVisible() is True
    dialog.close()
    qapp.processEvents()
    assert dialog.isVisible() is False
    assert mw.curve_shelf_if_open() is dialog
    assert dialog.shelf.entry_for_name("kept") is not None


def test_cross_page_transfer_end_to_end(loaded_window, qapp):
    mw = loaded_window
    page1 = mw.pages[0]
    pw1 = page1.plot_widgets[0].plot_widget
    assert pw1.plot_variable("speed")
    assert pw1.plot_variable("rpm")
    snapshot_color = pw1.curves["speed"].color
    assert pw1._multi_curve_manager.apply_curve_draw_mode("speed", "step-pre")
    # 模拟枚举通道自动判定的阶梯：来源应为 'auto'（经编辑器/菜单改则钉 'user'），
    # 用于验证搬运后不会被 apply 入口误钉
    pw1.curves["speed"].draw_mode_source = "auto"
    assert pw1._multi_curve_manager.apply_curve_symbol_policy("speed", "on")

    mw.curve_shelf_btn.setChecked(True)
    qapp.processEvents()
    shelf_dialog = mw.curve_shelf_if_open()
    assert shelf_dialog is not None and shelf_dialog.isVisible()

    # 1) tab1 图内：legend → 暂存区（无修饰=剪切）
    mime = build_legend_var_mimedata(["speed"], pw1)
    set_active_legend_drag(pw1, ["speed"])
    try:
        _drop(shelf_dialog.shelf, mime)
    finally:
        clear_active_legend_drag()
    qapp.processEvents()

    assert "speed" not in pw1.curves  # 剪切：源端延迟删除已落地
    assert "rpm" in pw1.curves
    entry = shelf_dialog.shelf.entry_for_name("speed")
    assert entry is not None
    assert "已剪切 speed" in mw._message_text

    # 2) 正常点击 tab2（浮窗非模态，不阻切页）
    page2 = mw.new_page()
    assert page2 is not None
    qapp.processEvents()
    assert mw.current_page is page2
    pw2 = page2.plot_widgets[0].plot_widget

    # 3) 暂存区 → tab2 目标图（无修饰=移出）
    shelf = shelf_dialog.shelf
    keys = [entry.key]
    shelf_mime = build_shelf_mimedata(keys, ["speed"], shelf)
    set_active_shelf_drag(shelf, keys)
    try:
        _drop(pw2, shelf_mime)
    finally:
        clear_active_shelf_drag()
    qapp.processEvents()

    assert "speed" in pw2.curves
    ci = pw2.curves["speed"]
    assert ci.color == snapshot_color
    assert ci.draw_mode == "step-pre"
    assert ci.symbol_policy == "on"
    # apply 入口不会把快照来源误钉成 'user'
    assert ci.draw_mode_source == "auto"
    assert shelf.count() == 0  # 移出：条目消失
    assert "已移动 speed：暂存区 → 页2·图1" in mw._message_text


def test_blf_abort_restore_skips_shelf_revalidate(loaded_window, qapp):
    """BLF 取消恢复路径（_restore_curves_after_abort）数据未变，不得触发置灰。"""
    mw = loaded_window
    mw.curve_shelf_btn.setChecked(True)
    qapp.processEvents()
    dialog = mw.curve_shelf_if_open()
    assert dialog is not None
    called = []
    dialog.revalidate_against_loader = lambda loader: called.append(1) or []

    mw.file_loader_manager._restore_curves_after_abort()
    qapp.processEvents()

    assert called == []


def _two_col_rows(n: int = 50) -> list[list]:
    return [[f"{i * 0.1:.1f}", f"{10.0 + i * 0.5:.2f}"] for i in range(n)]


def test_reload_marks_missing_shelf_entry_stale(
    loaded_window, qapp, qtbot, tmp_path, monkeypatch
):
    """reload 后：变量不在新数据 → 暂存条目置灰并播报。"""
    mw = loaded_window
    mw.curve_shelf_btn.setChecked(True)
    qapp.processEvents()
    dialog = mw.curve_shelf_if_open()
    assert dialog is not None

    _drop(dialog.shelf, build_var_mimedata(["rpm", "speed"]))
    qapp.processEvents()
    assert dialog.shelf.entry_for_name("rpm").stale is False

    broadcasts: list[str] = []
    original_show = mw._show_status_message
    monkeypatch.setattr(
        mw,
        "_show_status_message",
        lambda message: (broadcasts.append(message), original_show(message))[1],
    )

    # 新数据只保留 time/speed：rpm 应置灰
    csv_path = tmp_path / "e2e_demo.csv"
    write_csv(
        csv_path,
        header=["time", "speed"],
        units=["s", "km/h"],
        rows=_two_col_rows(),
    )
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    version_before = mw._data_version
    mw.file_loader_manager.reload_data()
    qtbot.waitUntil(lambda: mw._data_version > version_before, timeout=5000)
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)

    assert dialog.shelf.entry_for_name("rpm").stale is True
    assert dialog.shelf.entry_for_name("speed").stale is False
    assert any("rpm 在当前数据中不存在，已置灰" in m for m in broadcasts)


def test_reload_empty_data_marks_all_stale(
    loaded_window, qapp, qtbot, tmp_path, monkeypatch
):
    """空数据 reload：replots_after_loading 早退，钩子仍须把全部条目置灰。

    这条用例锁定 reload 钩子必须挂在调用方（file_loader_manager），
    不能挂进 replots_after_loading 方法内部。
    """
    mw = loaded_window
    mw.curve_shelf_btn.setChecked(True)
    qapp.processEvents()
    dialog = mw.curve_shelf_if_open()
    _drop(dialog.shelf, build_var_mimedata(["speed", "rpm"]))
    qapp.processEvents()

    broadcasts: list[str] = []
    original_show = mw._show_status_message
    monkeypatch.setattr(
        mw,
        "_show_status_message",
        lambda message: (broadcasts.append(message), original_show(message))[1],
    )

    # 仅表头 + 单位行：0 数据行（datalength == 0），且变量集不含暂存条目
    csv_path = tmp_path / "e2e_demo.csv"
    write_csv(
        csv_path,
        header=["time", "other"],
        units=["s", "-"],
        rows=[],
    )
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    version_before = mw._data_version
    mw.file_loader_manager.reload_data()
    qtbot.waitUntil(lambda: mw._data_version > version_before, timeout=5000)
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)

    assert dialog.shelf.entry_for_name("speed").stale is True
    assert dialog.shelf.entry_for_name("rpm").stale is True


# ---------------------------------------------------------------------------
# 入口迁移：页签栏右端胶囊（方案 A / T2）
# ---------------------------------------------------------------------------


def test_top_bar_shelf_button_is_replaced_by_capsule(loaded_window):
    mw = loaded_window
    assert mw.curve_shelf_btn is not None  # 撤入口不撤实现（回滚/兼容）
    assert mw.curve_shelf_btn.isVisible() is False
    capsule = mw.plot_tabs.shelf_capsule
    assert capsule is not None and capsule.isVisible() is True
    assert capsule.isEnabled() is True


def test_capsule_click_toggles_window_and_state(loaded_window, qapp, qtbot):
    mw = loaded_window
    capsule = mw.plot_tabs.shelf_capsule

    qtbot.mouseClick(capsule, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    dialog = mw.curve_shelf_if_open()
    assert dialog is not None and dialog.isVisible()
    assert capsule.isChecked() is True

    qtbot.mouseClick(capsule, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert dialog.isVisible() is False
    assert capsule.isChecked() is False


def test_capsule_badge_keeps_count_when_window_closed(loaded_window, qapp):
    mw = loaded_window
    capsule = mw.plot_tabs.shelf_capsule
    mw._show_curve_shelf()
    qapp.processEvents()
    shelf = mw.curve_shelf_if_open().shelf
    _drop(shelf, build_var_mimedata(["speed", "rpm"]))
    qapp.processEvents()
    assert capsule.count() == 2

    mw.hide_curve_shelf()
    qapp.processEvents()
    # 关窗不丢计数：正是本次改造要解决的「关窗即丢」
    assert capsule.count() == 2
    assert capsule.isChecked() is False


def test_capsule_drop_forwards_without_opening(loaded_window, qapp):
    mw = loaded_window
    capsule = mw.plot_tabs.shelf_capsule
    assert mw.curve_shelf_if_open() is None

    _drop(capsule, build_var_mimedata(["speed"]))
    qapp.processEvents()

    dialog = mw.curve_shelf_if_open()
    assert dialog is not None
    assert dialog.isVisible() is False  # D-A3：落点不自动弹窗
    assert dialog.shelf.count() == 1
    assert capsule.count() == 1


def test_rebuilt_shelf_resets_capsule_count(loaded_window, qapp):
    from tests.fixtures.waits import force_delete

    mw = loaded_window
    capsule = mw.plot_tabs.shelf_capsule
    mw._show_curve_shelf()
    qapp.processEvents()
    dialog = mw.curve_shelf_if_open()
    _drop(dialog.shelf, build_var_mimedata(["speed"]))
    qapp.processEvents()
    assert capsule.count() == 1

    mw.hide_curve_shelf()
    force_delete(dialog)  # C++ 已析构，Python 包装器尚存：走 RuntimeError 重建分支

    rebuilt = mw.get_curve_shelf()
    assert rebuilt is not dialog
    assert capsule.count() == 0
    assert capsule.isChecked() is False
    # 重建后 entries_changed 必须重接，否则徽章永远停在 0
    _drop(rebuilt.shelf, build_var_mimedata(["rpm"]))
    qapp.processEvents()
    assert capsule.count() == 1


def test_drag_leave_reaches_capsule_only(loaded_window, qapp):
    """D-A9：应用级过滤器只对胶囊放行 DragLeave，其余控件维持吞掉。"""
    from PySide6.QtGui import QDragLeaveEvent

    mw = loaded_window
    capsule = mw.plot_tabs.shelf_capsule
    capsule._set_drag_hot(True)
    assert capsule._drag_hot is True

    qapp.sendEvent(capsule, QDragLeaveEvent())
    qapp.processEvents()
    assert capsule._drag_hot is False

    # 非胶囊控件仍返回 True（既有行为不变，避免顺带改动其它拖拽路径）
    other = mw.plot_widgets[0]
    assert mw.eventFilter(other, QDragLeaveEvent()) is True


def test_shelf_open_is_rejected_while_loading(loaded_window, qapp, monkeypatch):
    mw = loaded_window
    monkeypatch.setattr(mw, "is_data_loading", lambda: True)
    capsule = mw.plot_tabs.shelf_capsule
    mw.plot_tabs.set_shelf_active(True)  # 模拟点按已把勾选态翻成 True

    mw.toggle_curve_shelf(True)
    qapp.processEvents()

    dialog = mw.curve_shelf_if_open()
    assert dialog is None or dialog.isVisible() is False
    assert capsule.isChecked() is False  # 拒绝后勾选态回滚
