"""e2e 加载与绘图链路测试：模拟 help.md 的用户旅程。

链路：导入数据文件（按钮 + 文件对话框替身）→ 变量列表刷新 →
拖拽变量到绘图区 → 曲线出现 → Ctrl+R 切换光标。

对应 help.md：
1️⃣ 加载数据（点击「导入数据文件」按钮）
2️⃣ 浏览数据（变量列表出现所有变量）
3️⃣ 绘制图表（从变量列表拖拽变量到绘图区域）
4️⃣ 交互操作（Ctrl+R 光标切换、Ctrl+Y 仅调节 Y 轴）
"""

import pytest

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QDropEvent

from src.ui.drag_drop import build_var_mimedata

# 保活容器：offscreen 下手工构造的 QMimeData 需防 GC（陷阱 #2）
_keep_alive: list = []


@pytest.mark.full_matrix
def test_default_matrix_builds_full_grid(loaded_window):
    """默认布局按 5×3 建满矩阵，并只显示默认可见区（3×1）。

    e2e 其余用例由 conftest 的 `_small_plot_matrix` 替身压到 1×1 提速
    （每例省约 124ms）；本用例以 `full_matrix` marker 跳过替身，把原先
    "54 例顺带覆盖"的建满矩阵行为显式锁定。

    容量（5×3）与默认可见（3×1）分开断言：提容量不得顺带改启动观感。
    """
    from src.core.config import (
        PLOT_COL_CURRENT_DEFAULT,
        PLOT_COL_MAX_DEFAULT,
        PLOT_ROW_CURRENT_DEFAULT,
        PLOT_ROW_MAX_DEFAULT,
    )

    mw = loaded_window
    assert (PLOT_ROW_MAX_DEFAULT, PLOT_COL_MAX_DEFAULT) == (5, 3)
    assert len(mw.plot_widgets) == PLOT_ROW_MAX_DEFAULT * PLOT_COL_MAX_DEFAULT == 15
    visible = [c for c in mw.plot_widgets if c.isVisible()]
    assert len(visible) == PLOT_ROW_CURRENT_DEFAULT * PLOT_COL_CURRENT_DEFAULT == 3


def test_load_populates_variable_list(loaded_window):
    """加载后：变量列表展示全部变量（time/speed/rpm/flag）"""
    mw = loaded_window
    assert mw.loader is not None
    assert mw.loader.datalength == 50

    var_names = set(mw.loader.var_names)
    assert {"time", "speed", "rpm", "flag"} <= var_names
    # 变量列表行数与变量数一致
    assert mw.list_widget.rowCount() == len(mw.loader.var_names)


def test_buttons_enabled_after_load(loaded_window):
    """加载后：分析类按钮全部启用"""
    mw = loaded_window
    for btn in (
        mw.clear_all_plots_btn,
        mw.auto_range_btn,
        mw.auto_y_btn,
        mw.cursor_btn,
        mw.mark_region_btn,
        mw.grid_layout_btn,
    ):
        assert btn.isEnabled(), f"{btn.text()} 应在加载后启用"


def test_drag_variable_to_plot(loaded_window, qapp):
    """拖拽变量到绘图区 = 添加曲线（help.md 3️⃣）"""
    mw = loaded_window
    assert len(mw.plot_widgets) >= 1
    pw = mw.plot_widgets[0].plot_widget

    mime = build_var_mimedata(["speed"])
    _keep_alive.append(mime)
    ev = QDropEvent(
        QPointF(50.0, 50.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    pw.dropEvent(ev)
    qapp.processEvents()

    assert "speed" in pw.curves
    assert len(pw.curves) == 1

    # 再拖一个变量 = 追加（默认添加语义）
    mime2 = build_var_mimedata(["rpm"])
    _keep_alive.append(mime2)
    ev2 = QDropEvent(
        QPointF(50.0, 50.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime2,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    pw.dropEvent(ev2)
    qapp.processEvents()

    assert list(pw.curves) == ["speed", "rpm"]


def test_shift_drag_replaces_curves(loaded_window, qapp):
    """Shift + 拖拽 = 替换（先清空再绘制，help.md 3️⃣）"""
    mw = loaded_window
    pw = mw.plot_widgets[0].plot_widget

    mime = build_var_mimedata(["speed", "rpm"])
    _keep_alive.append(mime)
    ev = QDropEvent(
        QPointF(50.0, 50.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    pw.dropEvent(ev)
    qapp.processEvents()
    assert len(pw.curves) == 2

    mime2 = build_var_mimedata(["flag"])
    _keep_alive.append(mime2)
    ev2 = QDropEvent(
        QPointF(50.0, 50.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime2,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ShiftModifier,
    )
    pw.dropEvent(ev2)
    qapp.processEvents()

    assert list(pw.curves) == ["flag"]


def test_ctrl_r_toggles_cursor(loaded_window, qapp, qtbot):
    """Ctrl+R 快捷键切换游标显示（help.md 5️⃣ / ⌨️）"""
    mw = loaded_window
    pw = mw.plot_widgets[0].plot_widget
    mime = build_var_mimedata(["speed"])
    _keep_alive.append(mime)
    pw.dropEvent(QDropEvent(
        QPointF(50.0, 50.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    ))
    qapp.processEvents()

    mw.activateWindow()
    qapp.processEvents()
    assert not mw.cursor_btn.isChecked()
    assert mw.cursor_btn.text() == "显示游标"

    qtbot.keyClick(mw, Qt.Key_R, Qt.KeyboardModifier.ControlModifier)
    qapp.processEvents()
    assert mw.cursor_btn.isChecked(), "Ctrl+R 应打开游标"
    assert mw.cursor_btn.text() == "隐藏游标", "按钮文案需随状态切换（术语统一为游标）"

    qtbot.keyClick(mw, Qt.Key_R, Qt.KeyboardModifier.ControlModifier)
    qapp.processEvents()
    assert not mw.cursor_btn.isChecked(), "再次 Ctrl+R 应关闭游标"
    assert mw.cursor_btn.text() == "显示游标"


def test_ctrl_y_auto_y_after_plot(loaded_window, qapp, qtbot):
    """Ctrl+Y「仅调节 Y 轴」：绘图后触发不崩溃且启用 Y autoRange"""
    mw = loaded_window
    pw = mw.plot_widgets[0].plot_widget
    mime = build_var_mimedata(["speed"])
    _keep_alive.append(mime)
    pw.dropEvent(QDropEvent(
        QPointF(50.0, 50.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    ))
    qapp.processEvents()

    mw.activateWindow()
    qapp.processEvents()
    qtbot.keyClick(mw, Qt.Key_Y, Qt.KeyboardModifier.ControlModifier)
    qapp.processEvents()

    assert bool(pw.view_box.state["autoRange"][1]) is True


# ---------- R01：按钮与快捷键共用动作闸门 ----------

_SHORTCUT_CASES = (
    pytest.param(
        "cursor", "cursor_btn", "cursor_sync_manager", "toggle_cursor_all",
        Qt.Key_R, Qt.KeyboardModifier.ControlModifier, id="ctrl-r",
    ),
    pytest.param(
        "auto_y", "auto_y_btn", "cursor_sync_manager", "auto_y_in_x_range",
        Qt.Key_Y, Qt.KeyboardModifier.ControlModifier, id="ctrl-y",
    ),
    pytest.param(
        "auto_range", "auto_range_btn", "cursor_sync_manager", "auto_range_all_plots",
        Qt.Key_Y,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        id="ctrl-shift-y",
    ),
    pytest.param(
        "mark_region", "mark_region_btn", "layout_manager", "toggle_mark_region",
        Qt.Key_T, Qt.KeyboardModifier.ControlModifier, id="ctrl-t",
    ),
)


@pytest.fixture()
def mark_stats_singleton_cleanup(qapp):
    from src.ui.mark_stats import MarkStatsWindow

    yield
    window = MarkStatsWindow._instance
    MarkStatsWindow._instance = None
    if window is not None:
        try:
            window.hide()
            window.deleteLater()
        except RuntimeError:
            pass
        qapp.processEvents()


def _plot_speed(mw, qapp):
    pw = mw.plot_widgets[0].plot_widget
    mime = build_var_mimedata(["speed"])
    _keep_alive.append(mime)
    pw.dropEvent(QDropEvent(
        QPointF(50.0, 50.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    ))
    qapp.processEvents()
    return pw


def _press_shortcut(mw, qtbot, qapp, key, modifiers):
    from PySide6.QtTest import QTest

    mw.activateWindow()
    qapp.processEvents()
    QTest.keyClick(mw, key, modifiers)
    qapp.processEvents()


def _spy_action(mw, monkeypatch, manager_name, method_name):
    calls = []
    original = getattr(getattr(mw, manager_name), method_name)

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(getattr(mw, manager_name), method_name, spy)
    return calls


def _plot_action_state(mw):
    pw = mw.plot_widgets[0].plot_widget
    return {
        "cursor_checked": mw.cursor_btn.isChecked(),
        "mark_checked": mw.mark_region_btn.isChecked(),
        "saved_mark_range": mw.saved_mark_range,
        "has_mark_region": pw.mark_region is not None,
        "has_stats_window": getattr(mw, "mark_stats_window", None) is not None,
        "view_range": tuple(tuple(axis) for axis in pw.view_box.viewRange()),
        "auto_range": tuple(pw.view_box.state["autoRange"]),
    }


@pytest.mark.parametrize(
    "action,button_name,manager_name,method_name,key,modifiers",
    _SHORTCUT_CASES,
)
def test_shortcuts_respect_disabled_buttons_without_data(
    loaded_window, qtbot, qapp, monkeypatch, mark_stats_singleton_cleanup,
    action, button_name, manager_name, method_name, key, modifiers,
):
    """无数据但已有子图：按钮与快捷键都不得进入动作，也不得留下假成功。"""
    mw = loaded_window
    _plot_speed(mw, qapp)
    old_loader = mw.loader
    mw.loader = None
    mw.file_loader_manager.set_button_status(False)
    btn = getattr(mw, button_name)
    assert not btn.isEnabled()
    calls = _spy_action(mw, monkeypatch, manager_name, method_name)
    before = _plot_action_state(mw)
    mw._message_label.setText("保持现状")

    try:
        btn.click()  # disabled 按钮的 click() 也不会发 clicked
        _press_shortcut(mw, qtbot, qapp, key, modifiers)
        assert calls == [], f"{action} 绕过了禁用按钮"
        assert _plot_action_state(mw) == before
        assert mw._message_label.text() == "保持现状"
    finally:
        mw.loader = old_loader
        mw.file_loader_manager.set_button_status(True)


@pytest.mark.parametrize(
    "action,button_name,manager_name,method_name,key,modifiers",
    _SHORTCUT_CASES,
)
def test_shortcuts_respect_disabled_buttons_while_loading(
    loaded_window, qtbot, qapp, monkeypatch, mark_stats_singleton_cleanup,
    action, button_name, manager_name, method_name, key, modifiers,
):
    """加载中即使旧 loader/子图仍在，四个动作也必须一致拦截。"""
    mw = loaded_window
    _plot_speed(mw, qapp)
    mw.begin_load_feedback("Demo.csv")
    mw.file_loader_manager.set_button_status(False)
    btn = getattr(mw, button_name)
    assert not btn.isEnabled()
    calls = _spy_action(mw, monkeypatch, manager_name, method_name)
    before = _plot_action_state(mw)
    mw._message_label.setText("保持现状")
    try:
        btn.click()
        _press_shortcut(mw, qtbot, qapp, key, modifiers)
        assert calls == [], f"加载中 {action} 绕过了禁用按钮"
        assert _plot_action_state(mw) == before
        assert mw._message_label.text() == "保持现状"
    finally:
        mw.end_load_feedback()
        mw.file_loader_manager.set_button_status(True)


def test_loading_feedback_immediately_syncs_action_buttons(loaded_window):
    """加载反馈先于 set_button_status(False)：按钮不能短暂留在可用态。"""
    mw = loaded_window
    mw.begin_load_feedback("Demo.csv")
    try:
        for button_name in (
            "cursor_btn", "auto_y_btn", "auto_range_btn", "mark_region_btn",
        ):
            assert not getattr(mw, button_name).isEnabled(), button_name
    finally:
        mw.end_load_feedback()
        mw.file_loader_manager.set_button_status(True)


@pytest.mark.parametrize(
    "action,button_name,manager_name,method_name,key,modifiers",
    _SHORTCUT_CASES,
)
def test_shortcuts_execute_when_data_action_is_available(
    loaded_window, qtbot, qapp, monkeypatch, mark_stats_singleton_cleanup,
    action, button_name, manager_name, method_name, key, modifiers,
):
    """有数据时保留快捷键效率，并确实产生对应业务效果。"""
    mw = loaded_window
    pw = _plot_speed(mw, qapp)
    btn = getattr(mw, button_name)
    assert btn.isEnabled()
    calls = _spy_action(mw, monkeypatch, manager_name, method_name)
    if action == "auto_y":
        pw.view_box.setYRange(100.0, 200.0, padding=0)
    elif action == "auto_range":
        pw.view_box.setXRange(10.0, 40.0, padding=0)
    qapp.processEvents()
    before_x = tuple(pw.view_box.viewRange()[0])

    _press_shortcut(mw, qtbot, qapp, key, modifiers)

    assert len(calls) == 1, f"{action} 有数据时应执行一次"
    if action == "cursor":
        assert mw.cursor_btn.isChecked()
    elif action == "auto_y":
        assert bool(pw.view_box.state["autoRange"][1]) is True
        assert mw._message_label.text() == "已按当前 x 范围调节各子图 Y 轴"
    elif action == "auto_range":
        assert tuple(pw.view_box.viewRange()[0]) != before_x
        assert mw._message_label.text() == "已按全部可见曲线自动缩放 XY 轴"
    else:
        assert mw.mark_region_btn.isChecked()
        assert mw.saved_mark_range is not None
        assert pw.mark_region is not None
        assert getattr(mw, "mark_stats_window", None) is not None


def test_x_lock_blocks_only_x_changing_actions(
    loaded_window, qtbot, qapp, monkeypatch,
):
    """X 锁禁用 XY 自动缩放；仅调 Y 仍可用，且不得改动冻结的 X 范围。"""
    mw = loaded_window
    pw = _plot_speed(mw, qapp)
    pw.view_box.setXRange(10.0, 40.0, padding=0)
    before_x = tuple(pw.view_box.viewRange()[0])
    range_calls = _spy_action(
        mw, monkeypatch, "cursor_sync_manager", "auto_range_all_plots"
    )
    y_calls = _spy_action(
        mw, monkeypatch, "cursor_sync_manager", "auto_y_in_x_range"
    )

    mw.set_x_axis_locked(True)
    assert not mw.auto_range_btn.isEnabled()
    assert mw.auto_y_btn.isEnabled()
    mw.auto_range_btn.click()
    _press_shortcut(
        mw, qtbot, qapp, Qt.Key_Y,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    )
    assert range_calls == []
    assert tuple(pw.view_box.viewRange()[0]) == before_x
    assert mw._message_label.text() == "X 轴已锁定，自动缩放不可用（按 Ctrl+Shift+X 解锁）"

    _press_shortcut(mw, qtbot, qapp, Qt.Key_Y, Qt.KeyboardModifier.ControlModifier)
    assert len(y_calls) == 1
    assert tuple(pw.view_box.viewRange()[0]) == before_x

    mw.set_x_axis_locked(False)
    assert mw.auto_range_btn.isEnabled()
    _press_shortcut(
        mw, qtbot, qapp, Qt.Key_Y,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    )
    assert len(range_calls) == 1
    assert tuple(pw.view_box.viewRange()[0]) != before_x


def test_x_lock_transition_does_not_reenable_actions_without_data(main_window):
    """无数据时解锁只改锁态，不得点亮自动缩放。"""
    mw = main_window
    mw.set_x_axis_locked(True)
    assert not mw.auto_range_btn.isEnabled()
    mw.set_x_axis_locked(False)
    assert not mw.auto_range_btn.isEnabled()


def test_x_lock_survives_loading_button_sync(loaded_window):
    """加载反馈的按钮刷新不得覆盖锁态；数据恢复后 XY 缩放仍禁用。"""
    mw = loaded_window
    mw.set_x_axis_locked(True)
    mw.begin_load_feedback("Demo.csv")
    mw.end_load_feedback()
    mw.file_loader_manager.set_button_status(True)
    assert not mw.auto_range_btn.isEnabled()
    assert mw.auto_y_btn.isEnabled()


def test_zero_row_asset_actions_do_not_enable_analysis_shortcuts(loaded_window):
    """零行数据只放行清图/暂存，不得顺带点亮四个分析动作。"""
    mw = loaded_window
    mw.file_loader_manager.set_button_status(False, asset_actions=True)
    assert mw.clear_all_plots_btn.isEnabled()
    assert mw.curve_shelf_btn.isEnabled()
    for button_name in (
        "cursor_btn", "auto_y_btn", "auto_range_btn", "mark_region_btn",
    ):
        assert not getattr(mw, button_name).isEnabled(), button_name
