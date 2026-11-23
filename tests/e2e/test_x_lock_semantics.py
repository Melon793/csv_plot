"""e2e：X 轴锁定 C2 边界语义（真实 MainWindow）。

- L1 两入口（右键菜单 action / Ctrl+Shift+X 槽）同规格播报；
- D2：自动缩放按钮置灰 + 快捷键槽守卫 + auto_range_page 入口守卫；
- D3：锁定切页各页保留冻结窗口，saved_x_range 跳过分支同样被消费，
  解锁后跟随恢复；
- D4：reload / 时间修正 / 模板应用 / reset_plot 四入口自动解锁+播报。

交互流（滚轮/拖动/框选）在 test_x_lock_flow.py（C3）。
"""

import pytest

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QDropEvent

from src.ui.drag_drop import build_var_mimedata
from src.ui.main_window import MainWindow

# README 陷阱 #2：QDropEvent 不持有 mimeData 所有权，模块级保活
_keep_alive = []


def _plot(mw, name="speed"):
    pw = mw.plot_widgets[0].plot_widget
    mime = build_var_mimedata([name])
    _keep_alive.append(mime)
    pw.dropEvent(QDropEvent(
        QPointF(10.0, 10.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    ))
    return pw


class _FakeMenuEvent:
    def scenePos(self):
        return QPointF(0.0, 0.0)


def _menu_lock_action(pw):
    menu = pw.view_box.getMenu(_FakeMenuEvent())
    return next(a for a in menu.actions() if a.text() == "锁定X轴")


def _add_page(mw, qtbot, var_name):
    page = mw.new_page()
    assert page is not None
    qtbot.waitUntil(lambda: len(page.plot_widgets) > 0, timeout=3000)
    assert page.plot_widgets[0].plot_widget.plot_variable(var_name)
    return page


def _wait_activation(mw, qtbot):
    qtbot.waitUntil(lambda: mw._pending_activation is None, timeout=3000)


# ---------- L1 两入口同规格 ----------

def test_menu_and_shortcut_entries_same_spec(loaded_window, qapp):
    mw = loaded_window
    pw = _plot(mw)
    qapp.processEvents()

    # 入口 1：快捷键槽
    mw._on_x_lock_shortcut()
    assert mw._x_axis_locked is True
    assert pw._x_mouse_locked is True
    assert mw._message_label.text() == MainWindow._x_lock_announce_text(True)
    assert mw._axis_segment.text().endswith("· 已锁定")
    assert not mw.auto_range_btn.isEnabled()

    # 入口 2：菜单 action（checked 已同步）→ 解锁
    act = _menu_lock_action(pw)
    assert act.isChecked()
    act.trigger()
    assert mw._x_axis_locked is False
    assert pw._x_mouse_locked is False
    assert mw._message_label.text() == MainWindow._x_lock_announce_text(False)
    assert "已锁定" not in mw._axis_segment.text()
    assert mw.auto_range_btn.isEnabled()

    # 菜单再锁 → 快捷键再解：两入口任意组合都走同一广播
    _menu_lock_action(pw).trigger()
    assert mw._x_axis_locked is True
    mw._on_x_lock_shortcut()
    assert mw._x_axis_locked is False


# ---------- D2：自动缩放全入口守卫 ----------

def test_d2_auto_range_guarded_on_all_entries(loaded_window, qapp):
    mw = loaded_window
    pw = _plot(mw)
    qapp.processEvents()
    pw.view_box.setXRange(10, 40, padding=0)
    qapp.processEvents()
    before = list(pw.view_box.viewRange()[0])

    mw.set_x_axis_locked(True)
    assert not mw.auto_range_btn.isEnabled(), "D2：按钮必须置灰"
    assert "锁定" in mw.auto_range_btn.toolTip()

    # 快捷键槽入口
    mw._on_auto_range_shortcut()
    qapp.processEvents()
    assert list(pw.view_box.viewRange()[0]) == before
    assert mw._message_label.text() == "X 轴已锁定，自动缩放不可用（先解锁）"

    # auto_range_page 直调入口（防其它调用旁路）
    mw.cursor_sync_manager.auto_range_page()
    qapp.processEvents()
    assert list(pw.view_box.viewRange()[0]) == before

    # 程序化静默路径不播报
    mw._message_label.setText("")
    mw.cursor_sync_manager.auto_range_page(announce=False)
    assert mw._message_label.text() == ""

    # 解锁后恢复
    mw.set_x_axis_locked(False)
    assert mw.auto_range_btn.isEnabled()
    mw.cursor_sync_manager.auto_range_page()
    qapp.processEvents()
    assert list(pw.view_box.viewRange()[0]) != before, "解锁后自动缩放应恢复"


# ---------- D3：锁定切页不跟随 ----------

def test_d3_locked_page_switch_keeps_frozen_ranges(loaded_window, qtbot, qapp):
    mw = loaded_window
    first = mw.current_page
    assert first.plot_widgets[0].plot_widget.plot_variable("speed")
    second = _add_page(mw, qtbot, "rpm")
    second_pw = second.plot_widgets[0].plot_widget
    first_pw = first.plot_widgets[0].plot_widget

    assert first.apply_x_range(10, 20)
    assert second.apply_x_range(30, 40)
    qapp.processEvents()

    mw.set_x_axis_locked(True)
    # 切回第一页：跟随写范围必须被跳过，第二页保留 (30, 40)
    mw.activate_page(first)
    _wait_activation(mw, qtbot)
    qapp.processEvents()
    assert second_pw.view_box.viewRange()[0] == pytest.approx((30, 40), abs=1e-6)
    # 消费侧是被激活页（first）：跳过分支同样置 None（m1），否则解锁后
    # 同页重激活写回陈旧范围。second.saved_x_range 是 grab 侧的副产物，不在此断言
    assert first.saved_x_range is None, "跳过分支必须消费 saved_x_range（m1）"
    assert first_pw.view_box.viewRange()[0] == pytest.approx((10, 20), abs=1e-6)
    # 广播含非当前页：两页掩码都在
    for pw in (first_pw, second_pw):
        assert [bool(v) for v in pw.view_box.state["mouseEnabled"]] == [False, True]

    # 再切回第二页：仍不跟随，第一页冻结窗口保持
    mw.activate_page(second)
    _wait_activation(mw, qtbot)
    qapp.processEvents()
    assert first_pw.view_box.viewRange()[0] == pytest.approx((10, 20), abs=1e-6)
    assert second_pw.view_box.viewRange()[0] == pytest.approx((30, 40), abs=1e-6)

    # 解锁后跟随恢复：改第一页窗口再切页，第二页应沿用
    mw.set_x_axis_locked(False)
    mw.activate_page(first)
    _wait_activation(mw, qtbot)
    assert first.apply_x_range(5, 15)
    mw.activate_page(second)
    _wait_activation(mw, qtbot)
    qapp.processEvents()
    assert second_pw.view_box.viewRange()[0] == pytest.approx((5, 15), abs=1e-6)


# ---------- D4：四入口自动解锁 ----------
# 播报断言用记录器：动作自身的收尾播报（加载反馈/已应用基准/匹配率）会
# 覆盖消息区，解锁播报是 best-effort transient；持久载体是 L3 段（「· 已锁定」
# 消失）与锁定态本身。记录器钉住「播报确实发出且文案同规格」。


def _spy_broadcast(mw):
    seen = []
    orig = mw._broadcast

    def spy(message, level="info"):
        seen.append(message)
        orig(message, level=level)

    mw._broadcast = spy
    return seen


def test_d4_reload_unlocks_and_announces(loaded_window, qtbot, qapp):
    mw = loaded_window
    _plot(mw)
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    mw.set_x_axis_locked(True)
    seen = _spy_broadcast(mw)
    version_before = mw._data_version

    mw.file_loader_manager.reload_data()
    qtbot.waitUntil(lambda: mw._data_version > version_before, timeout=5000)
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    qapp.processEvents()

    assert mw._x_axis_locked is False
    assert "X 轴已自动解锁（数据已重载）" in seen
    assert "已锁定" not in mw._axis_segment.text()
    pw = mw.plot_widgets[0].plot_widget
    assert pw._x_mouse_locked is False
    assert [bool(v) for v in pw.view_box.state["mouseEnabled"]] == [True, True]


def test_d4_time_correction_unlocks(loaded_window, qapp):
    mw = loaded_window
    _plot(mw)
    qapp.processEvents()
    mw.set_x_axis_locked(True)
    seen = _spy_broadcast(mw)

    assert mw.layout_manager.apply_time_correction(0.01, 0.0) is True
    assert mw._x_axis_locked is False
    assert "X 轴已自动解锁（时间基准已改变）" in seen
    assert "已锁定" not in mw._axis_segment.text()


def test_d4_template_apply_unlocks(loaded_window, qtbot, qapp):
    from uuid import uuid4

    mw = loaded_window
    _plot(mw)
    qapp.processEvents()
    manager = mw.plot_config_manager
    config = manager.export_current_config(mw)
    # 模板存储是会话级隔离目录但跨用例持久：名字必须唯一，防重跑冲突
    template = manager.template_manager.save_template(
        config, f"锁定解锁验证-{uuid4().hex[:8]}"
    )
    qtbot.wait(20)

    mw.set_x_axis_locked(True)
    seen = _spy_broadcast(mw)
    mw.apply_template(template.metadata.id)
    qtbot.waitUntil(
        lambda: bool(mw.plot_widgets[0].plot_widget.curves), timeout=5000
    )
    qapp.processEvents()

    assert mw._x_axis_locked is False
    assert "X 轴已自动解锁（已应用模板）" in seen
    assert "已锁定" not in mw._axis_segment.text()


def test_d4_reset_plot_unlocks(loaded_window, qapp):
    mw = loaded_window
    pw = _plot(mw)
    qapp.processEvents()
    mw.set_x_axis_locked(True)

    pw.reset_plot(1, 50)
    qapp.processEvents()

    assert mw._x_axis_locked is False
    assert mw._message_label.text() == "X 轴已自动解锁（绘图已重置）"
    # 未锁定时的 reset 不重复播报
    mw._message_label.setText("")
    pw.reset_plot(1, 50)
    assert mw._message_label.text() == ""