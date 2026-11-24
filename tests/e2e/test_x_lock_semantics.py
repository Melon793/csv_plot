"""e2e：X 轴锁定 C2 边界语义（真实 MainWindow）。

- L2 抽屉 chip 唯一手动入口同规格播报（main 收口撤菜单/快捷键）；
- D2：自动缩放按钮置灰 + 快捷键槽守卫 + auto_range_page 入口守卫；
- D3：锁定切页各页保留冻结窗口，saved_x_range 跳过分支同样被消费，
  解锁后跟随恢复；空页无窗口可冻结，激活时照常继承来源页范围；
- D4：reload / 时间修正 / 模板应用 / reset_plot 四入口自动解锁+播报。

交互流（滚轮/拖动/框选）在 test_x_lock_flow.py（C3）。
"""

import pytest

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QDropEvent

from src.ui.drag_drop import build_var_mimedata
from src.ui.main_window import MainWindow
from tests.fixtures.waits import settle

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


def _add_page(mw, qtbot, var_name):
    page = mw.new_page()
    assert page is not None
    qtbot.waitUntil(lambda: len(page.plot_widgets) > 0, timeout=3000)
    assert page.plot_widgets[0].plot_widget.plot_variable(var_name)
    return page


def _wait_activation(mw, qtbot):
    qtbot.waitUntil(lambda: mw._pending_activation is None, timeout=3000)


# ---------- L1 两入口同规格 ----------

def test_drawer_chip_entry_same_spec(loaded_window, qapp):
    """L2 抽屉 chip 是唯一手动入口：锁定/解锁同规格广播+段+按钮联动。"""
    mw = loaded_window
    pw = _plot(mw)
    qapp.processEvents()
    mw._axis_segment.clicked.emit()
    qapp.processEvents()
    drawer = mw._axis_drawer
    assert drawer is not None and drawer.isVisible()
    chip = drawer.lock_chip
    assert chip.text() == "锁定X轴可见范围"

    chip.click()
    assert mw._x_axis_locked is True
    assert pw._x_mouse_locked is True
    assert mw._message_label.text() == MainWindow._x_lock_announce_text(True)
    assert mw._axis_segment.text().endswith("· 已锁定")
    assert not mw.auto_range_btn.isEnabled()
    assert chip.text() == "解锁X轴可见范围"
    assert "自动解锁" in drawer.lock_hint.text()
    assert "应用将解锁" in drawer.apply_btn.toolTip()

    chip.click()
    assert mw._x_axis_locked is False
    assert mw._message_label.text() == MainWindow._x_lock_announce_text(False)
    assert "已锁定" not in mw._axis_segment.text()
    assert mw.auto_range_btn.isEnabled()
    assert chip.text() == "锁定X轴可见范围"


def test_drawer_locked_exit_and_esc_keeps_lock(loaded_window, qapp, qtbot):
    """先锁后开抽屉：chip=解锁出口；Esc 只收抽屉不回滚锁定。"""
    mw = loaded_window
    _plot(mw)
    qapp.processEvents()
    mw.set_x_axis_locked(True)
    mw._axis_segment.clicked.emit()
    qapp.processEvents()
    drawer = mw._axis_drawer
    assert drawer.lock_chip.text() == "解锁X轴可见范围"

    qtbot.keyClick(drawer, Qt.Key.Key_Escape)
    qapp.processEvents()
    assert not drawer.isVisible()
    assert mw._x_axis_locked is True, "Esc 不得回滚立即生效的锁定"

    mw._axis_segment.clicked.emit()
    qapp.processEvents()
    drawer.lock_chip.click()
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
    assert mw._message_label.text() == "X 轴已锁定，自动缩放不可用（在 x 轴抽屉解锁）"

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


def test_d3_locked_empty_new_page_inherits_frozen_window(loaded_window, qtbot, qapp):
    """锁定下新建页没有自己的窗口：激活即继承来源页，首绘不跳全域。

    覆盖两个可观测阶段：激活时不得停在空 ViewBox 的 0~3 占位范围；
    第一条曲线落下时不得被 autoRange 拉到全局数据域。
    """
    mw = loaded_window
    first = mw.current_page
    first_pw = _plot(mw)
    qtbot.waitUntil(lambda: len(first_pw.curves) == 1, timeout=3000)
    settle()  # 首绘排队刷新落地，避免随后重开 autoRange
    first_pw.view_box.enableAutoRange(x=False, y=False)
    assert first.apply_x_range(10, 20)
    qapp.processEvents()

    mw.set_x_axis_locked(True)
    second = mw.new_page()
    assert second is not None
    qtbot.waitUntil(lambda: len(second.plot_widgets) > 0, timeout=3000)
    _wait_activation(mw, qtbot)
    qapp.processEvents()
    second_pw = second.plot_widgets[0].plot_widget

    # 阶段 1：空页继承窗口（而不是 0~3 占位），且 X autoRange 已关
    assert second_pw.view_box.viewRange()[0] == pytest.approx((10, 20), abs=1e-6)
    assert bool(second_pw.view_box.state["autoRange"][0]) is False

    # 阶段 2：首绘不得把 X 拉到全局数据域
    mime = build_var_mimedata(["rpm"])
    _keep_alive.append(mime)
    second_pw.dropEvent(QDropEvent(
        QPointF(10.0, 10.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    ))
    qtbot.waitUntil(lambda: len(second_pw.curves) == 1, timeout=3000)
    qapp.processEvents()
    assert second_pw.view_box.viewRange()[0] == pytest.approx((10, 20), abs=1e-6)
    assert first_pw.view_box.viewRange()[0] == pytest.approx((10, 20), abs=1e-6)
    assert mw._x_axis_locked is True

    # 有内容后转冻结：改来源页窗口再切回，新页不得再跟随
    assert first.apply_x_range(5, 15)
    mw.activate_page(first)
    _wait_activation(mw, qtbot)
    mw.activate_page(second)
    _wait_activation(mw, qtbot)
    qapp.processEvents()
    assert first_pw.view_box.viewRange()[0] == pytest.approx((5, 15), abs=1e-6)
    assert second_pw.view_box.viewRange()[0] == pytest.approx((10, 20), abs=1e-6)


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
    # 解锁在收尾钩子 _post_reload_ui_refresh（50ms 延迟），等它落地
    qtbot.waitUntil(lambda: not mw._x_axis_locked, timeout=3000)
    qapp.processEvents()

    assert mw._x_axis_locked is False
    assert "X 轴可见范围已自动解锁（数据已重载）" in seen
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
    assert "X 轴可见范围已自动解锁（时间基准已改变）" in seen
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
    assert "X 轴可见范围已自动解锁（已应用模板）" in seen
    assert "已锁定" not in mw._axis_segment.text()


def test_d4_clear_plot_unlocks(loaded_window, qapp):
    """D4 真实用户入口（review M1）：清除绘图走 clear_current_plot，
    不经 reset_plot——解锁必须挂在这条链上。"""
    mw = loaded_window
    pw = _plot(mw)
    qapp.processEvents()
    mw.set_x_axis_locked(True)
    seen = _spy_broadcast(mw)

    pw.clear_current_plot()
    qapp.processEvents()

    assert mw._x_axis_locked is False
    # 清除播报会覆盖消息区（同 D4 其它入口），用记录器钉解锁播报
    assert "X 轴可见范围已自动解锁（绘图已清除）" in seen
    # 清空图再清一次：数据域未变，不解锁不播报
    mw.set_x_axis_locked(True)
    mw._message_label.setText("")
    pw.clear_current_plot()
    assert mw._x_axis_locked is True
    assert mw._message_label.text() == ""


def test_d4_remove_last_variable_unlocks(loaded_window, qapp):
    """D4 真实用户入口（review M1）：移除末变量走 _post_remove_refresh 空分支。"""
    mw = loaded_window
    pw = _plot(mw)
    qapp.processEvents()
    mw.set_x_axis_locked(True)

    assert pw.remove_variable_from_plot("speed") is True
    qapp.processEvents()

    assert not pw.curves
    assert mw._x_axis_locked is False
    assert mw._message_label.text() == "X 轴可见范围已自动解锁（末变量已移除）"


def test_d4_reset_plot_unlocks(loaded_window, qapp):
    mw = loaded_window
    pw = _plot(mw)
    qapp.processEvents()
    mw.set_x_axis_locked(True)

    pw.reset_plot(1, 50)
    qapp.processEvents()

    assert mw._x_axis_locked is False
    assert mw._message_label.text() == "X 轴可见范围已自动解锁（绘图已重置）"
    # 未锁定时的 reset 不重复播报
    mw._message_label.setText("")
    pw.reset_plot(1, 50)
    assert mw._message_label.text() == ""


def test_m1_template_failure_keeps_lock(main_window, qapp):
    """review m1：模板不存在/无数据等失败分支不得丢锁。"""
    mw = main_window
    mw.set_x_axis_locked(True)
    mw.apply_template("不存在的模板id")
    assert mw._x_axis_locked is True


def test_m2_same_value_time_correction_keeps_lock(loaded_window, qapp):
    """review m2：抽屉原值点「应用」不构成基准改变，不解锁。
    第二轮 m2：6 位小数频率量化往返（1/3 → 0.333333 Hz → 3.000003）也算同值。"""
    mw = loaded_window
    _plot(mw)
    qapp.processEvents()
    mw.set_x_axis_locked(True)

    assert mw.layout_manager.apply_time_correction(1.0, 0.0) is True
    assert mw._x_axis_locked is True, "同值应用不得解锁"

    assert mw.layout_manager.apply_time_correction(3.0, 0.0) is True
    assert mw._x_axis_locked is False
    mw.set_x_axis_locked(True)
    # factor=3 → 频率框 0.333333 Hz（6 位量化）→ 反推 3.000003：同值往返
    assert mw.layout_manager.apply_time_correction(1 / 0.333333, 0.0) is True
    assert mw._x_axis_locked is True, "量化往返误判基准改变"

    assert mw.layout_manager.apply_time_correction(0.01, 0.0) is True
    assert mw._x_axis_locked is False


def test_m3_sync_parse_failure_keeps_lock(loaded_window, qtbot, qapp, monkeypatch):
    """第二轮 m3：_begin_data_reload 之后的真失败（同步解析返 False）也经
    _end_data_reload 到达收尾钩子——成功标记按版本归属，失败不得解锁。"""
    mw = loaded_window
    _plot(mw)
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    mw.set_x_axis_locked(True)
    seen = _spy_broadcast(mw)

    monkeypatch.setattr(
        mw.file_loader_manager, "_load_sync", lambda *a, **k: False
    )
    mw.file_loader_manager.reload_data()
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    qtbot.wait(150)  # 覆盖收尾延迟链
    qapp.processEvents()

    assert mw._x_axis_locked is True, "解析失败仍丢锁"
    assert "X 轴可见范围已自动解锁（数据已重载）" not in seen
    assert mw.loader is not None, "旧数据应保留"


def test_initial_load_tail_does_not_unlock(loaded_window, qtbot, qapp):
    """初载与 reload 共用 _end_data_reload→_post_reload_ui_refresh 延迟链：
    初载收尾不得解掉用户加载后立刻设的锁（并行偶发失败的根因回归）。"""
    mw = loaded_window
    _plot(mw)
    mw.set_x_axis_locked(True)
    qtbot.wait(150)  # 覆盖初载 50ms+0ms 延迟收尾链
    qapp.processEvents()
    assert mw._x_axis_locked is True, "初载收尾链静默解锁"


def test_m3_reload_failure_keeps_lock(loaded_window, qtbot, qapp, monkeypatch):
    """review m3：加载异常时锁保留（解锁已后移到收尾钩子）。"""
    mw = loaded_window
    _plot(mw)
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    mw.set_x_axis_locked(True)

    def boom(*args, **kwargs):
        raise RuntimeError("合成加载失败")

    monkeypatch.setattr(mw.file_loader_manager, "_load_file", boom)
    mw.file_loader_manager.reload_data()
    qapp.processEvents()
    qtbot.wait(80)

    assert mw._x_axis_locked is True, "reload 失败不得丢锁"