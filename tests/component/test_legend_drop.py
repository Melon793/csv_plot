"""legend 拖拽 drop 路径 component 测试（offscreen）。

按 main 分支现行语义重构（原 tmp/test_legend_drop_e2e.py 基于旧设计 Alt/Shift
语义且会触发 QMessageBox 阻塞，无法在 offscreen 运行）。

现行语义（src/ui/widgets/plot_widget.py::_handle_legend_drop）：
- legend 来源：无修饰键=移动，Ctrl=复制，Shift 忽略
- 拖回原 plot：静默无操作
- 移动遇目标已含该变量：弹 question 确认窗，Yes=删源、No=两边都留
- 复制遇目标已含该变量：弹 information 提示，不改动

通过构造 QDropEvent + 手工登记拖拽注册表模拟 legend 来源 drop，
QMessageBox 静态方法用 monkeypatch 替身避免模态阻塞。
"""

import pytest

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QDropEvent
from PySide6.QtWidgets import QMessageBox

from src.ui.drag_drop import (
    build_legend_var_mimedata,
    build_var_mimedata,
    clear_active_legend_drag,
    set_active_legend_drag,
)


@pytest.fixture()
def two_plots(plot_factory, qapp):
    src = plot_factory()
    dst = plot_factory()
    return src, dst, qapp


@pytest.fixture()
def silent_dialogs(monkeypatch):
    """替换 QMessageBox 静态方法为记录器，避免 offscreen 模态阻塞"""
    calls = {"question": [], "information": []}
    answer = {"question": QMessageBox.StandardButton.No}

    def fake_question(parent, title, text, *args, **kwargs):
        calls["question"].append((title, text))
        return answer["question"]

    def fake_information(parent, title, text, *args, **kwargs):
        calls["information"].append((title, text))
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(fake_information))
    return calls, answer


def _drop_legend(dst, src, name, *, ctrl=False, shift=False):
    """模拟一次 legend 来源 drop（登记注册表 + 构造 QDropEvent）"""
    mods = Qt.KeyboardModifier.NoModifier
    if ctrl:
        mods |= Qt.KeyboardModifier.ControlModifier
    if shift:
        mods |= Qt.KeyboardModifier.ShiftModifier
    mime = build_legend_var_mimedata([name], src)
    _keep_alive.append(mime)  # 保活：PySide6 下事件不持有 mimeData 所有权
    set_active_legend_drag(src, [name])
    try:
        ev = QDropEvent(
            QPointF(10.0, 10.0),
            Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
            mime,
            Qt.MouseButton.LeftButton,
            mods,
        )
        dst.dropEvent(ev)
    finally:
        clear_active_legend_drag()


# 保活容器：offscreen 下手工构造的 QDropEvent/QMimeData 不受 Qt 事件系统
# 接管，局部变量被 GC 后 dropEvent 内访问 mimeData 会 SIGSEGV
_keep_alive: list = []


def test_default_move_removes_source(two_plots, silent_dialogs):
    """无修饰键=移动：目标新增 + 源移除（singleShot 延迟一拍生效）"""
    src, dst, qapp = two_plots
    assert src.plot_variable("a")
    assert src.plot_variable("b")
    assert dst.plot_variable("c")

    _drop_legend(dst, src, "a")

    # drop 栈返回时目标已添加；源端删除经 singleShot(0) 延迟
    assert "a" in dst.curves
    qapp.processEvents()  # 让 singleShot(0) 落地
    assert "a" not in src.curves
    assert "b" in src.curves  # 其余曲线不受影响


def test_ctrl_copy_keeps_source(two_plots, silent_dialogs):
    """Ctrl=复制：目标新增、源保留"""
    src, dst, _ = two_plots
    assert src.plot_variable("a")
    assert dst.plot_variable("b")

    _drop_legend(dst, src, "a", ctrl=True)

    assert "a" in dst.curves and "b" in dst.curves
    assert "a" in src.curves


def test_drop_on_same_plot_noop(two_plots, silent_dialogs):
    """拖回原 plot：视为取消，无操作（复制/移动统一）"""
    src, _, qapp = two_plots
    assert src.plot_variable("a")
    before = dict(src.curves)

    _drop_legend(src, src, "a")
    _drop_legend(src, src, "a", ctrl=True)
    qapp.processEvents()

    assert list(src.curves) == list(before)


def test_move_to_existing_confirm_yes_removes_source(two_plots, silent_dialogs):
    """移动遇目标已含该变量：确认 Yes → 目标不变、源删除（合并）"""
    calls, answer = silent_dialogs
    answer["question"] = QMessageBox.StandardButton.Yes
    src, dst, qapp = two_plots
    assert src.plot_variable("a")
    assert dst.plot_variable("a")
    dst_before = list(dst.curves)

    _drop_legend(dst, src, "a")
    qapp.processEvents()

    assert len(calls["question"]) == 1  # 确实弹出确认窗
    assert list(dst.curves) == dst_before  # 目标未重复添加
    assert "a" not in src.curves           # 源已删除（合并完成）


def test_move_to_existing_confirm_no_keeps_both(two_plots, silent_dialogs):
    """移动遇目标已含该变量：确认 No → 两边都留"""
    calls, answer = silent_dialogs
    answer["question"] = QMessageBox.StandardButton.No
    src, dst, qapp = two_plots
    assert src.plot_variable("a")
    assert dst.plot_variable("a")

    _drop_legend(dst, src, "a")
    qapp.processEvents()

    assert len(calls["question"]) == 1
    assert "a" in dst.curves
    assert "a" in src.curves  # 取消则源保留


def test_ctrl_copy_to_existing_shows_info(two_plots, silent_dialogs):
    """Ctrl 复制遇目标已含：仅弹 information 提示，曲线无改动"""
    calls, _ = silent_dialogs
    src, dst, _ = two_plots
    assert src.plot_variable("a")
    assert dst.plot_variable("a")

    _drop_legend(dst, src, "a", ctrl=True)

    assert len(calls["information"]) == 1
    assert list(dst.curves) == ["a"]
    assert "a" in src.curves


def test_legend_shift_ignored_treated_as_move(two_plots, silent_dialogs):
    """Shift 在 legend 来源被忽略：仍按移动语义处理"""
    src, dst, qapp = two_plots
    assert src.plot_variable("a")
    assert dst.plot_variable("b")

    _drop_legend(dst, src, "a", shift=True)
    qapp.processEvents()

    assert "a" in dst.curves and "b" in dst.curves
    assert "a" not in src.curves  # 移动语义：源删除


def test_varlist_shift_replace_unchanged(two_plots, silent_dialogs):
    """变量列表来源 + Shift 替换：现状行为不受 legend 分支影响"""
    _, dst, _ = two_plots
    assert dst.plot_variable("a")
    assert dst.plot_variable("b")

    mime = build_var_mimedata(["c"])
    _keep_alive.append(mime)
    ev = QDropEvent(
        QPointF(10.0, 10.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ShiftModifier,
    )
    dst.dropEvent(ev)

    assert list(dst.curves) == ["c"]  # 先清空再添加 = 替换


def test_varlist_plain_add_unchanged(two_plots, silent_dialogs):
    """变量列表来源普通拖入 = 添加（现状路径回归）"""
    _, dst, _ = two_plots
    assert dst.plot_variable("a")

    mime = build_var_mimedata(["b"])
    _keep_alive.append(mime)
    ev = QDropEvent(
        QPointF(10.0, 10.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    dst.dropEvent(ev)

    assert list(dst.curves) == ["a", "b"]


# ---------------------------------------------------------------------------
# 样式继承（方案 A）：drop 时从源 plot 读颜色走 preferred_color 通道。
# 断言一律用 drop 前快照的源色，与色板内容解耦。
# ---------------------------------------------------------------------------


def _pen_color_hex(ci) -> str:
    """取曲线 pen 的 hex 色（PlotDataItem.opts["pen"] 为 QPen）"""
    return QColor(ci.curve.opts["pen"].color()).name()


def test_move_preserves_color_to_empty_target(two_plots, silent_dialogs):
    """移动到空目标：继承源曲线颜色（快照断言，与色板内容解耦）"""
    src, dst, qapp = two_plots
    assert src.plot_variable("a")
    assert src.plot_variable("b")
    want = src.curves["b"].color

    _drop_legend(dst, src, "b")
    qapp.processEvents()

    assert "b" in dst.curves
    assert dst.curves["b"].color == want
    assert _pen_color_hex(dst.curves["b"]) == QColor(want).name()


def test_move_preserves_color_to_nonempty_target(two_plots, silent_dialogs):
    """移动到非空目标：新曲线继承源色，既有曲线颜色不变"""
    src, dst, qapp = two_plots
    assert src.plot_variable("a")
    assert src.plot_variable("b")
    want_b = src.curves["b"].color
    assert dst.plot_variable("c")
    want_c = dst.curves["c"].color

    _drop_legend(dst, src, "b")
    qapp.processEvents()

    assert dst.curves["b"].color == want_b
    assert dst.curves["c"].color == want_c


def test_ctrl_copy_preserves_custom_hex_color(two_plots, silent_dialogs):
    """Ctrl 复制：自定义 hex 色同样继承，且源曲线保留"""
    src, dst, _ = two_plots
    assert src.plot_variable("a")
    assert src._multi_curve_manager.apply_curve_color("a", "#123456")

    _drop_legend(dst, src, "a", ctrl=True)

    assert dst.curves["a"].color == "#123456"
    assert _pen_color_hex(dst.curves["a"]) == "#123456"
    assert "a" in src.curves  # 复制语义：源保留
    assert src.curves["a"].color == "#123456"


def test_ctrl_copy_preserves_color_to_nonempty_target(two_plots, silent_dialogs):
    """Ctrl 复制到非空目标：继承源色，目标既有曲线不受影响"""
    src, dst, _ = two_plots
    assert src.plot_variable("a")
    want_a = src.curves["a"].color
    assert dst.plot_variable("c")
    want_c = dst.curves["c"].color

    _drop_legend(dst, src, "a", ctrl=True)

    assert dst.curves["a"].color == want_a
    assert dst.curves["c"].color == want_c
    assert "a" in src.curves


# ---------------------------------------------------------------------------
# 显隐继承（Step 2）：源隐藏 → 目标保持隐藏（幂等恢复，延迟一拍生效）
# ---------------------------------------------------------------------------


def test_move_preserves_hidden_state(two_plots, silent_dialogs):
    """源隐藏曲线移动后：目标保持隐藏（singleShot 后落地）"""
    src, dst, qapp = two_plots
    assert src.plot_variable("a")
    src._multi_curve_manager.toggle_curve_visibility_by_name("a")
    assert src.curves["a"].visible is False

    _drop_legend(dst, src, "a")
    qapp.processEvents()  # 让显隐恢复 singleShot(0) 落地

    assert dst.curves["a"].visible is False
    assert dst.curves["a"].curve.isVisible() is False
    assert "a" not in src.curves  # 移动语义：源删除


def test_move_to_hidden_placeholder_keeps_hidden(two_plots, silent_dialogs):
    """目标存在同名隐藏占位（visible_preference=False）：复活即隐藏，
    占位色优先于源色（既有复活规则）。
    本用例源曲线可见（不调度显隐恢复），锁定占位复活路径本身；
    幂等早退由 test_move_hidden_to_hidden_placeholder_idempotent 覆盖"""
    src, dst, qapp = two_plots
    assert src.plot_variable("a")
    src._multi_curve_manager.apply_curve_color("a", "#123456")
    assert dst.plot_variable("a")
    dst.demote_to_invalid("a", "green", visible_preference=False)
    assert "a" in dst.invalid_curves

    _drop_legend(dst, src, "a")
    qapp.processEvents()

    # 占位复活色优先：源 #123456 不生效
    assert dst.curves["a"].color == "green"
    # 复活恢复 visible_preference=False；幂等恢复不得翻转
    assert dst.curves["a"].visible is False
    assert dst.curves["a"].curve.isVisible() is False
    # 移动语义：源已删除
    assert "a" not in src.curves


def test_move_hidden_to_hidden_placeholder_idempotent(two_plots, silent_dialogs):
    """源隐藏 + 目标同名隐藏占位：显隐恢复被调度但幂等早退，
    复活曲线保持隐藏（若实现为无条件 toggle，此处会被翻成可见）"""
    src, dst, qapp = two_plots
    assert src.plot_variable("a")
    src._multi_curve_manager.toggle_curve_visibility_by_name("a")
    assert src.curves["a"].visible is False
    assert dst.plot_variable("a")
    dst.demote_to_invalid("a", "green", visible_preference=False)
    assert "a" in dst.invalid_curves

    _drop_legend(dst, src, "a")
    qapp.processEvents()  # 显隐恢复 singleShot(0) 落地

    # 复活恢复 visible_preference=False；幂等早退：ci.visible(False) == want(False)
    assert dst.curves["a"].visible is False
    assert dst.curves["a"].curve.isVisible() is False
