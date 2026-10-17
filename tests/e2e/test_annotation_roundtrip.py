"""标注功能的端到端链路测试（真实 MainWindow，offscreen）。

component 层验的是"图元干了什么"，本文件验的是"用户按下去到结果出现"这条
纵向链路，用的是真实 ``MainWindow`` / ``PlotContext`` / ``LayoutManager``：

- 顶栏按钮与 Ctrl+E / Ctrl+P 三条入口共用同一份状态
- 换布局（含缩小再放大）时按子图索引保留标注
- 模板导出 → 套用之后标注回到原位
- Shift 拖入替换 / 顶栏「清除绘图」会把标注一起清掉并播报

断言保持链路级（"恢复了几条、按钮文案是什么"），细节下沉 component 层。
"""

from __future__ import annotations

import pytest

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QDropEvent

from src.ui.drag_drop import build_var_mimedata

# 保活容器：offscreen 下手工构造的 QMimeData 需防 GC（见 tests/README.md 陷阱）
_keep_alive: list = []

RECT = [(2.0, 2.0), (6.0, 6.0)]
TEXT = [(3.0, 3.0)]


def _pw(mw, index: int = 0):
    return mw.plot_widgets[index].plot_widget


def _add(pw, kind: str, points, **style) -> str:
    """在子图上放一条标注；几何显式给出，不依赖视图范围（离屏几何可能退化）"""
    aid = pw.annotation_manager.create(kind, points, **style)
    assert aid is not None, f"标注未创建: {kind}"
    return aid


def _broadcast_spy(mw) -> list:
    """顶掉真实状态栏播报，只记录 ``(message, level)``"""
    messages: list = []
    mw._broadcast = lambda *args, **kwargs: messages.append(args)
    return messages


def _messages_text(messages: list) -> str:
    return " | ".join(str(m[0]) for m in messages if m)


def _drop(pw, var_names, modifiers=Qt.KeyboardModifier.NoModifier) -> None:
    """模拟把变量拖进绘图区（同 test_load_and_plot.py 的做法）"""
    mime = build_var_mimedata(list(var_names))
    _keep_alive.append(mime)
    pw.dropEvent(
        QDropEvent(
            QPointF(50.0, 50.0),
            Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
            mime,
            Qt.MouseButton.LeftButton,
            modifiers,
        )
    )


def _key(mw, key, qtbot, modifier=Qt.KeyboardModifier.ControlModifier) -> None:
    mw.activateWindow()
    qtbot.keyClick(mw, key, modifier)


# ---------------------------------------------------------------------------
# 编辑模式：三条入口共用一份状态
# ---------------------------------------------------------------------------

class TestEditModeEntryPoints:
    def test_ctrl_e_toggles_every_subplot(self, loaded_window, qapp, qtbot):
        mw = loaded_window
        mw.layout_manager.create_subplots_matrix(2, 2)
        qapp.processEvents()

        assert mw.annotation_btn.isChecked() is False
        assert all(not _pw(mw, i).annotation_manager.edit_mode for i in range(4))

        _key(mw, Qt.Key_E, qtbot)
        qapp.processEvents()
        assert mw.annotation_btn.isChecked() is True
        assert all(_pw(mw, i).annotation_manager.edit_mode for i in range(4))

        _key(mw, Qt.Key_E, qtbot)
        qapp.processEvents()
        assert mw.annotation_btn.isChecked() is False
        assert all(not _pw(mw, i).annotation_manager.edit_mode for i in range(4))

    def test_button_click_agrees_with_shortcut(self, loaded_window, qapp, qtbot):
        """顶栏按钮与快捷键不能各说各话（同一个 checked 状态驱动）"""
        mw = loaded_window
        pw = _pw(mw)

        qtbot.mouseClick(mw.annotation_btn, Qt.MouseButton.LeftButton)
        qapp.processEvents()
        assert pw.annotation_manager.edit_mode is True

        # 快捷键关掉，按钮必须跟着弹起
        _key(mw, Qt.Key_E, qtbot)
        qapp.processEvents()
        assert mw.annotation_btn.isChecked() is False
        assert pw.annotation_manager.edit_mode is False

    def test_button_label_tracks_total_across_subplots(self, loaded_window, qapp, qtbot):
        mw = loaded_window
        mw.layout_manager.create_subplots_matrix(1, 2)
        qapp.processEvents()
        assert mw.annotation_btn.text() == "标注"

        _add(_pw(mw, 0), "rect", RECT)
        qapp.processEvents()
        assert mw.annotation_btn.text() == "标注 1"

        _add(_pw(mw, 1), "rect", RECT)
        qapp.processEvents()
        assert mw.annotation_btn.text() == "标注 2", "按钮文案应汇总全部子图"

        _key(mw, Qt.Key_E, qtbot)
        qapp.processEvents()
        assert mw.annotation_btn.text() == "退出标注 2"

    def test_new_subplot_inherits_active_edit_mode(self, loaded_window, qapp, qtbot):
        """编辑模式是全局状态：新建的子图不能一进去就"拖不动"而按钮显示已开"""
        mw = loaded_window
        _key(mw, Qt.Key_E, qtbot)
        qapp.processEvents()
        assert mw.annotation_btn.isChecked() is True

        mw.layout_manager.create_subplots_matrix(2, 2)
        qapp.processEvents()

        assert all(_pw(mw, i).annotation_manager.edit_mode for i in range(4))
        assert mw.annotation_btn.isChecked() is True


# ---------------------------------------------------------------------------
# 换布局：按子图索引保留
# ---------------------------------------------------------------------------

class TestLayoutSwitch:
    def test_rebuild_keeps_annotations_in_place(self, loaded_window, qapp):
        """换布局后每个子图的标注回到自己那一格（按 row-major 索引）

        只断言标注：曲线**不**参与回放（项目既有语义 —— 曲线可随时再拖一次，
        而标注是用户手工攒的产物，丢了就得重画）。这也是本功能存在的理由。
        """
        mw = loaded_window
        mw.layout_manager.create_subplots_matrix(2, 2)
        qapp.processEvents()

        payloads = {}
        for index in range(4):
            pw = _pw(mw, index)
            _add(pw, "rect", [(float(index), 1.0), (float(index) + 2.0, 3.0)])
            payloads[index] = pw.annotation_manager.dump()

        assert all(payloads.values()), f"前置条件不成立，部分子图没标注: {payloads}"

        mw.layout_manager.create_subplots_matrix(2, 2)
        qapp.processEvents()

        assert len(mw.plot_widgets) == 4
        for index, expected in payloads.items():
            assert _pw(mw, index).annotation_manager.dump() == expected, f"子图 {index}"

    def test_shrink_then_grow_restores_out_of_range_annotations(self, loaded_window, qapp):
        """2×2 缩到 1×1 时第 4 格的标注不能丢 —— 切回来还得在原位"""
        mw = loaded_window
        mw.layout_manager.create_subplots_matrix(2, 2)
        qapp.processEvents()
        last = _add(_pw(mw, 3), "ellipse", [(10.0, 10.0), (14.0, 14.0)])
        expected = _pw(mw, 3).annotation_manager.dump()

        mw.layout_manager.create_subplots_matrix(1, 1)
        qapp.processEvents()
        assert _pw(mw, 0).annotation_manager.count() == 0, "第 1 格本来就没有标注"

        mw.layout_manager.create_subplots_matrix(2, 2)
        qapp.processEvents()
        assert _pw(mw, 3).annotation_manager.dump() == expected
        assert _pw(mw, 3).annotation_manager.get(last) is not None

    def test_capture_writes_registry_before_widgets_die(self, loaded_window, qapp):
        """顺序错了就只能读到空表（widget 先没、标注后读）"""
        mw = loaded_window
        mw.layout_manager.create_subplots_matrix(1, 1)
        qapp.processEvents()
        _add(_pw(mw, 0), "rect", RECT)

        mw.layout_manager._capture_annotations()
        assert mw.annotation_registry.get(0), f"注册表为空: {mw.annotation_registry}"

    def test_double_drop_on_same_subplot_keeps_annotations(self, loaded_window, qapp):
        """追加绘曲线（非替换）不该动标注"""
        mw = loaded_window
        pw = _pw(mw)
        _drop(pw, ["speed"])
        qapp.processEvents()
        _add(pw, "rect", RECT)

        _drop(pw, ["rpm"])
        qapp.processEvents()

        assert list(pw.curves) == ["speed", "rpm"]
        assert pw.annotation_manager.count() == 1


# ---------------------------------------------------------------------------
# 清除入口
# ---------------------------------------------------------------------------

class TestClearEntries:
    def test_shift_drop_replace_clears_annotations(self, loaded_window, qapp):
        mw = loaded_window
        pw = _pw(mw)
        _drop(pw, ["speed", "rpm"])
        qapp.processEvents()
        _add(pw, "rect", RECT)
        _add(pw, "text", TEXT)

        _drop(pw, ["flag"], Qt.KeyboardModifier.ShiftModifier)
        qapp.processEvents()

        assert list(pw.curves) == ["flag"]
        assert pw.annotation_manager.count() == 0, "替换曲线后旧标注仍在"

    def test_middle_double_click_reports_annotation_count(self, loaded_window, qapp):
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QMouseEvent

        mw = loaded_window
        pw = _pw(mw)
        _drop(pw, ["speed"])
        qapp.processEvents()
        _add(pw, "rect", RECT)
        _add(pw, "text", TEXT)

        messages = _broadcast_spy(mw)
        ev = QMouseEvent(
            QEvent.Type.MouseButtonDblClick,
            QPointF(10.0, 10.0),
            QPointF(10.0, 10.0),
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.MiddleButton,
            Qt.KeyboardModifier.NoModifier,
        )
        pw.mouseDoubleClickEvent(ev)
        qapp.processEvents()

        assert pw.annotation_manager.count() == 0
        assert "2 条标注" in _messages_text(messages), _messages_text(messages)

    def test_top_bar_clear_clears_and_reports_annotations(self, loaded_window, qapp, qtbot):
        mw = loaded_window
        pw = _pw(mw)
        _drop(pw, ["speed"])
        qapp.processEvents()
        _add(pw, "rect", RECT)

        messages = _broadcast_spy(mw)
        qtbot.mouseClick(mw.clear_all_plots_btn, Qt.MouseButton.LeftButton)
        qapp.processEvents()

        assert list(pw.curves) == []
        assert pw.annotation_manager.count() == 0
        text = _messages_text(messages)
        assert "已清除全部绘图" in text and "1 条标注" in text, text


# ---------------------------------------------------------------------------
# 演示视图
# ---------------------------------------------------------------------------

class TestPresentationView:
    def test_ctrl_p_hides_handles_then_restores_state(self, loaded_window, qapp, qtbot):
        mw = loaded_window
        pw = _pw(mw)
        aid = _add(pw, "rect", RECT)
        _key(mw, Qt.Key_E, qtbot)
        qapp.processEvents()

        manager = pw.annotation_manager
        handle = manager._graphics[aid].item.handles[0]["item"]
        assert handle.isVisible() is True, "编辑态手柄应可见，否则本用例无内容"

        _key(mw, Qt.Key_P, qtbot)
        qapp.processEvents()

        assert mw._presentation_state is not None
        assert handle.isVisible() is False, "截图里不该出现拖拽手柄"
        assert manager.edit_mode is False
        assert manager._graphics[aid].item.isVisible() is True, "标注本体必须照常显示"

        _key(mw, Qt.Key_P, qtbot)
        qapp.processEvents()

        assert mw._presentation_state is None
        assert manager.edit_mode is True, "退出演示视图要恢复进入前的编辑模式"
        assert handle.isVisible() is True

    def test_ctrl_p_does_not_disturb_cursor_that_was_off(self, loaded_window, qapp, qtbot):
        mw = loaded_window
        assert mw.cursor_btn.isChecked() is False

        _key(mw, Qt.Key_P, qtbot)
        qapp.processEvents()
        _key(mw, Qt.Key_P, qtbot)
        qapp.processEvents()

        assert mw.cursor_btn.isChecked() is False, "本来没开的游标不该被演示视图打开"

    def test_presentation_view_is_idempotent_on_enter(self, loaded_window, qapp, qtbot):
        """连按两次进入不该把状态覆盖成"看起来是开着"的错误快照"""
        mw = loaded_window
        pw = _pw(mw)
        aid = _add(pw, "rect", RECT)
        _key(mw, Qt.Key_E, qtbot)
        qapp.processEvents()

        _key(mw, Qt.Key_P, qtbot)
        qapp.processEvents()
        snapshot = dict(mw._presentation_state)

        # 第二次 P 是"退出"，直接断言退出后状态正确
        _key(mw, Qt.Key_P, qtbot)
        qapp.processEvents()
        assert mw._presentation_state is None
        assert snapshot["annotation"] is True
        assert pw.annotation_manager.edit_mode is True
        assert pw.annotation_manager._graphics[aid].item.handles[0]["item"].isVisible() is True


# ---------------------------------------------------------------------------
# 模板往返
# ---------------------------------------------------------------------------

class TestTemplateRoundTrip:
    def test_export_then_apply_restores_annotations(self, loaded_window, qapp):
        mw = loaded_window
        pw = _pw(mw)
        assert pw.plot_variable("speed")
        _add(pw, "rect", RECT)
        _add(pw, "arrow", [(1.0, 1.0), (8.0, 8.0)])
        exported = pw.annotation_manager.dump()

        config = mw.plot_config_manager.export_current_config(mw)
        assert config.plots[0].annotations == exported, "导出没带上标注"
        assert config.plots[0].curves == ["speed"]

        # 清空后套用同一份配置
        pw.clear_plot_item(clear_annotations=True)
        qapp.processEvents()
        assert pw.annotation_manager.count() == 0

        assert mw.plot_config_manager.apply_config(mw, config) is True
        qapp.processEvents()

        assert pw.annotation_manager.dump() == exported
        assert "speed" in pw.curves, "曲线也应随模板恢复"

    def test_annotations_survive_even_if_curves_cannot_be_restored(self, loaded_window, qapp):
        """变量不在当前数据里时曲线加载失败，但标注必须恢复

        否则用户会以为模板把标注也弄丢了。
        """
        mw = loaded_window
        pw = _pw(mw)
        assert pw.plot_variable("speed")
        _add(pw, "rect", RECT)
        exported = pw.annotation_manager.dump()

        config = mw.plot_config_manager.export_current_config(mw)
        # 把曲线改成当前数据里没有的变量，让曲线加载必然失败
        config.plots[0].curves = ["__no_such_variable__"]

        pw.clear_plot_item(clear_annotations=True)
        qapp.processEvents()

        mw.plot_config_manager.apply_config(mw, config)
        qapp.processEvents()

        assert pw.annotation_manager.dump() == exported

    def test_annotation_only_subplot_is_exported(self, loaded_window, qapp):
        """只有标注、没有曲线的子图也要进模板，否则辛苦标的图一存就丢"""
        mw = loaded_window
        mw.layout_manager.create_subplots_matrix(1, 2)
        qapp.processEvents()

        _add(_pw(mw, 0), "rect", RECT)
        assert _add(_pw(mw, 1), "text", TEXT) is not None

        config = mw.plot_config_manager.export_current_config(mw)
        assert len(config.plots) == 2
        assert config.plots[0].curves == [] and config.plots[0].annotations
        assert config.plots[1].curves == [] and config.plots[1].annotations


@pytest.mark.parametrize(
    "kind", ["text", "rect", "ellipse", "line", "arrow", "polyline", "target"]
)
def test_every_supported_kind_survives_layout_switch(loaded_window, qapp, kind):
    """七类图元逐个走一遍换布局：漏掉任何一类都会在这里露出来

    折线尤其要过这一关 —— 它的几何是变长点列，回放时若按"两点"处理会静默丢折点。
    标靶同理，而且它还带一个**附属标签子图元**：只把 points 放回去、文案丢了，
    画面上就少一个标签，而只比几何的断言照样全绿。
    """
    mw = loaded_window
    pw = _pw(mw)
    points = {
        "text": TEXT,
        "target": TEXT,
        "polyline": [(1.0, 1.0), (3.0, 5.0), (6.0, 2.0), (9.0, 7.0)],
    }.get(kind) or [(1.0, 1.0), (9.0, 9.0)]
    # 花括号文案是刻意选的：pyqtgraph 会把非 callable 的含 {} 文案判成 format 模板
    style = {"text": "目标 {24℃}"} if kind == "target" else {}
    _add(pw, kind, points, **style)
    expected = pw.annotation_manager.dump()

    mw.layout_manager.create_subplots_matrix(1, 1)
    qapp.processEvents()

    assert _pw(mw, 0).annotation_manager.dump() == expected
    if kind == "polyline":
        assert len(expected[0]["points"]) == 4, "前置条件不成立：折点没存进去"
    if kind == "target":
        assert expected[0]["text"] == "目标 {24℃}", "前置条件不成立：标靶文案没进模板"
        restored = _pw(mw, 0).annotation_manager._graphics[expected[0]["id"]].item
        label = restored.label()
        assert label is not None, "回放丢了标靶标签（文案只活在模型里 = 静默丢失）"
        assert label.toPlainText() == "目标 {24℃}"
