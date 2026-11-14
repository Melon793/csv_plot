"""plot 侧暂存区 drop 分支 + hover 文案（offscreen）。

覆盖施工计划 §9.1 出区用例：
- 移出（条目删）/ 复制出（条目留）；
- 拖回来源图=合法撤销（不得早退）；
- 批量判重跳过 + 汇总播报；
- 全量样式保真（draw_mode_source 不误钉 'user'）；
- emit 合并（批量收尾 ≤1 次）；
- quiet 批量失败无模态 + 不删源/条目；
- loading 闸门；hover 文案 shelf 分支（Blocking 4）。
"""

import pytest

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QMessageBox

from src.core.data_types import (
    DRAW_MODE_SOURCE_AUTO,
    DRAW_MODE_STEP_PRE,
    SYMBOL_POLICY_ON,
)
from src.ui.drag_drop import (
    build_shelf_mimedata,
    clear_active_shelf_drag,
    set_active_shelf_drag,
)
from src.ui.widgets.curve_shelf import CurveShelfWidget, ShelfEntry

_keep_alive: list = []


class FakeContainer:
    def __init__(self, plot_widget):
        self.plot_widget = plot_widget


class FakePage:
    def __init__(self, index: int, containers):
        self.index = index
        self.plot_widgets = list(containers)


@pytest.fixture()
def dst(plot_factory):
    """目标 plot：预置 c，非空目标（避免首绘 emit 干扰计数）。"""
    widget = plot_factory()
    assert widget.plot_variable("c")
    host = widget.window()
    host.messages = []
    host._show_status_message = host.messages.append
    host.page_index = lambda page: getattr(page, "index", -1)
    host.reject_when_loading = lambda action: False
    widget.page = FakePage(1, [FakeContainer(widget)])  # 页2·图1
    return widget


@pytest.fixture()
def shelf(qapp):
    widget = CurveShelfWidget()
    yield widget
    widget.deleteLater()


def _style_entry(name: str, key: str = "k1") -> ShelfEntry:
    return ShelfEntry(
        key=key,
        var_name=name,
        color="#123456",
        visible=False,
        draw_mode=DRAW_MODE_STEP_PRE,
        symbol_policy=SYMBOL_POLICY_ON,
        draw_mode_source=DRAW_MODE_SOURCE_AUTO,
        origin_label="页1·图2",
    )


def _drop(dst, shelf, entries, *, copy=False):
    keys = [entry.key for entry in entries]
    mime = build_shelf_mimedata(keys, [entry.var_name for entry in entries], shelf)
    _keep_alive.append(mime)
    mods = (
        Qt.KeyboardModifier.AltModifier
        if copy
        else Qt.KeyboardModifier.NoModifier
    )
    set_active_shelf_drag(shelf, keys)
    try:
        event = QDropEvent(
            QPointF(10.0, 10.0),
            Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
            mime,
            Qt.MouseButton.LeftButton,
            mods,
        )
        dst.dropEvent(event)
    finally:
        clear_active_shelf_drag()


def test_shelf_move_applies_full_style_and_removes_entry(dst, shelf, qapp):
    entry = _style_entry("a")
    shelf.add_entry(entry)

    _drop(dst, shelf, [entry])

    assert "a" in dst.curves
    ci = dst.curves["a"]
    assert ci.color == "#123456"
    assert ci.draw_mode == DRAW_MODE_STEP_PRE
    # Blocking 2：apply 入口强制 source='user'，落地后必须回写快照 'auto'
    assert ci.draw_mode_source == DRAW_MODE_SOURCE_AUTO
    assert ci.symbol_policy == SYMBOL_POLICY_ON

    qapp.processEvents()  # 显隐恢复 singleShot(0)
    assert dst.curves["a"].visible is False
    assert shelf.count() == 0  # 移出：条目消失
    assert any("已移动 a：暂存区 → 页2·图1" in m for m in dst.window().messages)


def test_shelf_alt_copy_keeps_entry(dst, shelf):
    entry = _style_entry("a")
    shelf.add_entry(entry)

    _drop(dst, shelf, [entry], copy=True)

    assert "a" in dst.curves
    assert shelf.count() == 1  # 复制出：条目保留
    assert any("已复制 a：暂存区 → 页2·图1" in m for m in dst.window().messages)


def test_shelf_move_back_to_origin_is_legal_undo(dst, shelf):
    """拖回来源图 = 合法撤销：不得走 legend 的「source is self」早退。"""
    entry = _style_entry("a")
    shelf.add_entry(entry)

    _drop(dst, shelf, [entry])

    assert "a" in dst.curves
    assert shelf.count() == 0


def test_shelf_duplicate_skipped_and_summarized(dst, shelf):
    assert dst.plot_variable("a")
    entry = _style_entry("a")
    shelf.add_entry(entry)
    before = dst.curves["a"].color

    _drop(dst, shelf, [entry])

    assert dst.curves["a"].color == before  # 目标原曲线不动
    assert shelf.count() == 1  # 跳过条目不删
    assert any(
        "未搬运；已跳过 1 个已存在变量：a" in m for m in dst.window().messages
    )


def test_shelf_batch_emit_merged(dst, shelf):
    """批量搬运：curves_changed 只在收尾 emit 一次（≤1）。"""
    shelf.add_entry(_style_entry("a", "k1"))
    shelf.add_entry(_style_entry("b", "k2"))
    count = {"n": 0}
    dst.curves_changed.connect(lambda: count.__setitem__("n", count["n"] + 1))

    _drop(dst, shelf, shelf.entries)

    assert {"a", "b"} <= set(dst.curves)
    assert count["n"] == 1


def test_shelf_batch_finalize_called_once(dst, shelf, monkeypatch):
    shelf.add_entry(_style_entry("a", "k1"))
    shelf.add_entry(_style_entry("b", "k2"))
    shelf.add_entry(_style_entry("c", "k3"))
    calls = {"n": 0}
    original = dst._multi_curve_manager._finalize_batch_add

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(dst._multi_curve_manager, "_finalize_batch_add", counting)

    _drop(dst, shelf, shelf.entries)

    assert calls["n"] == 1


def test_shelf_quiet_failure_no_modal_and_keeps_entry(dst, shelf, monkeypatch):
    dialogs = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: dialogs.append(a))
    )
    monkeypatch.setattr(
        QMessageBox, "critical", staticmethod(lambda *a, **k: dialogs.append(a))
    )
    entry = ShelfEntry(key="k1", var_name="missing", origin_label="页1·图2")
    shelf.add_entry(entry)

    _drop(dst, shelf, [entry])

    assert "missing" not in dst.curves
    assert dialogs == []  # quiet：不弹模态
    assert shelf.count() == 1  # 失败条目不删托盘
    assert any("missing 数据校验失败，未搬运" in m for m in dst.window().messages)


def test_shelf_drop_rejected_while_loading(dst, shelf):
    dst.window().reject_when_loading = lambda action: True
    entry = _style_entry("a")
    shelf.add_entry(entry)

    _drop(dst, shelf, [entry])

    assert "a" not in dst.curves
    assert shelf.count() == 1


def test_shelf_drop_without_registry_ignored(dst, shelf):
    entry = _style_entry("a")
    shelf.add_entry(entry)
    mime = build_shelf_mimedata([entry.key], ["a"], shelf)
    _keep_alive.append(mime)
    # 不登记注册表：模拟托盘拖拽上下文失效
    event = QDropEvent(
        QPointF(1.0, 1.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    dst.dropEvent(event)
    assert "a" not in dst.curves


# ---------------------------------------------------------------------------
# hover 文案：SHELF 分支必须插在 else 之前（Blocking 4）
# ---------------------------------------------------------------------------


def _hover_text(dst, shelf, entries, *, copy=False):
    keys = [entry.key for entry in entries]
    mime = build_shelf_mimedata(keys, [entry.var_name for entry in entries], shelf)
    _keep_alive.append(mime)
    captured = {}

    def fake_notify(var_names=None, hide=False, source_widget=None, indicator_text=None):
        captured["names"] = var_names
        captured["hide"] = hide
        captured["text"] = indicator_text

    original = dst._notify_drag_indicator
    dst._notify_drag_indicator = fake_notify
    mods = (
        Qt.KeyboardModifier.AltModifier
        if copy
        else Qt.KeyboardModifier.NoModifier
    )
    set_active_shelf_drag(shelf, keys)
    try:
        event = QDropEvent(
            QPointF(1.0, 1.0),
            Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
            mime,
            Qt.MouseButton.LeftButton,
            mods,
        )
        dst._handle_drag_hover(event)
    finally:
        clear_active_shelf_drag()
        dst._notify_drag_indicator = original
    return captured


def test_shelf_hover_move_text(dst, shelf):
    entry = ShelfEntry(key="k1", var_name="a")
    captured = _hover_text(dst, shelf, [entry])
    assert captured["hide"] is False
    assert captured["text"] == "释放以移动"


def test_shelf_hover_copy_text(dst, shelf):
    entry = ShelfEntry(key="k1", var_name="a")
    captured = _hover_text(dst, shelf, [entry], copy=True)
    assert captured["text"] == "释放以复制"


def test_shelf_hover_duplicate_text(dst, shelf):
    assert dst.plot_variable("a")
    entry = ShelfEntry(key="k1", var_name="a")
    captured = _hover_text(dst, shelf, [entry])
    assert captured["text"] == "变量已存在，释放将跳过"


def test_shelf_hover_partial_duplicate_text(dst, shelf):
    """部分已存在（dst 预置 c）：文案提示将跳过，而不是承诺全部移动。"""
    entries = [
        ShelfEntry(key="k1", var_name="a"),
        ShelfEntry(key="k2", var_name="c"),
    ]
    captured = _hover_text(dst, shelf, entries)
    assert captured["text"] == "部分变量已存在，释放将跳过"
