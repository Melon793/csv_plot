"""第 5 轮批次 3 回归：隐藏页搬运数据锁 / shelf 悬停文案 / QDrag 释放。

P1-4：scene=False 搬运与隐藏页显隐恢复不得写游标场景（_is_updating_data 锁）。
P1-5：shelf 拖拽悬停的 120ms 轮询必须保持 移动/复制+skip 文案。
P1-6：所有 QDrag(self) 位点 exec 后必须 deleteLater（防长会话泄漏）。
"""
from __future__ import annotations

import ast
import types
from pathlib import Path

import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from src.ui import drag_drop


def _make_pw(plot_factory, df):
    pw = plot_factory(df)
    pw.resize(500, 300)
    pw.window().show()
    pw.show()
    for name in df.columns:
        assert pw.plot_variable(name)
    pw.show_values_only = False
    return pw


def _make_pw_single(plot_factory, df, plotted="a"):
    """只绘 plotted 一列：搬运用例必须搬「未绘制」变量，否则前置判重 skip，
    finalize 链根本不执行（评审 B1：空转护栏）。"""
    pw = plot_factory(df)
    pw.resize(500, 300)
    pw.window().show()
    pw.show()
    assert pw.plot_variable(plotted)
    pw.show_values_only = False
    return pw


def _spy_cursor(pw):
    calls = {"n": 0}
    cm = pw._cursor_manager
    orig = cm._update_multi_curve_cursor_label

    def spy():
        calls["n"] += 1
        return orig()

    cm._update_multi_curve_cursor_label = spy
    return calls


# ---------- P1-4a：scene=False 搬运全程置数据锁 ----------

def test_hidden_page_transfer_locks_cursor_chain(plot_factory, qapp):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    pw = _make_pw_single(plot_factory, df)
    pw.vline.setVisible(True)
    # 模拟隐藏页：page 与宿主 current_page 不同一
    pw.page = types.SimpleNamespace()
    pw.window().current_page = types.SimpleNamespace()
    # 先 flush 初始绘图触发的防抖刷新并复位交互态，隔离被测搬运段
    QTest.qWait(400)
    qapp.processEvents()
    pw._is_interacting = False
    timer = getattr(pw, "_interaction_timer", None)
    if timer is not None:
        timer.stop()
    pw._cancel_ui_refresh()
    calls = _spy_cursor(pw)

    result = pw._transfer_curves(["b"], batch=True, scene=False, quiet=True)
    assert result.added == ["b"], "必须真走 add+finalize 链（判重 skip 测不到锁）"
    assert pw._is_updating_data is False, "锁必须在 finally 复位"
    # 泵防抖/延迟 autoRange：修复前锁释放后落地的 range-change 会写隐藏页场景
    QTest.qWait(400)
    qapp.processEvents()
    assert calls["n"] == 0, "scene=False 搬运不得触发游标标签更新（隐藏页场景写入）"


def test_visible_page_transfer_still_updates_cursor(plot_factory, qapp):
    """正面护栏：可见页（scene=True）搬运的场景收尾不得被误锁。"""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    pw = _make_pw_single(plot_factory, df)
    pw.vline.setVisible(True)
    pw.page = types.SimpleNamespace()
    pw.window().current_page = pw.page  # 当前页可见
    QTest.qWait(400)
    qapp.processEvents()
    pw._is_interacting = False
    timer = getattr(pw, "_interaction_timer", None)
    if timer is not None:
        timer.stop()
    pw._cancel_ui_refresh()
    after_plot = {"n": 0}
    orig_after = pw._update_cursor_after_plot

    def after_spy(*a, **k):
        after_plot["n"] += 1
        return orig_after(*a, **k)

    pw._update_cursor_after_plot = after_spy
    calls = _spy_cursor(pw)
    # 评审 N-A：补 cursor_btn 替身，让场景收尾真走游标启用支，
    # 正面护栏同时断言标签刷新（只数 after_plot 抓不住「可见页被误锁」）
    pw.window().cursor_btn = types.SimpleNamespace(isChecked=lambda: True)
    pw.window().cursor_values_hidden = False

    result = pw._transfer_curves(["b"], batch=True, scene=True, quiet=True)
    assert result.added == ["b"]
    assert after_plot["n"] >= 1, "可见页搬运必须执行游标场景收尾"
    assert calls["n"] >= 1, "可见页搬运后游标标签必须照常刷新（不得被误锁）"
    assert pw._is_updating_data is False


# ---------- P1-4b：隐藏页延迟显隐恢复不写游标场景 ----------

def test_visibility_restore_on_hidden_page_skips_cursor_write(plot_factory, qapp):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    pw = _make_pw(plot_factory, df)
    pw.vline.setVisible(True)
    QTest.qWait(400)
    qapp.processEvents()
    calls = _spy_cursor(pw)
    # 模拟目标页已切走：page 与宿主 current_page 不同一
    pw.page = types.SimpleNamespace()
    pw.window().current_page = types.SimpleNamespace()

    pw._schedule_visibility_restore("a", False)
    qapp.processEvents()  # singleShot(0) 落地
    assert pw.curves["a"].visible is False, "显隐状态本身必须恢复"
    assert calls["n"] == 0, "隐藏页的 toggle 尾部不得写游标场景"
    assert pw._is_updating_data is False, "锁必须复位"


def test_visibility_restore_on_visible_page_keeps_cursor_write(plot_factory, qapp):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    pw = _make_pw(plot_factory, df)
    pw.vline.setVisible(True)
    QTest.qWait(400)
    qapp.processEvents()
    calls = _spy_cursor(pw)
    pw.page = types.SimpleNamespace()
    pw.window().current_page = pw.page  # 当前页可见

    pw._schedule_visibility_restore("a", False)
    qapp.processEvents()
    assert pw.curves["a"].visible is False
    assert calls["n"] >= 1, "可见页显隐恢复必须照常刷新游标"


# ---------- P1-5：shelf 悬停轮询文案 ----------

def test_shelf_drag_hover_polling_keeps_move_copy_text(plot_factory, qapp):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    pw = plot_factory(df)
    pw.resize(500, 300)
    pw.show()
    host = pw.window()
    recorded = []
    host.layout_manager = types.SimpleNamespace(
        _show_drag_indicator_for_plot=lambda p, names, text=None: recorded.append(text),
        _hide_drag_indicator_for_plot=lambda p: None,
        request_mark_stats_refresh=lambda immediate=False: None,
        _get_plot_container=lambda p: types.SimpleNamespace(
            plot_widget=p, isVisible=lambda: True),
    )
    host._active_drag_container = types.SimpleNamespace(plot_widget=pw)
    pw._drag_indicator_source = None
    pw._drag_indicator_var_names = ["a"]
    # 置一个与正确文案不同的旧值：轮询只在文案变化时记录
    pw._drag_indicator_last_text = "释放以添加"
    # 几何守卫不在靶区（offscreen 无真实光标），直通轮询分支
    pw._should_hide_drag_indicator = lambda mw: False
    drag_drop.set_active_shelf_drag(object(), ["k"])
    try:
        pw._enforce_drag_indicator_visibility()
    finally:
        drag_drop.clear_active_shelf_drag()
    assert recorded, "轮询必须产出文案"
    text = recorded[-1]
    assert "移动" in text, f"shelf 拖拽无修饰键=移出，实际文案={text!r}"
    assert "添加" not in text and "替换" not in text, (
        f"不得落入变量列表 Shift 分支，实际文案={text!r}"
    )


def test_shelf_drag_hover_polling_alt_switches_to_copy(plot_factory, qapp, monkeypatch):
    """P1-5 修饰键支：Alt=复制出，轮询静止悬停时也须实时切换（评审 N5）。"""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    pw = plot_factory(df)
    pw.resize(500, 300)
    pw.show()
    host = pw.window()
    recorded = []
    host.layout_manager = types.SimpleNamespace(
        _show_drag_indicator_for_plot=lambda p, names, text=None: recorded.append(text),
        _hide_drag_indicator_for_plot=lambda p: None,
        request_mark_stats_refresh=lambda immediate=False: None,
        _get_plot_container=lambda p: types.SimpleNamespace(
            plot_widget=p, isVisible=lambda: True),
    )
    host._active_drag_container = types.SimpleNamespace(plot_widget=pw)
    pw._drag_indicator_source = None
    pw._drag_indicator_var_names = ["a"]
    pw._drag_indicator_last_text = "释放以移动"
    pw._should_hide_drag_indicator = lambda mw: False
    monkeypatch.setattr(
        "src.ui.widgets.plot_widget.QApplication.queryKeyboardModifiers",
        staticmethod(lambda: Qt.KeyboardModifier.AltModifier),
    )
    drag_drop.set_active_shelf_drag(object(), ["k"])
    try:
        pw._enforce_drag_indicator_visibility()
    finally:
        drag_drop.clear_active_shelf_drag()
    assert recorded and "复制" in recorded[-1], f"Alt 须切换复制文案：{recorded}"


# ---------- P1-6：QDrag 释放 ----------

class _FakeDrag:
    """记录 deleteLater 是否被调用的 QDrag 替身（exec 不进嵌套循环）。"""

    instances: list["_FakeDrag"] = []

    def __init__(self, parent):
        self.parent = parent
        self.deleted = False
        _FakeDrag.instances.append(self)

    def setMimeData(self, mime):
        self._mime = mime

    def setPixmap(self, pixmap):
        self._pixmap = pixmap

    def setHotSpot(self, point):
        pass

    def exec(self, *args, **kwargs):
        return Qt.DropAction.MoveAction

    def deleteLater(self):
        self.deleted = True


def test_shelf_start_drag_releases_qdrag(qapp, monkeypatch):
    import src.ui.widgets.curve_shelf as cs
    from tests.component.test_curve_shelf import ShelfHost, _drop_into_shelf

    _FakeDrag.instances.clear()
    widget = cs.CurveShelfWidget(main_window=ShelfHost())
    try:
        mime = drag_drop.build_var_mimedata(["a"])
        _drop_into_shelf(widget, mime)
        widget.selectAll()
        assert widget.selected_entries(), "入区+全选后必须有可拖条目"
        monkeypatch.setattr(cs, "QDrag", _FakeDrag)
        widget.startDrag(Qt.DropAction.CopyAction | Qt.DropAction.MoveAction)
        assert _FakeDrag.instances, "startDrag 必须构造 QDrag"
        assert _FakeDrag.instances[-1].deleted is True, (
            "exec 返回后必须 deleteLater，否则 QDrag/QMimeData/缩略图累积到源控件销毁"
        )
    finally:
        widget.deleteLater()


def test_all_qdrag_sites_call_deletelater():
    """AST 护栏：src 内每个构造 QDrag 的函数必须有 deleteLater 调用。

    与三参 singleShot 护栏同思路：QDrag(self) 的 C++ 所有权挂源控件，
    exec 返回后 Python 包装器释放不会析构 C++ 对象，长会话反复拖拽会
    累积 QDrag+QMimeData+缩略图（DPR2 下约 1MB/次）。

    局限（评审 N2，粗护栏口径）：只匹配裸名 QDrag 构造与 FunctionDef  scope；
    属性式 QtGui.QDrag、模块/类体级构造漏报；函数内任意对象的 deleteLater
    即满足判定（不绑定到 drag 变量）。当前 5 处位点全覆盖。
    """
    src_root = Path(__file__).resolve().parents[2] / "src"
    offenders = []
    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            has_qdrag = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "QDrag"
                for n in ast.walk(node)
            )
            if not has_qdrag:
                continue
            has_release = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "deleteLater"
                for n in ast.walk(node)
            )
            if not has_release:
                offenders.append(f"{path.name}:{node.lineno} {node.name}")
    assert not offenders, f"QDrag 位点缺 deleteLater：{offenders}"