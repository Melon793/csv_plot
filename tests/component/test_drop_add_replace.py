"""变量列表来源拖拽（add/replace）component 测试（offscreen）。

模拟 help.md 3️⃣「绘制图表」中的变量拖拽操作：
- 默认拖拽到绘图区 = 添加（首绘/追加/多变量批量）
- Shift + 拖拽 = 替换（先清空再绘制）
- 拖拽悬停提示文字随 Shift 状态切换（"释放以添加"/"释放以替换"/"变量已存在…"）
- 拖入不存在变量时的防御路径（弹窗提示、不改动曲线）
- curves_changed 信号在曲线集合变化时 emit

test_legend_drop.py 已覆盖 legend 来源与 add/replace 基础回归，
本文件补齐多变量批量、悬停指示器与异常路径。
"""

import pytest

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QMessageBox

from src.ui.drag_drop import build_var_mimedata


# 保活容器：offscreen 下手工构造的 QMimeData 不受 Qt 事件系统接管，
# 局部变量被 GC 后 dropEvent 内访问 mimeData 会 SIGSEGV（README 陷阱 #2）
_keep_alive: list = []


@pytest.fixture()
def silent_dialogs(monkeypatch):
    """替换 QMessageBox 静态方法为记录器，避免 offscreen 模态阻塞（陷阱 #4）"""
    calls = {"warning": [], "information": [], "question": []}

    def fake_warning(parent, title, text, *args, **kwargs):
        calls["warning"].append((title, text))
        return QMessageBox.StandardButton.Ok

    def fake_information(parent, title, text, *args, **kwargs):
        calls["information"].append((title, text))
        return QMessageBox.StandardButton.Ok

    def fake_question(parent, title, text, *args, **kwargs):
        calls["question"].append((title, text))
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(fake_warning))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(fake_information))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))
    return calls


def _drop(pw, var_names: list[str], *, shift: bool = False):
    """模拟一次变量列表来源的 drop"""
    mods = Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
    mime = build_var_mimedata(var_names)
    _keep_alive.append(mime)
    ev = QDropEvent(
        QPointF(10.0, 10.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        mods,
    )
    pw.dropEvent(ev)


def _drag_enter(pw, var_names: list[str], *, shift: bool = False):
    """模拟一次变量列表来源的 dragEnter，返回指示器记录列表"""
    mods = Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
    mime = build_var_mimedata(var_names)
    _keep_alive.append(mime)
    records = []
    pw._notify_drag_indicator = (
        lambda var_names=None, hide=False, source_widget=None, indicator_text=None:
        records.append({"var_names": var_names, "hide": hide, "text": indicator_text})
    )
    ev = QDragEnterEvent(
        QPoint(10, 10),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        mods,
    )
    pw.dragEnterEvent(ev)
    return records


# ---------- drop 添加 / 替换 ----------

def test_drop_single_to_empty_plot_first_render(plot_factory, silent_dialogs):
    """拖入单变量到空白 plot = 首绘"""
    pw = plot_factory()
    assert len(pw.curves) == 0

    _drop(pw, ["a"])

    assert list(pw.curves) == ["a"]


def test_drop_multiple_appends_batch(plot_factory, silent_dialogs):
    """多变量一次拖入 = 批量添加（add_variables_to_plot 路径）"""
    pw = plot_factory()
    assert pw.plot_variable("a")

    _drop(pw, ["b", "c"])

    assert list(pw.curves) == ["a", "b", "c"]


def test_drop_multiple_to_empty_plot(plot_factory, silent_dialogs):
    """多变量拖入空白 plot：全部绘制"""
    pw = plot_factory()

    _drop(pw, ["a", "b", "c"])

    assert list(pw.curves) == ["a", "b", "c"]


def test_shift_drop_multiple_replaces_all(plot_factory, silent_dialogs):
    """Shift + 多变量拖入 = 全量替换"""
    pw = plot_factory()
    assert pw.plot_variable("a")

    _drop(pw, ["b", "c"], shift=True)

    assert list(pw.curves) == ["b", "c"]


def test_shift_drop_to_empty_plot(plot_factory, silent_dialogs):
    """Shift 拖入空白 plot：等价于首绘（清空空集合无副作用）"""
    pw = plot_factory()

    _drop(pw, ["b"], shift=True)

    assert list(pw.curves) == ["b"]


def test_drop_nonexistent_variable_is_safe(plot_factory, silent_dialogs):
    """拖入不存在的变量：弹窗提示、曲线不受影响、不抛异常"""
    pw = plot_factory()
    assert pw.plot_variable("a")

    _drop(pw, ["zzz_not_exist"])

    assert list(pw.curves) == ["a"]
    assert len(silent_dialogs["warning"]) + len(silent_dialogs["information"]) >= 1


def test_curves_changed_emitted_on_drop_add(plot_factory, silent_dialogs, qapp):
    """drop 添加成功时 curves_changed 信号 emit"""
    pw = plot_factory()
    emissions = []
    pw.curves_changed.connect(lambda: emissions.append(1))

    _drop(pw, ["a"])          # 首绘
    _drop(pw, ["b", "c"])     # 批量追加

    assert len(emissions) >= 2


# ---------- dragEnter 悬停指示器文案 ----------

def test_hover_text_add_without_shift(plot_factory, silent_dialogs):
    """无修饰键悬停：提示「释放以添加」"""
    pw = plot_factory()
    records = _drag_enter(pw, ["a"])

    assert records and records[-1]["hide"] is False
    assert records[-1]["text"] == "释放以添加"


def test_hover_text_replace_with_shift(plot_factory, silent_dialogs):
    """Shift 悬停：提示切换为「释放以替换」（help.md：随 Shift 状态实时切换）"""
    pw = plot_factory()
    records = _drag_enter(pw, ["a"], shift=True)

    assert records and records[-1]["hide"] is False
    assert records[-1]["text"] == "释放以替换"


def test_hover_text_existing_variable(plot_factory, silent_dialogs):
    """目标已含该变量：提示「变量已存在」（添加动作不产生重复曲线）"""
    pw = plot_factory()
    assert pw.plot_variable("a")

    records = _drag_enter(pw, ["a"])
    assert records[-1]["text"] == "变量已存在"

    records = _drag_enter(pw, ["a"], shift=True)
    assert records[-1]["text"] == "变量已存在，释放以替换"


def test_drag_enter_without_text_ignored(plot_factory, silent_dialogs):
    """无文本 MIME 的 dragEnter：ignore 且隐藏指示器"""
    from PySide6.QtCore import QMimeData
    pw = plot_factory()
    records = []
    pw._notify_drag_indicator = (
        lambda var_names=None, hide=False, source_widget=None, indicator_text=None:
        records.append({"hide": hide})
    )
    mime = QMimeData()  # 无 text
    _keep_alive.append(mime)
    ev = QDragEnterEvent(
        QPoint(10, 10),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    pw.dragEnterEvent(ev)

    assert records and records[-1]["hide"] is True
    assert not ev.isAccepted()
