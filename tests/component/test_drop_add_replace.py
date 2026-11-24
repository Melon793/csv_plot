"""变量列表来源拖拽（add/replace）component 测试（offscreen）。

模拟 help.md 3️⃣「绘制图表」中的变量拖拽操作：
- 默认拖拽到绘图区 = 添加（首绘/追加/多变量批量）
- Shift + 拖拽 = 先校验再替换当前子图（全部失败保留原图）
- 拖拽悬停提示文字随 Shift 状态切换（"释放以添加"/"释放以替换当前子图"）
- 拖入不存在变量时的防御路径（弹窗提示、不改动曲线）
- curves_changed 信号在曲线集合变化时 emit

test_legend_drop.py 已覆盖 legend 来源与 add/replace 基础回归，
本文件补齐多变量批量、悬停指示器与异常路径。
"""

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QMessageBox

from src.ui.drag_drop import build_var_mimedata
from tests.fixtures.data_factory import make_simple_rows


def _demo_data():
    """合成变量：有效、全 NaN、空值和非数值列。"""
    rows = make_simple_rows(20)
    df = pd.DataFrame(rows, columns=["time", "speed", "rpm", "flag"])
    df["all_nan"] = np.full(len(df), np.nan)
    df["empty"] = [None] * len(df)
    df["bad"] = ["not-a-number"] * len(df)
    return df


def _array_snapshot(array):
    array = np.asarray(array)
    return array.dtype.str, array.shape, array.tobytes()


def _curve_snapshot(pw):
    return [
        (
            name,
            ci,
            ci.curve,
            ci.curve.isVisible(),
            ci.color,
            ci.line_preset,
            ci.display_name,
            ci.factor,
            ci.offset,
            ci.draw_mode,
            ci.symbol_policy,
            ci.visible,
            _array_snapshot(ci.x_data),
            _array_snapshot(ci.y_data),
            _array_snapshot(ci.y_raw),
        )
        for name, ci in pw.curves.items()
    ]


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


@pytest.mark.parametrize(
    "var_names",
    [[], ["missing"], ["missing", "also_missing"], ["all_nan"], ["empty"], ["bad"]],
)
def test_shift_drop_all_invalid_preserves_current_subplot(
    plot_factory, silent_dialogs, var_names
):
    """空载荷/全部无效/解析失败：清空函数绝不能被调用，原曲线资产不变。"""
    pw = plot_factory(_demo_data())
    assert pw.plot_variable("speed")
    assert pw.plot_variable("rpm")
    assert pw.plot_variable("flag")
    manager = pw._multi_curve_manager
    assert manager.apply_curve_line_preset("speed", "dash")
    assert manager.apply_curve_display_name("speed", "Demo speed")
    assert manager.apply_curve_transform("rpm", 2.0, -1.0, allow_y_rescue=False)
    manager.toggle_curve_visibility_by_name("flag")
    assert pw.curves["flag"].visible is False
    pw.demote_to_invalid(
        "ghost_var", "#123456", False,
        display_name="Demo ghost", factor=3.0, offset=2.0,
    )
    before = _curve_snapshot(pw)
    invalid_before = dict(pw.invalid_curves)
    clear_calls = []
    pw.clear_plot_item = lambda **kwargs: clear_calls.append(kwargs)

    _drop(pw, var_names, shift=True)

    assert clear_calls == []
    assert _curve_snapshot(pw) == before
    assert pw.invalid_curves == invalid_before
    assert len(silent_dialogs["information"]) == 1
    assert "当前子图未更改" in silent_dialogs["information"][0][1]


def test_shift_drop_prepares_every_variable_before_clear(
    plot_factory, silent_dialogs, monkeypatch
):
    """部分有效也必须先完成全部校验/备数据，再清空；提交不重复解析。"""
    pw = plot_factory(_demo_data())
    assert pw.plot_variable("speed")
    events = []
    original_prepare = pw._prepare_plot_data
    original_validate = pw._validate_plot_data
    original_clear = pw.clear_plot_item

    def validate(name):
        events.append(("validate", name))
        return original_validate(name)

    def prepare(name):
        events.append(("prepare", name))
        return original_prepare(name)

    def clear(**kwargs):
        events.append(("clear", kwargs))
        return original_clear(**kwargs)

    monkeypatch.setattr(pw, "_prepare_plot_data", prepare)
    monkeypatch.setattr(pw, "_validate_plot_data", validate)
    monkeypatch.setattr(pw, "clear_plot_item", clear)

    _drop(pw, ["rpm", "bad", "missing"], shift=True)

    assert events[:5] == [
        ("validate", "rpm"),
        ("prepare", "rpm"),
        ("validate", "bad"),
        ("prepare", "bad"),
        ("validate", "missing"),
    ]
    assert events[5][0] == "clear"
    assert sum(event[0] == "prepare" for event in events) == 2
    assert list(pw.curves) == ["rpm"]
    message = silent_dialogs["information"][0][1]
    assert "成功 1" in message and "missing" in message and "bad" in message
    assert "当前子图" in message


def test_shift_drop_partial_success_skips_failures_without_false_success(
    plot_factory, silent_dialogs
):
    """部分有效：替换为有效项，反馈同时列出成功与失败，不假报全部成功。"""
    pw = plot_factory(_demo_data())
    assert pw.plot_variable("speed")
    assert pw.plot_variable("flag")

    _drop(pw, ["rpm", "missing"], shift=True)

    assert list(pw.curves) == ["rpm"]
    assert len(silent_dialogs["information"]) == 1
    message = silent_dialogs["information"][0][1]
    assert "成功 1" in message and "失败 1" in message
    assert "missing" in message


def test_shift_drop_same_name_is_replacement_not_duplicate_skip(
    plot_factory, silent_dialogs
):
    """目标已有同名变量：替换后仍绘制；输入重复项只绘制一次。"""
    pw = plot_factory(_demo_data())
    assert pw.plot_variable("speed")
    assert pw.plot_variable("flag")
    old_ci = pw.curves["speed"]

    _drop(pw, ["speed", "speed", "rpm"], shift=True)

    assert list(pw.curves) == ["speed", "rpm"]
    assert pw.curves["speed"] is not old_ci
    assert silent_dialogs["information"] == []


def test_shift_drop_parse_exception_preserves_current_subplot(
    plot_factory, silent_dialogs, monkeypatch
):
    """MIME 解析本身抛错时也不能清空。"""
    pw = plot_factory(_demo_data())
    assert pw.plot_variable("speed")
    before = _curve_snapshot(pw)

    def fail_parse(_mime):
        raise ValueError("Demo malformed payload")

    monkeypatch.setattr("src.ui.widgets.plot_widget.parse_var_names_from_mimedata", fail_parse)
    mime = build_var_mimedata(["rpm"])
    _keep_alive.append(mime)
    ev = QDropEvent(
        QPointF(10.0, 10.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ShiftModifier,
    )

    pw.dropEvent(ev)

    assert _curve_snapshot(pw) == before
    assert len(silent_dialogs["information"]) == 1
    assert "当前子图未更改" in silent_dialogs["information"][0][1]


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
    assert records[-1]["text"] == "释放以替换当前子图"


def test_hover_text_existing_variable(plot_factory, silent_dialogs):
    """目标已含该变量：提示「变量已存在」（添加动作不产生重复曲线）"""
    pw = plot_factory()
    assert pw.plot_variable("a")

    records = _drag_enter(pw, ["a"])
    assert records[-1]["text"] == "变量已存在"

    records = _drag_enter(pw, ["a"], shift=True)
    assert records[-1]["text"] == "变量已存在，释放以替换当前子图"


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
