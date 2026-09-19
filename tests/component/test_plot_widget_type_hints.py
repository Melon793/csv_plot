"""P2-1：`plot_widget` 的类型注解必须可求值。

`from __future__ import annotations` 让注解在导入期与运行期都不求值，所以
「用了没导入的名字」不会崩 —— 代价是它一直潜伏，直到有人真的求值注解
（`typing.get_type_hints`、文档生成、IDE/pydantic 式校验、`inspect` 深度内省）
才炸成 `NameError`。`_notify_drag_indicator` 的 `source_widget: QWidget | None`
就是这么一个（`QWidget` 全文件未导入）。
"""

from __future__ import annotations

import inspect
import typing

import pytest

from src.ui.widgets.plot_widget import DraggableGraphicsLayoutWidget


def test_notify_drag_indicator_hints_resolve(qapp):
    hints = typing.get_type_hints(
        DraggableGraphicsLayoutWidget._notify_drag_indicator
    )

    assert "source_widget" in hints
    assert hints["source_widget"] is not None


def test_every_method_annotation_resolves(qapp):
    """整类扫一遍：漏一个导入就是本用例红，而不是等下游工具去踩。"""
    broken = []
    for name, fn in inspect.getmembers(
        DraggableGraphicsLayoutWidget, predicate=inspect.isfunction
    ):
        try:
            typing.get_type_hints(fn)
        except Exception as e:  # noqa: BLE001
            broken.append(f"{name}: {type(e).__name__}: {e}")

    assert broken == [], "注解无法求值的方法：\n" + "\n".join(broken)


@pytest.mark.parametrize("name", ["QWidget", "QApplication", "QMessageBox"])
def test_annotated_qt_types_are_imported(qapp, name):
    import src.ui.widgets.plot_widget as pw_mod

    assert hasattr(pw_mod, name), f"plot_widget 用到 {name} 的注解却没导入它"
