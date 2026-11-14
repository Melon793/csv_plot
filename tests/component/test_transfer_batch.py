"""批量搬运共享助手（_transfer_curves）语义（offscreen）。

覆盖施工计划 §9.1 的批量协议护栏：
- 判重策略 ask / notify / skip 显式区分，单变量与批量不靠 len 推断；
- 失败（quiet）不弹模态、不触发源端删除；
- 样式保真含 draw_mode_source 回写（Blocking 2）；
- 批量收尾唯一且 emit 合并（≤1 次）；
- 异常路径 finally 复位 _batch_adding。
"""

import pytest

from PySide6.QtWidgets import QMessageBox

from src.core.data_types import (
    DRAW_MODE_SOURCE_AUTO,
    DRAW_MODE_STEP_POST,
    SYMBOL_POLICY_OFF,
)
from src.ui.drag_drop import CurveStyleSnapshot


@pytest.fixture()
def dst(plot_factory):
    widget = plot_factory()
    assert widget.plot_variable("c")
    return widget


def _record_dialogs(monkeypatch):
    records = {"question": [], "information": [], "warning": [], "critical": []}
    answer = {"question": QMessageBox.StandardButton.No}

    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(
            lambda parent, title, text, *a, **k: (
                records["question"].append((title, text))
                or answer["question"]
            )
        ),
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(
            lambda parent, title, text, *a, **k: (
                records["information"].append((title, text))
                or QMessageBox.StandardButton.Ok
            )
        ),
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: records["warning"].append(a)),
    )
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        staticmethod(lambda *a, **k: records["critical"].append(a)),
    )
    return records, answer


def test_ask_yes_removes_source_when_target_has_name(dst, monkeypatch):
    records, answer = _record_dialogs(monkeypatch)
    answer["question"] = QMessageBox.StandardButton.Yes
    assert dst.plot_variable("a")
    removed: list[str] = []

    result = dst._transfer_curves(
        ["a"],
        on_duplicate="ask",
        remove_source_cb=removed.extend,
        quiet=False,
        batch=False,
    )

    assert result.merged == ["a"] and result.added == []
    assert len(records["question"]) == 1  # 保留既有确认窗
    assert removed == ["a"]


def test_ask_no_keeps_both_sides(dst, monkeypatch):
    records, answer = _record_dialogs(monkeypatch)
    answer["question"] = QMessageBox.StandardButton.No
    assert dst.plot_variable("a")
    removed: list[str] = []

    result = dst._transfer_curves(
        ["a"],
        on_duplicate="ask",
        remove_source_cb=removed.extend,
        quiet=False,
        batch=False,
    )

    assert result.skipped == ["a"] and result.merged == []
    assert records["question"]
    assert removed == []


def test_notify_shows_information_and_skips(dst, monkeypatch):
    records, _ = _record_dialogs(monkeypatch)
    assert dst.plot_variable("a")

    result = dst._transfer_curves(
        ["a"], on_duplicate="notify", quiet=False, batch=False
    )

    assert result.skipped == ["a"]
    assert len(records["information"]) == 1  # 保留既有提示
    assert records["question"] == []


def test_skip_is_silent(dst, monkeypatch):
    records, _ = _record_dialogs(monkeypatch)
    assert dst.plot_variable("a")

    result = dst._transfer_curves(
        ["a"], on_duplicate="skip", quiet=True, batch=False
    )

    assert result.skipped == ["a"]
    assert records == {
        "question": [],
        "information": [],
        "warning": [],
        "critical": [],
    }


def test_quiet_failure_no_modal_and_no_source_removal(dst, monkeypatch):
    records, _ = _record_dialogs(monkeypatch)
    removed: list[str] = []

    result = dst._transfer_curves(
        ["missing"],
        on_duplicate="skip",
        remove_source_cb=removed.extend,
        quiet=True,
        batch=False,
    )

    assert result.failed == ["missing"]
    assert result.added == []
    assert removed == []  # 校验失败绝不能删源
    assert records["warning"] == [] and records["critical"] == []


def test_style_source_not_pinned_to_user(dst):
    style = CurveStyleSnapshot(
        color="#abcdef",
        visible=True,
        draw_mode=DRAW_MODE_STEP_POST,
        symbol_policy=SYMBOL_POLICY_OFF,
        draw_mode_source=DRAW_MODE_SOURCE_AUTO,
    )

    result = dst._transfer_curves(
        ["a"],
        style_map={"a": style},
        on_duplicate="skip",
        quiet=True,
        batch=False,
    )

    assert result.added == ["a"]
    ci = dst.curves["a"]
    assert ci.color == "#abcdef"
    assert ci.draw_mode == DRAW_MODE_STEP_POST
    assert ci.symbol_policy == SYMBOL_POLICY_OFF
    # apply_curve_draw_mode 内部会强制 'user'，助手必须回写快照来源
    assert ci.draw_mode_source == DRAW_MODE_SOURCE_AUTO


def test_batch_emits_once_and_finalizes_once(dst, monkeypatch):
    count = {"emit": 0, "finalize": 0}
    dst.curves_changed.connect(lambda: count.__setitem__("emit", count["emit"] + 1))
    original = dst._multi_curve_manager._finalize_batch_add

    def counting(*args, **kwargs):
        count["finalize"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(dst._multi_curve_manager, "_finalize_batch_add", counting)

    result = dst._transfer_curves(
        ["a", "b"], on_duplicate="skip", quiet=True
    )

    assert result.added == ["a", "b"]
    assert count["finalize"] == 1
    assert count["emit"] == 1


def test_batch_flag_reset_on_exception(dst, monkeypatch):
    real_plot = dst.plot_variable
    calls = {"n": 0}

    def flaky(name, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return real_plot(name, *args, **kwargs)

    monkeypatch.setattr(dst, "plot_variable", flaky)

    with pytest.raises(RuntimeError):
        dst._transfer_curves(["a", "b"], on_duplicate="skip", quiet=True)

    assert dst._batch_adding is False


def test_single_variable_batch_false_keeps_first_curve_emit_behavior(plot_factory):
    """batch=False 的单变量路径不改变既有首绘 emit 语义。"""
    widget = plot_factory()  # 空 plot：首绘应 emit（与改动前一致）
    count = {"n": 0}
    widget.curves_changed.connect(lambda: count.__setitem__("n", count["n"] + 1))

    result = widget._transfer_curves(
        ["a"], on_duplicate="skip", quiet=True, batch=False
    )

    assert result.added == ["a"]
    assert count["n"] == 1
