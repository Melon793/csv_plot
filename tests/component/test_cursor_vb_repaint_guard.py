"""CursorManager ViewBox 强制重绘的级联信号防护（V6.0 P0-2 防回归）。

背景：对 ViewBox 的强制重绘会触发 sigRangeChanged 等级联回调，回调重新进入
update_cursor_label 会交叉修改 BSP 树 → SIGSEGV（不可被 except 捕获）。
v5.3 只给多曲线路径加了 QSignalBlocker，show_values_only 默认路径漏修。
"""

import inspect

import pytest

from src.ui.widgets import cursor_manager as cursor_manager_module
from src.ui.widgets.cursor_manager import CursorManager


class _RecordingBlocker:
    """记录构造参数的 QSignalBlocker 替身（保留 with 协议）"""

    instances = []

    def __init__(self, obj):
        self.obj = obj
        _RecordingBlocker.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture()
def recording_blocker(monkeypatch):
    _RecordingBlocker.instances = []
    monkeypatch.setattr(
        cursor_manager_module, "QSignalBlocker", _RecordingBlocker
    )
    yield _RecordingBlocker


def test_repaint_view_box_uses_signal_blocker(plot_factory, qapp, recording_blocker):
    """_repaint_view_box 必须在阻断 view_box 信号的上下文中触发重绘"""
    pw = plot_factory()
    vb = pw.view_box
    assert vb.signalsBlocked() is False

    pw._cursor_manager._repaint_view_box(vb)

    assert [b.obj is vb for b in recording_blocker.instances] == [True]
    # 退出上下文后阻塞已解除，不影响后续正常信号
    assert vb.signalsBlocked() is False


def test_viewbox_repaint_has_single_unguarded_call_site():
    """全仓 view_box.update() 只允许出现在 _repaint_view_box 内部（禁止双实现漂移）"""
    src = inspect.getsource(CursorManager._repaint_view_box)
    assert src.count("view_box.update()") == 1

    module_src = inspect.getsource(cursor_manager_module)
    assert module_src.count("view_box.update()") == 1
