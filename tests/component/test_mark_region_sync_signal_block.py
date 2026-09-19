"""标记区跨子图同步的信号阻塞（V6.0 P1-3 防回归）。

`QSignalBlocker(mark)` 写成临时对象语句时，语句结束即析构并解除阻塞，
紧随其后的 `mark.setRegion(...)` 照常发射 sigRegionChanged —— 阻塞完全无效，
只能靠 `_is_syncing_mark_region` 标志兜底防递归，白白产生级联回调。
（同文件 `_unregister_global_event_filter` 的 `with QSignalBlocker(mw.main_splitter)`
是正确写法；V4.0 也在 plot_data_manager 修过同类回归。）
"""

from types import SimpleNamespace

import pytest

from PySide6.QtCore import QObject

from src.ui.layout_manager import LayoutManager


class _FakeMark(QObject):
    """记录 setRegion 时刻自身信号是否真被阻塞的替身"""

    def __init__(self, region=(0.0, 1.0)):
        super().__init__()
        self._region = region
        self.blocked_at_set_region = []

    def getRegion(self):
        return self._region

    def setRegion(self, region):
        self.blocked_at_set_region.append(self.signalsBlocked())
        self._region = tuple(region)


class _FakeMW:
    """可弱引用的主窗口替身（manager 基类以 weakref 持有宿主）"""

    def __init__(self, plot_widgets):
        self.plot_widgets = plot_widgets
        self._is_syncing_mark_region = False
        self.mark_stats_window = None


class _FakeContainer:
    def __init__(self, mark):
        self._mark = mark
        self.plot_widget = SimpleNamespace(mark_region=mark)

    def isVisible(self):
        return True


@pytest.fixture()
def synced_marks(qapp):
    """两个子图共享标记区，同步源为第一个"""
    source = _FakeMark((1.0, 2.0))
    sibling = _FakeMark((9.0, 9.5))
    mw = _FakeMW([_FakeContainer(source), _FakeContainer(sibling)])
    LayoutManager(mw).sync_mark_regions(source)
    return source, sibling


def test_region_value_is_synced(synced_marks):
    _, sibling = synced_marks
    assert sibling.getRegion() == (1.0, 2.0)


def test_set_region_runs_while_signals_blocked(synced_marks):
    """阻塞必须在 setRegion 执行期间仍然生效（修复前此处为 False）"""
    _, sibling = synced_marks
    assert sibling.blocked_at_set_region == [True]


def test_signals_unblocked_after_sync(synced_marks):
    source, sibling = synced_marks
    assert source.signalsBlocked() is False
    assert sibling.signalsBlocked() is False
