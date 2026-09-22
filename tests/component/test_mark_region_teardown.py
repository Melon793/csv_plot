"""标记区拆除（V6.0 P1-13 防回归）。

`add_mark_region` 直接覆盖 `pw.mark_region`，旧 `LinearRegionItem` 既不摘出场景
也不断开 `sigRegionChanged`；`remove_mark_region` 同样只 `removeItem`。连接挂在
发送端（item）上，只要旧 item 还被场景或引用持有，它就会继续响应拖拽并回调
`sync_mark_regions` 改写新区域 —— 一个拖得动、却再也关不掉的幽灵区域。
"""

from types import SimpleNamespace

import pytest
import pyqtgraph as pg
from shiboken6 import isValid

from tests.fixtures.waits import flush_deferred_deletes
from src.ui.widgets.mark_region_manager import MarkRegionManager


class _FakeLayoutManager:
    def __init__(self):
        self.synced = []

    def sync_mark_regions(self, region):
        self.synced.append(region)


class _FakePlotWidget:
    """真 PlotWidget（region 需要 ViewBox 宿主 + 活的 scene 才会走 removeItem 分支）"""

    def __init__(self, layout_manager):
        self.view = pg.PlotWidget()
        self.plot_item = self.view.plotItem
        self.mark_region = None
        self._window = SimpleNamespace(layout_manager=layout_manager)

    def window(self):
        return self._window


def _regions_in(plot_item):
    return [i for i in plot_item.items if isinstance(i, pg.LinearRegionItem)]


@pytest.fixture()
def mark_env(qapp):
    layout_manager = _FakeLayoutManager()
    pw = _FakePlotWidget(layout_manager)
    manager = MarkRegionManager(SimpleNamespace(pw=pw))
    yield manager, pw, layout_manager
    pw.view.deleteLater()
    flush_deferred_deletes()


def test_re_add_removes_previous_region_from_scene(mark_env):
    manager, pw, _ = mark_env
    manager.add_mark_region(0.0, 1.0)
    first = pw.mark_region
    manager.add_mark_region(2.0, 3.0)

    assert pw.mark_region is not first
    assert _regions_in(pw.plot_item) == [pw.mark_region]
    assert first.scene() is None


def test_re_add_old_region_no_longer_syncs(mark_env):
    """旧区域仍被测试引用着（C++ 未析构），断连必须发生在信号层"""
    manager, pw, layout_manager = mark_env
    manager.add_mark_region(0.0, 1.0)
    first = pw.mark_region
    manager.add_mark_region(2.0, 3.0)
    layout_manager.synced.clear()

    first.setRegion([5.0, 6.0])

    assert layout_manager.synced == []


def test_remove_disconnects_and_destroys(mark_env, qapp):
    manager, pw, layout_manager = mark_env
    manager.add_mark_region(0.0, 1.0)
    region = pw.mark_region

    manager.remove_mark_region()

    assert pw.mark_region is None
    region.setRegion([7.0, 8.0])
    assert layout_manager.synced == []
    flush_deferred_deletes()
    assert not isValid(region)


def test_remove_without_region_is_noop(mark_env):
    manager, pw, _ = mark_env
    manager.remove_mark_region()
    assert pw.mark_region is None
