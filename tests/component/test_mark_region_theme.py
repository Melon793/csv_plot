"""标记区域填充走 theme tint（库默认 (0,0,255,50) 纯蓝且源码不可见）。"""

from src.ui import theme
from src.ui.widgets.mark_region_manager import MarkRegionManager


class _FakePlotItem:
    def __init__(self):
        self.items = []

    def addItem(self, item):
        self.items.append(item)

    def removeItem(self, item):
        if item in self.items:
            self.items.remove(item)


class _FakePlotWidget:
    def __init__(self):
        self.mark_region = None
        self.plot_item = _FakePlotItem()


class _FakeCursorManager:
    def __init__(self, pw):
        self.pw = pw


def test_mark_region_uses_theme_tint(qapp):
    pw = _FakePlotWidget()
    manager = MarkRegionManager(_FakeCursorManager(pw))
    manager.add_mark_region(1.0, 2.0)
    try:
        color = pw.mark_region.brush.color()
        assert color.alpha() == theme.TINT_ALPHA == 64
        assert color.name().upper() == theme.ACCENT.upper()
    finally:
        manager.remove_mark_region()
