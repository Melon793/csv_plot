"""X 锁抽屉 B 状态行 component 测试（main 收口版）。

钉：状态行在 body 顶部+hairline 分离、440 宽不变、chip 换形不用 checked
视觉、立即生效走同一广播、懒创建补种、锁态应用钮 tooltip 披露、页脚不裁、
高度增量（+41px 口径：行 28+hairline 1+spacing 2×6）。
"""

import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QStatusBar, QWidget

from src.core.config import STATUS_DRAWER_WIDTH_AXIS
from src.ui.widgets.status_drawer import XAxisDrawer

_keep_alive = []


class _FakeMw(QWidget):
    def __init__(self, locked=False):
        super().__init__()
        self.loader = object()
        self.factor, self.offset = 1.0, 0.0
        self._x_axis_locked = locked
        self.calls = []

    def set_x_axis_locked(self, locked, reason=""):
        self.calls.append((bool(locked), reason))
        self._x_axis_locked = bool(locked)
        # 真 MainWindow 由广播回调 sync_lock_ui；替身手动模拟该回路
        drawer = getattr(self, "_drawer", None)
        if drawer is not None:
            drawer.sync_lock_ui(bool(locked))

    def _axis_segment_text(self, factor, offset, always_show_correction=False):
        return "x轴：Index"


@pytest.fixture()
def drawer(qapp, qtbot):
    mw = _FakeMw()
    _keep_alive.append(mw)
    d = XAxisDrawer(mw)
    mw._drawer = d
    qtbot.addWidget(d)
    d._fake_mw = mw
    return d


def test_lock_row_on_top_with_hairline(drawer):
    body = drawer._root.itemAt(0).widget()
    lay = body.layout()
    row_item = lay.itemAt(0)
    assert row_item.layout() is not None, "第 0 项应是状态行 hbox"
    assert row_item.layout().itemAt(0).widget() is drawer.lock_chip
    assert lay.itemAt(1).widget().maximumHeight() == 1, "hairline 分离状态行与表单"
    assert drawer.width_cap == STATUS_DRAWER_WIDTH_AXIS, "宽度维持 440"


def test_chip_swaps_without_checked_visual(drawer):
    chip = drawer.lock_chip
    assert chip.text() == "锁定X轴可见范围"
    assert not chip.isCheckable(), "不用 checked 视觉（mac 白字先例）"
    drawer.sync_lock_ui(True)
    assert chip.text() == "解锁X轴可见范围"
    assert not chip.isCheckable()
    assert "已冻结" in drawer.lock_hint.text()
    assert "应用将解锁" in drawer.apply_btn.toolTip()
    drawer.sync_lock_ui(False)
    assert chip.text() == "锁定X轴可见范围"
    assert "滚轮停用" in drawer.lock_hint.text()
    assert "应用将解锁" not in drawer.apply_btn.toolTip()


def test_chip_click_broadcasts_immediately(drawer):
    drawer.lock_chip.click()
    assert drawer._fake_mw.calls == [(True, "")]
    assert drawer.lock_chip.text() == "解锁X轴可见范围"
    drawer.lock_chip.click()
    assert drawer._fake_mw.calls == [(True, ""), (False, "")]


def test_lazy_creation_seeds_locked(qapp, qtbot):
    mw = _FakeMw(locked=True)
    _keep_alive.append(mw)
    d = XAxisDrawer(mw)
    qtbot.addWidget(d)
    assert d.lock_chip.text() == "解锁X轴可见范围"
    assert "已冻结" in d.lock_hint.text()


def test_open_geometry_footer_and_height(drawer, qapp):
    mw = drawer._fake_mw
    bar = QStatusBar(mw)
    mw.statusBar = lambda: bar
    mw.resize(1200, 800)
    mw.show()
    bar.setGeometry(0, mw.height() - 30, mw.width(), 30)
    qapp.processEvents()
    assert drawer.open_for(mw) is True
    assert drawer.width() == 440
    assert drawer.lock_chip.height() == 28
    # 高度增量口径：main 基线 Fusion 212 + 41 ≈ 253
    assert 245 <= drawer.height() <= 270, drawer.height()
    hint = next(lb for lb in drawer.findChildren(QLabel) if "Esc" in lb.text())
    assert hint.width() >= hint.sizeHint().width(), "页脚提示被裁"
    assert drawer.lock_hint.width() > 0

def test_enter_on_focused_chip_toggles_not_applies(drawer, qtbot, qapp):
    """review M1：非 QDialog 里 QPushButton 不吃 Return，聚焦 chip 的回车曾
    冒泡成「应用」（误改基准+连带 D4 解锁）；现在=切换锁定。"""
    mw = drawer._fake_mw
    drawer.show()
    qapp.processEvents()
    drawer.lock_chip.setFocus()
    qapp.processEvents()
    assert drawer.focusWidget() is drawer.lock_chip
    before_factor = mw.factor

    qtbot.keyClick(drawer.lock_chip, Qt.Key.Key_Return)
    qapp.processEvents()
    assert mw.calls == [(True, "")], "聚焦 chip 回车应切换锁定"
    assert mw.factor == before_factor, "不得误应用时间修正"
    assert drawer.lock_chip.text() == "解锁X轴可见范围"

    qtbot.keyClick(drawer.lock_chip, Qt.Key.Key_Return)
    qapp.processEvents()
    assert mw.calls == [(True, ""), (False, "")]
