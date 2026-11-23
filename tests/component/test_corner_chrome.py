"""角位 ⌄/胶囊迁 ChromeButton（C10b D-f）组件测试。

覆盖：身份与几何（52×24 / 24×24，F1 Mini 档，G1 sizeHint=fixedSize）、checked 壳态
（State_On spy）、drop 高亮＝内容色（字形/计数切 ACCENT，不再整面 paint bg）、
drop 事件链零改动存活（setAcceptDrops/_set_drag_hot/_drag_hot/定向放行）、
壳像素＝同矩形原生 QPushButton（Fusion offscreen）。
内容居中左右留白恒等回归在 test_plot_tabs_widget（零改动存活）。
"""

from __future__ import annotations

import pytest

from PySide6.QtCore import QByteArray, QMimeData, QPoint, QSize, Qt
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QImage
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QStyle, QStyleFactory

from src.ui.drag_drop import SHELF_MIME_FORMAT
from src.ui.drag_visuals import DRAG_ACCENT
from src.ui.widgets.chrome_button import ChromeButton, native_push_height
from src.ui.widgets.plot_tabs import PlotTabsWidget

S = QStyle.StateFlag

# 模块级保活：offscreen 下 dropEvent 期间 mime 被回收会段错误（既有教训）
_MIME_TEXT = QMimeData()
_MIME_TEXT.setText("rpm")
_MIME_SHELF = QMimeData()
_MIME_SHELF.setText("rpm")
_MIME_SHELF.setData(SHELF_MIME_FORMAT, QByteArray(b"shelf"))


@pytest.fixture()
def tabs(qapp, qtbot):
    widget = PlotTabsWidget()
    qtbot.addWidget(widget)
    widget.resize(640, 360)
    widget.show()
    widget.set_new_page_enabled(True)
    widget.addTab(QLabel("page"), "page 1")
    qapp.processEvents()
    yield widget
    widget.close()


@pytest.fixture()
def fusion_style(qapp):
    prev = qapp.style().objectName()
    qapp.setStyle(QStyleFactory.create("Fusion"))
    yield qapp
    qapp.setStyle(QStyleFactory.create(prev or "Fusion"))


def _img_diff(a: QImage, b: QImage) -> int:
    assert a.size() == b.size()
    return sum(
        1 for y in range(a.height()) for x in range(a.width())
        if a.pixel(x, y) != b.pixel(x, y)
    )


# ---------------------------------------------------------------------------
# 身份与几何（G1：sizeHint=fixedSize，角位只按 sizeHint 留位）
# ---------------------------------------------------------------------------


def test_corner_buttons_are_chrome_buttons(tabs):
    assert isinstance(tabs._dir_btn, ChromeButton)
    assert isinstance(tabs.shelf_capsule, ChromeButton)
    assert not tabs._dir_btn.styleSheet()
    assert not tabs.shelf_capsule.styleSheet()


def test_geometry_contracts_survive(tabs):
    cap = tabs.shelf_capsule
    assert cap.sizeHint() == cap.minimumSizeHint()
    assert cap.sizeHint().width() == cap.width() == 52
    assert cap.sizeHint().height() == cap.height() == native_push_height()  # F3 自然高
    assert cap.acceptDrops() is True
    d = tabs._dir_btn
    assert d.sizeHint() == d.minimumSizeHint()
    assert d.sizeHint() == QSize(24, native_push_height())  # G1 数值实断言（F3）
    assert d.width() == 24 and d.height() == native_push_height()
    # F3：自然高＝mac Large 档默认高→圆角无需尺寸属性（置 Mini 反落方壳）
    assert not cap.testAttribute(Qt.WidgetAttribute.WA_MacMiniSize)
    assert not d.testAttribute(Qt.WidgetAttribute.WA_MacMiniSize)


def test_natural_tier_and_active_state_spy(tabs, qapp):
    """F3 判据输入锁：高＝平台自然 push 高（mac Large 32＝档默认高→圆角；
    qmacstyle_mac.mm:356 pushButtonDefaultHeight{32,28,24}；:1826 不等→方）。
    激活态 State_Active 进 option（setActiveWindow 强制 Qt 激活态）；
    圆/方壳像素腿归真机抓屏（c10_mac_capture 激活态口径）。
    """
    for w in (tabs.shelf_capsule, tabs._dir_btn):
        assert w.height() == native_push_height()
        assert not w.testAttribute(Qt.WidgetAttribute.WA_MacMiniSize)
    QApplication.setActiveWindow(tabs)
    qapp.processEvents()
    tabs.shelf_capsule.grab()
    assert tabs.shelf_capsule._last_paint_option.state & S.State_Active


# ---------------------------------------------------------------------------
# checked 壳态（State_On spy）
# ---------------------------------------------------------------------------


def test_capsule_checked_maps_state_on(tabs, qapp):
    cap = tabs.shelf_capsule
    cap.grab()
    assert not (cap._last_paint_option.state & S.State_On)
    tabs.set_shelf_active(True)
    cap.grab()
    assert cap._last_paint_option.state & S.State_On
    assert cap._glyph_color() == QColor(DRAG_ACCENT)
    tabs.set_shelf_active(False)
    cap.grab()
    assert not (cap._last_paint_option.state & S.State_On)


def test_dir_button_rest_has_no_on(tabs):
    d = tabs._dir_btn
    assert not d.isCheckable()  # ⌄ 是动作钮非开关：State_On 无从置位（P2-4）
    d.grab()
    assert not (d._last_paint_option.state & S.State_On)


# ---------------------------------------------------------------------------
# drop 高亮＝内容色（不再整面 paint bg）
# ---------------------------------------------------------------------------


def test_drag_hot_switches_content_color(tabs, qtbot):
    cap = tabs.shelf_capsule
    hovered = []
    cap.drag_hovered.connect(hovered.append)
    normal = cap._glyph_color()
    cap._set_drag_hot(True)
    assert cap._drag_hot is True
    assert cap._glyph_color() == QColor(DRAG_ACCENT)
    cap.grab()  # hot 态绘制不崩，计数同色
    cap._set_drag_hot(False)
    assert cap._glyph_color() == normal
    assert hovered == [True, False]


def test_disabled_glyph_color(tabs):
    cap = tabs.shelf_capsule
    tabs.set_shelf_enabled(False)
    assert cap._glyph_color() == QColor(ChromeButton._COLOR_DISABLED)
    cap.grab()


# ---------------------------------------------------------------------------
# drop 事件链零改动存活
# ---------------------------------------------------------------------------


def test_drag_enter_accepts_text_rejects_shelf_mime(tabs, qapp):
    cap = tabs.shelf_capsule
    ev = QDragEnterEvent(
        QPoint(10, 10), Qt.DropAction.CopyAction, _MIME_TEXT,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(cap, ev)
    assert ev.isAccepted()
    assert cap._drag_hot is True
    cap._set_drag_hot(False)
    ev2 = QDragEnterEvent(
        QPoint(10, 10), Qt.DropAction.CopyAction, _MIME_SHELF,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(cap, ev2)
    assert not ev2.isAccepted()
    assert cap._drag_hot is False


def test_drop_forwards_event_and_resets_hot(tabs, qapp):
    cap = tabs.shelf_capsule
    received = []
    cap.drop_requested.connect(received.append)
    cap._set_drag_hot(True)
    ev = QDropEvent(
        QPoint(10, 10), Qt.DropAction.CopyAction, _MIME_TEXT,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    # 合成 drop 直调 dropEvent（既有口径：QApplication.notify 拦截合成
    # QDropEvent，sendEvent 投不进去；见 test_shelf_capsule._drop）
    cap.dropEvent(ev)
    assert received and received[0] is ev
    assert cap._drag_hot is False  # 兜底复位（G6）


# ---------------------------------------------------------------------------
# 壳像素＝同矩形原生 QPushButton（Fusion offscreen）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["rest", "checked"])
def test_capsule_shell_matches_native(fusion_style, tabs, mode):
    cap = tabs.shelf_capsule
    cap._draw_content = lambda painter: None  # 只对照壳
    ref = QPushButton()
    ref.setAutoDefault(False)
    ref.setDefault(False)
    ref.setFixedSize(cap.size())  # F3：跟消费方实际几何（自然高）
    for b in (cap, ref):
        b.ensurePolished()
        if mode == "checked":
            b.setCheckable(True)
            b.setChecked(True)
    assert _img_diff(ref.grab().toImage(), cap.grab().toImage()) == 0


def test_dir_shell_matches_native(fusion_style, tabs):
    d = tabs._dir_btn
    d._draw_content = lambda painter: None
    ref = QPushButton()
    ref.setAutoDefault(False)
    ref.setDefault(False)
    ref.setFixedSize(d.size())  # F3：跟消费方实际几何（自然高）
    for b in (d, ref):
        b.ensurePolished()
    assert _img_diff(ref.grab().toImage(), d.grab().toImage()) == 0


# ---------------------------------------------------------------------------
# 计数/各态绘制不崩（G1 裁切回归的绘制面）
# ---------------------------------------------------------------------------


def test_count_and_state_paint_smoke(tabs):
    for count in (0, 3, 100):
        tabs.set_shelf_count(count)
        tabs.shelf_capsule.grab()
    tabs.set_shelf_active(True)
    tabs.shelf_capsule.grab()
    tabs.set_shelf_enabled(False)
    tabs.shelf_capsule.grab()
    tabs._dir_btn.grab()
