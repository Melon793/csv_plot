"""C10c（D-h/D-i/D-j）组件测试：分段预设去 QSS＋内容标记、manual_check
palette＋字号、can_db 菜单钮迁 QToolButton popup。

预设 isChecked/单选语义的既有断言在 test_legend_line_preset_panel /
test_legend_draw_mode_menu / test_legend_quick_actions（零改动存活）。
"""

from __future__ import annotations

import pytest

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPalette
from PySide6.QtWidgets import (
    QPushButton,
    QStyle,
    QStyleFactory,
    QStyleOptionButton,
    QToolButton,
)

from src.core.data_types import LINE_PRESET_AUTO
from src.ui import theme
from src.ui.widgets.curve_style_panel import (
    _SEG_DRAW_ITEMS,
    _SEG_LINE_PRESET_ITEMS,
    _SEG_SYMBOL_ITEMS,
    _SegmentButton,
    _SegmentedControl,
    _draw_mode_icon,
    _line_preset_icon,
    _symbol_policy_icon,
)

ACCENT = QColor(theme.ACCENT)


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


def _accent_pixels(pm) -> int:
    img = pm.toImage()
    n = 0
    for y in range(img.height()):
        for x in range(img.width()):
            c = img.pixelColor(x, y)  # un-premultiply（pixel() 给预乘值）
            if c.alpha() > 0 and abs(c.red() - ACCENT.red()) < 40 \
                    and abs(c.green() - ACCENT.green()) < 40 \
                    and abs(c.blue() - ACCENT.blue()) < 40:
                n += 1
    return n


def _blue_ink(img) -> int:
    """渲染产物的品牌蓝墨迹：色相方向判据（b>r、b>g）。

    grab 像素是「半透明字形×壳底」的不透明合成色，RGB≠ACCENT（细线
    缩到 16px 全半透明时容差采样必漏，Win offscreen 实测）；灰壳/黑字形
    合成无蓝色相方向，故色相判据跨平台稳。
    """
    n = 0
    for y in range(img.height()):
        for x in range(img.width()):
            c = img.pixelColor(x, y)
            if c.alpha() > 0 and c.blue() - c.red() >= 25 and c.blue() - c.green() >= 10:
                n += 1
    return n


def _preset_items():
    return [
        (value, label, full, _line_preset_icon(value),
         _line_preset_icon(value, ACCENT))
        for value, label, full in _SEG_LINE_PRESET_ITEMS
    ]


def _seg(qtbot, current=LINE_PRESET_AUTO):
    seg = _SegmentedControl(_preset_items(), current, lambda v: None)
    qtbot.addWidget(seg)
    return seg


# ---------------------------------------------------------------------------
# D-h：分段预设壳归平台＋当前档内容标记
# ---------------------------------------------------------------------------


def test_segmented_control_has_no_qss(qtbot):
    seg = _seg(qtbot)
    assert not seg.styleSheet()
    for btn in seg.findChildren(QPushButton):
        assert not btn.styleSheet()
        assert btn.isCheckable() and btn.autoExclusive()


def test_checked_icon_variant_is_accent_content(qtbot):
    """当前档＝ACCENT 图标变体（QIcon On 态），常态无品牌蓝。"""
    seg = _seg(qtbot)
    for btn in seg.findChildren(QPushButton):
        icon = btn.icon()
        off = icon.pixmap(QSize(18, 18), QIcon.Mode.Normal, QIcon.State.Off)
        on = icon.pixmap(QSize(18, 18), QIcon.Mode.Normal, QIcon.State.On)
        assert _accent_pixels(off) == 0
        assert _accent_pixels(on) > 0
        assert btn.isChecked() == (btn.text() == "默认")


@pytest.mark.parametrize(
    "items,icon_fn",
    [
        (_SEG_DRAW_ITEMS, _draw_mode_icon),
        (_SEG_SYMBOL_ITEMS, _symbol_policy_icon),
        (_SEG_LINE_PRESET_ITEMS, _line_preset_icon),
    ],
)
def test_all_groups_carry_dual_state_icons(qtbot, items, icon_fn):
    tuples = [
        (v, label, full, icon_fn(v), icon_fn(v, ACCENT))
        for v, label, full in items
    ]
    seg = _SegmentedControl(tuples, items[0][0], lambda v: None)
    qtbot.addWidget(seg)
    for btn in seg.findChildren(QPushButton):
        icon = btn.icon()
        off = icon.pixmap(QSize(18, 18), QIcon.Mode.Normal, QIcon.State.Off)
        on = icon.pixmap(QSize(18, 18), QIcon.Mode.Normal, QIcon.State.On)
        assert _img_diff(off.toImage(), on.toImage()) > 0


def test_preset_shell_pixels_match_native(fusion_style, qtbot):
    """去 QSS 后段按钮＝同配置原生 QPushButton 逐像素（Fusion）。"""
    seg = _seg(qtbot)
    seg.ensurePolished()
    for btn in seg.findChildren(QPushButton):
        ref = QPushButton()
        ref.setAutoDefault(False)
        ref.setDefault(False)
        ref.setIcon(btn.icon())
        ref.setText(btn.text())
        ref.setCheckable(True)
        ref.setChecked(btn.isChecked())
        ref.ensurePolished()
        ref.resize(btn.sizeHint())
        btn.resize(btn.sizeHint())
        assert _img_diff(ref.grab().toImage(), btn.grab().toImage()) == 0


def test_checked_render_selects_on_variant(fusion_style, qtbot):
    """渲染选取回归锁（评审 r1 P1-1）：按钮自身 grab 必须经 State_On 选 On pixmap。

    壳像素对照把同一 icon＋checked 复制给参照钮，选取链路若坏两边同退
    Off 仍 diff=0（恒真）；这里直断渲染产物的品牌蓝墨迹：checked>0、rest=0。
    """
    for items, icon_fn in (
        (_SEG_DRAW_ITEMS, _draw_mode_icon),
        (_SEG_SYMBOL_ITEMS, _symbol_policy_icon),
        (_SEG_LINE_PRESET_ITEMS, _line_preset_icon),
    ):
        tuples = [
            (v, label, full, icon_fn(v), icon_fn(v, ACCENT))
            for v, label, full in items
        ]
        seg = _SegmentedControl(tuples, items[0][0], lambda v: None)
        qtbot.addWidget(seg)
        seg.ensurePolished()
        for btn in seg.findChildren(QPushButton):
            ink = _blue_ink(btn.grab().toImage())
            if btn.isChecked():
                assert ink > 0, f"{btn.text()} checked 渲染未选 On 变体"
            else:
                assert ink == 0, f"{btn.text()} rest 渲染漏出 On 变体"


# ---------------------------------------------------------------------------
# 回归锁：checked 标签不走 mac 白字分支（2026-11-23 真机缺陷）
# ---------------------------------------------------------------------------


def test_checked_segment_label_ignores_state_on(qtbot):
    """checked 段标签按常态色绘制（mac 原生会把选中文字刷白）。

    mac QMacStyle CE_PushButtonLabel：Enabled+Active+State_On → ButtonText
    强制白字；菜单弹层里 checked 壳又是白底 → 选项名整条消失。修复只接管
    checked 的标签绘制：清 On/Sunken/DefaultButton，当前档 ACCENT 图标显式
    喂入（清 On 后 style 不再自选 QIcon On pixmap）。
    """
    seg = _seg(qtbot)
    buttons = seg.findChildren(QPushButton)
    assert buttons and all(isinstance(b, _SegmentButton) for b in buttons)
    checked = next(b for b in buttons if b.isChecked())

    opt = QStyleOptionButton()
    checked.initStyleOption(opt)
    assert opt.state & QStyle.StateFlag.State_On  # 原生路径下会触发白字

    for pressed in (False, True):
        probe = QStyleOptionButton(opt)
        if pressed:
            probe.state |= QStyle.StateFlag.State_Sunken
        label = checked.label_style_option(probe)
        assert not label.state & QStyle.StateFlag.State_On
        assert not label.state & QStyle.StateFlag.State_Sunken
        assert not label.features & QStyleOptionButton.ButtonFeature.DefaultButton
        assert not label.icon.isNull()  # 当前档标记不能静默退化
        keep = probe.state & ~(
            QStyle.StateFlag.State_On | QStyle.StateFlag.State_Sunken
        )
        assert label.state == keep  # 其余状态位原样保留


def test_checked_segment_grab_smoke(fusion_style, qtbot):
    """checked 自绘路径在 rest/pressed/focused 下都不崩、不留空图。"""
    seg = _seg(qtbot)
    checked = next(b for b in seg.findChildren(QPushButton) if b.isChecked())
    checked.setFocus(Qt.FocusReason.OtherFocusReason)
    for down in (False, True):
        checked.setDown(down)
        assert not checked.grab().isNull()
    checked.setDown(False)


# ---------------------------------------------------------------------------
# D-i：manual_check palette WindowText＋QFont pixelSize 12
# ---------------------------------------------------------------------------


def test_manual_check_palette_and_font(qapp):
    from src.ui.widgets.status_drawer import XAxisDrawer

    drawer = XAxisDrawer()
    mc = drawer.manual_check
    assert not mc.styleSheet(), "chrome/文字层级归平台，不再带 QSS"
    assert mc.palette().color(QPalette.ColorRole.WindowText) == QColor(
        theme.TEXT_MUTED
    )
    assert mc.font().pixelSize() == theme.F_MUTED_PX == 12
    drawer.deleteLater()


# ---------------------------------------------------------------------------
# D-j：can_db 菜单钮迁 QToolButton popup
# ---------------------------------------------------------------------------


def test_can_db_binding_button_is_native_toolbutton(qapp, tmp_path):
    from src.ui.dialogs.can_db_selector import CanDatabaseSelectorDialog
    from tests.fixtures.data_factory import write_dbc

    dbc = write_dbc(tmp_path / "a.dbc")
    dlg = CanDatabaseSelectorDialog(initial_paths=[str(dbc)])
    btn = dlg._binding_button(0)
    assert isinstance(btn, QToolButton)
    assert not btn.styleSheet()
    assert btn.popupMode() == QToolButton.ToolButtonPopupMode.MenuButtonPopup
    assert btn.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextOnly
    assert btn.menu() is not None
    assert "▾" not in btn.text()  # 箭头交给原生 popup 指示器
    assert btn.text() == "全部库（默认）"
    dlg.deleteLater()


def test_can_db_text_area_click_opens_menu(qapp, tmp_path):
    """D-j 行为锁（评审 r1 P2-1）：文字区点击（clicked 语义）开菜单。

    Qt 6.12 公有 showMenu 是阻塞嵌套事件循环（qtoolbutton.cpp:670）；
    offscreen 无人关菜单，用 parented 安全阀在嵌套循环内 0ms hide 使其返回。
    """
    from PySide6.QtCore import QTimer

    from src.ui.dialogs.can_db_selector import CanDatabaseSelectorDialog
    from tests.fixtures.data_factory import write_dbc

    dbc = write_dbc(tmp_path / "a.dbc")
    dlg = CanDatabaseSelectorDialog(initial_paths=[str(dbc)])
    btn = dlg._binding_button(0)
    btn.show()
    qapp.processEvents()
    menu = btn.menu()
    opened = []
    menu.aboutToShow.connect(lambda: opened.append(True))
    valve = QTimer(menu)
    valve.setSingleShot(True)
    valve.timeout.connect(menu.hide)
    valve.start(0)
    btn.click()
    assert opened == [True]
    assert not menu.isVisible()
    btn.close()
    dlg.deleteLater()
