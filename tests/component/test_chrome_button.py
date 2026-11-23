"""ChromeButton 共享基类（C10a，M2）组件测试。

覆盖：state 映射契约（spy 最终 option，含 mac bevel 门槛位 State_Raised）、
壳像素＝同矩形原生 QPushButton（offscreen 只认 Fusion，计划 v1.1 §2 口径；
mac/vista 原生壳归真机抓屏）、glyph 钩子与禁用色语义、rail » 芯片迁移、
D-g 例外锁（×/＋ 保持 _FlatIconButton 扁平，不迁基类）。
"""

from __future__ import annotations

import pytest

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QPushButton, QStyle, QStyleFactory

from src.ui import theme
from src.ui.widgets.chrome_button import ChromeButton
from src.ui.widgets.plot_tabs import _CloseTabButton, _FlatIconButton, _NewPageButton
from src.ui.widgets.var_pane_rail import _RailExpandChip

S = QStyle.StateFlag


class _ShellOnly(ChromeButton):
    """只画壳不画 glyph：壳像素对照用。"""

    def _draw_glyph(self, painter, color):
        pass


class _GlyphSpy(ChromeButton):
    """记录 glyph 钩子收到的颜色。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.glyph_colors = []

    def _draw_glyph(self, painter, color):
        self.glyph_colors.append(QColor(color))


@pytest.fixture()
def fusion_style(qapp):
    """offscreen 像素对照只认 Fusion（mac/Win offscreen 默认均 Fusion，实测）。"""
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


def _apply_mode(btn, mode):
    if mode in ("down", "checked"):
        btn.setCheckable(True)
    if mode == "hover":
        btn.setAttribute(Qt.WidgetAttribute.WA_UnderMouse, True)
    elif mode == "down":
        btn.setDown(True)
        btn.setChecked(True)
    elif mode == "checked":
        btn.setChecked(True)
    elif mode == "disabled":
        btn.setEnabled(False)


# ---------------------------------------------------------------------------
# state 映射契约（spy 最终 option）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,expect",
    [
        ("rest", {S.State_Enabled, S.State_Raised}),
        ("hover", {S.State_Enabled, S.State_Raised, S.State_MouseOver}),
        ("down", {S.State_Enabled, S.State_Raised, S.State_Sunken, S.State_On}),
        ("checked", {S.State_Enabled, S.State_Raised, S.State_On}),
        ("disabled", {S.State_Raised}),
    ],
)
def test_state_mapping_spy(qapp, mode, expect):
    """initFrom 不含 Sunken/On/Raised，paintEvent 手工补（P-N3 契约）。

    State_Raised 是 QMacStyle bevel 门槛位（qmacstyle_mac.mm:3912），
    不补则 mac 原生壳 rest/hover/disabled 空渲染（real-QPA 探针实测）。
    """
    btn = _ShellOnly()
    btn.setFixedSize(52, 24)
    btn.ensurePolished()
    _apply_mode(btn, mode)
    btn.grab()
    opt = btn._last_paint_option
    assert opt is not None
    bits = (
        S.State_Enabled, S.State_Raised, S.State_MouseOver,
        S.State_Sunken, S.State_On, S.State_HasFocus,
    )
    assert {f for f in bits if opt.state & f} == expect


def test_focus_state_spy(qapp, qtbot):
    btn = _ShellOnly()
    btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    qtbot.addWidget(btn)
    btn.show()
    btn.setFocus()
    QApplication.processEvents()
    btn.grab()
    assert btn._last_paint_option.state & S.State_HasFocus


def test_default_no_focus(qapp):
    """消费方默认 NoFocus：点完焦点留在原交互入口。"""
    assert _ShellOnly().focusPolicy() == Qt.FocusPolicy.NoFocus


# ---------------------------------------------------------------------------
# 壳像素＝同矩形原生 QPushButton（Fusion offscreen）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(24, 24), (52, 24)])  # F1 消费方几何
@pytest.mark.parametrize("mode", ["rest", "hover", "down", "checked", "disabled"])
def test_shell_pixels_match_native_pushbutton(fusion_style, size, mode):
    ref = QPushButton()
    ref.setAutoDefault(False)
    ref.setDefault(False)
    ref.setFixedSize(*size)
    btn = _ShellOnly()
    btn.setFixedSize(*size)
    for b in (ref, btn):
        b.ensurePolished()
        _apply_mode(b, mode)
    assert _img_diff(ref.grab().toImage(), btn.grab().toImage()) == 0


def test_rail_chip_shell_matches_native(fusion_style, qapp):
    """rail » 芯片壳＝24×24 原生 QPushButton 壳（glyph 置空后逐像素）。"""
    chip = _RailExpandChip()
    chip.setFixedSize(24, 24)
    chip.ensurePolished()
    chip._draw_glyph = lambda painter, color: None  # 只对照壳
    ref = QPushButton()
    ref.setAutoDefault(False)
    ref.setDefault(False)
    ref.setFixedSize(24, 24)
    ref.ensurePolished()
    assert _img_diff(ref.grab().toImage(), chip.grab().toImage()) == 0


# ---------------------------------------------------------------------------
# glyph 钩子与颜色语义
# ---------------------------------------------------------------------------


def test_glyph_hook_receives_state_colors(qapp):
    btn = _GlyphSpy()
    btn.setFixedSize(24, 24)
    btn.ensurePolished()
    btn.grab()
    assert btn.glyph_colors[-1] == QColor(ChromeButton._COLOR_NORMAL)
    btn.setAttribute(Qt.WidgetAttribute.WA_UnderMouse, True)
    btn.grab()
    assert btn.glyph_colors[-1] == QColor(ChromeButton._COLOR_HOVER)
    btn.setAttribute(Qt.WidgetAttribute.WA_UnderMouse, False)
    btn.setEnabled(False)
    btn.grab()
    assert btn.glyph_colors[-1] == QColor(ChromeButton._COLOR_DISABLED)


def test_rail_chip_glyph_is_accent_content(qapp):
    """D-e：品牌蓝只作内容——glyph 恒 ACCENT（hover 不变色，状态交给壳）。"""
    chip = _RailExpandChip()
    chip.setFixedSize(24, 24)
    chip.ensurePolished()
    assert chip._glyph_color() == QColor(theme.ACCENT)
    chip.setAttribute(Qt.WidgetAttribute.WA_UnderMouse, True)
    assert chip._glyph_color() == QColor(theme.ACCENT)
    chip.setEnabled(False)
    assert chip._glyph_color() == QColor(theme.TEXT_MUTED)
    chip.grab()  # 禁用态绘制不崩
    chip.setEnabled(True)
    chip.grab()


# ---------------------------------------------------------------------------
# D-g 例外锁：×/＋ 保持扁平自绘，不迁基类
# ---------------------------------------------------------------------------


def test_tab_strip_buttons_stay_flat(qapp):
    for cls in (_CloseTabButton, _NewPageButton):
        assert issubclass(cls, _FlatIconButton)
        assert not issubclass(cls, ChromeButton)


# ---------------------------------------------------------------------------
# rail 芯片迁移（C8b 旧 QSS 断言的新语义，计划 §5.1 披露）
# ---------------------------------------------------------------------------


def test_rail_chip_is_chrome_button(qapp):
    from src.ui.widgets.var_pane_rail import VarPaneRail

    rail = VarPaneRail()
    chip = rail.expand_btn
    assert isinstance(chip, ChromeButton)
    assert not chip.styleSheet(), "chrome 归平台，芯片不再带 QSS"
    assert chip.size() == QSize(24, 24)  # F1 Mini 档
    assert chip.testAttribute(Qt.WidgetAttribute.WA_MacMiniSize)
    assert chip.toolTip() == "单击展开变量区 (Ctrl+B)"
    rail.deleteLater()
