"""ChromeButton：按钮 chrome 归平台的共享基类（M2，C10）。

教条（theme.py 模块 docstring）：**chrome 归平台，品牌色只作内容**。
壳委托当前 QStyle（``CE_PushButtonBevel``），hover/pressed/checked/disabled
随 windowsvista/Win 经典/mac/Fusion 自动与原生按钮一致；glyph 仍由子类
矢量自绘（``_draw_glyph`` 钩子），颜色＝内容色（ACCENT/TEXT_MUTED 按角色）。
不用 QIcon/QSS image：位图在 150% DPI 下模糊、qproperty-icon 不随状态变色
（与 plot_tabs._FlatIconButton 同一矢量理由，探针 P-N3 实测墨迹随 DPR² 增长）。

state 映射契约（探针 P-N1/N3，tmp/2026-11-23/probe-pn-results.md）：

- ``initFrom`` 不含 Sunken/On/Raised，paintEvent 手工补：
  ``State_Raised`` 恒置——QMacStyle 的 bevel 门槛位是 Raised|Sunken|On
  （qmacstyle_mac.mm:3912），不补则 mac 原生壳 rest/hover/disabled 空渲染
  （real-QPA 探针实测）；``State_Sunken=isDown()``、``State_On=isChecked()``。
- 最终 option 存 ``_last_paint_option`` 供测试 spy 断言（不依赖像素）。
- 焦点环走 ``PE_FrameFocusRect``；mac 焦点环超 rect 会被裁，消费方默认
  NoFocus，影响小（计划 §6 注记）。

例外判据（仅原生表达不了的信息通道才允许自绘 chrome，不迁本基类）：

- tab 条 ×/＋（plot_tabs._FlatIconButton）：扁平 glyph＋hover 灰圆底是
  IDE tab strip 语言，与原生 bevel/autoRaise 冲突（D-g，例外覆盖字形＋
  hover 壳整体，不只字形）；
- 「修改布局」15 格 CellButton：QSS 三态＝当前/将切同框，原生布尔态表达不了；
- _ColorSwatch：选中环＝当前色通道；_StatusSegment：状态栏内容面（palette hover）。

输入类 QSS（variable_search_bar / status_drawer 输入框）在按钮 scope 之外。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QPushButton,
    QStyle,
    QStyleOptionButton,
    QStyleOptionFocusRect,
    QToolButton,
)


def native_push_height() -> int:
    """当前 style 的原生 push button 自然高（mac Large 32 / Win 23 / Fusion ~24）。

    消费方以此定高：恰等于 mac 尺寸档默认高→圆角 aqua（qmacstyle_mac.mm:356
    /:1826 判据），且与顶栏/标题原生钮逐像素同高（F2/F3）。须在
    QApplication 存在后调用（构造期调用安全，收集期不调）。
    """
    return QPushButton().sizeHint().height()


class ChromeButton(QToolButton):
    """原生壳＋矢量内容 glyph 的按钮基类；子类实现 ``_draw_glyph``。

    几何由消费方自定（setFixedSize＋sizeHint 一致，角位 G1 口径）；
    本基类不预设尺寸。默认 NoFocus：点完后焦点留在原交互入口。
    **内容一律走 ``_draw_glyph``/``_draw_content``**：paintEvent 完全取代
    QToolButton 默认绘制，setText/setIcon/setArrowType 会被静默忽略；
    勿 setMenu（菜单可经鼠标路径打开却无可见箭头）。
    """

    _COLOR_NORMAL = "#8a8a8a"
    _COLOR_HOVER = "#1f1f1f"
    _COLOR_DISABLED = "#b5b5b5"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        # F4：mac size constrain 恒 Large 档。QMacStyle aquaSizeConstrain 无
        # mac 尺寸属性时按 ContentsType 猜档（qt_aqua_guess_size Q_UNUSED(widg)）：
        # CT_ToolButton 仅 Small 档有解→恒 Small 壳（比原生 QPushButton 矮 4
        # 逻辑 px）；CT_PushButton 三档全知→恒 Large。置 Normal 钉死 Large 与
        # 原生同壳高。mac-only 属性，Win/Fusion 忽略；Mini 属性优先级更高
        # （F1 默认分支 24+Mini 不受误伤）。
        self.setAttribute(Qt.WidgetAttribute.WA_MacNormalSize, True)
        #: 最近一次 paintEvent 的最终 QStyleOptionButton（测试 spy 钩子）
        self._last_paint_option: QStyleOptionButton | None = None

    def _base_color(self) -> str:
        """常态字形颜色，悬停/禁用时在 ``_glyph_color`` 里覆盖。"""
        return self._COLOR_NORMAL

    def _glyph_color(self) -> QColor:
        if not self.isEnabled():
            return QColor(self._COLOR_DISABLED)
        return QColor(self._COLOR_HOVER if self.underMouse() else self._base_color())

    def _draw_glyph(self, painter: QPainter, color: QColor) -> None:
        raise NotImplementedError

    def _draw_content(self, painter: QPainter) -> None:
        """内容层钩子：默认只画 glyph；胶囊等复合内容（字形＋计数）覆写。"""
        self._draw_glyph(painter, self._glyph_color())

    def _build_style_option(self) -> QStyleOptionButton:
        """组壳 option：initFrom＋手工补 Raised/Sunken/On（契约见模块 docstring）。"""
        opt = QStyleOptionButton()
        opt.initFrom(self)
        opt.state |= QStyle.StateFlag.State_Raised
        if self.isDown():
            opt.state |= QStyle.StateFlag.State_Sunken
        if self.isChecked():
            opt.state |= QStyle.StateFlag.State_On
        return opt

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        opt = self._build_style_option()
        self._last_paint_option = opt
        painter = QPainter(self)
        self.style().drawControl(
            QStyle.ControlElement.CE_PushButtonBevel, opt, painter, self
        )
        if opt.state & QStyle.StateFlag.State_HasFocus:
            focus_opt = QStyleOptionFocusRect()
            focus_opt.initFrom(self)
            focus_opt.backgroundColor = self.palette().window().color()
            self.style().drawPrimitive(
                QStyle.PrimitiveElement.PE_FrameFocusRect, focus_opt, painter, self
            )
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_content(painter)
        painter.end()

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        super().leaveEvent(event)
        self.update()
