"""ViewBox 交互事件构造器（component 层共享）。

``test_viewbox_zoom.py`` 里有一份等价实现。抽到这里是为了让标注的
"零干扰"对照测试不用再抄第三份 —— 既有文件保持原样，避免无关 diff 触碰
已被钉住的断言。

事件一律基于控件自身坐标构造（用 ``widget.mapToGlobal`` 换算），
不依赖绝对屏幕坐标，offscreen 下也可复现。
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent


def wheel_event(widget, local_pos: QPoint, delta_y: int,
                modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier) -> QWheelEvent:
    """构造一个落在 ``widget`` 指定局部坐标的滚轮事件（每齿 120）"""
    pos = QPointF(local_pos)
    global_pos = QPointF(widget.mapToGlobal(local_pos))
    return QWheelEvent(
        pos,
        global_pos,
        QPoint(0, 0),             # pixelDelta
        QPoint(0, delta_y),       # angleDelta（>0 向前=放大）
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def mouse_event(event_type, widget, local_pos: QPoint, button,
                buttons, modifiers) -> QMouseEvent:
    """构造一个落在 ``widget`` 指定局部坐标的鼠标事件"""
    pos = QPointF(local_pos)
    global_pos = QPointF(widget.mapToGlobal(local_pos))
    return QMouseEvent(event_type, pos, global_pos, button, buttons, modifiers)
