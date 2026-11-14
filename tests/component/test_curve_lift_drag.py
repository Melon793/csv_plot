"""曲线长按抬框搬运（组件层）：状态机、副框、QDrag 发起与清脏。

offscreen 下 ``QDrag.exec`` 立即返回，无法用真实拖拽验收，这里统一
monkeypatch ``plot_widget.QDrag`` 为同步记录的替身；真实 300ms 计时器与
跨 plot drop 走 ``tests/e2e/test_curve_lift_drag.py``。
"""

import pytest

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from src.core.config import CURVE_LIFT_LONG_PRESS_MS
from src.ui import drag_drop
from src.ui.widgets import plot_widget as plot_widget_module


@pytest.fixture()
def three_curves(plot_factory):
    pw = plot_factory()
    # 长按到期有 isVisible 兜底；component 宿主默认不显示，这里显式 show
    pw.window().show()
    pw.show()
    for name in ("a", "b", "c"):
        assert pw.plot_variable(name)
    yield pw
    pw.hide()


def _curve_pos(pw, name, frac=0.5):
    ci = pw.curves[name]
    i = min(len(ci.x_data) - 1, int(len(ci.x_data) * frac))
    scene = pw.view_box.mapViewToScene(
        QPointF(float(ci.x_data[i]), float(ci.y_data[i]))
    )
    return pw.mapFromScene(scene)


def _press(pw, pos, mods=Qt.KeyboardModifier.NoModifier):
    pw.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(pos),
            QPointF(pos),
            QPointF(0.0, 0.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            mods,
        )
    )


def _move(pw, pos, buttons=Qt.MouseButton.LeftButton,
          mods=Qt.KeyboardModifier.NoModifier):
    pw.mouseMoveEvent(
        QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(pos),
            QPointF(pos),
            QPointF(0.0, 0.0),
            Qt.MouseButton.NoButton,
            buttons,
            mods,
        )
    )


def _release(pw, pos, mods=Qt.KeyboardModifier.NoModifier):
    pw.mouseReleaseEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            QPointF(pos),
            QPointF(pos),
            QPointF(0.0, 0.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            mods,
        )
    )


class _FakeDrag:
    """同步替身：记录 MIME/注册表快照，不进入嵌套事件循环。"""

    instances: list["_FakeDrag"] = []

    def __init__(self, source):
        self.source = source
        self.mime = None
        self.exec_calls = 0
        self.pixmap = None
        self.hotspot = None
        self.snapshot = None
        _FakeDrag.instances.append(self)

    def setMimeData(self, mime):
        self.mime = mime

    def setPixmap(self, pixmap):
        self.pixmap = pixmap

    def setHotSpot(self, hotspot):
        self.hotspot = hotspot

    def exec(self, actions):
        self.exec_calls += 1
        # 在 exec 期间取注册表快照：返回后 _start_curve_drag 会清空注册表
        self.snapshot = {
            "active": drag_drop.is_legend_drag_active(),
            "origin": drag_drop.active_legend_drag_origin(),
            "batch": drag_drop.is_active_legend_batch_drag(),
            "source": drag_drop.get_active_legend_drag_source(self.mime),
            "vars": list(drag_drop._active_legend_drag.get("vars") or []),
        }
        return Qt.DropAction.IgnoreAction


@pytest.fixture()
def fake_drag(monkeypatch):
    _FakeDrag.instances = []
    monkeypatch.setattr(plot_widget_module, "QDrag", _FakeDrag)
    return _FakeDrag


def _lift(pw, pos):
    """press + 直接触发计时器回调（等价 300ms 到期）。"""
    _press(pw, pos)
    assert pw._lift_timer.isActive()
    pw._on_lift_timeout()


def test_press_starts_timer_without_hit_test(three_curves):
    pw = three_curves
    pos = _curve_pos(pw, "a")

    _press(pw, pos)

    assert pw._lift_timer.isActive()
    assert pw._lift_timer.interval() == CURVE_LIFT_LONG_PRESS_MS
    assert pw._lift_active is False
    assert pw._lift_names == []
    assert pw._curve_click_press_pos is not None
    pw._cancel_lift()


def test_ctrl_press_starts_lift_timer(three_curves):
    """macOS ⌘/Ctrl 先按再拖是习惯路径，长按不得被 Ctrl 排除。"""
    pw = three_curves
    _press(pw, _curve_pos(pw, "a"), Qt.KeyboardModifier.ControlModifier)
    assert pw._lift_timer.isActive()
    pw._cancel_lift()


def test_shift_press_does_not_start_lift(three_curves):
    """Shift 是框选缩放专属，不参与长按搬运。"""
    pw = three_curves
    _press(pw, _curve_pos(pw, "a"), Qt.KeyboardModifier.ShiftModifier)
    assert not pw._lift_timer.isActive()
    assert pw._lift_press_pos is None
    pw.rubberBand.hide()


def test_timeout_hit_shows_card_without_scene_item(three_curves):
    pw = three_curves
    pw.legend_label._toggle_legend_selection("a")
    items_before = len(pw.scene().items())

    _lift(pw, _curve_pos(pw, "a"))

    assert pw._lift_active is True
    assert pw._lift_names == ["a"]
    assert pw._lift_card is not None
    assert pw._lift_card.isVisibleTo(pw.viewport())
    assert len(pw.scene().items()) == items_before, "副框不得进入 scene"
    assert pw._legend_selection == {"a"}, "抬起不改变选择"
    pw._cancel_lift(restore_cursor=True)


def test_timeout_miss_cancels_silently(three_curves):
    pw = three_curves
    # 两条曲线之外的空白点
    pa = _curve_pos(pw, "a")
    pb = _curve_pos(pw, "b")
    blank = QPoint(pa.x(), (pa.y() + pb.y()) // 2)

    _press(pw, blank)
    pw._on_lift_timeout()

    assert pw._lift_active is False
    assert pw._lift_card is None or not pw._lift_card.isVisible()


def test_unselected_curve_does_not_lift(three_curves):
    """未选中曲线不参与长按：与慢速拖动平移的按下-静止阶段无法区分。"""
    pw = three_curves
    pw.legend_label._toggle_legend_selection("b")

    _press(pw, _curve_pos(pw, "a"))
    pw._on_lift_timeout()

    assert pw._lift_active is False
    assert pw._lift_names == []
    assert pw._lift_card is None or not pw._lift_card.isVisible()


def test_selected_member_lifts_whole_batch(three_curves):
    pw = three_curves
    pw.legend_label._toggle_legend_selection("a")
    pw.legend_label._toggle_legend_selection("b")

    _lift(pw, _curve_pos(pw, "a"))

    assert set(pw._lift_names) == {"a", "b"}
    pw._cancel_lift(restore_cursor=True)


def test_move_before_timeout_cancels_lift(three_curves):
    pw = three_curves
    start = _curve_pos(pw, "a")
    _press(pw, start)

    _move(pw, start + QPoint(0, QApplication.startDragDistance() + 4))

    assert pw._lift_active is False
    assert not pw._lift_timer.isActive()
    assert pw._lift_press_pos is None


def test_release_after_lift_does_nothing(three_curves, fake_drag):
    """抬框后原地松手：不搬运，裸单击语义（不选中/不清选）。"""
    pw = three_curves
    pw.legend_label._toggle_legend_selection("a")
    pos = _curve_pos(pw, "a")
    _lift(pw, pos)

    _release(pw, pos)

    assert fake_drag.instances == []
    assert pw._legend_selection == {"a"}
    assert pw._lift_active is False


def test_ctrl_slow_click_after_lift_still_toggles(three_curves, fake_drag):
    """⌘+长按后未拖动就松手：按点击收口，慢点击仍能 toggle（不丢手势）。"""
    pw = three_curves
    pw.legend_label._toggle_legend_selection("a")
    pos = _curve_pos(pw, "a")
    _press(pw, pos, Qt.KeyboardModifier.ControlModifier)
    pw._on_lift_timeout()
    assert pw._lift_active is True

    _release(pw, pos, Qt.KeyboardModifier.ControlModifier)

    assert fake_drag.instances == []
    assert pw._legend_selection == set(), "慢点击应取消选中而不是静默丢弃"


def test_move_after_lift_starts_drag_with_curve_origin(three_curves, fake_drag):
    pw = three_curves
    pw.legend_label._toggle_legend_selection("a")
    pos = _curve_pos(pw, "a")
    _lift(pw, pos)

    _move(pw, pos + QPoint(QApplication.startDragDistance() + 8, 0))

    assert len(fake_drag.instances) == 1
    drag = fake_drag.instances[0]
    assert drag.exec_calls == 1
    assert drag.mime is not None
    assert drag.snapshot["active"] is True
    assert drag.snapshot["origin"] == "curve"
    assert drag.snapshot["batch"] is False, "单条曲线也走 quiet 批量协议"
    assert drag.snapshot["source"] is pw
    assert drag.snapshot["vars"] == ["a"]
    assert pw._legend_selection == {"a"}, "发起搬运不得改变选择"


def test_batch_drag_marks_batch(three_curves, fake_drag):
    pw = three_curves
    pw.legend_label._toggle_legend_selection("a")
    pw.legend_label._toggle_legend_selection("b")
    pos = _curve_pos(pw, "a")
    _lift(pw, pos)

    _move(pw, pos + QPoint(QApplication.startDragDistance() + 8, 0))

    drag = fake_drag.instances[0]
    assert set(drag.snapshot["vars"]) == {"a", "b"}
    assert drag.snapshot["batch"] is True


def test_drag_finish_clears_registry_and_scene_state(three_curves, fake_drag):
    pw = three_curves
    scene = pw.scene()
    pw.legend_label._toggle_legend_selection("a")

    pos = _curve_pos(pw, "a")
    _lift(pw, pos)
    # 抬起后才伪造残留：press 期间让 pyqtgraph 自己维护 scene 状态
    scene.dragButtons = [Qt.MouseButton.LeftButton]
    scene.clickEvents = ["stale"]
    scene.lastDrag = "stale"
    scene.dragItem = "stale"
    _move(pw, pos + QPoint(QApplication.startDragDistance() + 8, 0))

    assert drag_drop.is_legend_drag_active() is False
    assert scene.dragButtons == []
    assert scene.clickEvents == []
    assert scene.lastDrag is None
    assert scene.dragItem is None


def test_suppress_click_after_drag_does_not_select(three_curves, fake_drag):
    pw = three_curves
    pw.legend_label._toggle_legend_selection("a")
    pos = _curve_pos(pw, "a")
    _lift(pw, pos)
    _move(pw, pos + QPoint(QApplication.startDragDistance() + 8, 0))

    # QDrag 消费 release 的兜底：释放同位置不得当成 Ctrl 单击
    _release(pw, pos, Qt.KeyboardModifier.ControlModifier)
    assert pw._legend_selection == {"a"}, "释放补刀不得改动选择"

    # 下一次正常 Ctrl 单击必须恢复
    _press(pw, pos, Qt.KeyboardModifier.ControlModifier)
    _release(pw, pos, Qt.KeyboardModifier.ControlModifier)
    assert pw._legend_selection == set()


def test_lost_release_cancels_lift(three_curves):
    pw = three_curves
    pw.legend_label._toggle_legend_selection("a")
    pos = _curve_pos(pw, "a")
    _lift(pw, pos)

    _move(pw, pos + QPoint(3, 0), buttons=Qt.MouseButton.NoButton)

    assert pw._lift_active is False
    assert pw._lift_names == []


def test_hide_cancels_lift_but_keeps_selection(three_curves):
    pw = three_curves
    pw.legend_label._toggle_legend_selection("a")
    pos = _curve_pos(pw, "a")
    _lift(pw, pos)
    assert pw._lift_active is True

    pw.hide()

    assert pw._lift_active is False
    assert pw._legend_selection == {"a"}, "切页隐藏不得清选择"
    pw.show()
