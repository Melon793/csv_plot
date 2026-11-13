"""曲线选中联动的手势/跨区回归（P3，真实 MainWindow，offscreen）。

把评审第 1 轮 §六 item 5 的探针结论固化为回归护栏：

- legend 真实变高后，press 落图例 / release 进绘图区 → 不触发选择
- 窗口 resize 后跨区判定仍按新几何生效，且正常点击不被误拦
- 真实双击序列（单击 → dblclick → 尾随 release）不改选择、不留陈旧记录
- 滚轮缩放不改变选择集
- 点击已隐藏曲线的位置 → 视为空白 → 裸单击清选

两处实测前提（探针 tmp/2026-11-14/probe_dclick_sequence.py 等）：

1. ``QTest.mouseDClick`` **只投递 MouseButtonDblClick**（实测序列：
   dblclick + press(type=DblClick)，没有前置 press/release）。真实双击是
   press→release→dblclick→release，故本文件自行补首次单击与尾随 release，
   否则选择手势根本没被触发过，断言会空转。
2. 隐藏曲线若走 ``_apply_visibility``，Y limits 随可见集合收缩，pyqtgraph
   会把视野钳回剩余曲线范围（multi_curve_manager.py 的 ``_update_axes_for_multi_curve``
   文档明示为预期行为，实测视野被钳到 [555, 1535]），被隐藏曲线的位置随之
   落到 vb 外 → 被区域校验拦下，测不到"视为空白"这一层。故可见性用
   ``ci.visible = False`` 钉住（与 component test_curve_selection 同口径），
   视野保持原样；本用例的被测对象是命中→清选→播报链路，不是显隐入口。

几何一律取真实 sceneBoundingRect，点击经 viewport 投递，口径与
test_curve_selection_flow 一致。
"""

import pytest

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from src.core.config import HIGHLIGHT_LINE_WIDTH, THICK_LINE_WIDTH
from src.core.data_types import SYMBOL_POLICY_OFF
from src.ui.widgets.multi_curve_manager import hit_test_curve
from tests.e2e.test_curve_selection_flow import (
    _blank_pos,
    _click_plot,
    _curve_pos,
    _pw,
    _width,
)
from tests.e2e.test_curve_selection_flow import (  # noqa: F401
    plotted_window as _flow_plotted_window,
)
from tests.fixtures.waits import settle


@pytest.fixture(autouse=True)
def _guard_app_exit(monkeypatch):
    """防毒化：任何路径若调用 QApplication.exit，同进程后续用例会被 quitNow 拖死。"""
    monkeypatch.setattr(QApplication, "exit", staticmethod(lambda *a, **k: None))


@pytest.fixture()
def plotted_window(request):
    """转发 flow 文件的绘图夹具（几何等待 + symbol 策略钉死）。"""
    return request.getfixturevalue("_flow_plotted_window")


def _mouse(kind, pos, button=Qt.MouseButton.LeftButton,
           mods=Qt.KeyboardModifier.NoModifier):
    p = QPointF(pos)
    return QMouseEvent(kind, p, p, QPointF(0.0, 0.0), button, button, mods)


def _press(pw, pos, **kw):
    pw.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, pos, **kw))


def _release(pw, pos, **kw):
    pw.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, pos, **kw))


def _cross_region_pair(pw):
    """图例底缘内 2px → 底缘外 4px（位移 < startDragDistance）。

    返回 (press_vp, release_vp)；前提不成立时直接 fail，避免空转断言。
    """
    proxy = pw.legend_label.graphicsProxyWidget()
    assert proxy is not None, "真实 legend 必须有 graphicsProxyWidget"
    lg = proxy.sceneBoundingRect()
    vb = pw.view_box.sceneBoundingRect()
    x = vb.center().x()
    press_scene = QPointF(x, lg.bottom() - 2)
    release_scene = QPointF(x, lg.bottom() + 4)
    assert lg.contains(press_scene), "press 须真的落在图例内"
    assert vb.contains(release_scene), "release 须真的落在绘图区内"
    p_vp, r_vp = pw.mapFromScene(press_scene), pw.mapFromScene(release_scene)
    assert (r_vp - p_vp).manhattanLength() < QApplication.startDragDistance()
    return p_vp, r_vp


def _wheel(pw, pos, delta=120):
    """向 viewport 投递真实滚轮事件（pyqtgraph 缩放路径）。"""
    p = QPointF(pos)
    evt = QWheelEvent(
        p,
        pw.viewport().mapToGlobal(pos),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(pw.viewport(), evt)
    QApplication.processEvents()


# ---------------------------------------------------------------------------
# 跨区：legend 变高 / resize
# ---------------------------------------------------------------------------


def test_legend_grown_taller_cross_region_not_triggered(plotted_window, qapp, qtbot):
    """三条曲线把 legend 撑高后，图例→绘图区的跨界手势仍不触发选择。"""
    mw = plotted_window
    pw = _pw(mw)
    assert pw.plot_variable("flag")
    pw.curves["flag"].symbol_policy = SYMBOL_POLICY_OFF
    pw._refresh_selection_projections()
    qapp.processEvents()
    qtbot.waitUntil(
        lambda: pw.view_box.sceneBoundingRect().height() > 100, timeout=5000
    )
    settle()

    _click_plot(pw, _curve_pos(pw, "rpm"))
    assert pw._legend_selection == {"rpm"}
    p_vp, r_vp = _cross_region_pair(pw)

    _press(pw, p_vp)
    assert pw._curve_click_press_pos is None, "图例上的 press 不得记录"
    _release(pw, r_vp)
    qapp.processEvents()

    assert pw._legend_selection == {"rpm"}
    assert _width(pw, "rpm") == HIGHLIGHT_LINE_WIDTH


def test_cross_region_after_resize_uses_new_geometry(plotted_window, qapp):
    """resize 后跨区判定按新几何生效，且正常点击不被误拦（无陈旧几何）。"""
    mw = plotted_window
    pw = _pw(mw)
    _click_plot(pw, _curve_pos(pw, "rpm"))
    assert pw._legend_selection == {"rpm"}

    mw.resize(900, 620)
    qapp.processEvents()
    settle()

    p_vp, r_vp = _cross_region_pair(pw)
    _press(pw, p_vp)
    assert pw._curve_click_press_pos is None
    _release(pw, r_vp)
    qapp.processEvents()
    assert pw._legend_selection == {"rpm"}

    # 新几何下正常单击仍须命中：证明区域校验没有把整块绘图区误判成图例
    _click_plot(pw, _curve_pos(pw, "speed"))
    assert pw._legend_selection == {"speed"}
    assert _width(pw, "speed") == HIGHLIGHT_LINE_WIDTH
    assert _width(pw, "rpm") == THICK_LINE_WIDTH


# ---------------------------------------------------------------------------
# 双击 / 滚轮
# ---------------------------------------------------------------------------


def test_double_click_keeps_selection_and_leaves_no_stale_record(plotted_window, qapp):
    """真实双击序列：选择保持单条、无陈旧 press 记录，随后空白单击仍能清选。

    序列 = 单击（选中）→ dblclick（绘图区内弹变量编辑器）→ 尾随 release。
    dblclick 事件本身也会走 mousePressEvent（Qt 语义），故尾随 release 是
    一次合法的"再点同一条"，选择集必须仍是单条 speed。
    """
    mw = plotted_window
    pw = _pw(mw)
    pos = _curve_pos(pw, "speed")

    _click_plot(pw, pos)
    assert pw._legend_selection == {"speed"}, "首次单击须先选中"

    QTest.mouseDClick(pw.viewport(), Qt.MouseButton.LeftButton,
                      Qt.KeyboardModifier.NoModifier, pos)
    qapp.processEvents()
    assert pw._legend_selection == {"speed"}, "dblclick 本身不该改选择"
    assert _width(pw, "speed") == HIGHLIGHT_LINE_WIDTH

    _release(pw, pos)
    qapp.processEvents()
    settle()

    assert pw._legend_selection == {"speed"}
    assert _width(pw, "speed") == HIGHLIGHT_LINE_WIDTH
    assert pw._curve_click_press_pos is None, "尾随 release 后不得留陈旧记录"

    _click_plot(pw, _blank_pos(pw))
    assert pw._legend_selection == set()
    assert _width(pw, "speed") == THICK_LINE_WIDTH
    assert mw._message_label.text() == "已清除选择"


def test_wheel_zoom_keeps_selection(plotted_window, qapp):
    """滚轮缩放是热路径：选择集与加粗投影都不该被它改动。"""
    pw = _pw(plotted_window)
    _click_plot(pw, _curve_pos(pw, "speed"))
    assert pw._legend_selection == {"speed"}

    _wheel(pw, _curve_pos(pw, "speed"), delta=120)
    _wheel(pw, _curve_pos(pw, "speed"), delta=-120)
    settle()

    assert pw._legend_selection == {"speed"}
    assert _width(pw, "speed") == HIGHLIGHT_LINE_WIDTH
    assert _width(pw, "rpm") == THICK_LINE_WIDTH


# ---------------------------------------------------------------------------
# 隐藏曲线：点它的位置 = 点空白
# ---------------------------------------------------------------------------


def test_click_hidden_curve_position_clears_selection(plotted_window, qapp):
    """隐藏曲线不参与命中 → 裸单击其位置视为空白，清选并播报。

    可见性用 ci.visible 钉住而非 _apply_visibility：后者会收缩 Y limits 并
    把视野钳回剩余曲线范围（产品预期行为），被隐藏曲线的位置会落到 vb 外，
    从而测不到"视为空白"这一层。详见模块 docstring 第 2 条。
    """
    mw = plotted_window
    pw = _pw(mw)
    _click_plot(pw, _curve_pos(pw, "rpm"))
    assert pw._legend_selection == {"rpm"}

    pos = _curve_pos(pw, "speed")
    scene = pw.mapToScene(pos)
    assert pw.view_box.sceneBoundingRect().contains(scene), "落点须在绘图区内"
    assert hit_test_curve(pw, scene) == "speed", "隐藏前须真的命中 speed"

    pw.curves["speed"].visible = False
    qapp.processEvents()
    settle()
    assert hit_test_curve(pw, scene) is None, "隐藏曲线不该被命中"

    _click_plot(pw, pos)

    assert pw._legend_selection == set()
    assert _width(pw, "rpm") == THICK_LINE_WIDTH
    assert mw._message_label.text() == "已清除选择"