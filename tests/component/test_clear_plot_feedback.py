"""「清除绘图」播报 component 测试（offscreen）。

三条用户入口（顶部按钮 / 右键菜单 / 双击中键）的文案收在
``MainWindow._announce_cleared_plots``；其中两条子图级入口（右键菜单、双击中键）
共用 ``PlotWidget.clear_current_plot()``。本文件钉住三件事：

1. 计数在清之前取 —— 两条子图级入口各验一次；
2. 空图不播报（守卫在文案函数里，子图级入口整条不调用）；
3. 文案函数自身的句式与守卫（不建真实状态栏，只记 _broadcast）。

真实状态栏上的呈现（文案落到右区消息格、顶部按钮那条入口）在
tests/e2e/test_status_bar.py。
"""

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from src.ui.main_window import MainWindow
from src.ui.widgets.custom_viewbox import ZH_CLEAR_PLOT


class _FakeMenuEvent:
    """只提供 getMenu 用到的 scenePos() 的伪事件（同 test_viewbox_context_menu_zh）"""

    def scenePos(self):
        return QPointF(0.0, 0.0)


def _stub_menu_context(pw):
    """补齐 CustomViewBox 菜单路径要查的 plot_context 成员（同既有菜单用例）"""
    ctx = pw.plot_context
    ctx.is_cursor_enabled = lambda: False
    ctx.cursor_mode = "1 free cursor"
    ctx.cursor_values_hidden = False
    ctx.get_row_height = lambda row: 100
    ctx._plot_col_max_default = 1
    ctx.plot_widgets = []


class _BroadcastRecorder:
    """只记 _broadcast 调用的最小宿主（_announce_cleared_plots 是纯文案函数）"""

    def __init__(self):
        self.calls = []

    def _broadcast(self, message, level="info"):
        self.calls.append((message, level))


def _middle_double_click(pw, qapp):
    ev = QMouseEvent(
        QEvent.Type.MouseButtonDblClick,
        QPointF(10.0, 10.0),
        QPointF(10.0, 10.0),
        Qt.MouseButton.MiddleButton,
        Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier,
    )
    pw.mouseDoubleClickEvent(ev)
    qapp.processEvents()


# ---------- 两条子图级入口 ----------

def test_middle_double_click_announces_cleared_count(plot_factory, qapp):
    """双击中键：清掉 2 条曲线，播报数目 2（清之前取）"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    assert pw.plot_variable("b")

    _middle_double_click(pw, qapp)

    assert list(pw.curves) == []
    assert pw.plot_context.cleared_announces == [("已清除绘图", 2)]


def test_right_click_menu_clear_announces_cleared_count(plot_factory, qapp):
    """右键菜单「清除绘图」：整链（QAction → 信号 → EventHandler → 落点）报数目"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    _stub_menu_context(pw)

    menu = pw.view_box.getMenu(_FakeMenuEvent())
    clear_act = next(a for a in menu.actions() if a.text() == ZH_CLEAR_PLOT)
    clear_act.trigger()
    qapp.processEvents()

    assert list(pw.curves) == []
    assert pw.plot_context.cleared_announces == [("已清除绘图", 1)]


def test_empty_plot_clear_stays_silent(plot_factory, qapp):
    """空图清一遍：落点照实数报 0 条，真文案函数把它咽掉 —— 屏上无声。

    守卫只此一处（在 _announce_cleared_plots 里），三条入口共用同一条规则。
    """
    pw = plot_factory()
    assert list(pw.curves) == []
    recorder = _BroadcastRecorder()
    pw.plot_context.announce_cleared = (
        lambda label, curves: MainWindow._announce_cleared_plots(recorder, label, curves)
    )

    _middle_double_click(pw, qapp)

    assert recorder.calls == []


# ---------- 文案函数本体 ----------

def test_announce_sentence_and_level():
    host = _BroadcastRecorder()
    MainWindow._announce_cleared_plots(host, "已清除全部绘图", 7)

    assert host.calls == [("已清除全部绘图 · 7 条曲线", "info")]


def test_announce_skips_zero_curves():
    host = _BroadcastRecorder()
    MainWindow._announce_cleared_plots(host, "已清除绘图", 0)

    assert host.calls == [], "0 条曲线时不该占用消息区"
