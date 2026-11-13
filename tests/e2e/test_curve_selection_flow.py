"""曲线选中联动端到端闭环（真实 MainWindow，offscreen）。

用户场景验收锚点：
加载 CSV → 绘图 → 在 plot 里单击曲线 → 选择集 + legend tint + 曲线加粗
→ Ctrl 加选第二条 → 空白单击清选；反向：legend Ctrl 点选 → 曲线加粗。
缩放热路径与游标开启态下高亮存活。

P1 断言"选择 + 两投影"；P2 在本文件末尾追加状态栏播报的**真实通道**断言
（_message_label / toolTip / info 定时回收 / 长名截断 / 游标读数隔离）。

关键隔离点由 e2e conftest 提供（sys.argv 重置 / app_settings 每例归零 /
QMessageBox 替身），本文件只补真实链路与几何前提。
"""

from urllib.parse import quote

import pytest

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from src.core.config import (
    HIGHLIGHT_CURVE_Z,
    HIGHLIGHT_LINE_WIDTH,
    THICK_LINE_WIDTH,
    UI_DEBOUNCE_DELAY_MS,
)
from src.core.data_types import SYMBOL_POLICY_OFF
from src.ui.drag_drop import ANCHOR_SCHEME
from src.ui.widgets.multi_curve_manager import hit_test_curve
from tests.fixtures.waits import pump, settle


@pytest.fixture(autouse=True)
def _guard_app_exit(monkeypatch):
    """防毒化：任何路径若调用 QApplication.exit，同进程后续用例会被 quitNow 拖死。"""
    monkeypatch.setattr(QApplication, "exit", staticmethod(lambda *a, **k: None))


@pytest.fixture()
def plotted_window(loaded_window, qapp, qtbot):
    """已绘 speed/rpm 两条曲线的真实 MainWindow，宿主布局完成后再量测。

    几何前提必须显式等待：未布局完成时 vb 只有 ~21px 高，8px 命中容差
    与像素断言全部失真（计划 §4.2 / 评审探针实测）。
    symbol 策略钉住为 off，让线宽断言不受可见点数阈值影响。
    """
    mw = loaded_window
    mw.resize(1200, 800)
    qapp.processEvents()
    pw = mw.plot_widgets[0].plot_widget
    for name in ("speed", "rpm"):
        assert pw.plot_variable(name), name
    for ci in pw.curves.values():
        ci.symbol_policy = SYMBOL_POLICY_OFF
    pw._refresh_selection_projections()
    qapp.processEvents()
    qtbot.waitUntil(
        lambda: pw.view_box.sceneBoundingRect().height() > 100, timeout=5000
    )
    settle()
    return mw


def _pw(mw):
    return mw.plot_widgets[0].plot_widget


def _curve_pos(pw, name, frac=0.5):
    """曲线上某数据点的 viewport 坐标（QTest 点击落点）。"""
    ci = pw.curves[name]
    i = min(len(ci.x_data) - 1, int(len(ci.x_data) * frac))
    scene = pw.view_box.mapViewToScene(
        QPointF(float(ci.x_data[i]), float(ci.y_data[i]))
    )
    return pw.mapFromScene(scene)


def _blank_pos(pw):
    """两条曲线之间的空带中央（speed 10~35 / rpm 800~1290，间隔极大）。"""
    pa = _curve_pos(pw, "speed")
    pb = _curve_pos(pw, "rpm")
    pos = QPoint(pa.x(), (pa.y() + pb.y()) // 2)
    assert hit_test_curve(pw, pw.mapToScene(pos)) is None, "空白点前提不成立"
    return pos


def _width(pw, name):
    return pw.curves[name].curve.opts["pen"].widthF()


def _tinted_texts(pw):
    return [es.cursor.selectedText() for es in pw.legend_label.extraSelections()]


def _is_tinted(pw, name):
    """tint 按 fragment 计（色块 + 名字各一段），真实 legend 名字带单位
    （如 `speed (km/h)`），故按前缀匹配而非全等。"""
    return any(t.startswith(name) for t in _tinted_texts(pw))


def _anchor_pos(browser, var_name):
    """legend 中变量名锚点的坐标（按 href 定位，截断安全）。"""
    href = f"{ANCHOR_SCHEME}:///{quote(var_name, safe='')}"
    for y in range(1, 60):
        for x in range(1, 600, 2):
            if browser.anchorAt(QPoint(x, y)) == href:
                return QPoint(x, y)
    pytest.fail(f"legend 中未找到 {href} 的锚点坐标")


def _click_plot(pw, pos, mods=Qt.KeyboardModifier.NoModifier):
    QTest.mouseClick(pw.viewport(), Qt.MouseButton.LeftButton, mods, pos)
    QApplication.processEvents()


def _legend_ctrl_click(pw, var_name):
    """legend Ctrl 点选：沿用既有 QMouseEvent 直调模式（计划 §4.2）。

    QTest 打 QTextBrowser 会经窗口投递到 viewport，绕过
    LegendTextBrowser.mousePressEvent 的 _drag_press_pos 记录 → 手势不成立。
    """
    browser = pw.legend_label
    pos = QPointF(_anchor_pos(browser, var_name))
    mods = Qt.KeyboardModifier.ControlModifier
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress, pos, pos, QPointF(0.0, 0.0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, mods,
    )
    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease, pos, pos, QPointF(0.0, 0.0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, mods,
    )
    browser.mousePressEvent(press)
    browser.mouseReleaseEvent(release)
    QApplication.processEvents()


# ---------------------------------------------------------------------------
# 正向：plot 点击 → 选择集 + 两投影
# ---------------------------------------------------------------------------


def test_click_curve_selects_thickens_and_tints(plotted_window, qapp):
    """单击曲线：选择集 / 加粗 2→4 / 提 Z / legend tint 四断言（真实通道）。"""
    pw = _pw(plotted_window)
    _click_plot(pw, _curve_pos(pw, "speed"))

    assert pw._legend_selection == {"speed"}
    assert _width(pw, "speed") == HIGHLIGHT_LINE_WIDTH
    assert pw.curves["speed"].curve.zValue() == HIGHLIGHT_CURVE_Z
    assert _is_tinted(pw, "speed")
    assert not _is_tinted(pw, "rpm")
    assert _width(pw, "rpm") == THICK_LINE_WIDTH


def test_ctrl_click_adds_second_curve(plotted_window, qapp):
    """Ctrl 加选第二条 → 两条都加粗；Ctrl 再击 → 减选并恢复。"""
    pw = _pw(plotted_window)
    _click_plot(pw, _curve_pos(pw, "speed"))
    _click_plot(pw, _curve_pos(pw, "rpm"), Qt.KeyboardModifier.ControlModifier)

    assert pw._legend_selection == {"speed", "rpm"}
    assert _width(pw, "speed") == HIGHLIGHT_LINE_WIDTH
    assert _width(pw, "rpm") == HIGHLIGHT_LINE_WIDTH

    _click_plot(pw, _curve_pos(pw, "rpm"), Qt.KeyboardModifier.ControlModifier)
    assert pw._legend_selection == {"speed"}
    assert _width(pw, "rpm") == THICK_LINE_WIDTH


def test_blank_click_clears_and_restores(plotted_window, qapp):
    """裸单击空白 → 清选 + 线宽/z/tint 全恢复。"""
    pw = _pw(plotted_window)
    _click_plot(pw, _curve_pos(pw, "speed"))
    _click_plot(pw, _curve_pos(pw, "rpm"), Qt.KeyboardModifier.ControlModifier)
    assert pw._legend_selection == {"speed", "rpm"}

    _click_plot(pw, _blank_pos(pw))

    assert pw._legend_selection == set()
    assert _width(pw, "speed") == THICK_LINE_WIDTH
    assert _width(pw, "rpm") == THICK_LINE_WIDTH
    assert pw.curves["speed"].curve.zValue() == 0
    assert pw.legend_label.extraSelections() == []


def test_ctrl_blank_keeps_selection(plotted_window, qapp):
    """Ctrl+空白 → 选择保持（与 legend 口径一致）。"""
    pw = _pw(plotted_window)
    _click_plot(pw, _curve_pos(pw, "speed"))
    _click_plot(pw, _blank_pos(pw), Qt.KeyboardModifier.ControlModifier)
    assert pw._legend_selection == {"speed"}
    assert _width(pw, "speed") == HIGHLIGHT_LINE_WIDTH


def test_pan_drag_does_not_change_selection(plotted_window, qapp):
    """N1：位移 ≥ startDragDistance 的左键拖拽是平移，不改选择集。

    断言口径为"press/release 序列不改变选择集"，不依赖 release 缺席
    （offscreen 下 QDrag.exec 立即返回，release 仍送达）。
    """
    pw = _pw(plotted_window)
    _click_plot(pw, _curve_pos(pw, "speed"))
    before = set(pw._legend_selection)

    start = _curve_pos(pw, "rpm")
    end = QPoint(start.x() + 40, start.y() + 30)
    assert (end - start).manhattanLength() >= QApplication.startDragDistance()
    QTest.mousePress(pw.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseRelease(pw.viewport(), Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, end)
    qapp.processEvents()
    settle()

    assert pw._legend_selection == before
    assert _width(pw, "speed") == HIGHLIGHT_LINE_WIDTH


def test_shift_drag_zoom_does_not_select(plotted_window, qapp):
    """R4：Shift 框选缩放专属，不被选择手势捕获。"""
    pw = _pw(plotted_window)
    start = _curve_pos(pw, "speed")
    end = QPoint(start.x() + 120, start.y() + 80)
    QTest.mousePress(pw.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.ShiftModifier, start)
    assert pw.rubberBand.isVisible()
    QTest.mouseRelease(pw.viewport(), Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.ShiftModifier, end)
    qapp.processEvents()
    settle()

    assert pw._legend_selection == set()
    assert not pw.rubberBand.isVisible()


# ---------------------------------------------------------------------------
# 反向：legend Ctrl 点选 → 曲线加粗
# ---------------------------------------------------------------------------


def test_legend_ctrl_click_thickens_curve(plotted_window, qapp):
    """legend 入口与 plot 入口共用同一真相源与同一投影。"""
    pw = _pw(plotted_window)
    _legend_ctrl_click(pw, "rpm")
    settle()

    assert pw._legend_selection == {"rpm"}
    assert _width(pw, "rpm") == HIGHLIGHT_LINE_WIDTH
    assert pw.curves["rpm"].curve.zValue() == HIGHLIGHT_CURVE_Z
    assert _is_tinted(pw, "rpm")


def test_plot_then_legend_mixed_multiselect(plotted_window, qapp):
    """两入口混合加选：plot 点 speed + legend Ctrl 点 rpm → 2 选。"""
    pw = _pw(plotted_window)
    _click_plot(pw, _curve_pos(pw, "speed"))
    _legend_ctrl_click(pw, "rpm")
    settle()

    assert pw._legend_selection == {"speed", "rpm"}
    assert _width(pw, "speed") == HIGHLIGHT_LINE_WIDTH
    assert _width(pw, "rpm") == HIGHLIGHT_LINE_WIDTH


# ---------------------------------------------------------------------------
# 共存：缩放热路径 / 游标开启态（风险 #2、验收 §6.4）
# ---------------------------------------------------------------------------


def test_highlight_survives_zoom(plotted_window, monkeypatch):
    """真实 debounce → update_plot_style 重刷后高亮存活（pen 缓存 _hl 维度）。

    评审 NB-5：原写法只 processEvents+settle，**不推进 50ms debounce
    定时器**，热路径调用数实测为 0，waitUntil 首次求值即真 → 断言空转
    （F3 反向注入实验证明对回归零检出力）。这里改为真实钟表推进 + spy
    计数断言，确保"重刷确实发生过"再断言高亮存活。
    """
    pw = _pw(plotted_window)
    _click_plot(pw, _curve_pos(pw, "speed"))
    assert _width(pw, "speed") == HIGHLIGHT_LINE_WIDTH

    manager = pw._plot_ui_manager
    original = manager.update_plot_style
    calls = []

    def _spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(manager, "update_plot_style", _spy)

    x0, x1 = pw.view_box.viewRange()[0]
    pw.view_box.setXRange(x0 + (x1 - x0) * 0.25, x0 + (x1 - x0) * 0.5, padding=0)
    pump(4 * UI_DEBOUNCE_DELAY_MS)
    settle()

    assert calls, "热路径未跑，本例对回归无检出力（NB-5）"
    assert pw._legend_selection == {"speed"}
    assert _width(pw, "speed") == HIGHLIGHT_LINE_WIDTH
    assert pw.curves["speed"].curve.zValue() == HIGHLIGHT_CURVE_Z
    assert _is_tinted(pw, "speed")


def test_selection_coexists_with_cursor(plotted_window, qapp, qtbot):
    """游标开启态：点击照常选中，游标 vline 仍可见且 z 序高于选中曲线。"""
    mw = plotted_window
    pw = _pw(mw)
    if not mw.cursor_btn.isChecked():
        qtbot.mouseClick(mw.cursor_btn, Qt.MouseButton.LeftButton)
        qapp.processEvents()
        settle()
    assert mw.cursor_btn.isChecked()

    _click_plot(pw, _curve_pos(pw, "speed"))

    assert pw._legend_selection == {"speed"}
    assert _width(pw, "speed") == HIGHLIGHT_LINE_WIDTH
    assert pw.vline.isVisible()
    assert pw.vline.zValue() > pw.curves["speed"].curve.zValue()


def test_selection_adds_no_scene_items(plotted_window, qapp):
    """D2 rev2：无徽章/无覆盖物 —— 选择只改 pen 与 ExtraSelections。"""
    pw = _pw(plotted_window)
    n0 = len(pw.scene().items())
    _click_plot(pw, _curve_pos(pw, "speed"))
    _click_plot(pw, _curve_pos(pw, "rpm"), Qt.KeyboardModifier.ControlModifier)
    assert len(pw.scene().items()) == n0


# ---------------------------------------------------------------------------
# P2 · 状态栏播报真实通道（计划 §3.4 / §4.2）
#
# 只有本层断言 _message_label / toolTip / info 定时回收；component 层
# 只断言文案与调用次数（FakeHost 无状态栏通道）。
# ---------------------------------------------------------------------------


def test_status_bar_shows_selected_name(plotted_window):
    """裸单击曲线 → 状态栏右区显示「已选中：speed」（真实通道）。"""
    mw = plotted_window
    pw = _pw(mw)

    _click_plot(pw, _curve_pos(pw, "speed"))

    assert mw._message_label.text() == "已选中：speed"
    assert mw._message_label.toolTip() == "已选中：speed"
    assert mw._message_text == "已选中：speed"


def test_status_bar_ctrl_select_and_deselect_wording(plotted_window):
    """Ctrl 加选/减选：三态文案逐条落到真实标签。"""
    mw = plotted_window
    pw = _pw(mw)
    ctrl = Qt.KeyboardModifier.ControlModifier

    _click_plot(pw, _curve_pos(pw, "speed"))
    assert mw._message_label.text() == "已选中：speed"
    _click_plot(pw, _curve_pos(pw, "rpm"), ctrl)
    assert mw._message_label.text() == "已选中：rpm"
    _click_plot(pw, _curve_pos(pw, "rpm"), ctrl)
    assert mw._message_label.text() == "已取消选中：rpm"
    assert pw._legend_selection == {"speed"}


def test_status_bar_blank_clear_snapshot(plotted_window):
    """BL-1 真实通道版：空选择清选不播报，非空才播「已清除选择」。"""
    mw = plotted_window
    pw = _pw(mw)
    mw.clear_status_message()
    assert mw._message_label.text() == "", "前置：右区应为空"

    _click_plot(pw, _blank_pos(pw))  # 本就为空 → 静默
    assert mw._message_label.text() == ""

    _click_plot(pw, _curve_pos(pw, "speed"))
    _click_plot(pw, _blank_pos(pw))
    assert mw._message_label.text() == "已清除选择"
    assert pw._legend_selection == set()


def test_status_bar_ctrl_blank_is_silent(plotted_window):
    """Ctrl+空白：选择保持，状态栏文案不被改写。"""
    mw = plotted_window
    pw = _pw(mw)

    _click_plot(pw, _curve_pos(pw, "speed"))
    _click_plot(pw, _blank_pos(pw), Qt.KeyboardModifier.ControlModifier)

    assert mw._message_label.text() == "已选中：speed"
    assert pw._legend_selection == {"speed"}


def test_status_bar_info_message_expires(plotted_window, monkeypatch, qtbot):
    """D8 瞬时语义：info 级到时自动回收（加速替身只改超时值）。

    先例 test_status_bar.py:249-264：先锁住真实配置字面值，再
    monkeypatch.setitem 把 5s 压到 50ms。
    """
    from src.ui.main_window import STATUS_MESSAGE_TIMEOUT_MS

    assert STATUS_MESSAGE_TIMEOUT_MS == {"info": 5000, "warn": 8000}
    monkeypatch.setitem(STATUS_MESSAGE_TIMEOUT_MS, "info", 50)

    mw = plotted_window
    pw = _pw(mw)
    _click_plot(pw, _curve_pos(pw, "speed"))
    assert mw._message_label.text() == "已选中：speed"

    qtbot.waitUntil(lambda: mw._message_label.text() == "", timeout=2000)
    # 回收只清文案，不动选择集与两投影（持久识别由加粗+tint 承担）
    assert pw._legend_selection == {"speed"}
    assert _width(pw, "speed") == HIGHLIGHT_LINE_WIDTH
    assert _is_tinted(pw, "speed")


def test_status_bar_later_broadcast_overwrites(plotted_window):
    """D8：后续播报覆盖前一条（不建常驻段、不排队）。"""
    mw = plotted_window
    pw = _pw(mw)

    _click_plot(pw, _curve_pos(pw, "speed"))
    assert mw._message_label.text() == "已选中：speed"
    _click_plot(pw, _curve_pos(pw, "rpm"), Qt.KeyboardModifier.ControlModifier)
    assert mw._message_label.text() == "已选中：rpm"

    mw._broadcast("其他动作播报")
    assert mw._message_label.text() == "其他动作播报"
    assert pw._legend_selection == {"speed", "rpm"}


def test_legend_and_plot_broadcast_identical_text(plotted_window):
    """D6 两入口同规格：真实通道上文案逐字相同。"""
    mw = plotted_window
    pw = _pw(mw)

    _click_plot(pw, _curve_pos(pw, "rpm"))
    plot_text = mw._message_label.text()
    _click_plot(pw, _blank_pos(pw))
    _legend_ctrl_click(pw, "rpm")
    settle()

    assert plot_text == mw._message_label.text() == "已选中：rpm"


@pytest.fixture()
def long_name_window(main_window, qapp, qtbot, dialog_stubs, tmp_path):
    """超长变量名的真实窗口（合成名，非现场标识符）。

    验证既有 ``_status_elided`` + ``setToolTip`` 全文出口对选择播报同样
    生效（计划 §3.4：零新增截断代码）。
    """
    from tests.fixtures.data_factory import make_simple_rows, write_csv

    long_name = "diag_" + "extended_channel_" * 9 + "tail"
    csv = write_csv(
        tmp_path / "e2e_long_name.csv",
        header=["time", long_name],
        units=["s", "degC"],
        rows=[[row[0], row[1]] for row in make_simple_rows(50)],
    )
    dialog_stubs["open_file"] = (str(csv), "CSV/TXT Files (*.csv)")
    qtbot.mouseClick(main_window.load_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: main_window.loader is not None, timeout=5000)
    qtbot.waitUntil(lambda: len(main_window.plot_widgets) > 0, timeout=5000)
    qapp.processEvents()
    assert not dialog_stubs["critical"], f"加载出现错误弹窗: {dialog_stubs['critical']}"

    mw = main_window
    mw.resize(1200, 800)
    qapp.processEvents()
    pw = _pw(mw)
    assert pw.plot_variable(long_name)
    pw.curves[long_name].symbol_policy = SYMBOL_POLICY_OFF
    pw._refresh_selection_projections()
    qapp.processEvents()
    qtbot.waitUntil(
        lambda: pw.view_box.sceneBoundingRect().height() > 100, timeout=5000
    )
    settle()
    return mw, long_name


def test_status_bar_long_name_elided_with_full_tooltip(long_name_window):
    """长变量名：标签按 … 截断，toolTip 给全文（既有通道，零新增代码）。"""
    mw, long_name = long_name_window
    pw = _pw(mw)
    full = f"已选中：{long_name}"

    _click_plot(pw, _curve_pos(pw, long_name))

    assert mw._message_text == full
    assert mw._message_label.toolTip() == full
    shown = mw._message_label.text()
    assert shown != full and shown.endswith("…"), f"未触发截断：{shown!r}"
    assert full.startswith(shown[:-1])
    assert pw._legend_selection == {long_name}


def test_status_bar_broadcast_keeps_cursor_readout(plotted_window, qapp, qtbot):
    """通道隔离：游标开启态下播报进状态栏，绘图区读数标签不受影响。

    读数标签的生成沿用 test_cursor_label_edge.py:53-63 的驱动口径
    （show_values_only=False + 越过时间节流直调更新链）：本例被测的是
    "播报通道与绘图区读数物理隔离"，不是游标更新链本身。
    """
    mw = plotted_window
    pw = _pw(mw)
    if not mw.cursor_btn.isChecked():
        qtbot.mouseClick(mw.cursor_btn, Qt.MouseButton.LeftButton)
        qapp.processEvents()
        settle()
    assert mw.cursor_btn.isChecked()

    ci = pw.curves["speed"]
    pw.show_values_only = False
    pw.vline.setVisible(True)
    pw.vline.setPos(float(ci.x_data[len(ci.x_data) // 2]))
    pw._is_interacting = False
    pw._last_cursor_update_time = 0.0
    pw._cursor_manager._update_multi_curve_cursor_label()
    qapp.processEvents()

    def _readouts():
        return [
            item.toPlainText()
            for item in pw._cursor_item_pool["labels"]
            if item.isVisible() and item.toPlainText()
        ]

    before = _readouts()
    assert before, "前置：游标读数应已生成"

    _click_plot(pw, _curve_pos(pw, "speed"))

    assert mw._message_label.text() == "已选中：speed"
    assert len(_readouts()) == len(before), "播报不得挤掉/清空绘图区读数"
    assert pw.vline.isVisible()