"""e2e：P0-2 状态栏 + 标题规范化（常驻四块版）。

口径（经本轮讨论定稿）：
1. 标题只放 文件名 - CSV Plot v{版本}，无「Alpha」字样；
2. 状态栏常驻四块：左文件段 / 中 x 轴段（均可点开抽屉）/ 右消息区 / 日志入口；
3. 加载反馈改为右区秒表文案 + 沙漏光标 + 加载期输入闸门；≥2MB 的模态进度框
   保留（百分比本来就不准，撤掉它反而丢了"还在干活"的证据）；
4. 消息按级别自动回收（错误常驻），且任何文案都不参与窗口最小宽度。

抽屉本身（点开后的内容与动作）在 tests/e2e/test_file_info_drawer.py。
"""

import time

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt

from tests.fixtures.data_factory import make_simple_rows, write_csv

# 保活容器：offscreen 下手工构造的 QMimeData 不受 Qt 事件系统接管，
# 局部变量被 GC 后 dropEvent 内访问 mimeData 会 SIGSEGV（README 陷阱 #2）
_drop_alive: list = []


def _pump_until(qapp, qtbot, predicate, timeout=8000):
    """轮询等待 + 显式泵事件：异步加载的回调靠事件循环投递。"""
    deadline = time.monotonic() + timeout / 1000
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        qtbot.wait(10)
    pytest.fail("等待状态栏条件超时")


def test_title_has_no_alpha_and_carries_version(main_window):
    """未加载文件：标题为「CSV Plot v{版本}」，与 defaultTitle 一致且无 Alpha。"""
    mw = main_window
    assert "Alpha" not in mw.windowTitle()
    assert mw.windowTitle() == mw.defaultTitle == f"CSV Plot v{mw.app_version}"


def test_status_bar_persistent_three_segments(main_window):
    """空闲态三段都在：左文件段占位、中 x 轴段未加载时隐藏、右消息区为空。"""
    mw = main_window
    assert mw.statusBar().isVisible()
    assert mw._file_segment.text() == "未加载文件"
    assert not mw._axis_segment.isVisible()
    assert not mw._segment_separator.isVisible(), "x 轴段隐藏时不该留孤立分隔线"
    assert mw._message_label.text() == ""


def test_title_and_segments_after_load(loaded_window):
    """加载后：标题退回纯文件名+版本，常态信息落在左/中两段。"""
    mw = loaded_window

    assert mw.windowTitle() == f"e2e_demo.csv - CSV Plot v{mw.app_version}"
    assert mw._file_segment.text() == f"csv文件 · {len(mw.var_names)} 个变量"
    assert "e2e_demo.csv" in mw._file_segment.toolTip()
    assert mw._axis_segment.isVisible()
    assert mw._segment_separator.isVisible(), "两段之间要有分隔"
    assert mw._message_label.text() == ""


def test_axis_segment_reports_axis_then_the_active_correction(loaded_window):
    """x 轴段：常态只报轴身份，修正过才把生效的系数与偏移摆出来。"""
    mw = loaded_window

    # 合成 CSV 没有可识别的时间列 → 轴身份如实是 Index，未修正时不带系数
    mw._update_axis_segment()
    assert mw._axis_segment.text() == "x轴：Index"

    mw.factor = 0.01
    mw._update_axis_segment()
    assert mw._axis_segment.text() == "x轴：Index（比例系数:0.01, 偏移量:0）"

    # 有时间列时轴身份换成轴名(单位)；系数不再反推成 Hz，统一走括号那一截
    mw.factor = 1
    mw.loader.time_column_name = "time"
    mw._update_axis_segment()
    assert mw._axis_segment.text() == "x轴：time (s)"

    mw.factor = 0.01
    mw.offset = 5.0
    mw._update_axis_segment()
    assert mw._axis_segment.text() == "x轴：time (s)（比例系数:0.01, 偏移量:5）"


def test_segments_are_clickable(main_window):
    """两段可点击：手型光标 + clicked 信号（点开抽屉的行为另有整链用例）。"""
    from PySide6.QtGui import QMouseEvent

    mw = main_window
    assert mw._file_segment.cursor().shape() == Qt.CursorShape.PointingHandCursor
    assert mw._axis_segment.cursor().shape() == Qt.CursorShape.PointingHandCursor

    hits = []
    mw._file_segment.clicked.connect(lambda: hits.append("file"))
    mw._axis_segment.clicked.connect(lambda: hits.append("axis"))
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(mw._file_segment.rect().center()),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    mw._file_segment.mousePressEvent(press)
    assert hits == ["file"]


def test_hover_paints_a_full_cell_block(main_window, qapp):
    """hover 反馈是整块底色（照 VS Code 的 item cell），不是 1 px 下划线。

    钉三件事：格子比文字宽出两侧内边距（命中区不再要求"点到字上"）、底色一直
    涂进内边距那一列、离开后收回。offscreen 与 cocoa 都会画，所以这条在 CI 上
    真有约束力（区别于抽屉里那两条只在真机才复现得出的样式）。
    """
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QFontMetrics
    from PySide6.QtWidgets import QApplication

    seg = main_window._file_segment
    # 阈值写死而不是取 PAD_X：取常量的话，把常量改成 0 这条用例照样通过（实测）
    text_w = QFontMetrics(seg.font()).horizontalAdvance(seg.text())
    assert seg.width() - text_w >= 8, "格子几乎等于文字宽：内边距丢了，命中区回到\"必须点到字\""

    before = seg.grab().toImage()
    # 投真的 QEnterEvent：类型是 Enter 的裸 QEvent 会被 QWidget::event 当成
    # QEnterEvent 来 static_cast，是未定义行为（同目录抽屉用例实测崩过）
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QEnterEvent

    QApplication.sendEvent(seg, QEnterEvent(QPointF(2, 2), QPointF(2, 2), QPointF(2, 2)))
    after = seg.grab().toImage()
    tinted = [
        (x, y)
        for x in range(0, 4)
        for y in range(after.height())
        if after.pixelColor(x, y) != before.pixelColor(x, y)
    ]
    assert len(tinted) >= 4 * after.height() * 0.9, "底色没涂到文字之外的内边距"
    assert not seg.font().underline(), "下划线已换成色块，别退回去做双重反馈"

    QApplication.sendEvent(seg, QEvent(QEvent.Type.Leave))
    back = seg.grab().toImage()
    assert all(
        back.pixelColor(x, y) == before.pixelColor(x, y)
        for x in range(before.width())
        for y in range(before.height())
    ), "离开后底色没收回"


def test_broadcast_expires_but_error_persists(main_window, qtbot):
    """右区消息：info 到时自动清理，error 常驻直到被新消息替换。"""
    mw = main_window
    mw._broadcast("模板已保存: demo")
    assert mw._message_label.text() == "模板已保存: demo"
    qtbot.waitUntil(lambda: mw._message_label.text() == "", timeout=8000)

    mw._broadcast("加载失败：文件被占用", level="error")
    assert mw._message_label.text() == "加载失败：文件被占用"
    mw.clear_status_message()
    assert mw._message_label.text() == "加载失败：文件被占用", "错误消息不应自动消失"

    mw._broadcast("后来的消息")
    assert mw._message_label.text() == "后来的消息"


def test_busy_gate_blocks_only_while_thread_runs(main_window):
    """输入闸门以"后台线程在跑"为准：UI 刷新链的锁不再误伤正常操作。"""
    mw = main_window

    class _FakeThread:
        def __init__(self, running):
            self._running = running

        def isRunning(self):
            return self._running

    mw._thread = _FakeThread(True)
    assert mw.reject_when_loading("拖入变量") is True
    assert "正在加载数据，请稍候再拖入变量" in mw._message_label.text()

    mw._thread = _FakeThread(False)
    assert mw.reject_when_loading("拖入变量") is False

    del mw._thread
    assert mw.is_data_loading() is False


def test_async_load_keeps_modal_and_elapsed_text(
    main_window, qapp, qtbot, tmp_path, monkeypatch
):
    """异步分支：模态摆动条保留；底部不再有百分比进度条，右区给秒表文案。"""
    mw = main_window
    csv = write_csv(
        tmp_path / "e2e_async.csv",
        header=["time", "speed", "rpm", "flag"],
        units=["s", "km/h", "rpm", "-"],
        rows=make_simple_rows(80),
    )

    created = {"count": 0, "closed": 0}

    class _RecordingProgressDialog:
        def __init__(self, *args, **kwargs):
            created["count"] += 1

        def close(self):
            created["closed"] += 1

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    monkeypatch.setattr("src.ui.file_loader_manager.QProgressDialog", _RecordingProgressDialog)
    monkeypatch.setattr("src.ui.file_loader_manager.FILE_SIZE_LIMIT_BACKGROUND_LOADING", 0)

    messages = []
    original = mw._broadcast

    def spy(message, level="info"):
        messages.append(message)
        original(message, level=level)

    monkeypatch.setattr(mw, "_broadcast", spy)

    mw.file_loader_manager.load_csv_file(str(csv))
    _pump_until(qapp, qtbot, lambda: mw.loader is not None)
    _pump_until(qapp, qtbot, lambda: "e2e_async.csv" in mw.windowTitle())

    assert created["count"] == 1, "≥2MB 应弹模态加载框"
    assert created["closed"] == 1, "加载完成要关掉模态框"
    assert any(m.startswith("正在加载 e2e_async.csv …") for m in messages)
    assert not hasattr(mw, "_load_progress_bar"), "百分比进度条已撤"
    assert mw._message_label.text() == "", "加载完成后消息区应自动清空"
    assert mw._file_segment.text() == f"csv文件 · {len(mw.var_names)} 个变量"


def test_log_entry_and_edge_padding(main_window, qapp):
    """第四块是日志入口；状态栏两端留白，避免圆角窗口切掉贴边文字。"""
    from PySide6.QtCore import QPoint

    mw = main_window
    qapp.processEvents()
    assert mw._log_segment.text() == "日志"
    assert mw._log_segment.cursor().shape() == Qt.CursorShape.PointingHandCursor

    bar = mw.statusBar()
    left_x = mw._file_segment.mapTo(bar, QPoint(0, 0)).x()
    log_x = mw._log_segment.mapTo(bar, QPoint(0, 0)).x()
    right_gap = bar.width() - (log_x + mw._log_segment.width())
    assert left_x >= 8, f"左端留白不足: {left_x}px"
    assert right_gap >= 8, f"右端留白不足: {right_gap}px"
    assert log_x > left_x, "日志块应在最右侧"


def test_long_message_does_not_widen_window(loaded_window, qapp):
    """回归：超长文案不得抬高窗口最小宽度（曾表现为拖动时窗口突然变宽）。"""
    mw = loaded_window
    qapp.processEvents()
    min_before = mw.minimumWidth()

    # 贴到布局最小宽度：此时任何最小宽度增量都会直接表现为窗口变宽
    mw.resize(min_before, mw.height())
    qapp.processEvents()
    width_at_min = mw.width()

    long_message = " · ".join(f"field_name_{i}=12345.678 unit" for i in range(24))
    mw._broadcast(long_message)
    qapp.processEvents()

    assert mw.minimumWidth() == min_before, "状态栏把窗口最小宽度顶高了"
    assert mw.width() == width_at_min, "窗口被长文案拉宽"
    assert mw._message_label.text().endswith("…"), "超长文案应截断显示"
    assert mw._message_text == long_message, "原始文案应完整缓存（tooltip/日志用）"


def test_segment_separator_is_a_single_plain_line(main_window):
    """段间竖线必须是"我们自绘的一条线"，不能交给样式画 3D 凹槽。

    回归：``Shadow.Sunken`` 会让平台样式把它画成"暗 + 亮"两笔（实测单元格
    x=[150,153) 里 150/151 两列都有墨迹），Windows 高 DPI 下表现为
    "细-粗-细"三条且不居中。
    """
    from PySide6.QtGui import QColor

    from PySide6.QtWidgets import QFrame

    from src.ui import theme

    for sep in (main_window._segment_separator, main_window._log_separator):
        assert sep.frameShape() == QFrame.Shape.VLine
        assert sep.frameShadow() == QFrame.Shadow.Plain, "Sunken 会被样式画成两笔凹槽"
        assert sep.lineWidth() == 1
        assert (
            sep.palette().color(sep.foregroundRole()).name()
            == QColor(theme.SEP_ON_BAR).name()
        ), "竖线取色必须走 foregroundRole(WindowText)，设 Text 不生效"


def test_log_separator_follows_the_message(main_window, qtbot):
    """空闲时日志前不该有一条左边没有内容的孤线；有消息才出现，回收后消失。"""
    mw = main_window
    assert mw._message_label.text() == ""
    assert not mw._log_separator.isVisible(), "空闲态不该留孤立分隔线"

    mw._broadcast("临时播报")
    qtbot.waitUntil(lambda: mw._log_separator.isVisible(), timeout=2000)

    mw.clear_status_message()
    qtbot.waitUntil(lambda: not mw._log_separator.isVisible(), timeout=2000)


def test_auto_restore_toggle_announces_what_it_will_do(main_window):
    """「自动恢复」勾完就收起菜单，那个 ✓ 太小：切换要在右区说一句后果。"""
    mw = main_window

    mw._on_auto_restore_toggled(True)
    assert "已开启" in mw._message_text
    assert "下次加载" in mw._message_text, "只重复开关名等于没说"

    mw._on_auto_restore_toggled(False)
    assert "已关闭" in mw._message_text


def test_quick_apply_of_a_deleted_template_broadcasts_instead(
    main_window, dialog_stubs
):
    """快速项指向已删模板：撤了 warning，改成 error 级常驻播报。

    这一条同时钉住"不许再弹框"和"清设置"两件事 —— 清完设置菜单里就没有这一
    项了，屏上若一个字都不留，用户只会以为菜单自己变短。
    """
    mw = main_window
    mw._last_template_id = "deadbeef"
    mw._last_template_name = "已失踪的模板"

    mw._quick_apply_template()

    assert not dialog_stubs["warning"], f"弹窗已撤，不该再有 warning: {dialog_stubs['warning']}"
    assert mw._message_text == "模板[已失踪的模板]已被删除"
    assert mw._message_level == "error", "要常驻到用户下一次动作，不能 5 秒自己没了"
    assert mw._last_template_id is None


def test_saved_template_announces_name_not_id(main_window, monkeypatch):
    """保存播报从前是 uuid 短码；换成模板名 + 规模，且与列表两列同口径。"""
    from types import SimpleNamespace

    mw = main_window
    mw._last_template_name = "台架四项"
    monkeypatch.setattr(
        mw.plot_config_manager.template_manager,
        "get_template",
        lambda _tid: SimpleNamespace(
            config={"plots": [{"curves": ["a", "b"]}, {"curves": ["b", "c"]}]}
        ),
    )

    text = mw._template_saved_text("模板已保存", "deadbeef")

    assert "deadbeef" not in text, "uuid 短码不该出现在界面上"
    assert text == "模板已保存：台架四项 · 3 个变量 / 2 个子图", "变量数按去重算，与列表「变量数」列一致"


def test_apply_template_announces_match(loaded_window, qapp):
    """套用一条通道映射不完整的模板：必须说清匹配度与缺失数，并按 warn 停留。"""
    from types import SimpleNamespace

    mw = loaded_window
    cfg = {
        "layout_rows": 1,
        "layout_cols": 2,
        "time_factor": 1.0,
        "time_offset": 0.0,
        "plots": [
            {"curves": ["speed"]},
            {"curves": ["rpm"]},
            {"curves": ["这根通道数据里没有"]},
        ],
    }

    mw._check_and_apply_template(SimpleNamespace(config=cfg), "abc12345", "台架三项")
    qapp.processEvents()

    assert mw._message_text == "已套用模板[台架三项] · 匹配 67%（1 个变量缺失）"
    assert mw._message_level == "warn", "图上有空格子，5 秒就收走等于没说"


def test_auto_restore_announces_in_the_same_shape(loaded_window):
    """自动恢复也是套用一套通道映射，播报必须与「已套用/已强制套用」同规格。"""
    from src.core.plot_config import PlotConfig, PlotSessionConfig

    mw = loaded_window
    config = PlotSessionConfig(
        plots=[PlotConfig(curves=["speed"]), PlotConfig(curves=["rpm"])]
    )

    mw.file_loader_manager._announce_auto_restore(config, ["speed", "rpm"])
    assert mw._message_text == "已自动恢复上次布局 · 匹配 100%"
    assert mw._message_level == "info", "全对上就不该用警告色粘在屏上"

    mw.file_loader_manager._announce_auto_restore(config, ["speed", "别的"])
    assert mw._message_text == "已自动恢复上次布局 · 匹配 50%（1 个变量缺失）"
    assert mw._message_level == "warn"


def test_status_bar_suppresses_style_item_borders(main_window):
    """状态栏必须自己掐掉样式给 item 画的边框，否则 Windows 上会长出多余竖线。

    样式在 item 之间的 6 px 间隙里画一对明暗边框，会把我们那一根
    ``_vline_separator`` 夹在中间，实测（Qt legacy Windows 样式代理）
    "变量 ↔ x轴" 区间有 3 条比背景暗的线、"csv" 左边还有 2 条；加
    ``QStatusBar::item { border: none; }`` 后收敛到 1 / 0 条。
    macOS/Fusion 本来就不画，所以这条在 mac 上是无副作用的保险。
    """
    sheet = main_window.statusBar().styleSheet()

    assert "QStatusBar::item" in sheet, "样式表被删了：Windows 上会长出多余竖线"
    assert "border:none" in sheet.replace(" ", ""), sheet


# ---------------- 清除绘图：三条入口同一套播报 ----------------

def _plot_variables(mw, var_names, qapp, plot_index=0):
    """拖入变量到指定子图（照 test_load_and_plot.py 的 dropEvent 手法）"""
    from PySide6.QtGui import QDropEvent

    from src.ui.drag_drop import build_var_mimedata

    mime = build_var_mimedata(var_names)
    _drop_alive.append(mime)  # offscreen 下手工 QMimeData 需防 GC（陷阱 #2）
    pw = mw.plot_widgets[plot_index].plot_widget
    pw.dropEvent(
        QDropEvent(
            QPointF(50.0, 50.0),
            Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    qapp.processEvents()
    return pw


def test_clear_all_button_announces_total_curves(loaded_window, qapp, qtbot):
    """顶部「清除绘图」：清的是全部子图，播报给总数（且数目在清之前取）。

    数目取在清之后的话这里会是 0 条 —— 一条只为"报个数字"的用例。
    """
    mw = loaded_window
    pw = _plot_variables(mw, ["speed", "rpm"], qapp)
    total = sum(len(c.plot_widget.curves) for c in mw.plot_widgets)
    assert total == 2, "前置条件：只往第一个子图画了 2 条曲线"

    qtbot.mouseClick(mw.clear_all_plots_btn, Qt.MouseButton.LeftButton)
    qapp.processEvents()

    assert list(pw.curves) == []
    assert mw._message_text == "已清除全部绘图 · 2 条曲线"
    assert mw._message_level == "info", "清成功不是问题，不该用 warn 粘在屏上"


def test_right_click_clear_announces_current_plot(loaded_window, qapp):
    """子图右键菜单「清除绘图」：只清当前子图，播报走同一条文案函数。"""
    from src.ui.widgets.custom_viewbox import ZH_CLEAR_PLOT

    mw = loaded_window
    pw = _plot_variables(mw, ["speed"], qapp)

    class _FakeMenuEvent:
        def scenePos(self):
            return QPointF(0.0, 0.0)

    menu = pw.view_box.getMenu(_FakeMenuEvent())
    clear_act = next(a for a in menu.actions() if a.text() == ZH_CLEAR_PLOT)
    clear_act.trigger()
    qapp.processEvents()

    assert list(pw.curves) == []
    assert mw._message_text == "已清除绘图 · 1 条曲线"


def test_middle_double_click_announces_current_plot(loaded_window, qapp):
    """双击中键清当前子图：最容易误触的入口，播报必须给（本操作无撤销）。"""
    from PySide6.QtGui import QMouseEvent

    mw = loaded_window
    pw = _plot_variables(mw, ["speed", "rpm"], qapp)

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

    assert list(pw.curves) == []
    assert mw._message_text == "已清除绘图 · 2 条曲线"


def test_clear_with_nothing_drawn_stays_silent(loaded_window, qapp):
    """数据加载了但一条曲线都没画：清除是空操作，屏上不该出现"已清除"字样。"""
    mw = loaded_window
    mw.clear_status_message()
    assert all(not c.plot_widget.curves for c in mw.plot_widgets)

    mw.cursor_sync_manager.clear_all_plots()
    qapp.processEvents()

    assert mw._message_text == "", "清了个空图还报一条，等于占着消息区说废话"
