"""e2e：状态栏中段 → x 轴时间基准上翻抽屉（P0-2 Step C）。

这一段要证的三件事，都不是"看起来对"：
1. 频率→系数的换算**不被 spinbox 小数位截断**（3 Hz 必须落成 1/3 全精度，
   实测按 6 位小数的 0.333333 存，3M 点横轴会漂出约 1 秒）；
2. 落地与顶部「时间修正」对话框共用 ``apply_time_correction``，所以可视 x
   范围要保住、状态栏中段要同步、恢复默认真的能回到原点；
3. 两扇抽屉互斥，且无数据 / 加载中不开抽屉。
"""

import pytest
from PySide6.QtCore import QEvent, QPointF, QPoint, Qt
from PySide6.QtGui import QDropEvent, QMouseEvent
from PySide6.QtWidgets import QToolButton

from src.ui.drag_drop import build_var_mimedata

# README 陷阱 #2：QDropEvent 不持有 mimeData 的所有权，局部变量被 GC 后
# dropEvent 内访问它就是 SIGSEGV，必须由模块级容器续命
_keep_alive = []


def _plot_first_variables(mw, names=("speed",)):
    """往首个绘图区拖几个变量：没有曲线就看不出基准有没有落地。"""
    pw = mw.plot_widgets[0].plot_widget
    mime = build_var_mimedata(list(names))
    _keep_alive.append(mime)
    pw.dropEvent(
        QDropEvent(
            QPointF(10.0, 10.0),
            Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    return pw


def _click(widget, qapp):
    local = QPointF(widget.rect().center())
    widget.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            local,
            QPointF(widget.mapToGlobal(local.toPoint())),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    qapp.processEvents()


def _open(loaded_window, qapp):
    _click(loaded_window._axis_segment, qapp)
    drawer = loaded_window._axis_drawer
    assert drawer is not None and drawer.isVisible(), "点中段应弹出 x 轴抽屉"
    return drawer


def _button(drawer, text):
    return next(b for b in drawer.findChildren(QToolButton) if b.text() == text)


def test_drawer_sits_above_status_bar(loaded_window, qapp):
    mw = loaded_window
    drawer = _open(mw, qapp)

    bar_top = mw.statusBar().mapToGlobal(QPoint(0, 0)).y()
    assert drawer.y() + drawer.height() <= bar_top, "抽屉压住了状态栏"
    assert drawer.x() >= mw.mapToGlobal(QPoint(0, 0)).x(), "抽屉左沿越出窗口"
    assert drawer.height() < mw.height() / 2, "抽屉太高会盖掉绘图区"


def test_seeded_from_current_base(loaded_window, qapp):
    """打开时输入框反映当前全局基准，预览与状态栏中段是同一句话。"""
    mw = loaded_window
    drawer = _open(mw, qapp)

    assert drawer.candidate() == (1.0, 0.0)
    assert drawer.preview.text() == mw._axis_segment.text()


def test_preset_sets_frequency_and_derives_factor(loaded_window, qapp):
    drawer = _open(loaded_window, qapp)

    _button(drawer, "100Hz").click()

    assert drawer.freq_spin.value() == 100.0
    assert drawer.candidate()[0] == 0.01
    assert drawer.preview.text() == "x轴：Index（比例系数:0.01, 偏移量:0）"


def test_preset_row_matches_config(loaded_window, qapp):
    """档位由 config 决定（作者定的 1/5/10/100 Hz），抽屉不得自带一份列表。

    按钮文字自带单位：这一行没有别的线索说明"1"是 Hz 还是系数。
    """
    from src.core.config import X_AXIS_FREQUENCY_PRESETS

    drawer = _open(loaded_window, qapp)
    labels = [label for _hz, label in X_AXIS_FREQUENCY_PRESETS]

    assert [b.text() for b in drawer._preset_buttons] == labels
    assert labels == ["1Hz", "5Hz", "10Hz", "100Hz"]


def test_preview_line_is_not_clipped(loaded_window, qapp):
    """大号预览行必须整行通栏：作者实测「系数 0.2（已修正）」被裁成"已修"。

    原先预览只占输入区那两列（cocoa 实测可用 228 px / 需要 258 px），确认键
    挤在同一行右侧。这里用最长的那句文案钉住宽度，防止再被挤回去。
    """
    drawer = _open(loaded_window, qapp)

    drawer.freq_spin.setValue(3.0)  # → "x轴：Index（比例系数:0.333333, 偏移量:0）"
    qapp.processEvents()

    assert drawer.preview.width() >= drawer.preview.sizeHint().width()


def test_active_preset_is_marked_current(loaded_window, qapp):
    """生效那一档预设显示"淡蓝=当前"，与网格选择器同一颜色语义。

    手写出来的非常规值（0.002 → 500 Hz）哪档都不对，此时**不高亮任何一档** ——
    这本身就是"当前不在预设档位上"的信息，不能拿最近的一档糊过去。
    """
    from src.core.config import X_AXIS_FREQUENCY_PRESETS
    from src.ui import theme

    drawer = _open(loaded_window, qapp)
    plain = theme.chip_style("QToolButton")

    assert _button(drawer, "1Hz").styleSheet() != plain, "默认 1 Hz 就该标成当前"
    _button(drawer, "100Hz").click()
    assert _button(drawer, "100Hz").styleSheet() != plain
    assert _button(drawer, "1Hz").styleSheet() == plain

    drawer.manual_check.setChecked(True)
    drawer.factor_spin.setValue(0.002)
    assert all(
        _button(drawer, label).styleSheet() == plain
        for _hz, label in X_AXIS_FREQUENCY_PRESETS
    )


def test_enter_applies_without_clicking(loaded_window, qapp, qtbot):
    """焦点不在输入框里时，Enter 直接应用（与网格选择器同一套键盘模型）。

    只覆盖"事件落到抽屉"这一种情形：offscreen 平台不给 popup 抓键盘（实测
    hasFocus 恒为 False），所以这里按仓库既有做法把键事件直接投给抽屉本体。
    """
    mw = loaded_window
    drawer = _open(mw, qapp)

    _button(drawer, "10Hz").click()
    qtbot.keyClick(drawer, Qt.Key.Key_Return)

    assert mw.factor == 0.1
    assert not drawer.isVisible()


def test_derived_factor_keeps_full_precision(loaded_window, qapp):
    """3 Hz 的落地系数必须是 1/3 全精度，而不是输入框显示的 0.333333。"""
    drawer = _open(loaded_window, qapp)

    drawer.freq_spin.setValue(3.0)

    assert drawer.factor_spin.value() == pytest.approx(0.333333, abs=1e-6)
    assert drawer.candidate()[0] == 1 / 3


def test_manual_toggle_moves_authority_between_fields(loaded_window, qapp):
    """勾"手写系数"后：频率框退成回显（禁用），系数框成为权威输入。"""
    drawer = _open(loaded_window, qapp)

    assert drawer.freq_spin.isEnabled() and not drawer.factor_spin.isEnabled()
    drawer.manual_check.setChecked(True)

    assert not drawer.freq_spin.isEnabled() and drawer.factor_spin.isEnabled()
    drawer.factor_spin.setValue(0.002)
    assert drawer.freq_spin.value() == 500.0
    assert drawer.candidate()[0] == 0.002


def test_apply_writes_global_base_and_recomputes_curve_x(loaded_window, qapp):
    """应用：全局基准、状态栏中段、消息区、抽屉四处同步，且曲线横轴真的重算了。"""
    mw = loaded_window
    pw = _plot_first_variables(mw)
    qapp.processEvents()
    drawer = _open(mw, qapp)

    _button(drawer, "100Hz").click()
    _button(drawer, "应用").click()

    assert mw.factor == 0.01 and mw.offset == 0.0
    assert mw._axis_segment.text() == "x轴：Index（比例系数:0.01, 偏移量:0）"
    assert not drawer.isVisible()
    assert mw._message_label.text() == "已应用 x 轴基准：系数 0.01，偏移 0"

    assert pw.factor == 0.01 and pw.offset == 0.0
    curve = next(iter(pw.curves.values()))
    # 合成 CSV 有 50 行、Index 从 1 起：最后一个点的 x 必须是 50 × 0.01
    assert curve.x_data[-1] == pytest.approx(50 * 0.01)


def test_reset_returns_to_the_original_axis(loaded_window, qapp):
    """恢复默认要**立即落地**：只清输入框会让人以为已恢复而横轴没变。"""
    mw = loaded_window
    pw = _plot_first_variables(mw)
    qapp.processEvents()
    drawer = _open(mw, qapp)
    before = pw.view_box.viewRange()[0]

    _button(drawer, "100Hz").click()
    _button(drawer, "应用").click()
    assert mw.factor == 0.01

    _open(mw, qapp)
    _button(drawer, "恢复默认").click()

    assert mw.factor == mw._factor_default and mw.offset == mw._offset_default
    assert mw._axis_segment.text() == "x轴：Index"
    assert pw.view_box.viewRange()[0] == pytest.approx(before, rel=1e-6)


def test_two_drawers_are_mutually_exclusive(loaded_window, qapp):
    """同一时刻只留一扇抽屉：两扇叠着用户分不清刚点的是哪一段。"""
    mw = loaded_window
    axis = _open(mw, qapp)

    _click(mw._file_segment, qapp)
    assert mw._file_info_drawer.isVisible() and not axis.isVisible()

    _click(mw._axis_segment, qapp)
    assert axis.isVisible() and not mw._file_info_drawer.isVisible()


def test_click_without_data_only_broadcasts(main_window, qapp):
    mw = main_window
    _click(mw._axis_segment, qapp)

    assert mw._axis_drawer is None
    assert mw._message_label.text() == "尚未加载数据文件"


def test_click_while_loading_is_gated(main_window, qapp):
    class _FakeThread:
        def isRunning(self):
            return True

    mw = main_window
    mw._thread = _FakeThread()
    _click(mw._axis_segment, qapp)

    assert mw._axis_drawer is None
    assert "正在加载数据，请稍候再改 x 轴基准" in mw._message_label.text()


def test_apply_rejects_non_positive_factor(loaded_window):
    """共用的 apply 对非法系数返回 False 且不动全局值。

    输入框本身有下限挡不住 0，而对话框那条路径的 spinbox 允许键入 0 ——
    守卫必须在共用的这一层，否则两个入口只有一个有校验。
    """
    mw = loaded_window

    assert mw.layout_manager.apply_time_correction(0.0, 1.0) is False
    assert mw.factor == 1 and mw.offset == 0
