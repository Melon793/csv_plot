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
from PySide6.QtWidgets import QPushButton

from src.ui.drag_drop import build_var_mimedata
from src.ui.widgets.status_drawer import XAxisDrawer

#: 当前档的圆点标记，直接取生产常量：改了标记这条测试跟着走，不会假绿
_MARK = XAxisDrawer._MARK

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
    # 抽屉按钮是 QPushButton 不是 QToolButton：两个类的"原生"长得不一样（macOS
    # 下 56x32 白胶囊 vs 36x22 灰方块），与顶栏同框必须是同一类。
    # 比对前先去掉当前档的圆点标记，否则被标记那一枚按原文查不到
    return next(
        b
        for b in drawer.findChildren(QPushButton)
        if b.text().removeprefix(f"{_MARK} ") == text
    )


def test_axis_drawer_uses_its_own_narrower_width(loaded_window, qapp):
    """x 轴抽屉用自己的宽度上限：共用文件抽屉那 600 会在右侧白留一块空。

    440 是扫出来的：最紧的一行是页脚（提示文字 + 恢复默认 + 应用），实测宽度
    420 时 Fusion 只剩 1 px（222/221），440 剩 21 px。这里同时钉住页脚与预览
    行都不被裁 —— 收窄的唯一硬约束就是这两行。
    """
    from PySide6.QtWidgets import QLabel

    from src.core.config import STATUS_DRAWER_WIDTH, STATUS_DRAWER_WIDTH_AXIS

    mw = loaded_window
    drawer = _open(mw, qapp)

    assert STATUS_DRAWER_WIDTH_AXIS < STATUS_DRAWER_WIDTH
    assert drawer.width() <= STATUS_DRAWER_WIDTH_AXIS, "x 轴抽屉比设计宽度更宽"

    hint = next(l for l in drawer.findChildren(QLabel) if "Esc" in l.text())
    assert hint.width() >= hint.sizeHint().width(), f"页脚提示被裁：{hint.width()}"
    assert drawer.preview.width() >= drawer.preview.sizeHint().width()


def test_drawer_sits_above_status_bar(loaded_window, qapp):
    mw = loaded_window
    drawer = _open(mw, qapp)

    bar_top = mw.statusBar().mapToGlobal(QPoint(0, 0)).y()
    assert drawer.y() + drawer.height() <= bar_top, "抽屉压住了状态栏"
    assert drawer.x() >= mw.mapToGlobal(QPoint(0, 0)).x(), "抽屉左沿越出窗口"
    assert drawer.height() < mw.height() / 2, "抽屉太高会盖掉绘图区"


def test_seeded_from_current_base(loaded_window, qapp):
    """打开时输入框反映当前全局基准；两处文案同一套格式，只差默认值那一段。

    状态栏是常驻段，默认基准只报轴身份；抽屉预览是"要落地成什么"的回执，
    默认值也写全（作者定）。非默认时两处必须逐字相同，否则就成了预览一个说法、
    落地另一个说法。
    """
    mw = loaded_window
    drawer = _open(mw, qapp)

    assert drawer.candidate() == (1.0, 0.0)
    assert drawer.preview.text() == "x轴：Index（比例系数:1, 偏移量:0）"
    assert mw._axis_segment.text() == "x轴：Index"

    mw.factor, mw.offset = 0.01, 5.0
    mw._update_axis_segment()
    drawer._load_from(0.01, 5.0)
    assert drawer.preview.text() == mw._axis_segment.text()
    assert drawer.preview.text() == "x轴：Index（比例系数:0.01, 偏移量:5）"


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

    # 去掉当前档的圆点前缀再比：抽屉一开就有一枚带着标记
    assert [b.text().removeprefix(f"{_MARK} ") for b in drawer._preset_buttons] == labels
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


def test_active_preset_is_marked_with_a_dot(loaded_window, qapp):
    """当前档前缀一个圆点，其余普通文字；四枚等宽，切档时整行不抖。

    刻意不用 checked 态：实测 macOS style 把选中按钮的文字画成白字压白底（深色
    像素 266 → 0，palette 四种设法都救不回来）。也不加粗（作者定：单一个点够认）。
    手写出来的非常规值（0.002 → 500 Hz）哪档都不对，此时**一枚都不标** —— 这本身
    就是"当前不在预设档位上"的信息，不能拿最近的一档糊过去。
    """
    from src.core.config import X_AXIS_FREQUENCY_PRESETS

    labels = [label for _hz, label in X_AXIS_FREQUENCY_PRESETS]
    drawer = _open(loaded_window, qapp)
    buttons = drawer._preset_buttons

    assert all(not b.styleSheet() for b in buttons), "按钮又自带样式表就退出平台绘制了"
    assert all(not b.isCheckable() for b in buttons), (
        "别退回 checked：mac 上选中态的文字是白字压白底，看不见"
    )
    assert len({b.width() for b in buttons}) == 1, "四枚不等宽，切档会抖行"

    assert [b.text() for b in buttons] == [f"{_MARK} 1Hz", "5Hz", "10Hz", "100Hz"]
    assert not any(b.font().bold() for b in buttons), "作者只要圆点，不加粗"

    _button(drawer, "100Hz").click()
    assert [b.text() for b in buttons] == ["1Hz", "5Hz", "10Hz", f"{_MARK} 100Hz"]

    drawer.manual_check.setChecked(True)
    drawer.factor_spin.setValue(0.002)
    assert [b.text() for b in buttons] == labels


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
