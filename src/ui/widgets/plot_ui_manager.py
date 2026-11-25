"""
PlotUIManager - UI初始化与刷新协调管理器

负责 DraggableGraphicsLayoutWidget 的所有 UI 初始化工作：
- 整体 setup_ui 编排
- 顶部 header 区域
- 主绘图区域和 ViewBox
- 坐标轴样式配置
- 交互元素（光标线、RubberBand）
- UI 更新调度协调（防抖 + 批量更新）

此模块从 csv_plot_pyqt6.py 迁移而来。
"""

from __future__ import annotations
from typing import Any

from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import (
    QAction,
    QFontMetrics,
    QPen,
    QColor,
    QDrag,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QMenu,
    QSizePolicy,
    QGraphicsProxyWidget,
    QGraphicsLinearLayout,
    QGraphicsWidget,
    QRubberBand,
    QApplication,
    QToolTip,
    QTextBrowser,
    QTextEdit,
    QFrame,
    QWidget,
    QWidgetAction,
)
import pyqtgraph as pg

from src.core.config import (
    DEFAULT_CURVE_COLORS,
    DEFAULT_PADDING_VAL_Y,
    DEFAULT_SHOW_X_AXIS_LABEL,
    UI_DEBOUNCE_DELAY_MS,
    widget_alive,
)
from src.core.logger import get_logger
from src.core.data_types import (
    DRAW_MODE_AUTO,
    DRAW_MODE_SOURCE_AUTO,
    DRAW_MODE_STEP_POST,
    LINE_PRESET_AUTO,
    SYMBOL_POLICY_AUTO,
    resolve_symbol_show,
)
from src.ui import theme
from src.ui.widgets.curve_style_panel import (
    DRAW_MODE_LABELS,
    LINE_PRESET_LABELS,
    SYMBOL_POLICY_LABELS,
    apply_curve_panel_picks,
    build_curve_panel_menu,
    build_style_panel_widget,
    exec_curve_panel,
)
from src.ui.drag_drop import (
    build_legend_var_mimedata,
    clear_active_legend_drag,
    parse_legend_anchor,
    set_active_legend_drag,
)
from src.ui.drag_visuals import (
    DragItem,
    build_drag_pixmap,
    drag_hotspot,
    transfer_hint,
)
from src.ui.transfer_broadcast import (
    announce_selection,
    broadcast_text,
    broadcast_transfer,
    loading_reject_message,
)

logger = get_logger("widget.plot_ui")
from src.core.scheduler import UnifiedUpdateScheduler
from src.ui.widgets.base_manager import BasePlotManager

LEGEND_MAX_LINES = 3  # legend 最大显示行数，超出部分通过滚动条查看
LEGEND_BOTTOM_PAD = 0  # legend 底部留白，保证末行文字不与 plot 上边框接触

# legend 滚动条样式：细条圆角灰色、无箭头按钮
LEGEND_SCROLLBAR_QSS = """
    QScrollBar:vertical {
        background: #f2f2f2;
        width: 8px;
        margin: 1px;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical {
        background: #b0b0b0;
        min-height: 16px;
        border-radius: 3px;
    }
    QScrollBar::handle:vertical:hover {
        background: #8f8f8f;
    }
    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {
        height: 0px;
    }
    QScrollBar::add-page:vertical,
    QScrollBar::sub-page:vertical {
        background: transparent;
    }
"""







class LegendTextBrowser(QTextBrowser):
    """支持将变量名拖出到其他 plot 的 legend 容器。

    点击（位移 < startDragDistance）保持原有 anchorClicked 切换显隐行为；
    超过阈值才发起 QDrag，两种手势物理互斥（见设计文档 §3.4）。
    右键（位移 < startDragDistance）弹出单变量操作菜单（设计 v1.2 方案 B）。
    图例快速操作（快速改色/快速删除方案）：色块左键弹调色板、中键单击
    删除条目，hover 锚点时以 tooltip 说明手势。
    """

    # 锚点 tooltip 引导文案（方案 §2.8）：按锚点类型区分
    _TOOLTIP_COLOR = "单击：改色"
    _TOOLTIP_TOGGLE = "单击：显示/隐藏 · 中键：删除"
    _TOOLTIP_INVALID = "失效占位 · 中键：删除"
    # 悬停延迟：需在锚点上停留该时长才显示（避免扫视图例时反复弹层）
    _TOOLTIP_DELAY_MS = 800

    def __init__(self, plot_widget):
        super().__init__()
        # 强引用：legend_label 经 proxy 由 pw 持有，二者生命周期同源；
        # 形成的 pw → proxy → label → _pw 引用环由 CPython 循环 GC 回收，
        # 无需 weakref（Qt 父子链保证 label 先于 pw 销毁）
        self._pw = plot_widget
        self._drag_press_pos: QPoint | None = None
        self._drag_var_name: str | None = None
        # 左键 press 时的修饰键（P1 多选手势：Ctrl 加选 / Shift 范围）
        self._drag_modifiers = Qt.KeyboardModifier.NoModifier
        self._ctx_press_pos: QPoint | None = None
        self._ctx_var_name: str | None = None
        # 中键手势状态（快速删除方案 §2.6）：press 记录、release 校验
        self._mid_press_pos: QPoint | None = None
        self._mid_var_name: str | None = None
        # 悬停提示状态：文案 + 最新坐标 + 是否已显示 + 延迟计时器
        self._tooltip_text: str | None = None
        self._tooltip_pos: QPoint | None = None
        self._tooltip_shown = False
        self._tooltip_timer = QTimer(self)
        self._tooltip_timer.setSingleShot(True)
        self._tooltip_timer.setInterval(self._TOOLTIP_DELAY_MS)
        self._tooltip_timer.timeout.connect(self._show_pending_tooltip)

    def mousePressEvent(self, event):
        self._hide_tooltip()  # 任何按下即隐藏手势引导（§2.8）
        if event.button() == Qt.MouseButton.LeftButton:
            href = self.anchorAt(event.position().toPoint())
            # 按压瞬间解析变量名（拖拽期间 setHtml 重建文档也不影响）。
            # 宽松解析（快速改色方案 §2.7）：curve:/// 与 color:/// 均可
            # 发起拖拽（色块上按住也能拖出），忽略 action 只取变量名。
            parsed = parse_legend_anchor(href)
            self._drag_var_name = parsed[1] if parsed else None
            self._drag_press_pos = event.position().toPoint()
            self._drag_modifiers = event.modifiers()
        elif event.button() == Qt.MouseButton.RightButton:
            # 右键：记录锚点与按压位置，release 时校验位移后弹菜单
            # （设计 v1.2 §3.2；与左键拖拽状态机同构）。同上宽松解析：
            # 色块上右键也弹单变量菜单。
            parsed = parse_legend_anchor(
                self.anchorAt(event.position().toPoint())
            )
            self._ctx_var_name = parsed[1] if parsed else None
            self._ctx_press_pos = event.position().toPoint()
        elif event.button() == Qt.MouseButton.MiddleButton:
            # 中键：记录锚点与按压位置，release 校验位移后删除（§2.6）。
            # 宽松解析：名称/色块/占位锚点均命中。press 显式 accept 且
            # 不进 QTextBrowser 处理链（中键无文本语义），避免未 accept
            # 的事件沿 scene 向下传播。
            parsed = parse_legend_anchor(
                self.anchorAt(event.position().toPoint())
            )
            self._mid_var_name = parsed[1] if parsed else None
            self._mid_press_pos = event.position().toPoint()
            event.accept()
            return
        super().mousePressEvent(event)  # 保留链接高亮/anchor 内部状态

    def mouseMoveEvent(self, event):
        if (
            self._drag_var_name
            and self._drag_press_pos is not None
            and event.buttons() & Qt.MouseButton.LeftButton
            and (event.position().toPoint() - self._drag_press_pos).manhattanLength()
            >= QApplication.startDragDistance()
        ):
            self._start_var_drag()
            return
        self._update_anchor_tooltip(event.position().toPoint())
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._hide_tooltip()
        super().leaveEvent(event)

    def _anchor_tooltip_text(self, pos: QPoint) -> str | None:
        """解析光标下锚点对应的手势引导文案；无提示则返回 None。

        空白处、陈旧锚点（变量刚被删、legend 尚未重建）返回 None。
        """
        parsed = parse_legend_anchor(self.anchorAt(pos))
        if parsed is None:
            return None
        action, var_name = parsed
        if var_name in self._pw.curves:
            return (
                self._TOOLTIP_COLOR if action == "color" else self._TOOLTIP_TOGGLE
            )
        if var_name in (getattr(self._pw, "invalid_curves", {}) or {}):
            return self._TOOLTIP_INVALID
        return None

    def _update_anchor_tooltip(self, pos: QPoint) -> None:
        """hover 锚点时延迟显示手势引导 tooltip（快速改色方案 §2.8）。

        停留超过 ``_TOOLTIP_DELAY_MS`` 才显示：扫视图例时不再反复弹层。
        同一锚点内移动只更新坐标、不重置计时；换锚点或移到空白处则
        收起旧提示并重新计时。QToolTip 为全局浮层，需换算
        viewport → global 坐标。
        """
        text = self._anchor_tooltip_text(pos)
        if text is None:
            self._hide_tooltip()
            return
        if text == self._tooltip_text:
            # 同一锚点内移动：只更新坐标；提示仍在计时或已显示则不重来
            self._tooltip_pos = pos
            if self._tooltip_timer.isActive() or self._tooltip_shown:
                return
            # 提示已被系统自动隐藏：同一锚点重新计时
            self._tooltip_timer.start()
            return
        self._hide_tooltip()
        self._tooltip_text = text
        self._tooltip_pos = pos
        self._tooltip_timer.start()

    def _show_pending_tooltip(self) -> None:
        """延迟计时到点：在最新悬停坐标显示待展示文案。"""
        if self._tooltip_text is None or self._tooltip_pos is None:
            return
        if not widget_alive(self):
            return
        QToolTip.showText(
            self.viewport().mapToGlobal(self._tooltip_pos),
            self._tooltip_text,
            self,
        )
        self._tooltip_shown = True

    def _hide_tooltip(self) -> None:
        """停止计时并隐藏提示（按下 / 离开 / 移到空白处）。"""
        self._tooltip_timer.stop()
        self._tooltip_text = None
        self._tooltip_pos = None
        self._tooltip_shown = False
        QToolTip.hideText()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            # 中键 release：位移小于拖拽阈值且锚点命中才删除（按住拖出
            # 视为取消，与左键/右键手势对称）。显式 accept 阻断沿
            # proxy→header→scene 传播（widget 侧护栏，view 侧见
            # PlotWidget.mouseDoubleClickEvent）。
            var_name = self._mid_var_name
            press_pos = self._mid_press_pos
            self._mid_var_name = None
            self._mid_press_pos = None
            event.accept()
            in_curves = var_name in self._pw.curves
            in_invalid = var_name in (getattr(self._pw, "invalid_curves", {}) or {})
            if (
                var_name
                and press_pos is not None
                and (event.position().toPoint() - press_pos).manhattanLength()
                < QApplication.startDragDistance()
                and (in_curves or in_invalid)
            ):
                # singleShot(0)：不在 release 处理器内改 scene / 重建
                # legend HTML（与右键菜单 singleShot 同构）
                QTimer.singleShot(
                    0, lambda: self._exec_middle_delete(var_name)
                )
            return
        if event.button() == Qt.MouseButton.RightButton:
            # 右键 release：位移小于拖拽阈值才弹菜单（按住拖出 legend 视为
            # 取消，与左键手势对称）；锚点解析失败或曲线已陈旧则静默不弹。
            # 状态无条件清空，防止悬空残留。
            var_name = self._ctx_var_name
            press_pos = self._ctx_press_pos
            self._ctx_var_name = None
            self._ctx_press_pos = None
            in_curves = var_name in self._pw.curves
            in_invalid = var_name in (getattr(self._pw, "invalid_curves", {}) or {})
            if (
                var_name
                and press_pos is not None
                and (event.position().toPoint() - press_pos).manhattanLength()
                < QApplication.startDragDistance()
                and (in_curves or in_invalid)
            ):
                global_pos = self.viewport().mapToGlobal(
                    event.position().toPoint()
                )
                # D5：右键未选条目只操作该条、不动选择集；右键已选且
                # 选择集 >1 时弹批量菜单。
                selection = getattr(self._pw, "_legend_selection", set())
                batch_menu = in_curves and var_name in selection and len(selection) > 1
                # singleShot(0)：不在 release 处理器内直接进 menu.exec 嵌套
                # 事件循环，让 Qt 先完成 release 收尾（与 _start_var_drag 同构）
                if batch_menu:
                    QTimer.singleShot(
                        0, lambda: self._exec_batch_var_menu(global_pos)
                    )
                else:
                    QTimer.singleShot(
                        0, lambda: self._exec_var_menu(var_name, global_pos)
                    )
            return
        # 左键 release：先判定 P1 选择手势（Ctrl 加选 / Shift 范围 / 空白清选）。
        # 命中即 return，不进 super —— 跳过 anchorClicked，显隐不切换（D4）。
        press_pos = self._drag_press_pos
        var_name = self._drag_var_name
        modifiers = self._drag_modifiers
        self._drag_press_pos = None
        self._drag_var_name = None
        self._drag_modifiers = Qt.KeyboardModifier.NoModifier
        if (
            press_pos is not None
            and (event.position().toPoint() - press_pos).manhattanLength()
            < QApplication.startDragDistance()
        ):
            if var_name is not None and var_name in self._pw.curves:
                if modifiers & Qt.KeyboardModifier.ControlModifier:
                    self._toggle_legend_selection(var_name)
                    selection = getattr(self._pw, "_legend_selection", None)
                    if selection is not None:
                        announce_selection(
                            self._pw,
                            "select" if var_name in selection else "deselect",
                            var_name,
                        )
                    return
                if modifiers & Qt.KeyboardModifier.ShiftModifier:
                    self._range_select_legend(var_name)
                    # 与 Ctrl 分支同口径：_range_select_legend 在选择集缺失时
                    # 早退（no-op），此时不得播报（评审 r2 NB-A3）。
                    if getattr(self._pw, "_legend_selection", None) is not None:
                        announce_selection(self._pw, "select", var_name)
                    return
            elif var_name is None and not modifiers:
                # 空白单击清空**本页**选择；裸单击条目不清（D4：显隐正交）
                clear_page = getattr(self._pw, "clear_page_selection", None)
                cleared = bool(clear_page()) if callable(clear_page) else False
                if cleared:
                    announce_selection(self._pw, "clear")
        super().mouseReleaseEvent(event)  # 未拖拽时正常触发 anchorClicked

    def mouseDoubleClickEvent(self, event):
        """中键双击在图例内终结（T0 探针实证的护栏第二层）。

        view 层（PlotWidget.mouseDoubleClickEvent）对图例区内中键双击
        放行进 scene 后，到达本控件时显式 accept：中键无文本双击语义，
        未处理的事件会沿 scene 继续传播（ViewBox 等），必须就地终结。
        """
        if event.button() == Qt.MouseButton.MiddleButton:
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        """纯抑制器：静默消费，不调 super（屏蔽 QTextBrowser 默认富文本菜单）。

        注：经 QGraphicsProxyWidget 的转发链路此方法实际不会被调用——
        Qt 源码 qgraphicsproxywidget.cpp 的 contextMenuEvent 存在
        !hasFocus() 早退，而 legend 为 NoFocus（探针 v7 已证零转发，
        见 tmp/proxy_context_menu_probe_v7.py）。保留覆写作为纵深防御：
        若未来 focus 策略变化导致事件到达，静默消费可避免默认富文本
        菜单与右键业务菜单双弹。
        """
        event.accept()

    def _exec_var_menu(self, var_name: str, global_pos: QPoint) -> None:
        """构建并弹出单变量操作菜单，按 exec 返回值分发动作（设计 §4.1）。

        显隐两项（仅显示此变量 / 显示全部变量）并列常驻，各自按「点击
        是否还会改变可见性」置灰（已处于目标态则不可点），不再共用一个
        按状态切换文案的互斥项。末尾「绘图变量编辑器」与 plot 自身右键
        菜单同源（均走 PlotWidget.open_variable_editor），总是可点。

        动作在菜单关闭后才执行（不在嵌套事件循环内改 scene / 重建
        legend HTML）。首行 + 分发前双查 var_name in pw.curves
        （§6 R3：菜单打开期间曲线可能被 reload/编辑器删除）。
        """
        # 调用方是 singleShot(0) 的 lambda：事件循环转回来之前 legend 可能已
        # 随 plot 关闭/重载销毁，只剩 Python 包装器（与 2a86e94 修的 X-link
        # 防抖同类）。先问存活，再去摸 pw.curves / 建 QMenu(pw)。
        if not widget_alive(self) or not widget_alive(self._pw):
            logger.debug("legend 变量菜单跳过：控件已销毁")
            return
        pw = self._pw
        if var_name not in pw.curves:
            # 失效占位专用轻量菜单（§6.2）：删除占位 + 复制变量名。
            # 销毁资产须用户显式操作；删除后由调用方显式 emit curves_changed。
            if var_name in (getattr(pw, "invalid_curves", {}) or {}):
                self._exec_invalid_menu(var_name, global_pos)
            return
        menu = QMenu(pw)
        act_solo = menu.addAction("仅显示此变量")
        act_solo.setEnabled(pw.can_solo(var_name))
        act_show_all = menu.addAction("显示全部变量")
        act_show_all.setEnabled(pw.can_show_all())
        menu.addSeparator()
        # 曲线样式：原「绘制方式」「数据点」两个子菜单合并为单一样式面板
        # （形态三），与左键色块快捷面板共享 _build_style_panel_widget；
        # 应用统一走 _apply_curve_panel_picks（MultiCurveManager 共享入口）
        ci = pw.curves[var_name]
        # 枚举通道自动阶梯时在面板分组标签标注来源（可解释性；手动选择无标注）
        auto_step = (
            getattr(ci, "draw_mode", DRAW_MODE_AUTO) == DRAW_MODE_STEP_POST
            and getattr(ci, "draw_mode_source", "") == DRAW_MODE_SOURCE_AUTO
        )
        picked: dict[str, str] = {}
        holder = [menu]
        sub_style = menu.addMenu("样式")
        style_panel = self._build_style_panel_widget(
            var_name, picked, holder, draw_group_suffix="（枚举通道自动）" if auto_step else ""
        )
        style_act = QWidgetAction(sub_style)
        style_act.setDefaultWidget(style_panel)
        sub_style.addAction(style_act)
        menu.addSeparator()
        act_remove = menu.addAction("删除变量")
        act_copy = menu.addAction("复制变量名")
        act_info = menu.addAction("变量信息")
        menu.addSeparator()
        act_editor = menu.addAction("绘图变量编辑器")

        chosen = menu.exec(global_pos)
        menu.deleteLater()
        if chosen is None and not picked:
            return
        if var_name not in pw.curves:  # 菜单打开期间可能已陈旧
            return
        # 样式面板选择（点段即记录并关闭整个菜单链，exec 返回 None）：
        # 关闭后统一执行，与色板同口径
        if picked:
            self._apply_curve_panel_picks(var_name, picked)
            return
        main_window = pw.window()
        if chosen is act_copy:
            QApplication.clipboard().setText(var_name)
        elif chosen is act_remove:
            pw.remove_variable_from_plot(var_name)
            if main_window is not None and hasattr(main_window, "layout_manager"):
                main_window.layout_manager.request_mark_stats_refresh()
        elif chosen is act_solo:
            pw.solo_curve_visibility(var_name)
        elif chosen is act_show_all:
            pw.show_all_curves()
        elif chosen is act_editor:
            pw.open_variable_editor()
        elif chosen is act_info:
            if main_window is None or getattr(main_window, "loader", None) is None:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(pw, "错误", "尚未加载数据文件")
                return
            from src.ui.dialogs.variable_info_dialog import VariableInfoDialog
            VariableInfoDialog.popup([var_name], parent=main_window)

    def _exec_invalid_menu(self, var_name: str, global_pos: QPoint) -> None:
        """失效占位专用轻量菜单（§6.2）：删除占位 + 复制变量名 + 替换变量。

        占位无曲线可切显隐/独占/看信息。删除是销毁用户资产，须显式操作；
        删除后显式 emit curves_changed（对齐 §4.5 发射口径），驱动编辑器/
        搜索栏增量刷新。动作在菜单关闭后执行（不进嵌套事件循环改状态）。
        「替换变量…」（身份匹配 D1/D7）：跨文件后缀漂移落占位后的手动
        兜底，执行序见 _replace_invalid_variable（D13）。
        """
        if not widget_alive(self) or not widget_alive(self._pw):
            return
        pw = self._pw
        if var_name not in (getattr(pw, "invalid_curves", {}) or {}):
            return
        menu = QMenu(pw)
        act_remove = menu.addAction("删除占位")
        act_copy = menu.addAction("复制变量名")
        act_replace = menu.addAction("替换变量…")

        chosen = menu.exec(global_pos)
        # 第 6 轮 E-1：exec 后释放（早退分支同样覆盖），与三个面板入口同口径
        menu.deleteLater()
        if chosen is None:
            return
        # 菜单打开期间占位可能已复活/被删（reload、编辑器）：分发前复查
        invalid = getattr(pw, "invalid_curves", None)
        if not invalid or var_name not in invalid:
            return
        if chosen is act_copy:
            QApplication.clipboard().setText(var_name)
        elif chosen is act_remove:
            del invalid[var_name]
            pw._update_header_for_curves()
            pw.curves_changed.emit()
        elif chosen is act_replace:
            self._replace_invalid_variable(var_name)

    def _replace_invalid_variable(self, var_name: str) -> None:
        """占位「替换变量…」执行序（身份匹配 D13，五步）：

        ①新名已在本格 curves 或恰为另一占位 → 拒绝并播报，不删任何资产；
        ②add_variable_to_plot 成功后才删旧占位（失败回滚不删）；
        ③显示名/换算走 _apply_curve_settings 静默口径；
        ④颜色经 preferred_color、可见性经 apply_curve_visibility 携带；
        ⑤回执播报一条。新名恰为占位自身时走同名复活（add 内部原子消费
        占位并带回全部偏好），不做手动搬运。
        """
        if not widget_alive(self) or not widget_alive(self._pw):
            return
        pw = self._pw
        invalid = getattr(pw, "invalid_curves", None)
        if not invalid or var_name not in invalid:
            return
        main_window = pw.window()
        broadcast = getattr(main_window, "_broadcast", None)

        def say(text: str, level: str = "info") -> None:
            if callable(broadcast):
                broadcast(text, level=level)

        loader = getattr(main_window, "loader", None)
        var_names = list(getattr(loader, "var_names", None) or [])
        if not var_names:
            say("尚未加载数据文件，无法替换占位", level="warn")
            return

        from src.ui.dialogs.variable_picker import VariablePickerDialog
        new_name = VariablePickerDialog.pick(var_names, parent=pw)
        if new_name is None:
            return
        # 对话框 exec 期间占位可能已复活/被删：执行前复查（:517-520 同款）
        invalid = getattr(pw, "invalid_curves", None)
        if not invalid or var_name not in invalid:
            return
        # D13①：判重 / 撞另一占位 → 拒绝，不删任何资产
        if new_name in pw.curves:
            say(f"变量 {new_name} 已在本格绘图中，替换取消", level="warn")
            return
        if new_name != var_name and new_name in invalid:
            say(f"{new_name} 已是本格占位，替换取消", level="warn")
            return

        ref = invalid[var_name]
        # D13②：先 add 后删；quiet 失败不弹模态框，占位原样保留
        ok = pw.add_variable_to_plot(
            new_name,
            preferred_color=ref.color,
            quiet=True,
            show_duplicate_warning=False,
            announce_step=False,
        )
        if not ok:
            say(f"替换失败：{new_name} 不可绘制，占位保留", level="warn")
            return
        if new_name != var_name:
            # D13③④：设置静默还原 + 可见性偏好携带（颜色已经 preferred_color）
            entry: dict = {}
            if ref.display_name:
                entry["display_name"] = ref.display_name
            if ref.factor != 1.0:
                entry["factor"] = ref.factor
            if ref.offset != 0.0:
                entry["offset"] = ref.offset
            if entry:
                from src.ui.plot_config_manager import PlotConfigManager
                PlotConfigManager._apply_curve_settings(pw, {new_name: entry})
            if not ref.visible_preference:
                pw._multi_curve_manager.apply_curve_visibility(new_name, False)
            del invalid[var_name]
        pw._update_header_for_curves()
        pw.curves_changed.emit()
        # D13⑤：回执一条
        say(f"已替换占位 {var_name} → {new_name}")

    def _exec_middle_delete(self, var_name: str) -> None:
        """中键单击条目：删除曲线（或失效占位），状态栏播报回执。

        快捷手势无菜单确认（快速删除方案 §2.6），成功时以状态栏播报
        作为可见回执，失败静默。执行前二次复查归属：singleShot(0) 转回
        事件循环期间 curves/invalid 可能已变（含中键双击第二次 release
        调度的天然幂等）。动作分支与右键菜单删除/占位删除完全一致。
        """
        if not widget_alive(self) or not widget_alive(self._pw):
            return
        pw = self._pw
        main_window = pw.window()
        show = getattr(main_window, "_show_status_message", None)

        if var_name in pw.curves:
            pw.remove_variable_from_plot(var_name)
            if main_window is not None and hasattr(main_window, "layout_manager"):
                main_window.layout_manager.request_mark_stats_refresh()
            if callable(show):
                show(f"已移除 {var_name}")
        elif var_name in (getattr(pw, "invalid_curves", {}) or {}):
            del pw.invalid_curves[var_name]
            pw._update_header_for_curves()
            pw.curves_changed.emit()
            if callable(show):
                show(f"已移除 {var_name}")

    def _build_style_panel_widget(
        self,
        var_name: str,
        picked: dict[str, str],
        holder: list,
        draw_group_suffix: str = "",
        title_text: str | None = None,
    ) -> QWidget:
        """薄委托：共享组件 curve_style_panel（C3 抽离，行为不变）。"""
        return build_style_panel_widget(
            self._pw, var_name, picked, holder, draw_group_suffix, title_text
        )

    def _build_curve_panel_menu(
        self, var_name: str, picked: dict[str, str], holder: list
    ) -> tuple[QMenu, QAction]:
        """薄委托：共享组件 curve_style_panel（C3）。"""
        return build_curve_panel_menu(self._pw, var_name, picked, holder)

    def _apply_curve_panel_picks(self, var_name: str, picked: dict[str, str]) -> None:
        """薄委托：共享组件 curve_style_panel（C3）。"""
        apply_curve_panel_picks(self._pw, var_name, picked)

    def _exec_curve_panel(self, var_name: str, global_pos: QPoint) -> None:
        """薄委托：共享组件 curve_style_panel（C3）。

        评审 P1-3：singleShot(0) 回环期间 legend 可能已销毁而 pw 仍活，
        共享函数只守卫 pw，故委托内补自身存活检查（与抽离前同口径）。
        """
        if not widget_alive(self):
            return
        exec_curve_panel(self._pw, var_name, global_pos)


    def _start_var_drag(self):
        var_name = self._drag_var_name
        self._drag_var_name = None
        self._drag_press_pos = None
        self._drag_modifiers = Qt.KeyboardModifier.NoModifier
        # 占位锚点不发起 legend 拖拽（§6.1）：拖到目标页后 plot_variable 会失败弹错
        if var_name in (getattr(self._pw, "invalid_curves", {}) or {}):
            return
        # P1：拖已选条目 → 整批（批量协议 skip+quiet）；拖未选条目 → 单变量，
        # 且选择集不动（D5 口径）。
        pw = self._pw
        selection = set(getattr(pw, "_legend_selection", set()) or set())
        if var_name in selection and len(selection) > 1:
            names = [name for name in pw.curves if name in selection]
            batch = True
        else:
            names = [var_name]
            batch = False
        drag = QDrag(self)
        drag.setMimeData(build_legend_var_mimedata(names, pw))
        pixmap = build_drag_pixmap(
            [
                DragItem(name, getattr(pw.curves.get(name), "color", None))
                for name in names
            ],
            self.font(),
            hint=transfer_hint(),
        )
        if pixmap:
            drag.setPixmap(pixmap)
            drag.setHotSpot(drag_hotspot())
        set_active_legend_drag(pw, names, batch=batch)
        try:
            # CopyAction|MoveAction 并存，drop 端按 Alt 决定语义（默认移动、Alt=复制）。
            # exec 进入嵌套事件循环，release 被拖拽会话消费，
            # anchorClicked 不会误发 —— 点击/拖拽互斥的核心保证。
            drag.exec(Qt.DropAction.CopyAction | Qt.DropAction.MoveAction)
        finally:
            clear_active_legend_drag()
            # P1-6（第 5 轮）：释放 QDrag/QMimeData/缩略图，防长会话累积；
            # exec 期间源控件被销毁时 C++ 对象已亡，静默跳过（评审 N1）
            try:
                drag.deleteLater()
            except RuntimeError:
                pass

    # ------------------------------------------------------------------
    # P1 legend 多选：选择集 + 高亮 + 批量入口
    # ------------------------------------------------------------------
    def _toggle_legend_selection(self, var_name: str) -> None:
        """Ctrl+单击：切换单个条目的选中态（不切显隐）。"""
        pw = self._pw
        selection = getattr(pw, "_legend_selection", None)
        if selection is None:
            return
        if var_name in selection:
            selection.discard(var_name)
        else:
            # 同页单持有：加选前静默清空同页兄弟（减选不触发）
            clear_others = getattr(pw, "clear_page_selection_except_self", None)
            if callable(clear_others):
                clear_others()
            selection.add(var_name)
        pw._legend_sel_anchor = var_name
        pw._refresh_selection_projections()

    def _range_select_legend(self, var_name: str) -> None:
        """Shift+单击：自锚点按 curves 插入序范围选（文件管理器语义）。

        范围选择替换旧选择集；锚点在范围选择后保持不变，便于连续 Shift
        调整范围。首次 Shift（无有效锚点）退化为加选当前条目并立锚。
        """
        pw = self._pw
        selection = getattr(pw, "_legend_selection", None)
        if selection is None:
            return
        order = list(pw.curves)
        anchor = getattr(pw, "_legend_sel_anchor", None)
        clear_others = getattr(pw, "clear_page_selection_except_self", None)
        if anchor in order and var_name in order:
            start, end = order.index(anchor), order.index(var_name)
            if start > end:
                start, end = end, start
            # 先算新集合再清兄弟：范围为空时不得误清别的 plot
            new_selection = set(order[start:end + 1])
            if new_selection and callable(clear_others):
                clear_others()
            selection.clear()
            selection.update(new_selection)
        else:
            if callable(clear_others):
                clear_others()
            selection.add(var_name)
            pw._legend_sel_anchor = var_name
        pw._refresh_selection_projections()

    def clear_legend_selection(self) -> None:
        """清空选择集（空白单击 / 菜单「清除选择」/ 占位态回归共用）。"""
        pw = self._pw
        selection = getattr(pw, "_legend_selection", None)
        if selection is not None:
            selection.clear()
        pw._legend_sel_anchor = None
        self.setExtraSelections([])
        # 两投影收口：tint 已就地清空（保留原语义，含无选择集属性的替身），
        # 这里补曲线加粗投影恢复。播报不在本函数内——用户/系统共用入口。
        # getattr 守卫与同函数内 selection/highlight 的防御风格一致：轻量
        # 替身 pw 可能没有该方法（评审 NB-4 / D13）。
        refresh = getattr(pw, "_refresh_selection_projections", None)
        if callable(refresh):
            refresh()

    def refresh_selection_highlight(self, *, prune: bool = True) -> None:
        """按当前选择集重放 ExtraSelections 高亮。

        - 按 fragment 的 anchorHref 匹配（**按 href 不按显示文本**，截断安全）；
        - ``prune=True`` 时把已不在 curves 中的条目从选择集修剪掉
          （reload/删除后的陈旧选择）；
        - 图例重建（setHtml）后由 PlotWidget.update_legend_label 调用重放。
        """
        pw = self._pw
        selection = getattr(pw, "_legend_selection", None)
        if selection is None:
            return
        if prune:
            valid = set(pw.curves)
            stale = selection - valid
            if stale:
                selection -= stale
            if getattr(pw, "_legend_sel_anchor", None) not in valid:
                pw._legend_sel_anchor = None
        if not selection:
            self.setExtraSelections([])
            return

        extras = []
        doc = self.document()
        block = doc.begin()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                if fragment.isValid():
                    parsed = parse_legend_anchor(
                        fragment.charFormat().anchorHref()
                    )
                    if parsed and parsed[1] in selection:
                        cursor = QTextCursor(doc)
                        cursor.setPosition(fragment.position())
                        cursor.setPosition(
                            fragment.position() + fragment.length(),
                            QTextCursor.MoveMode.KeepAnchor,
                        )
                        extra = QTextEdit.ExtraSelection()
                        extra.cursor = cursor
                        fmt = QTextCharFormat()
                        # 25% 档：与全 app 选中 tint 同一 token
                        fmt.setBackground(theme.tint_qcolor())
                        extra.format = fmt
                        extras.append(extra)
                it += 1
            block = block.next()
        self.setExtraSelections(extras)

    def _exec_batch_var_menu(self, global_pos: QPoint) -> None:
        """右键已选多条目：批量操作菜单（标题「已选 N 个变量」）。"""
        if not widget_alive(self) or not widget_alive(self._pw):
            return
        pw = self._pw
        selection = [name for name in pw.curves if name in pw._legend_selection]
        if len(selection) < 2:
            return
        menu = QMenu(pw)
        title = menu.addAction(f"已选 {len(selection)} 个变量")
        title.setEnabled(False)
        act_shelf_copy = menu.addAction("加入暂存区（复制）")
        act_shelf_cut = menu.addAction("剪切到暂存区")
        menu.addSeparator()
        act_solo = menu.addAction("仅显示所选")
        act_show_all = menu.addAction("显示全部变量")
        picked: dict[str, str] = {}
        holder = [menu]
        sub_style = menu.addMenu("样式")
        style_panel = self._build_style_panel_widget(
            selection[0],
            picked,
            holder,
            title_text=f"已选 {len(selection)} 个变量",
        )
        style_act = QWidgetAction(sub_style)
        style_act.setDefaultWidget(style_panel)
        sub_style.addAction(style_act)
        menu.addSeparator()
        act_remove = menu.addAction(f"删除变量（{len(selection)}）")
        act_copy = menu.addAction("复制变量名")
        act_clear = menu.addAction("清除选择")

        chosen = menu.exec(global_pos)
        menu.deleteLater()
        if chosen is None and not picked:
            return
        # 菜单打开期间可能已 reload/删除：分发前重新核对现存曲线
        selection = [name for name in selection if name in pw.curves]
        if picked:
            self._apply_batch_style_picks(selection, picked)
            return
        if not selection:
            return
        main_window = pw.window()
        if chosen is act_shelf_copy:
            self._send_selection_to_shelf(selection, cut=False)
        elif chosen is act_shelf_cut:
            self._send_selection_to_shelf(selection, cut=True)
        elif chosen is act_solo:
            pw._multi_curve_manager._apply_visibility(
                {name: name in selection for name in pw.curves}
            )
        elif chosen is act_show_all:
            pw.show_all_curves()
        elif chosen is act_remove:
            self._delete_selection_with_confirm(selection)
        elif chosen is act_copy:
            QApplication.clipboard().setText("、".join(selection))
        elif chosen is act_clear:
            # 页级清空：返回是否真清到东西，避免「什么都没清却播报」
            clear_page = getattr(pw, "clear_page_selection", None)
            cleared = bool(clear_page()) if callable(clear_page) else False
            if cleared:
                announce_selection(pw, "clear")

    def _apply_batch_style_picks(
        self, names: list[str], picked: dict[str, str]
    ) -> None:
        """批量样式应用：整批抑制逐条 emit，收尾一次刷新 + 一次播报。"""
        pw = self._pw
        mcm = pw._multi_curve_manager
        key, value = next(iter(picked.items()))
        changed = 0
        previous_batch = getattr(pw, "_batch_adding", False)
        pw._batch_adding = True
        try:
            for name in names:
                ci = pw.curves.get(name)
                if ci is None:
                    continue
                if key == "color":
                    if ci.color != value and mcm.apply_curve_color(name, value):
                        changed += 1
                elif key == "draw_mode":
                    if (
                        getattr(ci, "draw_mode", DRAW_MODE_AUTO) != value
                        and mcm.apply_curve_draw_mode(name, value)
                    ):
                        changed += 1
                elif key == "symbol_policy":
                    if (
                        getattr(ci, "symbol_policy", SYMBOL_POLICY_AUTO) != value
                        and mcm.apply_curve_symbol_policy(name, value)
                    ):
                        changed += 1
                elif key == "line_preset":
                    if (
                        getattr(ci, "line_preset", LINE_PRESET_AUTO) != value
                        and mcm.apply_curve_line_preset(name, value)
                    ):
                        changed += 1
        finally:
            pw._batch_adding = previous_batch
        if changed:
            pw.curves_changed.emit()
            main_window = pw.window()
            show = getattr(main_window, "_show_status_message", None)
            if key == "color":
                label = "颜色"
            elif key == "draw_mode":
                label = f"绘制方式：{DRAW_MODE_LABELS[value]}"
            elif key == "line_preset":
                label = f"样式：{LINE_PRESET_LABELS[value]}"
            else:
                label = f"数据点显示：{SYMBOL_POLICY_LABELS[value]}"
            if callable(show):
                show(f"已修改 {changed} 个变量的{label}")

    def _delete_selection_with_confirm(self, names: list[str]) -> None:
        """批量删除：N>1 一次汇总确认；删除后清空选择集。"""
        pw = self._pw
        try:
            from PySide6.QtWidgets import QMessageBox
        except Exception:  # pragma: no cover - 极端导入失败保持可用
            QMessageBox = None
        if len(names) > 1 and QMessageBox is not None:
            reply = QMessageBox.question(
                self,
                "删除变量",
                f"确定要删除所选的 {len(names)} 个变量吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        removed = 0
        for name in list(names):
            if name in pw.curves:
                pw.remove_variable_from_plot(name)
                removed += 1
        self.clear_legend_selection()
        main_window = pw.window()
        if main_window is not None and hasattr(main_window, "layout_manager"):
            main_window.layout_manager.request_mark_stats_refresh()
        show = getattr(main_window, "_show_status_message", None)
        if removed and callable(show):
            show(f"已删除 {removed} 个变量")

    def _send_selection_to_shelf(self, names: list[str], *, cut: bool) -> None:
        """批量入区（菜单入口）：快照后复制入托盘；剪切则延迟删源。"""
        pw = self._pw
        main_window = pw.window()
        # 加载期全入口拒绝（菜单入区与拖入区同口径）
        is_loading = getattr(main_window, "is_data_loading", None)
        if callable(is_loading) and is_loading():
            broadcast_text(main_window, loading_reject_message())
            return
        get_shelf = getattr(main_window, "get_curve_shelf", None)
        if not callable(get_shelf):
            return
        from src.ui.widgets.curve_shelf import ShelfEntry

        dialog = get_shelf()
        origin = self._badge_label_for(pw)
        added: list[str] = []
        updated = 0
        for name in names:
            ci = pw.curves.get(name)
            if ci is None:
                continue
            outcome = dialog.shelf.add_entry(
                ShelfEntry.from_curve_info(
                    name,
                    ci,
                    origin_label=origin,
                    # 菜单剪切同样删源：必须记来源，否则退回入口失效
                    origin_plot=pw if cut else None,
                )
            )
            # 第 6 轮 P2-34：同名覆盖更新计数，与拖入区播报同规格
            if outcome == "updated":
                updated += 1
            added.append(name)
        if not added:
            return
        # 统一走主窗口入口：最小化时用 showNormal 恢复，并同步胶囊开态
        show_shelf = getattr(main_window, "_show_curve_shelf", None)
        if callable(show_shelf):
            show_shelf()
        else:
            dialog.show()
            dialog.raise_()
        if cut:
            def _make_remove(target: Any, var: str):
                def _remove():
                    try:
                        target.remove_variable_from_plot(var)
                    except RuntimeError:
                        logger.debug("批量剪切入区：plot 已销毁，跳过删除")

                return _remove

            for name in added:
                QTimer.singleShot(0, _make_remove(pw, name))

            def _refresh_shelf_after_cut():
                # 排在所有删源 singleShot 之后：入区瞬间的 present 判定失效
                try:
                    dialog.shelf.refresh_entries(list(added))
                except RuntimeError:
                    logger.debug("批量剪切入区：托盘已销毁，跳过状态刷新")

            QTimer.singleShot(0, _refresh_shelf_after_cut)
            self.clear_legend_selection()
        broadcast_transfer(
            main_window,
            "cut" if cut else "shelf_copy",
            added,
            src_label=origin,
            updated=updated,
        )

    def _badge_label_for(self, plot_widget: Any) -> str:
        """反算「页N·图M」徽章（播报/来源标签用；不可用返回空串）。"""
        page = getattr(plot_widget, "page", None)
        host = getattr(self._pw, "window", lambda: None)()
        if page is None or host is None:
            return ""
        try:
            page_index = host.page_index(page)
            plot_index = next(
                (
                    index
                    for index, container in enumerate(page.plot_widgets)
                    if getattr(container, "plot_widget", None) is plot_widget
                ),
                -1,
            )
        except Exception:
            return ""
        if page_index < 0 or plot_index < 0:
            return ""
        return f"页{page_index + 1}·图{plot_index + 1}"


# 光标线抓取带在视觉笔宽之外的两侧外扩量（逻辑像素）。
# pyqtgraph 的命中矩形只由笔宽推导（InfiniteLine._computeBoundingRect：
# 半宽 = max(pen宽, hoverPen宽)/2 + 1），4px 笔宽只有 ±3px，拖动光标极易
# 点空变成平移视图。外扩 2px 后命中带 ±5px（共 10px），绘制线宽仍是 4px。
CURSOR_HIT_PAD_PX = 2


class GrabPaddedInfiniteLine(pg.InfiniteLine):
    """只扩命中区、不改绘制的 InfiniteLine。

    pyqtgraph 没有开放命中宽度参数，其 boundingRect 的垂直方向外扩量固定为
    max(pen宽, hoverPen宽)/2 + 1。这里在 boundingRect 上额外外扩
    ``hit_pad_px``，shape() 默认跟随 boundingRect（QGraphicsItem 默认实现），
    而 paint() 仍用 currentPen 绘制，红线粗细与悬停变红的手感都不变。

    局部坐标下 InfiniteLine 恒沿 x 轴绘制（painter 里做旋转），垂直方向为
    局部 y，故只需在 y 方向外扩；_computeBoundingRect 的 px 换算口径与这里保持
    一致（pixelVectors 的 perpendicular 分量 = 1 个像素对应的局部长度）。
    """

    hit_pad_px = CURSOR_HIT_PAD_PX

    def boundingRect(self):
        rect = super().boundingRect()
        _, ortho = self.pixelVectors(direction=pg.Point(1, 0))
        pad = 0.0 if ortho is None else abs(ortho.y()) * self.hit_pad_px
        if pad <= 0:
            return rect
        return rect.adjusted(0, -pad, 0, pad)


class PlotUIManager(BasePlotManager):
    """负责 UI 初始化和刷新协调的管理器"""

    def __init__(self, plot_widget: Any):
        super().__init__(plot_widget)
        # v5.3: cursor 场景修改护栏标志，阻止 paint event 在 BSP 中间态期间访问场景
        self.pw._is_cursor_modifying_scene = False

    # ========================================================================
    # 公开初始化入口
    # ========================================================================

    def setup_ui(
        self,
        units_dict: dict,
        dataframe: Any,
        time_channels_info: dict | None = None,
        synchronizer: Any = None,
    ) -> None:
        """初始化UI组件和布局（编排方法）

        设置图形布局控件的基本配置和数据结构，
        初始化绘图相关的属性和同步器。

        Args:
            units_dict: 单位字典
            dataframe: 数据框
            time_channels_info: 时间通道信息
            synchronizer: 同步器实例
        """
        pw = self.pw
        pw.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        pw.setAcceptDrops(True)
        pw.units = units_dict
        pw.data = dataframe
        pw.time_channels_info = time_channels_info or {}
        pw.synchronizer = synchronizer

        pw.time_column_name = None
        pw.time_axis_label = "Index"

        pw.x_name = ""
        pw.x_format = ""

        # 游标 X 域的唯一权威源是 AxisManager.cursor_x_domain()（随 factor/offset
        # 现算）。这里曾有一对 pw.xMin/pw.xMax 快照字段：只在 reset_plot 里按当时
        # 的 factor 换算一次，时间修正后再不更新，被 bounds 回退分支读到就会把空
        # plot 的游标钉在旧时间轴上，故删除。

        # 曲线统一管理
        pw.curves = {}
        # P1 legend 多选：纯 Python 真相源（按 plot 存）；高亮是投影，
        # 图例重建时由 update_legend_label 钩子重放 / 修剪。
        pw._legend_selection: set[str] = set()
        pw._legend_sel_anchor: str | None = None
        # 失效曲线占位（方案 B2）：与 curves 物理隔离，仅存轻量 InvalidCurveRef。
        # 不变量：同一变量名绝不同时出现在 curves 与 invalid_curves。
        pw.invalid_curves = {}
        pw._batch_adding = False
        pw.curve_colors = list(DEFAULT_CURVE_COLORS)
        pw.current_color_index = 0
        pw._max_point_density: float = 0.0

        self._setup_header(pw)
        self._setup_plot_area(pw)
        self._setup_axes(pw)
        self._setup_interaction(pw)
        self._init_ui_refresh_coordinator(pw)

        pw.ci.layout.setContentsMargins(0, 0, 10, 5)
        pw.ci.layout.setSpacing(0)
        pw.ci.layout.setRowStretchFactor(1, 1)

        pw.rubberBand = QRubberBand(QRubberBand.Shape.Rectangle, pw)
        pw.origin = QPoint()

    # ========================================================================
    # Header
    # ========================================================================

    def _setup_header(self, pw: Any) -> None:
        """完全修正的顶部文本区域设置方法"""
        header = pg.GraphicsWidget()
        layout = QGraphicsLinearLayout(Qt.Orientation.Horizontal)

        font = QApplication.font()
        fm = QFontMetrics(font)
        base_spacing = fm.horizontalAdvance("-10000.01")
        header.setFixedHeight(fm.height() * 2)
        # 清零默认 margin/spacing（样式默认各 ~12/10px），
        # 保证 legend 右缘可精确对齐 plot 右边界黑线
        layout.setContentsMargins(0, 0, 0, 0)

        # 左侧间距占位：宽度动态对齐 plot 左边界黑线（update_legend_height 维护，
        # 补偿左轴刻度/轴标题占位，随 y 轴宽度变化）
        pw._legend_left_spacer = QGraphicsWidget()
        pw._legend_left_spacer.setMinimumWidth(0)
        pw._legend_left_spacer.setMaximumWidth(0)
        layout.addItem(pw._legend_left_spacer)
        layout.setItemSpacing(0, base_spacing * 0)

        # 右侧间距占位：宽度动态对齐 plot 右边界黑线（update_legend_height 维护）
        pw._legend_right_spacer = QGraphicsWidget()
        pw._legend_right_spacer.setMinimumWidth(0)
        pw._legend_right_spacer.setMaximumWidth(0)

        pw.legend_label = LegendTextBrowser(pw)
        pw.legend_label.setOpenLinks(False)           # 禁止真实导航，只发 anchorClicked
        pw.legend_label.setFrameStyle(QFrame.Shape.NoFrame)
        pw.legend_label.setStyleSheet("""
            QTextBrowser {
                background-color: transparent;
                font-weight: bold;
                border: none;
            }
        """)
        pw.legend_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # 表意保留：右键菜单走 press/release 入口（LegendTextBrowser 方案 B），
        # contextMenuEvent 链路已死（proxy 无焦点不转发），此策略仅声明
        # "legend 不弹 Qt 默认富文本菜单" 的意图
        pw.legend_label.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        # 只允许鼠标点击链接(锚点)，禁止选中普通文本
        pw.legend_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        pw.legend_label.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        pw.legend_label.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        pw.legend_label.document().setDocumentMargin(1)
        pw.legend_label.setPlaceholderText("channel name")
        pw.legend_label.anchorClicked.connect(pw._on_legend_anchor_clicked)
        pw.legend_label.verticalScrollBar().setStyleSheet(LEGEND_SCROLLBAR_QSS)
        # 文档内容变化也触发高度重算（补齐宽度协商之外的触发源）
        pw.legend_label.document().contentsChanged.connect(self.update_legend_height)
        # label 自身宽度变化（任意布局通道引起）也触发高度重算，
        # 补齐 pw.resizeEvent 与宽度协商之间的时序缺口
        pw._legend_last_w = None
        pw.legend_label.installEventFilter(pw)

        proxy_left = QGraphicsProxyWidget()
        proxy_left.setWidget(pw.legend_label)
        layout.addItem(proxy_left)
        layout.setStretchFactor(proxy_left, 1)
        layout.setAlignment(
            proxy_left,
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
        )
        layout.addItem(pw._legend_right_spacer)
        layout.setItemSpacing(1, 0)
        layout.setItemSpacing(2, 0)
        # 供 wheelEvent 判断鼠标是否悬停在 legend 上，
        # 以及 update_legend_height 同步固定高度约束
        pw._legend_proxy = proxy_left

        header.setLayout(layout)
        pw._header_widget = header  # 供 update_legend_height 同步调整 header 高度
        pw.addItem(header, row=0, col=0, colspan=2)

    def update_legend_height(self) -> None:
        """根据内容与当前宽度重算 legend 高度（1 行 ~ LEGEND_MAX_LINES 行）"""
        pw = self.pw
        label = pw.legend_label

        def _recalc():
            # 让文档按当前 viewport 宽度排版后再测量
            doc = label.document()
            doc.setTextWidth(label.viewport().width())
            content_h = doc.size().height()
            # 取富文本排版的真实行高（fontMetrics 兜底），避免行高低估
            line_h = 0.0
            block = doc.firstBlock()
            if block.isValid() and block.layout() and block.layout().lineCount():
                line_h = block.layout().lineAt(0).height()
            if line_h <= 0:
                line_h = float(label.fontMetrics().height())
            min_h = line_h + 4
            max_h = line_h * LEGEND_MAX_LINES + 4
            # 底部留白保证末行文字不与 plot 上边框接触（label 底对齐 header 底）
            legend_h = int(max(min_h, min(content_h + 4, max_h))) + LEGEND_BOTTOM_PAD
            label.setFixedHeight(legend_h)
            # 同步固定 proxy 高度约束（多层防御：内嵌 widget 约束
            # 在图形布局协商中存在失效场景）
            proxy = getattr(pw, "_legend_proxy", None)
            if proxy is not None:
                proxy.setMinimumHeight(legend_h)
                proxy.setMaximumHeight(legend_h)
            # 同步增高 header 容器，避免 legend 溢出入侵绘图区；
            # 下限保持 2 行高（维持单行/空状态的旧版外观）
            header = getattr(pw, "_header_widget", None)
            if header is not None:
                header.setFixedHeight(max(line_h * 2, legend_h))
            label.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
                if content_h > max_h
                else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            # 左侧 spacer 对齐：legend 左缘对齐 plot 左边界黑线
            # （补偿左轴刻度/轴标题占位；宽度随 y 轴刻度位数变化，
            # 由左轴 Resize 的 eventFilter 触发重算）
            left_spacer = getattr(pw, "_legend_left_spacer", None)
            if left_spacer is not None and header is not None and hasattr(pw, "view_box"):
                try:
                    left_inset = (
                        pw.view_box.sceneBoundingRect().left()
                        - header.sceneBoundingRect().left()
                    )
                    if left_inset >= 0 and abs(left_spacer.rect().width() - left_inset) > 0.5:
                        left_spacer.setMinimumWidth(int(left_inset))
                        left_spacer.setMaximumWidth(int(left_inset))
                except Exception:
                    logger.debug("legend 左对齐计算失败", exc_info=True)
            # 右侧 spacer 对齐：滚动条右缘对齐 plot 右边界黑线
            # （补偿 ci 右边距 + 右侧轴占位）
            spacer = getattr(pw, "_legend_right_spacer", None)
            if spacer is not None and header is not None and hasattr(pw, "view_box"):
                try:
                    inset = (
                        header.sceneBoundingRect().right()
                        - pw.view_box.sceneBoundingRect().right()
                    )
                    if inset >= 0 and abs(spacer.rect().width() - inset) > 0.5:
                        spacer.setMinimumWidth(int(inset))
                        spacer.setMaximumWidth(int(inset))
                except Exception:
                    logger.debug("legend 右对齐计算失败", exc_info=True)
            # 强制外层布局立即落定，消除 header 增高未生效的中间态
            try:
                pw.ci.layout.activate()
            except Exception:
                logger.debug("强制布局落定失败", exc_info=True)

        def _safe_recalc():
            # singleShot 延迟执行期间 plot 可能已关闭（C++ 对象销毁），
            # 此时 label.document()/viewport() 等访问抛 RuntimeError，
            # 跳过本次重算即可（对齐 safe_callback 的销毁契约）
            try:
                _recalc()
            except RuntimeError:
                logger.debug(
                    "legend 高度重算跳过：C++ 对象已销毁", exc_info=True,
                )

        QTimer.singleShot(0, _safe_recalc)   # 等 graphics layout 分配完宽度

    # ========================================================================
    # Plot Area
    # ========================================================================

    def _setup_plot_area(self, pw: Any) -> None:
        """配置绘图区域基本属性（包含性能优化配置）"""
        from src.ui.widgets.custom_viewbox import CustomViewBox

        # defaultPadding=0.1：统一 Ctrl+Y（pyqtgraph suggestPadding）与
        # 「自动缩放」（setYRange padding=DEFAULT_PADDING_VAL_Y）两条路径的
        # Y 轴留白，消除切换跳变；X 轴无常开 autoRange，不受影响
        pw.plot_item = pw.addPlot(
            row=1, col=0, colspan=2,
            viewBox=CustomViewBox(defaultPadding=DEFAULT_PADDING_VAL_Y),
        )
        pw.view_box = pw.plot_item.vb
        pw.view_box.plot_widget = pw
        pw._connect_viewbox_signals()
        # 左轴宽度变化（y 轴刻度位数改变）→ 重算 legend 左右对齐
        pw._legend_left_axis = pw.plot_item.getAxis("left")
        pw._legend_left_axis.installEventFilter(pw)
        pw._legend_axis_last_w = None

        pw._is_interacting = False
        # 必须挂 parent：无父 QTimer 在 pw 销毁后仍存活，超时会对已销毁的
        # C++ 对象调用 _end_interaction（RuntimeError / 崩溃）
        pw._interaction_timer = QTimer(pw)
        pw._interaction_timer.setSingleShot(True)
        pw._interaction_timer.timeout.connect(pw._end_interaction)
        pw._is_syncing_range = False

        pw.view_box.setAutoVisible(x=False, y=True)
        pw.plot_item.setTitle(None)
        pw.plot_item.hideButtons()
        pw.plot_item.setClipToView(True)
        pw.plot_item.setDownsampling(mode="peak", auto=True)

        pw.setBackground("w")

        pw.plot_item.getAxis("left").setGrid(255)
        pw.plot_item.getAxis("bottom").setGrid(255)
        pw.plot_item.showGrid(x=True, y=True, alpha=0.1)

        pw.view_box.sigRangeChanged.connect(pw._on_range_changed)
        self.update_x_axis_label(pw)

    def update_x_axis_label(self, pw: Any) -> None:
        """更新 X 轴标签文本"""
        axis = pw.plot_item.getAxis("bottom")
        if DEFAULT_SHOW_X_AXIS_LABEL:
            label = pw.time_axis_label if pw.time_axis_label else "Index"
            axis.setLabel(label)
            axis.showLabel(True)
        else:
            axis.showLabel(False)

    # ========================================================================
    # Axes
    # ========================================================================

    def _setup_axes(self, pw: Any) -> None:
        """配置坐标轴样式和范围"""

        pw.axis_x = pw.plot_item.getAxis("bottom")
        pw.axis_x.setTextPen("black")
        pw.axis_x.setPen(QPen(QColor("black"), 1))
        pw.axis_x.setRange(0, 10)

        pw.axis_y = pw.plot_item.getAxis("left")
        pw.axis_y.enableAutoSIPrefix(False)
        pw.axis_y.setTextPen("black")
        pw.axis_y.setPen(QPen(QColor("black"), 1))

        for pos in ("top", "right"):
            ax = pw.plot_item.getAxis(pos)
            ax.setVisible(True)
            ax.setTicks([])
            ax.setStyle(showValues=False, tickLength=0)
            ax.setPen(QPen(QColor("black"), 1))

        font = QApplication.font()
        fm = QFontMetrics(font)
        pw.axis_y.setWidth(fm.horizontalAdvance("-10000.01"))

        # 轴标签走 HTML font-family，拿不到应用字体栈；用启动时解析到的
        # 中文字体，缺字时才不会又落回平台隐式回退（英文系统上可能是日文
        # 或繁中字形）
        from src.core.font_cache import resolved_chinese_family

        font_family = resolved_chinese_family() or font.family()
        pixel_size = font.pixelSize() + 2
        pw.axis_y.setLabel(
            color="black",
            angle=-90,
            **{
                "font-family": font_family,
                "font-size": f"{pixel_size}px",
                "font-weight": "bold",
            },
        )

    # ========================================================================
    # Interaction
    # ========================================================================

    def _setup_interaction(self, pw: Any) -> None:
        """配置交互元素（光标线、RubberBand、鼠标信号代理）"""
        from PySide6.QtCore import QTimer
        import pyqtgraph as _pg

        pw.vline = GrabPaddedInfiniteLine(
            angle=90,
            movable=False,
            pen=_pg.mkPen((255, 0, 0, 100), width=4),
        )
        pw.vline2 = GrabPaddedInfiniteLine(
            angle=90,
            movable=False,
            pen=_pg.mkPen((255, 0, 0, 100), width=4),
        )
        pw.vline.cursor_index = 0
        pw.vline2.cursor_index = 1
        pw.vline2.setZValue(100)
        pw.vline.setZValue(100)
        pw.cursor_label = _pg.TextItem("", anchor=(1, 1), color="red")
        pw.plot_item.addItem(pw.vline, ignoreBounds=True)
        pw.plot_item.addItem(pw.vline2, ignoreBounds=True)
        pw.plot_item.addItem(pw.cursor_label, ignoreBounds=True)
        pw.vline.setVisible(False)
        pw.vline2.setVisible(False)
        pw.cursor_label.setVisible(False)

        pw.multi_cursor_items = []
        pw.show_values_only = True

        pw._cursor_item_pool = {
            "circles": [],
            "labels": [],
            "x_labels": [],
            "leaders": [],
        }

        pw.proxy = _pg.SignalProxy(
            pw.scene().sigMouseMoved,
            rateLimit=20,
            slot=pw.mouse_moved,
        )
        pw.vline.sigPositionChanged.connect(pw.on_vline_position_changed)
        pw.vline2.sigPositionChanged.connect(pw.on_vline_position_changed)
        pw.setAntialiasing(False)

        pw._last_cursor_update_time = 0
        pw._cursor_update_throttle = 0.016
        pw._adaptive_throttle_enabled = True
        pw._cursor_refresh_timer = QTimer(pw)
        pw._cursor_refresh_timer.setSingleShot(True)
        pw._cursor_refresh_timer.timeout.connect(pw._refresh_cursor_geometry)
        pw._pending_cursor_geometry_update = False

    # ========================================================================
    # UI Refresh Coordinator
    # ========================================================================

    def _init_ui_refresh_coordinator(self, pw: Any) -> None:
        """初始化统一 UI 更新调度器"""
        pw._ui_refresh = UnifiedUpdateScheduler(
            delay_ms=UI_DEBOUNCE_DELAY_MS,
            order=("style", "cursor", "stats"),
            parent=pw,
        )
        pw._ui_refresh.register("style", pw._run_style_refresh)
        pw._ui_refresh.register("cursor", pw._run_cursor_refresh)
        pw._ui_refresh.register("stats", pw._run_stats_refresh)

    def _queue_ui_refresh(
        self,
        pw: Any,
        *,
        style: bool = True,
        cursor: bool = True,
        stats: bool = True,
        immediate: bool = False,
    ) -> None:
        """调度 UI 更新任务（防抖 + 批量更新）"""
        if not hasattr(pw, "_ui_refresh"):
            return
        tasks: list[str] = []
        if style:
            tasks.append("style")
        if cursor:
            tasks.append("cursor")
        if stats:
            tasks.append("stats")
        if not tasks:
            return
        if immediate:
            pw._ui_refresh.run_immediately(*tasks)
        else:
            pw._ui_refresh.schedule(*tasks)

    def _cancel_ui_refresh(self, pw: Any, *tasks: str) -> None:
        """取消已调度的 UI 更新任务"""
        if hasattr(pw, "_ui_refresh"):
            if tasks:
                pw._ui_refresh.cancel(*tasks)
            else:
                pw._ui_refresh.cancel()

    def _run_style_refresh(self, pw: Any) -> None:
        """执行样式刷新"""
        if (
            getattr(pw, "_is_updating_data", False)
            or getattr(pw, "_is_being_destroyed", False)
        ):
            return
        if hasattr(pw, "view_box") and hasattr(pw, "plot_item"):
            self.update_plot_style(pw, pw.view_box, pw.view_box.viewRange(), None)

    def update_plot_style(self, pw: Any, view_box, range, rect=None):
        """更新绘图样式 - 逐曲线判定 symbol 开关（per-curve，方案 A）

        每条曲线各自估算可见点数：x宽度 × 该曲线采样密度
        < XRANGE_THRESHOLD_FOR_SYMBOLS 时显示 symbols（细线+点），否则
        粗线无 symbol。稀疏曲线不再被同图密集曲线"连坐"。
        ``symbol_policy`` 非 'auto' 时以用户显式钉住值覆盖判定。
        与 factor 无关：无论 x 轴是行号（CSV）还是秒（MDF/BLF），
        阈值语义统一为"可见点数 < 100"。
        pen/symbol 缓存短路在 ``_apply_curve_style`` 内完成，无变化曲线零开销。
        """
        try:
            _mw = pw.window()

            if getattr(pw, '_is_updating_data', False) or getattr(pw, '_is_being_destroyed', False):
                return

            # 重载期间禁止样式刷新，避免中间状态触发错误的 symbol 切换
            if _mw and getattr(_mw, '_is_loading_new_data', False):
                return

            if not hasattr(pw, 'plot_item'):
                return

            is_interacting = getattr(pw, '_is_interacting', False)
            if is_interacting:
                return

            x_min, x_max = range[0]
            x_width = x_max - x_min

            symbol_on = 0
            for var_name, ci in pw.curves.items():
                show_i = resolve_symbol_show(ci, x_width)
                symbol_on += 1 if show_i else 0
                pw._apply_curve_style(var_name, show_i)

            logger.debug(
                "[PLOT_STYLE] x_width=%.4g, curves=%d, symbol_on=%d",
                x_width, len(pw.curves), symbol_on,
            )

        except Exception as e:
            logger.error("更新绘图样式时出错: %s", e)

    def _run_cursor_refresh(self, pw: Any) -> None:
        """执行光标刷新"""
        if getattr(pw, "_is_interacting", False):
            return
        if hasattr(pw, "vline") and pw.vline.isVisible():
            try:
                pw.update_cursor_label()
            except Exception:
                logger.debug("更新光标标签失败", exc_info=True)

    def _run_stats_refresh(self, pw: Any) -> None:
        """执行统计刷新"""
        # window() 非 MainWindow 时（如独立顶层/测试环境）没有 layout_manager，
        # 与 plot_widget._handle_variable_drop 的防御模式对齐，否则每次 [stats]
        # 调度回调都会抛 AttributeError。
        main_window = pw.window()
        if main_window is not None and hasattr(main_window, "layout_manager"):
            main_window.layout_manager.request_mark_stats_refresh(immediate=True)
