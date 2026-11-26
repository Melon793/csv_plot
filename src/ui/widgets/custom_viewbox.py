"""
CustomViewBox —— 信号化自定义 ViewBox

将原本通过 plot_widget.window() 直接访问 MainWindow 的操作
替换为 PyQt 信号，由 DraggableGraphicsLayoutWidget 负责连接。
"""

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QApplication, QMenu, QWidget
import pyqtgraph as pg

from src.core.config import (
    TRACKPAD_MODE_IDLE_MS,
    TRACKPAD_SCAN_UNITS_PER_SCREEN,
)
from src.ui import theme

# 右键菜单完全自建：保留一个本 ViewBox 的缓存 QMenu，每次右键清空重排。
# pyqtgraph 的 PlotItem / GraphicsScene 会在 getMenu 返回后往菜单尾部追加
# 「Plot Options」与「Export...」，这里在首次取菜单时摘走原生导出项并停用
# PlotItem 菜单，能力保留（scene.showExportDialog），只是入口改由我们排布。
ZH_JUMP_TO_DATA = "跳转至变量数值表"
ZH_AUTO_Y_IN_X = "按 X 范围调节 Y 轴"
ZH_SCAN_ZOOM = "扫选放大（拖动选择 X 区间）"
ZH_CURSOR_MENU = "游标选项"
ZH_SHOW_CURSOR_VALUE = "显示游标数值"
ZH_HIDE_CURSOR_VALUE = "隐藏游标数值"
ZH_COPY_NAME = "复制变量名"
ZH_VAR_EDITOR = "绘图变量编辑器"
ZH_ADJUST_HEIGHT = "调整高度"
ZH_ARRANGE_LAYOUT = "重排子图"
ZH_RESET_ALL_HEIGHT = "全部重置为 100%"
ZH_EXPORT = "导出图片…"
ZH_CLEAR_PLOT = "清除当前子图"

# 点击 / 扫选的判定线：取平台「开始拖动」距离（QStyleHints.startDragDistance，
# macOS=10px），8px 兜底。刻意**不随视宽缩放** —— 曾按「视宽 3%」定线，
# 2560px 宽的全屏图要 77px 才算扫选，几十像素的认真扫掠被判成点击。
SCAN_MIN_PX = 8

# 候选态（未过判定线）的虚线 alpha：过线前只画两条细虚线、不填充，
# 松手前就能看出「这次还不算数」——「虚线/实线」与「弹菜单/缩放」严格一一对应
SCAN_CANDIDATE_ALPHA = 160

# 游标模式显示文案：键是跨模块内部标识符（cursor_sync_manager 的模式分发、
# file_loader_manager 的重载恢复都按它比对），只改显示、绝不可改键值。
ZH_CURSOR_MODE_LABELS = {
    "1 free cursor": "单自由游标",
    "1 anchored cursor": "单固定游标",
    "2 anchored cursor": "双固定游标",
    "off": "关闭游标",
}


class CustomViewBoxSignals(QObject):
    """CustomViewBox 发出的信号集合 —— 用于解耦与 MainWindow 的直接依赖"""

    def __init__(self, parent=None):
        super().__init__(parent)

    request_jump_to_data = Signal(object, object)  # plot_widget, context_x
    request_clear_plot = Signal(object)  # plot_widget
    request_auto_y = Signal(object)  # plot_widget
    request_set_cursor_mode = Signal(
        str, object, object
    )  # mode, plot_widget, context_x
    request_show_cursor_value = Signal(object)  # plot_widget
    request_hide_cursor_value = Signal(object)  # plot_widget
    request_set_row_height = Signal(int, object)  # percentage, plot_widget
    request_set_all_row_height = Signal(int)  # percentage
    request_copy_name = Signal(object)  # plot_widget
    request_variable_editor = Signal(object)  # plot_widget
    request_arrange_layout = Signal(object)  # plot_widget


class CustomViewBox(pg.ViewBox):
    """
    自定义视图框 —— 信号化版本

    通过信号与上层的 MainWindow / PlotContext 通信，
    不再直接访问 plot_widget.window()。
    """

    signals: CustomViewBoxSignals

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signals = CustomViewBoxSignals(parent=self)
        self.context_x: float | None = None
        self.plot_widget = None
        self._context_menu: QMenu | None = None
        # 右键扫选会话：_scan_active 在 Esc 取消后仍要吞掉剩余拖拽事件，
        # 所以不能只靠 _scan_band 判活
        self._scan_active = False
        self._scan_armed = False
        self._scan_drag_button = None
        self._scan_cancelled_button = None
        self._scan_band = None
        self._scan_band_committed = None
        self._scan_filter_view = None
        self._scan_cursor_saved = None
        # Shift + 双指横滑刷选：Windows 无 ScrollPhase，用最后一次横滑后的
        # 静默超时代替“松手”；预览带与鼠标扫选共用，避免两套视觉语义。
        self._scroll_scan_active = False
        self._scroll_scan_anchor_x = None
        self._scroll_scan_target_x = None
        self._scroll_scan_dx = 0.0
        self._scroll_scan_timer = QTimer(self)
        self._scroll_scan_timer.setSingleShot(True)
        self._scroll_scan_timer.timeout.connect(self._finish_scroll_scan)

    # ================================================================
    # 右键拖拽扫选 X 区间
    # ================================================================

    def mouseDragEvent(self, ev, axis=None):
        """左/中键与轴区拖拽沿用父类；右键或 armed 左键在绘图区内扫选 X。

        pyqtgraph 0.14 的默认右键拖拽是双轴 ``scaleBy``（未写进文档，且会静默
        关掉两轴 autoRange）。整个默认行为在这里替换为「按下点 → 当前点」的
        预览带，松手后只缩 X、Y 一点不碰、autoRange 标记不动。

        松手时的点击/扫选判定只看净位移：不足平台拖动距离按点击补弹菜单，
        其余一律扫选 —— 判定线固定为像素、不随视宽放大。预览带的
        「虚线无填充 / 实线填充」与这个判定严格同源，松手前即可预知结果。
        """
        self.cancel_scroll_scan()
        button = ev.button()
        if button == self._scan_cancelled_button:
            ev.accept()
            if ev.isStart():
                self._scan_cancelled_button = None
            else:
                if ev.isFinish() or not (ev.buttons() & button):
                    self._scan_cancelled_button = None
                return
        armed_left = self._scan_armed and button == Qt.MouseButton.LeftButton
        if (
            axis is not None  # 轴区拖拽（AxisItem 转交，axis=0/1）
            or (button != Qt.MouseButton.RightButton and not armed_left)
            # 锁 X 时不写 X（生产上锁定时 x 已 setMouseEnabled(False)，父类
            # 只剩 Y 缩放），与 D13「锁定时框选只写 Y」同口径
            or getattr(self.plot_widget, "_x_mouse_locked", False)
        ):
            return super().mouseDragEvent(ev, axis=axis)

        ev.accept()  # 必须接受：GraphicsScene 靠它把后续 move/finish 都派给本视图
        if ev.isStart():
            self._begin_scan(button)

        if not self._scan_active:
            return  # Esc 已取消：剩余拖拽事件直接吞掉，不再处理

        # 松手事件可能被模态/切窗吃掉：中途发现拖动键已不在按下集合里就收摊，
        # 否则预览带与十字光标会一直留到下一次拖拽
        if not ev.isFinish() and not (ev.buttons() & button):
            self._teardown_scan(keep_armed=armed_left)
            return

        try:
            self._update_scan_band(ev)
        except Exception:
            self._teardown_scan()
            raise

        if ev.isFinish():
            self._end_scan(ev)

    def _arm_scan(self) -> None:
        if getattr(self.plot_widget, "_x_mouse_locked", False):
            return
        self.cancel_scroll_scan()
        self._scan_armed = True
        self._install_scan_key_filter()
        self._set_scan_cursor(Qt.CursorShape.CrossCursor)

    def _begin_scan(self, button) -> None:
        self.cancel_scroll_scan()
        self._scan_active = True
        self._scan_drag_button = button
        self._install_scan_key_filter()
        self._set_scan_cursor(Qt.CursorShape.CrossCursor)

    def _scan_commit_px(self) -> float:
        """点击/扫选的判定线（逻辑像素；视图不缩放时与 scene 坐标同尺度）。"""
        app = QApplication.instance()
        platform_px = (
            app.styleHints().startDragDistance() if app is not None else SCAN_MIN_PX
        )
        return max(SCAN_MIN_PX, platform_px)

    @property
    def scroll_scan_active(self) -> bool:
        return self._scroll_scan_active

    def _update_scan_band(self, ev) -> None:
        """按「按下点 → 当前点」建/更新预览带（数据坐标，无 padding）。"""
        down = ev.buttonDownScenePos(ev.button())
        x0 = self.mapSceneToView(down).x()
        x1 = self.mapSceneToView(ev.scenePos()).x()
        committed = abs(ev.scenePos().x() - down.x()) >= self._scan_commit_px()
        self._set_scan_band(x0, x1, committed)

    def _set_scan_band(self, x0: float, x1: float, committed: bool) -> None:
        """鼠标拖扫与 Shift+横滑刷选共用的预览带。"""
        band = self._scan_band
        if band is None:
            plot_item = getattr(self.plot_widget, "plot_item", None)
            if plot_item is None:
                return
            band = pg.LinearRegionItem(values=(x0, x1), movable=False)
            # 预览带只是视觉反馈，任何鼠标键都不得被它抢走：否则拖拽中的
            # 悬停/点击会落到区域上，松手后右键菜单也可能被它接管
            band.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            for line in band.lines:
                line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._scan_band = band
            self._scan_band_committed = None
            # ignoreBounds：预览带横跨全高，参与 auto-range 会把 Y 撑飞
            plot_item.addItem(band, ignoreBounds=True)
        else:
            band.setRegion((x0, x1))
        self._style_scan_band(band, committed)

    def _remove_scan_band(self) -> None:
        self._scan_band_committed = None
        band = self._scan_band
        self._scan_band = None
        if band is None:
            return
        plot_item = getattr(self.plot_widget, "plot_item", None)
        if plot_item is not None:
            plot_item.removeItem(band)
        else:
            # plot_item 已先一步没了：直接从场景摘，别留一张幽灵带
            scene = band.scene()
            if scene is not None:
                band.setParentItem(None)
                scene.removeItem(band)
        band.deleteLater()

    def _style_scan_band(self, band, committed: bool) -> None:
        """两态预览：未过判定线只画 accent 虚线、不填充；过线才有 tint 填充。"""
        if committed == self._scan_band_committed:
            return
        self._scan_band_committed = committed
        if committed:
            pen = pg.mkPen(theme.accent_qcolor(), width=1)
            band.setBrush(theme.tint_qcolor())
        else:
            pen = pg.mkPen(
                theme.accent_qcolor(SCAN_CANDIDATE_ALPHA),
                width=1,
                style=Qt.PenStyle.DashLine,
            )
            band.setBrush(None)  # mkBrush(None) == NoBrush
        for line in band.lines:
            line.setPen(pen)
            line.setHoverPen(pen)
        band.update()

    def _end_scan(self, ev) -> None:
        """松手：净位移够判定线 → 缩 X；armed 左键误触保持模式，右键误触弹菜单。"""
        down = ev.buttonDownScenePos(ev.button())
        pos = ev.scenePos()
        commit_px = self._scan_commit_px()
        x0 = self.mapSceneToView(down).x()
        x1 = self.mapSceneToView(pos).x()
        committed = abs(pos.x() - down.x()) >= commit_px
        keep_armed = ev.button() == Qt.MouseButton.LeftButton and not committed

        try:
            if committed:
                self._apply_scan_range(min(x0, x1), max(x0, x1))
            elif ev.button() == Qt.MouseButton.RightButton and self.menuEnabled():
                # 拖动已消费掉 mouseClickEvent → 手动补齐「点一下出菜单」
                self.raiseContextMenu(ev)
        finally:
            self._teardown_scan(keep_armed=keep_armed)

    def _apply_scan_range(self, xmin: float, xmax: float) -> None:
        """写 X 范围：走 AxisManager 入口（XLink 时写源视图，由原生广播到整组）。"""
        apply = getattr(self.plot_widget, "set_xrange_with_link_handling", None)
        if callable(apply):
            linked = self.linkedView(0)
            if linked is not None:
                local_range = self.viewRange()[0]
                source_range = linked.viewRange()[0]
                if max(abs(a - b) for a, b in zip(local_range, source_range)) > 1e-9:
                    # 同列错位 XLink 走像素对齐：子图范围是源范围的局部窗口。
                    # 要让广播后的子图恰好等于所选区间，需按两侧像素宽反推源范围；
                    # 低重叠仍走整段复制，不能把列间距折进数据坐标。
                    local_rect = self.screenGeometry()
                    source_rect = linked.screenGeometry()
                    local_width = max(local_rect.width(), 1)
                    source_width = max(source_rect.width(), 1)
                    overlap = min(local_rect.right(), source_rect.right()) - max(
                        local_rect.left(), source_rect.left()
                    )
                    if overlap >= min(local_width / 3, source_width / 3):
                        source_span = (xmax - xmin) * source_width / local_width
                        source_left = xmin - (
                            (local_rect.left() - source_rect.left())
                            / source_width
                            * source_span
                        )
                        xmin = source_left
                        xmax = source_left + source_span
            apply(xmin, xmax)
            return
        self.enableAutoRange(x=False)
        self.setXRange(xmin, xmax, padding=0)

    def feed_scroll_scan(self, dx: float, scene_pos) -> bool:
        """Shift+横滑：以光标为固定锚点，横滑量决定另一条边。"""
        if (
            getattr(self.plot_widget, "_x_mouse_locked", False)
            or getattr(self.plot_widget, "_is_updating_data", False)
            or getattr(self.plot_widget, "_is_being_destroyed", False)
        ):
            self.cancel_scroll_scan()
            return False
        if self._scan_active or self._scan_armed:
            return False

        if not self._scroll_scan_active:
            self._scroll_scan_active = True
            self._scroll_scan_anchor_x = self.mapSceneToView(scene_pos).x()
            self._scroll_scan_target_x = self._scroll_scan_anchor_x
            self._scroll_scan_dx = 0.0
            self._install_scan_key_filter()
            self._set_scan_cursor(Qt.CursorShape.CrossCursor)

        self._scroll_scan_dx += dx
        left, right = self.viewRange()[0]
        span = max(right - left, 1e-12)
        target = self._scroll_scan_anchor_x + (
            self._scroll_scan_dx / TRACKPAD_SCAN_UNITS_PER_SCREEN * span
        )
        target = max(left, min(right, target))
        self._scroll_scan_target_x = target
        pixel_delta = (
            abs(target - self._scroll_scan_anchor_x)
            / span
            * max(self.sceneBoundingRect().width(), 1.0)
        )
        self._set_scan_band(
            self._scroll_scan_anchor_x,
            target,
            pixel_delta >= self._scan_commit_px(),
        )
        # 这里故意用 debounce：横滑事件本身就是连续流，最后一次事件后的
        # 静默才代表手指离开；不能用 16ms 单发定时器提前提交。
        self._scroll_scan_timer.start(TRACKPAD_MODE_IDLE_MS)
        return True

    def _finish_scroll_scan(self, commit: bool = True) -> None:
        if not self._scroll_scan_active:
            return
        self._scroll_scan_timer.stop()
        self._scroll_scan_active = False
        anchor = self._scroll_scan_anchor_x
        target = self._scroll_scan_target_x
        committed = bool(self._scan_band_committed)
        self._scroll_scan_anchor_x = None
        self._scroll_scan_target_x = None
        self._scroll_scan_dx = 0.0
        self._remove_scan_band()
        self._remove_scan_key_filter()
        self._set_scan_cursor(None)

        blocked = (
            getattr(self.plot_widget, "_x_mouse_locked", False)
            or getattr(self.plot_widget, "_is_updating_data", False)
            or getattr(self.plot_widget, "_is_being_destroyed", False)
        )
        if (
            commit
            and committed
            and not blocked
            and anchor is not None
            and target is not None
        ):
            self._apply_scan_range(min(anchor, target), max(anchor, target))

    def cancel_scroll_scan(self) -> None:
        self._finish_scroll_scan(commit=False)

    def _cancel_scan(self) -> None:
        if self._scan_active:
            self._scan_cancelled_button = self._scan_drag_button
        self._teardown_scan()

    def _teardown_scan(self, keep_armed: bool = False) -> None:
        self._scan_active = False
        self._scan_drag_button = None
        self._remove_scan_band()
        if keep_armed:
            return
        self._scan_armed = False
        self._remove_scan_key_filter()
        self._set_scan_cursor(None)

    def eventFilter(self, obj, event):
        """扫选期间监听视图的 Esc：取消并撤掉预览带。"""
        if (
            (self._scan_active or self._scan_armed or self._scroll_scan_active)
            and obj is self._scan_filter_view
            and event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_Escape
        ):
            self.cancel_scroll_scan()
            self._cancel_scan()
            return True
        return super().eventFilter(obj, event)

    def _install_scan_key_filter(self) -> None:
        view = self.plot_widget
        if view is None or self._scan_filter_view is not None:
            return
        view.installEventFilter(self)
        self._scan_filter_view = view

    def _remove_scan_key_filter(self) -> None:
        view = self._scan_filter_view
        if view is None:
            return
        self._scan_filter_view = None
        try:
            view.removeEventFilter(self)
        except RuntimeError:
            pass  # C++ 侧已销毁

    def _set_scan_cursor(self, shape) -> None:
        """shape=None 表示还原。用 WA_SetCursor 区分「原本没设光标」的常态，
        没设过就 unsetCursor（不覆盖应用/主题光标，与视图其它路径同口径）。"""
        view = self.plot_widget
        viewport = view.viewport() if view is not None else None
        if viewport is None:
            return
        if shape is None:
            saved = self._scan_cursor_saved
            self._scan_cursor_saved = None
            if saved is None:
                return
            had_cursor, prev = saved
            if had_cursor:
                viewport.setCursor(prev)
            else:
                viewport.unsetCursor()
            return
        if self._scan_cursor_saved is None:
            self._scan_cursor_saved = (
                viewport.testAttribute(Qt.WidgetAttribute.WA_SetCursor),
                viewport.cursor(),
            )
        viewport.setCursor(shape)

    def getMenu(self, ev):
        view_pos = self.mapSceneToView(ev.scenePos())
        self.context_x = view_pos.x()

        menu = self._ensure_context_menu()
        self._clear_context_menu(menu)
        self._detach_native_menus()

        scan_act = QAction(ZH_SCAN_ZOOM, menu)
        scan_act.triggered.connect(self._arm_scan)
        scan_act.setEnabled(not getattr(self.plot_widget, "_x_mouse_locked", False))
        menu.addAction(scan_act)

        auto_y_act = QAction(ZH_AUTO_Y_IN_X, menu)
        auto_y_act.triggered.connect(self._emit_auto_y)
        menu.addAction(auto_y_act)

        menu.addMenu(self._build_cursor_menu(menu))
        menu.addSeparator()

        jump_act = QAction(ZH_JUMP_TO_DATA, menu)
        jump_act.triggered.connect(self._emit_jump_to_data)
        menu.addAction(jump_act)

        copy_act = QAction(ZH_COPY_NAME, menu)
        copy_act.triggered.connect(
            lambda: self.signals.request_copy_name.emit(self.plot_widget)
        )
        copy_act.setEnabled(self._has_data())
        menu.addAction(copy_act)

        editor_act = QAction(ZH_VAR_EDITOR, menu)
        editor_act.triggered.connect(
            lambda: self.signals.request_variable_editor.emit(self.plot_widget)
        )
        menu.addAction(editor_act)

        menu.addSeparator()
        menu.addMenu(self._build_adjust_height_menu(menu))

        arrange_act = QAction(ZH_ARRANGE_LAYOUT, menu)
        arrange_act.triggered.connect(
            lambda: self.signals.request_arrange_layout.emit(self.plot_widget)
        )
        menu.addAction(arrange_act)

        menu.addSeparator()
        export_act = self._export_action()
        if export_act is not None:
            menu.addAction(export_act)
            menu.addSeparator()

        clear_act = QAction(ZH_CLEAR_PLOT, menu)
        clear_act.triggered.connect(
            lambda: self.signals.request_clear_plot.emit(self.plot_widget)
        )
        menu.addAction(clear_act)
        return menu

    def _ensure_context_menu(self) -> QMenu:
        menu = self._context_menu
        if menu is not None:
            try:
                menu.actions()
            except RuntimeError:
                menu = None
        if menu is None:
            parent = self.plot_widget if isinstance(self.plot_widget, QWidget) else None
            menu = QMenu(parent)
            self._context_menu = menu
        return menu

    def _clear_context_menu(self, menu: QMenu) -> None:
        """清空上一轮的项。

        ``QMenu.clear()`` 只删自己拥有的 action，不删子 QMenu（子菜单的
        parent 仍是这个缓存菜单），所以子菜单必须单独 deleteLater，否则每次
        右键都会留下一整棵孤儿子树。
        """
        for act in menu.actions():
            submenu = act.menu()
            if submenu is not None:
                submenu.deleteLater()
        menu.clear()

    def _detach_native_menus(self):
        """停用 pyqtgraph 自动追加到菜单尾部的 PlotItem 菜单。

        只关入口：``ctrlMenu`` 与其中的动作原样保留，将来要恢复只需改回。
        Scene 的「Export...」由 ``_export_action`` 接管。

        必须显式传 ``enableViewBoxMenu=None``：``_end_scan`` 靠
        ``self.menuEnabled()`` 决定短拖是否补弹菜单，ViewBox 菜单本身不能关。
        """
        plot_item = getattr(self.plot_widget, "plot_item", None)
        if plot_item is None:
            return
        try:
            if plot_item.menuEnabled():
                plot_item.setMenuEnabled(False, enableViewBoxMenu=None)
        except (AttributeError, RuntimeError):
            pass

    def _export_action(self):
        """取 Scene 原生的导出动作，把它从自动追加列表里摘出来自用。

        首次调用后清空 ``scene.contextMenu``，之后每次右键复用同一个动作；
        导出实现仍是 ``scene.showExportDialog``（入口换了，能力不动）。
        """
        scene = self.scene()
        if scene is None:
            return None
        act = getattr(scene, "_csv_plot_export_action", None)
        if act is None:
            acts = list(getattr(scene, "contextMenu", None) or [])
            # 只认 Scene 自建的那个动作；列表被第三方改写时不误改别人的项
            if not acts or acts[0].parent() is not scene:
                return None
            act = acts[0]
            scene.contextMenu = []
            scene._csv_plot_export_action = act
        act.setText(ZH_EXPORT)
        return act

    def _build_cursor_menu(self, parent: QMenu) -> QMenu:
        cursor_enabled = self._get_cursor_enabled()
        current_mode = self._get_current_cursor_mode()

        cursor_menu = QMenu(ZH_CURSOR_MENU, parent)
        cursor_group = QActionGroup(cursor_menu)
        cursor_group.setExclusive(True)

        # 添加三个正常模式选项（mode 为内部标识符，仅显示文案中文化）
        for mode in ["1 free cursor", "1 anchored cursor", "2 anchored cursor"]:
            mode_act = QAction(ZH_CURSOR_MODE_LABELS[mode], cursor_menu)
            mode_act.setCheckable(True)
            # 选中逻辑：光标开启时检查是否匹配当前模式，光标关闭时不选中
            mode_act.setChecked(cursor_enabled and mode == current_mode)
            mode_act.triggered.connect(
                lambda checked, m=mode: self.signals.request_set_cursor_mode.emit(
                    m, self.plot_widget, self.context_x
                )
            )
            cursor_group.addAction(mode_act)
            cursor_menu.addAction(mode_act)

        # 添加 "off" 选项
        off_act = QAction(ZH_CURSOR_MODE_LABELS["off"], cursor_menu)
        off_act.setCheckable(True)
        off_act.setChecked(current_mode == "off" or not cursor_enabled)
        off_act.triggered.connect(
            lambda checked: self.signals.request_set_cursor_mode.emit(
                "off", self.plot_widget, self.context_x
            )
        )
        cursor_group.addAction(off_act)
        cursor_menu.addAction(off_act)

        # 数值开关与上面四个模式正交（任一已开启游标模式下都能单独隐藏数值）：
        # 分隔线与互斥组分隔，且不加入 QActionGroup——加了勾选态会被误读成
        # 第 5 种模式。文案随状态翻转，表示“点击后做什么”。
        cursor_menu.addSeparator()
        values_hidden = self._get_cursor_values_hidden()
        if values_hidden:
            cursor_value_act = QAction(ZH_SHOW_CURSOR_VALUE, cursor_menu)
            cursor_value_act.triggered.connect(
                lambda: self.signals.request_show_cursor_value.emit(self.plot_widget)
            )
        else:
            cursor_value_act = QAction(ZH_HIDE_CURSOR_VALUE, cursor_menu)
            cursor_value_act.triggered.connect(
                lambda: self.signals.request_hide_cursor_value.emit(self.plot_widget)
            )
        cursor_value_act.setEnabled(cursor_enabled)
        cursor_menu.addAction(cursor_value_act)
        return cursor_menu

    def _build_adjust_height_menu(self, parent: QMenu) -> QMenu:
        row = self._get_plot_row_index()
        current_pct = self._get_current_row_height(row)
        adjust_height_menu = QMenu(ZH_ADJUST_HEIGHT, parent)

        percentages = [25, 50, 75, 100, 125, 150, 200, 250, 300, 400]
        for pct in percentages:
            label = f"● {pct}%" if pct == current_pct else f"  {pct}%"
            act = QAction(label, adjust_height_menu)
            act.triggered.connect(
                lambda checked, p=pct: self.signals.request_set_row_height.emit(
                    p, self.plot_widget
                )
            )
            adjust_height_menu.addAction(act)

        adjust_height_menu.addSeparator()
        reset_act = QAction(ZH_RESET_ALL_HEIGHT, adjust_height_menu)
        reset_act.triggered.connect(
            lambda: self.signals.request_set_all_row_height.emit(100)
        )
        adjust_height_menu.addAction(reset_act)
        return adjust_height_menu

    def _emit_jump_to_data(self):
        self.signals.request_jump_to_data.emit(self.plot_widget, self.context_x)

    def _emit_auto_y(self):
        self.signals.request_auto_y.emit(self.plot_widget)

    def _get_cursor_enabled(self) -> bool:
        if self.plot_widget and hasattr(self.plot_widget, "plot_context"):
            return self.plot_widget.plot_context.is_cursor_enabled()
        return False

    def _get_current_cursor_mode(self) -> str:
        if self.plot_widget and hasattr(self.plot_widget, "plot_context"):
            return self.plot_widget.plot_context.cursor_mode
        return "1 free cursor"

    def _get_cursor_values_hidden(self) -> bool:
        if self.plot_widget and hasattr(self.plot_widget, "plot_context"):
            return self.plot_widget.plot_context.cursor_values_hidden
        return False

    def _get_current_row_height(self, row: int) -> int:
        if self.plot_widget and hasattr(self.plot_widget, "plot_context"):
            return self.plot_widget.plot_context.get_row_height(row)
        return 100

    def _has_data(self) -> bool:
        if not self.plot_widget:
            return False
        has_single = getattr(self.plot_widget, "curve", None) is not None and bool(
            getattr(self.plot_widget, "y_name", "")
        )
        has_multi = bool(getattr(self.plot_widget, "curves", {}))
        return has_single or has_multi

    def _get_plot_row_index(self) -> int:
        if not self.plot_widget or not hasattr(self.plot_widget, "plot_context"):
            return 0
        ctx = self.plot_widget.plot_context
        # 旧替身 ctx 没有 grid_cols：退回容量列数（单页建满时两者相等）
        ncols = getattr(ctx, "grid_cols", 0) or ctx._plot_col_max_default
        for idx, container in enumerate(ctx.plot_widgets):
            if container.plot_widget is self.plot_widget:
                row, _ = divmod(idx, ncols)
                return row
        return 0
