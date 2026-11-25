"""
CustomViewBox —— 信号化自定义 ViewBox

将原本通过 plot_widget.window() 直接访问 MainWindow 的操作
替换为 PyQt 信号，由 DraggableGraphicsLayoutWidget 负责连接。
"""

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMenu
import pyqtgraph as pg

from src.ui import theme

# 右键菜单中文文案：创建与「判重 / 移除旧项」比对必须共用这些常量——
# getMenu 返回的是 pyqtgraph 缓存的同一个 QMenu，两处文案不一致会导致
# 每次右键都重复插入一份（菜单项成对翻倍）。
ZH_JUMP_TO_DATA = "跳转至变量数值表"
ZH_AUTO_Y_IN_X = "按 X 范围调节 Y 轴"
ZH_CURSOR_MENU = "游标选项"
ZH_SHOW_CURSOR_VALUE = "显示游标数值"
ZH_HIDE_CURSOR_VALUE = "隐藏游标数值"
ZH_COPY_NAME = "复制变量名"
ZH_VAR_EDITOR = "绘图变量编辑器"
ZH_ADJUST_HEIGHT = "调整高度"
ZH_ARRANGE_LAYOUT = "重排子图"
ZH_RESET_ALL_HEIGHT = "全部重置为 100%"
ZH_CLEAR_PLOT = "清除当前子图"

# 右键拖拽扫选的最小有效宽度：8 px 与视宽 3% 取大。
# GraphicsScene 的拖拽认判是 moveDistance=5px（按住超 0.5s 后 1px 也算拖动），
# 5~8px 的抖动会走到「拖动已开始但不该缩放」的中间态 —— 这段按误触处理，
# 松手时补弹右键菜单，保证「点一下 = 菜单」的手感不变。
SCAN_MIN_PX = 8
SCAN_MIN_WIDTH_RATIO = 0.03

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
        # 右键扫选会话：_scan_active 在 Esc 取消后仍要吞掉剩余拖拽事件，
        # 所以不能只靠 _scan_band 判活
        self._scan_active = False
        self._scan_band = None
        self._scan_filter_view = None
        self._scan_cursor_saved = None

    # ================================================================
    # 右键拖拽扫选 X 区间
    # ================================================================

    def mouseDragEvent(self, ev, axis=None):
        """左/中键与轴区拖拽沿用父类；右键在绘图区内拖拽 = 扫选 X 区间。

        pyqtgraph 0.14 的默认右键拖拽是双轴 ``scaleBy``（未写进文档，且会静默
        关掉两轴 autoRange）。整个默认行为在这里替换为「按下点 → 当前点」的
        预览带，松手后只缩 X、Y 一点不碰、autoRange 标记不动。
        """
        if (
            axis is not None  # 轴区拖拽（AxisItem 转交，axis=0/1）
            or ev.button() != Qt.MouseButton.RightButton
            # 锁 X 时不写 X（生产上锁定时 x 已 setMouseEnabled(False)，父类
            # 只剩 Y 缩放），与 D13「锁定时框选只写 Y」同口径
            or getattr(self.plot_widget, "_x_mouse_locked", False)
        ):
            return super().mouseDragEvent(ev, axis=axis)

        ev.accept()  # 必须接受：GraphicsScene 靠它把后续 move/finish 都派给本视图
        if ev.isStart():
            self._begin_scan()

        if not self._scan_active:
            return  # Esc 已取消：剩余拖拽事件直接吞掉，不再处理

        # 松手事件可能被模态/切窗吃掉：中途发现右键已不在按下集合里就收摊，
        # 否则预览带与十字光标会一直留到下一次拖拽
        if not ev.isFinish() and not (
            ev.buttons() & Qt.MouseButton.RightButton
        ):
            self._teardown_scan()
            return

        try:
            self._update_scan_band(ev)
        except Exception:
            self._teardown_scan()
            raise

        if ev.isFinish():
            self._end_scan(ev)

    def _begin_scan(self) -> None:
        self._scan_active = True
        self._install_scan_key_filter()
        self._set_scan_cursor(Qt.CursorShape.CrossCursor)

    def _update_scan_band(self, ev) -> None:
        """按「按下点 → 当前点」建/更新预览带（数据坐标，无 padding）。"""
        x0 = self.mapSceneToView(
            ev.buttonDownScenePos(Qt.MouseButton.RightButton)
        ).x()
        x1 = self.mapSceneToView(ev.scenePos()).x()

        band = self._scan_band
        if band is None:
            plot_item = getattr(self.plot_widget, "plot_item", None)
            if plot_item is None:
                return
            band = pg.LinearRegionItem(
                values=(x0, x1), movable=False, brush=theme.tint_qcolor()
            )
            pen = pg.mkPen(theme.accent_qcolor(), width=1)
            for line in band.lines:
                line.setPen(pen)
                line.setHoverPen(pen)
                # 预览带只是视觉反馈，任何鼠标键都不得被它抢走：否则拖拽中的
                # 悬停/点击会落到区域上，松手后右键菜单也可能被它接管
                line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            band.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            # ignoreBounds：预览带横跨整高，参与 auto-range 会把 Y 撑飞
            plot_item.addItem(band, ignoreBounds=True)
            self._scan_band = band
        else:
            band.setRegion((x0, x1))

    def _end_scan(self, ev) -> None:
        """松手：宽度够就缩 X，不够按误触补弹右键菜单；无论如何先清理会话。"""
        down = ev.buttonDownScenePos(Qt.MouseButton.RightButton)
        pos = ev.scenePos()
        width_px = abs(pos.x() - down.x())
        # 视图脱离 ViewWidget 时 screenGeometry() 为 None：按 0 宽算，只剩 8px 绝对下限
        geometry = self.screenGeometry()
        view_width = geometry.width() if geometry is not None else 0
        min_px = max(SCAN_MIN_PX, SCAN_MIN_WIDTH_RATIO * view_width)
        x0 = self.mapSceneToView(down).x()
        x1 = self.mapSceneToView(pos).x()

        try:
            if width_px >= min_px:
                self._apply_scan_range(min(x0, x1), max(x0, x1))
            elif self.menuEnabled():
                # 拖动已消费掉 mouseClickEvent → 手动补齐「点一下出菜单」
                self.raiseContextMenu(ev)
        finally:
            self._teardown_scan()

    def _apply_scan_range(self, xmin: float, xmax: float) -> None:
        """写 X 范围：走 AxisManager 入口（XLink 时写源视图，由原生广播到整组）。"""
        apply = getattr(self.plot_widget, "set_xrange_with_link_handling", None)
        if callable(apply):
            apply(xmin, xmax)
            return
        self.enableAutoRange(x=False)
        self.setXRange(xmin, xmax, padding=0)

    def _cancel_scan(self) -> None:
        self._teardown_scan()

    def _teardown_scan(self) -> None:
        self._scan_active = False
        band = self._scan_band
        self._scan_band = None
        if band is not None:
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
        self._remove_scan_key_filter()
        self._set_scan_cursor(None)

    def eventFilter(self, obj, event):
        """扫选期间监听视图的 Esc：取消并撤掉预览带。"""
        if (
            self._scan_active
            and obj is self._scan_filter_view
            and event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_Escape
        ):
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
        scene_pos = ev.scenePos()
        view_pos = self.mapSceneToView(scene_pos)
        self.context_x = view_pos.x()

        menu = super().getMenu(ev)
        if menu is None:
            return None

        # 锁定标志一律 getattr 防 None（plot_widget 注入前为 None）
        x_locked = getattr(self.plot_widget, "_x_mouse_locked", False)
        for act in menu.actions():
            if act.text() == "Mouse Mode":
                act.setVisible(False)
            elif act.text() == "Plot Options":
                submenu = act.menu()
                if submenu:
                    for subact in submenu.actions():
                        if subact.text() == "Transforms":
                            subact.setVisible(False)
            elif act.text() in ("X axis", "View All"):
                # B2：锁定时藏 X axis 子菜单与 View All —— 子菜单的 mouseCheck
                # 经 xMouseToggled→setMouseEnabled 会静默解锁，View All 会
                # 双轴 autoRange 重置 X；Y axis 保留。getMenu 每次右键都执行，
                # per-open 同步可见性，解锁后自然恢复
                act.setVisible(not x_locked)

        existing_texts = [act.text() for act in menu.actions()]

        if ZH_JUMP_TO_DATA not in existing_texts:
            jump_act = QAction(ZH_JUMP_TO_DATA, menu)
            jump_act.triggered.connect(self._emit_jump_to_data)
            if menu.actions():
                menu.insertAction(menu.actions()[0], jump_act)
            else:
                menu.addAction(jump_act)

        if ZH_AUTO_Y_IN_X not in existing_texts:
            auto_y_act = QAction(ZH_AUTO_Y_IN_X, menu)
            auto_y_act.triggered.connect(self._emit_auto_y)
            if len(menu.actions()) >= 1:
                menu.insertAction(
                    menu.actions()[1] if len(menu.actions()) > 1 else None,
                    auto_y_act,
                )
            else:
                menu.addAction(auto_y_act)

        actions_to_remove = []
        for action in menu.actions():
            # "Pin Cursor" / "Free Cursor" 是 pyqtgraph 历史项名，保持英文原样
            if action.text() in ["Pin Cursor", "Free Cursor", ZH_CURSOR_MENU]:
                actions_to_remove.append(action)
        self._discard_actions(menu, actions_to_remove)

        cursor_enabled = self._get_cursor_enabled()

        cursor_menu = QMenu(ZH_CURSOR_MENU, menu)
        # 游标选项菜单始终可用
        cursor_menu.setEnabled(True)
        cursor_group = QActionGroup(cursor_menu)
        cursor_group.setExclusive(True)
        current_mode = self._get_current_cursor_mode()

        # 添加三个正常模式选项（mode 为内部标识符，仅显示文案中文化）
        for mode in ["1 free cursor", "1 anchored cursor", "2 anchored cursor"]:
            mode_act = QAction(ZH_CURSOR_MODE_LABELS[mode], cursor_menu)
            mode_act.setCheckable(True)
            # 选中逻辑：光标开启时检查是否匹配当前模式，光标关闭时不选中
            mode_act.setChecked(cursor_enabled and mode == current_mode)
            # 所有选项始终可用
            mode_act.setEnabled(True)
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
        # "off" 选项始终可用
        off_act.setEnabled(True)
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

        if len(menu.actions()) >= 2:
            menu.insertMenu(
                menu.actions()[2] if len(menu.actions()) > 2 else None,
                cursor_menu,
            )
        else:
            menu.addMenu(cursor_menu)

        copy_act = None
        for act in menu.actions():
            if act.text() == ZH_COPY_NAME:
                copy_act = act
                break
        if copy_act is None:
            copy_act = QAction(ZH_COPY_NAME, menu)
            copy_act.triggered.connect(
                lambda: self.signals.request_copy_name.emit(self.plot_widget)
            )
            menu.addAction(copy_act)

        has_data = self._has_data()
        copy_act.setEnabled(has_data)

        if ZH_VAR_EDITOR not in existing_texts:
            editor_act = QAction(ZH_VAR_EDITOR, menu)
            editor_act.triggered.connect(
                lambda: self.signals.request_variable_editor.emit(self.plot_widget)
            )
            menu.addAction(editor_act)

        # 重排子图（低频操作走右键菜单，入口分层）
        if ZH_ARRANGE_LAYOUT not in existing_texts:
            arrange_act = QAction(ZH_ARRANGE_LAYOUT, menu)
            arrange_act.triggered.connect(
                lambda: self.signals.request_arrange_layout.emit(self.plot_widget)
            )
            menu.addAction(arrange_act)

        actions_to_remove = []
        for action in menu.actions():
            if action.text() == ZH_ADJUST_HEIGHT:
                actions_to_remove.append(action)
        self._discard_actions(menu, actions_to_remove)

        row = self._get_plot_row_index()
        adjust_height_menu = QMenu(ZH_ADJUST_HEIGHT, menu)
        percentages = [25, 50, 75, 100, 125, 150, 200, 250, 300, 400]
        current_pct = self._get_current_row_height(row)

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

        insert_index = None
        for i, action in enumerate(menu.actions()):
            if action.text() == ZH_VAR_EDITOR:
                insert_index = i + 1
                break
        if insert_index is not None:
            if insert_index < len(menu.actions()):
                menu.insertMenu(menu.actions()[insert_index], adjust_height_menu)
            else:
                menu.addMenu(adjust_height_menu)
        else:
            menu.addMenu(adjust_height_menu)

        if ZH_CLEAR_PLOT not in existing_texts:
            menu.addSeparator()
            clear_act = QAction(ZH_CLEAR_PLOT, menu)
            clear_act.triggered.connect(
                lambda: self.signals.request_clear_plot.emit(self.plot_widget)
            )
            menu.addAction(clear_act)

        return menu

    @staticmethod
    def _discard_actions(menu: QMenu, actions: list[QAction]) -> None:
        """摘掉并在事件循环空闲时销毁旧菜单项。

        ``removeAction`` 只解除挂载、不销毁 QObject：被摘下的 QAction 连同其
        子 QMenu（parent 仍是这个缓存菜单）继续存活，每次右键累积一整棵子树
        （实测 4 轮后 QAction 9→965）。子菜单是其 menuAction 的父对象，必须
        一起 deleteLater，否则只删 action 仍漏掉子菜单与其中的项。
        """
        for action in actions:
            submenu = action.menu()
            menu.removeAction(action)
            action.deleteLater()
            if submenu is not None:
                submenu.deleteLater()

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
