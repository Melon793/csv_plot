"""
CustomViewBox —— 信号化自定义 ViewBox

将原本通过 plot_widget.window() 直接访问 MainWindow 的操作
替换为 PyQt 信号，由 DraggableGraphicsLayoutWidget 负责连接。
"""

from PySide6.QtCore import Signal, QObject
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMenu
import pyqtgraph as pg

# 右键菜单中文文案：创建与「判重 / 移除旧项」比对必须共用这些常量——
# getMenu 返回的是 pyqtgraph 缓存的同一个 QMenu，两处文案不一致会导致
# 每次右键都重复插入一份（菜单项成对翻倍）。
ZH_JUMP_TO_DATA = "跳转至变量数值表"
ZH_AUTO_Y_IN_X = "按 X 范围调节 Y 轴"
ZH_CURSOR_MODE = "游标模式"
ZH_SHOW_CURSOR_VALUE = "显示游标数值"
ZH_HIDE_CURSOR_VALUE = "隐藏游标数值"
ZH_COPY_NAME = "复制变量名"
ZH_VAR_EDITOR = "绘图变量编辑器"
ZH_ADJUST_HEIGHT = "调整高度"
ZH_RESET_ALL_HEIGHT = "全部重置为 100%"
ZH_CLEAR_PLOT = "清除绘图"

# --- 时间戳 X 轴 ---
ZH_X_AXIS = "X 轴"
ZH_X_AXIS_DEFAULT = "默认 (Index)"

# --- 标注功能（P1）---
ZH_ADD_ANNOTATION = "添加标注"
ZH_ANNOTATION_EDIT_MODE = "标注编辑模式"
ZH_ANNOTATION_LIST = "标注列表…"
ZH_ANNOTATION_CLEAR = "清除本图标注"
ZH_ANNOTATION_COPY = "复制选中标注"
ZH_ANNOTATION_COPY_ALL = "复制本图全部标注"
ZH_ANNOTATION_PASTE = "粘贴标注"
ZH_ANNOTATION_UNDO = "撤销"
ZH_ANNOTATION_REDO = "重做"

# 标注菜单项：(kind, 显示文案)
ANNOTATION_MENU_ITEMS = (
    ("text", "文字"),
    ("rect", "矩形"),
    ("ellipse", "椭圆"),
    ("line", "直线"),
    ("arrow", "箭头"),
    ("polyline", "折线"),
    ("target", "标靶"),
)

# 标注动作标识符：跨模块比对用，只改显示文案、绝不可改这些键值
ACTION_ANNOTATION_TOGGLE_EDIT = "toggle_edit"
ACTION_ANNOTATION_OPEN_LIST = "open_list"
ACTION_ANNOTATION_CLEAR = "clear"
ACTION_ANNOTATION_COPY = "copy"
ACTION_ANNOTATION_COPY_ALL = "copy_all"
ACTION_ANNOTATION_PASTE = "paste"
ACTION_ANNOTATION_UNDO = "undo"
ACTION_ANNOTATION_REDO = "redo"

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
    request_add_annotation = Signal(str, object)  # kind, plot_widget
    request_annotation_action = Signal(str, object)  # action, plot_widget
    request_set_x_axis_column = Signal(object, object)  # plot_widget, column|None


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
        self.context_y: float | None = None
        self.plot_widget = None

    def getMenu(self, ev):
        scene_pos = ev.scenePos()
        view_pos = self.mapSceneToView(scene_pos)
        self.context_x = view_pos.x()
        # context_y 是标注功能新增：新建标注需要一个具体落点，
        # 只有 context_x（游标跳转用）时只能取视图中心，右键位置就白点了
        self.context_y = view_pos.y()

        menu = super().getMenu(ev)
        if menu is None:
            return None

        for act in menu.actions():
            if act.text() == "Mouse Mode":
                act.setVisible(False)
            elif act.text() == "Plot Options":
                submenu = act.menu()
                if submenu:
                    for subact in submenu.actions():
                        if subact.text() == "Transforms":
                            subact.setVisible(False)

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
            if action.text() in ["Pin Cursor", "Free Cursor", ZH_CURSOR_MODE]:
                actions_to_remove.append(action)
        self._discard_actions(menu, actions_to_remove)

        cursor_enabled = self._get_cursor_enabled()

        cursor_menu = QMenu(ZH_CURSOR_MODE, menu)
        # 游标模式菜单始终可用
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

        if len(menu.actions()) >= 2:
            menu.insertMenu(
                menu.actions()[2] if len(menu.actions()) > 2 else None,
                cursor_menu,
            )
        else:
            menu.addMenu(cursor_menu)

        actions_to_remove = []
        for action in menu.actions():
            if action.text() in [ZH_SHOW_CURSOR_VALUE, ZH_HIDE_CURSOR_VALUE]:
                actions_to_remove.append(action)
        self._discard_actions(menu, actions_to_remove)

        values_hidden = self._get_cursor_values_hidden()
        if values_hidden:
            cursor_value_act = QAction(ZH_SHOW_CURSOR_VALUE, menu)
            cursor_value_act.triggered.connect(
                lambda: self.signals.request_show_cursor_value.emit(self.plot_widget)
            )
        else:
            cursor_value_act = QAction(ZH_HIDE_CURSOR_VALUE, menu)
            cursor_value_act.triggered.connect(
                lambda: self.signals.request_hide_cursor_value.emit(self.plot_widget)
            )
        cursor_value_act.setEnabled(cursor_enabled)

        if len(menu.actions()) >= 3:
            menu.insertAction(
                menu.actions()[3] if len(menu.actions()) > 3 else None,
                cursor_value_act,
            )
        else:
            menu.addAction(cursor_value_act)

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

        # ---- X 轴子菜单：默认 (Index) / 各时间列（时间戳模式） ----
        # 切换对全部子图生效（跨子图联动 / 游标要求 X 语义一致）。
        # 每次右键重建：勾选状态要反映当前模式（与标注子菜单同一理由）。
        self._discard_actions(
            menu, [act for act in menu.actions() if act.text() == ZH_X_AXIS]
        )
        x_axis_menu = self._build_x_axis_menu(menu)
        if x_axis_menu is not None:
            editor_index = None
            for i, action in enumerate(menu.actions()):
                if action.text() == ZH_VAR_EDITOR:
                    editor_index = i
                    break
            if editor_index is not None:
                menu.insertMenu(menu.actions()[editor_index], x_axis_menu)
            else:
                menu.addMenu(x_axis_menu)

        # ---- 标注子菜单：每次右键都重建 ----
        # 与上面各组的"判重后插入"不同，这里刻意重建：子菜单里有 checkable 的
        # 编辑模式开关，而 getMenu 返回的是 pyqtgraph 缓存的同一个 QMenu，
        # 不重建就会带着上一次的勾选状态显示。重建走 _discard_actions，
        # 旧子树会被 deleteLater，不会像 removeAction 那样累积。
        self._discard_actions(
            menu, [act for act in menu.actions() if act.text() == ZH_ADD_ANNOTATION]
        )
        annotation_menu = self._build_annotation_menu(menu)
        clear_index = None
        for i, action in enumerate(menu.actions()):
            if action.text() == ZH_CLEAR_PLOT:
                clear_index = i
                break
        if clear_index is not None:
            menu.insertMenu(menu.actions()[clear_index], annotation_menu)
        else:
            menu.addMenu(annotation_menu)

        if ZH_CLEAR_PLOT not in existing_texts:
            menu.addSeparator()
            clear_act = QAction(ZH_CLEAR_PLOT, menu)
            clear_act.triggered.connect(
                lambda: self.signals.request_clear_plot.emit(self.plot_widget)
            )
            menu.addAction(clear_act)

        return menu

    def _annotation_manager(self):
        """取本子图的标注管理器；独立 widget / 测试替身下返回 None"""
        pw = self.plot_widget
        if pw is None or not hasattr(pw, "annotation_manager"):
            return None
        try:
            return pw.annotation_manager
        except (RuntimeError, AttributeError):
            return None

    def _build_x_axis_menu(self, parent_menu: QMenu) -> "QMenu | None":
        """X 轴子菜单；没有可选时间列（无 loader / 非 datetime 列 / MDF）时为 None

        MDF 的 X 本来就是时间（master 通道秒值），不需要也不提供这个开关。
        FakePlotContext 等测试替身没有相关属性时整组跳过（getattr 守卫）。
        """
        pw = self.plot_widget
        context = getattr(pw, "plot_context", None)
        if context is None:
            return None
        try:
            columns = list(context.available_time_columns())
        except AttributeError:
            return None
        except Exception:
            columns = []
        try:
            current = getattr(context, "x_axis_time_column", None)
        except Exception:
            current = None
        if not columns and current is None:
            return None

        x_menu = QMenu(ZH_X_AXIS, parent_menu)

        default_act = QAction(ZH_X_AXIS_DEFAULT, x_menu)
        default_act.setCheckable(True)
        default_act.setChecked(current is None)
        default_act.triggered.connect(
            lambda: self.signals.request_set_x_axis_column.emit(
                self.plot_widget, None
            )
        )
        x_menu.addAction(default_act)

        if columns:
            x_menu.addSeparator()
            for column in columns:
                act = QAction(str(column), x_menu)
                act.setCheckable(True)
                act.setChecked(column == current)
                act.triggered.connect(
                    lambda checked=False, c=column: (
                        self.signals.request_set_x_axis_column.emit(
                            self.plot_widget, c
                        )
                    )
                )
                x_menu.addAction(act)
        return x_menu

    def _build_annotation_menu(self, parent_menu: QMenu) -> QMenu:
        """标注子菜单：新建各类图元 + 编辑模式开关 + 列表 + 清除本图"""
        menu = QMenu(ZH_ADD_ANNOTATION, parent_menu)

        for kind, label in ANNOTATION_MENU_ITEMS:
            act = QAction(label, menu)
            act.triggered.connect(
                lambda checked=False, k=kind: self.signals.request_add_annotation.emit(
                    k, self.plot_widget
                )
            )
            menu.addAction(act)
        menu.addSeparator()

        manager = self._annotation_manager()
        edit_act = QAction(ZH_ANNOTATION_EDIT_MODE, menu)
        edit_act.setCheckable(True)
        edit_act.setChecked(bool(manager is not None and manager.edit_mode))
        edit_act.triggered.connect(
            lambda: self.signals.request_annotation_action.emit(
                ACTION_ANNOTATION_TOGGLE_EDIT, self.plot_widget
            )
        )
        menu.addAction(edit_act)

        list_act = QAction(ZH_ANNOTATION_LIST, menu)
        list_act.triggered.connect(
            lambda: self.signals.request_annotation_action.emit(
                ACTION_ANNOTATION_OPEN_LIST, self.plot_widget
            )
        )
        menu.addAction(list_act)

        clear_act = QAction(ZH_ANNOTATION_CLEAR, menu)
        clear_act.setEnabled(bool(manager is not None and manager.count() > 0))
        clear_act.triggered.connect(
            lambda: self.signals.request_annotation_action.emit(
                ACTION_ANNOTATION_CLEAR, self.plot_widget
            )
        )
        menu.addAction(clear_act)

        menu.addSeparator()
        # 剪贴板与撤销栈的状态都在管理器里，enable 态每次右键重算 ——
        # 菜单是 pyqtgraph 缓存的同一个 QMenu，写死状态会带着上一次的结果显示
        clipboard = int(getattr(manager, "clipboard_count", 0)) if manager is not None else 0
        for label, action_id, enabled in (
            (
                ZH_ANNOTATION_COPY,
                ACTION_ANNOTATION_COPY,
                bool(manager is not None and manager.selected_id),
            ),
            (
                ZH_ANNOTATION_COPY_ALL,
                ACTION_ANNOTATION_COPY_ALL,
                bool(manager is not None and manager.count() > 0),
            ),
            (ZH_ANNOTATION_PASTE, ACTION_ANNOTATION_PASTE, clipboard > 0),
            (
                ZH_ANNOTATION_UNDO,
                ACTION_ANNOTATION_UNDO,
                bool(manager is not None and manager.can_undo),
            ),
            (
                ZH_ANNOTATION_REDO,
                ACTION_ANNOTATION_REDO,
                bool(manager is not None and manager.can_redo),
            ),
        ):
            act = QAction(label, menu)
            act.setEnabled(enabled)
            act.triggered.connect(
                lambda checked=False, a=action_id: self.signals.request_annotation_action.emit(
                    a, self.plot_widget
                )
            )
            menu.addAction(act)
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
        ncols = ctx._plot_col_max_default
        for idx, container in enumerate(ctx.plot_widgets):
            if container.plot_widget is self.plot_widget:
                row, _ = divmod(idx, ncols)
                return row
        return 0
