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
ZH_JUMP_TO_DATA = "跳转至数据表"
ZH_AUTO_Y_IN_X = "按 X 范围调节 Y 轴"
ZH_CURSOR_MODE = "游标模式"
ZH_SHOW_CURSOR_VALUE = "显示游标数值"
ZH_HIDE_CURSOR_VALUE = "隐藏游标数值"
ZH_COPY_NAME = "复制变量名"
ZH_VAR_EDITOR = "绘图变量编辑器"
ZH_ADJUST_HEIGHT = "调整高度"
ZH_RESET_ALL_HEIGHT = "全部重置为 100%"
ZH_CLEAR_PLOT = "清除绘图"

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

    def getMenu(self, ev):
        scene_pos = ev.scenePos()
        view_pos = self.mapSceneToView(scene_pos)
        self.context_x = view_pos.x()

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
        for action in actions_to_remove:
            menu.removeAction(action)

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
        for action in actions_to_remove:
            menu.removeAction(action)

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
        for action in actions_to_remove:
            menu.removeAction(action)

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
