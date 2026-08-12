"""
CustomViewBox —— 信号化自定义 ViewBox

将原本通过 plot_widget.window() 直接访问 MainWindow 的操作
替换为 PyQt 信号，由 DraggableGraphicsLayoutWidget 负责连接。
"""

from PySide6.QtCore import Signal, QObject
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMenu
import pyqtgraph as pg

import time


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

    def mouseDragEvent(self, ev, axis=None):
        # 标记真实用户事件（拖拽平移/右键拖拽缩放/Y 轴拖拽）：
        # 供 EventHandler._on_range_changed 区分用户触发与内部信号，
        # 防止交互所有权反转（auto-range 重算抢占所有权后抑制用户操作）
        if self.plot_widget is not None:
            self.plot_widget._user_event_pending = True
            self.plot_widget._user_event_ts = time.perf_counter()
        super().mouseDragEvent(ev, axis=axis)
        # 信号已在 super() 内同步消费则为 no-op；若本次事件未产生
        # range 变化（无信号发射）则清除残留标记防泄漏
        if self.plot_widget is not None:
            self.plot_widget._user_event_pending = False

    def wheelEvent(self, ev, axis=None):
        # 原生滚轮回退路径（带修饰键等）同样标记为用户事件
        if self.plot_widget is not None:
            self.plot_widget._user_event_pending = True
            self.plot_widget._user_event_ts = time.perf_counter()
        super().wheelEvent(ev, axis=axis)
        # 同上：未产生 range 变化时清除残留标记
        if self.plot_widget is not None:
            self.plot_widget._user_event_pending = False

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

        if "Jump to Data" not in existing_texts:
            jump_act = QAction("Jump to Data", menu)
            jump_act.triggered.connect(self._emit_jump_to_data)
            if menu.actions():
                menu.insertAction(menu.actions()[0], jump_act)
            else:
                menu.addAction(jump_act)

        if "Autoscale in x-Range" not in existing_texts:
            auto_y_act = QAction("Autoscale in x-Range", menu)
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
            if action.text() in ["Pin Cursor", "Free Cursor", "Cursor Mode"]:
                actions_to_remove.append(action)
        for action in actions_to_remove:
            menu.removeAction(action)

        cursor_enabled = self._get_cursor_enabled()

        cursor_menu = QMenu("Cursor Mode", menu)
        # Cursor Mode 菜单始终可用
        cursor_menu.setEnabled(True)
        cursor_group = QActionGroup(cursor_menu)
        cursor_group.setExclusive(True)
        current_mode = self._get_current_cursor_mode()

        # 添加三个正常模式选项
        for mode_text in ["1 free cursor", "1 anchored cursor", "2 anchored cursor"]:
            mode_act = QAction(mode_text, cursor_menu)
            mode_act.setCheckable(True)
            # 选中逻辑：光标开启时检查是否匹配当前模式，光标关闭时不选中
            mode_act.setChecked(cursor_enabled and mode_text == current_mode)
            # 所有选项始终可用
            mode_act.setEnabled(True)
            mode_act.triggered.connect(
                lambda checked, m=mode_text: self.signals.request_set_cursor_mode.emit(
                    m, self.plot_widget, self.context_x
                )
            )
            cursor_group.addAction(mode_act)
            cursor_menu.addAction(mode_act)

        # 添加 "off" 选项
        off_act = QAction("off", cursor_menu)
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
            if action.text() in ["Show Cursor Value", "Hide Cursor Value"]:
                actions_to_remove.append(action)
        for action in actions_to_remove:
            menu.removeAction(action)

        values_hidden = self._get_cursor_values_hidden()
        if values_hidden:
            cursor_value_act = QAction("Show Cursor Value", menu)
            cursor_value_act.triggered.connect(
                lambda: self.signals.request_show_cursor_value.emit(self.plot_widget)
            )
        else:
            cursor_value_act = QAction("Hide Cursor Value", menu)
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
            if act.text() == "Copy Name":
                copy_act = act
                break
        if copy_act is None:
            copy_act = QAction("Copy Name", menu)
            copy_act.triggered.connect(
                lambda: self.signals.request_copy_name.emit(self.plot_widget)
            )
            menu.addAction(copy_act)

        has_data = self._has_data()
        copy_act.setEnabled(has_data)

        if "Plot Variable Editor" not in existing_texts:
            editor_act = QAction("Plot Variable Editor", menu)
            editor_act.triggered.connect(
                lambda: self.signals.request_variable_editor.emit(self.plot_widget)
            )
            menu.addAction(editor_act)

        actions_to_remove = []
        for action in menu.actions():
            if action.text() == "Adjust Height":
                actions_to_remove.append(action)
        for action in actions_to_remove:
            menu.removeAction(action)

        row = self._get_plot_row_index()
        adjust_height_menu = QMenu("Adjust Height", menu)
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
        reset_act = QAction("100% to all", adjust_height_menu)
        reset_act.triggered.connect(
            lambda: self.signals.request_set_all_row_height.emit(100)
        )
        adjust_height_menu.addAction(reset_act)

        insert_index = None
        for i, action in enumerate(menu.actions()):
            if action.text() == "Plot Variable Editor":
                insert_index = i + 1
                break
        if insert_index is not None:
            if insert_index < len(menu.actions()):
                menu.insertMenu(menu.actions()[insert_index], adjust_height_menu)
            else:
                menu.addMenu(adjust_height_menu)
        else:
            menu.addMenu(adjust_height_menu)

        if "Clear Plot" not in existing_texts:
            menu.addSeparator()
            clear_act = QAction("Clear Plot", menu)
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
