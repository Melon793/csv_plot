import sys
import os
import csv
import math
import chardet
import numpy as np
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QMimeData, QRectF, QMargins, QTimer, QPointF
from PyQt6.QtGui import QFont, QFontMetrics, QDrag, QPen, QColor, QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QFileDialog, QPushButton, QAbstractItemView, QLabel, QLineEdit,
    QMessageBox, QDialog, QFormLayout, QSizePolicy
)
import pyqtgraph as pg

# 屏蔽 macOS ICC 警告
os.environ["QT_LOGGING_RULES"] = "qt5ct.debug=false"


class AxisDialog(QDialog):
    def __init__(self, axis, view_box, axis_type: str, plot_widget, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"调整 {axis_type} 轴")
        self.axis = axis
        self.view_box = view_box
        self.axis_type = axis_type
        self.plot_widget = plot_widget

        # 创建输入字段
        self.min_input = QLineEdit(str(view_box.viewRange()[0 if axis_type == "X" else 1][0]))
        self.max_input = QLineEdit(str(view_box.viewRange()[0 if axis_type == "X" else 1][1]))
        self.tick_count_input = QLineEdit()
        self.tick_count_input.setPlaceholderText("留空自动计算")

        # 创建布局
        layout = QFormLayout()
        layout.addRow("最小值:", self.min_input)
        layout.addRow("最大值:", self.max_input)
        layout.addRow("刻度数量:", self.tick_count_input)

        # 确定和取消按钮
        button_layout = QVBoxLayout()
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(self.apply_changes)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

        # 添加全选第一个文本框的功能
        QTimer.singleShot(0, lambda: self.min_input.selectAll())

    def apply_changes(self):
        try:
            min_val = float(self.min_input.text())
            max_val = float(self.max_input.text())
            if min_val >= max_val:
                QMessageBox.warning(self, "错误", "最小值必须小于最大值")
                return

            # 处理刻度数量
            tick_count_text = self.tick_count_input.text().strip()
            if tick_count_text:
                tick_count = int(tick_count_text)
                if tick_count < 2:
                    QMessageBox.warning(self, "错误", "刻度数量必须大于等于 2")
                    return
            else:
                tick_count = None  # 自动

            # 设置范围
            if self.axis_type == "X":
                self.view_box.setXRange(min_val, max_val, padding=0.00)
            else:
                self.view_box.setYRange(min_val, max_val, padding=0.00)

            # 设置固定刻度
            if tick_count:
                step = (max_val - min_val) / (tick_count - 1)
                ticks = [(min_val + i * step, str(round(min_val + i * step, 6)))
                         for i in range(tick_count)]
                self.axis.setTicks([ticks])
            else:
                self.axis.setTicks(None)
            self.accept()

        except ValueError:
            QMessageBox.warning(self, "错误", "请输入有效的数值（最小值、最大值、刻度数量）")


# ---------------- CSV Loader ----------------
def detect_encoding(file_path, n_bytes=4096):
    with open(file_path, "rb") as f:
        raw_data = f.read(n_bytes)
    result = chardet.detect(raw_data)
    return result['encoding'] if result['encoding'] else 'utf-8'


def load_csv(file_path):
    encoding = detect_encoding(file_path)
    try:
        with open(file_path, newline='', encoding=encoding) as f:
            reader = csv.reader(f)
            rows = list(reader)
    except UnicodeDecodeError:
        with open(file_path, newline='', encoding='gbk') as f:
            reader = csv.reader(f)
            rows = list(reader)

    if len(rows) < 3:
        raise ValueError("CSV 数据行数不足，至少包含变量名、单位和数据")

    var_names = rows[0]
    units_row = rows[1]
    data = {name: [] for name in var_names}
    units = {name: unit for name, unit in zip(var_names, units_row)}

    for row in rows[2:]:
        for i, value in enumerate(row):
            try:
                data[var_names[i]].append(float(value))
            except Exception:
                data[var_names[i]].append(value)
    return var_names, units, data


# ---------------- 自定义 QListWidget ----------------
class MyListWidget(QListWidget):
    def startDrag(self, supportedActions):
        item = self.currentItem()
        if item is None:
            return
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(item.text())
        drag.setMimeData(mime_data)
        drag.exec(Qt.DropAction.MoveAction)


# ---------------- 可拖拽 PlotWidget ----------------
class DraggablePlotWidget(pg.PlotWidget):
    def __init__(self, units_dict, data_dict, synchronizer=None):
        super().__init__()
        self.setAcceptDrops(True)
        self.units = units_dict
        self.data = data_dict
        self.synchronizer = synchronizer

        self.plot_item = self.getPlotItem()
        self.view_box = self.plot_item.vb
        self.plot_item.setTitle(None)
        self.plot_item.hideButtons()

        self.setBackground('w')
        self.plot_item.showGrid(x=True, y=True, alpha=0.1)

        # X 轴
        self.axis_x = self.plot_item.getAxis('bottom')
        self.axis_x.setStyle(tickFont=QFont("Arial", 12))
        self.axis_x.setTextPen('black')
        self.axis_x.setPen(QPen(QColor('black'), 1))
        self.axis_x.setRange(0, 10)

        # Y 轴
        self.axis_y = self.plot_item.getAxis('left')
        self.axis_y.setStyle(tickFont=QFont("Arial", 12))
        self.axis_y.enableAutoSIPrefix(False)
        self.axis_y.setTextPen('black')
        self.axis_y.setPen(QPen(QColor('black'), 1))

        # 其他边框
        for pos in ('top', 'right'):
            ax = self.plot_item.getAxis(pos)
            ax.setVisible(True)
            ax.setTicks([])
            ax.setStyle(showValues=False, tickLength=0)
            ax.setPen(QPen(QColor('black'), 1))

        # 计算固定 Y 轴宽度
        font = QFont("Arial", 12)
        fm = QFontMetrics(font)
        tick_w = fm.horizontalAdvance("-10000.01")
        margin = 16
        y_axis_width = tick_w + margin
        self.axis_y.setWidth(y_axis_width)

        # Y 轴 label 偏移
        title = "YYYYYYYYYYY"
        fm_f = QFontMetrics(font)
        title_rect = fm_f.tightBoundingRect(title)
        self.axis_y.setLabel(
            text=title,
            color='black',
            angle=-90,
            **{'font-family': 'Arial', 'font-size': '12pt', 'font-weight': 'bold'}
        )

        # 光标
        self.vline = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('r', width=4))
        self.cursor_label = pg.TextItem("", anchor=(1, 1), color="red")
        self.plot_item.addItem(self.vline, ignoreBounds=True)
        self.plot_item.addItem(self.cursor_label, ignoreBounds=True)
        self.vline.setVisible(False)
        self.cursor_label.setVisible(False)

        self.proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)
        self.vline.sigPositionChanged.connect(self.update_cursor_label)
        self.setAntialiasing(True)

        layout = self.plot_item.layout
        if layout is not None:
            layout.setContentsMargins(10, 10, 10, 10)

    # ---------------- 光标相关 ----------------
    def mouse_moved(self, evt):
        pos = evt[0]
        if not self.plot_item.sceneBoundingRect().contains(pos):
            return
        mousePoint = self.plot_item.vb.mapSceneToView(pos)
        window = self.window()
        if hasattr(window, 'sync_crosshair'):
            window.sync_crosshair(mousePoint.x(), self)

    def update_cursor_label(self):
        if len(self.plot_item.listDataItems()) == 0:
            self.cursor_label.setText("")
            return
        try:
            x = self.vline.value()
            curve = self.plot_item.listDataItems()[0]
            x_data, y_data = curve.getData()
            if x_data is None or len(x_data) == 0:
                self.cursor_label.setText("")
                return
            idx = min(range(len(x_data)), key=lambda i: abs(x_data[i] - x))
            y_val = y_data[idx]
            (x_min_view, x_max_view), (y_min_view, y_max_view) = self.view_box.viewRange()
            self.cursor_label.setText(f"x={x:.2f}, y={y_val:.2f}")

            fm = QFontMetrics(self.cursor_label.textItem.font())
            text_height = fm.height()
            dy = self.view_box.mapToView(QPointF(0, text_height)).y() - \
                self.view_box.mapToView(QPointF(0, 0)).y()
            margin_x = (x_max_view - x_min_view) * 0.02
            margin_y = dy * 1.2
            self.cursor_label.setPos(x_max_view - margin_x, y_max_view - abs(margin_y))
        except Exception as e:
            print(f"Cursor update error: {e}")
            self.cursor_label.setText("")

    def toggle_cursor(self, show: bool):
        self.vline.setVisible(show)
        self.cursor_label.setVisible(show)
        if show:
            self.update_cursor_label()

    # ---------------- 拖拽相关 ----------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        var_name = event.mimeData().text()
        if var_name not in self.data:
            QMessageBox.warning(self, "错误", f"变量 {var_name} 不存在")
            return

        raw_values = self.data[var_name]
        if not raw_values or all(v is None or str(v).strip() == '' for v in raw_values):
            QMessageBox.warning(self, "错误", f"变量 {var_name} 没有有效数据")
            for item in self.plot_item.listDataItems():
                self.plot_item.removeItem(item)
            self.axis_y.setLabel(text="")
            self.plot_item.update()
            event.acceptProposedAction()
            return

        y_values = []
        for v in raw_values:
            try:
                val = float(v)
                if math.isfinite(val):
                    y_values.append(val)
            except (ValueError, TypeError):
                pass

        if not y_values:
            sample_values = [str(v) for v in raw_values if v is not None and str(v).strip() != ''][:5]
            QMessageBox.information(self, "字符串变量", f"变量 {var_name} 包含字符串数据:\n{sample_values}")
            self.plot_item.clear()
            self.axis_y.setLabel(text="")
            self.plot_item.update()
            event.acceptProposedAction()
            return
        
        self.plot_item.clearPlots() 
        
        # for item in self.plot_item.listDataItems() + self.plot_item.items:
        #     if isinstance(item, pg.PlotDataItem):
        #         item.clear()
        #         self.plot_item.removeItem(item)

        x_values = list(range(len(y_values)))
        _pen = pg.mkPen(color='blue', width=4)
        self.plot_item.plot(x_values, y_values, pen=_pen, name=var_name)

        full_title = f"{var_name} ({self.units.get(var_name, '')})".strip()
        if len(full_title) > 30:
            full_title = full_title[:27] + "..."

        self.axis_y.setLabel(
            text=full_title,
            color='black',
            angle=-90,
            **{'font-family': 'Arial', 'font-size': '12pt', 'font-weight': 'bold'}
        )

        if min(y_values) == max(y_values):
            y_center = min(y_values)
            y_range = 1.0 if y_center == 0 else abs(y_center) * 0.2
            self.view_box.setYRange(y_center - y_range, y_center + y_range, padding=0.00)
        else:
            self.view_box.setYRange(min(y_values), max(y_values), padding=0.00)

        self.plot_item.update()
        if hasattr(self.window(), 'cursor_action'):
            self.vline.setBounds([min(x_values), max(x_values)])
            self.toggle_cursor(self.window().cursor_action.isChecked())
        else:
            self.toggle_cursor(False)

        event.acceptProposedAction()

    # ---------------- 双击轴弹出对话框 ----------------
    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseDoubleClickEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())
        y_axis_rect_scene = self.axis_y.mapToScene(self.axis_y.boundingRect()).boundingRect()
        x_axis_rect_scene = self.axis_x.mapToScene(self.axis_x.boundingRect()).boundingRect()

        if y_axis_rect_scene.contains(scene_pos):
            dialog = AxisDialog(self.axis_y, self.view_box, "Y", self)
            if dialog.exec():
                self.plot_item.update()
        elif x_axis_rect_scene.contains(scene_pos):
            dialog = AxisDialog(self.axis_x, self.view_box, "X", self)
            if dialog.exec():
                min_val, max_val = self.view_box.viewRange()[0]
                for view in self.window().findChildren(DraggablePlotWidget):
                    view.view_box.setXRange(min_val, max_val, padding=0.00)
                    view.plot_item.update()
        else:
            super().mouseDoubleClickEvent(event)


# ---------------- 主窗口 ----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV 拖放绘图 (PyQt6)")
        self.resize(1440, 900)

        self.var_names = []
        self.units = {}
        self.data = {}

        central = QWidget()
        main_layout = QHBoxLayout(central)

        # 光标菜单
        self.cursor_action = QAction("显示光标", self, checkable=True)
        self.cursor_action.triggered.connect(lambda checked: self.toggle_cursor_all(checked))
        self.menuBar().addMenu("选项").addAction(self.cursor_action)

        # 左侧变量列表
        left_widget = QWidget()
        left_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("变量列表"))

        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("输入关键词筛选变量（空格分隔）")
        self.filter_input.textChanged.connect(self.filter_variables)
        left_layout.addWidget(self.filter_input)

        self.load_btn = QPushButton("导入 CSV")
        self.load_btn.clicked.connect(self.load_csv_file)
        left_layout.addWidget(self.load_btn)

        self.list_widget = MyListWidget()
        self.list_widget.setDragEnabled(True)
        self.list_widget.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        left_layout.addWidget(self.list_widget)

        main_layout.addWidget(left_widget, 0)

        # 右侧绘图区
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        main_layout.addWidget(self.plot_widget, 4)

        self.setCentralWidget(central)

        self.plot_widgets = []
        self.create_subplots(4)

    # ---------------- 公用函数 ----------------
    def toggle_cursor_all(self, checked):
        for widget in self.plot_widgets:
            widget.toggle_cursor(checked)

    def sync_crosshair(self, x, sender_widget):
        if not self.cursor_action.isChecked():
            return
        for w in self.plot_widgets:
            w.vline.setVisible(True)
            w.vline.setPos(x)
            w.update_cursor_label()

    def create_subplots(self, n):
        for widget in self.plot_widgets:
            self.plot_layout.removeWidget(widget)
            widget.deleteLater()
        self.plot_widgets.clear()

        last_viewbox = None
        for _ in range(n):
            plot_widget = DraggablePlotWidget(self.units, self.data)
            plot_widget.toggle_cursor(self.cursor_action.isChecked())
            if last_viewbox is not None:
                plot_widget.view_box.setXLink(last_viewbox)
            last_viewbox = plot_widget.view_box

            wrapper = QVBoxLayout()
            wrapper.setContentsMargins(QMargins(0, 5, 5, 0))
            wrapper.addWidget(plot_widget)

            container = QWidget()
            container.setLayout(wrapper)
            self.plot_widgets.append(plot_widget)
            self.plot_layout.addWidget(container)

    def load_csv_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 CSV 文件", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            self.var_names, self.units, self.data = load_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))
            return
        self.list_widget.clear()
        self.list_widget.addItems(self.var_names)

        for widget in self.plot_widgets:
            widget.data = self.data
            widget.units = self.units

    def filter_variables(self, text):
        keywords = text.lower().split()
        self.list_widget.clear()
        if not keywords:
            self.list_widget.addItems(self.var_names)
            return
        filtered = [var for var in self.var_names if any(kw in var.lower() for kw in keywords)]
        self.list_widget.addItems(filtered)


# ---------------- 主程序 ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())