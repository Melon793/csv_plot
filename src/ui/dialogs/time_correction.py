"""时间修正对话框"""

from __future__ import annotations
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QFormLayout,
    QPushButton,
    QHBoxLayout,
    QDoubleSpinBox,
)


class TimeCorrectionDialog(QDialog):
    """
    时间校正对话框类
    用于校正时间序列数据的时间偏移和漂移
    提供时间同步和数据对齐功能
    """

    def __init__(self, cur_factor=1.0, cur_offset=0.0, parent=None):
        super().__init__(parent)
        self.window_geometry = None  # 存储几何信息
        self.setWindowTitle("时间修正")

        form = QFormLayout(self)

        self.factor_spin = QDoubleSpinBox()
        self.factor_spin.setRange(0.0001, 1e6)
        self.factor_spin.setValue(cur_factor)
        self.factor_spin.setDecimals(6)

        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-1e9, 1e9)
        self.offset_spin.setValue(cur_offset)
        self.offset_spin.setDecimals(6)

        form.addRow("比例因子:", self.factor_spin)
        form.addRow("偏移量:", self.offset_spin)

        btns = QHBoxLayout()
        ok_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        form.addRow(btns)

        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        QTimer.singleShot(0, lambda: self.factor_spin.selectAll())

        if self.parent().time_correction_geometry:
            self.restoreGeometry(self.parent().time_correction_geometry)

    def values(self):
        return self.factor_spin.value(), self.offset_spin.value()

    def closeEvent(self, event):
        self.parent().time_correction_geometry = self.saveGeometry()
        super().closeEvent(event)
