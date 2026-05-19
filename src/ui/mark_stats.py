"""标记统计窗口"""

from __future__ import annotations
import numpy as np
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QTreeWidget, QTreeWidgetItem)

class MarkStatsWindow(QDialog):
    """
    标记统计窗口类
    显示数据标记的统计信息和分析结果
    提供数据质量评估和异常检测功能
    使用单例模式确保只有一个统计窗口实例
    """
    _instance = None  # Singleton instance
    
    # 激活时不透明
    ACTIVE_OPACITY = 1.0
    # 非激活时 80% 透明
    INACTIVE_OPACITY = 0.8

    @classmethod
    def get_instance(cls, parent=None):
        if cls._instance is None:
            cls._instance = cls(parent)
        return cls._instance
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_geometry = None  # 存储几何信息
        self.setWindowTitle("标记区域统计")
        
        # 窗口标志：Tool窗口 + 仅标题栏和关闭按钮
        self.setWindowFlags(
            Qt.WindowType.Tool |
            Qt.WindowType.CustomizeWindowHint |
            Qt.WindowType.WindowTitleHint |
            Qt.WindowType.WindowCloseButtonHint
        )

        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabels(["Plot", "x1", "x2", "y1", "y2", "dx", "dy", "slope", "y_avg", "y_max", "y_min"])
        self.tree.setColumnWidth(0,200)
        #self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tree)
        self.no_curve_item = QTreeWidgetItem(self.tree, ["No Curve"])
        self.no_curve_item.setExpanded(False)
        if self.parent().mark_stats_geometry:
            self.restoreGeometry(self.parent().mark_stats_geometry)
        else:
            self.resize(1200, 300)

    def changeEvent(self, event):
        if event.type() == QEvent.Type.ActivationChange:
            self.setWindowOpacity(
                self.ACTIVE_OPACITY
                if self.isActiveWindow()
                else self.INACTIVE_OPACITY
            )
        super().changeEvent(event)

    def update_stats(self, stats_list):
        """更新统计信息显示
        
        Args:
            stats_list: 每个plot的统计信息列表
                - 单曲线模式：每个元素是一个包含11个值的元组列表
                - 多曲线模式：每个元素是一个包含多个元组的列表
        """
        self.tree.clear()
        self.no_curve_item = QTreeWidgetItem(self.tree, ["No Curve"])
        self.no_curve_item.setExpanded(False)
        has_no_curve = False
        
        for idx, stats in enumerate(stats_list):
            if stats:
                # stats现在是一个列表，可能包含多个曲线的统计信息
                if len(stats) == 1:
                    # 单曲线模式：直接显示
                    stat = stats[0]
                    item = QTreeWidgetItem(self.tree, [
                        f"Plot {idx+1} -> {stat[7]}",
                        f"{stat[0]:.2f}", f"{stat[1]:.2f}",
                        f"{stat[2]:.2f}", f"{stat[3]:.2f}",
                        f"{stat[4]:.2f}", f"{stat[5]:.2f}",
                        f"{stat[6]:.2f}" if not np.isinf(stat[6]) else "inf",
                        f"{stat[8]:.2f}", f"{stat[9]:.2f}", f"{stat[10]:.2f}"
                    ])
                else:
                    # 多曲线模式：创建父节点和子节点
                    parent_item = QTreeWidgetItem(self.tree, [f"Plot {idx+1} (多曲线)", "", "", "", "", "", "", "", "", "", ""])
                    parent_item.setExpanded(True)
                    
                    for stat in stats:
                        child_item = QTreeWidgetItem(parent_item, [
                            f"  → {stat[7]}",
                            f"{stat[0]:.2f}", f"{stat[1]:.2f}",
                            f"{stat[2]:.2f}", f"{stat[3]:.2f}",
                            f"{stat[4]:.2f}", f"{stat[5]:.2f}",
                            f"{stat[6]:.2f}" if not np.isinf(stat[6]) else "inf",
                            f"{stat[8]:.2f}", f"{stat[9]:.2f}", f"{stat[10]:.2f}"
                        ])
            else:
                has_no_curve = True
                sub_item = QTreeWidgetItem(self.no_curve_item, [f"Plot {idx+1}", "", "", "", "", "", "", "", "", "", ""])
        
        if not has_no_curve:
            self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(self.no_curve_item))
    
    def save_geom(self):
        if self.isMinimized() or self.isMaximized():
            self.parent().mark_stats_geometry = None  # 不保存，强制下次使用默认
        else:
            self.parent().mark_stats_geometry = self.saveGeometry()

    def load_geom(self):
        if self.parent().mark_stats_geometry is not None:
            geom = self.parent().mark_stats_geometry
            self.restoreGeometry(geom)
        else:
            # 默认大小和位置：resize并居中
            self.resize(1200, 300)
            screen = QApplication.primaryScreen().availableGeometry()
            x = (screen.width() - self.width()) // 2
            y = (screen.height() - self.height()) // 2
            self.move(x, y)
        
        # 新增：防御性重置状态，确保不是 min/max
        if self.isMinimized() or self.isMaximized():
            self.setWindowState(Qt.WindowState.WindowNoState)  # 强制正常状态

    def closeEvent(self, event):
        self.save_geom()
        super().closeEvent(event)

