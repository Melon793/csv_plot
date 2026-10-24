"""标记统计窗口"""

from __future__ import annotations
import numpy as np
from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
)
from shiboken6 import isValid


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
    # 非激活时 90% 透明
    INACTIVE_OPACITY = 0.9

    @classmethod
    def get_instance(cls, parent=None):
        # 主窗口销毁后子窗口的 C++ 对象已不在，类级单例不能继续复用
        if cls._instance is not None and not isValid(cls._instance):
            cls._instance = None
        if cls._instance is None:
            cls._instance = cls(parent)
        return cls._instance

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_geometry = None  # 存储几何信息
        self.setWindowTitle("标记区域统计")

        # 窗口标志：Tool窗口 + 仅标题栏和关闭按钮
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabels(
            [
                "Plot",
                "x1",
                "x2",
                "y1",
                "y2",
                "dx",
                "dy",
                "slope",
                "y_avg",
                "y_max",
                "y_min",
            ]
        )
        self.tree.setColumnWidth(0, 200)
        # self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tree)
        self.no_curve_item = QTreeWidgetItem(self.tree, ["No Curve"])
        self.no_curve_item.setExpanded(False)
        parent = self.parent()
        if parent is not None and getattr(parent, "mark_stats_geometry", None):
            self.restoreGeometry(parent.mark_stats_geometry)
        else:
            self.resize(1200, 300)

    def changeEvent(self, event):
        if event.type() == QEvent.Type.ActivationChange:
            self.setWindowOpacity(
                self.ACTIVE_OPACITY if self.isActiveWindow() else self.INACTIVE_OPACITY
            )
        super().changeEvent(event)

    @staticmethod
    def _stat_cells(stat) -> list[str]:
        return [
            f"{stat.x1:.2f}",
            f"{stat.x2:.2f}",
            f"{stat.y1:.2f}",
            f"{stat.y2:.2f}",
            f"{stat.dx:.2f}",
            f"{stat.dy:.2f}",
            f"{stat.slope:.2f}" if not np.isinf(stat.slope) else "inf",
            f"{stat.y_avg:.2f}",
            f"{stat.y_max:.2f}",
            f"{stat.y_min:.2f}",
        ]

    def update_stats(self, stats_list):
        """更新统计信息显示（按页分组，§6.4 #9）

        Args:
            stats_list: ``[(page_name, idx_in_page, stats), ...]``。
                stats 为该子图的曲线统计列表；``None`` 表示该子图没有可统计
                的曲线。仍兼容旧式 ``[stats, ...]``（视为单页「图 1」）。
        """
        self.tree.clear()
        self.no_curve_item = QTreeWidgetItem(self.tree, ["No Curve"])
        self.no_curve_item.setExpanded(False)
        has_no_curve = False
        page_items: dict[str, QTreeWidgetItem] = {}

        normalized: list[tuple[str, int, object]] = []
        for position, entry in enumerate(stats_list):
            if isinstance(entry, tuple) and len(entry) == 3:
                normalized.append(entry)
            else:
                normalized.append(("图 1", position, entry))

        def _page_item(page_name: str) -> QTreeWidgetItem:
            item = page_items.get(page_name)
            if item is None:
                item = QTreeWidgetItem(
                    self.tree,
                    [page_name, "", "", "", "", "", "", "", "", "", ""],
                )
                item.setExpanded(True)
                page_items[page_name] = item
            return item

        for page_name, idx, stats in normalized:
            if not stats:
                has_no_curve = True
                QTreeWidgetItem(
                    self.no_curve_item,
                    [f"{page_name} · Plot {idx + 1}", "", "", "", "", "", "", "", "", "", ""],
                )
                continue

            page_item = _page_item(page_name)
            if len(stats) == 1:
                # 单曲线模式：直接挂在页节点下
                stat = stats[0]
                QTreeWidgetItem(
                    page_item,
                    [f"Plot {idx + 1} -> {stat.label}", *self._stat_cells(stat)],
                )
            else:
                # 多曲线模式：创建子图父节点和曲线子节点
                plot_item = QTreeWidgetItem(
                    page_item,
                    [f"Plot {idx + 1} (多曲线)", "", "", "", "", "", "", "", "", "", ""],
                )
                plot_item.setExpanded(True)
                for stat in stats:
                    QTreeWidgetItem(
                        plot_item,
                        [f"  → {stat.label}", *self._stat_cells(stat)],
                    )

        if not has_no_curve:
            self.tree.takeTopLevelItem(
                self.tree.indexOfTopLevelItem(self.no_curve_item)
            )

    def save_geom(self):
        parent = self.parent()
        if parent is None:
            return
        if self.isMinimized() or self.isMaximized():
            parent.mark_stats_geometry = None  # 不保存，强制下次使用默认
        else:
            parent.mark_stats_geometry = self.saveGeometry()

    def load_geom(self):
        parent = self.parent()
        if parent is not None and parent.mark_stats_geometry is not None:
            geom = parent.mark_stats_geometry
            self.restoreGeometry(geom)
        else:
            # 默认大小和位置：resize并居中
            self.resize(1200, 300)
            screen = QApplication.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry()
                x = (avail.width() - self.width()) // 2
                y = (avail.height() - self.height()) // 2
                self.move(x, y)

        # 既有逻辑保留：防御性重置状态，确保不是 min/max
        if self.isMinimized() or self.isMaximized():
            self.setWindowState(Qt.WindowState.WindowNoState)  # 强制正常状态

    def closeEvent(self, event):
        self.save_geom()
        parent = self.parent()
        if hasattr(parent, "layout_manager"):
            parent.layout_manager.toggle_mark_region(False)
        super().closeEvent(event)
