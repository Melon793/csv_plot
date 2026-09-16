"""CanUnmatchedDialog - BLF 中未匹配 CAN ID 的详情表（方案 §5.5）

由左侧 banner 的「查看」或状态③的链接打开。刻意做成**结果反馈**而不是
错误提示：未定义 ID 在真实总线日志里是常态（诊断帧、其他厂商节点、
尚未入库的新报文），所以这里只说明「这些报文没生成信号」并给出补库出口。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.config import BLF_UNMATCHED_REPORT_LIMIT
from src.core.logger import get_logger

logger = get_logger("ui.can_unmatched_dialog")

#: 表格列头。顺序即 §5.5 规定的「通道 → ID → 帧数 → 占比」，
#: 用户顺着读就能定位到具体总线上的一条报文。
_COLUMNS = ("总线通道", "CAN ID (hex)", "帧数", "占比")


class CanUnmatchedDialog(QDialog):
    """未匹配报文 ID 列表。``reselect_requested`` 由调用方接去重开选择对话框。"""

    def __init__(
        self,
        unmatched: list[tuple[int, int, bool, int]],
        *,
        total_messages: int = 0,
        parent: QWidget | None = None,
        limit: int = BLF_UNMATCHED_REPORT_LIMIT,
    ):
        super().__init__(parent)
        self._unmatched = list(unmatched or [])
        self._total_messages = int(total_messages or 0)
        self._limit = int(limit)
        self.reselect_requested = False

        self.setWindowTitle("未匹配的 CAN 报文 ID")
        self.setMinimumSize(520, 360)
        self.resize(560, 400)
        self._build_ui()
        self._fill_rows()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(12, 10, 12, 10)

        tip = QLabel(
            "下列报文在已选数据库中未定义，故未生成信号；"
            "补充 DBC / ARXML 后重选即可，无需重新扫描报文。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("font-weight: bold;")
        layout.addWidget(tip)

        self.table = QTableWidget(0, len(_COLUMNS))
        # 显式尺寸规范：表格至少要有可滚动的高度，不能靠布局隐式给
        self.table.setMinimumHeight(200)
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        self.table.setColumnWidth(0, 90)
        self.table.setColumnWidth(1, 130)
        self.table.setColumnWidth(2, 100)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)

        self.more_label = QLabel("")
        self.more_label.setStyleSheet("color: #666;")
        layout.addWidget(self.more_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.addStretch(1)
        self.reselect_btn = QPushButton("重选数据库…")
        self.reselect_btn.clicked.connect(self._on_reselect)
        btn_row.addWidget(self.reselect_btn)
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

    def _fill_rows(self) -> None:
        rows = sorted(self._unmatched, key=lambda x: -int(x[3]))[: self._limit]
        self.table.setRowCount(len(rows))
        for r, (channel, can_id, extended, frames) in enumerate(rows):
            # 帧格式直接用数据层给的真值，不再按 ID 大小猜：11 位与 29 位
            # 取值会重叠，一个 0x123 的**扩展**帧被标成标准帧就会误导补库。
            hex_text = f"0x{int(can_id):X}" + ("E" if extended else "")
            ratio = (
                f"{frames / self._total_messages * 100:.2f}%"
                if self._total_messages
                else "-"
            )
            cells = (str(int(channel)), hex_text, f"{int(frames):,}", ratio)
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c >= 2:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.table.setItem(r, c, item)

        hidden = len(self._unmatched) - len(rows)
        if hidden > 0:
            self.more_label.setText(
                f"另有 {hidden:,} 条，完整列表见日志"
            )
        else:
            self.more_label.setText(
                f"共 {len(rows):,} 条" if rows else "没有未匹配的报文 ID"
            )

    def _on_reselect(self) -> None:
        self.reselect_requested = True
        self.accept()
