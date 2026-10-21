"""CanDbBanner - BLF 数据源的 CAN 数据库状态条（方案 §5.3）

放在左面板变量过滤框之上，常驻显示「当前这份 BLF 是用哪些数据库解出来的、
有没有没对上的报文 ID」。三态：

① 正常（绿）：全部报文都命中数据库；
② 截断（黄）：报文数超过 ``BLF_MAX_MESSAGES``，只扫了前一段；
③ 有未匹配（橙）：部分 CAN ID 在数据库里没有定义，因而没生成信号。

刻意做成**非阻塞反馈**：未匹配 ID 在真实日志里是常态（总线上有诊断帧、
其他厂商的报文），扫描完弹一个模态框会把人逼疯，所以这里只改横幅颜色并
给一个「查看」入口。
"""

from __future__ import annotations

from html import escape

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

from src.core.config import (
    BLF_BANNER_HEIGHT,
    BLF_BANNER_NAME_ELLIDE_AT,
    BLF_BANNER_STYLE_NORMAL,
    BLF_BANNER_STYLE_TRUNCATED,
    BLF_BANNER_STYLE_UNMATCHED,
)
from src.core.logger import get_logger

logger = get_logger("ui.can_db_banner")


def _ellide(text: str, limit: int = BLF_BANNER_NAME_ELLIDE_AT) -> str:
    """数据库文件名过长时省略中段，保留扩展名（用户靠后缀判断类型）。"""
    if len(text) <= limit:
        return text
    keep = limit - 1
    head = keep // 2
    tail = keep - head
    return f"{text[:head]}…{text[-tail:]}"


def _names_html(db_names: list[str] | None) -> str:
    """逗号拼接的库名（已转义）。

    ``info_label`` 是 RichText：文件名里的 ``&``、``<`` 不转义会被当成实体 /
    标签吞掉，用户看到的名字就和磁盘上的不一样了（评审 A8）。
    """
    return ", ".join(escape(_ellide(n)) for n in (db_names or []))


class CanDbBanner(QWidget):
    """BLF 数据库状态横幅。非 BLF 数据源时 ``setVisible(False)`` 且不占位。"""

    #: 用户点击「查看」/未匹配详情链接
    view_unmatched_requested = Signal()
    #: 用户点击「重选…」
    reselect_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(BLF_BANNER_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # 默认隐藏：只有 LOADER_TYPE == "blf" 时才由 main_window 显隐
        self.setVisible(False)
        self._state = ""
        self._unmatched_count = 0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 6, 2)
        layout.setSpacing(6)

        self.info_label = QLabel()
        self.info_label.setTextFormat(Qt.TextFormat.RichText)
        self.info_label.setWordWrap(False)
        # 文案里带 <a href> 链接（未匹配详情），必须开 links 且禁文本选择，
        # 否则点链接会被当成选词操作。
        self.info_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self.info_label.setOpenExternalLinks(False)
        self.info_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.info_label.linkActivated.connect(self._on_link_activated)
        layout.addWidget(self.info_label, 1)

        self.view_btn = QPushButton("查看")
        self.view_btn.setFixedWidth(46)
        self.view_btn.clicked.connect(self.view_unmatched_requested.emit)
        self.view_btn.setVisible(False)
        layout.addWidget(self.view_btn, 0)

        self.reselect_btn = QPushButton("重选…")
        self.reselect_btn.setFixedWidth(58)
        self.reselect_btn.setToolTip("重新指定 DBC / ARXML（无需重新扫描报文）")
        self.reselect_btn.clicked.connect(self.reselect_requested.emit)
        layout.addWidget(self.reselect_btn, 0)

        self._apply_style(*BLF_BANNER_STYLE_NORMAL)

    # ------------------------------------------------------------------
    # 三态入口
    # ------------------------------------------------------------------

    def show_normal(
        self,
        *,
        db_count: int,
        frame_count: int,
        signal_count: int,
        db_names: list[str] | None = None,
    ) -> None:
        """状态①：全部报文命中数据库。"""
        line1 = (
            f"已加载 {db_count} 个数据库 · "
            f"{frame_count} 报文 / {signal_count} 信号"
        )
        names = _names_html(db_names)
        self._set_text(line1, names)
        self._state = "normal"
        self._unmatched_count = 0
        self.view_btn.setVisible(False)
        self._apply_style(*BLF_BANNER_STYLE_NORMAL)
        self.setVisible(True)

    def show_truncated(
        self,
        *,
        limit: int,
        scanned: int,
        db_count: int,
        db_names: list[str] | None = None,
        reason: str = "messages",
    ) -> None:
        """状态②：报文数 / payload 字节数超过上限，后半段没生成信号。"""
        what = (
            f"报文数据超过 {limit // (1024 * 1024)} MB"
            if reason == "payload"
            else f"报文数超过 {limit:,}"
        )
        line1 = (
            f"{what}，仅扫描前 {scanned:,} 条"
            f" · 已用 {db_count} 个数据库"
        )
        self._set_text(line1, _names_html(db_names))
        self._state = "truncated"
        self._unmatched_count = 0
        self.view_btn.setVisible(False)
        self._apply_style(*BLF_BANNER_STYLE_TRUNCATED)
        self.setVisible(True)

    def show_unmatched(
        self,
        *,
        unmatched_count: int,
        db_count: int,
        frame_count: int,
        signal_count: int,
        db_names: list[str] | None = None,
    ) -> None:
        """状态③：有 CAN ID 未定义，给「查看」入口。"""
        names = _names_html(db_names)
        line1 = (
            f'<a href="unmatched">{unmatched_count} 个报文 ID 未在数据库中定义</a>'
            f"（未生成信号）· {db_count} 库 / {frame_count} 报文 / {signal_count} 信号"
        )
        self.info_label.setText(f"{line1}<br>{names}" if names else line1)
        self._state = "unmatched"
        self._unmatched_count = int(unmatched_count)
        self.view_btn.setVisible(unmatched_count > 0)
        self._apply_style(*BLF_BANNER_STYLE_UNMATCHED)
        self.setVisible(True)

    def hide_banner(self) -> None:
        """非 BLF 数据源：隐藏且不留占位（不动态增删控件，避免 splitter 抖动）。"""
        self._state = ""
        self._unmatched_count = 0
        self.setVisible(False)

    @property
    def state(self) -> str:
        return self._state

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _set_text(self, line1: str, line2: str) -> None:
        self.info_label.setText(f"{line1}<br>{line2}" if line2 else line1)

    def _apply_style(self, bg: str, fg: str) -> None:
        self.setStyleSheet(
            f"CanDbBanner {{ background-color: {bg}; border-radius: 3px; }}"
            f"CanDbBanner > QLabel {{ color: {fg}; background: transparent; border: none; }}"
        )
        self.info_label.setStyleSheet(f"color: {fg}; background: transparent;")

    def _on_link_activated(self, link: str) -> None:
        if link == "unmatched" and self._unmatched_count:
            self.view_unmatched_requested.emit()
        else:
            logger.debug("banner 链接忽略: %s", link)
