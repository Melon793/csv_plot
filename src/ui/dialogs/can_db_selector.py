"""CanDatabaseSelectorDialog - 指定 CAN 数据库（DBC / ARXML）（方案 §5.2）

BLF 只有原始报文，必须搭配数据库才能还原出信号，所以「导入 BLF」这一步必须
先把数据库定下来。本对话框承担四件事：

1. 让用户增删 / 排序多个 DBC、ARXML 文件；**列表顺序即优先级**（同一报文 ID
   在多库中重复时靠前的库生效，见 :meth:`CanDatabaseSet.find_frame`）；
2. 逐文件预解析并把结果直接摊在状态列里（✔ / ⚠ / ✖），避免「导入了才发现
   一个库都没读进去」；
3. data_file 是 BLF 时，后台线程预扫报文 ID 并在「命中」列回填每个库能解出
   的帧占比；用户没动过行序时自动按命中降序排序，把最匹配的库顶上去；
4. 记忆选择（勾选后由调用方写入 QSettings），使连续分析多份 BLF 不必重选。

单个文件解析失败**不阻断**其余行：真实的整车数据库工程里总有几个 canmatrix
读不动的历史文件。
"""

from __future__ import annotations

import os
import threading
from html import escape

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.config import (
    BLF_DB_COL_WIDTHS,
    BLF_DB_COLOR_ERROR,
    BLF_DB_COLOR_OK,
    BLF_DB_COLOR_WARN,
    BLF_DB_DIALOG_MIN_SIZE,
    BLF_DB_DIALOG_SIZE,
    BLF_PRESCAN_MAX_BYTES,
    BLF_TEXT_PRESCAN_SKIPPED,
)
from src.core.settings import AppSettings
from src.data.can_database import SUPPORTED_DB_EXTS, CanDatabaseSet, import_type_for
from src.data.blf_loader import prescan_blf_frames
from src.core.logger import get_logger
from src.utils.paths import normalize_input_path

logger = get_logger("ui.can_db_selector")

_COLUMNS = ("优先级", "文件", "类型", "报文数 / 信号数", "命中", "备注", "状态")

#: 备注列下标（抽屉清单表与选库框同序）：放在状态列前，紧邻「这个库现在
#: 好不好用」的结论，备注解释的正是「为什么选 / 为什么弃用它」。
_HIT_COL = 4
_NOTE_COL = 5
_STATUS_COL = 6

#: 状态列的四种取值 → (文字, 颜色)。⚠ 与 ✔ 用黄绿、✖ 用红，与 banner 三态
#: 同一套色觉约定，用户不必二次学习「红色 = 这个库没生效」。
_STATUS_OK = ("✔ 已加载", BLF_DB_COLOR_OK)
_STATUS_NO_FRAME = ("⚠ 无 CAN 报文", BLF_DB_COLOR_WARN)
_STATUS_ERROR = ("✖ 解析失败: {reason}", BLF_DB_COLOR_ERROR)
_STATUS_MISSING = ("✖ 文件不存在", BLF_DB_COLOR_ERROR)


class _Row:
    """一行 = 一个数据库文件及其预解析结果。"""

    __slots__ = ("path", "ok", "frame_count", "signal_count", "error", "missing",
                 "note", "db_ids", "hit_frames")

    def __init__(self, path: str):
        self.path = path
        self.ok = False
        self.frame_count = 0
        self.signal_count = 0
        self.error = ""
        self.missing = False
        # 用户备注：跟文件走（跨会话持久化在 QSettings），缺失/坏文件也可写
        self.note = AppSettings().get_blf_db_note(path)
        # 预解析时顺手记下帧定义集合，供预扫完成后估算命中率；-1 = 未知
        self.db_ids: frozenset = frozenset()
        self.hit_frames = -1

    @property
    def type_text(self) -> str:
        itype = import_type_for(self.path)
        return itype.upper() if itype else "?"

    def status(self) -> tuple[str, str]:
        if self.missing:
            return _STATUS_MISSING
        if self.ok:
            if self.frame_count == 0:
                return _STATUS_NO_FRAME
            return _STATUS_OK
        reason = (self.error or "未知错误").splitlines()[0]
        return _STATUS_ERROR[0].format(reason=reason[:48]), _STATUS_ERROR[1]


class CanDatabaseSelectorDialog(QDialog):
    """模态选择对话框。``exec()`` 返回 Accepted 后用 :meth:`get_selected_paths`。"""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        data_file: str = "",
        initial_paths: list[str] | None = None,
        missing_paths: list[str] | None = None,
        estimated_messages: int | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("指定 CAN 数据库（DBC / ARXML）")
        self.setAcceptDrops(True)
        self.setMinimumSize(*BLF_DB_DIALOG_MIN_SIZE)
        self.resize(*BLF_DB_DIALOG_SIZE)

        self._rows: list[_Row] = []
        self._missing = {normalize_input_path(p) for p in (missing_paths or [])}
        self._data_file = data_file
        self._estimated_messages = estimated_messages

        # ---- 后台预扫（方案 §5.2 命中列）---------------------------------
        # BLF 没有索引，报文 ID 集合只有完整读一遍才知道；预扫在后台线程跑
        # （不阻塞选库），完成后回填命中列。线程与 UI 的交接刻意用「普通
        # list + 轮询定时器」而不是跨线程信号：GIL 保证单次 append 原子，
        # 定时器随对话框析构，不用处理接收者已销毁的竞态。
        self._prescan_state = "off"     # off / running / done / failed
        self._prescan_counts = None     # Counter[(channel, can_id, extended)]
        self._prescan_total = 0
        self._prescan_box: list = []    # 线程 → UI 的单槽交接（[Counter|None]）
        self._prescan_cancel = False
        self._prescan_thread: threading.Thread | None = None
        self._prescan_timer: QTimer | None = None
        # 用户动过行（增删/排序）就不许再自动排序，尊重用户的手工顺序
        self._user_edited = False

        self._build_ui()
        if initial_paths:
            self.add_paths(initial_paths)
        self._refresh_buttons()
        self._start_prescan()

    # ------------------------------------------------------------------
    # 后台预扫：BLF 帧计数 → 命中列
    # ------------------------------------------------------------------

    def _start_prescan(self) -> None:
        """data_file 是存在的 .blf 时启动后台预扫；其余情况命中列保持「—」。"""
        if (not self._data_file
                or os.path.splitext(self._data_file)[1].lower() != ".blf"
                or not os.path.isfile(self._data_file)):
            return
        try:
            size = os.path.getsize(self._data_file)
        except OSError:
            size = 0
        if size >= BLF_PRESCAN_MAX_BYTES:
            # 预扫是第二次完整 I/O，大文件上会让「选库」空转几十秒。宁可
            # 不显示命中率，也不能让这一步看起来像卡死（评审 M4）。
            logger.info("BLF 体积 %d MB 达到预扫上限，跳过命中列: %s",
                        size // (1024 * 1024), self._data_file)
            self._prescan_state = "off"
            self._append_notice(BLF_TEXT_PRESCAN_SKIPPED)
            return
        self._prescan_state = "running"
        self._prescan_box = []
        self._prescan_cancel = False
        self._prescan_thread = threading.Thread(
            target=self._prescan_worker, name="blf-prescan", daemon=True,
        )
        self._prescan_thread.start()
        self._prescan_timer = QTimer(self)
        self._prescan_timer.setInterval(120)
        self._prescan_timer.timeout.connect(self._poll_prescan)
        self._prescan_timer.start()
        self._rebuild_table()

    def _prescan_worker(self) -> None:
        try:
            counts = prescan_blf_frames(
                self._data_file, should_cancel=lambda: self._prescan_cancel
            )
        except Exception as e:  # 文件坏 / 无报文 / python-can 缺失：命中列退化为「—」
            logger.info("BLF 预扫未完成，命中列不可用: %s", e)
            self._prescan_box.append(None)
        else:
            self._prescan_box.append(counts)

    def _poll_prescan(self) -> None:
        """轮询线程交接槽；结果迟到时对话框多半已关（cancel 位已置）→ 丢弃。"""
        if not self._prescan_box:
            return
        counts = self._prescan_box[0]
        self._prescan_timer.stop()
        self._prescan_timer = None
        self._prescan_thread = None
        # 不能拿 isVisible() 判断生死：预扫在 __init__ 就启动，文件很小时结果
        # 会先于 exec() 落地，那时对话框还没显示。关窗路径统一走
        # _shutdown_prescan（reject/closeEvent），cancel 位才是可靠的收口信号。
        if self._prescan_cancel:
            self._prescan_state = "off"
            return
        if counts is None:
            self._prescan_state = "failed"
        else:
            self._prescan_state = "done"
            self._prescan_counts = counts
            self._prescan_total = sum(counts.values())
        self._apply_prescan_results()

    def _hit_frames_of(self, row: _Row) -> int:
        """该库能解出的帧数：预扫计数按 (can_id, extended) 归并后查库内帧定义。

        ``find_frame`` 当前不区分通道（can_database.py），所以跨通道的同一 ID
        都算命中；预解析失败的行没有帧集合，返回 -1（未知）。
        """
        if self._prescan_state != "done" or not row.ok:
            return -1
        ids = row.db_ids
        return sum(n for (_ch, aid, ext), n in self._prescan_counts.items()
                   if (aid, ext) in ids)

    def _apply_prescan_results(self) -> None:
        for row in self._rows:
            row.hit_frames = self._hit_frames_of(row)
        self._rebuild_table()
        if (self._prescan_state == "done" and not self._user_edited
                and len(self._rows) > 1):
            self._rows.sort(key=lambda r: r.hit_frames, reverse=True)
            self._rebuild_table()
            self._show_sort_notice()

    def _append_notice(self, text: str) -> None:
        """提示条已有内容（如缺文件黄条）时追加，不覆盖更重要的告警。"""
        if self.notice.isVisibleTo(self):
            self.notice.setText(self.notice.text() + "<br>" + text)
        else:
            self.set_notice(text)

    def _show_sort_notice(self) -> None:
        self._append_notice(
            "已按命中帧数对列表降序排序（「命中」= 该库能解出的帧占比）。"
        )

    def _shutdown_prescan(self) -> None:
        """关窗 / 点「导入」时收口：置取消位，让后台线程在下个分块边界退出。"""
        self._prescan_cancel = True
        if self._prescan_timer is not None:
            self._prescan_timer.stop()
            self._prescan_timer = None

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel(
            "BLF 只记录原始报文，需数据库才能还原出信号。"
        )
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        title.setWordWrap(True)
        layout.addWidget(title)

        self.subtitle = QLabel(self._subtitle_text())
        self.subtitle.setStyleSheet("color: #666; font-size: 12px;")
        self.subtitle.setWordWrap(True)
        layout.addWidget(self.subtitle)

        # 路径记忆命中但有文件缺失（§5.1 的 C 路径）时的黄条说明
        self.notice = QLabel("")
        self.notice.setWordWrap(True)
        self.notice.setTextFormat(Qt.TextFormat.RichText)
        self.notice.setStyleSheet(
            f"background-color: #fff8e1; color: {BLF_DB_COLOR_WARN};"
            " padding: 4px 6px; border-radius: 3px;"
        )
        self.notice.setVisible(False)
        layout.addWidget(self.notice)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.setAlternatingRowColors(True)
        # 只有备注列可编辑：QTableWidget 的 edit trigger 是表级的，但 item
        # flags 是逐格的 —— 触发放开、非备注列的 item 摘掉 ItemIsEditable，
        # 双击其他列仍然是选择而不是进入编辑。
        self.table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked
            | QTableWidget.EditTrigger.EditKeyPressed
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, BLF_DB_COL_WIDTHS["priority"])
        self.table.setColumnWidth(2, BLF_DB_COL_WIDTHS["type"])
        self.table.setColumnWidth(3, BLF_DB_COL_WIDTHS["counts"])
        self.table.setColumnWidth(4, BLF_DB_COL_WIDTHS["hit"])
        self.table.setColumnWidth(5, BLF_DB_COL_WIDTHS["note"])
        self.table.setColumnWidth(6, BLF_DB_COL_WIDTHS["status"])
        self.table.setMinimumHeight(180)
        self.table.currentCellChanged.connect(lambda *_: self._refresh_buttons())
        self.table.itemChanged.connect(self._on_note_changed)
        layout.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.add_btn = QPushButton("添加文件…")
        self.add_btn.clicked.connect(self._on_add_files)
        btn_row.addWidget(self.add_btn)
        self.remove_btn = QPushButton("移除")
        self.remove_btn.clicked.connect(self._on_remove)
        btn_row.addWidget(self.remove_btn)
        self.up_btn = QPushButton("↑ 提高优先级")
        self.up_btn.clicked.connect(lambda: self._move_current(-1))
        btn_row.addWidget(self.up_btn)
        self.down_btn = QPushButton("↓ 降低优先级")
        self.down_btn.clicked.connect(lambda: self._move_current(1))
        btn_row.addWidget(self.down_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        hint = QLabel("ℹ 同一报文 ID 在多库中重复时，排在上面的库生效。")
        hint.setStyleSheet("color: #666;")
        layout.addWidget(hint)

        self.remember_check = QCheckBox("记住此列表，下次自动预选")
        self.remember_check.setChecked(True)
        layout.addWidget(self.remember_check)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel
        )
        # 语义是「导入」而不是「打开」；按钮文案直接决定用户是否明白下一步
        self.button_box.button(QDialogButtonBox.StandardButton.Open).setText("导 入")
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText("取 消")
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _subtitle_text(self) -> str:
        from src.data.var_info import format_size

        if not self._data_file:
            return "请选择一个或多个 DBC / ARXML 文件。"
        parts = [f"数据文件：{os.path.basename(self._data_file)}"]
        try:
            parts.append(format_size(os.path.getsize(self._data_file)))
        except OSError:
            pass
        if self._estimated_messages:
            parts.append(f"预计 {self._estimated_messages / 1_000_000:.1f} M 帧")
        return "   ".join(parts)

    def set_notice(self, text: str) -> None:
        """C 路径：告诉用户「为什么上次的库不能直接用」。"""
        if text:
            self.notice.setText(text)
            self.notice.setVisible(True)
        else:
            self.notice.setText("")
            self.notice.setVisible(False)

    # ------------------------------------------------------------------
    # 行管理
    # ------------------------------------------------------------------

    def add_paths(self, paths: list[str]) -> int:
        """加入文件并逐份预解析。返回真正新增的行数。"""
        added = 0
        existing = {r.path for r in self._rows}
        missing_names: list[str] = []
        for raw in paths or []:
            path = normalize_input_path(str(raw))
            if not path or path in existing:
                continue
            if import_type_for(path) is None:
                logger.info("忽略非 CAN 数据库文件: %s", path)
                continue
            row = _Row(path)
            if not os.path.isfile(path):
                row.missing = True
                row.error = "文件不存在"
                missing_names.append(os.path.basename(path))
            else:
                self._probe(row)
                if self._prescan_state == "done":
                    # 预扫已完成时补加的行立即能算命中，不必等下一轮
                    row.hit_frames = self._hit_frames_of(row)
            self._rows.append(row)
            existing.add(path)
            added += 1
        if missing_names:
            # 提示条是 RichText：文件名里的 & / < 必须转义，否则会被当标记吞掉
            self.set_notice(
                "上次使用的数据库中有 "
                f"{len(missing_names)} 个文件不存在或无法读取，已标红："
                + "、".join(escape(n) for n in missing_names)
            )
        self._rebuild_table()
        return added

    @staticmethod
    def _probe(row: _Row) -> None:
        """逐文件预解析：只关心「能不能读、多少报文/信号」，不做跨库合并。"""
        db = CanDatabaseSet().load([row.path])
        report = db.reports[0]
        row.ok = bool(report.ok)
        row.frame_count = int(report.frame_count)
        row.signal_count = int(report.signal_count)
        row.error = str(report.error or "")
        # 帧定义集合供命中列估算（见 _hit_frames_of）；解析失败的行没有可靠集合
        row.db_ids = (
            frozenset((f.can_id, bool(f.extended)) for f in db.frames)
            if report.ok else frozenset()
        )

    def _hit_cell(self, row: _Row) -> tuple[str, str]:
        """命中列的 (文字, tooltip)。预扫进行中显示「…」，不可用显示「—」。"""
        if self._prescan_state == "running":
            return "…", "正在后台预扫 BLF 报文 ID，完成后显示各库命中率"
        if row.hit_frames >= 0 and self._prescan_total:
            pct = row.hit_frames / self._prescan_total * 100
            text = "100%" if pct >= 99.95 else f"{pct:.1f}%"
            return (text,
                    f"命中 {row.hit_frames:,} / {self._prescan_total:,} 帧"
                    f"（该库能解出的报文占比）")
        return "—", "预扫不可用，无法估算匹配度"

    def _rebuild_table(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._rows))
        for r, row in enumerate(self._rows):
            text, color = row.status()
            hit_text, hit_tip = self._hit_cell(row)
            cells = (
                str(r + 1),
                os.path.basename(row.path),
                row.type_text,
                f"{row.frame_count} / {row.signal_count}"
                if not row.missing
                else "— / —",
                hit_text,
                row.note,
                text,
            )
            for c, value in enumerate(cells):
                item = QTableWidgetItem(value)
                if c == 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if c == 1:
                    item.setToolTip(row.path)
                if c == _HIT_COL:
                    item.setToolTip(hit_tip)
                if c == _NOTE_COL:
                    # 编辑开关只挂在这一列的 flags 上（见 _build_ui）
                    item.setToolTip(row.note or "双击填写备注（跨会话记忆）")
                else:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if c == _STATUS_COL:
                    item.setForeground(_qcolor(color))
                    item.setToolTip(row.error or text)
                self.table.setItem(r, c, item)
        self.table.blockSignals(False)
        self._refresh_buttons()

    def _on_note_changed(self, item: QTableWidgetItem) -> None:
        """备注编辑落盘：随改随存，不等「导入」（备注是文件的属性，与本次
        导入是否发生无关）。strip 后与原文本不同时回写 item，触发一次再入
        后文本相等，递归自然终止。"""
        if item.column() != _NOTE_COL:
            return
        r = item.row()
        if not (0 <= r < len(self._rows)):
            return
        row = self._rows[r]
        text = item.text().strip()
        row.note = text
        AppSettings().set_blf_db_note(row.path, text)
        if item.text() != text:
            item.setText(text)
        elif not text:
            item.setToolTip("双击填写备注（跨会话记忆）")
        else:
            item.setToolTip(text)

    def _refresh_buttons(self) -> None:
        has_row = self.table.currentRow() >= 0
        self.remove_btn.setEnabled(has_row)
        self.up_btn.setEnabled(has_row and self.table.currentRow() > 0)
        self.down_btn.setEnabled(
            has_row and 0 <= self.table.currentRow() < len(self._rows) - 1
        )
        importable = [r for r in self._rows if r.ok and r.frame_count > 0]
        self.button_box.button(QDialogButtonBox.StandardButton.Open).setEnabled(
            bool(importable)
        )
        if not importable:
            tip = (
                "至少选择一个可解析的数据库文件"
                if self._rows
                else "请先添加 DBC / ARXML 文件"
            )
        else:
            tip = ""
        self.button_box.button(QDialogButtonBox.StandardButton.Open).setToolTip(tip)

    def _move_current(self, delta: int) -> None:
        r = self.table.currentRow()
        dst = r + delta
        if r < 0 or dst < 0 or dst >= len(self._rows):
            return
        self._user_edited = True
        self._rows[r], self._rows[dst] = self._rows[dst], self._rows[r]
        self._rebuild_table()
        self.table.selectRow(dst)

    def _on_remove(self) -> None:
        r = self.table.currentRow()
        if 0 <= r < len(self._rows):
            self._user_edited = True
            del self._rows[r]
            self._rebuild_table()

    def _on_add_files(self) -> None:
        self._user_edited = True
        initial_dir = (
            os.path.dirname(self._rows[-1].path) if self._rows else ""
        )
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "添加 CAN 数据库",
            initial_dir,
            "CAN Database (*.dbc *.arxml);;DBC Files (*.dbc);;ARXML Files (*.arxml)",
        )
        if files:
            self.add_paths(files)

    def _on_accept(self) -> None:
        importable = [r for r in self._rows if r.ok and r.frame_count > 0]
        if not importable:
            QMessageBox.warning(
                self,
                "没有可用的数据库",
                "所选文件全部无法解析出 CAN 报文。\n"
                "请确认文件是 DBC 或 ARXML，且内容与本工具支持的格式一致。",
            )
            return  # 对话框保持打开
        self._shutdown_prescan()  # 用户已定夺，预扫不必再跑
        self.accept()

    # ------------------------------------------------------------------
    # 结果
    # ------------------------------------------------------------------

    def get_selected_paths(self) -> list[str]:
        """按优先级返回可用数据库路径（跳过缺失/解析失败的行）。"""
        return [r.path for r in self._rows if r.ok and r.frame_count > 0]

    def all_paths(self) -> list[str]:
        """含失败行的完整列表，供调用方写日志。"""
        return [r.path for r in self._rows]

    def remember_choice(self) -> bool:
        return self.remember_check.isChecked()

    @property
    def rows(self) -> list[_Row]:
        return self._rows

    # ------------------------------------------------------------------
    # 拖拽：把 .dbc / .arxml 直接拖进列表
    # ------------------------------------------------------------------

    def reject(self) -> None:  # noqa: N802 - Qt 命名
        self._shutdown_prescan()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        self._shutdown_prescan()
        super().closeEvent(event)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if self._drag_candidate_paths(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        paths = self._drag_candidate_paths(event)
        if paths:
            self._user_edited = True
            self.add_paths(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    @staticmethod
    def _drag_candidate_paths(event) -> list[str]:
        """只收 .dbc / .arxml；其余后缀直接忽略（不给错误框，避免打断）。"""
        mime = event.mimeData()
        if not mime.hasUrls():
            return []
        out = []
        for url in mime.urls():
            local = url.toLocalFile()
            if not local:
                continue
            if os.path.splitext(local)[1].lower() in SUPPORTED_DB_EXTS:
                out.append(local)
        return out


def _qcolor(hex_text: str) -> QColor:
    return QColor(hex_text) if hex_text else QColor("#000000")
