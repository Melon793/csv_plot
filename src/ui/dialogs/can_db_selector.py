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

from PySide6.QtCore import QMimeData, Qt, QTimer
from PySide6.QtGui import QColor, QDrag
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
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
from src.data.can_bindings import (
    CanBindings,
    format_bus_channel,
    is_dir_writable,
    load_yaml,
    save_yaml,
    sidecar_config_path,
)
from src.data.can_database import SUPPORTED_DB_EXTS, CanDatabaseSet, import_type_for
from src.data.blf_loader import prescan_blf_frames
from src.core.gc_guard import no_autogc
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


_DB_MIME = "application/x-canview-db"


class _PoolTable(QTableWidget):
    """库池表：拖行调序 + 拖到通道表绑定 + 外部文件拖入转发。

    外部文件（text/uri-list）显式转发给对话框处理，不依赖「ignore 后是否
    冒泡到父控件」的平台行为——评审 P1 指出那是未验证假设。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_path = ""
        #: (path, dst_row) -> None，由对话框注入；按路径而不是行号定位，
        #: 避免预扫自动排序在拖拽期间改变行号（评审 P2 的竞态）
        self.on_move = None
        #: (event) -> bool，外部文件拖入的接受/落盘委托给对话框
        self.on_external_accept = None
        self.on_external_drop = None

    def startDrag(self, supported_actions):  # noqa: N802 - Qt 命名
        row = self.currentRow()
        if row < 0 or self.on_move is None:
            return
        item = self.item(row, 1)
        path = item.toolTip() if item is not None else ""
        if not path:
            return
        self._drag_path = path
        mime = QMimeData()
        mime.setData(_DB_MIME, path.encode("utf-8"))
        mime.setText(path)
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction | Qt.DropAction.MoveAction)
        # P1-6（第 5 轮）：释放 QDrag/QMimeData，防反复拖拽累积；
        # exec 期间源控件被销毁时 C++ 对象已亡，静默跳过（评审 N1）
        try:
            drag.deleteLater()
        except RuntimeError:
            pass

    def dragEnterEvent(self, event):  # noqa: N802 - Qt 命名
        if event.mimeData().hasFormat(_DB_MIME):
            event.acceptProposedAction()
        elif self.on_external_accept is not None and self.on_external_accept(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):  # noqa: N802 - Qt 命名
        if event.mimeData().hasFormat(_DB_MIME):
            event.acceptProposedAction()
        elif self.on_external_accept is not None and self.on_external_accept(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):  # noqa: N802 - Qt 命名
        if event.mimeData().hasFormat(_DB_MIME) and event.source() is self:
            row = self.rowAt(int(event.position().toPoint().y()))
            if row < 0:
                row = self.rowCount() - 1
            if self.on_move is not None and self._drag_path:
                self.on_move(self._drag_path, row)
            self._drag_path = ""
            event.acceptProposedAction()
            return
        if self.on_external_drop is not None and self.on_external_drop(event):
            event.acceptProposedAction()
            return
        event.ignore()


class _ChannelTable(QTableWidget):
    """通道绑定表：接受从库池拖来的库行（drop = 加入该通道绑定）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        #: (0 基通道, 库路径) -> None，由对话框注入
        self.on_bind = None
        #: 外部文件拖入：显式转发给对话框，不依赖 ignore 冒泡
        self.on_external_accept = None
        self.on_external_drop = None

    def _accepts(self, event) -> bool:
        return event.mimeData().hasFormat(_DB_MIME) and self.on_bind is not None

    def _accepts_external(self, event) -> bool:
        return (
            self.on_external_accept is not None
            and self.on_external_accept(event)
        )

    def dragEnterEvent(self, event):  # noqa: N802 - Qt 命名
        if self._accepts(event) or self._accepts_external(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):  # noqa: N802 - Qt 命名
        if self._accepts(event) or self._accepts_external(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):  # noqa: N802 - Qt 命名
        if self._accepts(event):
            row = self.rowAt(int(event.position().toPoint().y()))
            item = self.item(row, 0) if row >= 0 else None
            channel = (
                item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            )
            if channel is None:
                event.ignore()
                return
            path = bytes(event.mimeData().data(_DB_MIME)).decode("utf-8")
            self.on_bind(int(channel), path)
            event.acceptProposedAction()
            return
        if self.on_external_drop is not None and self.on_external_drop(event):
            event.acceptProposedAction()
            return
        event.ignore()


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
        initial_cfg: CanBindings | None = None,
        known_channels: list[tuple[int, int]] | None = None,
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

        # ---- 通道绑定（0 基通道 → 库路径子集 / 别名）---------------------
        cfg = initial_cfg or CanBindings.empty()
        self._bindings: dict[int, tuple[str, ...]] = dict(cfg.by_channel)
        self._aliases: dict[int, str] = dict(cfg.alias)
        #: 重选/抽屉场景由 loader 已扫桶给出（预扫被跳过时也能列通道）
        self._known_channels: dict[int, int] = dict(known_channels or [])
        #: 首次导入 >128MB（预扫跳过）时用户手工补的通道号
        self._manual_channels: list[int] = []
        #: 预扫阶段的错误帧计数（0 基通道 → 帧数），用于通道行提示
        self._prescan_errors: dict[int, int] = {}

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
        #: 用户手动折叠过通道绑定分组时，重建不再覆盖其选择（D1）
        self._bind_user_toggled = False

        self._build_ui()
        if initial_paths:
            self.add_paths(initial_paths)
        self._refresh_buttons()
        self._rebuild_channel_table()
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
        # 整段预扫包在 no_autogc() 窗口内：本线程每 4k 帧构造 Message/list/Counter
        # 临时对象，分配量足够触发自动分代回收；收集若落在这里，会把主线程创建的
        # QObject 包装器拿到本线程析构（详见 src/core/gc_guard.py）。线程是 daemon
        # 且关窗只置取消位，窗口若漏，分配会跨用例继续。
        with no_autogc():
            errors: dict[int, int] = {}
            try:
                counts = prescan_blf_frames(
                    self._data_file,
                    should_cancel=lambda: self._prescan_cancel,
                    error_counts=errors,
                )
            except Exception as e:  # 文件坏 / 无报文 / python-can 缺失：命中列退化为「—」
                logger.info("BLF 预扫未完成，命中列不可用: %s", e)
                self._prescan_box.append(None)
            else:
                self._prescan_errors = errors
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
        self._rebuild_channel_table()

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

        self.table = _PoolTable(self)
        self.table.setColumnCount(len(_COLUMNS))
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
        # 拖拽：行内调序（与 ↑/↓ 按钮共用 _move_row）+ 拖到通道行绑定
        self.table.setDragEnabled(True)
        self.table.setAcceptDrops(True)
        self.table.setDragDropMode(QTableWidget.DragDropMode.DragDrop)
        self.table.on_move = self._move_row_path
        self.table.on_external_accept = self._accept_external_drag
        self.table.on_external_drop = self._handle_external_drop
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

        hint = QLabel(
            "ℹ 同一报文 ID 在多库中重复时，排在上面的库生效。"
            "拖拽：库行上下拖 = 调优先级；拖到下方通道行 = 绑定。"
        )
        hint.setStyleSheet("color: #666;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # ---- 通道绑定分组（D1：多通道默认展开，单通道默认折叠）----------
        self.bind_box = QGroupBox("通道绑定")
        self.bind_box.setCheckable(True)
        self.bind_box.setChecked(True)
        group_layout = QVBoxLayout(self.bind_box)
        group_layout.setContentsMargins(8, 4, 8, 6)
        self.bind_content = QWidget()
        bind_layout = QVBoxLayout(self.bind_content)
        bind_layout.setContentsMargins(0, 0, 0, 0)
        bind_layout.setSpacing(4)
        group_layout.addWidget(self.bind_content)

        def _toggle_content(checked: bool) -> None:
            self._bind_user_toggled = True
            self.bind_content.setVisible(bool(checked))

        self.bind_box.toggled.connect(_toggle_content)

        self.bind_summary = QLabel("")
        self.bind_summary.setTextFormat(Qt.TextFormat.RichText)
        self.bind_summary.setWordWrap(True)
        bind_layout.addWidget(self.bind_summary)

        self.bind_table = _ChannelTable(self)
        self.bind_table.setColumnCount(5)
        self.bind_table.setHorizontalHeaderLabels(
            ["通道", "别名（可选）", "绑定库", "该通道命中", ""]
        )
        self.bind_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.bind_table.setSelectionMode(
            QTableWidget.SelectionMode.NoSelection
        )
        self.bind_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.bind_table.verticalHeader().setVisible(False)
        self.bind_table.setMinimumHeight(96)
        bind_header = self.bind_table.horizontalHeader()
        bind_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        bind_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.bind_table.setColumnWidth(0, 86)
        self.bind_table.setColumnWidth(1, 130)
        self.bind_table.setColumnWidth(3, 170)
        self.bind_table.setColumnWidth(4, 44)
        self.bind_table.on_bind = self._on_drag_bind
        self.bind_table.on_external_accept = self._accept_external_drag
        self.bind_table.on_external_drop = self._handle_external_drop
        bind_layout.addWidget(self.bind_table)

        bind_btn_row = QHBoxLayout()
        bind_btn_row.setSpacing(6)
        self.add_ch_btn = QPushButton("＋ 手动添加通道…")
        self.add_ch_btn.clicked.connect(self._on_add_channel)
        bind_btn_row.addWidget(self.add_ch_btn)
        self.reset_bind_btn = QPushButton("全部使用全局库")
        self.reset_bind_btn.clicked.connect(self._reset_all_bindings)
        bind_btn_row.addWidget(self.reset_bind_btn)
        bind_btn_row.addStretch(1)
        bind_layout.addLayout(bind_btn_row)

        self.bind_hint = QLabel("")
        self.bind_hint.setStyleSheet("color: #666; font-size: 11px;")
        self.bind_hint.setWordWrap(True)
        bind_layout.addWidget(self.bind_hint)
        layout.addWidget(self.bind_box)

        self.remember_check = QCheckBox("记住此配置，下次打开同一文件自动预选")
        self.remember_check.setChecked(True)
        layout.addWidget(self.remember_check)

        cfg_row = QHBoxLayout()
        cfg_row.setSpacing(6)
        self.save_cfg_btn = QPushButton("保存配置…")
        self.save_cfg_btn.clicked.connect(self._on_save_config)
        cfg_row.addWidget(self.save_cfg_btn)
        self.load_cfg_btn = QPushButton("载入配置…")
        self.load_cfg_btn.clicked.connect(self._on_load_config)
        cfg_row.addWidget(self.load_cfg_btn)
        cfg_row.addStretch(1)
        layout.addLayout(cfg_row)

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
        self._rebuild_channel_table()
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
        self._move_row(r, dst)

    def _move_row_path(self, path: str, dst: int) -> None:
        """拖拽调序：按路径定位源行，避免预扫自动排序把行号挪走。"""
        for src, row in enumerate(self._rows):
            if row.path == path:
                self._move_row(src, dst)
                return

    def _accept_external_drag(self, event) -> bool:
        return bool(self._drag_candidate_paths(event))

    def _handle_external_drop(self, event) -> bool:
        paths = self._drag_candidate_paths(event)
        if not paths:
            return False
        self._user_edited = True
        self.add_paths(paths)
        return True

    def _move_row(self, src: int, dst: int) -> None:
        """调序唯一入口：↑/↓ 按钮与池表拖拽都走这里（播报 parity）。"""
        if not (0 <= src < len(self._rows)) or src == dst:
            return
        dst = max(0, min(dst, len(self._rows) - 1))
        self._user_edited = True
        row = self._rows.pop(src)
        self._rows.insert(dst, row)
        self._rebuild_table()
        self._rebuild_channel_table()
        self.table.selectRow(dst)
        logger.info("CAN 数据库优先级调整: %s -> 第 %d 位",
                    os.path.basename(row.path), dst + 1)

    def _on_remove(self) -> None:
        r = self.table.currentRow()
        if not (0 <= r < len(self._rows)):
            return
        row = self._rows[r]
        dropped = [
            ch for ch, files in self._bindings.items() if row.path in files
        ]
        self._user_edited = True
        del self._rows[r]
        if dropped:
            # 被移除的库仍在绑定里：立即收敛并明确告知，不能静默放宽到全局池。
            # 措辞按收敛结果分支：还有别的绑定库 ≠ 退回全局池。
            for ch in dropped:
                self._set_channel_binding(
                    ch, [p for p in self._bindings.get(ch, ()) if p != row.path],
                    rebuild=False,
                )
            kept = [format_bus_channel(ch) for ch in dropped
                    if ch in self._bindings]
            freed = [format_bus_channel(ch) for ch in dropped
                     if ch not in self._bindings]
            parts = []
            if kept:
                parts.append(f"{'、'.join(kept)} 保留其余绑定库")
            if freed:
                parts.append(f"{'、'.join(freed)} 改为使用全局库")
            self._append_notice(
                f"库 {escape(os.path.basename(row.path))} 已被移除："
                + "；".join(parts) + "。"
            )
        self._rebuild_table()
        self._rebuild_channel_table()

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
    # 通道绑定（内部 0 基，显示 1 基；绑定 ⊂ 池）
    # ------------------------------------------------------------------

    def _pool_paths(self) -> list[str]:
        """可用于绑定的库（按优先级）。缺失/解析失败的行不参与。"""
        return [r.path for r in self._rows if r.ok and r.frame_count > 0]

    def _channel_numbers(self) -> list[int]:
        channels: set[int] = set(self._known_channels)
        channels.update(self._manual_channels)
        channels.update(self._bindings)
        channels.update(self._aliases)
        if self._prescan_counts:
            channels.update(ch for ch, _aid, _ext in self._prescan_counts)
        return sorted(channels)

    def _binding_allowed(self, channel: int, pool: dict[str, frozenset]) -> set[str]:
        bound = self._bindings.get(channel)
        return set(bound) if bound is not None else set(pool)

    def _channel_hit(self, channel: int) -> tuple[str, str]:
        if self._prescan_state != "done" or not self._prescan_counts:
            return "—（导入后可知）", "预扫不可用；导入后按绑定解码，未匹配 ID 会列在状态栏"
        pool = {
            r.path: r.db_ids for r in self._rows if r.ok and r.frame_count > 0
        }
        allowed = self._binding_allowed(channel, pool)
        ok = total = 0
        missing: list[str] = []
        pool_only: list[str] = []
        for (ch, aid, ext), n in self._prescan_counts.items():
            if ch != channel:
                continue
            total += n
            hex_id = f"0x{aid:X}" + ("E" if ext else "")
            defined = [p for p, ids in pool.items() if (aid, ext) in ids]
            if any(p in allowed for p in defined):
                ok += n
            else:
                missing.append(hex_id)
                if any(p not in allowed for p in defined):
                    pool_only.append(hex_id)
        pct = ok / total * 100 if total else 0.0
        tip = [f"按当前绑定可解出 {ok:,} / {total:,} 帧"]
        if missing:
            tip.append("未定义：" + "、".join(sorted(set(missing))[:8]))
        if pool_only:
            tip.append(
                "仅全局池有定义（按「不回落」不自动使用）："
                + "、".join(sorted(set(pool_only))[:8])
            )
        return f"{ok:,} / {total:,} 帧（{pct:.1f}%）", "\n".join(tip)

    def _duplicate_cross_channel_ids(self) -> list[str]:
        if not self._prescan_counts:
            return []
        seen: dict[tuple[int, bool], set[int]] = {}
        for ch, aid, ext in self._prescan_counts:
            seen.setdefault((aid, ext), set()).add(ch)
        return [
            f"0x{aid:X}" + ("E" if ext else "")
            for (aid, ext), chans in seen.items()
            if len(chans) > 1
        ]

    def _summary_text(self) -> str:
        if not self._channel_numbers():
            return (
                "尚无通道信息（大文件未预扫）。可手动添加通道行，"
                "或先导入后从「重选数据库」补绑定。"
            )
        if self._prescan_state == "running":
            return "正在后台预扫…完成后显示按绑定的命中率。"
        if self._prescan_state != "done" or not self._prescan_counts:
            return (
                "预扫不可用：导入后按当前绑定解码，未匹配 ID 会列在状态栏。"
            )
        pool = {
            r.path: r.db_ids for r in self._rows if r.ok and r.frame_count > 0
        }
        ok = total = 0
        missing: set[str] = set()
        pool_only: set[str] = set()
        for (ch, aid, ext), n in self._prescan_counts.items():
            total += n
            hex_id = f"0x{aid:X}" + ("E" if ext else "")
            allowed = self._binding_allowed(ch, pool)
            defined = [p for p, ids in pool.items() if (aid, ext) in ids]
            if any(p in allowed for p in defined):
                ok += n
            else:
                missing.add(hex_id)
                if any(p not in allowed for p in defined):
                    pool_only.add(hex_id)
        pct = ok / total * 100 if total else 0.0
        text = f"按当前绑定可解出 <b>{ok:,} / {total:,}</b> 帧（{pct:.1f}%）"
        if missing:
            text += f" · 未定义 ID {len(missing)} 个"
        if pool_only:
            text += (
                '<br><span style="color:#9a6b00">ⓘ '
                + "、".join(sorted(pool_only)[:8])
                + " 仅全局池其他库有定义，按「不回落」不会自动使用。</span>"
            )
        if not self._bindings:
            dup = self._duplicate_cross_channel_ids()
            if dup:
                text += (
                    '<br><span style="color:#9a6b00">⚠ '
                    + "、".join(dup[:6])
                    + " 跨通道同 ID：未绑定时按池顺序取定义，可能解错其中一路。</span>"
                )
        return text

    def _binding_button(self, channel: int) -> QPushButton:
        pool = self._pool_paths()
        current = self._bindings.get(channel)
        if current is None:
            text = "全部库（默认）"
        else:
            text = "、".join(os.path.basename(p) for p in current)
        btn = QPushButton(f"{text} ▾")
        btn.setToolTip("勾选只在该通道使用的库；全选 = 未绑定（用全局池）")
        btn.setStyleSheet("text-align: left; padding: 2px 6px;")
        menu = QMenu(btn)
        for path in pool:
            action = menu.addAction(os.path.basename(path))
            action.setCheckable(True)
            action.setChecked(current is None or path in current)
            action.toggled.connect(
                lambda checked, p=path, c=channel:
                self._toggle_binding(c, p, checked)
            )
        menu.addSeparator()
        menu.addAction("全选（用全局池）").triggered.connect(
            lambda _checked=False, c=channel: self._reset_channel_binding(c)
        )
        btn.setMenu(menu)
        return btn

    def _toggle_binding(self, channel: int, path: str, checked: bool) -> None:
        current = self._bindings.get(channel)
        paths = list(current) if current is not None else list(self._pool_paths())
        if checked and path not in paths:
            paths.append(path)
        elif not checked:
            paths = [p for p in paths if p != path]
        # 菜单属于即将被重建的按钮：先只改状态，等本轮事件收尾再重建，
        # 避免在 QAction.toggled 信号里销毁打开中的 QMenu。
        self._set_channel_binding(channel, paths, rebuild=False)
        QTimer.singleShot(0, self._rebuild_channel_table)

    def _set_channel_binding(self, channel: int,
                             paths: list[str] | tuple[str, ...],
                             *, rebuild: bool = True) -> None:
        """绑定变更唯一入口：勾选面板 / 拖拽 / 一键恢复都走这里。"""
        pool = self._pool_paths()
        allowed = [p for p in pool if p in set(paths)]
        if not allowed or set(allowed) == set(pool):
            self._bindings.pop(channel, None)
        else:
            self._bindings[channel] = tuple(allowed)
        logger.info(
            "通道 %s 绑定: %s",
            format_bus_channel(channel),
            "全局池" if channel not in self._bindings
            else "、".join(os.path.basename(p) for p in self._bindings[channel]),
        )
        if rebuild:
            self._rebuild_channel_table()

    def _reset_channel_binding(self, channel: int) -> None:
        self._set_channel_binding(channel, [])

    def _reset_all_bindings(self) -> None:
        self._bindings.clear()
        logger.info("已恢复：所有通道使用全局库")
        self._rebuild_channel_table()

    def _on_drag_bind(self, channel: int, path: str) -> None:
        if path not in self._pool_paths():
            return
        current = self._bindings.get(channel)
        if current is None:
            paths = [path]
        elif path in current:
            paths = list(current)
        else:
            paths = list(current) + [path]
        self._set_channel_binding(channel, paths)

    def _on_alias_edited(self, channel: int, text: str) -> None:
        text = (text or "").strip()
        if text:
            self._aliases[channel] = text
        else:
            self._aliases.pop(channel, None)
        # 只改第 0 列文本：重建表会销毁正在收尾的 QLineEdit（评审 P2）
        for row in range(self.bind_table.rowCount()):
            item = self.bind_table.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == channel:
                item.setText(format_bus_channel(channel, self._aliases.get(channel)))
                break

    def _on_add_channel(self) -> None:
        value, ok = QInputDialog.getInt(
            self, "手动添加通道",
            "通道号（1 基，与 CANoe / 记录仪一致）：", 1, 1, 9999,
        )
        if not ok:
            return
        channel = int(value) - 1
        if channel not in self._manual_channels:
            self._manual_channels.append(channel)
        self._rebuild_channel_table()

    def _on_remove_channel(self, channel: int) -> None:
        self._manual_channels = [c for c in self._manual_channels if c != channel]
        self._bindings.pop(channel, None)
        self._aliases.pop(channel, None)
        self._rebuild_channel_table()

    def _rebuild_channel_table(self) -> None:
        channels = self._channel_numbers()
        self.bind_table.blockSignals(True)
        self.bind_table.setRowCount(len(channels))
        for row, channel in enumerate(channels):
            errors = int(self._prescan_errors.get(channel, 0) or 0)
            text = format_bus_channel(channel, self._aliases.get(channel))
            if errors:
                text += f"  ⚠{errors}"
            item = QTableWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, channel)
            item.setToolTip(
                f"内部通道号 {channel}（0 基）"
                + (f" · 可识别错误帧 {errors:,}（不产生采样点）" if errors else "")
            )
            self.bind_table.setItem(row, 0, item)

            edit = QLineEdit(self._aliases.get(channel, ""))
            edit.setPlaceholderText("如：动力CAN")
            edit.editingFinished.connect(
                lambda c=channel, e=edit: self._on_alias_edited(c, e.text())
            )
            self.bind_table.setCellWidget(row, 1, edit)

            self.bind_table.setCellWidget(row, 2, self._binding_button(channel))

            hit_text, hit_tip = self._channel_hit(channel)
            hit_item = QTableWidgetItem(hit_text)
            hit_item.setToolTip(hit_tip)
            self.bind_table.setItem(row, 3, hit_item)

            if channel in self._manual_channels:
                remove = QPushButton("删")
                remove.setFixedWidth(34)
                remove.clicked.connect(
                    lambda _checked=False, c=channel: self._on_remove_channel(c)
                )
                self.bind_table.setCellWidget(row, 4, remove)
            else:
                self.bind_table.removeCellWidget(row, 4)
        self.bind_table.blockSignals(False)

        self.bind_table.setVisible(bool(channels))
        self.bind_summary.setText(self._summary_text())
        self.bind_hint.setText(
            "通道号填 CANoe / 记录仪里看到的编号（1 基）。"
            "已绑定通道只在其绑定库中找定义；找不到进未匹配列表，不会借用其他通道的库。"
        )
        if not self._bind_user_toggled:
            # D1：多通道默认展开；单通道 / 尚无通道信息时默认折叠（后者展开
            # 只为让「手动添加 / 先导入后重选」的出口可见）
            expand = len(channels) > 1 or not channels
            self.bind_box.blockSignals(True)
            self.bind_box.setChecked(expand)
            self.bind_box.blockSignals(False)
            self.bind_content.setVisible(expand)

    # ------------------------------------------------------------------
    # 显式配置：保存 / 载入（旁车 YAML 或用户配置目录）
    # ------------------------------------------------------------------

    def _on_save_config(self) -> None:
        cfg = self.get_result_config()
        if not cfg.pool:
            QMessageBox.warning(
                self, "没有可保存的配置", "请先添加可用的数据库文件。"
            )
            return
        if self._data_file:
            folder = os.path.dirname(os.path.abspath(self._data_file))
            default = (
                sidecar_config_path(self._data_file)
                if is_dir_writable(folder)
                else AppSettings.blf_binding_store_path(self._data_file)
            )
        else:
            default = ""
        path, _ = QFileDialog.getSaveFileName(
            self, "保存 CAN 配置", default,
            "CAN 配置 (*.canview.yaml *.yaml)",
        )
        if not path:
            return
        issues = save_yaml(cfg, path)
        if issues:
            self.set_notice("保存配置失败：" + "；".join(escape(i) for i in issues))
        else:
            self._append_notice(
                f"配置已保存：{escape(os.path.basename(path))}"
            )

    def _on_load_config(self) -> None:
        initial_dir = (
            os.path.dirname(os.path.abspath(self._data_file))
            if self._data_file else ""
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "载入 CAN 配置", initial_dir,
            "CAN 配置 (*.canview.yaml *.yaml)",
        )
        if not path:
            return
        cfg, issues = load_yaml(path, self._data_file)
        if cfg.pool:
            self.add_paths(cfg.pool)
            # P1-3（第 5 轮）：YAML 有序池 = 分享者的优先级意图。add_paths 只
            # 追加不重排，已有行会保持旧序 → first-match 取错库、静默解出错误
            # 物理值。按载入顺序重排已有行；YAML 未包含的行保持相对顺序垫底
            # （不删除，非破坏性）。精确字符串匹配与 add_paths 同口径。
            order: dict[str, int] = {}
            for i, p in enumerate(cfg.pool):
                # setdefault 保留首次出现索引：重复路径按 first-match 语义
                order.setdefault(normalize_input_path(str(p)), i)
            self._rows.sort(key=lambda r: order.get(r.path, len(order)))
            self._rebuild_table()
            # 显式载入 = 用户意图：预扫完成后的自动排序不得覆盖（与拖拽/勾选
            # 等手动操作同口径置位）。
            self._user_edited = True
        pool = set(self._pool_paths())
        self._bindings = {
            ch: tuple(p for p in files if p in pool)
            for ch, files in cfg.by_channel.items()
        }
        self._bindings = {
            ch: files for ch, files in self._bindings.items() if files
        }
        self._aliases = dict(cfg.alias)
        self._rebuild_channel_table()
        message = f"已载入配置：{escape(os.path.basename(path))}"
        if issues:
            message += "<br>" + "；".join(escape(i) for i in issues)
        self._append_notice(message)

    def get_result_config(self) -> CanBindings:
        """按当前池顺序 + 绑定 + 别名生成配置（绑定自动收敛到可用池）。"""
        pool = self._pool_paths()
        allowed = set(pool)
        by_channel: dict[int, tuple[str, ...]] = {}
        for channel, files in self._bindings.items():
            subset = tuple(p for p in files if p in allowed)
            # 初始配置里的「绑定 = 当前全池」原样保留（将来加库时不应悄悄放宽）；
            # 只有 UI 操作（勾选/拖拽）才会把「全选」归一成未绑定。
            if subset:
                by_channel[channel] = subset
        return CanBindings.from_paths(pool, by_channel, self._aliases)

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
        if self._handle_external_drop(event):
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
