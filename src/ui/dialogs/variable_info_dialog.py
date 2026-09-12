"""变量信息窗口 —— 全局单例 + 标签页累积。

交互模型：
    变量列表右键「变量信息」→ 单选弹一个标签页，多选每个变量一个标签页。
    窗口已存在时新变量**追加**为标签页，已存在的变量直接激活其标签页。
    标签可逐个关闭；只剩一个标签时隐藏标签栏（视觉上退化为单变量窗口）。
    关闭窗口等同「关闭全部」：清空所有标签页并取消后台统计，多标签时
    先弹确认。刻意不保留：保留会让“关窗后重新右键打开”与旧标签页混在
    一起（实测变成 2 个 tab），甚至加载新数据后旧变量名仍挂在标签上。

线程模型（实测数据驱动）：
    元数据快照构建**零磁盘 I/O**（六组块属性全访问约 99 μs），在 UI 线程
    同步完成，因此窗口打开即有内容，不需要"加载中"占位。
    统计特征需读磁盘（776 MB .mf4 的 428k 点通道约 18.8 ms，为元数据的
    190 倍），交给 ``VarInfoWorker`` 后台线程逐条计算、逐条回填，
    任一时刻 UI 线程都不被阻塞。

缓存与失效：
    只缓存统计（``main_window.var_stats_cache``），不缓存快照 —— 快照重建
    仅 99 μs，缓存它反而引入 sections 陈旧风险。每条统计自带 generation
    令牌，reload 后版本递增，陈旧结果即使到达也会被丢弃。
"""

from __future__ import annotations

import threading
import weakref
from collections import deque

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.config import (
    VAR_INFO_COL0_MIN_WIDTH,
    VAR_INFO_MAX_TABS,
    VAR_INFO_STATS_CACHE_MAX,
)
from src.core.logger import get_logger
from src.data import var_info
from src.data.metadata import VALID, CONST, INVALID
from src.ui.variable_actions import (
    add_variables_to_blank_plot,
    add_variables_to_data_table,
)

logger = get_logger(__name__)


def _make_ref(obj):
    """weakref 失败时降级为强引用闭包，保证调用方接口一致。

    项目内 4 个 Loader 均为普通类（无 ``__slots__``）故支持 weakref；
    这里兜底是为了让测试中的替身对象也能直接传入。
    """
    try:
        return weakref.ref(obj)
    except TypeError:
        return lambda: obj


def _validity_color(validity: int) -> QColor | None:
    """与 ``NoHoverDelegate._get_validity_color`` 保持一致的绿/橙/红语义。"""
    if validity == VALID:
        return QColor(0, 200, 0)
    if validity == CONST:
        return QColor(255, 140, 0)
    if validity == INVALID:
        return QColor(255, 0, 0)
    return None


# ---------------------------------------------------------------------------
# 后台统计线程
# ---------------------------------------------------------------------------


class VarInfoWorker(QThread):
    """统计特征的后台计算线程（FIFO 队列 + 可取消）。

    持 loader 用 **weakref**：队列里若残留强引用，reload 后旧 loader
    （776 MB 量级的 ``_signal_cache``）会被拖住无法释放。
    """

    item_ready = Signal(str, object)  # (var_name, VarStats)
    progress = Signal(int, int)  # (已完成条数, 本批总数)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cond = threading.Condition()
        # 元素为 (var_name, loader_ref, generation)
        self._jobs: deque = deque()
        self._cancel: set = set()
        self._stop = threading.Event()
        self._current: str | None = None
        self._pending_total = 0
        self._done_count = 0

    # -- 生产者侧（UI 线程） ------------------------------------------------

    def submit(self, jobs: list) -> None:
        """提交任务并唤醒线程；线程未启动时自动启动。"""
        if not jobs or self._stop.is_set():
            return
        with self._cond:
            for job in jobs:
                self._jobs.append(job)
                # 重新提交同名变量时必须清除旧的取消标记，否则会被立即跳过
                self._cancel.discard(job[0])
            self._pending_total += len(jobs)
            self._cond.notify_all()
        if not self.isRunning():
            self.start()

    def cancel(self, var_name: str) -> None:
        """取消单个变量（关闭标签页时调用）。"""
        with self._cond:
            self._cancel.add(var_name)
            self._jobs = deque(j for j in self._jobs if j[0] != var_name)
            self._cond.notify_all()

    def cancel_all(self) -> None:
        """清空队列（reload / 窗口销毁时调用）。

        ``_cancel`` 必须一并清空：队列已空，残留的取消标记会让**下一次**
        同名 submit 被误判为已取消而永久跳过。随后单独把正在运行的那条
        重新加入，使其在下一个分块边界尽快退出。
        """
        with self._cond:
            self._jobs.clear()
            self._cancel.clear()
            self._pending_total = 0
            self._done_count = 0
            if self._current is not None:
                self._cancel.add(self._current)
            self._cond.notify_all()

    def shutdown(self, timeout_ms: int = 3000) -> None:
        """彻底终止线程。主窗口 closeEvent 调用，避免
        "QThread: Destroyed while thread is still running" 崩溃。
        """
        self._stop.set()
        self.cancel_all()
        if self.isRunning():
            self.wait(timeout_ms)

    def queue_size(self) -> int:
        with self._cond:
            return len(self._jobs)

    # -- 消费者侧（子线程） -------------------------------------------------

    def _should_cancel(self, var_name: str) -> bool:
        return var_name in self._cancel or self._stop.is_set()

    def run(self) -> None:
        while not self._stop.is_set():
            with self._cond:
                while not self._jobs and not self._stop.is_set():
                    # 带超时的 wait：即使 notify 丢失也能周期性复检 _stop
                    self._cond.wait(0.2)
                if self._stop.is_set():
                    break
                if not self._jobs:
                    continue
                var_name, loader_ref, generation = self._jobs.popleft()
                self._current = var_name

            if self._should_cancel(var_name):
                self._cancel.discard(var_name)
                self._advance_progress()
                continue

            loader = loader_ref()
            if loader is None:
                # loader 已被 GC（reload 后旧数据释放），静默跳过
                self._advance_progress()
                continue

            try:
                stats = var_info.compute_stats(loader, var_name, self._should_cancel)
            except Exception as e:  # noqa: BLE001 - 子线程须自行兜住所有异常
                logger.debug("统计线程异常 %s", var_name, exc_info=True)
                stats = var_info.VarStats(error=f"{type(e).__name__}: {e}")

            stats.generation = generation
            with self._cond:
                self._current = None
            self._advance_progress()
            self.item_ready.emit(var_name, stats)

        with self._cond:
            self._current = None

    def _advance_progress(self) -> None:
        with self._cond:
            self._done_count += 1
            done, total = self._done_count, self._pending_total
        self.progress.emit(done, total)


# ---------------------------------------------------------------------------
# 单变量信息页
# ---------------------------------------------------------------------------


class VarInfoPage(QWidget):
    """单个变量的信息页：标题条 + 分组属性树 + 联动工具条。"""

    def __init__(self, var_name: str, dialog: "VariableInfoDialog", parent=None):
        super().__init__(parent)
        self.var_name = var_name
        # dialog 已是本页的 Qt 祖先（addTab 会 reparent），此处仅作逻辑回指
        self._dialog = dialog
        self.snapshot: var_info.VarInfoSnapshot | None = None
        self.stats: var_info.VarStats | None = None
        self._stats_item: QTreeWidgetItem | None = None
        self._stale = False
        # 列宽只在首次渲染时按内容定一次，之后交给用户拖动（详见
        # _apply_col0_width）。_auto_sizing 用于区分“程序定宽”与“用户拖动”，
        # 否则 resizeColumnToContents 自己触发的 sectionResized 会被当成
        # 用户意图并广播到其它标签页。
        self._col0_sized = False
        self._auto_sizing = False

        self._build_ui()

    # -- UI 构建 ------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addLayout(self._build_header())

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["属性", "值"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setWordWrap(True)
        header = self.tree.header()
        # 「属性」列用 Interactive 而不是 ResizeToContents：后者的定义就是
        # “由内容决定、用户拖不动”，且每次展开/收起都要重算，表现为
        # 列宽自行跳动。改为 Interactive 后两个毛病一并消失。
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionsMovable(False)
        header.setMinimumSectionSize(VAR_INFO_COL0_MIN_WIDTH)
        header.setStretchLastSection(False)
        header.sectionResized.connect(self._on_section_resized)
        layout.addWidget(self.tree, 1)

        layout.addLayout(self._build_toolbar())

    def _build_header(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(8)

        # 有效性色块：与变量列表的绿/橙/红标识一致，便于跨窗口对应
        self.color_chip = QLabel()
        self.color_chip.setFixedSize(14, 14)
        bar.addWidget(self.color_chip)

        self.title_label = QLabel(self.var_name)
        font = self.title_label.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 1)
        self.title_label.setFont(font)
        self.title_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        bar.addWidget(self.title_label)

        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("color: #666;")
        self.summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        bar.addWidget(self.summary_label, 1)
        return bar

    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(6)

        btn_table = QPushButton("添加至数值变量表")
        btn_table.setToolTip("把本变量加入「变量数值表」窗口")
        btn_table.clicked.connect(self._on_add_to_table)
        bar.addWidget(btn_table)

        btn_plot = QPushButton("添加至空白绘图区")
        btn_plot.setToolTip("在当前布局中找到首个空白绘图区并绘制本变量")
        btn_plot.clicked.connect(self._on_add_to_plot)
        bar.addWidget(btn_plot)

        bar.addStretch(1)

        self.btn_refresh = QPushButton("刷新统计")
        self.btn_refresh.setToolTip("强制重新计算并覆盖缓存（忽略已缓存结果）")
        self.btn_refresh.clicked.connect(self._on_refresh_stats)
        bar.addWidget(self.btn_refresh)

        btn_copy = QPushButton("复制本页 Markdown")
        btn_copy.setToolTip("把本页全部内容（含统计）复制到剪贴板")
        btn_copy.clicked.connect(self._on_copy_self)
        bar.addWidget(btn_copy)

        shortcut = QShortcut(QKeySequence.StandardKey.Copy, self.tree)
        shortcut.activated.connect(self._on_copy_selection)
        return bar

    # -- 内容渲染 -----------------------------------------------------------

    def render(self, snapshot, stats) -> None:
        """渲染快照与统计。``snapshot`` 为 None 表示变量已不存在。"""
        self.snapshot = snapshot
        self.stats = stats
        self.tree.clear()
        self._stats_item = None

        if snapshot is None:
            self._stale = True
            self._render_error("变量不在当前数据中")
            self.btn_refresh.setEnabled(False)
            return
        self._stale = False
        # 非数值通道（字符串 / pandas category）无法统计，禁用刷新避免提交
        # 注定失败的任务
        self.btn_refresh.setEnabled(snapshot.is_numeric)
        if stats is None and not snapshot.is_numeric:
            # 这类变量永远不会被提交到统计队列，若不给出终态说明，
            # 「统计特征」会永远停在"计算中…"，让用户误以为后台还在算。
            # 全空列单独给文案：它本质上是 float 数据只是没值，说
            # "非数值列（object）" 会让用户以为自己的列类型被误判了。
            if snapshot.all_empty:
                reason = "该列全部为空值，无有效样本可统计"
            else:
                reason = f"非数值列（{snapshot.dtype or '-'}），不适用统计"
            stats = var_info.VarStats(error=reason)
            self.stats = stats

        color = _validity_color(snapshot.validity)
        if color is None:
            self.color_chip.setStyleSheet(
                "background-color: #cccccc; border: 1px solid #999;"
            )
        else:
            self.color_chip.setStyleSheet(
                f"background-color: {color.name()};"
                f"border: 1px solid {color.darker(130).name()};"
            )

        self.title_label.setText(snapshot.name)
        self.summary_label.setText(
            f"{snapshot.unit} · {snapshot.length} 点 · {snapshot.dtype or '-'} · "
            f"{var_info.validity_label(snapshot.validity)}"
        )

        # 统计特征置顶：这是用户最常看的内容，无需展开分组即可读到
        self._stats_item = QTreeWidgetItem(self.tree, ["统计特征", ""])
        self._stats_item.setFirstColumnSpanned(False)
        font = self._stats_item.font(0)
        font.setBold(True)
        self._stats_item.setFont(0, font)
        self._fill_stats_rows()

        for title, rows in snapshot.sections.items():
            top = QTreeWidgetItem(self.tree, [title, ""])
            top_font = top.font(0)
            top_font.setBold(True)
            top.setFont(0, top_font)
            for key, value in rows:
                child = QTreeWidgetItem(top, [str(key), str(value)])
                child.setToolTip(1, str(value))

        # 全部分组默认展开：MDF 原本的五个块（基本信息 / 通道 CN /
        # 通道组 CGB / 源信息 SB / 时间基准）已合并为单个「基本信息」，
        # 顶层分组从 8 个降到 4 个（统计特征 + 3 个 sections）；实测真实
        # 通道总行数 57 → 41。原先“全展开会把关键信息挤出可视区”的
        # 前提已不成立。最长的枚举映射受
        # VAR_INFO_ENUM_DISPLAY_LIMIT=200 封顶且行高统一，展开代价可控。
        self.tree.expandAll()
        self._apply_col0_width()

    def _apply_col0_width(self) -> None:
        """首次渲染时按内容定一次「属性」列宽，此后不再自动重算。

        只定一次是关键：异步统计回填走 ``_fill_stats_rows`` 而不走
        ``render``，若在那里重算就会把用户手动调好的宽度顶掉。
        同一窗口内已有用户调过的宽度时直接沿用，切标签页不会跳回去。
        """
        if self._col0_sized:
            return
        self._col0_sized = True
        self._auto_sizing = True
        try:
            shared = self._dialog.shared_col0_width
            if shared:
                self.tree.setColumnWidth(0, shared)
            else:
                self.tree.resizeColumnToContents(0)
        finally:
            self._auto_sizing = False

    def _on_section_resized(self, index: int, _old: int, new: int) -> None:
        """用户拖动「属性」列时，把宽度共享给同窗口的其它标签页。"""
        if index != 0 or self._auto_sizing or self._dialog._syncing_col0:
            return
        self._dialog.set_shared_col0_width(new, origin=self)

    def _fill_stats_rows(self) -> None:
        """把统计结果写入「统计特征」节点。可反复调用（异步回填 / 刷新）。"""
        if self._stats_item is None:
            return
        self._stats_item.takeChildren()
        for key, value in var_info.stats_to_rows(self.stats):
            child = QTreeWidgetItem(self._stats_item, [str(key), str(value)])
            child.setToolTip(1, str(value))
        # 显式括号：and 优先级高于 or，此处语义为
        # 「stats 为空」或「尚未算完且无错误」→ 标注计算中
        pending = self.stats is None or (
            not self.stats.computed and not self.stats.error
        )
        self._stats_item.setText(1, "计算中…" if pending else "")

    def set_stats(self, stats) -> None:
        self.stats = stats
        self._fill_stats_rows()

    def _render_error(self, reason: str) -> None:
        self.tree.clear()
        self._stats_item = None
        top = QTreeWidgetItem(self.tree, ["状态", reason])
        top.setToolTip(1, reason)

    @property
    def is_stale(self) -> bool:
        return self._stale

    def to_markdown(self) -> str:
        if self.snapshot is None:
            return f"## 变量信息：{self.var_name}\n\n- 状态: 变量不在当前数据中\n"
        return var_info.snapshot_to_markdown(self.snapshot, self.stats)

    # -- 按钮回调 -----------------------------------------------------------

    def _main_window(self):
        return self._dialog._get_main_window()

    def _on_add_to_table(self) -> None:
        mw = self._main_window()
        if mw is None:
            return
        if self._stale:
            self._dialog._notify("变量已失效，无法添加")
            return
        add_variables_to_data_table([self.var_name], mw)

    def _on_add_to_plot(self) -> None:
        mw = self._main_window()
        if mw is None:
            return
        if self._stale:
            self._dialog._notify("变量已失效，无法添加")
            return
        add_variables_to_blank_plot([self.var_name], mw, msg_parent=self)

    def _on_refresh_stats(self) -> None:
        self._dialog.recompute(self.var_name, force=True)

    def _on_copy_self(self) -> None:
        QApplication.clipboard().setText(self.to_markdown())
        self._dialog._notify(f"已复制「{self.var_name}」的 Markdown")

    def _on_copy_selection(self) -> None:
        """Ctrl+C 复制树中选中的行（tab 分隔，可直接粘贴进表格）。"""
        items = self.tree.selectedItems()
        if not items:
            return
        lines = [
            "\t".join(it.text(c) for c in range(self.tree.columnCount()))
            for it in items
        ]
        QApplication.clipboard().setText("\n".join(lines))


# ---------------------------------------------------------------------------
# 单例对话框
# ---------------------------------------------------------------------------


class VariableInfoDialog(QDialog):
    """变量信息窗口（全局单例，标签页累积）。"""

    _instance: "VariableInfoDialog | None" = None

    # -- 生命周期 -----------------------------------------------------------

    @classmethod
    def _live_instance(cls) -> "VariableInfoDialog | None":
        """返回仍然存活的单例；C++ 对象已销毁时顺手清空引用。

        单例的 parent 是主窗口，主窗口销毁会连带销毁本对话框的 C++
        对象，但类属性 ``_instance`` 仍持有 Python 包装器。此时任何 Qt
        方法调用都会抛 ``RuntimeError: Internal C++ object already
        deleted``；若不回收引用，下一次 popup 会直接崩溃。
        """
        dlg = cls._instance
        if dlg is None:
            return None
        try:
            dlg.isVisible()
        except RuntimeError:
            cls._instance = None
            return None
        return dlg

    @classmethod
    def popup(cls, var_names, parent=None) -> "VariableInfoDialog | None":
        """打开（或复用）窗口并追加变量标签页。

        返回 dialog 实例；``parent`` 为主窗口，用于解析 loader 与保存几何。
        """
        names = [n for n in (var_names or []) if n]
        if not names:
            return None

        dlg = cls._live_instance()
        if dlg is None:
            dlg = cls._instance = cls(parent)
        else:
            dlg._update_owner(parent)

        dlg.load_geom()
        dlg.add_variables(names)
        dlg.show()
        if dlg.isMinimized():
            dlg.showNormal()
        dlg.raise_()
        dlg.activateWindow()
        return dlg

    @classmethod
    def on_loader_released(cls) -> None:
        """旧 loader 即将释放时调用（``_release_old_data`` 内，**释放之前**）。

        必须先取消后台任务：worker 队列里的条目持 loader 的 weakref，
        但**正在运行**的那条已把 loader 取到局部变量，若不取消会继续读
        一个即将被 close() 的文件句柄（改进 I 保证此时抛 KeyError 降级，
        但取消可以让它立即停下，不做无谓 I/O）。
        """
        dlg = cls._live_instance()
        if dlg is None:
            return
        try:
            dlg.worker.cancel_all()
        except Exception:
            logger.debug("取消变量信息统计任务失败", exc_info=True)

    @classmethod
    def refresh_after_reload(cls, loader) -> None:
        """新 loader 就位后调用（``_apply_loader`` 末尾）：重建所有页面。

        仍存在的变量原地重建（用户 reload 的往往是同一批测量，原地刷新
        才能对比新旧）；新数据中已不存在的变量**摘掉标签页**并在状态栏
        告知数量 —— 留一个标着"(失效)"的旧变量名既点不动也统计不出，
        只会让人以为换了文件后窗口没跟上。
        """
        dlg = cls._live_instance()
        if dlg is None or loader is None:
            return
        try:
            dlg._rebuild_all_pages(loader)
        except Exception:
            logger.debug("刷新变量信息窗口失败", exc_info=True)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("变量信息")
        self.setModal(False)
        # 独立窗口按钮（关闭语义见 closeEvent：清空所有标签页）
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )
        self.resize(760, 640)

        self._owner_ref = None
        self._pages: dict[str, VarInfoPage] = {}
        # 用户手动调过的「属性」列宽，同窗口内所有标签页共享。
        # 0 表示“用户还没拖过”，此时新建页按自己的内容定宽。
        self.shared_col0_width = 0
        # 应用退出流程标记：由 shutdown_worker() 置位（主窗口 closeEvent
        # 会调它）。退出时不得弹模态确认框，否则会把关闭流程卡住。
        self._shutting_down = False
        # 广播宽度时的重入闸：setColumnWidth 会反过来触发对方的
        # sectionResized，不拦住会形成页与页之间的信号往返。
        self._syncing_col0 = False
        # 状态栏由两部分组成，必须分开保存：
        #   _notice        —— 用户必须看到的一次性提示（如“标签页已达上限”）
        #   _progress_text —— 后台统计的实时进度，随任务推进不断刷新
        # 共用一个字段时，异步的进度信号会在数十毫秒内把提示覆盖掉
        # （实测：提交 2 个统计任务后，“已达上限”提示从未显示过）。
        self._notice = ""
        self._progress_text = ""
        self._update_owner(parent)

        self.worker = VarInfoWorker(self)
        self.worker.item_ready.connect(self._on_stats_ready)
        self.worker.progress.connect(self._on_progress)

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.tabCloseRequested.connect(self._on_tab_close)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)
        root.addWidget(self.tabs, 1)
        root.addLayout(self._build_status_bar())

        self._sync_tab_bar_visibility()

    def _build_status_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666;")
        bar.addWidget(self.status_label, 1)

        btn_copy_all = QPushButton("复制全部 Markdown")
        btn_copy_all.setToolTip("把所有标签页的内容复制到剪贴板（以分隔线拼接）")
        btn_copy_all.clicked.connect(self._on_copy_all)
        bar.addWidget(btn_copy_all)

        btn_close_all = QPushButton("关闭全部")
        btn_close_all.clicked.connect(self._close_all_tabs)
        bar.addWidget(btn_close_all)
        return bar

    # -- owner / loader 解析 ------------------------------------------------

    def _update_owner(self, widget) -> None:
        window = None
        if isinstance(widget, QWidget):
            window = widget.window()
        if window is None:
            window = widget
        self._owner_ref = _make_ref(window) if window is not None else None

    def _get_main_window(self):
        if self._owner_ref is not None:
            window = self._owner_ref()
            if window is not None and hasattr(window, "loader"):
                return window
        active = QApplication.activeWindow()
        if isinstance(active, QMainWindow) and hasattr(active, "loader"):
            return active
        parent = self.parent()
        if parent is not None and hasattr(parent, "loader"):
            return parent
        return None

    def _resolve_loader(self):
        mw = self._get_main_window()
        return getattr(mw, "loader", None) if mw is not None else None

    def _generation(self) -> int:
        mw = self._get_main_window()
        return int(getattr(mw, "_data_version", 0)) if mw is not None else 0

    # -- 几何信息 -----------------------------------------------------------

    def save_geom(self) -> None:
        mw = self._get_main_window()
        if mw is not None and hasattr(mw, "var_info_geometry"):
            mw.var_info_geometry = self.saveGeometry()

    def load_geom(self) -> None:
        mw = self._get_main_window()
        if mw is None or not hasattr(mw, "var_info_geometry"):
            return
        geom = mw.var_info_geometry
        if geom is not None:
            self.restoreGeometry(geom)

    # -- 标签页管理 ---------------------------------------------------------

    def add_variables(self, names) -> None:
        """追加标签页并触发统计。已存在的变量直接激活其标签页。

        受 ``VAR_INFO_MAX_TABS`` 约束：标签页总数达上限后，本次提交中
        超出的部分被截断并提示。这不是理论风险：在变量列表里全选后
        右键是很容易发生的误操作，990 通道的 .dat 全选即 990 个标签页，
        每页都要建 QTreeWidget 并渲染十几个分组，UI 会直接卡死。
        """
        loader = self._resolve_loader()
        if loader is None:
            self._notify("尚未加载数据文件")
            return

        # 去重且保序：重复名字只对应同一个标签页，不得占用新增额度
        unique = list(dict.fromkeys(names))
        fresh = [n for n in unique if n not in self._pages]

        capacity = max(VAR_INFO_MAX_TABS - len(self._pages), 0)
        # 截断按提交顺序保留靠前的；用 set 做名单而不直接切片 fresh，
        # 是为了下面的主循环仍能按原始提交顺序遍历（否则“激活最后
        # 提到的那个标签”这一语义会变成“总是激活最后一个新建页”）。
        allowed = set(fresh[:capacity])
        notice = ""
        if len(fresh) > capacity:
            notice = (
                f"标签页已达上限 {VAR_INFO_MAX_TABS}，"
                f"本次忽略 {len(fresh) - capacity} 个变量"
            )

        generation = self._generation()
        cache = self._cache()
        jobs = []
        last_index = -1

        for name in unique:
            page = self._pages.get(name)
            if page is not None:
                last_index = self.tabs.indexOf(page)
                continue
            if name not in allowed:
                continue  # 被上限截断

            snapshot = self._safe_snapshot(loader, name, generation)
            page = VarInfoPage(name, self)
            stats = self._lookup_cache(cache, name, generation)
            if stats is not None:
                # 命中缓存：标记为缓存来源，UI 显示"（缓存）"
                stats.cached = True
            page.render(snapshot, stats)
            self._pages[name] = page
            last_index = self.tabs.addTab(page, self._tab_title(name, snapshot))

            if stats is None and snapshot is not None and snapshot.is_numeric:
                jobs.append((name, _make_ref(loader), generation))

        self._sync_tab_bar_visibility()
        if last_index >= 0:
            self.tabs.setCurrentIndex(last_index)

        if jobs:
            self.worker.submit(jobs)
            self._progress_text = f"统计中 0/{len(jobs)}…"
        # 每次都重置提示：截断信息属于本次提交，不得沿用上一轮的
        self._notice = notice
        self._refresh_status()

    def _tab_title(self, name: str, snapshot) -> str:
        """标签标题带上单位，便于同名不同单位的变量区分。"""
        if snapshot is None:
            return f"{name} (失效)"
        unit = snapshot.unit
        return f"{name} [{unit}]" if unit and unit != "-" else name

    def _safe_snapshot(self, loader, name, generation):
        """快照构建失败（变量不存在 / 数据异常）时返回 None 而不抛。

        构建过程零磁盘 I/O（约 99 μs），因此在 UI 线程同步调用是安全的。
        """
        try:
            return var_info.build_snapshot(loader, name, generation)
        except KeyError:
            return None
        except Exception:
            logger.debug("构建 %s 快照失败", name, exc_info=True)
            return None

    def _on_tab_close(self, index: int) -> None:
        page = self.tabs.widget(index)
        if page is None:
            return
        name = getattr(page, "var_name", "")
        # 先取消后台任务再摘除页面：顺序颠倒会让结果回填到已销毁的页面上
        self.worker.cancel(name)
        self._remove_page(name)
        self._sync_tab_bar_visibility()
        if self.tabs.count() == 0:
            self.hide()

    def _remove_page(self, name: str) -> None:
        """摘除并**销毁**指定变量的标签页（不取消后台任务，由调用方决定）。

        必须 deleteLater：实测 Qt 6.11.1 下 removeTab / tabs.clear() 都只
        摘掉标签而保留页面，页面仍作为 QStackedWidget 的子对象驻留内存
        直到整个对话框销毁。
        """
        page = self._pages.pop(name, None)
        if page is None:
            return
        index = self.tabs.indexOf(page)
        if index >= 0:
            self.tabs.removeTab(index)
        page.deleteLater()

    def _on_tab_changed(self, index: int) -> None:
        page = self.tabs.widget(index)
        if page is not None:
            # 用户已切换关注点，清掉一次性提示（进度文本保留）
            self._notice = ""
            self._refresh_status()

    def _close_all_tabs(self) -> None:
        self.worker.cancel_all()
        # 逐个 removeTab + deleteLater，刻意不用 tabs.clear()：实测 clear() 不
        # 销毁页面，它们会继续挂在 QStackedWidget 下。关窗即清空后这条
        # 路径从“偶尔点按钮”变成“每次关窗”，不销毁就会逐次累积。
        # 以 tabs 而不是 _pages 为遍历源：前者是真正需要清空的容器。
        while self.tabs.count():
            page = self.tabs.widget(0)
            self.tabs.removeTab(0)
            if page is not None:
                page.deleteLater()
        self._pages.clear()
        # 窗口隐藏后下次仍会复用同一实例，不清会看到陈旧的提示/进度
        self._notice = ""
        self._progress_text = ""
        self._refresh_status()
        self._sync_tab_bar_visibility()
        self.hide()

    def set_shared_col0_width(self, width: int, origin=None) -> None:
        """把用户拖出的「属性」列宽同步到本窗口所有标签页。

        不同步的话，用户在 A 页调好宽度、切到 B 页会看到完全不同的列宽，
        视觉上像“设置没生效”。刻意不跨会话持久化：这属于临时阅读偏好，
        且不同变量的标签长度差异很大，记住一个宽度反而可能不合用。
        """
        self.shared_col0_width = width
        self._syncing_col0 = True
        try:
            for page in self._pages.values():
                if page is not origin:
                    page.tree.setColumnWidth(0, width)
        finally:
            self._syncing_col0 = False

    def _sync_tab_bar_visibility(self) -> None:
        """只剩一个标签页时隐藏标签栏，视觉上退化为单变量窗口。"""
        self.tabs.tabBar().setVisible(self.tabs.count() > 1)

    # -- 缓存 ---------------------------------------------------------------

    def _cache(self) -> dict | None:
        """返回主窗口的统计缓存字典；不可用时返回 **None**。

        必须用 None 而不是空字典表示"不可用"：空字典是 falsy，
        会让 ``_store_cache`` 把"缓存尚空"误判为"无处可存"而直接返回，
        导致缓存永远写不进任何条目。
        """
        mw = self._get_main_window()
        cache = getattr(mw, "var_stats_cache", None) if mw is not None else None
        return cache if isinstance(cache, dict) else None

    @staticmethod
    def _lookup_cache(cache: dict | None, name: str, generation: int):
        """命中条件：键存在 **且** generation 与当前数据版本一致。

        generation 校验是失效的第二道保险 —— 即使 reload 时忘记清空缓存，
        陈旧条目也不会被采用。
        """
        if not cache:
            return None
        stats = cache.get(name)
        if stats is None:
            return None
        if getattr(stats, "generation", -1) != generation:
            cache.pop(name, None)
            return None
        if not getattr(stats, "computed", False) and not getattr(stats, "error", ""):
            return None
        return stats

    def _store_cache(self, name: str, stats) -> None:
        cache = self._cache()
        if cache is None:
            # 主窗口已销毁或属性缺失：无处可存，静默放弃（统计结果仍会回填 UI）
            return
        cache[name] = stats
        overflow = len(cache) - VAR_INFO_STATS_CACHE_MAX
        if overflow > 0:
            # dict 自 Python 3.7 起保序，故最前面的键即最早写入者（FIFO 淘汰）
            for key in list(cache.keys())[:overflow]:
                cache.pop(key, None)

    # -- 统计回填 -----------------------------------------------------------

    def _on_stats_ready(self, name: str, stats) -> None:
        generation = self._generation()
        if getattr(stats, "generation", -1) != generation:
            # reload 期间完成的陈旧结果：既不写缓存也不回填，直接丢弃
            return

        self._store_cache(name, stats)

        page = self._pages.get(name)
        if page is None:
            return  # 标签页已被用户关闭
        page.set_stats(stats)
        if not stats.computed and stats.error:
            self._progress_text = f"{name}: {stats.error}"
            self._refresh_status()

    def _on_progress(self, done: int, total: int) -> None:
        if total <= 0 or done >= total:
            if self.worker.queue_size() == 0:
                self._progress_text = ""
        else:
            self._progress_text = f"统计中 {done}/{total}…"
        self._refresh_status()

    def recompute(self, name: str, force: bool = False) -> None:
        """重新计算统计。``force=True`` 时先丢弃缓存条目。"""
        loader = self._resolve_loader()
        page = self._pages.get(name)
        if loader is None or page is None:
            self._notify("尚未加载数据文件")
            return
        if page.is_stale:
            self._notify("变量已失效，无法重新统计")
            return

        generation = self._generation()
        if force:
            cache = self._cache()
            if cache is not None:
                cache.pop(name, None)
            self.worker.cancel(name)
            page.set_stats(None)
        self.worker.submit([(name, _make_ref(loader), generation)])
        self._progress_text = "统计中 0/1…"
        self._refresh_status()

    # -- reload 后重建 ------------------------------------------------------

    def _rebuild_all_pages(self, loader) -> None:
        self.worker.cancel_all()
        generation = self._generation()
        cache = self._cache()
        jobs = []
        gone = []

        for name, page in self._pages.items():
            # 旧数据的统计一律作废：键相同但数值可能完全不同
            if cache is not None:
                cache.pop(name, None)
            snapshot = self._safe_snapshot(loader, name, generation)
            if snapshot is None:
                # 迭代中不得改 _pages（下面要摘除页面），先记下名字
                gone.append(name)
                continue
            page.render(snapshot, None)
            self.tabs.setTabText(self.tabs.indexOf(page), self._tab_title(name, snapshot))
            if snapshot.is_numeric:
                jobs.append((name, _make_ref(loader), generation))

        for name in gone:
            self._remove_page(name)
        self._sync_tab_bar_visibility()
        if self.tabs.count() == 0:
            self.hide()

        if jobs:
            self.worker.submit(jobs)
            self._progress_text = f"统计中 0/{len(jobs)}…"
        else:
            self._progress_text = ""
        # 摘除后要告知：静默消失会让“我刚才明明开了 3 个”变成无从解释
        self._notice = (
            f"已移除 {len(gone)} 个新数据中不存在的变量标签页" if gone else ""
        )
        self._refresh_status()

    # -- 其他 ---------------------------------------------------------------

    def _on_copy_all(self) -> None:
        items = [(p.snapshot, p.stats) for p in self._ordered_pages() if p.snapshot]
        if not items:
            self._notify("没有可复制的内容")
            return
        text = var_info.snapshots_to_markdown(items)
        QApplication.clipboard().setText(text)
        self._notify(f"已复制 {len(items)} 个变量的 Markdown")

    def _ordered_pages(self) -> list:
        """按标签页当前顺序返回页面（用户可拖动调整顺序，导出应尊重之）。"""
        pages = []
        for i in range(self.tabs.count()):
            page = self.tabs.widget(i)
            if isinstance(page, VarInfoPage):
                pages.append(page)
        return pages

    def _notify(self, text: str) -> None:
        """显示一条用户提示，直到用户切换标签页或下一次提交才被替换。

        不走 ``status_label.setText``：那会被异步的进度信号立即覆盖。
        """
        self._notice = text
        self._refresh_status()

    def _refresh_status(self) -> None:
        """拼接提示与进度写入状态栏（提示在前，因为它更重要）。"""
        parts = [p for p in (self._notice, self._progress_text) if p]
        self.status_label.setText("　".join(parts))

    def shutdown_worker(self) -> None:
        """主窗口关闭时调用：确保 QThread 不在运行中被销毁。"""
        # 先置退出标记：closeEvent 靠它判定“现在不该弹确认框”。
        # 刻意不用 QApplication.closingDown()：实测 closeAllWindows() 期间
        # 它仍为 False（Qt 只在 ~QCoreApplication 里置位），拦不住。
        # 而主窗口 closeEvent 会先调到这里，标记因此总是先于关窗就位。
        self._shutting_down = True
        try:
            self.worker.shutdown()
        except Exception:
            logger.debug("终止统计线程失败", exc_info=True)

    def closeEvent(self, event) -> None:
        """关窗等同「关闭全部」：清空所有标签页，多标签时先弹确认。

        刻意不保留标签页：旧实现只 hide()，于是"关窗 → 再右键打开另一个
        变量"会变成 2 个 tab，用户无从预期。后台任务必须一并取消：页面
        已销毁，统计结果无处可去（_on_stats_ready 虽有 page is None 防护，
        但让它继续读磁盘纯属浪费）。

        单个标签页不弹确认：关了再开就是它自己，没有误操作损失。
        退出流程中也不弹（``_shutting_down`` 由 shutdown_worker 置位，主窗口
        closeEvent 会先调它）：实测主窗口关闭并不会给子对话框发
        closeEvent，但 closeAllWindows() 会，那时弹模态框会把退出卡住。
        """
        count = self.tabs.count()
        if count > 1 and not self._shutting_down:
            ret = QMessageBox.question(
                self,
                "关闭变量信息",
                f"当前有 {count} 个变量标签页，关闭窗口将全部清空。\n"
                "确定关闭吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self.save_geom()
        self._close_all_tabs()
        event.accept()

    @classmethod
    def reset_for_tests(cls) -> None:
        """仅供自动化测试：彻底销毁单例与后台线程。"""
        dlg = cls._instance
        cls._instance = None
        if dlg is None:
            return
        try:
            dlg.shutdown_worker()
        except Exception:
            logger.debug("测试清理：终止统计线程失败", exc_info=True)
        try:
            dlg.hide()
            dlg.deleteLater()
        except RuntimeError:
            # C++ 对象已随父窗口销毁，无需也无法再释放
            pass
