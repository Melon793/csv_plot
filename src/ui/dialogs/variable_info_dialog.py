"""变量信息窗口 —— 全局单例 + 标签页累积。

交互模型：
    变量列表右键「变量信息」→ 单选弹一个标签页，多选每个变量一个标签页。
    窗口已存在时新变量**追加**为标签页，已存在的变量直接激活其标签页。
    标签可逐个关闭；只剩一个标签时隐藏标签栏（视觉上退化为单变量窗口）。
    关闭窗口等同「关闭全部」：清空所有标签页并取消后台统计，多标签时
    先弹确认。刻意不保留：保留会让“关窗后重新右键打开”与旧标签页混在
    一起（实测变成 2 个 tab），甚至加载新数据后旧变量名仍挂在标签上。

线程模型（实测数据驱动）：
    元数据快照构建**零磁盘 I/O 且 O(1)**（MDF 路径六组块属性全访问约
    99 μs；表格路径只读 pandas 的 dtype/categories/长度，不物化列数据），
    在 UI 线程同步完成，因此窗口打开即有内容，不需要"加载中"占位。
    统计特征需读磁盘（776 MB .mf4 的 428k 点通道约 18.8 ms，为元数据的
    190 倍），交给 ``VarInfoWorker`` 后台线程逐条计算、逐条回填，
    任一时刻 UI 线程都不被阻塞。

缓存与失效：
    只缓存统计（``main_window.var_stats_cache``），不缓存快照 —— 快照重建
    零磁盘 I/O 且 O(1)，缓存它反而引入 sections 陈旧风险。每条统计自带
    generation 令牌，reload 后版本递增，陈旧结果即使到达也会被丢弃。

复制粒度：
    三个入口各管一档 —— 单个字段值走值列行尾的悬停复制按钮
    （``CopyFieldDelegate``），整页走「复制本页 Markdown」，多页走
    「复制全部 Markdown」；``Ctrl+C`` 复制树里选中的若干行（tab 分隔）。
"""

from __future__ import annotations

import threading
import weakref
from collections import deque
from dataclasses import replace

from PySide6.QtCore import (
    QEvent,
    QRect,
    Qt,
    QThread,
    QPersistentModelIndex,
    Signal,
)
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStyleOptionViewItem,
    QStyledItemDelegate,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.config import (
    PATH_COPY_QUOTE,
    PATH_COPY_STYLE,
    VAR_INFO_COL0_MIN_WIDTH,
    VAR_INFO_COPY_BTN_MARGIN,
    VAR_INFO_COPY_BTN_SIZE,
    VAR_INFO_MAX_TABS,
    VAR_INFO_ROW_HEIGHT,
    VAR_INFO_STATS_CACHE_MAX,
)
from src.core.logger import get_logger
from src.core.gc_guard import no_autogc
from src.data import var_info
from src.data.metadata import VALID, CONST, INVALID
from src.utils.paths import STYLE_POSIX, STYLE_WINDOWS, format_for_copy
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
# 「值」列悬停复制按钮
# ---------------------------------------------------------------------------

# 按钮所在列与无内容占位文本。「值」列的占位短横由 var_info._fmt 对
# None/空白统一产出，给它画一个复制按钮等于让用户复制一个无意义的 "-"
_COPY_COLUMN = 1
_DASH_PLACEHOLDER = "-"


def copy_button_rect(
    cell_rect: QRect,
    size: int | None = None,
    margin: int | None = None,
) -> QRect:
    """值列单元格里复制按钮的热区：右端对齐、垂直居中的正方形。

    paint 与 editorEvent 必须共用本函数，并且都从**未经收缩的整格 rect**
    求值：委托在画文本时会把文本区右边收掉一个按钮宽，若热区改用收缩后
    的 rect 计算，命中区就会与看到的图标错位若干个 margin。

    size/margin 的默认值刻意取 None 而不是直接写常量：默认参数在 def
    执行时就绑定，等于把 config 的值焊死在这里，而 paint 里的文本区收缩
    读的是运行时全局 —— 两处会错位。实测踩坑：monkeypatch 常量后按钮
    边长仍是旧值，据此做的“谁决定行高”实验数据全废。取 None 后两边
    都走运行时查找，不重启进程也能预览尺寸效果。
    """
    if size is None:
        size = VAR_INFO_COPY_BTN_SIZE
    if margin is None:
        margin = VAR_INFO_COPY_BTN_MARGIN
    # QRect 的 right() 是闭区间右边界（x + width - 1），故 +1 回退一个像素
    x = cell_rect.right() - margin - size + 1
    y = cell_rect.center().y() - size // 2
    return QRect(x, y, size, size)


class CopyFieldDelegate(QStyledItemDelegate):
    """在「值」列右端自绘复制图标，仅在鼠标悬停该行时出现。

    刻意不用 ``setItemWidget`` 挂 QPushButton：一页 40~60 行、最多
    ``VAR_INFO_MAX_TABS`` 个标签页，常驻控件数量会到数千个，而这份信息
    绝大多数时候只是看一眼。图标用 QPainter 画两张错开的圆角矩形，不走
    Unicode 字符：字形在 macOS/Windows 字体里的可用性不可控（Legend 方块
    已为此踩过坑），也不用往打包里加资源文件。
    """

    copy_requested = Signal(object)  # QModelIndex

    def __init__(self, parent=None):
        super().__init__(parent)
        # 悬停/按下态一律用 QPersistentModelIndex 而非 QModelIndex：
        # render() 会 tree.clear()、_fill_stats_rows() 会 takeChildren()，
        # 裸索引在行被移除后仍持有已销毁项的内部指针，再取用就是
        # “访问已删除 C++ 对象”那类崩溃的成因；持久索引会随模型复位
        # 自动失效。
        self._hover = QPersistentModelIndex()
        self._pressed = QPersistentModelIndex()

    # -- 状态 --------------------------------------------------------------

    @staticmethod
    def is_copyable(index) -> bool:
        """该行值列是否有可复制的内容（分组标题行与占位行没有）。"""
        if index.column() != _COPY_COLUMN:
            return False
        text = index.data(Qt.ItemDataRole.DisplayRole)
        if not isinstance(text, str) or not text:
            return False
        return text != _DASH_PLACEHOLDER

    def set_hover(self, index) -> bool:
        """更新悬停位置，返回是否变化（调用方据此决定要不要重绘）。"""
        return self._swap_hover(QPersistentModelIndex(index))

    def clear_hover(self) -> bool:
        """清除悬停态，返回是否发生变化。"""
        return self._swap_hover(QPersistentModelIndex())

    def _swap_hover(self, candidate: QPersistentModelIndex) -> bool:
        # 鼠标在同一行内移动是数量级更高的事件，只有跨行才值得整个
        # viewport 重绘；两个持久索引相等即位置未变
        if candidate == self._hover:
            return False
        self._hover = candidate
        return True

    # -- 绘制 --------------------------------------------------------------

    def paint(self, painter, option, index) -> None:
        show_btn = self.is_copyable(index)
        btn = copy_button_rect(option.rect) if show_btn else QRect()
        painter.save()
        try:
            if show_btn:
                # 复制一份再收缩：option 由视图在同一次绘制周期里复用，
                # 就地改写会污染后续列
                opt = QStyleOptionViewItem(option)
                # 预留宽度与悬停无关：若只在悬停时收缩，鼠标划过的那一行
                # 文本会在“截断位置前移/后移”之间反复跳，视觉上就是整列在抖
                opt.rect = option.rect.adjusted(
                    0, 0, -(VAR_INFO_COPY_BTN_SIZE + 2 * VAR_INFO_COPY_BTN_MARGIN), 0
                )
            else:
                opt = option
            super().paint(painter, opt, index)
            if show_btn and self._highlighted(index):
                pressed = QPersistentModelIndex(index) == self._pressed
                self._paint_icon(painter, btn, pressed)
        finally:
            painter.restore()

    def sizeHint(self, option, index):
        hint = super().sizeHint(option, index)
        if self.is_copyable(index):
            # 预留量计入理想宽度。实测 QTreeView 并不支持 word wrap（视图
            # 会忽略 setWordWrap(True)，长文本一律用省略号截断而非折行），
            # 所以此处今天不影响行高；保留是为了委托被复到支持折行的
            # 视图上时，预留宽度能算进行数而不是把第二行截掉
            hint.setWidth(
                hint.width() + VAR_INFO_COPY_BTN_SIZE + 2 * VAR_INFO_COPY_BTN_MARGIN
            )
        if VAR_INFO_ROW_HEIGHT > 0:
            # 必须对所有 index 生效：视图开着 setUniformRowHeights(True)，
            # 任意一行的高度都会被拉平成全表最大值（实测给分组行 40、
            # 数据行 24 的结果是全部 40），逐行给不同值在这里没有意义
            hint.setHeight(VAR_INFO_ROW_HEIGHT)
        return hint

    def _highlighted(self, index) -> bool:
        # 显式包装再比：QPersistentModelIndex 与 QModelIndex 的 == 依赖隐式
        # 转换，在 PySide6 上不可靠，走同类型比较
        persistent = QPersistentModelIndex(index)
        return persistent == self._hover or persistent == self._pressed

    def _paint_icon(self, painter: QPainter, rect: QRect, pressed: bool) -> None:
        """两张错开的圆角矩形：后层右上、前层左下。"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor("#333333") if pressed else QColor("#8a8a8a"))
        pen.setWidthF(1.2)
        painter.setPen(pen)
        # 后层不填充：选中行/交替行/深浅主题下底色各不相同，用任何固定底色
        # 去遮罩都会画出一块“补丁”；两张轮廓线错开在 16 px 下已足够可读
        painter.setBrush(Qt.BrushStyle.NoBrush)
        w = rect.width() - 6
        h = rect.height() - 4
        painter.drawRoundedRect(
            QRect(rect.left() + 5, rect.top(), w, h), 1.5, 1.5
        )
        if pressed:
            painter.setBrush(QColor(33, 150, 243, 70))
        painter.drawRoundedRect(
            QRect(rect.left() + 1, rect.top() + 4, w, h), 1.5, 1.5
        )

    # -- 交互 --------------------------------------------------------------

    def editorEvent(self, event, model, option, index) -> bool:
        """处理热区内的左键按下/释放，其余事件一律放行给视图。"""
        et = event.type()
        if et not in (
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.MouseButtonDblClick,
        ):
            return False
        if not self.is_copyable(index):
            return False
        btn = copy_button_rect(option.rect)
        pos = event.position().toPoint()

        if et == QEvent.Type.MouseButtonPress:
            if event.button() != Qt.MouseButton.LeftButton or not btn.contains(pos):
                return False
            self._pressed = QPersistentModelIndex(index)
            self._repaint(option)
            # 返回 True 吞掉这一下：复制是“读”操作，不该顺带把用户原本的
            # 选区弄乱（按下未吞的话，紧随其后的 release 会把该行选中）
            return True

        if et == QEvent.Type.MouseButtonDblClick:
            # 双击必然伴随一次已吞掉的按下；不接住这一就会落到展开/选中上
            return self._pressed.isValid()

        if not self._pressed.isValid():
            return False
        pressed = self._pressed
        self._pressed = QPersistentModelIndex()
        self._repaint(option)
        if pressed == QPersistentModelIndex(index) and btn.contains(pos):
            self.copy_requested.emit(index)
        # 起手在热区内，收尾就不该再让视图拿去改选中
        return True

    @staticmethod
    def _repaint(option) -> None:
        # option.widget 由视图填成 viewport，比从 parent() 强转一层更可靠
        widget = option.widget
        if widget is not None:
            widget.update()


class VarInfoTree(QTreeWidget):
    """属性树：在 :class:`CopyFieldDelegate` 之上补悬停追踪与事件出口。

    悬停必须放在视图层：委托的 ``editorEvent`` 拿不到“鼠标离开”事件，
    而不清悬停的话，鼠标移出树外时最后那一行的图标会永久留着。
    """

    copy_requested = Signal(object)  # QTreeWidgetItem

    def __init__(self, parent=None):
        super().__init__(parent)
        self._delegate = CopyFieldDelegate(self)
        self.setItemDelegate(self._delegate)
        self.setMouseTracking(True)
        self._delegate.copy_requested.connect(self._on_delegate_copy)
        # 滚动后旧悬停索引指向的内容已经换人：不清就会有一枚图标悬停在
        # 鼠标根本没碰到的行上，而且要等下一次鼠标移动才消失
        self.verticalScrollBar().valueChanged.connect(self._reset_hover)

    def _on_delegate_copy(self, index) -> None:
        item = self.itemFromIndex(index)
        if item is None:
            # 按下到投递之间树被重建（异步统计回填就在这个窗口里）：
            # 宁可不复制，也不能把陈旧的键对到新的值上
            return
        self.copy_requested.emit(item)

    def _reset_hover(self, *_args) -> None:
        if self._delegate.clear_hover():
            self.viewport().update()

    def mouseMoveEvent(self, event) -> None:
        if self._delegate.set_hover(self.indexAt(event.position().toPoint())):
            self.viewport().update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self._reset_hover()
        super().leaveEvent(event)


# ---------------------------------------------------------------------------
# 后台统计线程
# ---------------------------------------------------------------------------


class VarInfoWorker(QThread):
    """统计特征的后台计算线程（FIFO 队列 + 可取消）。

    持 loader 用 **weakref**：队列里若残留强引用，reload 后旧 loader
    （776 MB 量级的 ``_signal_cache``）会被拖住无法释放。
    """

    item_ready = Signal(str, object)  # (var_name, VarStats)
    progress = Signal(int, int)  # (本波已完成条数, 本波总数)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cond = threading.Condition()
        # 元素为 (var_name, loader_ref, generation)
        self._jobs: deque = deque()
        self._cancel: set = set()
        self._stop = threading.Event()
        self._current: str | None = None
        # **本波**提交的任务总数：submit 累加、cancel 扣减、队列排空时复位
        # （一波 = 从队列非空到彻底排空）。已完成条数不单独计数，而是由
        # 「本波总数 - 仍在队列或正在运行的条数」反推，见 _progress_snapshot。
        self._pending_total = 0

    # -- 生产者侧（UI 线程） ------------------------------------------------

    def submit(self, jobs: list) -> None:
        """提交任务并唤醒线程；线程未启动时自动启动。

        分母在本波内累加是**正确**的：用户可能在上一批还没跑完时继续追加
        变量，此时"还剩几条"确实等于两批之和。跨波不会漂移，因为
        _advance_progress 与 cancel 都会在队列排空时把分母复位。
        """
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
        """取消单个变量（关闭标签页时调用）。

        分母必须随队列**同步缩减**：被摘除的任务永远不会走到
        ``_advance_progress``，不扣减则进度永远追不平，队列排空后再无
        progress 信号发出，状态栏会永久停在「统计中 N/M…」而不自愈。
        全部清空时当场补发一次信号，让文案立刻恢复。
        """
        with self._cond:
            self._cancel.add(var_name)
            pending_before = self._pending_total
            before = len(self._jobs)
            self._jobs = deque(j for j in self._jobs if j[0] != var_name)
            removed = before - len(self._jobs)
            # 正在运行的那条不在扣减之列：它仍会走到 _finish_current 推进
            # 进度（取消只是在下一个分块边界提前返回），替它扣分母会让
            # 已完成数算多一条
            self._pending_total = max(pending_before - removed, 0)
            idle = not self._jobs and self._current is None
            if idle:
                self._pending_total = 0
            done, total = self._progress_snapshot()
            # 无任务在途时不发信号：cancel 也可能发生在队列本已为空的时刻
            # （如 recompute 的 force 路径），补发 (0, 0) 会把状态栏里
            # _on_stats_ready 刚写入的错误提示清掉
            notify = removed > 0 or (idle and pending_before > 0)
            self._cond.notify_all()
        if notify:
            self.progress.emit(done, total)

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

            # 一条任务的计算与发信号都包在 no_autogc() 窗口内：自动分代收集若
            # 落在本 worker 线程，会把主线程创建的 QObject 包装器拿到这里析构
            # （详见 src/core/gc_guard.py）。窗口按「单条任务」而不是整个 run()
            # 划分——本线程与对话框同寿，长期关着 GC 会让引用环垃圾一直攒着。
            with no_autogc():
                self._process_job(var_name, loader_ref, generation)

        with self._cond:
            self._current = None

    def _process_job(self, var_name: str, loader_ref, generation) -> None:
        """算一条变量的统计并回填。调用方须已置好 ``_current``。"""
        if self._should_cancel(var_name):
            self._cancel.discard(var_name)
            self._finish_current()
            return

        loader = loader_ref()
        if loader is None:
            # loader 已被 GC（reload 后旧数据释放），静默跳过
            self._finish_current()
            return

        try:
            stats = var_info.compute_stats(loader, var_name, self._should_cancel)
        except Exception as e:  # noqa: BLE001 - 子线程须自行兜住所有异常
            logger.debug("统计线程异常 %s", var_name, exc_info=True)
            stats = var_info.VarStats(error=f"{type(e).__name__}: {e}")

        stats.generation = generation
        self._finish_current()
        self.item_ready.emit(var_name, stats)

    def _finish_current(self) -> None:
        """结束当前任务：清空 ``_current`` 后推进本波进度。

        三个出口（正常完成 / 被取消跳过 / loader 已释放跳过）都必须走这里。
        ``_current`` 残留会让 ``cancel_all`` 把已结束的任务重新加入取消集，
        也会让 ``_progress_snapshot`` 把分母多算一条。
        """
        with self._cond:
            self._current = None
        self._advance_progress()

    def _progress_snapshot(self) -> tuple:
        """按「本波」语义算出 (已完成, 总数)。调用方须持有 ``_cond``。

        已完成数由「本波总数 - 仍在队列或正在运行的条数」反推，而不是维护
        一个单调递增的计数器。后者在三类场景下都会与用户眼中的"本批"脱节：
        跨批提交（分母累加成两批之和，实测显示「统计中 4/6…」而用户只提交
        了 3 个）、cancel 摘除任务（已完成数永远追不平）、reload 后旧任务
        迟到完成（旧计数混进新批次）。反推法天然自洽：任何一条任务的进出
        都会同时反映在分子与分母上。
        """
        remaining = len(self._jobs) + (1 if self._current is not None else 0)
        total = self._pending_total
        return max(total - remaining, 0), total

    def _advance_progress(self) -> None:
        with self._cond:
            # 本波排空即复位分母，下一批从 0/N 重新开始。复位后发出的
            # (0, 0) 由 _on_progress 识别为"无在途任务"并清空文案
            if not self._jobs and self._current is None:
                self._pending_total = 0
            done, total = self._progress_snapshot()
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
        # 方案 E 的行为开关（R4）：本会话内是否已有现算结果回填本页。
        # 命中缓存渲染的页面为 False，成为当前可见页时触发一次静默再
        # 验证；现算回填后置 True —— "每 tab 每会话至多一次"由此保证。
        self.validated_this_session = False
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

        self.tree = VarInfoTree()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["属性", "值"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(True)
        # 行高由 CopyFieldDelegate.sizeHint 按 VAR_INFO_ROW_HEIGHT 统一给出
        # （PySide6 6.11 的 QTreeView 没有 setRowHeight）。开着本开关时全表
        # 取最高的一行，所以想要不等高行必须先关掉它
        self.tree.setUniformRowHeights(True)
        # 实测空操作：QTreeView 不支持折行，长文本一律省略号截断。留着只为
        # 表明意图，真正的长值出口是 tooltip 与行尾复制按钮
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
        self.tree.copy_requested.connect(self._on_copy_field)
        # 「文件路径」行的跨平台复制靠右键菜单（见 _on_tree_context_menu）
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_context_menu)
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

        btn_table = QPushButton("添加至变量数值表")
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
        # 每次渲染都重置验证标记：reload 重建（stats=None）后页面回到
        # "待现算"状态，现算回填时重新置位
        self.validated_this_session = False
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
        """回填统计。展示口径未变时跳过重建（R7）。

        方案 E 的再验证回填值与缓存值几乎必然相同（同一 _data_version
        内数据不变），仅诊断字段 from_cache 不同；按对象相等判断会退化为
        每次回填都 takeChildren+重建、肉眼可见地闪动一次。引用本身仍要
        更新（诊断字段以现算结果为准）。
        """
        if (
            self.stats is not None
            and stats is not None
            and var_info.stats_to_rows(self.stats) == var_info.stats_to_rows(stats)
        ):
            self.stats = stats
            return
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

    def _on_copy_field(self, item: QTreeWidgetItem) -> None:
        """行尾悬停按钮：只把「值」列原文送进剪贴板。

        刻意不拼上属性名：用户要的是能直接粘进报告表格单元格/公式里的
        干净数值或路径。也不去取未格式化的原值：树里的文本就是
        ``stats_to_rows`` / ``_fmt`` 产出的那份，与 tooltip 同源。窄窗口下
        长路径会被视图用省略号截断显示，复制到的是全量 —— 与 tooltip
        一致，正是想要的。

        例外是「文件路径」：按 PATH_COPY_STYLE / PATH_COPY_QUOTE 转写法与
        包引号，因为这一行的用途就是在另一台机器 / 命令行里打开该文件。

        状态栏只报键名、不回显数值：一条完整文件路径 120+ 字符，回显
        出来既读不出新信息（刚点的就是它），又会把底部标签顶长。
        """
        key = item.text(0)
        value = item.text(1)
        if key == var_info.ROW_KEY_FILE_PATH:
            value = self._format_path(value, None)
        QApplication.clipboard().setText(value)
        self._dialog._notify(f"已复制「{key}」")

    # -- 文件路径的跨平台复制 ------------------------------------------------

    def _format_path(self, raw: str, style: str | None) -> str:
        """按风格与引号策略格式化路径；``style=None`` 用配置里的默认风格。

        值为占位符 ``-``（``var_info._fmt`` 在路径缺失时给的）时必须原样返回：
        ``display_path`` 只认“非空串即路径”，喂它会得到 ``<工作目录>/-`` ——
        一条看着能用、实际打不开的假路径。守卫放在这个唯一出口，行尾按钮 /
        Ctrl+C / 右键菜单三条路径一并覆盖，不必各自在调用点重复一份。
        """
        if not raw or raw == _DASH_PLACEHOLDER:
            return raw
        return format_for_copy(
            raw, style or PATH_COPY_STYLE, PATH_COPY_QUOTE
        )

    def _copy_path_variant(self, raw: str, style: str | None, label: str) -> None:
        QApplication.clipboard().setText(self._format_path(raw, style))
        self._dialog._notify(f"已复制文件路径（{label}）")

    def _on_tree_context_menu(self, pos) -> None:
        r"""「文件路径」行的右键菜单：挑目标平台的写法复制。

        入口已把路径统一成当前平台的绝对写法，但跨平台协作时还需要另一种
        分隔符：实测 Windows 拖拽产出的 ``//host/share/x.csv`` 粘回 Windows
        会被 Shell 当 URL 交给浏览器（跳 Edge），只有 ``\\host\share`` 能跳转
        网盘；macOS 上复制给 Windows 侧同理。其他行不提供菜单：它们的
        复制语义就是“原样取走”，多一层转换反而不可预期。
        """
        item = self.tree.itemAt(pos)
        if item is None or item.text(0) != var_info.ROW_KEY_FILE_PATH:
            return
        raw = item.text(1)
        if not raw or raw == "-":
            # 空路径没有内容可复制，不必弹菜单（_format_path 另有同一守卫）
            return
        menu = QMenu(self.tree)
        # 闭包必须用默认参数绑定循环变量（增量删行后陈旧值误删的老坑同源）
        for label, style in (
            ("当前平台", None),
            (r"Windows \host\share", STYLE_WINDOWS),
            ("POSIX //host/share", STYLE_POSIX),
        ):
            action = menu.addAction(f"复制为 {label}")
            action.triggered.connect(
                lambda _checked=False, s=style, lab=label: self._copy_path_variant(raw, s, lab)
            )
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _on_copy_selection(self) -> None:
        """Ctrl+C 复制树中选中的行（tab 分隔，可直接粘贴进表格）。

        「文件路径」的值列同样走跨平台格式化，与行尾复制按钮口径一致。
        """
        items = self.tree.selectedItems()
        if not items:
            return
        lines = []
        for it in items:
            cells = []
            for c in range(self.tree.columnCount()):
                text = it.text(c)
                if c == 1 and it.text(0) == var_info.ROW_KEY_FILE_PATH:
                    text = self._format_path(text, None)
                cells.append(text)
            lines.append("\t".join(cells))
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
            # 排队中的再验证任务被静默丢弃（不 emit），标记一并整批释放（R1）
            dlg._revalidating.clear()
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
        # 方案 E 的在途再验证去重集合：同一变量的再验证未完成前不得重复
        # 提交（快速连切标签页时尤其重要）。泄漏会让该变量永久无法再验证
        # 且无任何报错（R1），因此在每条完成/丢弃路径与整批清理点都要释放。
        self._revalidating: set[str] = set()
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
        # 宽度策略取 Ignored：QLabel 的 sizeHint 会经布局抬高整个窗口的
        # minimumWidth，长提示一显示窗口就被顶宽且再也拖不回去（实测
        # 复制一条 60+ 字符的文件路径：700 → 769，minimumWidth 同步变
        # 769）。Ignored 意为不参与宽度计算，窗口宽度只由用户决定，
        # 提示放不下就裁尾（高度仍按内容走）
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
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
        # 仅对不可见窗口生效：restoreGeometry 对已显示窗口非幂等（会把
        # frame margin 反复计入导致逐次漂移），而窗口开着时当前几何就是
        # 最新状态。popup 每次都对复用实例 load_geom，若不设守卫，用户
        # 调好的位置大小会在下一次右键"变量信息"时被旧快照重置。
        # 与 DataTableDialog.load_geom 的守卫同口径。
        if self.isVisible():
            return
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
                # 命中缓存：复制副本后标记来源。不得原地改写缓存对象（R4）：
                # 缓存母本必须恒为 from_cache=False，否则回写与诊断语义漂移
                stats = replace(stats, from_cache=True)
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
        # 方案 E 触发点（共三个，此处兜底第二个）：setCurrentIndex 在索引
        # 未变（右键已存在变量）或标签栏隐藏（单标签）时不发 currentChanged，
        # 必须对当前页显式补一次再验证，否则最典型的"关窗后再开同一变量"
        # 场景会静默漏掉
        self._revalidate_current_page()

    def _tab_title(self, name: str, snapshot) -> str:
        """标签标题带上单位，便于同名不同单位的变量区分。"""
        if snapshot is None:
            return f"{name} (失效)"
        unit = snapshot.unit
        return f"{name} [{unit}]" if unit and unit != "-" else name

    def _safe_snapshot(self, loader, name, generation):
        """快照构建失败（变量不存在 / 数据异常）时返回 None 而不抛。

        构建过程零磁盘 I/O 且 O(1)（MDF 路径约 99 μs，表格路径只读 pandas
        元数据），因此在 UI 线程同步调用是安全的；即使一次提交
        ``VAR_INFO_MAX_TABS`` 个变量，累计耗时也在毫秒量级。
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
        # 排队未启动的再验证任务被 cancel 后不会 emit（worker 只 emit
        # 运行中的那条），去重标记必须在此释放（R1），否则重开该变量后
        # 永远无法再次触发再验证
        self._revalidating.discard(name)
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
            # 方案 E 触发点一：切到的页若显示的是缓存值，静默提交一次现算
            self._revalidate_page_if_needed(page)

    # -- 可见即再验证（方案 E）---------------------------------------------

    def _revalidate_current_page(self) -> None:
        """对当前可见页触发再验证（add_variables 末尾的显式触发点）。"""
        self._revalidate_page_if_needed(self.tabs.currentWidget())

    def _revalidate_page_if_needed(self, page) -> None:
        """页面成为当前可见页时：若显示的是缓存值，静默提交一次现算覆盖。

        触发条件（全部满足才提交）：
        - 未处于在途再验证（``_revalidating`` 去重，快速连切不重复入队）
        - 页面未在本会话内现算过（``validated_this_session`` 为 False；
          该标记为 False 且 stats.computed 为 True 蕴含"值来自缓存"）
        - 已有算完的统计（首批计算在途 / 错误终态无从覆盖，等回填）

        静默 = 不清空页面、不显示"计算中…"，回填直接覆盖；值相同时
        ``set_stats`` 跳过重绘（R7），无闪动。成本从方案 B 的 ×标签数
        降到 ×1，且与用户注意力对齐 —— 不看的标签页不读盘。
        """
        if not isinstance(page, VarInfoPage):
            return
        name = page.var_name
        if name in self._revalidating:
            return
        if page.validated_this_session or page.is_stale:
            return
        stats = page.stats
        if stats is None or not stats.computed:
            return
        loader = self._resolve_loader()
        if loader is None:
            return
        self._revalidating.add(name)
        self.worker.submit([(name, _make_ref(loader), self._generation())])

    def _close_all_tabs(self) -> None:
        self.worker.cancel_all()
        # cancel_all 对排队未启动的任务静默丢弃（不 emit），整批去重标记
        # 必须就地释放（R1）：本方法绕过 _on_tab_close 自行循环摘页，
        # 残留的名字会让"关闭全部后再打开"永远不再触发再验证
        self._revalidating.clear()
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
        if getattr(stats, "cancelled", False):
            # 纵深防御（缺-5）：cancelled 条目不是有效结果，拒绝采用并
            # 顺手剔除。正常路径下 _store_cache 已拒绝写入，此处兜住
            # 历史版本残留与异常路径
            cache.pop(name, None)
            return None
        if getattr(stats, "generation", -1) != generation:
            cache.pop(name, None)
            return None
        if not getattr(stats, "computed", False) and not getattr(stats, "error", ""):
            return None
        return stats

    def _store_cache(self, name: str, stats) -> None:
        if getattr(stats, "cancelled", False):
            # 取消不是有效统计结果（缺-5）：写入会让下一次命中直接展示
            # 「已取消」，而 add_variables 只对 stats is None 的页面提交
            # 重算，用户将永远卡在错误终态
            return
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
            # reload 期间完成的陈旧结果：既不写缓存也不回填，直接丢弃。
            # 去重标记必须一并释放（R1）—— 此分支直接 return，不清理的
            # 话该变量在本会话内将永远无法再次触发再验证
            self._revalidating.discard(name)
            return

        self._store_cache(name, stats)
        # 尽早释放去重标记：无论页面是否还在、结果是否被守卫拦截，
        # 本次验证机会都已用掉
        self._revalidating.discard(name)

        page = self._pages.get(name)
        if page is None:
            return  # 标签页已被用户关闭

        # R2 覆盖守卫：页面已有算完的正确值（缓存命中 + 可见即再验证的
        # 典型场景）时，迟到的失败结果（取消 / loader 关闭 / 读盘异常）
        # 只在状态栏提示，不得把数字覆盖成"已取消"
        current = page.stats
        if current is not None and current.computed and not stats.computed:
            if stats.error:
                self._progress_text = f"{name}: {stats.error}"
                self._refresh_status()
            page.validated_this_session = True
            return

        page.set_stats(stats)
        # 现算结果已回填：本会话内不再对该页触发再验证。「刷新统计」的
        # force 路径同样经此回填，因此手动刷新过的页也不会被 E 重复触发
        page.validated_this_session = True
        if not stats.computed and stats.error:
            self._progress_text = f"{name}: {stats.error}"
            self._refresh_status()

    def _on_progress(self, done: int, total: int) -> None:
        """渲染后台统计进度。

        ``(0, 0)`` 是 worker 的**终态信号**（本波队列彻底排空），此时清空
        文案。分母取的是「本波总数」而非进程内累加值，因此第二批提交会从
        「0/3」重新开始，不会出现「统计中 4/6…」这类既非本批、又永不
        自愈的错误分母。
        """
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
        # 与再验证共用在途去重：recompute 计算期间用户切换标签页不得
        # 对同一变量重复提交（结果到达时统一在 _on_stats_ready 释放）
        self._revalidating.add(name)
        self._progress_text = "统计中 0/1…"
        self._refresh_status()

    # -- reload 后重建 ------------------------------------------------------

    def _rebuild_all_pages(self, loader) -> None:
        self.worker.cancel_all()
        # 整批释放去重标记（R1）：reload 后所有页面重新走"渲染 → 现算
        # → 回填置位"流程，旧标记全部作废
        self._revalidating.clear()
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
