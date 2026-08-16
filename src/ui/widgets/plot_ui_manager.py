"""
PlotUIManager - UI初始化与刷新协调管理器

负责 DraggableGraphicsLayoutWidget 的所有 UI 初始化工作：
- 整体 setup_ui 编排
- 顶部 header 区域
- 主绘图区域和 ViewBox
- 坐标轴样式配置
- 交互元素（光标线、RubberBand）
- UI 更新调度协调（防抖 + 批量更新）

此模块从 csv_plot_pyqt6.py 迁移而来。
"""

from __future__ import annotations
from typing import Any

from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import QFontMetrics, QPen, QColor, QDrag
from PySide6.QtWidgets import (
    QSizePolicy,
    QGraphicsProxyWidget,
    QGraphicsLinearLayout,
    QGraphicsWidget,
    QRubberBand,
    QApplication,
    QTextBrowser,
    QFrame,
)
import pyqtgraph as pg

from src.core.config import (
    DEFAULT_SHOW_X_AXIS_LABEL,
    UI_DEBOUNCE_DELAY_MS,
    XRANGE_THRESHOLD_FOR_SYMBOLS,
)
from src.core.logger import get_logger
from src.ui.drag_drop import (
    build_legend_var_mimedata,
    clear_active_legend_drag,
    create_drag_pixmap,
    parse_anchor_var_name,
    set_active_legend_drag,
)

logger = get_logger("widget.plot_ui")
from src.core.scheduler import UnifiedUpdateScheduler
from src.ui.widgets.base_manager import BasePlotManager

LEGEND_MAX_LINES = 3  # legend 最大显示行数，超出部分通过滚动条查看
LEGEND_BOTTOM_PAD = 0  # legend 底部留白，保证末行文字不与 plot 上边框接触

# legend 滚动条样式：细条圆角灰色、无箭头按钮
LEGEND_SCROLLBAR_QSS = """
    QScrollBar:vertical {
        background: #f2f2f2;
        width: 8px;
        margin: 1px;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical {
        background: #b0b0b0;
        min-height: 16px;
        border-radius: 3px;
    }
    QScrollBar::handle:vertical:hover {
        background: #8f8f8f;
    }
    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {
        height: 0px;
    }
    QScrollBar::add-page:vertical,
    QScrollBar::sub-page:vertical {
        background: transparent;
    }
"""


class LegendTextBrowser(QTextBrowser):
    """支持将变量名拖出到其他 plot 的 legend 容器。

    点击（位移 < startDragDistance）保持原有 anchorClicked 切换显隐行为；
    超过阈值才发起 QDrag，两种手势物理互斥（见设计文档 §3.4）。
    """

    def __init__(self, plot_widget):
        super().__init__()
        self._pw = plot_widget  # 弱语义持有（同生命周期，无需 weakref）
        self._drag_press_pos: QPoint | None = None
        self._drag_var_name: str | None = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            href = self.anchorAt(event.position().toPoint())
            # 按压瞬间解析变量名（拖拽期间 setHtml 重建文档也不影响）
            self._drag_var_name = parse_anchor_var_name(href)
            self._drag_press_pos = event.position().toPoint()
        super().mousePressEvent(event)  # 保留链接高亮/anchor 内部状态

    def mouseMoveEvent(self, event):
        if (
            self._drag_var_name
            and self._drag_press_pos is not None
            and event.buttons() & Qt.MouseButton.LeftButton
            and (event.position().toPoint() - self._drag_press_pos).manhattanLength()
            >= QApplication.startDragDistance()
        ):
            self._start_var_drag()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # 无条件清空 press 状态，防止悬空状态残留（拖拽取消兜底）
        self._drag_press_pos = None
        self._drag_var_name = None
        super().mouseReleaseEvent(event)  # 未拖拽时正常触发 anchorClicked

    def _start_var_drag(self):
        var_name = self._drag_var_name
        self._drag_var_name = None
        self._drag_press_pos = None
        drag = QDrag(self)
        drag.setMimeData(build_legend_var_mimedata([var_name], self._pw))
        pixmap = create_drag_pixmap([var_name], self.font())
        if pixmap:
            drag.setPixmap(pixmap)
            drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))
        set_active_legend_drag(self._pw, [var_name])
        try:
            # CopyAction|MoveAction 并存，drop 端按 Ctrl 决定语义（默认移动、Ctrl=复制）。
            # exec 进入嵌套事件循环，release 被拖拽会话消费，
            # anchorClicked 不会误发 —— 点击/拖拽互斥的核心保证。
            drag.exec(Qt.DropAction.CopyAction | Qt.DropAction.MoveAction)
        finally:
            clear_active_legend_drag()


class PlotUIManager(BasePlotManager):
    """负责 UI 初始化和刷新协调的管理器"""

    def __init__(self, plot_widget: Any):
        super().__init__(plot_widget)
        # v5.3: cursor 场景修改护栏标志，阻止 paint event 在 BSP 中间态期间访问场景
        self.pw._is_cursor_modifying_scene = False

    # ========================================================================
    # 公开初始化入口
    # ========================================================================

    def setup_ui(
        self,
        units_dict: dict,
        dataframe: Any,
        time_channels_info: dict | None = None,
        synchronizer: Any = None,
    ) -> None:
        """初始化UI组件和布局（编排方法）

        设置图形布局控件的基本配置和数据结构，
        初始化绘图相关的属性和同步器。

        Args:
            units_dict: 单位字典
            dataframe: 数据框
            time_channels_info: 时间通道信息
            synchronizer: 同步器实例
        """
        pw = self.pw
        pw.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        pw.setAcceptDrops(True)
        pw.units = units_dict
        pw.data = dataframe
        pw.time_channels_info = time_channels_info or {}
        pw.synchronizer = synchronizer

        pw.time_column_name = None
        pw.time_axis_label = "Index"

        pw.x_name = ""
        pw.x_format = ""

        pw.xMin: int = 0
        pw.xMax: int = 1

        # 曲线统一管理
        pw.curves = {}
        pw._batch_adding = False
        pw.curve_colors = [
            "blue",
            "red",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
        pw.current_color_index = 0
        pw._max_point_density: float = 0.0

        self._setup_header(pw)
        self._setup_plot_area(pw)
        self._setup_axes(pw)
        self._setup_interaction(pw)
        self._init_ui_refresh_coordinator(pw)

        pw.ci.layout.setContentsMargins(0, 0, 10, 5)
        pw.ci.layout.setSpacing(0)
        pw.ci.layout.setRowStretchFactor(1, 1)

        pw.rubberBand = QRubberBand(QRubberBand.Shape.Rectangle, pw)
        pw.origin = QPoint()

    # ========================================================================
    # Header
    # ========================================================================

    def _setup_header(self, pw: Any) -> None:
        """完全修正的顶部文本区域设置方法"""
        header = pg.GraphicsWidget()
        layout = QGraphicsLinearLayout(Qt.Orientation.Horizontal)

        font = QApplication.font()
        fm = QFontMetrics(font)
        base_spacing = fm.horizontalAdvance("-10000.01")
        header.setFixedHeight(fm.height() * 2)
        # 清零默认 margin/spacing（样式默认各 ~12/10px），
        # 保证 legend 右缘可精确对齐 plot 右边界黑线
        layout.setContentsMargins(0, 0, 0, 0)

        # 左侧间距占位：宽度动态对齐 plot 左边界黑线（update_legend_height 维护，
        # 补偿左轴刻度/轴标题占位，随 y 轴宽度变化）
        pw._legend_left_spacer = QGraphicsWidget()
        pw._legend_left_spacer.setMinimumWidth(0)
        pw._legend_left_spacer.setMaximumWidth(0)
        layout.addItem(pw._legend_left_spacer)
        layout.setItemSpacing(0, base_spacing * 0)

        # 右侧间距占位：宽度动态对齐 plot 右边界黑线（update_legend_height 维护）
        pw._legend_right_spacer = QGraphicsWidget()
        pw._legend_right_spacer.setMinimumWidth(0)
        pw._legend_right_spacer.setMaximumWidth(0)

        pw.legend_label = LegendTextBrowser(pw)
        pw.legend_label.setOpenLinks(False)           # 禁止真实导航，只发 anchorClicked
        pw.legend_label.setFrameStyle(QFrame.Shape.NoFrame)
        pw.legend_label.setStyleSheet("""
            QTextBrowser {
                background-color: transparent;
                font-weight: bold;
                border: none;
            }
        """)
        pw.legend_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        pw.legend_label.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        # 只允许鼠标点击链接(锚点)，禁止选中普通文本
        pw.legend_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        pw.legend_label.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        pw.legend_label.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        pw.legend_label.document().setDocumentMargin(1)
        pw.legend_label.setPlaceholderText("channel name")
        pw.legend_label.anchorClicked.connect(pw._on_legend_anchor_clicked)
        pw.legend_label.verticalScrollBar().setStyleSheet(LEGEND_SCROLLBAR_QSS)
        # 文档内容变化也触发高度重算（补齐宽度协商之外的触发源）
        pw.legend_label.document().contentsChanged.connect(self.update_legend_height)
        # label 自身宽度变化（任意布局通道引起）也触发高度重算，
        # 补齐 pw.resizeEvent 与宽度协商之间的时序缺口
        pw._legend_last_w = None
        pw.legend_label.installEventFilter(pw)

        proxy_left = QGraphicsProxyWidget()
        proxy_left.setWidget(pw.legend_label)
        layout.addItem(proxy_left)
        layout.setStretchFactor(proxy_left, 1)
        layout.setAlignment(
            proxy_left,
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
        )
        layout.addItem(pw._legend_right_spacer)
        layout.setItemSpacing(1, 0)
        layout.setItemSpacing(2, 0)
        # 供 wheelEvent 判断鼠标是否悬停在 legend 上，
        # 以及 update_legend_height 同步固定高度约束
        pw._legend_proxy = proxy_left

        header.setLayout(layout)
        pw._header_widget = header  # 供 update_legend_height 同步调整 header 高度
        pw.addItem(header, row=0, col=0, colspan=2)

    def update_legend_height(self) -> None:
        """根据内容与当前宽度重算 legend 高度（1 行 ~ LEGEND_MAX_LINES 行）"""
        pw = self.pw
        label = pw.legend_label

        def _recalc():
            # 让文档按当前 viewport 宽度排版后再测量
            doc = label.document()
            doc.setTextWidth(label.viewport().width())
            content_h = doc.size().height()
            # 取富文本排版的真实行高（fontMetrics 兜底），避免行高低估
            line_h = 0.0
            block = doc.firstBlock()
            if block.isValid() and block.layout() and block.layout().lineCount():
                line_h = block.layout().lineAt(0).height()
            if line_h <= 0:
                line_h = float(label.fontMetrics().height())
            min_h = line_h + 4
            max_h = line_h * LEGEND_MAX_LINES + 4
            # 底部留白保证末行文字不与 plot 上边框接触（label 底对齐 header 底）
            legend_h = int(max(min_h, min(content_h + 4, max_h))) + LEGEND_BOTTOM_PAD
            label.setFixedHeight(legend_h)
            # 同步固定 proxy 高度约束（多层防御：内嵌 widget 约束
            # 在图形布局协商中存在失效场景）
            proxy = getattr(pw, "_legend_proxy", None)
            if proxy is not None:
                proxy.setMinimumHeight(legend_h)
                proxy.setMaximumHeight(legend_h)
            # 同步增高 header 容器，避免 legend 溢出入侵绘图区；
            # 下限保持 2 行高（维持单行/空状态的旧版外观）
            header = getattr(pw, "_header_widget", None)
            if header is not None:
                header.setFixedHeight(max(line_h * 2, legend_h))
            label.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
                if content_h > max_h
                else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            # 左侧 spacer 对齐：legend 左缘对齐 plot 左边界黑线
            # （补偿左轴刻度/轴标题占位；宽度随 y 轴刻度位数变化，
            # 由左轴 Resize 的 eventFilter 触发重算）
            left_spacer = getattr(pw, "_legend_left_spacer", None)
            if left_spacer is not None and header is not None and hasattr(pw, "view_box"):
                try:
                    left_inset = (
                        pw.view_box.sceneBoundingRect().left()
                        - header.sceneBoundingRect().left()
                    )
                    if left_inset >= 0 and abs(left_spacer.rect().width() - left_inset) > 0.5:
                        left_spacer.setMinimumWidth(int(left_inset))
                        left_spacer.setMaximumWidth(int(left_inset))
                except Exception:
                    logger.debug("legend 左对齐计算失败", exc_info=True)
            # 右侧 spacer 对齐：滚动条右缘对齐 plot 右边界黑线
            # （补偿 ci 右边距 + 右侧轴占位）
            spacer = getattr(pw, "_legend_right_spacer", None)
            if spacer is not None and header is not None and hasattr(pw, "view_box"):
                try:
                    inset = (
                        header.sceneBoundingRect().right()
                        - pw.view_box.sceneBoundingRect().right()
                    )
                    if inset >= 0 and abs(spacer.rect().width() - inset) > 0.5:
                        spacer.setMinimumWidth(int(inset))
                        spacer.setMaximumWidth(int(inset))
                except Exception:
                    logger.debug("legend 右对齐计算失败", exc_info=True)
            # 强制外层布局立即落定，消除 header 增高未生效的中间态
            try:
                pw.ci.layout.activate()
            except Exception:
                logger.debug("强制布局落定失败", exc_info=True)

        QTimer.singleShot(0, _recalc)   # 等 graphics layout 分配完宽度

    # ========================================================================
    # Plot Area
    # ========================================================================

    def _setup_plot_area(self, pw: Any) -> None:
        """配置绘图区域基本属性（包含性能优化配置）"""
        from src.ui.widgets.custom_viewbox import CustomViewBox

        pw.plot_item = pw.addPlot(row=1, col=0, colspan=2, viewBox=CustomViewBox())
        pw.view_box = pw.plot_item.vb
        pw.view_box.plot_widget = pw
        pw._connect_viewbox_signals()
        # 左轴宽度变化（y 轴刻度位数改变）→ 重算 legend 左右对齐
        pw._legend_left_axis = pw.plot_item.getAxis("left")
        pw._legend_left_axis.installEventFilter(pw)
        pw._legend_axis_last_w = None

        pw._is_interacting = False
        pw._interaction_timer = QTimer()
        pw._interaction_timer.setSingleShot(True)
        pw._interaction_timer.timeout.connect(pw._end_interaction)
        pw._is_syncing_range = False

        pw.view_box.setAutoVisible(x=False, y=True)
        pw.plot_item.setTitle(None)
        pw.plot_item.hideButtons()
        pw.plot_item.setClipToView(True)
        pw.plot_item.setDownsampling(mode="peak", auto=True)

        pw.setBackground("w")

        pw.plot_item.getAxis("left").setGrid(255)
        pw.plot_item.getAxis("bottom").setGrid(255)
        pw.plot_item.showGrid(x=True, y=True, alpha=0.1)

        pw.view_box.sigRangeChanged.connect(pw._on_range_changed)
        self.update_x_axis_label(pw)

    def update_x_axis_label(self, pw: Any) -> None:
        """更新 X 轴标签文本"""
        axis = pw.plot_item.getAxis("bottom")
        if DEFAULT_SHOW_X_AXIS_LABEL:
            label = pw.time_axis_label if pw.time_axis_label else "Index"
            axis.setLabel(label)
            axis.showLabel(True)
        else:
            axis.showLabel(False)

    # ========================================================================
    # Axes
    # ========================================================================

    def _setup_axes(self, pw: Any) -> None:
        """配置坐标轴样式和范围"""

        pw.axis_x = pw.plot_item.getAxis("bottom")
        pw.axis_x.setTextPen("black")
        pw.axis_x.setPen(QPen(QColor("black"), 1))
        pw.axis_x.setRange(0, 10)

        pw.axis_y = pw.plot_item.getAxis("left")
        pw.axis_y.enableAutoSIPrefix(False)
        pw.axis_y.setTextPen("black")
        pw.axis_y.setPen(QPen(QColor("black"), 1))

        for pos in ("top", "right"):
            ax = pw.plot_item.getAxis(pos)
            ax.setVisible(True)
            ax.setTicks([])
            ax.setStyle(showValues=False, tickLength=0)
            ax.setPen(QPen(QColor("black"), 1))

        font = QApplication.font()
        fm = QFontMetrics(font)
        pw.axis_y.setWidth(fm.horizontalAdvance("-10000.01"))

        font_family = font.family()
        pixel_size = font.pixelSize() + 2
        pw.axis_y.setLabel(
            color="black",
            angle=-90,
            **{
                "font-family": font_family,
                "font-size": f"{pixel_size}px",
                "font-weight": "bold",
            },
        )

    # ========================================================================
    # Interaction
    # ========================================================================

    def _setup_interaction(self, pw: Any) -> None:
        """配置交互元素（光标线、RubberBand、鼠标信号代理）"""
        from PySide6.QtCore import QTimer
        import pyqtgraph as _pg

        pw.vline = _pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=_pg.mkPen((255, 0, 0, 100), width=4),
        )
        pw.vline2 = _pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=_pg.mkPen((255, 0, 0, 100), width=4),
        )
        pw.vline.cursor_index = 0
        pw.vline2.cursor_index = 1
        pw.vline2.setZValue(100)
        pw.vline.setZValue(100)
        pw.cursor_label = _pg.TextItem("", anchor=(1, 1), color="red")
        pw.plot_item.addItem(pw.vline, ignoreBounds=True)
        pw.plot_item.addItem(pw.vline2, ignoreBounds=True)
        pw.plot_item.addItem(pw.cursor_label, ignoreBounds=True)
        pw.vline.setVisible(False)
        pw.vline2.setVisible(False)
        pw.cursor_label.setVisible(False)

        pw.multi_cursor_items = []
        pw.show_values_only = True

        pw._cursor_item_pool = {
            "circles": [],
            "labels": [],
            "x_labels": [],
        }

        pw.proxy = _pg.SignalProxy(
            pw.scene().sigMouseMoved,
            rateLimit=20,
            slot=pw.mouse_moved,
        )
        pw.vline.sigPositionChanged.connect(pw.on_vline_position_changed)
        pw.vline2.sigPositionChanged.connect(pw.on_vline_position_changed)
        pw.setAntialiasing(False)

        pw._last_cursor_update_time = 0
        pw._cursor_update_throttle = 0.016
        pw._adaptive_throttle_enabled = True
        pw._cursor_refresh_timer = QTimer(pw)
        pw._cursor_refresh_timer.setSingleShot(True)
        pw._cursor_refresh_timer.timeout.connect(pw._refresh_cursor_geometry)
        pw._pending_cursor_geometry_update = False

    # ========================================================================
    # UI Refresh Coordinator
    # ========================================================================

    def _init_ui_refresh_coordinator(self, pw: Any) -> None:
        """初始化统一 UI 更新调度器"""
        pw._ui_refresh = UnifiedUpdateScheduler(
            delay_ms=UI_DEBOUNCE_DELAY_MS,
            order=("style", "cursor", "stats"),
            parent=pw,
        )
        pw._ui_refresh.register("style", pw._run_style_refresh)
        pw._ui_refresh.register("cursor", pw._run_cursor_refresh)
        pw._ui_refresh.register("stats", pw._run_stats_refresh)

    def _queue_ui_refresh(
        self,
        pw: Any,
        *,
        style: bool = True,
        cursor: bool = True,
        stats: bool = True,
        immediate: bool = False,
    ) -> None:
        """调度 UI 更新任务（防抖 + 批量更新）"""
        if not hasattr(pw, "_ui_refresh"):
            return
        tasks: list[str] = []
        if style:
            tasks.append("style")
        if cursor:
            tasks.append("cursor")
        if stats:
            tasks.append("stats")
        if not tasks:
            return
        if immediate:
            pw._ui_refresh.run_immediately(*tasks)
        else:
            pw._ui_refresh.schedule(*tasks)

    def _cancel_ui_refresh(self, pw: Any, *tasks: str) -> None:
        """取消已调度的 UI 更新任务"""
        if hasattr(pw, "_ui_refresh"):
            if tasks:
                pw._ui_refresh.cancel(*tasks)
            else:
                pw._ui_refresh.cancel()

    def _run_style_refresh(self, pw: Any) -> None:
        """执行样式刷新"""
        if (
            getattr(pw, "_is_updating_data", False)
            or getattr(pw, "_is_being_destroyed", False)
        ):
            return
        if hasattr(pw, "view_box") and hasattr(pw, "plot_item"):
            self.update_plot_style(pw, pw.view_box, pw.view_box.viewRange(), None)

    def _calculate_visible_points(self, pw: Any, range) -> tuple:
        """计算当前可见范围的点数估算"""
        x_min, x_max = range[0]
        x_range_width = x_max - x_min

        if hasattr(pw, 'factor') and pw.factor != 0:
            index_range_width = x_range_width / abs(pw.factor)
        else:
            index_range_width = x_range_width

        curve_count = len(pw.curves) if hasattr(pw, 'curves') and pw.curves else 0

        curve_count = max(curve_count, 1)
        visible_points = index_range_width * curve_count

        return index_range_width, visible_points

    def update_plot_style(self, pw: Any, view_box, range, rect=None):
        """更新绘图样式 - 基于xRange宽度判断细线+symbol或粗线无symbol"""
        try:
            _mw = pw.window()

            if getattr(pw, '_is_updating_data', False) or getattr(pw, '_is_being_destroyed', False):
                return

            # 重载期间禁止样式刷新，避免中间状态触发错误的 symbol 切换
            if _mw and getattr(_mw, '_is_loading_new_data', False):
                return

            if not hasattr(pw, 'factor') or not hasattr(pw, 'plot_item'):
                return

            is_interacting = getattr(pw, '_is_interacting', False)
            if is_interacting:
                return

            index_range_width, visible_points = self._calculate_visible_points(pw, range)

            density = getattr(_mw, '_global_max_density', 0.0) if _mw else 0.0
            if density > 0:
                raw_threshold = XRANGE_THRESHOLD_FOR_SYMBOLS / density
                # clamp 到 [100, 200]，防止 density 变化导致阈值异常（如 factor=10 时 threshold≈999）
                effective_threshold = max(XRANGE_THRESHOLD_FOR_SYMBOLS,
                                          min(raw_threshold, XRANGE_THRESHOLD_FOR_SYMBOLS * 2))
            else:
                effective_threshold = XRANGE_THRESHOLD_FOR_SYMBOLS
            show_symbols = index_range_width < effective_threshold

            pw._apply_plot_style(show_symbols)

        except Exception as e:
            logger.error("更新绘图样式时出错: %s", e)

    def _run_cursor_refresh(self, pw: Any) -> None:
        """执行光标刷新"""
        if getattr(pw, "_is_interacting", False):
            return
        if hasattr(pw, "vline") and pw.vline.isVisible():
            try:
                pw.update_cursor_label()
            except Exception:
                logger.debug("更新光标标签失败", exc_info=True)

    def _run_stats_refresh(self, pw: Any) -> None:
        """执行统计刷新"""
        main_window = pw.window()
        if main_window is not None:
            main_window.layout_manager.request_mark_stats_refresh(immediate=True)
