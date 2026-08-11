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
from PySide6.QtGui import QFontMetrics, QPen, QColor
from PySide6.QtWidgets import (
    QLabel,
    QSizePolicy,
    QGraphicsProxyWidget,
    QGraphicsLinearLayout,
    QGraphicsWidget,
    QRubberBand,
    QApplication,
)
import pyqtgraph as pg

from src.core.config import (
    DEFAULT_SHOW_X_AXIS_LABEL,
    UI_DEBOUNCE_DELAY_MS,
    XRANGE_THRESHOLD_FOR_SYMBOLS,
)
from src.core.logger import get_logger

logger = get_logger("widget.plot_ui")
from src.core.scheduler import UnifiedUpdateScheduler
from src.ui.widgets.base_manager import BasePlotManager


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

        left_margin = QGraphicsWidget()
        layout.addItem(left_margin)
        layout.setItemSpacing(0, base_spacing * 0)

        pw.legend_label = QLabel("channel name")
        pw.legend_label.setStyleSheet("""
            color: #000;
            font-weight: bold;
            background-color: transparent;
        """)
        pw.legend_label.setSizePolicy(
            QSizePolicy.Policy.Minimum,
            QSizePolicy.Policy.Preferred,
        )
        pw.legend_label.setTextFormat(Qt.TextFormat.RichText)
        pw.legend_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction,
        )
        pw.legend_label.setContextMenuPolicy(
            Qt.ContextMenuPolicy.NoContextMenu,
        )
        pw.legend_label.mousePressEvent = pw._on_legend_clicked

        proxy_left = QGraphicsProxyWidget()
        proxy_left.setWidget(pw.legend_label)
        layout.addItem(proxy_left)
        layout.setStretchFactor(proxy_left, 1)
        layout.setAlignment(
            proxy_left,
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
        )

        header.setLayout(layout)
        pw.addItem(header, row=0, col=0, colspan=2)

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

        pw._is_interacting = False
        pw._interaction_timer = QTimer()
        pw._interaction_timer.setSingleShot(True)
        pw._interaction_timer.timeout.connect(pw._end_interaction)
        pw._is_syncing_range = False
        pw._interaction_x_only = False  # 标记当前交互是否为 X-only（用于 _end_interaction 决策）

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
