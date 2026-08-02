"""
ui.widgets —— 绘图相关的组件模块

包含 DraggableGraphicsLayoutWidget 的所有组合模式管理器:
- BasePlotManager: 管理器基类 (weakref + 生命周期钩子)
- PlotUIManager: UI 初始化 + 防抖刷新协调
- AxisManager: X/Y 轴范围、标签、限制管理
- PlotDataManager: 单曲线绘图 + 时间修正 + 数据验证
- MultiCurveManager: 多曲线 + 图例 + 样式切换
- CursorManager: 光标模式/标签/对象池/off 模式
- MarkRegionManager: 区域选择 + NumPy 统计计算
- EventHandler: ViewBox 信号路由 + 交互事件
"""

from src.ui.widgets.custom_viewbox import CustomViewBox
from src.ui.widgets.plot_container import PlotContainerWidget
from src.ui.widgets.base_manager import BasePlotManager
from src.ui.widgets.plot_ui_manager import PlotUIManager
from src.ui.widgets.axis_manager import AxisManager
from src.ui.widgets.plot_data_manager import PlotDataManager
from src.ui.widgets.multi_curve_manager import MultiCurveManager
from src.ui.widgets.cursor_manager import CursorManager
from src.ui.widgets.mark_region_manager import MarkRegionManager
from src.ui.widgets.event_handler import EventHandler
from src.ui.widgets.log_viewer import LogViewer
from src.ui.widgets.variable_search_bar import VariableSearchBar

__all__ = [
    "BasePlotManager",
    "PlotUIManager",
    "AxisManager",
    "PlotDataManager",
    "MultiCurveManager",
    "CursorManager",
    "MarkRegionManager",
    "EventHandler",
    "CustomViewBox",
    "PlotContainerWidget",
    "LogViewer",
    "VariableSearchBar",
]
