"""
ui.widgets —— 绘图相关的组件模块

此模块包含用于绘图界面的自定义组件，
支持 DraggableGraphicsLayoutWidget 的功能。
"""

from src.ui.widgets.custom_viewbox import CustomViewBox
from src.ui.widgets.plot_container import PlotContainerWidget
from src.ui.widgets.base_manager import BasePlotManager

__all__ = [
    "CustomViewBox",
    "PlotContainerWidget",
    "BasePlotManager",
]
