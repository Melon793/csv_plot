"""
BasePlotManager —— 所有绘图组件管理器的基类

提供通用的弱引用模式，避免管理器与主类之间的循环引用，
以及统一的生命周期钩子方法。
"""

from __future__ import annotations
import weakref
from typing import Any


class BasePlotManager:
    """所有绘图管理器的基类，提供统一的弱引用和生命周期管理。
    
    子类应该继承此类，并在需要时重写生命周期钩子方法。
    """
    
    def __init__(self, plot_widget: Any):
        """初始化管理器
        
        Args:
            plot_widget: 关联的绘图组件 (DraggableGraphicsLayoutWidget 或 MainWindow)
        """
        self._pw_ref = weakref.ref(plot_widget)
    
    @property
    def pw(self) -> Any:
        """获取关联的 plot_widget 引用，安全检查是否已被销毁。
        
        Returns:
            关联的 plot_widget 对象
            
        Raises:
            RuntimeError: 如果 plot_widget 已被垃圾回收
        """
        pw = self._pw_ref()
        if pw is None:
            raise RuntimeError(
                f"{type(self).__name__}: PlotWidget has been garbage collected"
            )
        return pw
    
    def initialize(self) -> None:
        """在主类 setup_ui 完成后调用，用于执行初始化后的额外设置。
        
        子类可重写此方法以初始化计时器、连接信号等。
        """
        pass
    
    def cleanup(self) -> None:
        """在 plot_widget 销毁前调用，用于释放资源、断开信号等。
        
        子类可重写此方法以执行清理工作。
        """
        pass
    
    def reset(self) -> None:
        """在 reset_plot 或 clear_plot_item 后调用，用于重置管理器特有状态。
        
        子类可重写此方法以清空内部缓存、重置标志位等。
        """
        pass
