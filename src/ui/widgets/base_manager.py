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
