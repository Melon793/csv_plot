"""
BasePlotManager —— 绘图组件管理器的基类

采用"单一 weakref 锚点"设计：仅链首 (PlotUIManager) 持有 plot_widget 的
weakref，其余管理器经依赖链委托访问 pw，避免循环引用。链中各 manager 的
pw property 在依赖被外部置 None 时抛统一的 RuntimeError（详见各 manager）。
"""

from __future__ import annotations
import weakref
from typing import Any


class BasePlotManager:
    """绘图管理器基类，提供弱引用锚点与统一的 GC 错误契约。

    设计上仅链首管理器继承本类持有 weakref；链中其余管理器通过依赖链
    委托访问 pw，并在断链时抛 RuntimeError（错误信息含管理器名）。
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
