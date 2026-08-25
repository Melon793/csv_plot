"""component 测试共享夹具：可独立绘图的 plot widget"""

import pandas as pd
import pytest

from PySide6.QtWidgets import QWidget


class FakePlotContext:
    """最小 PlotContext 替身（真实实现见 src/app/plot_context.py，依赖 MainWindow 服务）。

    覆盖绘图主路径访问的属性：value_cache / loader / _enum_text_maps /
    request_mark_stats_refresh，使 DraggableGraphicsLayoutWidget 可脱离
    MainWindow 独立执行 plot_variable / dropEvent。
    """

    def __init__(self):
        self.value_cache = {}
        self.loader = None
        self._enum_text_maps = {}

    def request_mark_stats_refresh(self, immediate: bool = False):
        pass


class FakeLayoutManager:
    """layout_manager 替身：覆盖 drop/拖拽路径访问的方法（均 no-op）"""

    def request_mark_stats_refresh(self, immediate: bool = False):
        pass

    def _hide_drag_indicator_for_plot(self, pw):
        pass

    def _show_drag_indicator_for_plot(self, pw, var_names, text=None):
        pass

    def _get_plot_container(self, pw):
        return None


class FakeHost(QWidget):
    """伪宿主窗口：提供 layout_manager 替身。

    plot_widget.dropEvent 末尾会调 self.window().layout_manager.
    request_mark_stats_refresh()，独立顶层 widget 缺少该属性会抛
    AttributeError（真实应用中 window() 为 MainWindow，不暴露）。
    """

    def __init__(self):
        super().__init__()
        self.layout_manager = FakeLayoutManager()


@pytest.fixture()
def plot_factory(qapp):
    """构造可独立绘图的 DraggableGraphicsLayoutWidget 工厂"""
    created = []
    hosts = []

    def _make(df: pd.DataFrame | None = None):
        from src.ui.widgets.plot_widget import DraggableGraphicsLayoutWidget
        if df is None:
            df = pd.DataFrame({
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0],
                "c": [7.0, 8.0, 9.0],
            })
        pw = DraggableGraphicsLayoutWidget({}, df)
        pw.plot_context = FakePlotContext()
        host = FakeHost()
        pw.setParent(host)  # window() 返回 host，提供 layout_manager 替身
        created.append(pw)
        hosts.append(host)
        return pw

    yield _make

    for pw in created:
        pw._is_being_destroyed = True
        pw.deleteLater()
    for host in hosts:
        host.deleteLater()
