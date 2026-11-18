"""PlotConfigManager._refresh_axis_segment 的守卫回归。

状态栏 x 轴段是缓存文案：模板/auto-save 套用写完全局 factor/offset 后必须
补刷（否则抽屉现读 mw.factor 显示 0.1、底部还停在「x轴：Index」）。但补刷
发生在写入之后，属装饰性收尾——控件已销毁等异常不得把已经落地的套用翻成
「应用失败」。
"""

from src.ui.plot_config_manager import PlotConfigManager


def test_missing_method_is_ignored():
    """旧式替身没有 _update_axis_segment：静默跳过，不抛异常。"""

    class Stub:
        pass

    PlotConfigManager._refresh_axis_segment(Stub())


def test_refresh_exception_is_swallowed():
    """补刷路径抛异常（如控件已销毁）：不得向外传播。"""

    class Stub:
        def _update_axis_segment(self):
            raise RuntimeError("Internal C++ object already deleted")

    PlotConfigManager._refresh_axis_segment(Stub())


def test_refresh_calls_through():
    """正常宿主：确实调用一次，且不额外要求返回值。"""
    calls = []

    class Stub:
        def _update_axis_segment(self):
            calls.append(True)

    PlotConfigManager._refresh_axis_segment(Stub())
    assert calls == [True]