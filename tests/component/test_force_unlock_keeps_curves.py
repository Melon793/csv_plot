"""兜底解锁不得销毁曲线 —— 打包形态「重载 3 秒后曲线全消失」的直接回归。

历史事故链（tmp/2026-09-24/rca-lazy-reload-curve-vanish.md）：Nuitka exe 里三参
``QTimer.singleShot(msec, ctx, slot)`` 重载解析失败抛 ``TypeError`` →
``_post_reload_ui_refresh`` 永不调度 → ``_is_loading_new_data`` /
``_is_updating_data`` 双锁不释放 → 3 秒兜底 ``_force_unlock_all`` 接管，而它当时
的第一步就是 ``_safe_clear_plot_items()``：销毁全部曲线。用户看到的是「重载完
3 秒后曲线全消失」，而 ``curves`` 字典与 cursor 读数还在 —— 状态和画面彻底对不上。

修复后兜底解锁只做两件事：放平锁与按钮，以及**只重建真正失效**的曲线
（``scene() is None`` / ``_dataset`` 被清空），正常曲线原样保留。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QPushButton, QWidget

from src.ui.file_loader_manager import FileLoaderManager


class _FakeCurve:
    """``PlotDataItem`` 替身：只暴露失效判据用到的两个接口。"""

    def __init__(self, in_scene: bool = True, has_dataset: bool = True):
        self._scene = object() if in_scene else None
        self._dataset = object() if has_dataset else None

    def scene(self):
        return self._scene


class _FakeMultiCurveManager:
    def __init__(self):
        self.recreated: list[str] = []

    def _recreate_curve(self, var_name):
        self.recreated.append(var_name)


class _FakePlotWidget:
    """最小 plot widget 替身：记录兜底解锁对它做了什么。"""

    def __init__(self, curves: dict | None = None):
        self.plot_item = object()
        self.vline = None
        self.vline2 = None
        self.curves = curves if curves is not None else {}
        self._multi_curve_manager = _FakeMultiCurveManager()
        # 布景成「双锁还没放平」的样子，正好验证兜底解锁把它们放掉
        self._is_updating_data = True
        self._cached_data_version = -1
        self.updates_enabled: list[bool] = []
        self.clear_calls = 0
        self.refresh_calls: list[bool] = []

    def setUpdatesEnabled(self, flag):
        self.updates_enabled.append(flag)

    def _safe_clear_plot_items(self):
        self.clear_calls += 1

    def _queue_ui_refresh(self, immediate=False, cursor=True):
        # cursor= 是 P3 新增的场景分级参数：兜底解锁只对当前页写 cursor
        self.refresh_calls.append(immediate)


class _FakeMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.load_btn = QPushButton(self)
        self.reload_btn = QPushButton(self)
        self.reload_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self._is_loading_new_data = True
        self._is_being_destroyed = False
        self._data_version = 3
        self.plot_widgets = []


@pytest.fixture()
def host(qtbot):
    mw = _FakeMainWindow()
    qtbot.addWidget(mw)
    mgr = FileLoaderManager(mw)
    return mw, mgr


def _curve_info(curve):
    return SimpleNamespace(curve=curve)


class TestForceUnlockIsNonDestructive:
    def test_healthy_curves_survive_force_unlock(self, host):
        """正常曲线既不被清除，也不需要重建；锁与按钮照常放平。"""
        mw, mgr = host
        widget = _FakePlotWidget({"speed": _curve_info(_FakeCurve())})
        mw.plot_widgets = [SimpleNamespace(plot_widget=widget)]
        mgr._safety_unlock_version = mw._data_version

        mgr._force_unlock_all()

        assert widget.clear_calls == 0, "兜底解锁不得销毁曲线"
        assert widget.curves == {"speed": widget.curves["speed"]}
        assert widget._multi_curve_manager.recreated == []
        # 解锁语义必须完整：双锁放平 + reload 互斥锁释放 + 触发一次重绘
        assert widget._is_updating_data is False
        assert widget._cached_data_version == mw._data_version
        assert widget.updates_enabled == [True]
        assert widget.refresh_calls == [True]
        assert mw._is_loading_new_data is False
        assert mw.reload_btn.isEnabled() is True
        assert mgr._reload_in_progress is False
        assert getattr(mgr, "_safety_timer", None) is None

    def test_only_invalid_curves_get_rebuilt(self, host):
        """失效曲线（scene=None / _dataset 被清空）走重建，健康曲线不动。"""
        mw, mgr = host
        healthy = _FakeCurve()
        gone_from_scene = _FakeCurve(in_scene=False)
        dataset_wiped = _FakeCurve(has_dataset=False)
        widget = _FakePlotWidget(
            {
                "ok": _curve_info(healthy),
                "no_scene": _curve_info(gone_from_scene),
                "no_dataset": _curve_info(dataset_wiped),
            }
        )
        mw.plot_widgets = [SimpleNamespace(plot_widget=widget)]
        mgr._safety_unlock_version = mw._data_version

        mgr._force_unlock_all()

        assert widget.clear_calls == 0
        assert sorted(widget._multi_curve_manager.recreated) == [
            "no_dataset",
            "no_scene",
        ]

    def test_destroying_window_touches_nothing(self, qtbot):
        """主窗口 C++ 已销毁时原地收口（既有守卫的回归基线）。

        守卫问的是 ``_live_mw()``（shiboken 存活），不是 ``_is_being_destroyed``
        标志位，所以这里必须真把 C++ 对象删掉才等价于「兜底定时器落地时窗口没了」。
        """
        import shiboken6

        mw = _FakeMainWindow()
        qtbot.addWidget(mw)
        mgr = FileLoaderManager(mw)
        widget = _FakePlotWidget({"speed": _curve_info(_FakeCurve())})
        mw.plot_widgets = [SimpleNamespace(plot_widget=widget)]
        shiboken6.delete(mw)

        mgr._force_unlock_all()

        assert widget.clear_calls == 0
        assert widget.updates_enabled == []
        assert widget._is_updating_data is True
        assert widget._multi_curve_manager.recreated == []


class TestSchedulingFailureFallback:
    def test_scheduling_typeerror_fallback_keeps_curves(self, host, monkeypatch):
        """事故直接回归：调度失败 → 兜底解锁接管 → 曲线必须还在。

        打包形态下三参 singleShot 抛的就是这个 ``TypeError``；这里让
        ``_schedule_delayed`` 抛同样的错，复现「收尾链路整条没跑」的现场。
        """
        mw, mgr = host
        widget = _FakePlotWidget({"speed": _curve_info(_FakeCurve())})
        mw.plot_widgets = [SimpleNamespace(plot_widget=widget)]

        def boom(msec, slot):
            raise TypeError("QTimer.singleShot(...) is wrong (missing signature)")

        monkeypatch.setattr(mgr, "_schedule_delayed", boom)
        mgr._reload_in_progress = True

        with pytest.raises(TypeError):
            mgr._end_data_reload()

        # 收尾没跑：双锁仍在，但 3 秒兜底定时器已经就位
        assert mw._is_loading_new_data is True
        assert mgr._safety_timer is not None
        assert mgr._safety_unlock_version == mw._data_version

        # 兜底接管（等价于 3 秒后定时器触发）
        mgr._safety_force_unlock()

        assert widget.clear_calls == 0, "兜底解锁不得把曲线清掉"
        assert widget._multi_curve_manager.recreated == []
        assert mw._is_loading_new_data is False
        assert widget._is_updating_data is False
        assert mw.reload_btn.isEnabled() is True
        assert mgr._reload_in_progress is False

    def test_safety_unlock_skips_stale_version(self, host, monkeypatch):
        """版本不匹配的兜底解锁不得动新一轮加载的状态。"""
        mw, mgr = host
        widget = _FakePlotWidget({"speed": _curve_info(_FakeCurve())})
        mw.plot_widgets = [SimpleNamespace(plot_widget=widget)]
        monkeypatch.setattr(mgr, "_schedule_delayed", lambda msec, slot: None)
        mgr._end_data_reload()
        mw._data_version += 1  # 新一轮加载已经开始

        mgr._safety_force_unlock()

        assert mw._is_loading_new_data is True, "过期兜底不得解锁新一轮"
        assert widget.updates_enabled == []
