"""P2-13：绘图区被主动隐藏时，`_ensure_splitter_ready` 不该继续 20Hz 轮询。

轮询的判据是「两个 pane 都 > 0」，用来等首次布局落地。但隐藏绘图区就是把右
pane 压成 0（`setSizes([width, 0])`）—— 隐藏期间这条判据永远不会成立，于是每
50ms 自我续排一次，直到窗口关闭。
"""

from __future__ import annotations

import pytest

from PySide6.QtWidgets import QPushButton, QSplitter, QWidget

from src.ui import layout_manager as lm_mod
from src.ui.layout_manager import LayoutManager


@pytest.fixture()
def env(qapp, monkeypatch):
    """伪主窗口 + 记录型 singleShot：轮询是否续排、间隔多少，全看排了什么。"""
    scheduled: list[int] = []
    monkeypatch.setattr(
        lm_mod.QTimer,
        "singleShot",
        staticmethod(lambda delay, cb: scheduled.append(delay)),
    )

    mw = QWidget()
    mw.main_splitter = QSplitter(mw)
    left, right = QWidget(mw.main_splitter), QWidget(mw.main_splitter)
    mw.main_splitter.addWidget(left)
    mw.main_splitter.addWidget(right)
    mw.plot_widget = QWidget(mw)
    mw.toggle_plot_btn = QPushButton(mw)
    mw._splitter_ready = False
    mw._plot_area_visible = True
    mw.var_table_user_adjusted = False
    mw._saved_splitter_sizes = [300, 400]
    mw._saved_geometry = None
    mw._was_maximized = False
    mw._was_fullscreen = False
    manager = LayoutManager(mw)
    mw._kept = manager
    return mw, manager, scheduled


def _sizes(mw, left: int, right: int):
    """绕开 QSplitter 自己的分配逻辑，直接钉住两个 pane 的宽度。"""
    mw.main_splitter.setSizes([left, right])
    mw.main_splitter.widget(1).setMaximumWidth(right)
    mw.main_splitter.widget(1).setMinimumWidth(right)
    mw.main_splitter.widget(0).setMaximumWidth(left)
    mw.main_splitter.widget(0).setMinimumWidth(left)
    return mw.main_splitter.sizes()


class TestEnsureSplitterReadyPolling:
    def test_ready_once_both_panes_have_width(self, env):
        mw, manager, scheduled = env
        assert all(s > 0 for s in _sizes(mw, 300, 400)), "现场自检：两个 pane 都有宽度"

        manager._ensure_splitter_ready()

        assert mw._splitter_ready is True
        assert scheduled == []

    def test_rearms_while_plot_area_is_visible(self, env):
        """「还没布局好」的常态：继续按 50ms 等。"""
        mw, manager, scheduled = env
        assert _sizes(mw, 700, 0)[1] == 0, "现场自检：右 pane 还没拿到宽度"

        manager._ensure_splitter_ready()

        assert scheduled == [50], f"应续排一次 50ms，实际 {scheduled}"
        assert mw._splitter_ready is False

    def test_stops_polling_while_plot_area_hidden(self, env):
        """隐藏绘图区 = 右 pane 恒为 0 的预期状态，不是没布局好。"""
        mw, manager, scheduled = env
        mw._plot_area_visible = False
        assert _sizes(mw, 700, 0)[1] == 0

        manager._ensure_splitter_ready()

        assert scheduled == [], f"隐藏期间不该再空转: {scheduled}"
        assert mw._splitter_ready is False, "也不该把就绪标记谎报成 True"


class TestResumeOnShow:
    def test_showing_the_plot_area_restarts_one_poll(self, env):
        """补起轮询，否则 _splitter_ready 永久为 False，自动宽度机制跟着失效。"""
        mw, manager, scheduled = env
        mw._plot_area_visible = False

        manager.toggle_plot_area(False)  # False → 恢复显示

        assert mw._plot_area_visible is True
        assert scheduled == [0], f"恢复可见后须补起一次轮询: {scheduled}"

    def test_no_extra_poll_when_already_ready(self, env):
        mw, manager, scheduled = env
        mw._plot_area_visible = False
        mw._splitter_ready = True

        manager.toggle_plot_area(False)

        assert scheduled == []
