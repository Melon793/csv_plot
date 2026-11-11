"""P1-I（第 4 轮全仓审查）：模板套用失败路径兜底。

「变量存在于数据但不可绘制」（文本列 df_validity=-1、整列 CAN 哨兵值被
过滤成全 NaN 的常量列 validity=0 等）时，修复前的 _apply_plot_config：

* 只按 ``var_name in loader.var_names`` 判定，失败不 demote —— 格子既无
  曲线也无占位，资产静默丢失（与 reload 路径的 _var_is_plottable 口径分裂）；
* 返回值被丢弃，``added_count += 1`` 无条件执行 —— 「已应用 N/M」虚高，
  全失败的格子也算 applied；
* 失败弹 QMessageBox 模态框，打断程序化批量套用。

修复后：接住返回值，失败 demote 为占位且不计数；quiet=True 抑制弹框。

harness 与 test_invalid_curve_reload.py 同构（真实 PlotWidget + 最小宿主）。
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox, QPushButton, QWidget

from src.core.plot_config import PlotConfig
from src.ui.cursor_sync_manager import CursorSyncManager
from src.ui.plot_config_manager import PlotConfigManager
from src.ui.widgets.plot_widget import DraggableGraphicsLayoutWidget

N = 60
SENTINEL = -1.7e38


class StaticLoader:
    """CSV loader 替身：b 列为全哨兵常量列（validity=0 ≠ -1 → 成员判定通过）。"""

    LOADER_TYPE = "csv"

    def __init__(self, df: pd.DataFrame, validity: dict):
        self.df = df
        self.datalength = int(len(df))
        self.var_names = list(df.columns)
        self.df_validity = dict(validity)
        self.units = {}
        self.time_channels_info = {}
        self.time_column_name = None
        self.time_axis_label = "Index"

    @property
    def global_time_range(self) -> tuple[float, float]:
        return (1.0, float(self.datalength))


class FakeCtx:
    def __init__(self, host, loader):
        self._host = host
        self.loader = loader
        self.plot_widgets = []
        self.pinned_x_values = []
        self.cursor_mode = "1 free cursor"

    @property
    def value_cache(self):
        return self._host.value_cache

    @value_cache.setter
    def value_cache(self, value):
        self._host.value_cache = value

    @property
    def _enum_text_maps(self):
        return self._host._enum_text_maps

    @_enum_text_maps.setter
    def _enum_text_maps(self, value):
        self._host._enum_text_maps = value

    def request_mark_stats_refresh(self, immediate: bool = False):
        pass


class FakeLayoutManager:
    def request_mark_stats_refresh(self, immediate: bool = False):
        pass


class FakeMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.cursor_btn = QPushButton()
        self.cursor_btn.setCheckable(True)
        self.cursor_btn.setChecked(True)
        self.cursor_values_hidden = False
        self.layout_manager = FakeLayoutManager()
        self.cursor_mode = "1 free cursor"
        self.pinned_x_values = []
        self._is_loading_new_data = False
        self._is_syncing_crosshair = False
        self._pending_crosshair_x = None
        self._crosshair_update_timer = QTimer(self)
        self.plot_widgets = []
        self.loader = None
        self.value_cache = OrderedDict()
        self._enum_text_maps = {}
        self._baseline_density = 0.0
        self._global_max_density = 0.0
        self.cursor_sync_manager = None
        self.status_messages: list[str] = []

    def _show_status_message(self, message: str):
        self.status_messages.append(message)


@pytest.fixture(autouse=True)
def modal_records(monkeypatch):
    """任何模态弹窗都替身化并记录：模板套用中途不得弹框。"""
    shown = []

    def make_recorder(kind):
        def fake(parent, title, text, *args, **kwargs):
            shown.append((kind, title, text))
            return QMessageBox.StandardButton.Ok
        return fake

    for kind in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(make_recorder(kind)))
    return shown


@pytest.fixture()
def env(qapp):
    host = FakeMainWindow()
    df = pd.DataFrame(
        {
            "a": list(np.arange(N, dtype=float)),
            "b": [SENTINEL] * N,
        }
    )
    # b：常量列 validity=0（≠-1 → 套用的成员判定与 reload 的可绘制判定都放行），
    # 但 _prepare_plot_data 的哨兵过滤会把它变成全 NaN → 备数据失败
    loader = StaticLoader(df, validity={"a": 1, "b": 0})
    ctx = FakeCtx(host, loader)
    pw = DraggableGraphicsLayoutWidget({}, df)
    pw.plot_context = ctx
    pw.setParent(host)
    pw.resize(700, 260)
    pw.show()
    host.loader = loader
    # add_variable_to_plot 收尾会走 pw.window().cursor_sync_manager._sync_min_xrange()：
    # 宿主必须有真管理器（None 属性会让 hasattr 放行后炸掉，曲线被误降级）
    host.cursor_sync_manager = CursorSyncManager(host)

    yield host, pw, loader

    pw._is_being_destroyed = True
    pw.deleteLater()
    host.deleteLater()


def test_mixed_plot_unplottable_demotes_and_not_counted(env, modal_records):
    """a 可绘、b 不可绘：a 进曲线、b 降级占位；不弹模态框。"""
    host, pw, loader = env
    mgr = PlotConfigManager()

    ok = mgr._apply_plot_config(
        pw, PlotConfig(curves=["a", "b"]), host, 0, scene=True
    )

    assert ok is True
    assert "a" in pw.curves
    assert "b" not in pw.curves
    assert "b" in pw.invalid_curves, (
        "存在但不可绘制的变量必须降级为占位，不得静默留空格子"
    )
    assert not pw.curves.keys() & pw.invalid_curves.keys()
    assert not modal_records, f"模板套用中途不得弹模态框: {modal_records}"


def test_all_unplottable_plot_returns_false(env, modal_records):
    """全部不可绘：added_count 必须为 0 → 返回 False（修复前虚高返回 True）。"""
    host, pw, loader = env
    mgr = PlotConfigManager()

    ok = mgr._apply_plot_config(
        pw, PlotConfig(curves=["b"]), host, 0, scene=True
    )

    assert ok is False, "全失败的格子不得计入 applied_count"
    assert not pw.curves
    assert "b" in pw.invalid_curves
    assert not modal_records


def test_missing_variable_still_demotes_as_before(env, modal_records):
    """正向对照：「不存在于数据」的既有 demote 行为不受影响。"""
    host, pw, loader = env
    mgr = PlotConfigManager()

    ok = mgr._apply_plot_config(
        pw, PlotConfig(curves=["a", "zzz"]), host, 0, scene=True
    )

    assert ok is True
    assert "a" in pw.curves
    assert "zzz" in pw.invalid_curves
    assert not modal_records


def test_duplicate_curve_names_do_not_remove_drawn_curve(env, modal_records):
    """评审 B1：畸形模板重名 ["a","a"]——第二次失败是「重复」不是「不可绘」，
    不得把已画好的曲线 demote 成占位（demote 会移除已成功绘制的曲线）。"""
    host, pw, loader = env
    mgr = PlotConfigManager()

    ok = mgr._apply_plot_config(
        pw, PlotConfig(curves=["a", "a"]), host, 0, scene=True
    )

    assert ok is True
    assert "a" in pw.curves, "已成功绘制的曲线不得被重名条目移除"
    assert "a" not in pw.invalid_curves
    assert not modal_records
