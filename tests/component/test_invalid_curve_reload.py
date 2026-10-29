"""失效曲线占位（方案 B2）reload 集成 component 测试（offscreen）。

钉住 Commit 2 的 reload 契约（真实 CursorSyncManager + 真实 PlotWidget + 最小宿主）：
- 部分失效：匹配曲线重建、失效曲线降级为占位（不清空、不进全清 reset）；
- 全失效：所有曲线降级为占位，curves 空但 invalid_curves 保留，不 reset；
- 纯占位页复活：curves 空、仅剩占位时 reload，占位变量重新有效 → 自动复活同色；
- 双页混合：一页正常重建、另一页失效降级，互不干扰；
- 本轮 demote 的条目不参与复活（invalid_snapshot 快照口径）；
- 状态栏汇总取代旧模态「更新通知」（无 QMessageBox）。
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox, QPushButton, QWidget

from src.ui.cursor_sync_manager import CursorSyncManager
from src.ui.table_dialog import DataTableDialog
from src.ui.widgets.plot_widget import DraggableGraphicsLayoutWidget

N = 60


class MutableLoader:
    """CSV loader 替身：df/var_names/df_validity 可就地替换以模拟 reload。"""

    LOADER_TYPE = "csv"

    def __init__(self, df: pd.DataFrame, validity: dict | None = None):
        self.set_data(df, validity)
        self.units = {}
        self.time_channels_info = {}
        self.time_column_name = None
        self.time_axis_label = "Index"

    def set_data(self, df: pd.DataFrame, validity: dict | None = None):
        self.df = df
        self.datalength = int(len(df))
        self.var_names = list(df.columns)
        self.df_validity = (
            dict(validity) if validity is not None else {v: 1 for v in df.columns}
        )

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
        self.cleared_announces = []

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

    def announce_cleared(self, label: str, curves: int) -> None:
        self.cleared_announces.append((label, curves))


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


class Container:
    def __init__(self, pw):
        self.plot_widget = pw

    def isVisible(self):
        return True

    def isHidden(self):
        return False


@pytest.fixture(autouse=True)
def _no_data_table_dialog(monkeypatch):
    monkeypatch.setattr(DataTableDialog, "_instance", None)


@pytest.fixture(autouse=True)
def modal_records(monkeypatch):
    """任何模态弹窗都替身化并记录：reload 不该再弹「更新通知」（§5.6）"""
    shown = []

    def make_recorder(kind):
        def fake(parent, title, text, *args, **kwargs):
            shown.append((kind, title, text))
            return QMessageBox.StandardButton.Ok
        return fake

    for kind in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(make_recorder(kind)))
    return shown


def _df(columns: dict[str, list[float]]):
    return pd.DataFrame(columns)


@pytest.fixture()
def env(qapp):
    """真实 CursorSyncManager + 单页可绘图 plot widget + 可变 loader"""
    host = FakeMainWindow()
    df0 = _df({"a": list(np.arange(N, dtype=float)), "b": list(np.sin(np.arange(N) / 5.0))})
    loader = MutableLoader(df0)
    ctx = FakeCtx(host, loader)
    pw = DraggableGraphicsLayoutWidget({}, df0)
    pw.plot_context = ctx
    pw.setParent(host)
    pw.resize(700, 260)
    pw.show()

    container = Container(pw)
    host.plot_widgets.append(container)
    ctx.plot_widgets = list(host.plot_widgets)
    host.loader = loader

    manager = CursorSyncManager(host)
    host.cursor_sync_manager = manager

    def reload_with(columns: dict[str, list[float]], validity: dict | None = None):
        """模拟 _apply_loader：换 loader 数据 + 同步 widget.data，再 replots"""
        new_df = _df(columns)
        loader.set_data(new_df, validity)
        for c in host.plot_widgets:
            c.plot_widget.data = new_df
        manager.replots_after_loading()

    yield host, manager, pw, loader, reload_with

    pw._is_being_destroyed = True
    pw.deleteLater()
    host.deleteLater()


def _assert_disjoint(pw):
    overlap = set(pw.curves) & set(pw.invalid_curves)
    assert not overlap, f"curves 与 invalid_curves 重叠: {overlap}"


# ---------- 部分失效 ----------

def test_partial_invalid_demotes_missing_keeps_matched(env, modal_records):
    """a 仍有效、b 缺失：a 重建、b 降级为占位（不清空、不弹窗）"""
    host, manager, pw, loader, reload_with = env
    assert pw.plot_variable("a")
    assert pw.add_variable_to_plot("b")
    pw.curves["b"].color = "red"
    assert set(pw.curves) == {"a", "b"}

    reload_with({"a": list(np.arange(N, dtype=float))})

    assert "a" in pw.curves
    assert "b" not in pw.curves
    assert "b" in pw.invalid_curves
    assert pw.invalid_curves["b"].color == "red"
    _assert_disjoint(pw)
    assert not modal_records, f"reload 不该弹模态框: {modal_records}"


def test_status_summary_reports_invalid_count(env, modal_records):
    """失效发生 → 状态栏汇总；无模态框"""
    host, manager, pw, loader, reload_with = env
    assert pw.plot_variable("a")
    assert pw.add_variable_to_plot("b")

    reload_with({"a": list(np.arange(N, dtype=float))})

    assert not modal_records
    assert any("失效" in m and "占位" in m for m in host.status_messages), host.status_messages


# ---------- 全失效 ----------

def test_all_invalid_keeps_placeholders_no_reset(env, modal_records):
    """全部变量缺失：curves 空但占位保留，不进全清 reset（§5.2）"""
    host, manager, pw, loader, reload_with = env
    assert pw.plot_variable("a")
    pw.add_variable_to_plot("b")
    pw.curves["a"].color = "green"

    reload_with({"zzz": list(np.arange(N, dtype=float))})

    assert list(pw.curves) == []
    assert set(pw.invalid_curves) == {"a", "b"}
    assert pw.invalid_curves["a"].color == "green"
    _assert_disjoint(pw)
    assert not modal_records


# ---------- 纯占位页复活 ----------

def test_pure_placeholder_page_revives_on_reload(env, modal_records):
    """curves 空、仅剩占位时 reload：占位变量重新有效 → 自动复活同色（§5.4）"""
    host, manager, pw, loader, reload_with = env
    assert pw.plot_variable("a")
    pw.curves["a"].color = "purple"
    # 先换成缺 a 的数据 → a 降级为占位
    reload_with({"zzz": list(np.arange(N, dtype=float))})
    assert list(pw.curves) == []
    assert "a" in pw.invalid_curves

    # 再换回含 a 的数据 → 复活
    reload_with({"a": list(np.arange(N, dtype=float))})

    assert "a" in pw.curves
    assert "a" not in pw.invalid_curves
    assert pw.curves["a"].color == "purple"
    _assert_disjoint(pw)


def test_this_round_demoted_not_revived_same_round(env, modal_records):
    """本轮刚 demote 的条目不参与本轮复活（invalid_snapshot 快照口径，§5.4）"""
    host, manager, pw, loader, reload_with = env
    assert pw.plot_variable("a")
    # 新数据里 a 无效（INVALID=-1）：应 demote，且因不在入口快照中不复活
    reload_with({"a": list(np.arange(N, dtype=float))}, validity={"a": -1})

    assert "a" not in pw.curves
    assert "a" in pw.invalid_curves
    _assert_disjoint(pw)


# ---------- 双页混合 ----------

def test_two_pages_mixed_valid_invalid(env, modal_records):
    """两容器：各自独立重建/降级，互不干扰"""
    host, manager, pw, loader, reload_with = env
    # 第二个容器（同宿主，均 scene=True）
    df0 = loader.df
    ctx2 = FakeCtx(host, loader)
    pw2 = DraggableGraphicsLayoutWidget({}, df0)
    pw2.plot_context = ctx2
    pw2.setParent(host)
    pw2.resize(700, 260)
    pw2.show()
    host.plot_widgets.append(Container(pw2))
    ctx2.plot_widgets = list(host.plot_widgets)

    assert pw.plot_variable("a")       # 页1 绘 a（reload 后仍有效）
    assert pw2.plot_variable("b")      # 页2 绘 b（reload 后缺失）
    pw2.curves["b"].color = "orange"

    reload_with({"a": list(np.arange(N, dtype=float))})

    assert "a" in pw.curves and not pw.invalid_curves
    assert list(pw2.curves) == [] and "b" in pw2.invalid_curves
    assert pw2.invalid_curves["b"].color == "orange"
    _assert_disjoint(pw)
    _assert_disjoint(pw2)

    pw2._is_being_destroyed = True
    pw2.deleteLater()


# ---------- 无失效时静默 ----------

def test_all_valid_reload_stays_silent(env, modal_records):
    """全部有效：正常重建，无占位、无状态栏失效汇总、无模态框"""
    host, manager, pw, loader, reload_with = env
    assert pw.plot_variable("a")
    pw.add_variable_to_plot("b")
    host.status_messages.clear()

    reload_with({
        "a": list(np.arange(N, dtype=float)),
        "b": list(np.sin(np.arange(N) / 5.0)),
    })

    assert set(pw.curves) == {"a", "b"}
    assert not pw.invalid_curves
    assert not modal_records
    assert not any("失效" in m for m in host.status_messages)
