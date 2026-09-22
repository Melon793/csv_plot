"""P5 枚举列绘图新路径 component 测试（D8 双入口，offscreen）。

覆盖 §6.1 的枚举专项：
- 低基数文本列在惰性 loader 下可绘（validity=1 的有意正向差异）
- 曲线 y = 码值、y_format="enum"、_enum_text_maps 提供 {码: 标签}
  （光标读数的唯一消费源，cursor_manager 的 enum_map.get(int(y))）
- get_series 仍返回文本 category（数值表/统计的数据源，不得是码值）
- 高基数文本列保持 validity=-1、不可绘（与今天一致）
- MDF 枚举夹具兜底：D8 把 y 来源改成 get_value_from_name 后，
  MDF 枚举通道的曲线值必须与 get_series 逐位相同（对 MDF 是值等价空操作）
"""

from __future__ import annotations

import numpy as np
import pytest

from PySide6.QtWidgets import QMessageBox

from src.data.mdf_lazy_loader import MDFLazyLoader
from src.data.parquet_converter import convert_to_parquet
from src.data.parquet_lazy_loader import ParquetLazyLoader
from src.data.temp_cache_dir import TempCacheDir
from src.data.var_info import compute_stats
from src.ui.table_dialog import DataTableDialog
from tests.fixtures.data_factory import ENUM_TEXTS, write_csv, write_mdf
from tests.fixtures.waits import flush_deferred_deletes

STATES = ["ON", "OFF", "STANDBY"]
N_ROWS = 12
N_HIGH_CARD = 210  # > ENUM_LABEL_MAX(200)：高基数文本列


def _rows():
    rows = []
    for i in range(N_HIGH_CARD):
        rows.append(
            [f"{i * 0.1:.1f}", f"{10.0 + i * 0.5:.2f}", STATES[i % 3], f"n{i:04d}"]
        )
    return rows


@pytest.fixture()
def enum_loader(qapp, tmp_path):
    """state=低基数枚举列；noise=高基数文本列（210 unique）。"""
    csv = write_csv(
        tmp_path / "enum_flow.csv",
        header=["time", "speed", "state", "noise"],
        units=["s", "km/h", "-", "-"],
        rows=_rows(),
    )
    temp = TempCacheDir.create()
    convert_to_parquet(str(csv), outdir=temp.path(), has_unit=True, sep=",")
    loader = ParquetLazyLoader(str(csv), temp)
    yield loader
    loader.close()


@pytest.fixture()
def enum_pw(plot_factory, enum_loader):
    pw = plot_factory(df=None)
    pw.plot_context.loader = enum_loader
    pw.data = enum_loader.df  # None
    pw.units = enum_loader.units
    pw.time_channels_info = enum_loader.time_channels_info
    return pw


@pytest.fixture()
def silent_dialogs(monkeypatch):
    calls = {"warning": [], "information": [], "critical": []}

    def _rec(kind):
        def fake(parent, title, text, *args, **kwargs):
            calls[kind].append((title, text))
            return QMessageBox.StandardButton.Ok
        return staticmethod(fake)

    for kind in calls:
        monkeypatch.setattr(QMessageBox, kind, _rec(kind))
    return calls


# ---------- 有意差异清单（§6.2，D8） ----------

def test_enum_validity_is_intentional_diff(enum_loader):
    """低基数文本列 validity：内存 loader 今天 -1，惰性 loader 有意为 1"""
    assert enum_loader.df_validity["state"] == 1
    assert enum_loader.meta("state")["is_enum"] is True
    # 高基数文本列与今天一致：不可绘
    assert enum_loader.df_validity["noise"] == -1
    assert enum_loader.meta("noise")["is_enum"] is False


def test_enum_dual_entry_contract(enum_loader):
    """get_series 给文本（数值表/统计），get_value_from_name 给码值+标签表"""
    series = enum_loader.get_series("state")
    assert str(series.dtype) == "category"
    assert series.tolist() == [STATES[i % 3] for i in range(N_HIGH_CARD)]

    _, codes, _, text_map = enum_loader.get_value_from_name("state")
    assert codes.dtype.kind in "iu"
    assert list(codes[:6]) == [0, 1, 2, 0, 1, 2]  # polars 出现序：ON,OFF,STANDBY
    assert text_map == {0: "ON", 1: "OFF", 2: "STANDBY"}

    # 高基数列 text_map 为空 → 不进枚举分支
    _, _, _, tm_noise = enum_loader.get_value_from_name("noise")
    assert tm_noise == {}


def test_enum_stats_still_non_numeric(enum_loader):
    """统计不得对码值算出数字（比崩更糟的静默降级）"""
    stats = compute_stats(enum_loader, "state")
    assert stats.error is not None
    assert "非数值" in stats.error


# ---------- 绘图新路径 ----------

def test_enum_column_plots_as_codes(enum_pw, enum_loader):
    """曲线 y = 码值、y_format="enum"、标签表进 _enum_text_maps（光标读数源）"""
    assert enum_pw.plot_variable("state")
    ci = enum_pw.curves["state"]
    assert ci.y_format == "enum"
    np.testing.assert_array_equal(
        ci.y_data.astype(np.int64),
        np.asarray([i % 3 for i in range(N_HIGH_CARD)], dtype=np.int64),
    )
    enum_map = enum_pw.plot_context._enum_text_maps["state"]
    assert enum_map == {0: "ON", 1: "OFF", 2: "STANDBY"}
    # 光标读数消费形态（cursor_manager: enum_map.get(int(y_val), str(y_val))）
    assert enum_map.get(int(ci.y_data[1])) == "OFF"


def test_enum_column_shows_text_in_table(qapp, monkeypatch, enum_loader, silent_dialogs):
    """数值表显示文本，不是 0/1/2"""
    monkeypatch.setattr(DataTableDialog, "_instance", None)
    dlg = DataTableDialog()
    dlg._resolve_loader = lambda: enum_loader
    try:
        dlg._handle_dropped_variable("state")
        assert "state" in dlg.get_column_names()
        shown = dlg._df["state"].tolist()
        assert shown == [STATES[i % 3] for i in range(N_HIGH_CARD)]
    finally:
        from src.core.config import widget_alive

        if widget_alive(dlg):
            dlg.hide()
            dlg.deleteLater()
        flush_deferred_deletes()


def test_high_cardinality_column_not_plottable(enum_pw):
    """高基数文本列：与今天一致，取数失败（validity=-1 在变量列表侧灰显）"""
    success, _, _, _, _ = enum_pw._plot_data_manager._prepare_plot_data("noise")
    assert success is False


# ---------- MDF 枚举兜底（防 D8 的 y 来源改动回归 MDF） ----------

def test_mdf_enum_curve_unchanged(qapp, tmp_path, plot_factory):
    """MDF 枚举通道：曲线值必须与 get_series 逐位相同（两入口同源）"""
    path = write_mdf(tmp_path / "enum.mf4", version="4.10", n=N_ROWS)
    loader = MDFLazyLoader(str(path))
    try:
        pw = plot_factory(df=None)
        pw.plot_context.loader = loader
        pw.data = loader.df  # None
        pw.units = loader.units
        pw.time_channels_info = loader.time_channels_info

        assert pw.plot_variable("State")
        ci = pw.curves["State"]
        assert ci.y_format == "enum"
        raw = np.asarray(loader.get_series("State"), dtype=np.float32)
        np.testing.assert_array_equal(ci.y_data, raw)
        enum_map = pw.plot_context._enum_text_maps["State"]
        assert enum_map == ENUM_TEXTS
    finally:
        loader.close()
