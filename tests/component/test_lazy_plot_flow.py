"""P5 惰性 loader UI 解耦 component 测试（offscreen）。

验证 §2.4 的 UI 取数点在 ParquetLazyLoader（df 恒为 None）下可用：
- 拖进子图：_validate_plot_data / plot_variable（§2.4 #1/#2）
- 取数四元组与 value_cache（光标读数的数据源，§2.4 #2）
- 数值表添加列（#4/#5）与重载重建（#7 的静默清空护栏）

夹具全部走 data_factory.write_csv → convert_to_parquet 合成，
不引用 data/ 下的真实文件（隐私红线）。
"""

from __future__ import annotations

import numpy as np
import pytest

from PySide6.QtWidgets import QMessageBox

from src.data.parquet_converter import convert_to_parquet
from src.data.parquet_lazy_loader import ParquetLazyLoader
from src.data.temp_cache_dir import TempCacheDir
from src.ui.table_dialog import DataTableDialog
from tests.fixtures.data_factory import make_simple_rows, write_csv
from tests.fixtures.waits import flush_deferred_deletes, settle

N_ROWS = 20


@pytest.fixture()
def lazy_loader(qapp, tmp_path):
    """合成 CSV → parquet 惰性 loader（收尾 close 删临时目录）。"""
    csv = write_csv(
        tmp_path / "lazy_flow.csv",
        header=["time", "speed", "rpm", "flag"],
        units=["s", "km/h", "rpm", "-"],
        rows=make_simple_rows(N_ROWS),
    )
    temp = TempCacheDir.create()
    convert_to_parquet(str(csv), outdir=temp.path(), has_unit=True, sep=",")
    loader = ParquetLazyLoader(str(csv), temp)
    yield loader
    loader.close()


@pytest.fixture()
def lazy_pw(plot_factory, lazy_loader):
    """pw.data=None + plot_context.loader=惰性 loader（P6 接线后的真实形态）。"""
    pw = plot_factory(df=None)
    pw.plot_context.loader = lazy_loader
    pw.data = lazy_loader.df  # 硬性契约：None
    pw.units = lazy_loader.units
    pw.time_channels_info = lazy_loader.time_channels_info
    return pw


@pytest.fixture()
def silent_dialogs(monkeypatch):
    """QMessageBox 静态方法替身：offscreen 下模态弹窗会永久阻塞（陷阱 #4）"""
    calls = {"warning": [], "information": [], "critical": []}

    def _rec(kind):
        def fake(parent, title, text, *args, **kwargs):
            calls[kind].append((title, text))
            return QMessageBox.StandardButton.Ok
        return staticmethod(fake)

    for kind in calls:
        monkeypatch.setattr(QMessageBox, kind, _rec(kind))
    return calls


@pytest.fixture()
def table(qapp, monkeypatch, lazy_loader):
    """单表模式的 DataTableDialog，_resolve_loader 指向惰性 loader。"""
    monkeypatch.setattr(DataTableDialog, "_instance", None)
    dlg = DataTableDialog()
    dlg._resolve_loader = lambda: lazy_loader
    yield dlg
    from src.core.config import widget_alive

    if widget_alive(dlg):
        dlg.hide()
        dlg.deleteLater()
    flush_deferred_deletes()


# ---------- 拖进子图（§2.4 #1 硬阻塞 + #2 取数） ----------

def test_lazy_df_is_none_precondition(lazy_loader):
    assert lazy_loader.df is None
    assert lazy_loader.IS_LAZY is True


def test_validate_accepts_lazy_loader(lazy_pw, lazy_loader):
    """#1：非 mdf 的惰性 loader 不得落到 `pw.data is None` → (False, 没有可用的数据)"""
    ok, msg = lazy_pw._plot_data_manager._validate_plot_data("speed")
    assert ok is True
    assert msg == ""


def test_plot_numeric_column(lazy_pw, lazy_loader):
    """#2：曲线 y 值与 get_series 逐位相等（df=None 不再 TypeError）"""
    assert lazy_pw.plot_variable("speed")
    ci = lazy_pw.curves["speed"]
    expected = lazy_loader.get_series("speed").to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(ci.y_data, expected)
    # x 轴是 1 起的行号（CSV 语义，来自 get_value_from_name 四元组）
    np.testing.assert_array_equal(
        ci.original_index, np.arange(1, N_ROWS + 1, dtype=np.float32)
    )


def test_missing_variable_fails_without_touching_df(lazy_pw):
    """不存在的变量：get_series 抛 KeyError → 失败返回，不得 AttributeError on None"""
    success, msg, _, _, _ = lazy_pw._plot_data_manager._prepare_plot_data("nope")
    assert success is False
    assert "nope" in msg or msg


def test_value_cache_populated_for_cursor(lazy_pw):
    """光标读数取数入口：value_cache 缓存 (y, format)"""
    assert lazy_pw.plot_variable("rpm")
    y, fmt = lazy_pw.plot_context.value_cache["rpm"]
    assert fmt == "number"
    np.testing.assert_array_equal(
        np.asarray(y, dtype=np.float32),
        lazy_pw.plot_context.loader.get_series("rpm").to_numpy(dtype=np.float32),
    )


# ---------- 数值表（§2.4 #4/#5） ----------

def test_table_add_single_column(table, lazy_loader, silent_dialogs):
    table._handle_dropped_variable("speed")
    assert "speed" in table.get_column_names()
    got = table._df["speed"].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(
        got, lazy_loader.get_series("speed").to_numpy(dtype=np.float32)
    )
    assert not silent_dialogs["warning"]
    assert not silent_dialogs["critical"]


def test_table_missing_variable_raises_keyerror_like_mdf(table, silent_dialogs):
    """单变量入口对不存在变量抛 KeyError（§4.4 契约，与 MDF 今天行为一致；
    dropEvent 无 try/except，拖拽源只提供已存在变量。批量入口才转警告弹窗）"""
    with pytest.raises(KeyError):
        table._handle_dropped_variable("nope")
    assert "nope" not in table.get_column_names()


def test_table_batch_add_mixed(table, lazy_loader, silent_dialogs):
    """批量入口（#4）：有效列添加、无效列进警告，df=None 不崩"""
    table._handle_dropped_variables(["speed", "nope", "rpm"])
    names = table.get_column_names()
    assert "speed" in names and "rpm" in names
    assert "nope" not in names
    assert any("nope" in text for _, text in silent_dialogs["warning"])


# ---------- 重载重建（§2.4 #7 静默清空护栏） ----------

def test_update_data_keeps_columns_for_lazy(table, lazy_loader, silent_dialogs):
    """#7：修复前 df=None → 每列都进 removed，数值表被静默清空"""
    table._handle_dropped_variables(["speed", "rpm"])
    assert set(table.get_column_names()) >= {"speed", "rpm"}

    table.update_data(lazy_loader)
    settle(2)

    names = table.get_column_names()
    assert "speed" in names and "rpm" in names
    np.testing.assert_array_equal(
        table._df["speed"].to_numpy(dtype=np.float32),
        lazy_loader.get_series("speed").to_numpy(dtype=np.float32),
    )
