"""P2-6：向单表赋一条比当前表更长的列，尾部真实值不得被截断成 NaN。

`df[col] = series` 是**按现有索引对齐**的：长出来的部分先被丢掉，随后那句
`reindex(range(max_len))` 只把表撑高、给新列补 NaN —— 「支持变长列」的意图被前
一步抵消，真实数据静默丢失。修法是先扩容再赋列。
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from PySide6.QtCore import QCoreApplication

from src.ui.table_dialog import DataTableDialog


@pytest.fixture()
def dlg(qapp, app_settings):
    dialog = DataTableDialog()
    yield dialog
    dialog.hide()
    dialog.deleteLater()
    end = time.monotonic() + 0.02
    while time.monotonic() < end:
        QCoreApplication.processEvents()


def test_longer_column_keeps_its_tail_values(dlg):
    dlg._add_variable_to_table("a", pd.Series([1.0, 2.0, 3.0], name="a"))
    dlg._add_variable_to_table("b", pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], name="b"))

    assert len(dlg._df) == 5
    assert list(dlg._df["b"]) == [10.0, 20.0, 30.0, 40.0, 50.0], "新列尾部不得变 NaN"
    # 短列补空是原有语义（P2-6 之前的 reindex 就是为此），保持缺值而非伪造
    assert list(dlg._df["a"][:3]) == [1.0, 2.0, 3.0]
    assert dlg._df["a"].iloc[3:].isna().all()


def test_shorter_column_still_grows_the_frame(dlg):
    """反向对照：先加长列、再加短列时，表高不回缩，短列尾部为空。"""
    dlg._add_variable_to_table("long", pd.Series([1.0, 2.0, 3.0, 4.0], name="long"))
    dlg._add_variable_to_table("short", pd.Series([9.0, 8.0], name="short"))

    assert len(dlg._df) == 4
    assert list(dlg._df["long"]) == [1.0, 2.0, 3.0, 4.0]
    assert list(dlg._df["short"][:2]) == [9.0, 8.0]
    assert dlg._df["short"].iloc[2:].isna().all()


def test_model_row_count_matches_the_grown_frame(dlg):
    """表高变了 model 必须跟着变，否则界面只画前 3 行、第 4/5 个值看不见。"""
    dlg._add_variable_to_table("a", pd.Series([1.0, 2.0, 3.0], name="a"))
    dlg._add_variable_to_table("b", pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], name="b"))

    assert dlg.model.rowCount() == 5
    assert dlg.model.columnCount() == 2
    assert "50" in str(dlg.model.data(dlg.model.index(4, 1)))


def test_equal_length_column_is_unaffected(dlg):
    dlg._add_variable_to_table("a", pd.Series([1.0, 2.0, 3.0], name="a"))
    dlg._add_variable_to_table("b", pd.Series([4.0, 5.0, 6.0], name="b"))

    assert len(dlg._df) == 3
    assert list(dlg._df["b"]) == [4.0, 5.0, 6.0]
