"""P1-M（上轮 P1-11）：剪贴板复制的向量化 TSV 组装。

逐格 ``df.iloc[r, c]`` 标量访问在大表「复制全部」时是百万级调用，GUI
线程冻结分钟级。``_rows_to_tsv_lines`` 改为整块 ``to_numpy(object)`` +
NaN 掩码 + ``str()`` 列表推导。本文件锁住文本语义：NaN → 空串、其余
``str(val)``、列序/行序按传入子矩阵保持。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ui.table_dialog import DataTableDialog


@pytest.fixture()
def fake_clipboard(qapp, monkeypatch):
    """offscreen 下不依赖平台剪贴板：直接截获 setText。"""

    class _Fake:
        def __init__(self):
            self.text = None

        def setText(self, value):
            self.text = value

    fake = _Fake()
    monkeypatch.setattr(
        "src.ui.table_dialog.QApplication.clipboard",
        staticmethod(lambda: fake),
    )
    return fake


class TestRowsToTsvLines:
    def test_nan_becomes_empty_string(self):
        df = pd.DataFrame({"a": [1.5, np.nan], "b": [np.nan, 2.0]})
        assert DataTableDialog._rows_to_tsv_lines(df) == ["1.5\t", "\t2.0"]

    def test_str_semantics_match_scalar_str(self):
        """向量化路径的 str() 必须与逐格 df.iloc 的 str() 一致。"""
        df = pd.DataFrame(
            {
                "int": [1, 2],
                "float": [0.1, 1e20],
                "str": ["x", "含\t制表"],
                "bool": [True, False],
            }
        )
        expected = []
        for r in range(df.shape[0]):
            row = [str(df.iloc[r, c]) for c in range(df.shape[1])]
            expected.append("\t".join(row))
        assert DataTableDialog._rows_to_tsv_lines(df) == expected

    def test_column_and_row_order_follow_input(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        sub = df.iloc[[2, 0], [1, 0]]
        assert DataTableDialog._rows_to_tsv_lines(sub) == ["6\t3", "4\t1"]

    def test_empty_matrix_returns_no_lines(self):
        assert DataTableDialog._rows_to_tsv_lines(pd.DataFrame()) == []
        assert (
            DataTableDialog._rows_to_tsv_lines(pd.DataFrame({"a": []})) == []
        )


class TestCopyPathsUseVectorizedRows:
    def test_copy_selected_tsv_layout(self, fake_clipboard):
        dlg = DataTableDialog()
        try:
            df = pd.DataFrame(
                {"time": [0.0, 0.1, 0.2], "speed": [10.0, np.nan, 30.0]}
            )
            dlg.units = {"time": "s", "speed": "km/h"}
            # 行序倒置 + 只取第 2 列：验证按传入索引组装
            dlg._copy_selected_to_clipboard([1], [2, 0], df=df)
            assert fake_clipboard.text == (
                "speed\nkm/h\n30.0\n10.0"
            )
        finally:
            dlg.deleteLater()

    def test_copy_all_tab_mode_tsv_layout(self, fake_clipboard):
        dlg = DataTableDialog()
        try:
            tab_df = pd.DataFrame(
                {"time": [0.0, 0.1], "flag": [1.0, np.nan]}
            )
            dlg.units = {"time": "s", "flag": "-"}
            # tab 模式：df 非 self._df，按列序直接复制
            dlg._copy_all_to_clipboard(df=tab_df)
            assert fake_clipboard.text == (
                "time\tflag\ns\t-\n0.0\t1.0\n0.1\t"
            )
        finally:
            dlg.deleteLater()
