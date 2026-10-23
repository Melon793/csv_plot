"""CSV/TXT 空表头单元格归一化回归测试。

pandas 3.0 的 ``astype(str)`` 会把缺失值保留为 float NaN。CSV 表头末尾多一个
分隔符时，多出来的空列名会以 float 进入后续 schema 推断与 UI，导致
``col.lower()`` / 排序 / 筛选 / 绘制抛异常。加载层必须把它归一为
``Unnamed: N``（0-based 列位置）。
"""

from __future__ import annotations

import pytest

from src.data.loader import FastDataLoader
from tests.fixtures.data_factory import write_csv


def test_trailing_empty_header_becomes_unnamed(tmp_path):
    """表尾空表头 + 全空列：名称全为 str，末列归一为 Unnamed: 3 且无效。"""
    path = write_csv(
        tmp_path / "Demo.csv",
        header=["alpha", "beta", "gamma", ""],
        units=["m", "s", "m", ""],
        rows=[[i, i * 2, i * 3, ""] for i in range(6)],
    )

    loader = FastDataLoader(str(path))

    assert all(isinstance(name, str) for name in loader.var_names)
    assert loader.var_names == ["alpha", "beta", "gamma", "Unnamed: 3"]
    # 全空列没有数据 → 无效；名称归一不应改变有效性判定
    assert loader.df_validity["Unnamed: 3"] == -1


def test_empty_header_with_data_is_normalized_before_schema_inference(tmp_path):
    """空表头列有数据：归一化必须发生在 _infer_schema 之前，文件应能加载。"""
    path = write_csv(
        tmp_path / "Demo2.csv",
        header=["alpha", "", "gamma"],
        units=["m", "s", "m"],
        rows=[[i, i * 10, i * 3] for i in range(6)],
    )

    loader = FastDataLoader(str(path))

    assert loader.var_names == ["alpha", "Unnamed: 1", "gamma"]
    assert loader.df_validity["Unnamed: 1"] == 1
    assert loader.datalength == 6
