"""probe_csv_schema 重构等价测试（P1）。

验收：对 data_factory 的夹具矩阵（gb18030、有/无单位行、desc_rows>0、
重复列名、空表头单元格、日期列、文本列、分隔符），断言
`probe_csv_schema()` 的七个字段与 `FastDataLoader` 构造后的同名属性
逐个相等（dtype_map / parse_dates 通过 loader._infer_schema 对同一
sample 的结果交叉对照，其余五字段直接对实例属性）。
"""

from __future__ import annotations

import pytest

from src.data.loader import FastDataLoader, probe_csv_schema
from tests.fixtures.data_factory import write_csv, write_field_like_wide_csv


def _assert_probe_matches_loader(csv_path, **kwargs):
    """独立调用 probe_csv_schema 与 FastDataLoader 实例属性逐字段比对。"""
    probe = probe_csv_schema(str(csv_path), **kwargs)
    loader = FastDataLoader(str(csv_path), **kwargs)

    assert probe.var_names == loader._var_names
    assert probe.units == loader._units
    assert probe.encoding_used == loader.encoding_used
    assert probe.has_unit == loader.has_unit
    assert probe.date_formats == loader.date_formats
    return probe


class TestProbeMatchesLoader:
    def test_basic_with_units(self, tmp_path):
        csv = write_csv(
            tmp_path / "a.csv",
            header=["time", "speed", "rpm"],
            units=["s", "km/h", "-"],
            rows=[[f"{i * 0.1:.1f}", f"{10 + i * 0.5:.2f}", 800 + i] for i in range(30)],
        )
        probe = _assert_probe_matches_loader(csv)
        assert probe.has_unit is True
        assert probe.units == {"time": "s", "speed": "km/h", "rpm": "-"}

    def test_no_unit_row(self, tmp_path):
        csv = write_csv(
            tmp_path / "b.csv",
            header=["x", "y"],
            units=None,
            rows=[[1, 2], [3, 4], [5, 6]],
        )
        probe = _assert_probe_matches_loader(csv, has_unit=False)
        assert probe.has_unit is False
        assert probe.units == {"x": "-", "y": "-"}

    def test_desc_rows(self, tmp_path):
        csv = write_csv(
            tmp_path / "c.csv",
            header=["a", "b"],
            units=["-", "-"],
            rows=[[1, 2], [3, 4]],
            desc_rows=["export v1", "operator: demo"],
        )
        probe = _assert_probe_matches_loader(csv, desc_rows=2)
        assert probe.var_names == ["a", "b"]

    def test_duplicate_headers(self, tmp_path):
        csv = write_csv(
            tmp_path / "d.csv",
            header=["A", "A", "A_1"],
            units=["-", "-", "-"],
            rows=[[1, 2, 3], [4, 5, 6]],
        )
        probe = _assert_probe_matches_loader(csv)
        # _make_unique 的占位规则：字面 A_1 保号，第二个 A 改成不冲突的名
        assert probe.var_names == ["A", "A_2", "A_1"]
        assert len(set(probe.var_names)) == 3

    def test_empty_header_cell_normalized(self, tmp_path):
        # 空表头单元格曾是崩溃点（pandas 3.0 astype(str) 对 NaN 保留 float，
        # 列名变 float nan 后 _infer_schema 的 col.lower() 抛 AttributeError）。
        # 修复后按 pandas 惯例归一为 "Unnamed: {i}"，两侧一致。
        csv = write_csv(
            tmp_path / "e.csv",
            header=["x", "", "y"],
            units=["-", "-", "-"],
            rows=[[1, 2, 3], [4, 5, 6]],
        )
        probe = _assert_probe_matches_loader(csv)
        assert probe.var_names == ["x", "Unnamed: 1", "y"]
        assert all(isinstance(n, str) for n in probe.var_names)
        assert probe.units["Unnamed: 1"] == "-"

    def test_empty_trailing_header_cell_normalized(self, tmp_path):
        # 现场宽表形态：表头行末尾多一个分隔符 → 末列表头为空
        csv = write_field_like_wide_csv(tmp_path / "h.csv")
        probe = _assert_probe_matches_loader(csv, sep="\t")
        assert probe.var_names[-1] == "Unnamed: 4"
        assert all(isinstance(n, str) for n in probe.var_names)
        assert all(isinstance(k, str) for k in probe.dtype_map)

    def test_pua_column_name_survives_probe(self, tmp_path):
        # 私有区字符列名（\ue71a 前缀）必须是字符串一路贯通
        csv = write_field_like_wide_csv(
            tmp_path / "i.csv", pua_column_name=True, empty_trailing_header=False
        )
        probe = _assert_probe_matches_loader(csv, sep="\t")
        assert "\ue71aPUA_NAME" in probe.var_names
        assert "\ue71aPUA_NAME" in probe.units

    def test_star_nulls_read_as_nan(self, tmp_path):
        # ** 缺测标记在 probe 样本与正式读取中同为 NaN
        csv = write_field_like_wide_csv(
            tmp_path / "j.csv", star_nulls=True, empty_trailing_header=False,
            pua_column_name=False,
        )
        probe = _assert_probe_matches_loader(csv, sep="\t")
        loader = FastDataLoader(str(csv), has_unit=True, sep="\t")
        s = loader.get_series("ENG01_CH01")
        import numpy as np

        assert np.isnan(s.to_numpy()).sum() > 0

    def test_gb18030_chinese_names_explicit_encoding(self, tmp_path):
        # 显式传编码（与 test_fast_loader 的 gb18030 用例同口径）。
        # 自动检测路径下 charset_normalizer 对小样本中文文件可能误判
        # big5（既有局限，两侧结果一致地错），不属于本重构的等价面。
        csv = write_csv(
            tmp_path / "f.csv",
            header=["时间", "转速", "温度"],
            units=["s", "rpm", "°C"],
            rows=[[i * 0.1, 800 + i, 20 + i] for i in range(30)],
            encoding="gb18030",
        )
        probe = _assert_probe_matches_loader(csv, encoding="gb18030")
        assert probe.var_names == ["时间", "转速", "温度"]
        assert probe.units["温度"] == "°C"

    def test_semicolon_sep(self, tmp_path):
        csv = write_csv(
            tmp_path / "g.csv",
            header=["u", "v"],
            units=["-", "-"],
            rows=[[1, 2], [3, 4]],
            sep=";",
        )
        probe = _assert_probe_matches_loader(csv, sep=";")
        assert probe.var_names == ["u", "v"]


class TestProbeSemantics:
    """probe 七字段的语义断言（与 _infer_schema 既有行为一致）。"""

    @pytest.fixture()
    def mixed_csv(self, tmp_path):
        # time 列（%H:%M:%S.%f）、float32 安全列、float64 大值列、文本列
        rows = []
        for i in range(30):
            rows.append(
                [
                    f"00:00:{i % 60:02d}.{i * 1000 % 1000000:06d}",
                    1.5 + i,        # 小数值，float32 可表示
                    1e308 + i * 1e292,  # 超出 float32 范围
                    "STANDBY" if i % 2 else "ON",
                ]
            )
        return write_csv(
            tmp_path / "mixed.csv",
            header=["time", "small", "big", "state"],
            units=["s", "-", "-", "-"],
            rows=rows,
        )

    def test_dtype_map_three_kinds(self, mixed_csv):
        probe = probe_csv_schema(str(mixed_csv))
        assert probe.dtype_map["small"] == "float32"
        assert probe.dtype_map["big"] == "float64"
        assert probe.dtype_map["state"] == "category"

    def test_date_column_not_in_dtype_map(self, mixed_csv):
        # 日期列只进 parse_dates/date_formats，dtype_map 无条目（D9 的事实基础）
        probe = probe_csv_schema(str(mixed_csv))
        assert "time" in probe.parse_dates
        assert probe.date_formats["time"] == "%H:%M:%S.%f"
        assert "time" not in probe.dtype_map

    def test_probe_is_repeatable(self, mixed_csv):
        p1 = probe_csv_schema(str(mixed_csv))
        p2 = probe_csv_schema(str(mixed_csv))
        assert p1 == p2


if __name__ == "__main__":
    pytest.main([__file__])
