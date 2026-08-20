"""ExcelDataLoader 单元测试：标题行检测、加载、sheet 信息。"""

from __future__ import annotations

import openpyxl
import pytest

from src.data.excel_loader import ExcelDataLoader


def _write_xlsx(path, header, units, rows, sheet_name="Sheet1"):
    """生成 标题行 + 单位行 + 数据行 结构的小型 xlsx"""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append(header)
    if units is not None:
        ws.append(units)
    for row in rows:
        ws.append(row)
    wb.save(path)
    wb.close()
    return path


HEADER = ["time", "speed", "rpm"]
UNITS = ["s", "km/h", "rpm"]
ROWS = [[0.0, 10.5, 800], [0.1, 11.0, 810], [0.2, 11.5, 820]]


class TestDetectHeaderRowFromRows:
    def test_plain_header_first_row(self):
        rows = [
            ("time", "speed", "rpm"),
            (0.0, 10.5, 800),
        ]
        assert ExcelDataLoader._detect_header_row_from_rows(rows) == 0

    def test_metadata_region_jump(self):
        """元数据区（2 列）后接真实标题行（9 列）→ 跳跃定位"""
        rows = [
            ("File", "test.xlsx"),
            ("Author", "tester"),
            ("time", "speed", "rpm", "temp", "a", "b", "c", "d", "e"),
            (0.0, 10.5, 800, 25, 1, 2, 3, 4, 5),
        ]
        assert ExcelDataLoader._detect_header_row_from_rows(rows) == 2

    def test_numeric_only_rows_return_zero(self):
        rows = [
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
        ]
        assert ExcelDataLoader._detect_header_row_from_rows(rows) == 0


class TestExcelLoad:
    @pytest.fixture
    def simple_xlsx(self, tmp_path):
        return _write_xlsx(tmp_path / "simple.xlsx", HEADER, UNITS, ROWS)

    def test_load_with_explicit_unit_row(self, simple_xlsx):
        loader = ExcelDataLoader(
            str(simple_xlsx), sheet_name=0, desc_rows=0, has_unit=True
        )
        assert loader.var_names == HEADER
        assert loader.units["speed"] == "km/h"
        assert loader.datalength == 3
        assert loader.LOADER_TYPE == "excel"

    def test_load_auto_detect_unit_row(self, simple_xlsx):
        """has_unit=None → 自动检测单位行（单位关键字比例）"""
        loader = ExcelDataLoader(str(simple_xlsx), sheet_name=0, desc_rows=0)
        assert loader.has_unit is True
        assert loader.units["rpm"] == "rpm"
        assert loader.datalength == 3

    def test_load_without_unit_row(self, tmp_path):
        f = _write_xlsx(tmp_path / "no_unit.xlsx", ["a", "b"], None, [[1, 2], [3, 4]])
        loader = ExcelDataLoader(str(f), sheet_name=0, desc_rows=0, has_unit=False)
        assert loader.datalength == 2
        assert loader.units["a"] == "-"

    def test_validity_and_data_values(self, simple_xlsx):
        loader = ExcelDataLoader(
            str(simple_xlsx), sheet_name=0, desc_rows=0, has_unit=True
        )
        assert loader.df_validity["speed"] == 1
        assert loader.df["rpm"].iloc[0] == pytest.approx(800)

    def test_select_sheet_by_name(self, tmp_path):
        f = _write_xlsx(tmp_path / "named.xlsx", HEADER, UNITS, ROWS, sheet_name="Data")
        loader = ExcelDataLoader(
            str(f), sheet_name="Data", desc_rows=0, has_unit=True
        )
        assert loader.datalength == 3


class TestGetSheetInfo:
    def test_sheet_info(self, tmp_path):
        f = _write_xlsx(tmp_path / "info.xlsx", HEADER, UNITS, ROWS)
        info = ExcelDataLoader.get_sheet_info(str(f))
        assert len(info) == 1
        assert info[0]["name"] == "Sheet1"
        assert info[0]["rows"] == 5  # 标题 + 单位 + 3 数据行
        assert info[0]["cols"] == 3
