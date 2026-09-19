"""ExcelDataLoader 单元测试：标题行检测、加载、sheet 信息、时间通道识别。"""

from __future__ import annotations

import datetime as dt

import openpyxl
import pandas as pd
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


class TestTimeChannelColumns:
    """时间通道识别回归测试：CSV 转 xlsx 后日期/时间变为原生类型单元格，
    应与 CSV（字符串单元格）行为一致地进入 time_channels_info。"""

    HEADER3 = ["Date", "Time", "RPM"]
    UNITS3 = ["-", "-", "rpm"]

    @staticmethod
    def _load(path, monkeypatch=None):
        if monkeypatch is not None:
            # 强制走 openpyxl fallback 路径（跳过 calamine）
            monkeypatch.setattr(
                ExcelDataLoader, "_read_with_calamine",
                lambda self: (_ for _ in ()).throw(ImportError("测试强制 openpyxl")),
            )
        return ExcelDataLoader(str(path), sheet_name=0, desc_rows=0, has_unit=True)

    @pytest.fixture(params=["calamine", "openpyxl"])
    def typed_time_xlsx(self, tmp_path, request, monkeypatch):
        """生成含原生 date/time 单元格的 xlsx（模拟 Excel 另存 CSV 的默认行为）"""
        f = _write_xlsx(
            tmp_path / "typed.xlsx",
            self.HEADER3, self.UNITS3,
            [
                [dt.date(2025, 1, 2), dt.time(13, 4, 34), 800.0],
                [dt.date(2025, 1, 2), dt.time(13, 4, 35), 810.0],
                [dt.date(2025, 1, 3), dt.time(13, 5, 12), 820.0],
            ],
        )
        return self._load(f, monkeypatch if request.param == "openpyxl" else None)

    def test_native_time_cells_recognized(self, typed_time_xlsx):
        """纯时间单元格 → 登记为 %H:%M:%S 时间通道，数据不丢失（旧代码整列 NaT）"""
        loader = typed_time_xlsx
        assert loader.time_channels_info["Time"] == "%H:%M:%S"
        assert loader.df["Time"].tolist() == ["13:04:34", "13:04:35", "13:05:12"]
        assert loader.df_validity["Time"] == 1

    def test_native_time_cells_monotonic_unix(self, typed_time_xlsx):
        """时间通道按 fmt 解析后应单调递增（复刻下游 startswith 分支语义）"""
        loader = typed_time_xlsx
        fmt = loader.time_channels_info["Time"]
        assert fmt.startswith("%H:%M:%S")
        times = pd.to_datetime(loader.df["Time"], format=fmt, errors="coerce")
        assert times.notna().all()
        assert times.is_monotonic_increasing

    def test_date_only_column_keeps_real_dates(self, typed_time_xlsx):
        """纯日期列 → fmt 不含时间分量（旧代码误入 today 分支导致日期被抹平）"""
        loader = typed_time_xlsx
        fmt = loader.time_channels_info["Date"]
        assert not fmt.startswith("%H:%M:%S")
        ts = pd.to_datetime(loader.df["Date"], format=fmt, errors="coerce")
        assert ts.iloc[0] == pd.Timestamp("2025-01-02 00:00:00")
        assert ts.iloc[2] == pd.Timestamp("2025-01-03 00:00:00")

    def test_time_cells_with_microseconds(self, tmp_path, monkeypatch):
        """带微秒的纯时间单元格 → 列级统一 %H:%M:%S.%f（混合格式会导致匹配失败）"""
        f = _write_xlsx(
            tmp_path / "usec.xlsx",
            ["TMOD", "val"], ["HH:MM:SS.mmm", "-"],
            [
                [dt.time(14, 59, 45, 731000), 1.0],
                [dt.time(14, 59, 45, 831000), 2.0],
            ],
        )
        loader = self._load(f, monkeypatch)
        assert loader.time_channels_info["TMOD"] == "%H:%M:%S.%f"
        assert loader.df["TMOD"].tolist() == ["14:59:45.731000", "14:59:45.831000"]
        # 方案 3-A：时间通道保留在变量列表中（与 CSV 行为对齐）
        assert loader.time_column_name is None
        assert "TMOD" in loader.var_names

    def test_full_datetime_column_fmt(self, tmp_path):
        """完整 datetime 单元格 → fmt 为日期+时间（下游走日期分支，保留真实时刻）"""
        f = _write_xlsx(
            tmp_path / "dt.xlsx",
            ["timestamp", "val"], ["-", "-"],
            [
                [dt.datetime(2025, 1, 2, 13, 4, 34), 1.0],
                [dt.datetime(2025, 1, 2, 13, 4, 35), 2.0],
            ],
        )
        loader = ExcelDataLoader(str(f), sheet_name=0, desc_rows=0, has_unit=True)
        fmt = loader.time_channels_info["timestamp"]
        assert fmt == "%Y-%m-%d %H:%M:%S"
        assert not fmt.startswith("%H:%M:%S")  # 不误入 today+time-of-day 分支
        ts = pd.to_datetime(loader.df["timestamp"], format=fmt, errors="coerce")
        assert ts.iloc[0] == pd.Timestamp("2025-01-02 13:04:34")

    def test_time_channels_stay_in_var_names(self, typed_time_xlsx):
        """方案 3-A：时间通道始终保留在 var_names 中（与 CSV 行为对齐）"""
        loader = typed_time_xlsx
        # Date 和 Time 都是时间通道，但都应保留在变量列表中
        assert "Date" in loader.var_names
        assert "Time" in loader.var_names
        assert "RPM" in loader.var_names
        # X 轴标签回归 "Index"（行号基准）
        assert loader.time_axis_label == "Index"
        # time_column_name 不再被设置
        assert loader.time_column_name is None
        # df_validity 包含所有列
        assert "Date" in loader.df_validity
        assert "Time" in loader.df_validity

    def test_mdf_behavior_unchanged(self):
        """方案 3-A 仅影响 ExcelDataLoader，MDF 行为不变"""
        # MDFLazyLoader 的 var_names 独立实现，不剔除时间通道
        # 此用例仅作为文档性验证，实际 MDF 测试在 test_mdf_lazy_loader.py
        from src.data.mdf_lazy_loader import MDFLazyLoader
        # 确认 MDFLazyLoader 有自己的 var_names 实现
        assert hasattr(MDFLazyLoader, 'var_names')
        # 确认不是继承自 BaseDataLoader 的剔除逻辑
        import inspect
        source = inspect.getsource(MDFLazyLoader.var_names.fget)
        assert 'time_column_name' not in source  # 不引用 time_column_name


class TestObjectColumnNumericFallback:
    """object 列的数值兜底转换不得销毁文本列（V6.0 P0-6）。

    旧实现对所有非时间 object 列无条件 `to_numeric(errors="coerce")`，
    状态/枚举/备注列整列变 NaN 且无日志，与 CSV 路径（保留 category）不一致。
    兜底只为救「数字被存成文本」的列，故按可解析比例判定。
    """

    HEADER = ["time", "speed", "status", "note"]
    UNITS = ["s", "km/h", "-", "-"]
    ROWS = [
        [0.0, "10.5", "OK", "first"],
        [0.1, "11.0", "FAIL", "second"],
        [0.2, "11.5", "OK", "third"],
    ]

    def _assert_kept_text_and_rescued_numbers(self, loader):
        assert loader.df["status"].tolist() == ["OK", "FAIL", "OK"]
        assert loader.df["note"].tolist() == ["first", "second", "third"]
        # 数字被存成文本的列仍被兜底救回
        assert pd.api.types.is_numeric_dtype(loader.df["speed"])
        assert loader.df["speed"].tolist() == pytest.approx([10.5, 11.0, 11.5])

    def test_calamine_path(self, tmp_path):
        path = _write_xlsx(
            tmp_path / "text_cols.xlsx", self.HEADER, self.UNITS, self.ROWS
        )
        loader = ExcelDataLoader(
            str(path), sheet_name=0, desc_rows=0, has_unit=True
        )
        self._assert_kept_text_and_rescued_numbers(loader)

    def test_openpyxl_fallback_path(self, tmp_path, monkeypatch):
        path = _write_xlsx(
            tmp_path / "text_cols2.xlsx", self.HEADER, self.UNITS, self.ROWS
        )

        def boom(*args, **kwargs):
            raise ImportError("calamine 不可用")

        monkeypatch.setattr(ExcelDataLoader, "_read_with_calamine", boom)
        loader = ExcelDataLoader(
            str(path), sheet_name=0, desc_rows=0, has_unit=True
        )
        self._assert_kept_text_and_rescued_numbers(loader)

    def test_mixed_object_column_text_is_preserved(self, tmp_path):
        """object dtype 列（数值与文本混排）不得被无条件 to_numeric 销毁。

        与 dtype 判断直接挂钩，pandas 2（文本列 → object）与 pandas 3
        （文本列 → str）下均为回归红线。
        """
        mixed = [1, 2, 3, 4, 5, 6, 7, "OK", "FAIL", "WARN"]
        rows = [[float(i), "10.5", val, "x"] for i, val in enumerate(mixed)]
        path = _write_xlsx(tmp_path / "mixed.xlsx", self.HEADER, self.UNITS, rows)

        loader = ExcelDataLoader(
            str(path), sheet_name=0, desc_rows=0, has_unit=True
        )

        assert loader.df["status"].dtype == object
        assert loader.df["status"].tolist() == mixed
        assert loader.df["status"].isna().sum() == 0

    def test_below_threshold_column_is_not_converted(self, tmp_path):
        """可解析比例低于阈值的混合列保留原文，不得产出半 NaN 结果"""
        mixed = ["1", "2", "3", "4", "5", "6", "7", "OK", "FAIL", "WARN"]
        rows = [[float(i), "10.5", status, "x"] for i, status in enumerate(mixed)]
        path = _write_xlsx(tmp_path / "mixed.xlsx", self.HEADER, self.UNITS, rows)

        loader = ExcelDataLoader(
            str(path), sheet_name=0, desc_rows=0, has_unit=True
        )

        assert loader.df["status"].tolist() == mixed
        assert not pd.api.types.is_numeric_dtype(loader.df["status"])


def _break_dimension(path, ref="A1"):
    """改写 sheet1.xml 的 <dimension>，模拟只写 A1（或压根不写）的导出工具"""
    import re
    import zipfile

    with zipfile.ZipFile(path) as zf:
        entries = [(i.filename, zf.read(i.filename)) for i in zf.infolist()]

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, payload in entries:
            if name.startswith("xl/worksheets/"):
                xml = payload.decode("utf-8")
                if re.search(r"<dimension\b", xml):
                    xml = re.sub(r"<dimension\b[^>]*/>", f'<dimension ref="{ref}"/>', xml)
                else:
                    xml = xml.replace(
                        "<sheetData>", f'<dimension ref="{ref}"/><sheetData>', 1
                    )
                payload = xml.encode("utf-8")
            zf.writestr(name, payload)
    return path


class TestBrokenSheetDimension:
    """dimension 声明失真时，openpyxl 回退路径不得静默读成空表"""

    @staticmethod
    def _load_openpyxl(path, monkeypatch):
        monkeypatch.setattr(
            ExcelDataLoader, "_read_with_calamine",
            lambda self: (_ for _ in ()).throw(ImportError("测试强制 openpyxl")),
        )
        return ExcelDataLoader(str(path), sheet_name=0, desc_rows=0, has_unit=True)

    def test_a1_dimension_keeps_all_columns_and_rows(self, tmp_path, monkeypatch):
        path = _write_xlsx(tmp_path / "dim.xlsx", HEADER, UNITS, ROWS)
        _break_dimension(path, "A1")

        loader = self._load_openpyxl(path, monkeypatch)

        assert loader.var_names == HEADER
        assert loader.datalength == len(ROWS)
        assert loader.df["rpm"].tolist() == [800, 810, 820]

    def test_undersized_dimension_warns_instead_of_returning_empty(
        self, tmp_path, monkeypatch, caplog
    ):
        path = _write_xlsx(tmp_path / "empty.xlsx", HEADER, UNITS, [])
        _break_dimension(path, "A1")

        with caplog.at_level("WARNING"):
            loader = self._load_openpyxl(path, monkeypatch)

        assert loader.datalength == 0
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "dimension" in messages or "无数据" in messages

    def test_sound_dimension_still_uses_preallocation_path(self, tmp_path, monkeypatch):
        path = _write_xlsx(tmp_path / "ok.xlsx", HEADER, UNITS, ROWS)
        streamed = []
        monkeypatch.setattr(
            ExcelDataLoader, "_read_with_calamine",
            lambda self: (_ for _ in ()).throw(ImportError("测试强制 openpyxl")),
        )
        monkeypatch.setattr(
            ExcelDataLoader, "_read_rows_streaming",
            lambda self, data_start: streamed.append(data_start),
        )

        loader = ExcelDataLoader(str(path), sheet_name=0, desc_rows=0, has_unit=True)

        assert streamed == []
        assert loader.datalength == len(ROWS)
