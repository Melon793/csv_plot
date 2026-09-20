"""unit：文件级信息快照（状态栏左段抽屉的数据源）。

关注三件事：口径按格式分叉（CSV/Excel/MDF 各自能说什么）、缺字段不占行、
以及"哪些信息刻意不放"（有效性计数被撤，必须有用例钉住，防止回潮）。
"""

import pytest

from src.data import file_info
from src.data.file_info import KEY_FILE_NAME, KEY_FOLDER, build_file_rows
from src.data.metadata import CONST, INVALID, UNKNOWN, VALID
from src.data.var_info import ROW_KEY_FILE_PATH


class _FakeLoader:
    """表格型 loader 的最小替身：只给抽屉真正会读的字段。"""

    LOADER_TYPE = "csv"

    def __init__(self, **overrides):
        self.path = "/data/run/a.csv"
        self.file_size = 2048
        self.encoding_used = "utf-8-sig"
        self.sep = ","
        self.has_unit = True
        self.desc_rows = 0
        self.row_count = 120
        self.var_names = ["time", "speed", "flag"]
        self.df_validity = {"time": VALID, "speed": CONST, "flag": INVALID}
        for key, value in overrides.items():
            setattr(self, key, value)


def _as_dict(loader, **kwargs) -> dict:
    return dict(build_file_rows(loader, **kwargs))


def test_csv_rows_cover_path_size_and_format():
    rows = _as_dict(_FakeLoader(), elapsed_s=1.234)

    assert rows[KEY_FILE_NAME] == "a.csv"
    assert rows[ROW_KEY_FILE_PATH].endswith("a.csv")
    assert rows[KEY_FOLDER] == rows[ROW_KEY_FILE_PATH][: -len("/a.csv")]
    assert rows["文件大小"] == "2.0 KB"
    assert rows["格式"] == "CSV 文本（编码：utf-8-sig，分隔符：','）"
    assert rows["数据行数"] == "120 行"
    assert rows["加载耗时"] == "1.23 s"


def test_no_validity_rows_and_no_variable_count():
    """有效性/常量/无效计数与变量数一律不进抽屉：前者对绘图不可执行，
    后者状态栏左段已经常驻。"""
    rows = _as_dict(_FakeLoader())

    for banned in ("有效变量", "常量变量", "无效变量", "数据质量", "变量数", "有效性"):
        assert banned not in rows, f"{banned} 已从抽屉撤掉，不得回潮"
    # 编码/分隔符并进格式行，不再各占一行
    assert "文本编码" not in rows and "分隔符" not in rows


def test_format_notes_skip_defaults():
    """默认值不占字数：单位行=有、说明行=0 时格式行保持短。"""
    default = _as_dict(_FakeLoader())["格式"]
    assert "单位行" not in default and "说明行" not in default

    odd = _as_dict(_FakeLoader(has_unit=False, desc_rows=3))["格式"]
    assert odd == "CSV 文本（编码：utf-8-sig，分隔符：','，无单位行，说明行：3）"


def test_missing_fields_do_not_take_a_row():
    """缺字段直接不给行，不用 "-" 占位稀释可信度。"""
    loader = _FakeLoader(encoding_used="", sep="", file_size=None)
    del loader.desc_rows
    rows = _as_dict(loader)

    assert "文件大小" not in rows  # file_size=None 且路径不存在
    assert _as_dict(_FakeLoader(sep=""))["格式"] == "CSV 文本（编码：utf-8-sig）"


@pytest.mark.parametrize("sep,expect", [("\t", "Tab"), (" ", "空格"), (";", "';'")])
def test_separator_is_written_readably(sep, expect):
    assert expect in _as_dict(_FakeLoader(sep=sep))["格式"]


def test_excel_row_names_the_sheet():
    loader = _FakeLoader(LOADER_TYPE="excel", sheet_name="Run 07")
    del loader.encoding_used
    del loader.sep
    rows = _as_dict(loader)

    assert rows["格式"] == "Excel 工作簿（工作表：Run 07）"


def test_mdf_row_states_its_record_count_semantics():
    """MDF 的行数口径与 CSV 不同，标签必须写清楚，不冒充"N 行"。"""
    loader = _FakeLoader(
        LOADER_TYPE="mdf", path="/data/run/x.mf4", group_count=12, row_count=428_000
    )
    del loader.encoding_used
    del loader.sep
    rows = _as_dict(loader, elapsed_s=0.5)

    assert rows["格式"] == "MDF 二进制（通道组：12）"
    assert rows["记录数"] == "428000 行（最大通道组的声明记录数）"
    assert "数据行数" not in rows


def test_empty_file_reports_zero_rows_rather_than_omitting_the_row():
    assert _as_dict(_FakeLoader(row_count=0))["数据行数"] == "0 行"


def test_elapsed_absent_when_unknown():
    assert "加载耗时" not in _as_dict(_FakeLoader())


def test_no_loader_yields_nothing():
    assert build_file_rows(None) == []


def test_unknown_loader_type_falls_back_to_suffix():
    loader = _FakeLoader(LOADER_TYPE="", path="/data/run/x.dat")
    del loader.encoding_used
    del loader.sep

    assert _as_dict(loader)["格式"] == "DAT 文件"


def test_windows_unc_path_splits_on_backslash():
    r"""Windows 拖进来的 UNC 路径在 macOS 上也要拆对：os.path 只认 "/"。"""
    split = file_info._split_path

    assert split(r"\\host\share\run a.csv") == (r"\\host\share", "run a.csv")
    assert split("/data/run/a.csv") == ("/data/run", "a.csv")
    assert split("a.csv") == ("", "a.csv")


def test_broken_attribute_does_not_break_the_drawer():
    """单个字段读失败只丢那一行，不能让抽屉打不开。"""

    class _Exploding:
        LOADER_TYPE = "csv"
        path = "/data/run/a.csv"
        file_size = 2048
        encoding_used = "ascii"
        sep = ","
        has_unit = True
        var_names = ["time"]
        df_validity = {}

        @property
        def row_count(self):
            raise RuntimeError("loader 已释放")

    rows = _as_dict(_Exploding())
    assert "数据行数" not in rows
    assert rows[KEY_FILE_NAME] == "a.csv"


def test_unknown_validity_values_are_not_reported():
    """df_validity 全未知（MDF 的真实情形）时也不产出任何有效性行。"""
    loader = _FakeLoader(
        df_validity={name: UNKNOWN for name in ["time", "speed", "flag"]}
    )
    rows = _as_dict(loader)
    assert not any("变量" in key or "有效性" in key for key in rows)
