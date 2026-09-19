"""var_info 信息提取层的单元测试（零 Qt 依赖，可脱离 GUI 运行）。

覆盖改进 C（版本感知常量标签）、F（聚合名歧义）、H（统计与缓存分离），
以及 Markdown 导出（改进 E 的数据侧）。
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from src.data import var_info
from src.data.loader import FastDataLoader
from src.data.mdf_lazy_loader import MDFLazyLoader
from src.data.metadata import CONST, INVALID, UNKNOWN, VALID
from tests.fixtures.data_factory import ENUM_TEXTS, write_csv, write_mdf


@pytest.fixture(scope="module")
def mdf4_loader(tmp_path_factory):
    path = write_mdf(
        tmp_path_factory.mktemp("vi4") / "syn4.dat",
        version="4.10",
        n=12,
        with_empty_group=True,
    )
    loader = MDFLazyLoader(str(path))
    yield loader
    loader.close()


@pytest.fixture(scope="module")
def mdf3_loader(tmp_path_factory):
    path = write_mdf(
        tmp_path_factory.mktemp("vi3") / "syn3.dat", version="3.30", n=12
    )
    loader = MDFLazyLoader(str(path))
    yield loader
    loader.close()


@pytest.fixture(scope="module")
def attr4_loader(tmp_path_factory):
    """带归属信息的 MDF4（SI 源块 + 报文组 + 层级显示名 + HD 试验注释）。"""
    path = write_mdf(
        tmp_path_factory.mktemp("attr4") / "syn4.dat",
        version="4.10",
        n=12,
        with_attribution=True,
    )
    loader = MDFLazyLoader(str(path))
    yield loader
    loader.close()


@pytest.fixture(scope="module")
def attr3_loader(tmp_path_factory):
    """带归属信息的 MDF3：无 SI 块，设备只能从通道名 '\\XCP:1' 后缀推断。"""
    path = write_mdf(
        tmp_path_factory.mktemp("attr3") / "syn3.dat",
        version="3.30",
        n=12,
        with_attribution=True,
    )
    loader = MDFLazyLoader(str(path))
    yield loader
    loader.close()


@pytest.fixture()
def csv_loader(tmp_path):
    path = write_csv(
        tmp_path / "s.csv",
        header=["time", "speed", "flag", "note"],
        units=["s", "km/h", "-", "-"],
        rows=[
            ["0.0", "10.0", "1", "a"],
            ["0.1", "nan", "1", "b"],
            ["0.2", "30.0", "1", "c"],
        ],
    )
    return FastDataLoader(str(path), has_unit=True, sep=",")


# ---------------------------------------------------------------------------
# 改进 C：常量标签必须分版本取表
# ---------------------------------------------------------------------------


class TestConstantLabels:
    def test_conversion_label_differs_by_version(self):
        """ct=7 在 MDF4 是 TABX、在 MDF3 是 EXPO —— 同一数值语义相反。"""
        assert var_info.conversion_label("4.10", 7) == "TABX"
        assert var_info.conversion_label("3.00", 7) == "EXPO"

    def test_conversion_label_ct9(self):
        assert var_info.conversion_label("4.10", 9) == "TTAB"
        assert var_info.conversion_label("3.00", 9) == "RAT"

    def test_mdf3_none_conversion_labelled(self):
        """MDF3 用 65535 表示"无转换"，常量表里有对应项而非 UNKNOWN。"""
        assert var_info.conversion_label("3.00", 65535) == "NONE"

    def test_unknown_value_reports_code(self):
        assert var_info.conversion_label("4.10", 999) == "UNKNOWN(999)"

    def test_none_value_renders_dash(self):
        assert var_info.conversion_label("4.10", None) == "-"

    def test_channel_type_table_is_version_aware(self):
        """四个 label 包装函数已随 block 合并移除，改测底层取表机制。

        版本感知本身仍必须覆盖：CHANNEL_TYPE_TO_STRING 两版语义不同。
        """
        table = "CHANNEL_TYPE_TO_STRING"
        assert var_info._constant_label(table, "3.00", 1, "TYPE=") == "MASTER"
        assert var_info._constant_label(table, "3.00", 0, "TYPE=") == "VALUE"

    def test_missing_table_falls_back_to_raw_code(self):
        """MDF3 无 SYNC_TYPE_TO_STRING 常量表，需回退为带前缀的原始码。

        不能因缺表而抛异常或显示 UNKNOWN —— 缺表是版本差异的正常情况。
        这一分支 conversion_label 走不到（两版都有
        CONVERSION_TYPE_TO_STRING），因此刻意用 SYNC_TYPE_TO_STRING 覆盖。
        """
        label = var_info._constant_label("SYNC_TYPE_TO_STRING", "3.00", 5, "SYNC=")
        assert label == "SYNC=5"

    def test_sync_type_table_exists_in_mdf4(self):
        """同一张表在 MDF4 存在，因此不得回退到原始码。"""
        label = var_info._constant_label("SYNC_TYPE_TO_STRING", "4.10", 1, "SYNC=")
        assert label != "SYNC=1"


# ---------------------------------------------------------------------------
# 格式化辅助
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_validity_labels_cover_all_states(self):
        assert var_info.validity_label(VALID) == "有效"
        assert var_info.validity_label(CONST) == "常量"
        assert var_info.validity_label(INVALID) == "无效"
        assert var_info.validity_label(UNKNOWN) == "未知"

    def test_validity_label_unknown_code(self):
        assert "42" in var_info.validity_label(42)

    @pytest.mark.parametrize(
        "num,expected",
        [(0, "0 B"), (512, "512 B"), (2048, "2.0 KB"), (5 * 1024**2, "5.0 MB")],
    )
    def test_format_size(self, num, expected):
        assert var_info.format_size(num) == expected

    def test_format_size_rejects_garbage(self):
        assert var_info.format_size(None) == "-"
        assert var_info.format_size("abc") == "-"

    def test_fmt_decodes_bytes_and_strips_nul(self):
        assert var_info._fmt(b"abc\x00") == "abc"

    def test_fmt_empty_string_becomes_dash(self):
        assert var_info._fmt("   ") == "-"
        assert var_info._fmt(None) == "-"

    def test_fmt_bool_localised(self):
        assert var_info._fmt(True) == "是"
        assert var_info._fmt(False) == "否"

    def test_fmt_nan_becomes_dash(self):
        assert var_info._fmt(float("nan")) == "-"

    def test_fmt_suffix_applied_to_numbers_only(self):
        assert var_info._fmt(12, " bit") == "12 bit"

    def test_fmt_rate_annotates_estimation_basis(self):
        """采样率必须注明是推算值：变速/事件型采样下它会失真。"""
        label = var_info._fmt_rate(100.0)
        assert "100" in label and "平均" in label

    def test_fmt_rate_none(self):
        assert var_info._fmt_rate(None) == "-"

    def test_fmt_raster_shows_nominal_frequency(self):
        assert "1000" in var_info._fmt_raster(0.001)

    def test_fmt_raster_zero_not_divided(self):
        """raster=0 时不得做 1/0 除法。"""
        assert var_info._fmt_raster(0.0) == "0 s"


# ---------------------------------------------------------------------------
# 快照构建：MDF 路径
# ---------------------------------------------------------------------------


class TestMdfSnapshot:
    def test_sections_are_merged(self, mdf4_loader):
        """五个 MDF 块合并为单个「基本信息」（合并方案 2-B）。

        转换规则 / 文件信息 / 枚举映射刻意保持独立块不动。

        归属信息块对 MDF 路径恒存在（合成组的 acq_name / comment 就能填出
        「测量组 / 报文」行），因此列入固定集合；它只在 MDF 路径出现，
        CSV 路径由 test_no_mdf_blocks_leak_into_csv_snapshot 守住。
        """
        snap = var_info.build_snapshot(mdf4_loader, "Press_G0", generation=3)
        assert set(snap.sections) == {
            "基本信息",
            "归属信息",
            "转换规则 (CCBLOCK)",
            "文件信息 (HDBLOCK)",
        }

    def test_attribution_section_follows_basic_section(self, mdf4_loader):
        """渲染顺序即 dict 插入顺序：归属块紧跟基本信息。"""
        snap = var_info.build_snapshot(mdf4_loader, "Press_G0")
        assert list(snap.sections)[:2] == ["基本信息", "归属信息"]

    def test_mdf3_snapshot_also_carries_attribution(self, mdf3_loader):
        snap = var_info.build_snapshot(mdf3_loader, "Press_G0")
        assert "归属信息" in snap.sections

    def test_merged_blocks_no_longer_exist(self, mdf4_loader):
        """被合并的四个块名不得残留，否则 Markdown 导出会出现空块。"""
        snap = var_info.build_snapshot(mdf4_loader, "Press_G0")
        for gone in (
            "通道 (CNBLOCK)",
            "通道组 (CGBLOCK)",
            "源信息 (SBLOCK)",
            "时间基准",
        ):
            assert gone not in snap.sections

    def test_basic_section_row_order(self, mdf4_loader):
        """行序按用户关注顺序固定，不得回退成按 MDF 块结构排列。

        刻意取 State 而不是 Press_G0：工厂的 with_dup_group 会让 Press 跨组
        重名而被改名为 Press_G0 / Press_G1，多出「原始通道名」与「名称
        改写原因」两个条件行。State 唯一且为数值列，正好得到完整的
        16 行固定序列；条件行另有专测覆盖。
        """
        snap = var_info.build_snapshot(mdf4_loader, "State")
        keys = [k for k, _ in snap.sections["基本信息"]]
        assert keys == [
            "变量名",
            "单位",
            "通道注释",
            "记录 ID",
            "组注释",
            "数据类型",
            "数据点总数",
            "是否枚举",
            "位宽 (bit_count)",
            "精度 (precision)",
            "下限 (lower_limit)",
            "上限 (upper_limit)",
            "标称采样间隔",
            "有效采样率",
            "起始时间戳",
            "结束时间戳",
        ]

    def test_dropped_rows_are_reachable_elsewhere(self, mdf4_loader):
        """合并删掉的两行必须在别处仍可见，否则就是真丢信息。

        有效性：页头摘要与 snapshot_to_markdown 首行；
        转换类型：「转换规则 (CCBLOCK)」块。

        用 State 而不是 Press_G0：后者在合成 fixture 里没有 CCBLOCK
        （只得到“无转换块”占位行），无法证明转换类型仍可见。
        """
        snap = var_info.build_snapshot(mdf4_loader, "State")
        basic = dict(snap.sections["基本信息"])
        assert "有效性" not in basic
        assert "转换类型" not in basic
        assert var_info.validity_label(snap.validity)  # 页头摘要的数据源
        conv = dict(snap.sections["转换规则 (CCBLOCK)"])
        assert "转换类型" in conv

    def test_generation_propagated(self, mdf4_loader):
        snap = var_info.build_snapshot(mdf4_loader, "Press_G0", generation=7)
        assert snap.generation == 7

    def test_source_kind_is_mdf(self, mdf4_loader):
        assert var_info.build_snapshot(mdf4_loader, "Press_G0").source_kind == "mdf"

    def test_snapshot_carries_no_stats(self, mdf4_loader):
        """快照刻意不含统计字段，避免与 var_stats_cache 形成两份真相。"""
        snap = var_info.build_snapshot(mdf4_loader, "Press_G0")
        assert not hasattr(snap, "stats")
        assert not hasattr(snap, "min")

    def test_enum_section_appended_for_enum_channel(self, mdf4_loader):
        snap = var_info.build_snapshot(mdf4_loader, "State")
        assert snap.is_enum is True
        assert "枚举映射 (CCBLOCK)" in snap.sections
        rows = dict(snap.sections["枚举映射 (CCBLOCK)"])
        assert rows == {str(k): v for k, v in ENUM_TEXTS.items()}

    def test_enum_section_absent_for_numeric_channel(self, mdf4_loader):
        snap = var_info.build_snapshot(mdf4_loader, "Press_G0")
        assert "枚举映射 (CCBLOCK)" not in snap.sections

    def test_conversion_section_states_no_conversion(self, mdf4_loader):
        """无 CCBLOCK 时必须显式说明，而不是留空让用户猜。"""
        snap = var_info.build_snapshot(mdf4_loader, "Label")
        rows = dict(snap.sections["转换规则 (CCBLOCK)"])
        assert "说明" in rows
        assert "无转换块" in rows["说明"]

    def test_aggregate_renaming_disclosed(self, mdf4_loader):
        """改进 F：跨组重名时同时显示聚合名与原始通道名。

        用户按原始名 "Press" 搜索会找不到变量，必须告知名称已被改写。
        """
        snap = var_info.build_snapshot(mdf4_loader, "Press_G1")
        rows = dict(snap.sections["基本信息"])
        assert snap.original_name == "Press"
        assert rows["变量名（聚合显示名）"] == "Press_G1"
        assert rows["原始通道名"] == "Press"
        assert "跨通道组重名" in rows["名称改写原因"]

    def test_unique_name_has_no_renaming_rows(self, mdf4_loader):
        snap = var_info.build_snapshot(mdf4_loader, "State")
        rows = dict(snap.sections["基本信息"])
        assert "变量名" in rows
        assert "原始通道名" not in rows

    def test_string_channel_marked_non_numeric(self, mdf4_loader):
        snap = var_info.build_snapshot(mdf4_loader, "Label")
        assert snap.is_numeric is False
        rows = dict(snap.sections["基本信息"])
        assert "不支持绘图与统计" in rows["说明"]

    def test_empty_group_explains_zero_samples(self, mdf4_loader):
        """0 点必须解释成因，否则用户会以为是软件读取失败。"""
        snap = var_info.build_snapshot(mdf4_loader, "EmptyCh")
        assert snap.length == 0
        rows = dict(snap.sections["基本信息"])
        assert rows["数据点总数"] == "0"
        assert "预留组" in rows["数据点总数说明"]

    def test_sampling_rows_survive_merge(self, mdf4_loader):
        """合并方案 2-B 的核心约束：采样两行必须留下。

        它们是全软件唯一展示点（grep 过 src/ 全部 .py），而实测真实
        文件存在标称 0.001 s（1000 Hz）但有效仅 17.68 Hz 的严重偏差——
        正是 VarMetadata.sampling_rate_hz 语义修复要在 UI 上体现的场景。
        删掉这两行等于让那次修复彻底不可见。
        """
        snap = var_info.build_snapshot(mdf4_loader, "Press_G0")
        rows = dict(snap.sections["基本信息"])
        assert "10" in rows["有效采样率"]
        # 标称采样间隔在合成 fixture 里恒为 "-"：MDF4 的 raster 藏在 CN
        # comment 的 <raster> 标签内，而工厂没写 comment。所以这里断言的
        # 是“行必须存在”，真实文件的取值证据记在 _from_mdf 的注释里。
        assert "标称采样间隔" in rows
        assert rows["起始时间戳"].endswith("s")
        assert rows["结束时间戳"].endswith("s")

    def test_master_channel_name_row_dropped(self, mdf4_loader):
        """master 通道名实测恒为 "time"，合并时作为冗余行删除。"""
        snap = var_info.build_snapshot(mdf4_loader, "Press_G0")
        basic = dict(snap.sections["基本信息"])
        assert "master 通道名" not in basic

    def test_unknown_variable_raises_keyerror(self, mdf4_loader):
        with pytest.raises(KeyError):
            var_info.build_snapshot(mdf4_loader, "NoSuchChannel__")

    def test_zero_disk_io_for_metadata(self, mdf4_loader):
        """快照构建须足够快以支持 UI 线程同步调用（实测约 99 μs）。

        阈值放宽到 5 ms 以吸收 CI 抖动：仍比统计（18.8 ms 量级）低一个
        数量级，足以证明没有触发磁盘读取。
        """
        import time

        var_info.build_snapshot(mdf4_loader, "Press_G0")  # 预热
        t0 = time.perf_counter()
        for _ in range(20):
            var_info.build_snapshot(mdf4_loader, "Press_G0")
        per_call_ms = (time.perf_counter() - t0) / 20 * 1000
        assert per_call_ms < 5.0, f"单次快照 {per_call_ms:.3f} ms 超出预期"


# ---------------------------------------------------------------------------
# 归属信息与注释解析（对标 ETAS MDA 的 variable 所属 function）
# ---------------------------------------------------------------------------


class TestMdfAttributionSection:
    def test_mdf4_rows_from_source_information(self, attr4_loader):
        snap = var_info.build_snapshot(attr4_loader, "BMS_CellVolt082")
        rows = dict(snap.sections["归属信息"])
        assert rows["设备 / 总线"] == "CAN-Monitoring:1"
        assert rows["ECU / 源节点"] == "BMCe"
        assert rows["总线类型"] == "CAN"
        assert rows["采集源类型"] == "ECU"
        assert rows["测量组 / 报文"] == "BMS_CellVoltInfo"
        assert rows["来源数据库"] == "CANDB: BMS_CellVolt082"
        assert rows["层级显示名"] == "CAN-Monitoring:1.BMS_CellVolt082"
        # 中文注释不是标识符样式，绝不能被当成函数
        assert "所属函数（推断）" not in rows

    def test_mdf3_function_inferred_from_comment(self, attr3_loader):
        snap = var_info.build_snapshot(attr3_loader, "WaterTemp\\XCP:1")
        rows = dict(snap.sections["归属信息"])
        assert rows["设备 / 总线"] == "XCP:1"
        assert rows["所属函数（推断）"] == "RBArithmeticElement"
        assert rows["函数推断依据"] == "通道注释字段(推断)"
        # 注释推断是唯一需要向用户声明“非 A2L 真值”的路径
        assert "非 A2L 定义" in rows["提示"]
        # v3 无 SI 块，而 asammdf 注入的噪声源名必须被过滤
        assert "ECU / 源节点" not in rows

    def test_mdf3_function_from_channel_name_hierarchy(self, attr3_loader):
        """'Fkt/Var\\Device' 形态优先于注释推断，且 '#Index' 不产生任何行。"""
        snap = var_info.build_snapshot(attr3_loader, "EpmCaS_phiSegOfs_CA/isx\\XCP:1")
        rows = dict(snap.sections["归属信息"])
        assert rows["所属函数（推断）"] == "EpmCaS_phiSegOfs_CA"
        assert rows["函数推断依据"] == "通道名层级结构"
        assert "提示" not in rows
        assert "#Index" not in " ".join(rows.values())

    def test_channel_comment_is_parsed_not_raw_xml(self, attr4_loader):
        """缺陷 1 的锁：通道注释不得直出整段 <CNcomment> XML。"""
        snap = var_info.build_snapshot(attr4_loader, "BMS_CellVolt082")
        comment = dict(snap.sections["基本信息"])["通道注释"]
        for token in ("<CNcomment", "<TX>", "</TX>", "<raster>"):
            assert token not in comment, f"通道注释仍含 {token}: {comment}"
        assert comment.startswith("82号单体电压")

    def test_channel_comment_falls_back_to_description_for_mdf3(self, attr3_loader):
        """实测合成 v3 的 Signal(comment=) 落到 CN.description、comment 为空。"""
        snap = var_info.build_snapshot(attr3_loader, "WaterTemp\\XCP:1")
        assert dict(snap.sections["基本信息"])["通道注释"] == "RBArithmeticElement"

    def test_channel_comment_is_single_line(self, attr4_loader):
        """行高固定，多行注释必须拼接成单行而不是嵌换行符。"""
        snap = var_info.build_snapshot(attr4_loader, "BMS_CellVolt082")
        assert "\n" not in dict(snap.sections["基本信息"])["通道注释"]

    @staticmethod
    def file_rows(loader, var_name="Press_G0") -> dict[str, str]:
        snap = var_info.build_snapshot(loader, var_name)
        return dict(snap.sections["文件信息 (HDBLOCK)"])

    def test_header_start_time_is_not_bound_method(self, attr4_loader, attr3_loader):
        """缺陷 3 的锁：start_time_string 在 v3/v4 都是方法而非属性。"""
        for loader in (attr4_loader, attr3_loader):
            value = self.file_rows(loader)["起始时间"]
            assert "bound method" not in value, value

    def test_file_comment_is_parsed_text(self, attr4_loader, attr3_loader):
        """缺陷 2 同类：文件注释不得直出 <HDcomment> XML（v3 还会二次转义）。"""
        for loader in (attr4_loader, attr3_loader):
            text = self.file_rows(loader)["文件注释"]
            assert "<HDcomment" not in text and "&lt;" not in text, text
            assert text.startswith("Database: SYN_DB")

    def test_trial_level_attribution_rows_in_file_section(self, attr4_loader):
        """HD 注释里的试验级归属行拆成独立行。

        WP/RP 的中文标签必须是「工作页 / 参考页」（working / reference page）：
        实测 5 个真实文件两者成对出现、且 WP 基本是 RP 基名加改动后缀；
        “写保护参数集”是错的（那是 CANape 的另一个功能）。
        """
        rows = self.file_rows(attr4_loader)
        assert rows["数据库（Database）"] == "SYN_DB"
        assert rows["试验（Experiment）"] == "SYN_EXP"
        assert rows["工作空间（Workspace）"] == "SYN_WS"
        assert rows["设备清单（Devices）"] == "XCP:1,CAN-Monitoring:1,CalcDev"
        assert rows["工作页（WP）"] == "SYN_WP"
        assert rows["参考页（RP）"] == "SYN_RP"
        # Date / Time 不在展示集合里（已有「起始时间」行）
        assert not any(key.startswith("日期") for key in rows)

    def test_mdf3_file_section_parses_double_escaped_header(self, attr3_loader):
        """实测 asammdf 写 v3 header 注释时会套两层，两层都必须能解析。"""
        rows = self.file_rows(attr3_loader)
        assert rows["数据库（Database）"] == "SYN_DB"
        assert rows["设备清单（Devices）"] == "XCP:1,CAN-Monitoring:1,CalcDev"

    def test_disabled_attribution_keeps_comment_parsing(self, attr4_loader, monkeypatch):
        """开关只关掉新块，不能把缺陷修复一并回退。"""
        from src.data import mdf_attribution

        monkeypatch.setattr(mdf_attribution, "MDF_ATTRIBUTION_ENABLED", False)
        snap = var_info.build_snapshot(attr4_loader, "BMS_CellVolt082")
        assert "归属信息" not in snap.sections
        assert "<CNcomment" not in dict(snap.sections["基本信息"])["通道注释"]


# ---------------------------------------------------------------------------
# 快照构建：表格（CSV / Excel）路径
# ---------------------------------------------------------------------------


class TestTabularSnapshot:
    def test_sections_are_tabular_specific(self, csv_loader):
        snap = var_info.build_snapshot(csv_loader, "speed")
        assert snap.source_kind == "csv"
        assert set(snap.sections) == {"基本信息", "列信息", "文件信息"}

    def test_no_mdf_blocks_leak_into_csv_snapshot(self, csv_loader):
        snap = var_info.build_snapshot(csv_loader, "speed")
        assert "归属信息" not in snap.sections  # 零回归：表格路径不受影响
        for title in snap.sections:
            assert "CNBLOCK" not in title
            assert "CGBLOCK" not in title

    def test_nan_count_comes_from_stats_not_snapshot(self, csv_loader):
        """NaN 计数是统计量而非元数据：快照不得提供，统计必须提供。

        旧实现在 UI 线程对整列做 np.isnan 扫描（O(N)），并带来三处不一致：
        与统计层的「NaN 数」在同一棵树里并存两个标签、MDF 快照从不提供该
        值、非浮点列恒报 0。移除后 NaN 数与 min/max/mean 一样在统计回填
        后才出现，用户心智模型统一。
        """
        rows = dict(var_info.build_snapshot(csv_loader, "speed").sections["列信息"])
        assert "NaN 数量" not in rows
        stats = var_info.compute_stats(csv_loader, "speed")
        assert stats.nan_count == 1
        assert dict(var_info.stats_to_rows(stats))["NaN 数"] == "1"

    def test_integer_column_stats_have_zero_nan(self, csv_loader):
        """整数列没有 NaN，统计层须报 0 而不是抛 TypeError。

        快照侧不再做 isnan，整数列的判定只剩 dtype.kind —— 它必须仍被认作
        数值列，否则连统计任务都不会被提交。
        """
        snap = var_info.build_snapshot(csv_loader, "flag")
        assert snap.is_numeric is True
        assert "NaN 数量" not in dict(snap.sections["列信息"])
        stats = var_info.compute_stats(csv_loader, "flag")
        assert stats.computed is True
        assert stats.nan_count == 0
        assert stats.inf_count == 0

    def test_time_column_flagged(self, csv_loader):
        snap = var_info.build_snapshot(csv_loader, "time")
        rows = dict(snap.sections["列信息"])
        assert rows["是否时间格式列"] in ("是", "否")

    def test_string_column_marked(self, csv_loader):
        snap = var_info.build_snapshot(csv_loader, "note")
        assert snap.is_numeric is False
        rows = dict(snap.sections["列信息"])
        assert "非数值列" in rows["说明"]

    def test_file_info_present(self, csv_loader):
        snap = var_info.build_snapshot(csv_loader, "speed")
        rows = dict(snap.sections["文件信息"])
        assert rows["文件路径"].endswith("s.csv")
        assert rows["行数"] == "3"
        assert rows["列数"] == "4"

    def test_relative_loader_path_is_absolutized(self):
        """相对路径进快照必须是绝对写法（用户会直接复制这一行）。

        入口 normalize 只覆盖 GUI（对话框 / 拖拽 / 命令行）；脚本与测试直接
        构造 loader 时传的就是 ``data/a.csv`` 这种串，不兜一次就会把在别的
        目录下打不开的相对路径送进剪贴板。
        """
        import os

        rows = dict(var_info._tabular_file_rows(SimpleNamespace(path="data/a.csv")))
        assert os.path.isabs(rows["文件路径"])
        assert rows["文件路径"].endswith(os.path.join("data", "a.csv"))

    def test_missing_path_stays_placeholder(self):
        """无路径的替身 loader 仍得 `-`：display_path 不得把空串变成 cwd。"""
        rows = dict(var_info._tabular_file_rows(SimpleNamespace(path=None)))
        assert rows[var_info.ROW_KEY_FILE_PATH] == "-"

    def test_unknown_column_raises_keyerror(self, csv_loader):
        with pytest.raises(KeyError):
            var_info.build_snapshot(csv_loader, "nope")


# ---------------------------------------------------------------------------
# 表格快照必须是 O(1) 元数据读取
# ---------------------------------------------------------------------------


class _TabularStub:
    """只带 df 的表格 loader 替身。

    ``_from_tabular`` 对 path / units / df_validity / time_channels_info 全部走
    ``getattr(..., None)`` 默认值，因此一个 df 属性就足以驱动它 —— 计时与
    dtype 语义测试无需真的往磁盘写 CSV。
    """

    LOADER_TYPE = "csv"

    def __init__(self, df):
        self.df = df


class TestTabularSnapshotIsO1:
    """快照在 UI 线程同步调用，因此不得触碰列数据本身。

    旧实现调 ``series.to_numpy()``（浮点列再叠一次 ``np.isnan``），两者都是
    O(N)。实测 200 万行 × 50 列：文本 category 列物化 223 ms、float 列
    61 ms —— 与模块 docstring 承诺的「零磁盘 I/O、可在 UI 线程同步调用」
    直接矛盾，用户看到的是批量打开标签页时数百毫秒的界面冻结。
    """

    ROWS = 2_000_000
    # 阈值取实测新值（约 0.7 ms）的 ~30 倍：既留足 CI 抖动余量，又能稳稳
    # 捕获旧实现的 61~223 ms 回归
    BUDGET_MS = 20.0

    @staticmethod
    def _time_snapshots(loader, names) -> float:
        import time

        var_info.build_snapshot(loader, names[0])  # 预热，排除首次导入开销
        t0 = time.perf_counter()
        for name in names:
            var_info.build_snapshot(loader, name)
        return (time.perf_counter() - t0) * 1000

    def test_category_columns_are_not_materialized(self):
        """50 列文本 category：旧实现会物化出 50 份完整 object 数组。

        DataFrame 的 50 列共享同一个 Series 对象以压住内存（实测独占
        100 MB、构造 0.01 s；float64 × 50 列则要 800 MB）。快照只读元数据，
        共享与否不影响被测行为。
        """
        import pandas as pd

        codes = np.tile(np.arange(3, dtype=np.int8), self.ROWS // 3 + 1)[: self.ROWS]
        base = pd.Series(
            pd.Categorical.from_codes(
                codes, categories=pd.Index(["alpha", "beta", "gamma"])
            )
        )
        df = pd.DataFrame({f"v{i}": base for i in range(50)})
        names = list(df.columns)

        elapsed = self._time_snapshots(_TabularStub(df), names)
        assert elapsed < self.BUDGET_MS, (
            f"50 列 category 快照 {elapsed:.1f} ms，超出 O(1) 预算 "
            f"{self.BUDGET_MS} ms（旧实现实测约 223 ms）"
        )

    def test_float_columns_are_not_scanned(self):
        """浮点列的 ``np.isnan`` 全列扫描同样是 O(N)，必须一并去掉。

        刻意用**单列跑 50 次**而非 50 列：float64 × 200 万行 × 50 列要占
        800 MB，而被测的是单次调用成本，重复调用同一列等价。
        """
        import pandas as pd

        df = pd.DataFrame({"v": np.linspace(0.0, 1.0, self.ROWS)})

        elapsed = self._time_snapshots(_TabularStub(df), ["v"] * 50)
        assert elapsed < self.BUDGET_MS, (
            f"50 次 float 快照 {elapsed:.1f} ms，超出 O(1) 预算 "
            f"{self.BUDGET_MS} ms（旧实现实测约 61 ms）"
        )

    def test_all_nan_category_with_float_categories_is_all_empty(self):
        """全空列的 categories dtype 可能是 float64，不得因此判成数值列。

        实测两条路径给出的空 categories dtype 不同：真实 CSV 读回是
        ``Index([], dtype='object')``，而对全 NaN 的 float 列显式
        ``astype('category')`` 得到 ``Index([], dtype='float64')``。后者会让
        旧实现算出 is_numeric=True —— 于是白白提交一个注定返回"全部为
        NaN/Inf"的统计任务，且 all_empty 的专属文案永远出不来。
        """
        import pandas as pd

        series = pd.Series([np.nan, np.nan, np.nan]).astype("category")
        cats = series.cat.categories
        assert len(cats) == 0, "前提已变，请同步本测试"
        assert cats.dtype.kind == "f", "前提已变，请同步本测试"

        snap = var_info.build_snapshot(_TabularStub(pd.DataFrame({"v": series})), "v")
        assert snap.all_empty is True
        assert snap.is_numeric is False
        assert snap.dtype == "object"
        assert "全部为空值" in dict(snap.sections["列信息"])["说明"]

    def test_extended_dtype_degrades_to_object(self):
        """扩展 dtype（StringDtype）不得让快照抛 TypeError。

        实测 pandas 3.0.3 下文本 category 列的 ``categories.dtype`` 是
        ``StringDtype(storage='python')``，``np.dtype()`` 对它直接抛
        TypeError；而 ``Series.to_numpy()`` 对同一列物化出的就是 object
        数组。``_effective_numpy_dtype`` 取同样口径，但只读元数据。
        """
        import pandas as pd

        series = pd.Series(["a", "b"], dtype="category")
        cat_dtype = series.cat.categories.dtype
        try:
            np.dtype(cat_dtype)
        except TypeError:
            pass  # 扩展 dtype：正是本用例要覆盖的场景
        else:
            pytest.skip(f"本 pandas 版本的 {cat_dtype} 可直接转 numpy dtype")

        assert var_info._effective_numpy_dtype(cat_dtype) == np.dtype("O")
        snap = var_info.build_snapshot(_TabularStub(pd.DataFrame({"v": series})), "v")
        assert snap.is_numeric is False
        assert snap.all_empty is False
        assert "非数值列" in dict(snap.sections["列信息"])["说明"]

    def test_is_numeric_semantics_preserved_across_dtypes(self):
        """O(1) 改写必须与旧的 to_numpy() 物化口径逐类一致。

        探针 tmp/probe_tabular_o1_equiv.py 对比了 14 种列类型：除上面两个
        全空列用例外完全一致。这里把一致的部分固化成回归网，防止后续再为
        性能改动时悄悄漂移（漂移的后果是数值列不提交统计、或文本列提交后
        必然报错）。
        """
        import pandas as pd

        cases = [
            ("int64", pd.Series([1, 2, 3]), True, False),
            ("float_with_nan", pd.Series([1.0, np.nan, 3.0]), True, False),
            ("bool", pd.Series([True, False, True]), True, False),
            ("cat_int", pd.Series([1, 2, 3], dtype="category"), True, False),
            ("cat_float", pd.Series([1.0, np.nan, 3.0], dtype="category"), True, False),
            ("cat_text", pd.Series(["a", "b"], dtype="category"), False, False),
            ("object_text", pd.Series(["a", "b"], dtype=object), False, False),
        ]
        for label, series, want_numeric, want_all_empty in cases:
            snap = var_info.build_snapshot(
                _TabularStub(pd.DataFrame({"v": series})), "v"
            )
            assert snap.is_numeric is want_numeric, label
            assert snap.all_empty is want_all_empty, label
            assert snap.length == len(series), label


# ---------------------------------------------------------------------------
# 全空列的文案（设计文档 §12.2 方案 C 窄化版）
# ---------------------------------------------------------------------------


class TestAllEmptyColumn:
    """整列为空的表格列必须说"没数据"，而不是"类型不对"。

    pandas 把全空列推断为 category dtype（0 个 categories），dtype 字符串
    只剩 "category"，于是 UI 原本显示"非数值列（object），不适用统计"
    —— 技术上准确，但用户看到的是"我这列是空的"，会以为自己的列类型被
    误判了。

    判定刻意收窄为「category dtype 且 categories 为空」这一 O(1) 检查，
    不用通用的 ``pd.isna(series).all()``：后者在千万行的列上约需 50 ms，
    会破坏"快照构建零磁盘 I/O 且 O(1)、可在 UI 线程同步调用"的前提。
    """

    @pytest.fixture
    def loader(self, tmp_path):
        path = write_csv(
            tmp_path / "empty.csv",
            header=["time", "v", "note"],
            units=["s", "-", "-"],
            rows=[
                ["0.0", "", "a"],
                ["0.1", "", "b"],
            ],
        )
        return FastDataLoader(str(path), has_unit=True, sep=",")

    def test_all_empty_column_flagged(self, loader):
        snap = var_info.build_snapshot(loader, "v")
        assert snap.all_empty is True
        assert snap.is_numeric is False

    def test_nan_text_variant_also_flagged(self, tmp_path):
        """字段写成 "nan" 与写成空串都必须识别（两种真实 CSV 都遇到过）。"""
        path = write_csv(
            tmp_path / "nantext.csv",
            header=["time", "v"],
            units=["s", "-"],
            rows=[["0.0", "nan"], ["0.1", "nan"]],
        )
        loader = FastDataLoader(str(path), has_unit=True, sep=",")
        assert var_info.build_snapshot(loader, "v").all_empty is True

    def test_all_empty_column_explains_itself(self, loader):
        rows = dict(var_info.build_snapshot(loader, "v").sections["列信息"])
        assert "全部为空值" in rows["说明"]
        assert "非数值列" not in rows["说明"]
        # 关键：行数必须写进文案。快照不再提供 NaN 计数（那属统计层的
        # 「NaN 数」行），若只说"全部为空值"而不给规模，用户无法区分这是
        # 2 行的空列还是 200 万行的空列
        assert "共 2 行" in rows["说明"]
        assert "NaN 数量" not in rows

    def test_text_column_is_not_all_empty(self, loader):
        """对照：本 loader 把**所有**文本列都编码为 category。

        实测 note 列（值 a/b）的 pandas dtype 同样是 category，因此
        "是 category 就当全空" 的粗判会误伤所有文本列；只有
        ``len(categories) == 0`` 能把两者区分开。
        """
        snap = var_info.build_snapshot(loader, "note")
        assert str(loader.df["note"].dtype) == "category", "前提已变，请同步本测试"
        assert snap.all_empty is False
        rows = dict(snap.sections["列信息"])
        assert "非数值列" in rows["说明"]
        assert "NaN 数量" not in rows

    def test_numeric_column_never_flagged(self, loader):
        """数值列即使含 NaN 也不算全空：判定必须先过 is_numeric 短路。"""
        snap = var_info.build_snapshot(loader, "time")
        assert snap.all_empty is False
        assert snap.is_numeric is True

    def test_markdown_carries_explanation(self, loader):
        """导出给评审报告时同样不能只剩一句"非数值列（object）"。"""
        snap = var_info.build_snapshot(loader, "v")
        md = var_info.snapshot_to_markdown(snap, None)
        assert "全部为空值" in md
        assert "共 2 行" in md


class TestRangeTextHint:
    """RTABX 无法展示文本表时，提示必须说“不支持”而不是“提取失败”（§12.1）。

    两者对用户的含义完全不同：“提取失败”暗示解析出了 bug、数据本可
    读到；而 RTABX 是结构性的不支持（区间表无法用 dict[int, str] 表达），
    用户无需也不应该去怀疑自己的文件坏了。
    """

    @staticmethod
    def _rows(version, ct, enum_map=None):
        conv = {
            "conversion_type": ct,
            "unit": "-",
            "name": "range_conv",
            "ref_param_nr": 3,
        }
        meta = SimpleNamespace(is_enum=True, enum_map=enum_map)
        return dict(var_info._conversion_rows(version, conv, meta))

    def test_mdf3_rtabx_says_unsupported(self):
        rows = self._rows("3.00", 12)
        assert "范围文本表" in rows["提示"]
        assert "提取失败" not in rows["提示"], "不得让用户以为是解析 bug"
        assert "原始码值" in rows["提示"], "必须告知当前看到的是码值"

    def test_mdf4_rtabx_says_unsupported(self):
        rows = self._rows("4.10", 8)
        assert "范围文本表" in rows["提示"]
        assert "提取失败" not in rows["提示"]

    def test_tabx_still_says_extraction_failed(self):
        """对照：TABX 属于可提取结构，此时 enum_map 为 None 确实是异常。"""
        rows = self._rows("3.00", 11)
        assert "提取失败" in rows["提示"]
        assert "范围文本表" not in rows["提示"]

    def test_no_hint_when_enum_map_present(self):
        """文本表已成功提取时不得出现任何降级提示。"""
        rows = self._rows("3.00", 12, enum_map={0: "off", 1: "on"})
        assert "提示" not in rows


# ---------------------------------------------------------------------------
# 统计计算
# ---------------------------------------------------------------------------


class TestComputeStats:
    def test_csv_numeric_stats(self, csv_loader):
        stats = var_info.compute_stats(csv_loader, "speed")
        assert stats.computed is True
        assert stats.min == pytest.approx(10.0)
        assert stats.max == pytest.approx(30.0)
        assert stats.mean == pytest.approx(20.0)
        assert stats.nan_count == 1
        assert stats.finite_count == 2

    def test_csv_string_column_rejected(self, csv_loader):
        stats = var_info.compute_stats(csv_loader, "note")
        assert stats.computed is False
        assert "非数值" in stats.error

    def test_csv_unknown_column_reports_error_not_raise(self, csv_loader):
        """后台线程不得让异常逃逸，须转成可展示的 error。"""
        stats = var_info.compute_stats(csv_loader, "nope")
        assert stats.computed is False
        assert stats.error

    def test_mdf_numeric_stats(self, mdf4_loader):
        stats = var_info.compute_stats(mdf4_loader, "Press_G0")
        assert stats.computed is True
        assert stats.min == pytest.approx(1.0, rel=1e-5)
        assert stats.max == pytest.approx(5.0, rel=1e-5)
        assert stats.finite_count == 12

    def test_mdf_mean_matches_numpy(self, mdf4_loader):
        """分块累加的结果必须与一次性 numpy 计算一致。"""
        samples = np.asarray(mdf4_loader.get_samples_chunked("Press_G0", 0, -1))
        stats = var_info.compute_stats(mdf4_loader, "Press_G0")
        assert stats.mean == pytest.approx(float(np.mean(samples)), rel=1e-6)
        assert stats.std == pytest.approx(float(np.std(samples)), rel=1e-4)

    def test_mdf_enum_channel_reports_text_values(self, mdf4_loader):
        """枚举通道物理值是文本标签，不适用数值统计，须引导去看枚举表。

        统计恒定 raw=False：若取码值，min/max/mean 得到的是无意义的枚举码。
        """
        stats = var_info.compute_stats(mdf4_loader, "State")
        assert stats.computed is False
        assert "枚举" in stats.error
        assert "枚举映射" in stats.error

    def test_mdf_string_channel_rejected(self, mdf4_loader):
        stats = var_info.compute_stats(mdf4_loader, "Label")
        assert stats.computed is False
        assert "字符串" in stats.error

    def test_mdf_empty_group_reports_zero_data(self, mdf4_loader):
        stats = var_info.compute_stats(mdf4_loader, "EmptyCh")
        assert stats.computed is False
        assert "无数据" in stats.error

    def test_mdf_unknown_variable_reports_error(self, mdf4_loader):
        stats = var_info.compute_stats(mdf4_loader, "nope")
        assert stats.computed is False
        assert "不存在" in stats.error

    def test_closed_loader_degrades_to_error(self, tmp_path):
        """改进 I 的下游收益：loader 关闭后统计降级为错误而非崩溃。"""
        path = write_mdf(tmp_path / "c.dat", version="4.10", n=8)
        loader = MDFLazyLoader(str(path))
        loader.close()
        stats = var_info.compute_stats(loader, "Press_G0")
        assert stats.computed is False
        assert stats.error

    def test_cancellation_checked_between_chunks(self, mdf4_loader):
        """取消回调在每块之间被调用，实现即时中断。"""
        calls = []

        def should_cancel(name):
            calls.append(name)
            return len(calls) >= 1  # 第一次检查即取消

        stats = var_info.compute_stats(mdf4_loader, "Press_G0", should_cancel)
        assert stats.computed is False
        assert stats.error == "已取消"
        # cancelled 标记供缓存层识别并拒绝该条目（缺-5）
        assert stats.cancelled is True
        assert calls, "取消回调从未被调用"

    def test_compute_stats_does_not_write_cache(self, csv_loader):
        """缓存写入属 UI 层职责：子线程不得触碰 main_window 属性。"""
        stats = var_info.compute_stats(csv_loader, "speed")
        assert stats.from_cache is False
        assert stats.generation == 0

    def test_all_nan_csv_column_is_refused_as_non_numeric(self, tmp_path):
        """整列为空的 CSV 列：pandas 推断为 category dtype，按非数值拒绝。

        实测行为（非猜测）：``time,v\\n0.0,nan\\n0.1,nan`` 读入后 v 列的
        dtype 是 ``category``（0 个 categories），``to_numpy()`` 得到
        kind='O' 的 object 数组。真实数据中也出现过（某 .dat 转出的 CSV
        里 FC_I_RON95 整列为空）。

        因此这里**不会**走到 nan_count 统计：kind='O' 在数值判定之前就被
        拦下。断言当前契约，避免日后误改判定顺序而无人察觉。
        """
        path = write_csv(
            tmp_path / "nan.csv",
            header=["time", "v"],
            units=["s", "-"],
            rows=[["0.0", "nan"], ["0.1", "nan"]],
        )
        loader = FastDataLoader(str(path), has_unit=True, sep=",")
        series = loader.get_series("v")
        assert str(series.dtype) == "category", "pandas 行为已变化，请同步本测试"

        stats = var_info.compute_stats(loader, "v")
        assert stats.computed is False
        assert stats.nan_count == 0
        assert "非数值类型" in stats.error

    def test_stats_from_array_all_nan_float(self):
        """float 列全为 NaN：必须返回可读错误，**不得** computed=True + nan。

        回归防护（实测缺陷）：numpy 2.4.6 的 ``np.nanmin`` 对全 NaN 数组只发
        RuntimeWarning 并返回 nan，不抛 ValueError。旧代码仅靠
        ``except ValueError`` 兵来将挡，于是得到
        ``computed=True, min=nan, max=nan, mean=nan, std=nan``：
        UI 四个统计行全部显示 "nan"，且缓存层把 computed=True 视为有效
        结果而长期复用这条脏数据。

        本测试同时断言不得泄露 RuntimeWarning（否则在将警告升级为错误的
        环境下会炸）。
        """
        a = np.array([np.nan, np.nan], dtype=np.float64)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stats = var_info._stats_from_array(a)

        assert stats.computed is False
        assert stats.nan_count == 2
        assert stats.finite_count == 0
        assert "NaN" in stats.error
        assert stats.min is None and stats.max is None and stats.mean is None
        assert stats.std is None
        assert not [w for w in caught if issubclass(w.category, RuntimeWarning)], (
            "全 NaN 路径不应调用 nanmin/nanmax，否则会产生 RuntimeWarning 噪声"
        )

    def test_stats_from_array_all_nan_float32(self):
        """float32 同样适用：绘图与统计大量使用降精度后的 float32 列。"""
        stats = var_info._stats_from_array(np.array([np.nan] * 5, dtype=np.float32))
        assert stats.computed is False
        assert stats.nan_count == 5
        assert "NaN" in stats.error

    def test_stats_from_array_partial_nan_unaffected(self):
        """部分 NaN 的正常路径不得因新增守卫而改变结果。"""
        stats = var_info._stats_from_array(np.array([1.0, np.nan, 3.0]))
        assert stats.computed is True
        assert stats.error == ""
        assert stats.min == 1.0 and stats.max == 3.0
        assert stats.mean == pytest.approx(2.0)
        assert stats.nan_count == 1
        assert stats.finite_count == 2


class TestInfSemantics:
    """Inf 必须在两条统计路径上得到**相同**处理（设计文档 §12.3）。

    实测缺陷：``_stats_from_array`` 用 ``isnan`` 计数、``_stats_mdf`` 用
    ``~isfinite`` 剔除，导致同一个数组 ``[1, 2, inf, nan]`` 在两条路径上
    给出 4 个不同的数字，且各自都有一个字段名不符实：

    - 表格路径 ``finite_count = size - nan_count = 3``，把 Inf 当成了有限值
    - MDF 路径 ``nan_count = 2``，把 Inf 当成了 NaN
    - 表格路径 ``max=inf, mean=inf, std=nan``（std 因 inf-inf），而
      ``computed=True`` 会让这些脏值进缓存长期复用（与全 NaN 缺陷同源）
    """

    SAMPLE = np.array([1.0, 2.0, np.inf, np.nan], dtype=np.float64)

    class _FakeMdfLoader:
        """只实现 ``_stats_mdf`` 用到的两个方法。

        刻意用 stub 而不是造真 MDF 文件：``write_mdf`` 靠 conversion dict
        自动推断信号内容，无法注入 Inf，为测一个计数语义去扩展合成工厂
        成本过高。而跨块累加逻辑只依赖这两个方法，stub 足以覆盖。
        """

        def __init__(self, values):
            self._values = np.asarray(values, dtype=np.float64)

        def get_metadata(self, var_name):
            class _Meta:
                sample_count = self._values.size
                is_enum = False

            return _Meta()

        def get_samples_chunked(self, var_name, offset, count):
            if count is None or count < 0:
                return self._values[offset:]
            return self._values[offset:offset + count]

    def test_inf_excluded_from_stats_and_counted_separately(self):
        stats = var_info._stats_from_array(self.SAMPLE)
        assert stats.computed is True
        # Inf 不得污染统计量：旧行为是 max=inf, mean=inf, std=nan
        assert stats.max == 2.0
        assert stats.mean == pytest.approx(1.5)
        assert np.isfinite(stats.std)
        assert stats.nan_count == 1, "nan_count 只数 NaN"
        assert stats.inf_count == 1, "Inf 必须单独计数，不得混入 nan_count"
        assert stats.finite_count == 2, "finite_count 不得把 Inf 算成有限值"

    def test_three_counts_partition_the_sample(self):
        """三个计数之和恒等于样本总数：字段名与实际含义不得脱节。"""
        stats = var_info._stats_from_array(self.SAMPLE)
        assert stats.nan_count + stats.inf_count + stats.finite_count == self.SAMPLE.size

    def test_all_inf_rejected_with_readable_error(self):
        """全 Inf 与全 NaN 同样属于“无有效样本”，不得 computed=True。"""
        stats = var_info._stats_from_array(np.array([np.inf, -np.inf]))
        assert stats.computed is False
        assert stats.inf_count == 2
        assert stats.nan_count == 0
        assert "NaN/Inf" in stats.error
        assert stats.min is None and stats.max is None

    def test_integer_column_unaffected_by_new_counts(self):
        """整数列不可能有 NaN/Inf，不得因新增计数而改变结果。"""
        stats = var_info._stats_from_array(np.array([1, 2, 3], dtype=np.int32))
        assert stats.computed is True
        assert (stats.nan_count, stats.inf_count, stats.finite_count) == (0, 0, 3)

    def test_mdf_path_agrees_with_tabular_path(self, monkeypatch):
        """同一数组经两条路径必须得到相同数字（§12.3 的核心诉求）。

        刻意把块大小压到 2，把 4 个样本切成两块且 Inf 与 NaN 分属不同块，
        以验证**跨块累加**仍然正确——这是真实 MDF 路径特有的风险点，
        CSV 路径一次拿到整个数组，测不到。
        """
        monkeypatch.setattr("src.core.config.MDF_STATS_CHUNK_SIZE", 2)
        stats = var_info._stats_mdf(self._FakeMdfLoader(self.SAMPLE), "ch", None)
        ref = var_info._stats_from_array(self.SAMPLE)

        assert stats.computed is True
        assert (stats.nan_count, stats.inf_count, stats.finite_count) == (
            ref.nan_count, ref.inf_count, ref.finite_count
        )
        assert stats.max == ref.max
        assert stats.mean == pytest.approx(ref.mean)
        assert stats.std == pytest.approx(ref.std)


# ---------------------------------------------------------------------------
# MDF 分块统计：声明点数 > 实际可读点数
# ---------------------------------------------------------------------------


class TestMdfShortRead:
    """``CGBLOCK.cycles_nr`` 是文件头声明值，不可全信。

    实测存在采集提前中断的 MDF：声明点数远多于数据块里实际写入的点数，
    asammdf 对越界 offset 返回**空数组**而不是抛异常。旧实现有三处连锁
    问题：

    1. 遇空块静默 ``break`` 后仍报 ``computed=True`` —— 页面「基本信息」显示
       的 sample_count 是声明值，统计却只覆盖了前面的样本，两者自相矛盾
       且都标为有效，用户无从察觉数据缺了九成
    2. ``offset += count`` 按**请求量**推进 —— asammdf 返回不足 count 时会越过
       中间样本，在统计里凭空挖掉一段连续数据（比尾部截断更糟）
    3. 首块就读不到数据时落入 ``n == 0`` 的通用分支，报"全部为 NaN/Inf" ——
       把文件头不可信误报成数据质量问题，用户会去查根本不存在的 NaN
    """

    class _ShortMdfLoader:
        """声明 ``declared`` 点、实际只有 ``actual`` 点可读的 loader 替身。

        样本值取 ``arange``，使 min / max / finite_count 能直接反推出"到底数了
        哪些下标"，从而区分「尾部截断」与「中间挖洞」。同
        ``TestInfSemantics._FakeMdfLoader`` 的理由：跨块累加逻辑只依赖这两个
        方法，而合成 MDF 工厂无法注入"声明点数与实际不符"这种损坏。
        """

        def __init__(self, declared: int, actual: int):
            self.declared = declared
            self.actual = actual
            self.calls = []  # [(offset, count), ...]，供断言推进方式

        def get_metadata(self, var_name):
            return SimpleNamespace(sample_count=self.declared, is_enum=False)

        def get_samples_chunked(self, var_name, offset, count):
            self.calls.append((offset, count))
            if count is None or count < 0:
                count = self.declared - offset
            end = min(offset + count, self.actual)
            if end <= offset:
                return np.array([], dtype=np.float64)
            return np.arange(offset, end, dtype=np.float64)

    @pytest.fixture
    def chunk10(self, monkeypatch):
        """把分块压到 10，让 declared=1000 的读取真的跨越多块。"""
        monkeypatch.setattr("src.core.config.MDF_STATS_CHUNK_SIZE", 10)

    def test_short_read_is_reported_not_silent(self, chunk10):
        loader = self._ShortMdfLoader(declared=1000, actual=100)
        stats = var_info._stats_mdf(loader, "ch", None)

        assert stats.computed is True, "前 100 个样本确实读到了，结果仍有效"
        assert stats.error == ""
        assert stats.finite_count == 100
        assert stats.min == pytest.approx(0.0)
        assert stats.max == pytest.approx(99.0)
        assert "仅统计到前 100 个样本" in stats.note
        assert "1000" in stats.note, "note 必须带上声明值，否则用户不知道差多少"
        assert dict(var_info.stats_to_rows(stats))["说明"] == stats.note

    def test_partial_tail_counts_every_sample(self, chunk10):
        """尾块返回不足请求量时，必须按**实际返回量**推进 offset。

        declared=100 / actual=95 / chunk=10：第 10 块只能返回 5 个样本。旧写法
        ``offset += count`` 会把 offset 推到 100 并当作正常读完，既不报截断、
        也可能在更极端的分块下越过中间样本。
        """
        loader = self._ShortMdfLoader(declared=100, actual=95)
        stats = var_info._stats_mdf(loader, "ch", None)

        assert stats.computed is True
        assert stats.finite_count == 95, "95 个样本一个都不能少"
        assert stats.max == pytest.approx(94.0)
        assert "仅统计到前 95 个样本" in stats.note
        # 读到不足量就该停：再请求下一块只会拿到空数组，白花一次锁竞争
        assert loader.calls[-1][0] == 90

    def test_no_note_when_read_is_complete(self, chunk10):
        """正常文件不得多出「说明」行。

        否则每份健康文件都会带上一句看似警告的提示，用户很快就学会忽略
        它，真正需要它的时候反而失效。
        """
        loader = self._ShortMdfLoader(declared=100, actual=100)
        stats = var_info._stats_mdf(loader, "ch", None)

        assert stats.computed is True
        assert stats.finite_count == 100
        assert stats.max == pytest.approx(99.0)
        assert stats.note == ""
        assert "说明" not in dict(var_info.stats_to_rows(stats))

    def test_first_chunk_empty_reports_unreadable_not_all_nan(self, chunk10):
        loader = self._ShortMdfLoader(declared=100, actual=0)
        stats = var_info._stats_mdf(loader, "ch", None)

        assert stats.computed is False
        assert "无可读样本" in stats.error
        assert "100" in stats.error, "必须带上文件头声明值，便于定位是哪份文件"
        assert "NaN" not in stats.error, "真实原因是读不出样本，不是数据质量问题"


# ---------------------------------------------------------------------------
# 统计结果渲染
# ---------------------------------------------------------------------------


class TestStatsToRows:
    def test_pending_placeholder(self):
        assert var_info.stats_to_rows(None) == [("状态", "计算中…")]

    def test_error_message_surfaced(self):
        stats = var_info.VarStats(error="数据为空")
        assert var_info.stats_to_rows(stats) == [("状态", "数据为空")]

    def test_computed_rows(self):
        stats = var_info.VarStats(
            min=1.0, max=3.0, mean=2.0, std=0.8,
            nan_count=2, finite_count=10, computed=True,
        )
        rows = dict(var_info.stats_to_rows(stats))
        assert rows["最小值"] == "1"
        assert rows["最大值"] == "3"
        assert rows["有效样本数"] == "10"
        assert rows["NaN 数"] == "2"
        # 旧标签 "NaN / 非有限值数" 的模糊措辞恰好掩盖了两条路径的分歧：
        # 表格路径数的是纯 NaN，MDF 路径数的是所有非有限值。计数口径
        # 统一后标签必须说真话，Inf 另起一行。
        assert "Inf 数" not in rows

    def test_inf_row_appears_only_when_present(self):
        """Inf 行仅在出现时显示，不给绝大多数正常数据增加噪声。"""
        rows = dict(var_info.stats_to_rows(var_info.VarStats(
            min=1.0, max=2.0, mean=1.5, std=0.5,
            nan_count=1, inf_count=1, finite_count=2, computed=True,
        )))
        assert rows["Inf 数"] == "1"
        assert rows["NaN 数"] == "1", "Inf 不得再混入 NaN 计数"

    def test_stats_rows_contain_no_cached_marker(self):
        """D′：缓存来源（from_cache）不得写入任何数值行。

        旧实现只给 min/max/mean 加「（缓存）」后缀：一是同源同趟的
        标准差/计数行不一致，二是 snapshot_to_markdown 复用本函数、
        复制进报告会破坏数值字段，三是用户极易读成"数值可能是旧的"。
        """
        stats = var_info.VarStats(
            min=1.0, max=3.0, mean=2.0, std=0.8, computed=True, from_cache=True
        )
        for _, value in var_info.stats_to_rows(stats):
            assert "（缓存）" not in value

    def test_note_rendered_last_and_only_when_present(self):
        """``note`` 是对上面数值**适用范围**的限制说明，必须排在最后。

        放在数值行之前会让用户带着"这结果不完整"的先入为主去读 min/max；
        而 ``note`` 为空时（绝大多数正常数据）不得多出一行噪声。
        """
        rows = var_info.stats_to_rows(var_info.VarStats(
            min=1.0, max=2.0, mean=1.5, std=0.5, finite_count=2,
            computed=True, note="仅统计到前 2 个样本",
        ))
        assert rows[-1] == ("说明", "仅统计到前 2 个样本")
        assert "说明" not in dict(var_info.stats_to_rows(var_info.VarStats(
            min=1.0, max=2.0, mean=1.5, std=0.5, finite_count=2, computed=True,
        )))

    def test_note_does_not_make_result_look_failed(self):
        """``note`` 与 ``error`` 必须分开：带 note 的结果仍是**有效**结果。

        若把截断说明写进 ``error``，``stats_to_rows`` 的 ``not computed`` 分支会
        把整张表压成一行提示，min/max/mean 全部丢失。
        """
        stats = var_info.VarStats(
            min=0.0, max=9.0, mean=4.5, std=2.9, finite_count=10,
            computed=True, note="仅统计到前 10 个样本",
        )
        rows = dict(var_info.stats_to_rows(stats))
        assert "状态" not in rows
        assert rows["最大值"] == "9"
        assert rows["有效样本数"] == "10"


# ---------------------------------------------------------------------------
# Markdown 导出（改进 E）
# ---------------------------------------------------------------------------


class TestMarkdownExport:
    def test_single_snapshot_structure(self, mdf4_loader):
        snap = var_info.build_snapshot(mdf4_loader, "Press_G0")
        md = var_info.snapshot_to_markdown(snap)
        assert md.startswith("## 变量信息：Press_G0")
        assert "### 统计特征" in md
        assert "### 基本信息" in md
        assert "### 通道 (CNBLOCK)" not in md, "已合并的块不得再导出为空块"
        assert "| 指标 | 值 |" in md

    def test_stats_omitted_renders_pending(self, mdf4_loader):
        """统计未回填时导出不得报错，而是标注"计算中…"。"""
        snap = var_info.build_snapshot(mdf4_loader, "Press_G0")
        md = var_info.snapshot_to_markdown(snap, None)
        assert "计算中…" in md

    def test_stats_included_when_available(self, mdf4_loader):
        snap = var_info.build_snapshot(mdf4_loader, "Press_G0")
        stats = var_info.compute_stats(mdf4_loader, "Press_G0")
        md = var_info.snapshot_to_markdown(snap, stats)
        assert "计算中…" not in md
        assert "有效样本数" in md

    def test_original_name_exported_when_renamed(self, mdf4_loader):
        snap = var_info.build_snapshot(mdf4_loader, "Press_G1")
        md = var_info.snapshot_to_markdown(snap)
        assert "- 原始通道名: Press" in md

    def test_original_name_omitted_when_identical(self, mdf4_loader):
        snap = var_info.build_snapshot(mdf4_loader, "State")
        assert "原始通道名" not in var_info.snapshot_to_markdown(snap)

    def test_multiple_snapshots_joined_by_rule(self, mdf4_loader):
        snaps = [
            (var_info.build_snapshot(mdf4_loader, n), None)
            for n in ("Press_G0", "State")
        ]
        md = var_info.snapshots_to_markdown(snaps)
        assert md.count("## 变量信息：") == 2
        assert "\n---\n" in md

    def test_pipe_and_newline_escaped(self):
        """表格单元格里的竖线与换行会破坏 Markdown 结构，必须转义。"""
        assert var_info._md_escape("a|b") == "a\\|b"
        assert var_info._md_escape("a\nb") == "a b"
        assert var_info._md_escape("a\r\nb") == "a b"
        assert var_info._md_escape(None) == "-"

    def test_md_value_wraps_file_path_in_backticks(self):
        r"""「文件路径」包反引号：实测文件名里成串出现 ``_`` 与 ``\``。

        ``EGR_new_20%SOC``、``07-05-2026new ECU_old SW0009`` 这类名字里的 `_`
        会被 Markdown 渲染器当斜体标记吃掉、`\` 在部分渲染器里是转义符 ——
        粘进评审/缺陷单的路径就与真值不一致了。其他键一律不参与包裹。
        """
        win = r"D:\Messung\EGR_new_20%SOC\CMP21_1am=0.8_Map.dat"
        assert var_info._md_value(var_info.ROW_KEY_FILE_PATH, win) == f"`{win}`"
        assert var_info._md_value("变量名", "a|b") == "a\\|b"

    def test_md_value_falls_back_when_path_contains_backtick(self):
        """值自身含反引号时包裹不安全 → 退回普通转义（不包、也不静默删字符）。"""
        out = var_info._md_value(var_info.ROW_KEY_FILE_PATH, "a`b_c")
        assert not out.startswith("`")
        assert out == "a`b_c"

    def test_exported_markdown_path_row_is_a_code_span(self, csv_loader):
        """端到端：导出的表里路径行整格被反引号包住，路径本体逐字保留。"""
        import os

        snap = var_info.build_snapshot(csv_loader, "speed")
        md = var_info.snapshot_to_markdown(snap)
        line = next(ln for ln in md.splitlines() if ln.startswith("| 文件路径 |"))
        expect = os.path.abspath(str(csv_loader.path))
        assert line == f"| 文件路径 | `{expect}` |"

    def test_comment_with_pipe_survives_export(self, mdf4_loader):
        """端到端确认转义生效：导出的每行表格都恰好有 3 个未转义竖线。"""
        snap = var_info.build_snapshot(mdf4_loader, "Press_G0")
        for line in var_info.snapshot_to_markdown(snap).splitlines():
            if line.startswith("|") and not line.startswith("|---"):
                assert line.count("|") - line.count("\\|") >= 3

    def test_enum_limit_respected(self):
        """超大枚举表须截断，否则单页会渲染上万行。"""
        from src.core.config import VAR_INFO_ENUM_DISPLAY_LIMIT

        big = {i: f"label_{i}" for i in range(VAR_INFO_ENUM_DISPLAY_LIMIT + 50)}
        rows = var_info._enum_rows(big)
        assert len(rows) == VAR_INFO_ENUM_DISPLAY_LIMIT + 1
        assert "仅显示前" in rows[-1][1]


class TestCrossChunkVariance:
    """P2-31：跨块方差必须先中心化，再套 E[x²]-E[x]²。

    绝对时间戳通道（1.7e9 量级、真实波动不到 1）下，`E[x²]` 与 `E[x]²` 都是
    1e18 量级，float64 只有 ~16 位有效数字：实测旧实现把 std 抵成 **0**（被
    `max(..., 0.0)` 夹住），带噪声的那条则抵出 **22.6**（真值 0.498）。而单数组
    路径走 `np.nanstd`（内部先减均值）没这问题 —— 同一个变量两条路径结论不同。
    """

    class _SeriesLoader:
        def __init__(self, data):
            self.data = np.asarray(data, dtype=np.float64)

        def get_metadata(self, var_name):
            return SimpleNamespace(sample_count=self.data.size, is_enum=False)

        def get_samples_chunked(self, var_name, offset, count):
            return self.data[offset : offset + count]

    @pytest.fixture(autouse=True)
    def small_chunk(self, monkeypatch):
        """块压到 1000，20000 点要走 20 次跨块累加。"""
        monkeypatch.setattr("src.core.config.MDF_STATS_CHUNK_SIZE", 1000)

    def _std(self, data):
        stats = var_info._stats_mdf(self._SeriesLoader(data), "ch", None)
        assert stats.computed is True
        return stats

    def test_timestamp_channel_keeps_its_real_std(self):
        data = 1.7e9 + np.arange(20000) * 0.001
        stats = self._std(data)

        assert stats.std == pytest.approx(float(np.nanstd(data)), rel=1e-9)
        assert stats.std > 1.0, "旧实现这里是 0：方差被整体抵消掉了"
        assert stats.mean == pytest.approx(float(np.mean(data)), rel=1e-12)

    def test_noisy_timestamp_channel_not_inflated(self):
        data = 1.7e9 + np.random.default_rng(0).standard_normal(20000) * 0.5
        stats = self._std(data)

        assert stats.std == pytest.approx(float(np.nanstd(data)), rel=1e-6)
        assert stats.std < 1.0, f"旧实现给出 22.6（真值 ≈0.498）: {stats.std}"

    def test_chunked_path_agrees_with_single_array_path(self):
        """两条统计路径的精度口径必须一致，否则切不分块会看到数字跳变。"""
        data = 1.7e9 + np.arange(5000) * 0.002

        chunked = self._std(data)
        single = var_info._stats_from_array(data)

        assert chunked.std == pytest.approx(single.std, rel=1e-9)
        assert chunked.mean == pytest.approx(single.mean, rel=1e-12)

    def test_constant_channel_with_large_offset_is_still_exactly_zero(self):
        """中心化不能把别名义值算出来：常量通道 std 仍须严格为 0。"""
        stats = self._std(np.full(20000, 1.7e9))

        assert stats.std == 0.0
        assert stats.mean == pytest.approx(1.7e9)

    def test_ordinary_magnitude_channel_is_unchanged(self):
        data = 800.0 + np.arange(20000) % 97
        stats = self._std(data)

        assert stats.std == pytest.approx(float(np.nanstd(data)), rel=1e-12)

    def test_nan_and_inf_samples_still_excluded(self):
        """参考点取的是首块均值，必须建立在剔除非有限值之后。"""
        data = 1.7e9 + np.arange(3000) * 0.001
        data[7] = np.nan
        data[1500] = np.inf

        stats = self._std(data)

        assert (stats.nan_count, stats.inf_count) == (1, 1)
        clean = np.delete(data, [7, 1500])
        assert stats.std == pytest.approx(float(np.std(clean)), rel=1e-9)
