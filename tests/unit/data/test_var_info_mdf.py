"""var_info 的 MDF 侧：版本感知标签、快照合并、归属信息与短读。

由 test_var_info.py 拆分而来（只搬运，未改任何断言）。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from src.data import var_info
from tests.fixtures.data_factory import ENUM_TEXTS

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
