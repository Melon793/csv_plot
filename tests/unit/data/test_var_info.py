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

    def test_channel_type_label_version_aware(self):
        assert var_info.channel_type_label("3.00", 1) == "MASTER"
        assert var_info.channel_type_label("3.00", 0) == "VALUE"

    def test_missing_table_falls_back_to_raw_code(self):
        """MDF3 无 SYNC_TYPE_TO_STRING 常量表，需回退为带前缀的原始码。

        不能因缺表而抛异常或显示 UNKNOWN —— 缺表是版本差异的正常情况。
        """
        label = var_info.sync_type_label("3.00", 5)
        assert label == "SYNC=5"

    def test_sync_type_label_mdf4(self):
        assert var_info.sync_type_label("4.10", 1) != "SYNC=1"


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
    def test_sections_cover_all_blocks(self, mdf4_loader):
        snap = var_info.build_snapshot(mdf4_loader, "Press_G0", generation=3)
        for title in (
            "基本信息",
            "通道 (CNBLOCK)",
            "通道组 (CGBLOCK)",
            "源信息 (SBLOCK)",
            "转换规则 (CCBLOCK)",
            "时间基准",
            "文件信息 (HDBLOCK)",
        ):
            assert title in snap.sections, f"缺少分组 {title}"

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

    def test_time_base_section_values(self, mdf4_loader):
        snap = var_info.build_snapshot(mdf4_loader, "Press_G0")
        rows = dict(snap.sections["时间基准"])
        assert "10" in rows["有效采样率"]
        assert rows["master 通道名"] == "time"

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
# 快照构建：表格（CSV / Excel）路径
# ---------------------------------------------------------------------------


class TestTabularSnapshot:
    def test_sections_are_tabular_specific(self, csv_loader):
        snap = var_info.build_snapshot(csv_loader, "speed")
        assert snap.source_kind == "csv"
        assert set(snap.sections) == {"基本信息", "列信息", "文件信息"}

    def test_no_mdf_blocks_leak_into_csv_snapshot(self, csv_loader):
        snap = var_info.build_snapshot(csv_loader, "speed")
        for title in snap.sections:
            assert "CNBLOCK" not in title
            assert "CGBLOCK" not in title

    def test_nan_count_reported(self, csv_loader):
        snap = var_info.build_snapshot(csv_loader, "speed")
        rows = dict(snap.sections["列信息"])
        assert rows["NaN 数量"] == "1"

    def test_integer_column_has_zero_nan(self, csv_loader):
        """整数列调 np.isnan 会抛 TypeError，必须跳过。"""
        snap = var_info.build_snapshot(csv_loader, "flag")
        rows = dict(snap.sections["列信息"])
        assert rows["NaN 数量"] == "0"

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

    def test_unknown_column_raises_keyerror(self, csv_loader):
        with pytest.raises(KeyError):
            var_info.build_snapshot(csv_loader, "nope")


# ---------------------------------------------------------------------------
# 全空列的文案（设计文档 §12.2 方案 C 窄化版）
# ---------------------------------------------------------------------------


class TestAllEmptyColumn:
    """整列为空的表格列必须说"没数据"，而不是"类型不对"。

    pandas 把全空列推断为 category dtype（0 个 categories），to_numpy()
    后只剩 object 数组，于是 UI 原本显示"非数值列（object），不适用统计"
    —— 技术上准确，但用户看到的是"我这列是空的"，会以为自己的列类型被
    误判了。

    判定刻意收窄为「category dtype 且 categories 为空」这一 O(1) 检查，
    不用通用的 ``pd.isna(series).all()``：后者在千万行的列上约需 50 ms，
    会破坏"快照构建零磁盘 I/O、约 99 μs、可在 UI 线程同步调用"的前提。
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
        # 关键：object 数组不做 isnan，nan_count 恒为 0，直写会与
        # "全部为空值" 自相矛盾，让用户以为这列一个空值都没有
        assert rows["NaN 数量"] == "2（整列为空）"

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
        assert rows["NaN 数量"] == "0"

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
        assert "整列为空" in md


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
        assert calls, "取消回调从未被调用"

    def test_compute_stats_does_not_write_cache(self, csv_loader):
        """缓存写入属 UI 层职责：子线程不得触碰 main_window 属性。"""
        stats = var_info.compute_stats(csv_loader, "speed")
        assert stats.cached is False
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

    def test_cached_marker_on_key_metrics(self):
        """缓存标注只加在 min/max/mean 上，避免每行重复噪声。"""
        stats = var_info.VarStats(
            min=1.0, max=3.0, mean=2.0, std=0.8, computed=True, cached=True
        )
        rows = dict(var_info.stats_to_rows(stats))
        assert rows["最小值"].endswith("（缓存）")
        assert not rows["标准差"].endswith("（缓存）")


# ---------------------------------------------------------------------------
# Markdown 导出（改进 E）
# ---------------------------------------------------------------------------


class TestMarkdownExport:
    def test_single_snapshot_structure(self, mdf4_loader):
        snap = var_info.build_snapshot(mdf4_loader, "Press_G0")
        md = var_info.snapshot_to_markdown(snap)
        assert md.startswith("## 变量信息：Press_G0")
        assert "### 统计特征" in md
        assert "### 通道 (CNBLOCK)" in md
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
