"""MDFLazyLoader 只读信息接口与并发加固的单元测试。

覆盖本次改进的 loader 侧落点：
- 改进 A：``nominal_raster_s`` / ``effective_rate_hz`` 的语义分离
- 改进 B：``sample_count`` / ``time_min`` / ``time_max`` 零成本回填
- 改进 F：跨组重名的聚合改名与 ``original_name`` 保留
- 改进 G：``is_enum`` 按 MDF 版本判定（合成文件在 MDF3 落地 ct=12、
  MDF4 落地 ct=7，同一逻辑通道跨版本 ct 不同，故可确定性验证）
- 改进 I：``close()`` 幂等、关闭后统一抛 ``KeyError``、``__del__`` 安全
"""

import gc
import threading
import weakref

import numpy as np
import pytest

from src.data.mdf_lazy_loader import MDFLazyLoader
from tests.fixtures.data_factory import ENUM_TEXTS, write_mdf

# 把 __del__ 里的异常从"警告"升级为"失败"。
# close() 在部分构造状态下抛 AttributeError 时，Python 只会把它变成
# "Exception ignored in __del__" 噪声，默认不会让测试失败；本标记使其显式暴露。
pytestmark = pytest.mark.filterwarnings(
    "error::pytest.PytestUnraisableExceptionWarning"
)


@pytest.fixture(scope="module")
def mdf4_path(tmp_path_factory):
    return write_mdf(
        tmp_path_factory.mktemp("mdf4") / "syn4.dat",
        version="4.10",
        n=12,
        with_single_shot_group=True,
        with_empty_group=True,
    )


@pytest.fixture(scope="module")
def mdf3_path(tmp_path_factory):
    return write_mdf(
        tmp_path_factory.mktemp("mdf3") / "syn3.dat",
        version="3.30",
        n=12,
        with_single_shot_group=True,
        with_empty_group=True,
    )


@pytest.fixture()
def loader4(mdf4_path):
    loader = MDFLazyLoader(str(mdf4_path))
    yield loader
    loader.close()


@pytest.fixture()
def loader3(mdf3_path):
    loader = MDFLazyLoader(str(mdf3_path))
    yield loader
    loader.close()


# ---------------------------------------------------------------------------
# 改进 A + B：采样率语义与零成本回填
# ---------------------------------------------------------------------------


class TestTimeBaseMetadata:
    def test_effective_rate_derived_from_master(self, loader4):
        """有效采样率 = (cycles-1)/(t_max-t_min)。

        合成数据 12 点、间隔 0.1 s → t 范围 0~1.1 s → (12-1)/1.1 = 10 Hz。
        """
        meta = loader4.get_metadata("Press_G0")
        assert meta is not None
        assert meta.effective_rate_hz == pytest.approx(10.0, rel=1e-6)

    def test_sample_count_and_time_range_filled(self, loader4):
        meta = loader4.get_metadata("Press_G0")
        assert meta.sample_count == 12
        assert meta.time_min == pytest.approx(0.0)
        assert meta.time_max == pytest.approx(1.1)

    def test_second_group_has_own_time_base(self, loader4):
        """每个组的时间基准独立：第二组 6 点、间隔 0.2 s → 5 Hz。"""
        meta = loader4.get_metadata("Press_G1")
        assert meta.sample_count == 6
        assert meta.time_max == pytest.approx(1.0)
        assert meta.effective_rate_hz == pytest.approx(5.0, rel=1e-6)

    def test_nominal_raster_absent_when_file_has_none(self, loader4):
        """asammdf 合成的文件不写 ``<raster>`` 标签，故标称间隔为 None。

        这里断言 None 而非猜测值：``nominal_raster_s`` 与
        ``effective_rate_hz`` 是两个独立事实，前者来自文件声明、
        后者由数据推算，合成数据只有后者。
        """
        meta = loader4.get_metadata("Press_G0")
        assert meta.nominal_raster_s is None

    def test_empty_group_has_zero_samples(self, loader4):
        """空组（cycles_nr=0 且无数据块）如实报告 0 点而非编造数值。

        实测真实 MDF3 文件中约 12.5% 的通道属于此类预留组。
        """
        meta = loader4.get_metadata("EmptyCh")
        assert meta is not None
        assert meta.sample_count == 0
        assert meta.effective_rate_hz is None


# ---------------------------------------------------------------------------
# 改进 F：聚合改名与原始通道名
# ---------------------------------------------------------------------------


class TestAggregateRenaming:
    def test_duplicate_names_across_groups_get_suffix(self, loader4):
        names = loader4.var_names
        assert "Press_G0" in names
        assert "Press_G1" in names
        assert "Press" not in names

    def test_original_name_preserved(self, loader4):
        """改进 F 的数据基础：用户按原始名搜索时窗口能显示歧义来源。"""
        assert loader4.get_metadata("Press_G0").original_name == "Press"
        assert loader4.get_metadata("Press_G1").original_name == "Press"

    def test_unique_name_keeps_original(self, loader4):
        assert loader4.get_metadata("State").original_name == "State"

    def test_renamed_channels_keep_distinct_units(self, loader4):
        """改名后两个通道的单位仍各自正确，证明索引映射未错位。"""
        assert loader4.get_metadata("Press_G0").unit == "bar"
        assert loader4.get_metadata("Press_G1").unit == "degC"


# ---------------------------------------------------------------------------
# 改进 G：枚举判定的版本感知
# ---------------------------------------------------------------------------


class TestVersionAwareEnum:
    def test_mdf4_enum_channel_detected(self, loader4):
        """MDF4 的文本表转换落地为 ct=7 (TABX)。"""
        info = loader4.get_channel_info("State")
        assert info["version"].startswith("4")
        assert info["conversion"]["conversion_type"] == 7
        assert info["meta"].is_enum is True
        assert info["meta"].enum_map == ENUM_TEXTS

    def test_mdf3_enum_channel_detected(self, loader3):
        """同一逻辑通道在 MDF3 落地为 ct=12 (RTABX)。

        这正是版本感知的必要性：12 在 MDF4 语义里根本不存在，
        而 MDF3 的枚举恰恰用它。改动前的版本无关判定 ``ct in (7,9,10,11)``
        会漏判 MDF3 的 RTABX 枚举。
        """
        info = loader3.get_channel_info("State")
        assert info["version"].startswith("3")
        assert info["conversion"]["conversion_type"] == 12
        assert info["meta"].is_enum is True

    def test_mdf3_numeric_channel_not_enum(self, loader3):
        """MDF3 的 ct=65535 (NONE) 不得判为枚举。"""
        info = loader3.get_channel_info("Press_G0")
        assert info["conversion"]["conversion_type"] == 65535
        assert info["meta"].is_enum is False

    def test_mdf4_numeric_channel_not_enum(self, loader4):
        info = loader4.get_channel_info("Press_G0")
        assert info["meta"].is_enum is False


# ---------------------------------------------------------------------------
# 字符串通道识别（跨版本 dtype_fmt 语义不同）
# ---------------------------------------------------------------------------


class TestStringChannelDetection:
    def test_mdf3_string_channel(self, loader3):
        """MDF3 字符串通道的 dtype_fmt 就是真实的 |S1。"""
        info = loader3.get_channel_info("Label")
        assert info["is_numeric"] is False
        assert info["dtype"] == "|S1"

    def test_mdf4_string_channel_despite_uint64_dtype_fmt(self, loader4):
        """回归防护：MDF4 字符串通道的 dtype_fmt 失真为 uint64。

        MDF4 的可变长字符串在记录里存的是指向 SDblock 的索引，
        asammdf 因此把 dtype_fmt 报为 uint64（data_type=7 STRING_UTF_8），
        实际 get() 之后才是 |S1。仅凭 dtype_fmt 会把它误判为可统计的
        数值通道，进而提交注定失败的统计任务、且不显示"不支持统计"说明。
        """
        info = loader4.get_channel_info("Label")
        assert info["channel"]["data_type"] == 7
        assert info["is_numeric"] is False
        # dtype 字段如实标注失真原因，避免用户看到 uint64 却读到文本
        assert "uint64" in info["dtype"]
        assert "字符串索引" in info["dtype"]

    def test_numeric_channel_dtype_not_annotated(self, loader4):
        assert loader4.get_channel_info("Press_G0")["dtype"] == "float32"


# ---------------------------------------------------------------------------
# get_channel_info 结构完整性
# ---------------------------------------------------------------------------


class TestGetChannelInfo:
    REQUIRED_KEYS = (
        "meta", "version", "dtype", "is_numeric", "channel",
        "channel_group", "source", "conversion", "header",
        "time_base", "file", "enum_map",
    )

    def test_returns_all_sections(self, loader4):
        info = loader4.get_channel_info("Press_G0")
        for key in self.REQUIRED_KEYS:
            assert key in info, f"缺少 {key}"

    def test_unknown_variable_raises_keyerror(self, loader4):
        with pytest.raises(KeyError):
            loader4.get_channel_info("NoSuchChannel__")

    def test_channel_group_carries_cycles(self, loader4):
        """cycles_nr 挂在 CGBLOCK 上而非 Group 对象（曾踩过的坑）。"""
        cg = loader4.get_channel_info("Press_G0")["channel_group"]
        assert cg["cycles_nr"] == 12
        assert cg["channel_count"] == 4  # time + Press + State + Label

    def test_time_base_mirrors_metadata(self, loader4):
        info = loader4.get_channel_info("Press_G0")
        tb = info["time_base"]
        meta = info["meta"]
        assert tb["sample_count"] == meta.sample_count
        assert tb["effective_rate_hz"] == meta.effective_rate_hz
        assert tb["nominal_raster_s"] == meta.nominal_raster_s

    def test_file_section_reports_group_count(self, loader4):
        fi = loader4.get_channel_info("Press_G0")["file"]
        assert fi["path"] == str(loader4._path)
        assert fi["size"] > 0
        assert fi["group_count"] >= 3

    def test_enum_map_present_for_enum_channel(self, loader4):
        info = loader4.get_channel_info("State")
        assert info["enum_map"] == ENUM_TEXTS

    def test_channel_carries_attribution_text_keys(self, loader3, loader4):
        """归属提取依赖的两个字段必须在两版下都存键（值可空）。

        description 是 v3 长文本真身、display_names 是 v4 层级显示名，
        另一版无此属性时 getattr 置 None —— 锁的是**键在**，不是值在。
        """
        for loader in (loader3, loader4):
            ch = loader.get_channel_info("Press_G0")["channel"]
            assert "description" in ch, f"{loader} 的 channel 缺 description"
            assert "display_names" in ch, f"{loader} 的 channel 缺 display_names"


# ---------------------------------------------------------------------------
# _block_attrs：可调用属性守卫与类型归一
# ---------------------------------------------------------------------------


class _FakeBlock:
    """模拟 asammdf 块对象：数据属性 / bytes / 方法 / 抛异常的方法混在一起。"""

    def __init__(self):
        self.author = "xiaolin"
        self.raw = b"proj\x00\x00"
        self.missing = None

    def start_time_string(self):
        return "local time = 22-Apr-2026 14:58:14 + 088990u [GMT+8.00]"

    def boom(self):
        raise RuntimeError("boom")


class TestBlockAttrs:
    NAMES = ("author", "raw", "missing", "start_time_string", "boom", "absent")

    def test_none_object_returns_empty(self):
        assert MDFLazyLoader._block_attrs(None, self.NAMES) == {}

    def test_callable_attr_is_invoked_not_repr(self):
        """实测 v3/v4 的 HeaderBlock.start_time_string 都是**方法**（缺陷 3）。

        不取返回值的话，信息窗口会直出 ``<bound method ...&gt;``。
        """
        out = MDFLazyLoader._block_attrs(_FakeBlock(), self.NAMES)
        assert out["start_time_string"].startswith("local time = ")

    def test_raising_callable_becomes_none(self):
        """宁可少一行，也不能让归属/文件信息渲染出垃圾。"""
        out = MDFLazyLoader._block_attrs(_FakeBlock(), self.NAMES)
        assert out["boom"] is None

    def test_data_and_bytes_attrs_normalized(self):
        out = MDFLazyLoader._block_attrs(_FakeBlock(), self.NAMES)
        assert out["author"] == "xiaolin"
        assert out["raw"] == "proj"
        assert out["missing"] is None
        assert out["absent"] is None


# ---------------------------------------------------------------------------
# SingleShotGroup 过滤
# ---------------------------------------------------------------------------


class TestSingleShotGroupFilter:
    def test_single_shot_group_excluded(self, loader4):
        """comment 含 ``SingleShotGroup`` 的组不进入变量列表。

        这直接影响缺陷影响面评估：真实文件中被 ct 判定修正的通道若
        落在单发组内，用户实际看不到差异。
        """
        assert "SingleCh" not in loader4.var_names

    def test_empty_group_is_not_filtered(self, loader4):
        """空组不属于单发组，仍应出现在列表中（并如实报 0 点）。"""
        assert "EmptyCh" in loader4.var_names


# ---------------------------------------------------------------------------
# 改进 I：close() 加固
# ---------------------------------------------------------------------------


class TestCloseHardening:
    def test_close_is_idempotent(self, mdf4_path):
        loader = MDFLazyLoader(str(mdf4_path))
        loader.close()
        loader.close()  # 二次关闭不得抛异常
        loader.close()
        assert loader._closed is True

    @pytest.mark.parametrize(
        "method",
        [
            "get_channel_info",
            "get_metadata",
            "get_samples_chunked",
            "get_group_variables",
        ],
    )
    def test_closed_loader_raises_keyerror(self, mdf4_path, method):
        """关闭后所有数据访问统一抛 ``KeyError``。

        统一异常类型是刻意设计：调用方（如后台统计线程）只需
        ``except KeyError`` 即可降级，不必同时兜住 AttributeError。
        改动前 ``del self._mdf`` 会让并发方拿到
        ``AttributeError('NoneType' object has no attribute 'get')``。
        本用例同时是 get_group_variables docstring 里"close 后仍抛
        KeyError"的口径凭据（table_dialog 两个调用方据此做 try/except 降级）。
        """
        loader = MDFLazyLoader(str(mdf4_path))
        name = loader.var_names[0]  # 必须在 close 前取：close 会清空 _metadata
        loader.close()
        with pytest.raises(KeyError):
            if method == "get_samples_chunked":
                loader.get_samples_chunked(name, 0, 4)
            elif method == "get_group_variables":
                loader.get_group_variables(0)
            else:
                getattr(loader, method)(name)

    def test_get_metadata_returns_none_for_unknown_even_when_open(self, loader4):
        """未关闭时对未知变量返回 None（与 get_channel_info 的 KeyError 区分）。"""
        assert loader4.get_metadata("NoSuchChannel__") is None

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MDFLazyLoader(str(tmp_path / "absent.mf4"))

    def test_failed_init_instance_is_collectable(self, tmp_path):
        """构造失败的对象被 GC 时，``__del__`` → ``close()`` 不得二次抛异常。

        依靠模块级 ``filterwarnings`` 把 unraisable 异常升级为失败：
        若 close() 在容器未建的中间态抛 AttributeError，本用例会报错。
        """
        with pytest.raises(FileNotFoundError):
            MDFLazyLoader(str(tmp_path / "absent.mf4"))
        # 不绑定名字：异常传播后对象仅由 traceback 持有，gc 即可触发 __del__
        gc.collect()

    def test_close_survives_bare_instance(self):
        """完全未构造的实例（连锁都没有）调 close() 也不得抛异常。

        对应 close() 首行的 ``getattr(self, "_access_lock", None)`` 兜底。
        """
        MDFLazyLoader.__new__(MDFLazyLoader).close()

    def test_close_survives_lock_only_instance(self):
        """锁已建但容器未建的中间态：close() 仍需完整执行并置位。

        修复前此处抛 AttributeError，且会跳过后续的 ``_mdf.close()``，
        导致文件句柄泄露。
        """
        loader = MDFLazyLoader.__new__(MDFLazyLoader)
        loader._access_lock = threading.RLock()
        loader._closed = False
        loader._mdf = None
        loader.close()
        assert loader._closed is True

    def test_close_releases_mdf_handle(self, mdf4_path):
        """close() 必须真正关闭 asammdf 句柄并置 None（不保留可误用的引用）。"""
        loader = MDFLazyLoader(str(mdf4_path))
        ref = weakref.ref(loader)
        loader.close()
        assert loader._mdf is None
        del loader
        gc.collect()
        assert ref() is None, "close 后对象应可被回收"

    def test_zero_byte_file_raises_valueerror(self, tmp_path):
        bad = tmp_path / "empty.mf4"
        bad.write_bytes(b"")
        with pytest.raises(ValueError):
            MDFLazyLoader(str(bad))


# ---------------------------------------------------------------------------
# get_samples_chunked：分块读取
# ---------------------------------------------------------------------------


class TestGetSamplesChunked:
    def test_offset_and_count_honoured(self, loader4):
        full = np.asarray(loader4.get_samples_chunked("Press_G0", 0, -1))
        head = np.asarray(loader4.get_samples_chunked("Press_G0", 0, 4))
        tail = np.asarray(loader4.get_samples_chunked("Press_G0", 8, 4))
        assert full.size == 12
        assert head.size == 4
        assert tail.size == 4
        np.testing.assert_allclose(head, full[:4])
        np.testing.assert_allclose(tail, full[8:12])

    def test_chunks_cover_whole_channel_without_gap(self, loader4):
        """分块统计的正确性前提：各块拼接后与全量一致。"""
        full = np.asarray(loader4.get_samples_chunked("Press_G0", 0, -1))
        parts = [
            np.asarray(loader4.get_samples_chunked("Press_G0", off, 5))
            for off in (0, 5, 10)
        ]
        np.testing.assert_allclose(np.concatenate(parts), full)

    def test_returns_physical_values_for_enum(self, loader4):
        """统计恒定 ``raw=False``：枚举通道取物理值（文本）而非码值。

        绘图路径用 ``raw=is_enum`` 取码值配合文本标签，但统计必须基于
        物理值，否则 min/max/mean 得到的是无意义的枚举码。
        """
        samples = np.asarray(loader4.get_samples_chunked("State", 0, -1))
        assert samples.dtype.kind in "SU"
        assert set(samples.tolist()) <= {b"off", b"on", b"err"}

    def test_empty_group_returns_empty(self, loader4):
        samples = np.asarray(loader4.get_samples_chunked("EmptyCh", 0, -1))
        assert samples.size == 0

    def test_does_not_populate_signal_cache(self, loader4):
        """统计分块读取不得污染 LRU 缓存。

        否则一次全量统计就会把绘图用的热数据挤出缓存
        （单条 1.7 MB、上限 256 条）。
        """
        loader4.clear_cache()
        loader4.get_samples_chunked("Press_G0", 0, -1)
        assert len(loader4._signal_cache) == 0


# ---------------------------------------------------------------------------
# Group-level access：变量数值表 tab 模式所需的 group 级接口
# ---------------------------------------------------------------------------


class TestGroupLevelAccess:
    """覆盖 ``get_var_group_index`` / ``get_group_time_array`` /
    ``get_group_label`` / ``get_group_variables`` / ``search_variables`` 五个新方法。

    合成文件结构（write_mdf n=12, with_single_shot_group=True,
    with_empty_group=True）：
    - Group 0 (NormalGroup,  comment="synthetic test group")
        → Press_G0, State, Label
    - Group 1 (DupGroup,     comment="duplicate channel name group")
        → Press_G1
    - Group 2 (SingleShotGroup, comment="SingleShotGroup") — 被 loader 跳过
    - Group 3 (EmptyGroup,   comment="empty reserved group")
        → EmptyCh（0 采样点）
    """

    # -- get_var_group_index -----------------------------------------------

    def test_get_var_group_index_first_group(self, loader4):
        """Group 0 中的变量应返回 group_index=0。"""
        assert loader4.get_var_group_index("Press_G0") == 0
        assert loader4.get_var_group_index("State") == 0
        assert loader4.get_var_group_index("Label") == 0

    def test_get_var_group_index_second_group(self, loader4):
        """Group 1 中的变量应返回 group_index=1。"""
        assert loader4.get_var_group_index("Press_G1") == 1

    def test_get_var_group_index_empty_group(self, loader4):
        """空组中的变量也应正确返回其 group 索引。"""
        assert loader4.get_var_group_index("EmptyCh") == 3

    def test_get_var_group_index_nonexistent_raises(self, loader4):
        """不存在的变量应抛 KeyError。"""
        with pytest.raises(KeyError, match="不存在"):
            loader4.get_var_group_index("NoSuchVariable")

    # -- get_group_time_array -----------------------------------------------

    def test_get_group_time_array_first_group(self, loader4):
        """Group 0：12 点、间隔 0.1 s → [0.0, 0.1, ..., 1.1]。"""
        t = loader4.get_group_time_array(0)
        assert t.dtype == np.float64
        assert len(t) == 12
        np.testing.assert_allclose(t, np.arange(12) * 0.1)

    def test_get_group_time_array_second_group(self, loader4):
        """Group 1：6 点、间隔 0.2 s → [0.0, 0.2, ..., 1.0]。"""
        t = loader4.get_group_time_array(1)
        assert len(t) == 6
        np.testing.assert_allclose(t, np.arange(6) * 0.2)

    def test_get_group_time_array_cached(self, loader4):
        """第二次调用同一 group 应命中缓存（返回同一数组对象）。"""
        t1 = loader4.get_group_time_array(0)
        t2 = loader4.get_group_time_array(0)
        assert t1 is t2

    def test_get_group_time_array_different_groups_differ(self, loader4):
        """不同 group 的时间数组应不同（长度和内容均不同）。"""
        t0 = loader4.get_group_time_array(0)
        t1 = loader4.get_group_time_array(1)
        assert len(t0) != len(t1)
        assert not np.array_equal(t0[:6], t1)

    # -- get_group_label ----------------------------------------------------

    def test_get_group_label_with_acq_name(self, loader4):
        """有 acq_name 时标签格式为 '{acq_name} (G{index})'。"""
        assert loader4.get_group_label(0) == "NormalGroup (G0)"
        assert loader4.get_group_label(1) == "DupGroup (G1)"

    def test_get_group_label_unknown_index(self, loader4):
        """不存在的 group 索引应兜底为 'G{index}'。"""
        assert loader4.get_group_label(999) == "G999"

    def test_get_group_label_empty_group(self, loader4):
        """EmptyGroup 有 acq_name='EmptyGroup'。"""
        label = loader4.get_group_label(3)
        assert "G3" in label
        assert "EmptyGroup" in label

    # -- get_group_variables ------------------------------------------------

    def test_get_group_variables_returns_display_names(self, loader4):
        """必须返回聚合后显示名 Press_G0，而不是文件里的原始通道名 Press。

        表格列名用的就是显示名，调用方拿原始名去 get_series 会 KeyError；
        而 _raw_metadata 恰好存的也是原始名，这一步读错来源不会报错、只会在
        跨组重名的真实文件上静默加错列。
        """
        assert loader4.get_group_variables(0) == ["Press_G0", "State", "Label"]
        assert loader4.get_group_variables(1) == ["Press_G1"]
        assert "Press" not in loader4.get_group_variables(0)

    def test_get_group_variables_excludes_time_channel(self, loader4):
        """时间通道（master）不得混进来：tab 的 time 首列就是它，再加一次会得到两个同义列。"""
        for gi in (0, 1, 3):
            assert "time" not in loader4.get_group_variables(gi)

    def test_get_group_variables_matches_var_names_membership(self, loader4):
        """各组变量不重叠、且都是 var_names 的完整子集。"""
        grouped = [n for gi in range(loader4.group_count)
                   for n in loader4.get_group_variables(gi)]
        assert len(grouped) == len(set(grouped))
        assert set(grouped) <= set(loader4.var_names)

    def test_get_group_variables_single_shot_group_is_absent(self, loader4):
        """被 loader 跳过的 SingleShotGroup（gi=2）无元数据 → 空列表。"""
        assert loader4.get_group_variables(2) == []

    def test_get_group_variables_unknown_index(self, loader4):
        """未知组返回空列表，与 get_group_time_array 的宽容处理一致（UI 侧不必 try）。"""
        assert loader4.get_group_variables(999) == []

    # -- search_variables ---------------------------------------------------

    def test_search_variables_matches_press(self, loader4):
        """搜索 'press' 应匹配 Press_G0 和 Press_G1（大小写不敏感）。"""
        results = loader4.search_variables("press")
        names = [r[0] for r in results]
        assert "Press_G0" in names
        assert "Press_G1" in names

    def test_search_variables_returns_group_info(self, loader4):
        """搜索结果应包含正确的 group_index 和 group_label。"""
        results = loader4.search_variables("Press_G0")
        assert len(results) >= 1
        name, gi, label = results[0]
        assert name == "Press_G0"
        assert gi == 0
        assert "NormalGroup" in label

    def test_search_variables_case_insensitive(self, loader4):
        """搜索 'STATE' 应匹配 State。"""
        results = loader4.search_variables("STATE")
        names = [r[0] for r in results]
        assert "State" in names

    def test_search_variables_no_match(self, loader4):
        """无匹配时返回空列表。"""
        results = loader4.search_variables("zzz_nonexistent")
        assert results == []

    def test_search_variables_respects_limit(self, loader4):
        """limit=1 时最多返回 1 条结果。"""
        results = loader4.search_variables("press", limit=1)
        assert len(results) == 1

    def test_search_variables_empty_keyword_matches_all(self, loader4):
        """空关键词应匹配所有变量。"""
        results = loader4.search_variables("")
        all_names = loader4.var_names
        # 空关键词匹配所有非时间通道
        assert len(results) >= len(all_names) - 2  # 减去可能的时间通道
