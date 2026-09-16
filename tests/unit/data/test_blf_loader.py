"""``src/data/blf_loader.py`` 的单元测试（方案 §8 unit 层 + §7 P2/P3 验收）。

断言分三类，各管不同的失效模式：

1. **契约逐项**（§3.4）：BLF 走的是 MDF 已验证的分组路径，契约缺一个名字就是
   UI 侧一处 ``AttributeError``，所以这里按方案里那份清单逐个点名，不用
   ``dir()`` 模糊扫描。
2. **语义**：时间归零、group label、三段式改名、mux NaN、未匹配上报、错误帧
   不产生采样点。
3. **生命周期**：取消、``close()`` 后统一 ``KeyError``、``__del__`` 不残留、
   并发读、LRU 上限、``bind_databases`` **不重扫**。

合成样本走 :func:`write_blf_pair`（CI 主路径，无网络）；真实公开夹具只用来
证明「python-can 的每一种 CAN 对象都能读出来」，缺夹具即 skip。
"""

from __future__ import annotations

import gc
import threading
import weakref

import numpy as np
import pytest

import src.data.blf_loader as blf_module
from src.core.config import is_grouped_loader
from src.data.blf_loader import BLFLazyLoader
from src.data.metadata import UNKNOWN
from tests.fixtures import bus_data_access as bd
from tests.fixtures.data_factory import (
    CAN_FRAMES,
    CAN_UNMATCHED_ID,
    make_blf_messages,
    write_blf,
    write_blf_pair,
    write_dbc,
)

# ``close()`` 在部分构造状态下抛异常时，Python 只会打成 "Exception ignored in
# __del__" 噪声；本标记使其显式失败（与 test_mdf_lazy_loader 同一手法）。
pytestmark = pytest.mark.filterwarnings(
    "error::pytest.PytestUnraisableExceptionWarning"
)

T0 = 1_700_000_000.0


def _load(directory, *, n=20, buses=(0,), basename="syn_can", **kw):
    blf, dbc = write_blf_pair(
        directory, n=n, buses=buses, basename=basename, **kw
    )
    return BLFLazyLoader(str(blf), db_paths=[str(dbc)]), blf, dbc


@pytest.fixture(scope="module")
def loader_and_paths(tmp_path_factory):
    directory = tmp_path_factory.mktemp("blfmain")
    loader, blf, dbc = _load(directory)
    yield loader, blf, dbc
    loader.close()


@pytest.fixture(scope="module")
def dual_bus_loader(tmp_path_factory):
    """两条总线收全量报文：同一 CAN ID 出两个桶，触发第三段改名。"""
    loader, _blf, _dbc = _load(tmp_path_factory.mktemp("blfdual"), buses=(0, 1))
    yield loader
    loader.close()


# ---------------------------------------------------------------------------
# 契约面（§3.4 清单逐项）
# ---------------------------------------------------------------------------

CONTRACT_ATTRS = [
    "path", "file_size", "var_names", "units", "df", "df_validity",
    "datalength", "max_row_count", "row_count", "column_count",
    "global_time_range", "baseline_density", "time_column_name",
    "time_axis_label", "time_values", "time_channels_info", "groups",
    "group_count", "current_group_index", "LOADER_TYPE", "GROUP_MODE",
]
CONTRACT_METHODS = [
    "get_series", "get_value_from_name", "get_var_group_index",
    "get_group_time_array", "get_group_label", "get_group_variables",
    "search_variables", "get_metadata", "get_channel_info",
    "get_samples_chunked", "clear_cache", "release_memory", "close",
]
BLF_ONLY_ATTRS = [
    "db_paths", "load_errors", "unmatched_ids", "total_messages", "truncated",
]


def test_contract_surface_is_complete(loader_and_paths):
    loader = loader_and_paths[0]
    missing = [a for a in CONTRACT_ATTRS if not hasattr(loader, a)]
    assert not missing, f"契约属性缺失: {missing}"
    missing = [m for m in CONTRACT_METHODS if not callable(getattr(loader, m, None))]
    assert not missing, f"契约方法缺失: {missing}"
    missing = [a for a in BLF_ONLY_ATTRS if not hasattr(loader, a)]
    assert not missing, f"BLF 专有属性缺失: {missing}"
    assert callable(loader.bind_databases)


def test_loader_is_recognised_as_a_grouped_source(loader_and_paths):
    """分组判定收敛（P0）的真正落点：BLF 必须和 MDF 一样走分组路径。"""
    loader = loader_and_paths[0]
    assert loader.LOADER_TYPE == "blf"
    assert loader.GROUP_MODE == "can_message"
    assert is_grouped_loader(loader) is True


def test_df_is_none_and_validity_is_unknown(loader_and_paths):
    """``df`` 恒为 None 是契约的一部分：消费方以它判定「惰性数据源」。"""
    loader = loader_and_paths[0]
    assert loader.df is None
    assert set(loader.df_validity.values()) == {UNKNOWN}
    assert loader.time_column_name == "time"
    assert loader.time_axis_label == "time (s)"
    assert loader.time_channels_info == {}


# ---------------------------------------------------------------------------
# 时间轴
# ---------------------------------------------------------------------------

def test_time_axis_is_zero_based_relative_seconds(loader_and_paths):
    """时间轴必须归零；但“归零”是**全局**首帧，不是每个桶各自首帧。

    每个桶只包含自己那条报文的帧，所以它的起点必然 ≥ 0（且通常 > 0）。
    另外 BLF 的时间戳有固定粒度（实测 0.03 存回来是 0.0299999714），
    因此不能按 1e-9 断言，只能给到微秒量级。
    """
    loader = loader_and_paths[0]
    mins = []
    for gi in range(loader.group_count):
        t = loader.get_group_time_array(gi)
        assert t.size, f"group {gi} 没有时间轴"
        assert np.isfinite(t).all()
        assert t.min() >= -1e-6, f"group {gi} 出现了负时间"
        assert (t - t.min()).min() >= -1e-9
        mins.append(float(t.min()))
    assert min(mins) == pytest.approx(0.0, abs=1e-6)
    assert loader.global_time_range[0] == pytest.approx(0.0, abs=1e-6)
    assert loader.global_time_range[1] > 0.0


def test_absolute_start_time_only_lives_in_the_header(loader_and_paths):
    """§2：绝对起始时间进 ``header.start_time_string``，不污染数值轴。"""
    loader = loader_and_paths[0]
    info = loader.get_channel_info(loader.var_names[0])
    text = info["header"]["start_time_string"]
    assert text and text[:4] == "2023"      # T0 = 2023-11-14（本地时区）
    assert info["time_base"]["time_min"] == pytest.approx(
        loader.get_group_time_array(loader.get_var_group_index(loader.var_names[0]))[0]
    )


def test_single_frame_bucket_does_not_make_a_zero_length_range(tmp_path):
    """只有 1 帧的桶没有 span：周期必须是 None 而不是除零，X 轴必须非退化。"""
    blf = write_blf(
        tmp_path / "one.blf",
        [{"timestamp": T0, "channel": 0, "can_id": 0x1A0, "data": b"\x00" * 8}],
    )
    dbc = write_dbc(tmp_path / "one.dbc")
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        assert loader.total_messages == 1
        lo, hi = loader.global_time_range
        assert hi > lo
        assert loader.baseline_density >= 0.0
    finally:
        loader.close()


# ---------------------------------------------------------------------------
# 分桶与组
# ---------------------------------------------------------------------------

def test_groups_are_channels_times_ids_and_labels_read_well(loader_and_paths):
    loader, _blf, dbc = loader_and_paths
    labels = [loader.get_group_label(gi) for gi in range(loader.group_count)]
    assert any(lbl.startswith("Ch0 0x1A0 EngineData (G") for lbl in labels), labels
    # 未匹配的 0x7FF 桶不生成变量 ⇒ 不在 group_count 里（否则 UI 出现空 tab）
    assert loader.group_count == len(loader.groups)
    for group in loader.groups:
        assert group["var_names"]
        assert loader.get_group_variables(group["index"]) == group["var_names"]
    assert all("0x7FF" not in loader.get_group_label(gi)
               for gi in range(loader.group_count))


def test_same_id_on_two_buses_creates_two_buckets(dual_bus_loader):
    loader = dual_bus_loader
    engine_groups = [
        gi for gi in range(loader.group_count)
        if "0x1A0" in loader.get_group_label(gi)
    ]
    assert len(engine_groups) == 2
    channels = {loader.get_group_label(gi).split()[0] for gi in engine_groups}
    assert channels == {"Ch0", "Ch1"}


def test_three_tier_renaming(dual_bus_loader):
    """§2 命名：裸名唯一 → 裸名；重名 → ``报文名.信号名``；仍重名 → ``_G{gi}``。

    单总线时 ``EngineSpeed`` 已在 EngineData / BrakeData 间重名 ⇒ 加前缀；
    双总线时同一报文又出现在两条总线上 ⇒ 前缀仍冲突 ⇒ 必须补 ``_G{gi}``。
    """
    loader = dual_bus_loader
    names = loader.var_names
    assert len(names) == len(set(names)), "变量名必须全局唯一"
    prefixed = [n for n in names if n.startswith("EngineData.EngineSpeed")]
    assert prefixed, names
    assert all("_G" in n for n in prefixed), prefixed
    # original_name 永远是裸信号名（导出/搜索按它匹配）
    for n in prefixed:
        assert loader.get_metadata(n).original_name == "EngineSpeed"


def test_single_bus_names_only_get_the_frame_prefix(loader_and_paths):
    loader = loader_and_paths[0]
    names = set(loader.var_names)
    # 单总线：前缀已唯一，不该再多缀 _G
    assert "EngineData.EngineSpeed" in names
    assert "BrakeData.EngineSpeed" in names
    assert not any(n.endswith("_G0") for n in names), names
    # 只出现一次的信号保留裸名
    assert "CoolantTemp" in names


def test_unmatched_ids_are_reported_without_blocking_signals(loader_and_paths):
    """§2：未匹配 ID **不生成信号**，但必须被上报（banner 状态③的数据源）。"""
    loader, _blf, _dbc = loader_and_paths
    assert loader.unmatched_ids, "样本里应有 0x7FF"
    channels = {ch for ch, _cid, _ext, _n in loader.unmatched_ids}
    ids = {cid for _ch, cid, _ext, _n in loader.unmatched_ids}
    assert ids == {CAN_UNMATCHED_ID}
    assert channels == {0}
    # 按帧数降序
    counts = [n for _c, _i, _e, n in loader.unmatched_ids]
    assert counts == sorted(counts, reverse=True)
    assert all("0x7FF" not in n for n in loader.var_names)
    assert loader.unmatched_report(limit=1) == loader.unmatched_ids[:1]


def test_an_unmatched_extended_frame_keeps_its_format_flag(tmp_path):
    """11 位与 29 位的取值会重叠：帧格式必须是数据层给的真值，不能由 ID 大小猜。

    一个 0x123 的**扩展**帧与同值的**标准**帧在库里是两条报文，且两条都匹配
    不上。详情表若按 ``can_id > 0x7FF`` 判，两条会标成同一副模样，用户按这张
    表去补 DBC 就会找错报文。
    """
    rows = [
        {"timestamp": T0, "channel": 0, "can_id": 0x123, "data": b"\x00" * 8},
        {"timestamp": T0 + 0.01, "channel": 0, "can_id": 0x123,
         "data": b"\x00" * 8, "extended": True},
    ]
    blf = write_blf(tmp_path / "ext.blf", rows)
    dbc = write_dbc(tmp_path / "ext.dbc")
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        flags = {(cid, ext) for _ch, cid, ext, _n in loader.unmatched_ids}
        assert flags == {(0x123, False), (0x123, True)}, "两条必须按帧格式各住一条"
    finally:
        loader.close()


def test_error_and_remote_frames_produce_no_samples(tmp_path):
    """§2 最后一行：错误帧 / 远程帧跳过，不进桶、不产生采样点。"""
    blf, dbc = write_blf_pair(
        tmp_path, n=40, buses=(0,), with_remote=True, with_error=True
    )
    raw = make_blf_messages(n=40, buses=(0,), with_remote=True, with_error=True)
    wanted = sum(1 for r in raw if not r.get("error") and not r.get("remote"))

    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        assert loader.total_messages == wanted
        all_ids = {
            loader.get_group_label(gi) for gi in range(loader.group_count)
        }
        assert not any("0x5" in lbl.split()[1] for lbl in all_ids), all_ids
    finally:
        loader.close()


# ---------------------------------------------------------------------------
# 周期：标称来自数据库，实测来自报文
# ---------------------------------------------------------------------------

def test_nominal_raster_comes_from_the_dbc_not_from_measurement(loader_and_paths):
    """§3.1 / §5.7：「标称间隔」必须是数据库声明值，不是实测均值的别名。

    DBC 里 EngineData 是 20 ms，而模块级样本每 0.04 s 才轮到它一帧（工厂按
    10 ms 全局步长发 4 帧轮转）。两个读数不相等正是 var_info 坚持两行都
    展示的理由：只看实测会把「采集丢帧」误读成「本来就这么慢」。
    """
    loader = loader_and_paths[0]
    name = next(n for n in loader.var_names if n.endswith("EngineSpeed"))
    assert name.startswith("EngineData")     # 拿错报文则下面两个数全部失去意义
    meta = loader.get_metadata(name)
    assert meta.nominal_raster_s == pytest.approx(0.02)
    assert meta.effective_rate_hz == pytest.approx(25.0)
    # §5.7 明文：「CAN 报文」分区里的周期用实测均值，并且要报出读数来源
    can = loader.get_channel_info(name)["can"]
    assert can["cycle_time_s"] == pytest.approx(0.04)
    assert can["cycle_time_measured"] is True


def test_single_shot_signal_falls_back_to_the_declared_cycle(tmp_path):
    """只有一帧时没有均值可言，周期必须回退到数据库声明值而不是空。"""
    rows = [{
        "timestamp": T0, "channel": 0, "can_id": 0x1A0,
        "data": b"\x00" * 8,
    }]
    blf = write_blf(tmp_path / "one.blf", rows)
    dbc = write_dbc(tmp_path / "one.dbc")
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        name = next(n for n in loader.var_names if n.endswith("EngineSpeed"))
        meta = loader.get_metadata(name)
        assert meta.nominal_raster_s == pytest.approx(0.02)
        assert meta.effective_rate_hz is None
        # 单帧桶没有实测均值，cycle_time_s 退成了数据库声明值 —— 标志必须是
        # False，否则 var_info 会把注释当成测量结果展示（L-1）。
        can = loader.get_channel_info(name)["can"]
        assert can["cycle_time_s"] == pytest.approx(0.02)
        assert can["cycle_time_measured"] is False
    finally:
        loader.close()


# ---------------------------------------------------------------------------
# 解码结果
# ---------------------------------------------------------------------------

def test_series_agree_with_canmatrix_frame_decode(loader_and_paths):
    """端到端真值：loader 的曲线必须与 canmatrix 逐帧解码一致。

    这条不重复 can_decode 的单测 —— 它验的是「桶里存的字节确实是那条报文」，
    即扫描/分块/合并路径没有错位或串帧。
    """
    loader, _blf, _dbc = loader_and_paths
    checked = 0
    for name in loader.var_names:
        sig = loader._var_to_sig[name]
        if sig.mux is not None or sig.is_mux_selector:
            continue        # 非本 mux 值的采样点本来就是 NaN，与上游不可比
        gi = loader.get_var_group_index(name)
        bucket = loader._buckets[gi]
        frame = bucket.frame
        values = loader.get_series(name).to_numpy()
        assert values.shape[0] == bucket.count
        for i in range(bucket.count):
            row = bucket.payload[i].tobytes()[: frame.dlc]
            truth = frame.cm_frame.decode(bytearray(row))[sig.name]
            assert values[i] == pytest.approx(float(truth.phys_value)), \
                f"{name} 第 {i} 帧"
            checked += 1
    # 8 个非 mux 信号 × 每桶 5 帧；掉下来说明某个报文或信号没被覆盖
    assert checked >= 32, f"实际比对 {checked} 个点，覆盖不足"


def test_decoded_series_matches_cantools_reference_decode(loader_and_paths):
    """第三只眼（§6.4）：cantools 逐帧解码同一批 payload，物理值必须一致。

    上面那条拿 canmatrix 作真值，而本项目的解码器本身就是「对齐 canmatrix」
    写出来的 —— 只比对它，「两边错得一致」永远发现不了。cantools 与二者都无
    关系，而且它同时反向验了**合成夹具自己的编码**：如果
    ``data_factory._encode_bits`` 的位序写错，这一条会报，而
    ``test_series_agree_with_canmatrix_frame_decode`` 不会（它两边用的是同一
    份字节）。
    """
    cantools = pytest.importorskip("cantools")
    loader, _blf, dbc = loader_and_paths
    db = cantools.database.load_file(str(dbc))
    checked = 0
    for name in loader.var_names:
        sig = loader._var_to_sig[name]
        if sig.mux is not None or sig.is_mux_selector:
            continue
        gi = loader.get_var_group_index(name)
        bucket = loader._buckets[gi]
        message = db.get_message_by_frame_id(bucket.can_id)
        assert message is not None, hex(bucket.can_id)
        values = loader.get_series(name).to_numpy()
        for i in range(bucket.count):
            row = bucket.payload[i].tobytes()[: bucket.stride]
            truth = message.decode(row, decode_choices=False)
            assert values[i] == pytest.approx(float(truth[sig.name])), \
                f"{name} 第 {i} 帧"
            checked += 1
    assert checked >= 32, f"实际比对 {checked} 个点，覆盖不足"


def test_multiplexed_branch_is_nan_outside_its_group(tmp_path):
    """mux 分支：非本 mux 值的采样点必须是 NaN（§2）。"""
    blf, dbc = write_blf_pair(tmp_path, n=12, buses=(0,))
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        a = next(n for n in loader.var_names if "BranchA" in n)
        b = next(n for n in loader.var_names if "BranchB" in n)
        va = loader.get_series(a).to_numpy()
        vb = loader.get_series(b).to_numpy()
        assert va.shape == vb.shape
        # 样本按 mux 帧自己的出现次数交替选路 ⇒ 两者都应有 NaN 且互斥
        assert np.isnan(va).any() and np.isnan(vb).any()
        assert not (np.isnan(va) & np.isnan(vb)).any()
        assert (np.isnan(va) | np.isnan(vb)).all()
        # 选择器本身每帧都有效，绝不能被掩成 NaN
        sel = next(n for n in loader.var_names if "MuxSel" in n)
        assert not np.isnan(loader.get_series(sel).to_numpy()).any()
    finally:
        loader.close()


def test_enum_series_uses_raw_codes_with_an_enum_map(loader_and_paths):
    loader = loader_and_paths[0]
    name = next(n for n in loader.var_names if n.endswith("StarterMode"))
    meta = loader.get_metadata(name)
    assert meta.is_enum is True
    assert meta.enum_map == {0: "off", 1: "run", 2: "error"}
    values = loader.get_series(name).to_numpy()
    assert set(np.unique(values[~np.isnan(values)])) <= {0.0, 1.0, 2.0}
    # 统计路径（get_samples_chunked）刻意**不**做码值化
    chunked = loader.get_samples_chunked(name, 0, 5)
    assert chunked.dtype == np.float64


def test_get_value_from_name_returns_time_unit_and_enum_map(loader_and_paths):
    loader = loader_and_paths[0]
    name = next(n for n in loader.var_names if n.endswith("StarterMode"))
    x, values, unit, enum_map = loader.get_value_from_name(name)
    gi = loader.get_var_group_index(name)
    np.testing.assert_array_equal(x, loader.get_group_time_array(gi))
    assert values.shape[0] == x.shape[0]
    assert unit == loader.units[name]
    assert enum_map == {0: "off", 1: "run", 2: "error"}
    plain = next(n for n in loader.var_names if n.endswith("EngineSpeed"))
    assert loader.get_value_from_name(plain)[3] == {}


# ---------------------------------------------------------------------------
# get_channel_info 的 can 子 dict（§3.4 / §5.7）
# ---------------------------------------------------------------------------

CAN_KEYS = [
    "message_name", "can_id_hex", "is_extended", "dlc", "bus_channel",
    "start_bit", "bit_size", "byte_order", "is_signed", "is_float",
    "factor", "offset", "cycle_time_s", "cycle_time_measured",
    "receivers", "db_file", "mux_text",
]


def test_channel_info_has_the_can_sub_dict_and_mdf_shaped_keys(loader_and_paths):
    loader, _blf, dbc = loader_and_paths
    name = next(n for n in loader.var_names if n.endswith("EngineSpeed"))
    info = loader.get_channel_info(name)
    missing = [k for k in CAN_KEYS if k not in info["can"]]
    assert not missing, f"can 子 dict 缺键: {missing}"
    # var_info._from_blf 复用 _from_mdf 骨架的前提：这些键名必须还在
    for section in ("channel", "channel_group", "conversion", "header",
                    "time_base", "file"):
        assert section in info, section
    assert info["version"] == "BLF"
    assert info["can"]["message_name"] == "EngineData"
    assert info["can"]["can_id_hex"] == "0x1A0"
    assert info["can"]["db_file"] == str(dbc)
    assert info["channel"]["bit_count"] == 16
    assert info["can"]["byte_order"] == "Intel（小端）"


def test_channel_info_start_bit_is_the_dbc_number(loader_and_paths):
    """用户拿 CANoe / DBC 编辑器对照，看到的必须是 DBC 文本位号。"""
    loader = loader_and_paths[0]
    fl = next(n for n in loader.var_names if "WheelSpeedFL" in n)
    can = loader.get_channel_info(fl)["can"]
    assert can["start_bit"] == 7
    assert can["start_bit_internal"] == 0
    assert can["byte_order"] == "Motorola（大端）"


def test_channel_info_mux_text_is_human_readable(loader_and_paths):
    loader = loader_and_paths[0]
    sel = next(n for n in loader.var_names if "MuxSel" in n)
    branch = next(n for n in loader.var_names if "BranchA" in n)
    texts = {loader.get_channel_info(v)["can"]["mux_text"] for v in (sel, branch)}
    assert any("选择器" in t for t in texts), texts
    assert any("==" in t for t in texts), texts
    plain = next(n for n in loader.var_names if n.endswith("EngineSpeed"))
    assert loader.get_channel_info(plain)["can"]["mux_text"] == ""


# ---------------------------------------------------------------------------
# bind_databases：换库不重扫
# ---------------------------------------------------------------------------

def test_bind_databases_rebinds_without_rescanning(loader_and_paths):
    """§5.6 的全部前提：payload 常驻，换库只重建元数据。"""
    loader, _blf, dbc = loader_and_paths
    before_messages = loader.total_messages
    before_buckets = [id(b) for b in loader._buckets]
    before_strides = [b.stride for b in loader._buckets]
    assert loader.var_names

    stats = loader.bind_databases([])
    assert stats["var_count"] == 0
    assert stats["frame_count"] == 0
    assert loader.var_names == []
    assert loader.unmatched_ids, "清空数据库后所有报文都应变为未匹配"

    assert loader.total_messages == before_messages
    assert [id(b) for b in loader._buckets] == before_buckets, "桶对象被换掉即重扫"
    assert [b.stride for b in loader._buckets] == before_strides

    stats = loader.bind_databases([str(dbc)])
    assert stats["var_count"] > 0
    assert [id(b) for b in loader._buckets] == before_buckets
    # 重绑必须清缓存，否则会拿到上一个库的旧解码结果
    assert not loader._signal_cache and not loader._time_cache


def test_bind_databases_switching_to_a_conflicting_db_changes_factors(tmp_path):
    """first-match-wins 的 UI 落点：重选到另一份库，因子必须跟着变。"""
    blf, dbc_a = write_blf_pair(tmp_path / "a", n=6, buses=(0,))
    engine = CAN_FRAMES[0]
    dbc_b = write_dbc(tmp_path / "b" / "alt.dbc", [{
        **engine,
        "signals": [
            {**s, "factor": 2.0} if s["name"] == "EngineSpeed" else s
            for s in engine["signals"]
        ],
    }])
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc_a)])
    try:
        name = next(n for n in loader.var_names if n.endswith("EngineSpeed"))
        assert loader.get_channel_info(name)["can"]["factor"] == pytest.approx(0.125)
        loader.bind_databases([str(dbc_b)])
        name = next(n for n in loader.var_names if n.endswith("EngineSpeed"))
        assert loader.get_channel_info(name)["can"]["factor"] == pytest.approx(2.0)
        assert loader.db_paths == [str(dbc_b)]
    finally:
        loader.close()


# ---------------------------------------------------------------------------
# 进度 / 取消 / 上限
# ---------------------------------------------------------------------------

def test_progress_is_monotonic_and_ends_at_100(tmp_path):
    blf, dbc = write_blf_pair(tmp_path, n=60, buses=(0,))
    seen: list[int] = []
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)], _progress=seen.append)
    try:
        assert seen == sorted(seen)
        assert seen[0] >= 10
        assert seen[-1] == 100
        assert all(0 <= v <= 100 for v in seen)
    finally:
        loader.close()


def test_stats_dict_exposes_live_message_count(tmp_path):
    """进度框要显示「已读 N 帧」，而 loader 实例在扫描完成前拿不到 ⇒ 共享 dict。"""
    blf, dbc = write_blf_pair(tmp_path, n=30, buses=(0,))
    stats: dict = {}
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)], _stats=stats)
    try:
        assert stats["messages"] == loader.total_messages > 0
        assert stats["buckets"] == len(loader._buckets)
    finally:
        loader.close()


def test_cancel_stops_at_a_chunk_boundary_and_releases_the_handle(tmp_path):
    """§5.4：取消必须在 chunk 边界生效，且不能泄漏句柄。"""
    blf, dbc = write_blf_pair(tmp_path, n=400, buses=(0,))
    calls = {"n": 0}

    def cancel_after_one_chunk():
        calls["n"] += 1
        return calls["n"] >= 1

    with pytest.raises(InterruptedError):
        BLFLazyLoader(
            str(blf), db_paths=[str(dbc)], _should_cancel=cancel_after_one_chunk
        )
    # 取消发生在**构造期**，没有半只_loader_泄漏到外部可回收
    gc.collect()
    assert calls["n"] >= 1


def test_truncation_flag_is_set_at_the_message_cap(tmp_path, monkeypatch):
    blf, dbc = write_blf_pair(tmp_path, n=200, buses=(0,))
    monkeypatch.setattr(blf_module, "BLF_MAX_MESSAGES", 30)
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        assert loader.truncated is True
        assert loader.total_messages >= 30
        assert loader.var_names, "截断后仍应给出已扫到部分的信号"
    finally:
        loader.close()


def test_not_truncated_below_the_cap(loader_and_paths):
    assert loader_and_paths[0].truncated is False


# ---------------------------------------------------------------------------
# 生命周期与并发
# ---------------------------------------------------------------------------

def test_missing_empty_and_signal_free_files_fail_cleanly(tmp_path):
    dbc = write_dbc(tmp_path / "x.dbc")
    with pytest.raises(FileNotFoundError):
        BLFLazyLoader(str(tmp_path / "nope.blf"), db_paths=[str(dbc)])

    empty = tmp_path / "empty.blf"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="为空"):
        BLFLazyLoader(str(empty), db_paths=[str(dbc)])

    # 只有错误帧 ⇒ 一条报文都没有：明确的「未包含任何 CAN 报文」而不是空数据态
    only_err = write_blf(tmp_path / "err.blf", [
        {"timestamp": T0, "channel": 0, "can_id": 0, "error": True, "data": b""},
    ])
    with pytest.raises(ValueError, match="未包含任何 CAN 报文"):
        BLFLazyLoader(str(only_err), db_paths=[str(dbc)])


def test_no_arguments_still_scans_but_yields_no_vars(tmp_path):
    """不带数据库也必须能扫（§5.1 F 路径要能判断「0 个信号」）。"""
    blf, _dbc = write_blf_pair(tmp_path, n=5, buses=(0,))
    loader = BLFLazyLoader(str(blf))
    try:
        assert loader.var_names == []
        assert loader.total_messages > 0
        assert loader.unmatched_ids
        assert loader.db_paths == []
    finally:
        loader.close()


CLOSED_READERS = [
    lambda ldr: ldr.get_series("EngineSpeed"),
    lambda ldr: ldr.get_value_from_name("EngineSpeed"),
    lambda ldr: ldr.get_group_time_array(0),
    lambda ldr: ldr.get_group_label(0),
    lambda ldr: ldr.get_group_variables(0),
    lambda ldr: ldr.get_var_group_index("EngineSpeed"),
    lambda ldr: ldr.search_variables("Engine"),
    lambda ldr: ldr.get_metadata("EngineSpeed"),
    lambda ldr: ldr.get_channel_info("EngineSpeed"),
    lambda ldr: ldr.get_samples_chunked("EngineSpeed", 0, 10),
]


@pytest.mark.parametrize("reader", CLOSED_READERS, ids=lambda f: f.__name__)
def test_every_reader_raises_keyerror_after_close(tmp_path_factory, reader):
    """§3.3：close 后统一 ``KeyError``（不是 ValueError / 不是静默返回空）。

    UI 侧的后台线程与 close 有竞态，只有统一的异常类型才接得住。
    """
    loader, _blf, _dbc = _load(tmp_path_factory.mktemp("blfclose"))
    loader.close()
    with pytest.raises(KeyError):
        reader(loader)


def test_close_is_idempotent_and_del_is_silent(tmp_path):
    """必须用**自有** loader：模块级 fixture 一旦被 close，后面的用例全废。"""
    loader, _blf, _dbc = _load(tmp_path)
    loader.close()
    loader.close()
    assert loader._closed is True


def test_partially_constructed_loader_can_be_collected(tmp_path):
    """构造期就抛异常时 ``__del__ → close()`` 必须不报 AttributeError。

    用一个**存在但为空**的文件：它走得到 ``_scan()``，因而那些只在扫描之后
    才建立的属性（``_buckets`` 等）真的缺失 —— 文件不存在的话
    ``_validate_file()`` 第一步就抛，根本测不到 ``__del__``。
    """
    empty = tmp_path / "half-built.blf"
    empty.write_bytes(b"")
    gc.disable()
    try:
        with pytest.raises(ValueError, match="为空"):
            BLFLazyLoader(str(empty))
        gc.collect()
    finally:
        gc.enable()


def test_gc_of_an_unclosed_loader_does_not_warn(tmp_path):
    blf, dbc = write_blf_pair(tmp_path, n=8, buses=(0,))
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    ref = weakref.ref(loader)
    del loader
    gc.collect()
    assert ref() is None


def test_concurrent_reads_agree_with_serial_reads(loader_and_paths):
    """多线程并发读必须给同一个数：契约里有 RLock，但要用例证明它真的生效。"""
    loader = loader_and_paths[0]
    names = [n for n in loader.var_names if not n.startswith("Branch")]
    serial = {n: loader.get_series(n).to_numpy().copy() for n in names}
    errors: list[BaseException] = []

    def worker(name):
        try:
            for _ in range(5):
                got = loader.get_series(name).to_numpy()
                if not np.array_equal(got, serial[name], equal_nan=True):
                    errors.append(AssertionError(f"{name} 并发读结果不一致"))
                loader.get_group_time_array(loader.get_var_group_index(name))
        except BaseException as e:  # noqa: BLE001 - 收集后统一断言
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in names * 2]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors[:3]
    loader.clear_cache()


def test_signal_cache_is_lru_bounded_and_clearable(loader_and_paths, monkeypatch):
    loader = loader_and_paths[0]
    monkeypatch.setattr(BLFLazyLoader, "MAX_CACHE_SIZE", 2)
    monkeypatch.setattr(BLFLazyLoader, "MAX_TIME_CACHE_SIZE", 1)
    loader.clear_cache()
    names = list(loader.var_names)
    for n in names:
        loader.get_series(n)
    assert len(loader._signal_cache) <= 2
    assert set(loader._signal_cache) == set(names[-2:])
    assert len(loader._time_cache) <= 1

    # release_memory 只清解码缓存，不动 payload（原始字节还得留着）
    counts = [b.count for b in loader._buckets]
    loader.release_memory()
    assert not loader._signal_cache
    assert [b.count for b in loader._buckets] == counts


def test_search_variables_and_aggregates(loader_and_paths):
    loader = loader_and_paths[0]
    hits = loader.search_variables("enginespeed")     # 大小写不敏感
    assert hits
    for name, gi, label in hits:
        assert gi == loader.get_var_group_index(name)
        assert label == loader.get_group_label(gi)
    assert loader.search_variables("no_such_signal") == []
    assert len(loader.search_variables("e", limit=2)) == 2
    assert loader.units[loader.var_names[0]] == loader.get_metadata(
        loader.var_names[0]).unit
    assert loader.column_count == len(loader.var_names)


def test_row_count_and_density_are_consistent(loader_and_paths):
    loader = loader_and_paths[0]
    biggest = max(b.count for b in loader._buckets)
    assert loader.datalength == biggest
    assert loader.max_row_count == loader.datalength
    assert loader.row_count == loader.datalength
    assert loader.message_count == loader.total_messages
    assert loader.baseline_density > 0
    assert loader.time_values.shape[0] >= 1
    assert len(loader.time_values) == loader.get_group_time_array(
        loader.get_var_group_index(loader.var_names[0])
    ).shape[0] or loader.datalength > 0
    assert loader.current_group_index == 0
    loader.current_group_index = 2
    assert loader.current_group_index == 2


def test_load_errors_surface_a_broken_second_db(loader_and_paths):
    loader, _blf, dbc = loader_and_paths
    stats = loader.bind_databases([str(dbc), "/nonexistent/old_fota.dbc"])
    assert stats["var_count"] > 0, "坏库不得拖垮好库"
    assert len(loader.load_errors) == 1
    assert "nonexistent" in loader.load_errors[0]
    loader.bind_databases([str(dbc)])
    assert loader.load_errors == []


# ---------------------------------------------------------------------------
# 跨块合并：payload 长度在扫描过程中增长
# ---------------------------------------------------------------------------

def test_payload_that_grows_between_blocks_is_padded_not_broken(tmp_path, monkeypatch):
    """同一桶里「前几帧短、后几帧长」时，窄历史块必须被补 0 合并。

    真实 BLF 里这很常见（CAN FD 的 dlc 逐帧变化、valid_bytes 小于实际数据长度
    的历史回归样本）。原实现直接把 ``block`` 赋给窄切片，会
    ``ValueError: could not broadcast``。这里把攒块阈值压到 2 行以复现多块。
    """
    monkeypatch.setattr(blf_module, "_BUCKET_BLOCK_ROWS", 2)
    rows = []
    for i in range(6):
        width = 2 if i < 3 else 8            # 前 3 帧 2 字节，后 3 帧 8 字节
        rows.append({
            "timestamp": T0 + i * 0.01, "channel": 0, "can_id": 0x1A0,
            "data": bytes(range(width)),
        })
    blf = write_blf(tmp_path / "grow.blf", rows)
    dbc = write_dbc(tmp_path / "grow.dbc")

    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        bucket = loader._buckets[0]
        assert bucket.stride == 8 and bucket.count == 6
        assert bucket.payload.shape == (6, 8)
        np.testing.assert_array_equal(bucket.payload[0], [0, 1] + [0] * 6)
        np.testing.assert_array_equal(bucket.payload[5], list(range(8)))
        name = next(n for n in loader.var_names if n.endswith("EngineSpeed"))
        values = loader.get_series(name).to_numpy()
        assert values.shape[0] == 6
        assert values[0] == pytest.approx(0x0100 * 0.125)
    finally:
        loader.close()


# ---------------------------------------------------------------------------
# 真实公开夹具：python-can 的每一种 CAN 对象类型
# ---------------------------------------------------------------------------

# 夹具名 -> 该文件里应能读出的帧数下界（0 表示「一个可用采样点都没有」）
@pytest.mark.parametrize("fixture_name,min_messages", [
    ("pcan_test_CanMessage.blf", 1),
    ("pcan_test_CanMessage2.blf", 1),
    ("pcan_test_CanFdMessage.blf", 1),
    ("pcan_issue_1905.blf", 1),
    ("pcan_test_CanErrorFrameExt.blf", 0),
    ("pcan_test_CanFdMessage64.blf", 0),
])
def test_real_python_can_fixtures_scan_or_fail_cleanly(tmp_path, fixture_name,
                                                       min_messages):
    """P2 验收：python-can 的 6 份真实 BLF 全部要能读出**正确类型**的对象。

    两份「0 帧」的（只有错误帧的那份、两条都是远程 FD 帧的那份）必须走
    「未包含任何 CAN 报文」而不是崩在 BLFReader 里 —— 这正是 §5.1 的 F 路径
    能判定「0 个信号」的前提。
    """
    path = bd.require(fixture_name)
    dbc = write_dbc(tmp_path / "any.dbc")
    if min_messages == 0:
        with pytest.raises(ValueError, match="未包含任何 CAN 报文"):
            BLFLazyLoader(str(path), db_paths=[str(dbc)])
        return
    loader = BLFLazyLoader(str(path), db_paths=[str(dbc)])
    try:
        assert loader.total_messages >= min_messages
        assert loader.file_size == path.stat().st_size
        assert loader.LOADER_TYPE == "blf"
    finally:
        loader.close()


def test_real_fd64_fixture_is_remote_frames_so_no_samples(tmp_path):
    """§2「远程帧跳过」的实测依据，钉住上游解析器的判定。

    ``pcan_test_CanFdMessage64.blf`` 看名字像「64 字节 FD 数据帧」，但它的
    ``fd_flags`` 里 0x0010（vblf 的 ``CanFdFlags.REMOTE_FRAME``）是置位的，
    ``valid_bytes`` 又是 0 —— 即它真是两条**远程** FD 帧，本就没有数据区。
    这条用例不参与解码断言，只负责「上游一改这个判定我们立刻知道」。
    """
    pytest.importorskip("can")
    path = bd.require("pcan_test_CanFdMessage64.blf")
    from can import BLFReader

    msgs = list(BLFReader(str(path)))
    assert msgs
    for m in msgs:
        assert m.is_fd and m.is_remote_frame and not m.is_error_frame
        assert len(bytes(m.data)) == 0
    dbc = write_dbc(tmp_path / "any.dbc")
    with pytest.raises(ValueError, match="未包含任何 CAN 报文"):
        BLFLazyLoader(str(path), db_paths=[str(dbc)])


def test_real_fd_64byte_bucket_keeps_the_full_stride(tmp_path):
    """CAN FD 64 B 决定桶 stride 上限：截到 8 字节会让所有高字节信号变 0。

    用 ``pcan_issue_1905.blf``（valid_bytes 大于实际可读数据的历史回归样本，
    读回 dlc=64 / 64 字节），而不是名字更像的 ``test_CanFdMessage64.blf`` ——
    后者其实是远程帧，见上一条用例。
    """
    path = bd.require("pcan_issue_1905.blf")
    dbc = write_dbc(tmp_path / "any.dbc")
    loader = BLFLazyLoader(str(path), db_paths=[str(dbc)])
    try:
        assert max(b.stride for b in loader._buckets) > 8
    finally:
        loader.close()


def test_real_two_channel_pair_end_to_end_produces_signals(tmp_path):
    """唯一的「真实 BLF + 配对其真实 DBC」样本：必须能出信号。"""
    blf = bd.require("blfviewer_two-channel.blf")
    dbc = bd.require("blfviewer_two-channel.dbc")
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        assert loader.var_names, "配对样本不该 0 信号"
        assert loader.group_count >= 1
        channels = {b.channel for b in loader._buckets}
        assert len(channels) >= 2, f"双通道样本应看到 2 条总线: {channels}"
        for name in loader.var_names[:5]:
            values = loader.get_series(name).to_numpy()
            assert values.shape[0] == loader._buckets[
                loader.get_var_group_index(name)].count
    finally:
        loader.close()


# ---------------------------------------------------------------------------
# 与 vblf 的逐帧交叉验证（§6.4 第 1 层）
# ---------------------------------------------------------------------------

def test_python_can_and_vblf_agree_frame_by_frame(tmp_path):
    """二进制层交叉：两个独立实现读同一份 BLF，逐帧一致即认为解析正确。

    ``vblf`` 是 dev 依赖；没装就 skip（不能让 CI 因可选依赖红）。它的
    ``BlfReader`` 只在 ``vblf.reader`` 里（顶层 __init__ 不导出任何类），
    递出来的是 BLF **对象**而非 python-can 的 ``Message``，所以字段名与
    口径都要换算：channel 保留 1 基、时间戳是相对文件头的纳秒整数、
    ``data`` 不按 dlc 截断（python-can 会 ``can_data[:dlc]``）。直接比对原始
    字节会把「谁更贴合规范」这个上游差异误报成解析错误。
    """
    pytest.importorskip("vblf")
    from can import BLFReader
    from vblf.reader import BlfReader

    blf, _dbc = write_blf_pair(tmp_path, n=30, buses=(0, 1))

    ours = [
        (m.timestamp, int(m.arbitration_id), bytes(m.data), int(m.channel or 0))
        for m in BLFReader(str(blf))
        if not m.is_error_frame and not m.is_remote_frame
    ]
    with BlfReader(str(blf)) as reader:
        theirs = [
            (o.header.object_time_stamp / 1e9, int(o.frame_id) & 0x1FFFFFFF,
             bytes(o.data)[: int(o.dlc)], int(o.channel) - 1)
            for o in reader
            if type(o).__name__.startswith("CanMessage")
        ]
    assert len(ours) == len(theirs) > 0
    base = ours[0][0]
    for a, b in zip(ours, theirs):
        assert a[1:] == b[1:], f"{a} != {b}"
        # BLF 时间戳有固定粒度（实测 0.01 s 存回来是 0.00999999），只能微秒级
        assert a[0] - base == pytest.approx(b[0], abs=1e-6), f"{a} != {b}"
