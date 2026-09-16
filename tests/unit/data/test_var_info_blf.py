"""BLF 侧的变量信息快照与统计（方案 §5.7 + §8 的 unit 层）。

这里刻意**不重复** ``test_blf_loader.py`` 已经钉过的解码数值，只管三件事：

1. **分派**：``kind == "blf"`` 必须走 ``_from_blf`` / ``_stats_blf``，并且
   既不能漏掉 CAN 语义分区，也不能把 MDF 专有的块名/行名带进来 —— BLF 复用
   MDF 的快照骨架是刻意的省事，但一旦骨架里的 MDF 术语漏出来，用户看到的
   就是「CCBLOCK」「通道组」这种他文件里根本不存在的东西。
2. **「CAN 报文」分区的每一行**（§5.7 的两列键/值清单）。
3. **统计的流式路径**：分块累加、取消、关闭后不抛异常、全 NaN 分支。
"""

from __future__ import annotations

import numpy as np
import pytest

import src.core.config as config_module
from src.data import var_info
from src.data.blf_loader import BLFLazyLoader
from src.data.metadata import UNKNOWN
from tests.fixtures.data_factory import (
    CAN_FRAMES,
    encode_frame_payload,
    write_blf,
    write_blf_pair,
    write_dbc,
)

T0 = 1_700_000_000.0

#: 「CAN 报文」分区应有的行（§5.7 的键列，顺序不敏感、内容必须齐）
CAN_ROW_LABELS = [
    "报文名", "CAN ID", "总线通道", "DLC (字节)", "起始位 (DBC)", "位长",
    "字节序", "有符号", "浮点信号", "factor", "offset", "周期（实测均值）",
    "接收节点", "多路复用", "数据库文件",
]

#: 只属于 MDF 的分区名 / 行名，BLF 快照里出现即为回归
MDF_ONLY_TOKENS = [
    "转换规则 (CCBLOCK)", "文件信息 (HDBLOCK)", "枚举映射 (CCBLOCK)",
    "归属信息", "MDF 版本", "通道组数", "CCBLOCK", "HDBLOCK", "CGBLOCK",
]


@pytest.fixture(scope="module")
def blf_loader(tmp_path_factory):
    """双总线样本：同时覆盖「同 ID 多通道」与三段式改名后的显示名。"""
    directory = tmp_path_factory.mktemp("varinfo_blf")
    blf, dbc = write_blf_pair(directory, n=24, buses=(0, 1))
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    yield loader, blf, dbc
    loader.close()


def _name(loader, needle, frame=None):
    """按**裸信号名**找显示名。

    刻意不拿 ``endswith`` 碰运气：§2 的命名是三段式的（裸名唯一就用裸名），
    单库/多库、单报文/多报文下得到的名字不同，而
    ``metadata.original_name`` 永远是裸名 —— 那才是可以跨场景用的锚。
    """
    hits = [n for n in loader.var_names
            if loader.get_metadata(n).original_name == needle]
    if frame is not None:
        hits = [n for n in hits
                if loader.get_channel_info(n)["can"]["message_name"] == frame]
    assert hits, f"没有叫 {needle} 的变量（frame={frame}）: {loader.var_names}"
    return hits[0]


def _rows(snap, section):
    assert section in snap.sections, list(snap.sections)
    return dict(snap.sections[section])


# ---------------------------------------------------------------------------
# 分派与骨架复用
# ---------------------------------------------------------------------------

def test_dispatch_sets_source_kind_for_every_variable(blf_loader):
    loader, _blf, _dbc = blf_loader
    assert loader.var_names, "样本必须能出信号"
    for name in loader.var_names:
        snap = var_info.build_snapshot(loader, name, generation=3)
        assert snap.source_kind == "blf"
        assert snap.generation == 3
        assert snap.name == name
        assert snap.length == loader.get_metadata(name).sample_count
        assert snap.unit == loader.units[name]
        assert snap.validity == UNKNOWN
        assert snap.is_numeric is True


def test_no_mdf_only_block_names_leak_into_a_blf_snapshot(blf_loader):
    """复用骨架的代价必须被看住：MDF 术语一个都不能出现在 BLF 快照里。"""
    loader, _blf, _dbc = blf_loader
    for name in loader.var_names:
        snap = var_info.build_snapshot(loader, name)
        text = var_info.snapshot_to_markdown(snap)
        for token in MDF_ONLY_TOKENS:
            assert token not in snap.sections, f"{name} 多出分区 {token}"
            assert token not in text, f"{name} 的导出里漏出了 {token}"


def test_can_section_sits_right_after_the_basic_section(blf_loader):
    """§5.7：「CAN 报文」紧跟「基本信息」，在转换规则之前。"""
    loader, _blf, _dbc = blf_loader
    snap = var_info.build_snapshot(
        loader, _name(loader, "EngineSpeed")
    )
    order = list(snap.sections)
    assert order[0] == "基本信息"
    assert order[1] == "CAN 报文"
    assert order.index("CAN 报文") < order.index("转换规则")
    assert "文件信息" in order


def test_rename_suffix_is_explained_rather_than_mysteriously_appended(blf_loader):
    loader, _blf, _dbc = blf_loader
    name = _name(loader, "EngineSpeed")
    basic = _rows(var_info.build_snapshot(loader, name), "基本信息")
    assert basic["变量名（聚合显示名）"] == name
    assert basic["原始通道名"] == "EngineSpeed"
    # 行文案沿用 MDF 侧的「跨通道组重名」措辞：BLF 的组 = 报文桶，同一套
    # 骨架下这句话依旧成立，刻意不另开一份文案以免两处漂移。
    assert "名称改写原因" in basic
    assert "组" in basic["名称改写原因"]


# ---------------------------------------------------------------------------
# 「CAN 报文」分区（§5.7 的两列清单）
# ---------------------------------------------------------------------------

def test_can_section_carries_every_planned_row(blf_loader):
    loader, _blf, dbc = blf_loader
    snap = var_info.build_snapshot(loader, _name(loader, "EngineSpeed"))
    rows = _rows(snap, "CAN 报文")
    missing = [k for k in CAN_ROW_LABELS if k not in rows]
    assert not missing, f"CAN 分区缺行: {missing}"
    assert list(rows) == CAN_ROW_LABELS, "行序即 UI 展示序，不该随意变动"


def test_can_section_values_for_an_intel_signal(blf_loader):
    loader, _blf, dbc = blf_loader
    rows = _rows(
        var_info.build_snapshot(loader, _name(loader, "EngineSpeed",
                                              frame="EngineData")),
        "CAN 报文",
    )
    assert rows["报文名"] == "EngineData"
    assert rows["CAN ID"] == "0x1A0"
    assert rows["总线通道"] == "0"
    assert rows["DLC (字节)"] == "8"
    assert rows["起始位 (DBC)"] == "0"
    assert rows["位长"] == "16 bit"
    assert rows["字节序"] == "Intel（小端）"
    assert rows["有符号"] == "否"
    assert rows["浮点信号"] == "否"
    assert rows["factor"] == "0.125"
    assert rows["offset"] == "0"
    assert rows["数据库文件"] == str(dbc)


def test_can_section_marks_float_and_signed_signals(blf_loader):
    loader, _blf, _dbc = blf_loader
    flt = _rows(
        var_info.build_snapshot(loader, _name(loader, "TorqueLimit",
                                              frame="EngineData")),
        "CAN 报文",
    )
    assert flt["浮点信号"] == "是"
    assert flt["位长"] == "32 bit"

    sgn = _rows(
        var_info.build_snapshot(loader, _name(loader, "CoolantTemp",
                                              frame="EngineData")),
        "CAN 报文",
    )
    assert sgn["有符号"] == "是"
    assert sgn["浮点信号"] == "否"
    assert sgn["offset"] == "-40"


def test_motorola_signal_shows_the_dbc_number_not_the_internal_one(blf_loader):
    """用户拿 CANoe / DBC 编辑器对照时必须对得上号（§5.7）。"""
    loader, _blf, _dbc = blf_loader
    name = _name(loader, "WheelSpeedFL", frame="ChassisData")
    rows = _rows(var_info.build_snapshot(loader, name), "CAN 报文")
    assert rows["起始位 (DBC)"] == "7"
    assert rows["字节序"] == "Motorola（大端）"
    # 内部位号（解码器真正用的）仍留在 get_channel_info 里供排错
    internal = loader.get_channel_info(name)["can"]["start_bit_internal"]
    assert internal == 0
    assert internal != int(rows["起始位 (DBC)"])


def test_receivers_are_listed_when_the_dbc_declares_them(blf_loader):
    loader, _blf, _dbc = blf_loader
    named = _name(loader, "WheelSpeedFL", frame="ChassisData")
    rows = _rows(var_info.build_snapshot(loader, named), "CAN 报文")
    assert rows["接收节点"] == "Cluster, DiagTool"
    bare = _name(loader, "EngineSpeed", frame="EngineData")
    assert _rows(var_info.build_snapshot(loader, bare), "CAN 报文")["接收节点"] == "-"


def test_mux_rows_read_as_sentences(blf_loader):
    loader, _blf, _dbc = blf_loader
    sel = _name(loader, "MuxSel", frame="DiagMux")
    branch_a = _name(loader, "BranchA", frame="DiagMux")
    plain = _name(loader, "EngineSpeed", frame="EngineData")
    can = lambda n: _rows(var_info.build_snapshot(loader, n), "CAN 报文")  # noqa: E731
    assert "选择器" in can(sel)["多路复用"]
    assert "==" in can(branch_a)["多路复用"]
    # 非多路复用帧的信号不留空占位，但必须是「-」而不是句子
    assert can(plain)["多路复用"] == "-"


def test_period_row_is_the_measured_mean(blf_loader):
    """§5.7 明文「周期用实测 mean_cycle_s」，不是 DBC 声明值。

    双总线样本里 EngineData 每 0.08 s 才轮到一次（声明 0.02 s），两个读数
    不同才能证明这两行不是同一个值的两次复读。
    """
    loader, _blf, _dbc = blf_loader
    name = _name(loader, "EngineSpeed", frame="EngineData")
    snap = var_info.build_snapshot(loader, name)
    assert _rows(snap, "CAN 报文")["周期（实测均值）"] == "0.08 s"
    basic = _rows(snap, "基本信息")
    assert basic["标称采样间隔"].startswith("0.02 s"), basic["标称采样间隔"]
    assert "50 Hz" in basic["标称采样间隔"], basic["标称采样间隔"]
    assert basic["有效采样率"].startswith("12.5 Hz"), basic["有效采样率"]


def test_the_period_label_names_where_the_number_came_from():
    """算不出均值时标题必须跟着换（§5.7 的诚实性）。

    单帧桶（偶发帧 / 触发型报文）没有实测均值可言，数据层会退到 DBC 的
    GenMsgCycleTime。两个读数本该能互相揭示偏差，把声明值挂在「实测」
    标题下等于把注释当成测量结果。
    """
    declared = dict(var_info._can_message_rows(
        {"cycle_time_s": 0.02, "cycle_time_measured": False}))
    assert declared["周期（数据库声明）"] == "0.02 s"
    assert "周期（实测均值）" not in declared

    measured = dict(var_info._can_message_rows(
        {"cycle_time_s": 0.04, "cycle_time_measured": True}))
    assert measured["周期（实测均值）"] == "0.04 s"
    assert "周期（数据库声明）" not in measured

    # 两边都没有：标题不带来源定语，不给一个「-」凭空冠上含义
    none = dict(var_info._can_message_rows({"cycle_time_s": None}))
    assert none["周期"] == "-"


# ---------------------------------------------------------------------------
# 转换规则 / 文件信息 / 枚举
# ---------------------------------------------------------------------------

def test_conversion_section_reuses_factor_and_offset(blf_loader):
    loader, _blf, _dbc = blf_loader
    conv = _rows(var_info.build_snapshot(loader, _name(loader, "EngineSpeed")),
                 "转换规则")
    assert conv["线性系数 a"] == "0.125"
    assert conv["线性系数 b"] == "0"
    assert conv["物理单位"] == "rpm"


def test_file_section_reports_blf_wording_and_real_numbers(blf_loader):
    loader, blf, _dbc = blf_loader
    fi = _rows(var_info.build_snapshot(loader, _name(loader, "EngineSpeed")),
               "文件信息")
    assert fi["数据格式"] == "BLF"
    assert fi["文件路径"].endswith("syn_can.blf")
    assert fi["文件大小"] != "-"
    assert fi["报文桶数"] == str(loader.group_count)
    assert fi["变量总数"] == str(len(loader.var_names))
    # 绝对起始时间只活在这一行（§2），数值轴始终是归零的相对秒
    assert fi["起始时间"].startswith("2023")


def test_enum_section_uses_the_blf_title_and_the_val_table(blf_loader):
    loader, _blf, _dbc = blf_loader
    enum_var = _name(loader, "StarterMode", frame="EngineData")
    snap = var_info.build_snapshot(loader, enum_var)
    assert snap.is_enum is True
    assert "枚举映射" in snap.sections
    assert _rows(snap, "枚举映射") == {"0": "off", "1": "run", "2": "error"}
    plain = var_info.build_snapshot(
        loader, _name(loader, "EngineSpeed", frame="EngineData")
    )
    assert "枚举映射" not in plain.sections


def test_markdown_export_contains_the_can_section(blf_loader):
    loader, _blf, _dbc = blf_loader
    snap = var_info.build_snapshot(loader, _name(loader, "EngineSpeed"))
    stats = var_info.compute_stats(loader, snap.name)
    text = var_info.snapshot_to_markdown(snap, stats)
    assert "### CAN 报文" in text
    assert "| 起始位 (DBC) | 0 |" in text
    assert "| 最小值 |" in text


# ---------------------------------------------------------------------------
# 统计（§4 的 ``_stats_blf``：流式累加 + 取消 + 关闭兜底）
# ---------------------------------------------------------------------------

def test_stats_match_numpy_over_the_same_series(blf_loader):
    loader, _blf, _dbc = blf_loader
    name = _name(loader, "EngineSpeed")
    stats = var_info.compute_stats(loader, name)
    values = loader.get_series(name).to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    assert stats.computed is True
    assert stats.min == pytest.approx(float(finite.min()))
    assert stats.max == pytest.approx(float(finite.max()))
    assert stats.mean == pytest.approx(float(finite.mean()))
    assert stats.std == pytest.approx(float(finite.std()))
    assert stats.finite_count == finite.size
    assert stats.nan_count == int(np.isnan(values).sum())
    assert stats.cancelled is False
    assert stats.note == ""


def test_stats_accumulate_identically_across_chunk_boundaries(blf_loader,
                                                              monkeypatch):
    """分块只该改变读取粒度，不该改变结果（float64 累加的口径一致）。"""
    loader, _blf, _dbc = blf_loader
    name = _name(loader, "EngineSpeed")
    whole = var_info.compute_stats(loader, name)
    monkeypatch.setattr(config_module, "MDF_STATS_CHUNK_SIZE", 1)
    one_by_one = var_info.compute_stats(loader, name)
    assert one_by_one.computed is True
    for field in ("min", "max", "mean", "std", "finite_count", "nan_count"):
        assert getattr(one_by_one, field) == pytest.approx(getattr(whole, field)), \
            field


def test_mixed_sign_and_enum_signals_are_counted_with_nan_separate(blf_loader):
    """有符号走负值、枚举走物理值：两条都是「统计≠绘图」的口径差异点。"""
    loader, _blf, _dbc = blf_loader
    coolant = var_info.compute_stats(loader, _name(loader, "CoolantTemp"))
    assert coolant.computed is True and coolant.min < 0

    enum_var = _name(loader, "StarterMode")
    stats = var_info.compute_stats(loader, enum_var)
    drawn = loader.get_series(enum_var).to_numpy()
    assert stats.finite_count == drawn.size
    assert stats.min == pytest.approx(float(drawn.min()))


def test_cancelled_stats_are_flagged_so_the_cache_refuses_them(blf_loader,
                                                              monkeypatch):
    loader, _blf, _dbc = blf_loader
    name = _name(loader, "EngineSpeed")
    seen = {"n": 0}

    def cancel(_var):
        seen["n"] += 1
        return seen["n"] >= 2

    # 默认块大于单桶行数时 ``should_cancel`` 只会被问一次，取消路径根本没被
    # 走过却照样「没算完」—— 压成 1 行一块才能真正确信是取消导致的中断。
    monkeypatch.setattr(config_module, "MDF_STATS_CHUNK_SIZE", 1)
    stats = var_info.compute_stats(loader, name, should_cancel=cancel)
    assert stats.cancelled is True
    assert stats.error == "已取消"
    assert stats.computed is False
    assert seen["n"] >= 2


def test_stats_on_a_closed_loader_become_an_error_row(tmp_path):
    """后台统计线程与 close() 有竞态：必须把 KeyError 折成可展示的 error。"""
    blf, dbc = write_blf_pair(tmp_path, n=12, buses=(0,))
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    name = _name(loader, "EngineSpeed")
    loader.close()
    stats = var_info.compute_stats(loader, name)
    assert stats.computed is False
    assert stats.error and "关闭" in stats.error


def test_stats_for_an_unknown_variable(blf_loader):
    loader, _blf, _dbc = blf_loader
    stats = var_info.compute_stats(loader, "no_such_signal")
    assert stats.computed is False
    assert "不存在" in stats.error


def test_a_branch_that_is_never_selected_reports_all_nan(tmp_path):
    """mux 恒选 1 ⇒ BranchA 全 NaN：不能出「min/max 都 0」的假结果。"""
    diag = next(f for f in CAN_FRAMES if f["name"] == "DiagMux")
    rows = [
        {
            "timestamp": T0 + i * 0.05, "channel": 0, "can_id": 0x3B0,
            "data": encode_frame_payload(diag, {"MuxSel": 1.0, "BranchB": 3.0}),
        }
        for i in range(6)
    ]
    blf = write_blf(tmp_path / "mux1.blf", rows)
    dbc = write_dbc(tmp_path / "mux1.dbc")
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc)])
    try:
        branch_a = _name(loader, "BranchA", frame="DiagMux")
        assert np.isnan(loader.get_series(branch_a).to_numpy()).all()
        stats = var_info.compute_stats(loader, branch_a)
        assert stats.computed is False
        assert "全部为 NaN" in stats.error
        assert stats.nan_count == 6

        branch_b = _name(loader, "BranchB", frame="DiagMux")
        ok = var_info.compute_stats(loader, branch_b)
        assert ok.computed is True and ok.nan_count == 0
    finally:
        loader.close()


def test_stats_survive_a_database_rebind_without_stale_factors(tmp_path):
    """换库后统计必须按新库的 factor 重算（缓存清没清，这里就见分晓）。"""
    blf, dbc_a = write_blf_pair(tmp_path, n=8, buses=(0,))
    engine = CAN_FRAMES[0]
    dbc_b = write_dbc(
        tmp_path / "alt.dbc",
        [{**engine, "signals": [
            {**s, "factor": 2.0} if s["name"] == "EngineSpeed" else s
            for s in engine["signals"]
        ]}],
    )
    loader = BLFLazyLoader(str(blf), db_paths=[str(dbc_a)])
    try:
        name = _name(loader, "EngineSpeed")
        before = var_info.compute_stats(loader, name)
        loader.bind_databases([str(dbc_b)])
        name = _name(loader, "EngineSpeed")
        after = var_info.compute_stats(loader, name)
        assert after.max == pytest.approx(before.max * 2.0 / 0.125)
        rows = _rows(var_info.build_snapshot(loader, name), "CAN 报文")
        assert rows["factor"] == "2"
    finally:
        loader.close()
