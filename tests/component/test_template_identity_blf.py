"""身份匹配 BLF 漂移语义测试（C3 / MJ-2 / MJ-4 / MN-6，真实 loader + 合成夹具）。

漂移触发用**注入低 ID unmatched 帧**（桶序位置整体后移）——反转帧序已
实证 no-op（gi 是 (channel,extended,can_id) 排序位置，非首现顺序）。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.plot_config import PlotConfig, PlotPageConfig, PlotSessionConfig
from src.core.template_identity import resolve_identities
from src.data.blf_loader import BLFLazyLoader
from src.ui.plot_config_manager import PlotConfigManager
from tests.fixtures.data_factory import (
    CAN_FRAMES,
    make_blf_messages,
    write_blf,
    write_blf_pair,
    write_dbc,
)

T0 = 1_700_000_000.0


def _load(path, dbc):
    return BLFLazyLoader(str(path), db_paths=[str(dbc)])


@pytest.fixture()
def blf_pair(tmp_path):
    """A：buses=(0,1)；B：同帧序 + 注入低 ID 0x010 unmatched 帧 → gi 整体 +1。"""
    blf_a, dbc = write_blf_pair(tmp_path, n=20, buses=(0, 1), basename="syn_a")
    msgs = make_blf_messages(T0, n=20, buses=(0, 1))
    for bus in (0, 1):
        msgs.append({
            "timestamp": T0 - 1.0, "channel": bus,
            "can_id": 0x010, "data": b"\x00" * 8,
        })
    msgs.sort(key=lambda r: r["timestamp"])
    blf_b = write_blf(tmp_path / "syn_b.blf", msgs)
    la, lb = _load(blf_a, dbc), _load(blf_b, dbc)
    yield la, lb
    la.close()
    lb.close()


def _template_from(loader, names):
    cfg = PlotSessionConfig(
        pages=[PlotPageConfig(plots=[PlotConfig(curves=list(names))])]
    )
    PlotConfigManager._collect_curve_identities(cfg, SimpleNamespace(loader=loader))
    return cfg


def test_fixture_suffix_drifted(blf_pair):
    la, lb = blf_pair
    assert "EngineData.EngineSpeed_G0" in la.var_names
    assert "EngineData.EngineSpeed_G0" not in lb.var_names
    assert "EngineData.EngineSpeed_G1" in lb.var_names


def test_remap_targets_correct_frame_and_factor(blf_pair):
    """语义断言：remap 目标是 EngineData 帧（factor 0.125），不是 BrakeData
    的同名信号（factor 1.0）；can_id/总线归属正确。"""
    la, lb = blf_pair
    cfg = _template_from(la, ["EngineData.EngineSpeed_G0"])
    entry = cfg.pages[0].plots[0].curve_identities["EngineData.EngineSpeed_G0"]
    assert entry["kind"] == "blf"
    assert entry["can_id"] == CAN_FRAMES[0]["can_id"]
    assert entry["bus_channel"] == 0

    resolved, remap, _demoted = resolve_identities(cfg, lb)
    assert remap == {"EngineData.EngineSpeed_G0": "EngineData.EngineSpeed_G1"}
    info = lb.get_channel_info("EngineData.EngineSpeed_G1")
    assert info["can"]["can_id"] == 0x1A0
    assert info["can"]["bus_channel_index"] == 0
    assert info["conversion"]["a"] == 0.125  # EngineData 的因子，非 BrakeData 1.0


def test_old_doc_qualified_multi_candidate_l3(blf_pair):
    """旧档无身份：限定基名索引 2 候选（两条总线）→ 不猜，落占位。"""
    _la, lb = blf_pair
    cfg = PlotSessionConfig(
        pages=[PlotPageConfig(plots=[PlotConfig(curves=["EngineData.EngineSpeed_G0"])])]
    )
    resolved, remap, _demoted = resolve_identities(cfg, lb)
    assert remap == {}
    assert resolved.pages[0].plots[0].curves == ["EngineData.EngineSpeed_G0"]


def test_old_doc_qualified_to_bare_unique(tmp_path):
    """旧档限定名 → 当前裸名唯一（BrakeData 帧移除后 EngineSpeed 全局唯一）。"""
    blf_a, dbc_a = write_blf_pair(tmp_path, n=20, buses=(0,), basename="syn_a")
    frames_b = tuple(f for f in CAN_FRAMES if f["name"] != "BrakeData")
    dbc_b = write_dbc(tmp_path / "syn_b.dbc", frames_b)
    msgs = make_blf_messages(T0, n=20, frames=frames_b, buses=(0,))
    blf_b = write_blf(tmp_path / "syn_b.blf", msgs)
    la, lb = _load(blf_a, dbc_a), _load(blf_b, dbc_b)
    try:
        assert "EngineData.EngineSpeed" in la.var_names
        assert "EngineSpeed" in lb.var_names
        cfg = PlotSessionConfig(
            pages=[PlotPageConfig(plots=[PlotConfig(curves=["EngineData.EngineSpeed"])])]
        )
        resolved, remap, _demoted = resolve_identities(cfg, lb)
        assert remap == {"EngineData.EngineSpeed": "EngineSpeed"}
        assert resolved.pages[0].plots[0].curves == ["EngineSpeed"]
    finally:
        la.close()
        lb.close()


def test_bus_renumber_subset_multi_candidate_fail_closed(tmp_path):
    """MN-6 风险场景 + M-1 方案乙：buses=(0,1)→(1,2) 全改号，_G0 同名仍在
    但总线归属矛盾 → D9 检出、L1 四元组失配、L2 多候选 → 降占位不绘制
    （方案甲时照画错总线数据，方案乙收口），走 C4 手动替换。"""
    blf_a, dbc = write_blf_pair(tmp_path, n=20, buses=(0, 1), basename="syn_a")
    msgs = make_blf_messages(T0, n=20, buses=(1, 2))
    blf_b = write_blf(tmp_path / "syn_b.blf", msgs)
    la, lb = _load(blf_a, dbc), _load(blf_b, dbc)
    try:
        cfg = _template_from(la, ["EngineData.EngineSpeed_G0"])
        resolved, remap, demoted = resolve_identities(cfg, lb)
        assert remap == {}
        assert demoted == ["EngineData.EngineSpeed_G0"]
        p = resolved.pages[0].plots[0]
        assert p.curves == []
        assert p.invalid_curves == ["EngineData.EngineSpeed_G0"]
    finally:
        la.close()
        lb.close()