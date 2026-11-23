"""身份匹配 MDF 组序对调语义测试（C3 / BL-1 / MJ-9，真实 loader + 合成夹具）。

不只断言名字命中：remap 目标必须单位/设备归属正确（画的是 bar 那条
Press，而不是同名 degC 那条）。SI 有/无两态夹具显式构造。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.plot_config import PlotConfig, PlotPageConfig, PlotSessionConfig
from src.core.template_identity import resolve_identities
from src.data.mdf_lazy_loader import MDFLazyLoader
from src.ui.plot_config_manager import PlotConfigManager
from tests.fixtures.data_factory import write_mdf


@pytest.fixture()
def mdf_pair(tmp_path):
    """A：主组在前（Press_G0=bar/Dev1）；B：组序对调（Press_G0=degC/Dev2）。"""
    a = write_mdf(
        tmp_path / "syn_a.dat", version="4.10", n=10,
        with_enum=False, with_string=False, with_dup_group=True, press_si=True,
    )
    b = write_mdf(
        tmp_path / "syn_b.dat", version="4.10", n=10,
        with_enum=False, with_string=False, with_dup_group=True, press_si=True,
        dup_group_first=True,
    )
    la, lb = MDFLazyLoader(str(a)), MDFLazyLoader(str(b))
    yield la, lb
    la.close()
    lb.close()


def _template_from(loader, names):
    cfg = PlotSessionConfig(
        pages=[PlotPageConfig(plots=[PlotConfig(curves=list(names))])]
    )
    PlotConfigManager._collect_curve_identities(cfg, SimpleNamespace(loader=loader))
    return cfg


def test_fixture_group_order_swapped(mdf_pair):
    la, lb = mdf_pair
    assert la.get_channel_info("Press_G0")["meta"].unit == "bar"
    assert lb.get_channel_info("Press_G0")["meta"].unit == "degC"
    assert lb.get_channel_info("Press_G1")["meta"].unit == "bar"


def test_remap_targets_correct_unit_and_attribution(mdf_pair):
    """BL-1 语义断言：Press_G0→Press_G1，且目标是 bar/Dev1/ECU_A 那条。"""
    la, lb = mdf_pair
    cfg = _template_from(la, ["Press_G0"])
    entry = cfg.pages[0].plots[0].curve_identities["Press_G0"]
    assert entry["device"] == "Dev1"
    assert entry["ecu"] == "ECU_A"
    assert entry["unit"] == "bar"

    resolved, remap = resolve_identities(cfg, lb)
    assert remap == {"Press_G0": "Press_G1"}
    assert resolved.pages[0].plots[0].curves == ["Press_G1"]
    info = lb.get_channel_info("Press_G1")
    assert info["meta"].unit == "bar"
    assert info["source"]["path"] == "Dev1"
    assert info["source"]["name"] == "ECU_A"


def test_si_absent_state_collected_without_fingerprint(tmp_path):
    """MJ-9 两态：press_si=False → 无 device/ecu（伪影源被过滤）。"""
    p = write_mdf(
        tmp_path / "syn_nosi.dat", version="4.10", n=10,
        with_enum=False, with_string=False, with_dup_group=True, press_si=False,
    )
    loader = MDFLazyLoader(str(p))
    try:
        cfg = _template_from(loader, ["Press_G0"])
        entry = cfg.pages[0].plots[0].curve_identities["Press_G0"]
        assert entry["original_name"] == "Press"
        assert "device" not in entry
        assert "ecu" not in entry
    finally:
        loader.close()


def test_si_absent_swapped_multi_candidate_l3(tmp_path):
    """无 SI + 组序对调：L1 无指纹、L2 多候选 → 落占位（不猜，D4）。"""
    a = write_mdf(
        tmp_path / "syn_a.dat", version="4.10", n=10,
        with_enum=False, with_string=False, with_dup_group=True,
    )
    b = write_mdf(
        tmp_path / "syn_b.dat", version="4.10", n=10,
        with_enum=False, with_string=False, with_dup_group=True,
        dup_group_first=True,
    )
    la, lb = MDFLazyLoader(str(a)), MDFLazyLoader(str(b))
    try:
        cfg = _template_from(la, ["Press_G2"])  # 两边都不存在的旧后缀
        resolved, remap = resolve_identities(cfg, lb)
        assert remap == {}
        assert resolved.pages[0].plots[0].curves == ["Press_G2"]
    finally:
        la.close()
        lb.close()


def test_si_absent_swapped_original_still_present_residual(tmp_path):
    """终审 M-1 护栏（已披露残余风险，fail-closed 属新决策留用户拍板）：

    无 SI + 组序对调 + 原名仍在 + 单位矛盾：D9 检出矛盾降级，但 L1 无
    指纹不命中、L2 多候选不猜（D4/D10）→ L3 保持原名，套用端见原名
    仍在照画 degC 那条——「检而不防」，与无引擎现状一致、不劣于现状。
    本测试显式钉住该行为为残余风险，而非安全子例（Press_G2 不存在）。
    """
    a = write_mdf(
        tmp_path / "syn_a.dat", version="4.10", n=10,
        with_enum=False, with_string=False, with_dup_group=True,
    )
    b = write_mdf(
        tmp_path / "syn_b.dat", version="4.10", n=10,
        with_enum=False, with_string=False, with_dup_group=True,
        dup_group_first=True,
    )
    la, lb = MDFLazyLoader(str(a)), MDFLazyLoader(str(b))
    try:
        cfg = _template_from(la, ["Press_G0"])
        entry = cfg.pages[0].plots[0].curve_identities["Press_G0"]
        assert entry["unit"] == "bar"
        assert "device" not in entry  # 无 SI 指纹
        resolved, remap = resolve_identities(cfg, lb)
        # 残余风险行为：不改名、不告警播报（remap=0），原名仍在 → 照画
        assert remap == {}
        assert resolved.pages[0].plots[0].curves == ["Press_G0"]
        assert lb.get_channel_info("Press_G0")["meta"].unit == "degC"
    finally:
        la.close()
        lb.close()


def test_old_doc_l0_residual_risk(mdf_pair):
    """D9 披露：旧档无身份条目，同名换内容不可检测，L0 维持现状。"""
    _la, lb = mdf_pair
    cfg = PlotSessionConfig(
        pages=[PlotPageConfig(plots=[PlotConfig(curves=["Press_G0"])])]
    )
    resolved, remap = resolve_identities(cfg, lb)
    assert remap == {}
    assert resolved.pages[0].plots[0].curves == ["Press_G0"]


def test_v3_artifact_source_not_collected(tmp_path):
    """MJ-9：合成 v3 被 asammdf 注入伪影 source.name，采集侧视为空。"""
    p = write_mdf(
        tmp_path / "syn_v3.dat", version="3.30", n=10,
        with_enum=False, with_string=False, with_dup_group=True, press_si=True,
    )
    loader = MDFLazyLoader(str(p))
    try:
        cfg = _template_from(loader, ["Press_G0"])
        entry = cfg.pages[0].plots[0].curve_identities["Press_G0"]
        assert entry["kind"] == "mdf"
        assert "ecu" not in entry
        assert "device" not in entry
    finally:
        loader.close()