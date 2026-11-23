"""template_identity 匹配引擎单测（身份匹配 C2 / D4/D6/D9/D10/D12）。

全部用合成 loader 替身（var_names + get_channel_info），零 Qt、零磁盘。
info dict 形状与 MDFLazyLoader/BLFLazyLoader.get_channel_info 同构。
"""

from types import SimpleNamespace

from src.core.plot_config import PlotConfig, PlotPageConfig, PlotSessionConfig
from src.core.template_identity import (
    identity_from_channel_info,
    resolve_identities,
)


def mdf_info(original_name, unit="", device=None, ecu=None):
    return {
        "meta": SimpleNamespace(original_name=original_name, unit=unit),
        "source": {"path": device, "name": ecu},
    }


def blf_info(original_name, frame, can_id, *, unit="", extended=False, bus=0):
    return {
        "meta": SimpleNamespace(original_name=original_name, unit=unit),
        "can": {
            "message_name": frame,
            "can_id": can_id,
            "is_extended": extended,
            "bus_channel_index": bus,
        },
    }


class FakeLoader:
    def __init__(self, infos: dict):
        self._infos = infos
        self.var_names = list(infos)

    def get_channel_info(self, name):
        return self._infos[name]


def session(*plots, v2=True):
    plots = list(plots)
    if v2:
        return PlotSessionConfig(
            pages=[PlotPageConfig(plots=list(plots))],
            plots=[
                PlotConfig.from_dict(p.to_dict()) for p in plots
            ],  # from_dict 口径：顶层与 pages 独立对象
        )
    return PlotSessionConfig(plots=plots)


def mdf_identity(original_name, unit=None, device=None, ecu=None):
    entry = {"original_name": original_name, "kind": "mdf"}
    if unit:
        entry["unit"] = unit
    if device:
        entry["device"] = device
    if ecu:
        entry["ecu"] = ecu
    return entry


def blf_identity(original_name, frame, can_id, *, unit=None, extended=False, bus=0):
    entry = {
        "original_name": original_name, "kind": "blf", "frame": frame,
        "can_id": can_id, "extended": extended, "bus_channel": bus,
    }
    if unit:
        entry["unit"] = unit
    return entry


class TestL0:
    def test_identity_hit_no_remap(self):
        cfg = session(PlotConfig(curves=["Press_G0", "Speed"]))
        loader = FakeLoader({
            "Press_G0": mdf_info("Press", "bar"),
            "Press_G1": mdf_info("Press", "degC"),
            "Speed": mdf_info("Speed", "km/h"),
        })
        resolved, remap = resolve_identities(cfg, loader)
        assert remap == {}
        assert resolved.pages[0].plots[0].curves == ["Press_G0", "Speed"]

    def test_old_doc_without_identity_keeps_l0(self):
        """D9 残余风险披露：旧档无身份条目，同名换内容不可检测，维持现状。"""
        cfg = session(PlotConfig(curves=["Press_G0"]))
        loader = FakeLoader({"Press_G0": mdf_info("Press", "degC")})
        resolved, remap = resolve_identities(cfg, loader)
        assert remap == {}
        assert resolved.pages[0].plots[0].curves == ["Press_G0"]

    def test_consistent_identity_keeps_l0(self):
        cfg = session(PlotConfig(
            curves=["Press_G0"],
            curve_identities={"Press_G0": mdf_identity("Press", "bar", "Dev1", "ECU_A")},
        ))
        loader = FakeLoader({"Press_G0": mdf_info("Press", "bar", "Dev1", "ECU_A")})
        _resolved, remap = resolve_identities(cfg, loader)
        assert remap == {}


class TestD9:
    def _swapped_loader(self):
        # BL-1 场景：组序对调后 Press_G0 同名仍在但单位 bar↔degC
        return FakeLoader({
            "Press_G0": mdf_info("Press", "degC", "Dev2", "ECU_B"),
            "Press_G1": mdf_info("Press", "bar", "Dev1", "ECU_A"),
        })

    def test_unit_contradiction_demotes_and_l1_finds_correct(self):
        cfg = session(PlotConfig(
            curves=["Press_G0"],
            curve_identities={"Press_G0": mdf_identity("Press", "bar", "Dev1", "ECU_A")},
        ))
        resolved, remap = resolve_identities(cfg, self._swapped_loader())
        assert remap == {"Press_G0": "Press_G1"}
        assert resolved.pages[0].plots[0].curves == ["Press_G1"]

    def test_original_name_mismatch_demotes(self):
        cfg = session(PlotConfig(
            curves=["X"],
            curve_identities={"X": mdf_identity("Press", "bar", device="D")},
        ))
        loader = FakeLoader({
            "X": mdf_info("Other", "bar", "D"),
            "Y": mdf_info("Press", "bar", "D"),
        })
        resolved, remap = resolve_identities(cfg, loader)
        assert remap == {"X": "Y"}
        assert resolved.pages[0].plots[0].curves == ["Y"]

    def test_fingerprint_contradiction_demotes(self):
        cfg = session(PlotConfig(
            curves=["Press_G0"],
            curve_identities={"Press_G0": mdf_identity("Press", "bar", "Dev1")},
        ))
        loader = FakeLoader({
            "Press_G0": mdf_info("Press", "bar", "Dev9"),
            "Press_G1": mdf_info("Press", "bar", "Dev1"),
        })
        _resolved, remap = resolve_identities(cfg, loader)
        assert remap == {"Press_G0": "Press_G1"}

    def test_blf_quadruple_contradiction_demotes(self):
        cfg = session(PlotConfig(
            curves=["EngineData.EngineSpeed_G0"],
            curve_identities={
                "EngineData.EngineSpeed_G0": blf_identity("EngineSpeed", "EngineData", 0x1A0, bus=0)
            },
        ))
        loader = FakeLoader({
            "EngineData.EngineSpeed_G0": blf_info("EngineSpeed", "EngineData", 0x1A0, bus=1),
            "EngineData.EngineSpeed_G1": blf_info("EngineSpeed", "EngineData", 0x1A0, bus=0),
        })
        _resolved, remap = resolve_identities(cfg, loader)
        assert remap == {"EngineData.EngineSpeed_G0": "EngineData.EngineSpeed_G1"}

    def test_demote_unique_candidate_l2_reselects_self(self):
        """D10 字面口径：D9 降级后 L2 唯一候选仍可回选自身（无更优候选时
        维持现状绘制），不构成改名、不计入 remap。"""
        cfg = session(PlotConfig(
            curves=["Press"],
            curve_identities={"Press": mdf_identity("Press", "bar")},
        ))
        loader = FakeLoader({"Press": mdf_info("Press", "degC")})
        resolved, remap = resolve_identities(cfg, loader)
        assert remap == {}
        assert resolved.pages[0].plots[0].curves == ["Press"]

    def test_empty_units_not_compared(self):
        """双方单位均非空才比对：'-' 与 '' 都视为空。"""
        cfg = session(PlotConfig(
            curves=["Press"],
            curve_identities={"Press": mdf_identity("Press")},
        ))
        loader = FakeLoader({"Press": mdf_info("Press", "-")})
        _resolved, remap = resolve_identities(cfg, loader)
        assert remap == {}


class TestL1Mdf:
    def test_unique_fingerprint_hit(self):
        cfg = session(PlotConfig(
            curves=["Press_G0"],
            curve_identities={"Press_G0": mdf_identity("Press", device="Dev1", ecu="ECU_A")},
        ))
        loader = FakeLoader({
            "Press_G5": mdf_info("Press", "bar", "Dev1", "ECU_A"),
            "Press_G9": mdf_info("Press", "degC", "Dev2", "ECU_B"),
        })
        _resolved, remap = resolve_identities(cfg, loader)
        assert remap == {"Press_G0": "Press_G5"}

    def test_contradiction_veto_mj6(self):
        """MJ-6：{Dev1,ECU_A} vs {Dev1,ECU_B} 不得按 OR 命中。"""
        cfg = session(PlotConfig(
            curves=["Press_G0"],
            curve_identities={"Press_G0": mdf_identity("Press", device="Dev1", ecu="ECU_A")},
        ))
        loader = FakeLoader({"Press_G7": mdf_info("Press", "bar", "Dev1", "ECU_B")})
        resolved, remap = resolve_identities(cfg, loader)
        # L1 矛盾否决；L2 键 Press 唯一候选 → 兜底命中（D10）
        assert remap == {"Press_G0": "Press_G7"}

    def test_contradiction_veto_with_alternative(self):
        cfg = session(PlotConfig(
            curves=["Press_G0"],
            curve_identities={"Press_G0": mdf_identity("Press", device="Dev1", ecu="ECU_A")},
        ))
        loader = FakeLoader({
            "Press_G7": mdf_info("Press", "bar", "Dev1", "ECU_B"),
            "Press_G8": mdf_info("Press", "bar", "Dev1", "ECU_A"),
        })
        _resolved, remap = resolve_identities(cfg, loader)
        assert remap == {"Press_G0": "Press_G8"}

    def test_no_si_entry_l1_impossible_falls_l2(self):
        """至少一对非空相等：无 SI 条目 L1 不可能命中，退 L2。"""
        cfg = session(PlotConfig(
            curves=["Press_G0"],
            curve_identities={"Press_G0": mdf_identity("Press")},
        ))
        loader = FakeLoader({"Press_G3": mdf_info("Press", "bar")})
        _resolved, remap = resolve_identities(cfg, loader)
        assert remap == {"Press_G0": "Press_G3"}

    def test_multiple_fingerprint_hits_no_guess(self):
        cfg = session(PlotConfig(
            curves=["Press_G0"],
            curve_identities={"Press_G0": mdf_identity("Press", device="Dev1")},
        ))
        loader = FakeLoader({
            "Press_G1": mdf_info("Press", "bar", "Dev1"),
            "Press_G2": mdf_info("Press", "bar", "Dev1"),
        })
        resolved, remap = resolve_identities(cfg, loader)
        assert remap == {}
        assert resolved.pages[0].plots[0].curves == ["Press_G0"]


class TestL1Blf:
    def test_bus_renumber_unique_hit(self):
        """总线改号 buses=(0,1)→(1,2)：四元组失配 → L2 限定基名唯一候选。"""
        cfg = session(PlotConfig(
            curves=["EngineData.EngineSpeed_G0"],
            curve_identities={
                "EngineData.EngineSpeed_G0": blf_identity("EngineSpeed", "EngineData", 0x1A0, bus=0)
            },
        ))
        loader = FakeLoader({
            "EngineData.EngineSpeed_G1": blf_info("EngineSpeed", "EngineData", 0x1A0, bus=1),
        })
        _resolved, remap = resolve_identities(cfg, loader)
        assert remap == {"EngineData.EngineSpeed_G0": "EngineData.EngineSpeed_G1"}

    def test_quadruple_hit_across_suffix_drift(self):
        cfg = session(PlotConfig(
            curves=["EngineData.EngineSpeed_G0"],
            curve_identities={
                "EngineData.EngineSpeed_G0": blf_identity("EngineSpeed", "EngineData", 0x1A0, bus=1)
            },
        ))
        loader = FakeLoader({
            "EngineData.EngineSpeed_G2": blf_info("EngineSpeed", "EngineData", 0x1A0, bus=1),
            "EngineData.EngineSpeed_G3": blf_info("EngineSpeed", "EngineData", 0x1A0, bus=2),
        })
        _resolved, remap = resolve_identities(cfg, loader)
        assert remap == {"EngineData.EngineSpeed_G0": "EngineData.EngineSpeed_G2"}

    def test_bus_subset_multi_candidate_l3(self):
        """MN-6 风险披露：换总线子集 → L1 失配且 L2 多候选 → 落占位。"""
        cfg = session(PlotConfig(
            curves=["EngineData.EngineSpeed_G0"],
            curve_identities={
                "EngineData.EngineSpeed_G0": blf_identity("EngineSpeed", "EngineData", 0x1A0, bus=0)
            },
        ))
        loader = FakeLoader({
            "EngineData.EngineSpeed_G1": blf_info("EngineSpeed", "EngineData", 0x1A0, bus=1),
            "EngineData.EngineSpeed_G2": blf_info("EngineSpeed", "EngineData", 0x1A0, bus=2),
        })
        resolved, remap = resolve_identities(cfg, loader)
        assert remap == {}
        assert resolved.pages[0].plots[0].curves == ["EngineData.EngineSpeed_G0"]


class TestL2OldDoc:
    def test_mdf_suffix_drift_unique(self):
        cfg = session(PlotConfig(curves=["Press_G0"]))
        loader = FakeLoader({"Press_G1": mdf_info("Press", "bar")})
        _resolved, remap = resolve_identities(cfg, loader)
        assert remap == {"Press_G0": "Press_G1"}

    def test_mdf_multi_candidate_l3(self):
        cfg = session(PlotConfig(curves=["Press_G0"]))
        loader = FakeLoader({
            "Press_G0": mdf_info("Press", "degC"),
            "Press_G1": mdf_info("Press", "bar"),
        })
        # Press_G0 同名仍在 → L0 命中（旧档无身份不校验）
        _resolved, remap = resolve_identities(cfg, loader)
        assert remap == {}

    def test_mdf_absent_multi_candidate_l3(self):
        cfg = session(PlotConfig(curves=["Press_G2"]))
        loader = FakeLoader({
            "Press_G0": mdf_info("Press", "bar"),
            "Press_G1": mdf_info("Press", "degC"),
        })
        resolved, remap = resolve_identities(cfg, loader)
        assert remap == {}
        assert resolved.pages[0].plots[0].curves == ["Press_G2"]

    def test_blf_qualified_to_bare(self):
        """BLF 限定名↔裸名双向（BL-2）：旧档限定名，当前裸名唯一。"""
        cfg = session(PlotConfig(curves=["EngineData.EngineSpeed_G0"]))
        loader = FakeLoader({"EngineSpeed": blf_info("EngineSpeed", "EngineData", 0x1A0)})
        _resolved, remap = resolve_identities(cfg, loader)
        assert remap == {"EngineData.EngineSpeed_G0": "EngineSpeed"}

    def test_blf_bare_to_qualified(self):
        cfg = session(PlotConfig(curves=["EngineSpeed_G0"]))
        loader = FakeLoader({
            "EngineData.EngineSpeed": blf_info("EngineSpeed", "EngineData", 0x1A0),
        })
        _resolved, remap = resolve_identities(cfg, loader)
        assert remap == {"EngineSpeed_G0": "EngineData.EngineSpeed"}

    def test_signal_name_with_dots_no_split(self):
        """D10：信号名本身含点，限定基名索引由 loader 构造，不做点号拆分。"""
        cfg = session(PlotConfig(curves=["Msg.Sig.Sub_G2"]))
        loader = FakeLoader({
            "Msg.Sig.Sub": blf_info("Sig.Sub", "Msg", 0x100),
        })
        _resolved, remap = resolve_identities(cfg, loader)
        assert remap == {"Msg.Sig.Sub_G2": "Msg.Sig.Sub"}

    def test_placeholder_name_heuristic(self):
        """占位无身份（D2），跨文件回退靠 L2；改名后仍是占位。"""
        cfg = session(PlotConfig(curves=[], invalid_curves=["Press_G0"]))
        loader = FakeLoader({"Press_G1": mdf_info("Press", "bar")})
        resolved, remap = resolve_identities(cfg, loader)
        assert remap == {"Press_G0": "Press_G1"}
        assert resolved.pages[0].plots[0].invalid_curves == ["Press_G1"]
        assert resolved.pages[0].plots[0].curves == []


class TestConflictAndStructure:
    def test_conflict_first_come_first_served(self):
        cfg = session(PlotConfig(
            curves=["A_G0", "B_G0"],
            curve_identities={
                "A_G0": mdf_identity("X", device="D"),
                "B_G0": mdf_identity("X", device="D"),
            },
        ))
        loader = FakeLoader({"X_G7": mdf_info("X", device="D")})
        resolved, remap = resolve_identities(cfg, loader)
        assert remap == {"A_G0": "X_G7"}
        assert resolved.pages[0].plots[0].curves == ["X_G7", "B_G0"]

    def test_same_name_two_plots_both_kept(self):
        """同一变量画在两个子图是正常资产，不得当冲突。"""
        cfg = session(
            PlotConfig(curves=["Press_G0"]),
            PlotConfig(curves=["Press_G0"]),
        )
        loader = FakeLoader({"Press_G1": mdf_info("Press")})
        resolved, remap = resolve_identities(cfg, loader)
        assert remap == {"Press_G0": "Press_G1"}
        assert resolved.pages[0].plots[0].curves == ["Press_G1"]
        assert resolved.pages[0].plots[1].curves == ["Press_G1"]

    def test_from_dict_dual_structure_synced(self):
        """MJ-5：from_dict 顶层与 pages 独立对象，两份都要换名。"""
        cfg = PlotSessionConfig.from_dict({
            "plots": [{"curves": ["Press_G0"]}],
            "pages": [{"plots": [{"curves": ["Press_G0"]}]}],
        })
        assert cfg.plots[0] is not cfg.pages[0].plots[0]
        loader = FakeLoader({"Press_G1": mdf_info("Press")})
        resolved, remap = resolve_identities(cfg, loader)
        assert remap == {"Press_G0": "Press_G1"}
        assert resolved.plots[0].curves == ["Press_G1"]
        assert resolved.pages[0].plots[0].curves == ["Press_G1"]

    def test_export_shared_structure_synced(self):
        shared = PlotConfig(curves=["Press_G0"])
        cfg = PlotSessionConfig(
            plots=[shared], pages=[PlotPageConfig(plots=[shared])]
        )
        loader = FakeLoader({"Press_G1": mdf_info("Press")})
        resolved, remap = resolve_identities(cfg, loader)
        assert remap == {"Press_G0": "Press_G1"}
        assert resolved.plots[0] is resolved.pages[0].plots[0]
        assert resolved.plots[0].curves == ["Press_G1"]

    def test_settings_and_identity_keys_synced_orphan_kept(self):
        cfg = session(PlotConfig(
            curves=["Press_G0"],
            curve_settings={
                "Press_G0": {"factor": 2.0},
                "orphan": {"display_name": "手改遗留"},
            },
            curve_identities={
                "Press_G0": mdf_identity("Press"),
                "orphan": mdf_identity("Ghost"),
            },
        ))
        loader = FakeLoader({"Press_G1": mdf_info("Press")})
        resolved, _remap = resolve_identities(cfg, loader)
        p = resolved.pages[0].plots[0]
        assert p.curve_settings == {
            "Press_G1": {"factor": 2.0},
            "orphan": {"display_name": "手改遗留"},
        }
        assert set(p.curve_identities) == {"Press_G1", "orphan"}

    def test_original_config_untouched_d6(self):
        cfg = session(PlotConfig(
            curves=["Press_G0"],
            curve_identities={"Press_G0": mdf_identity("Press")},
        ))
        loader = FakeLoader({"Press_G1": mdf_info("Press")})
        resolved, _remap = resolve_identities(cfg, loader)
        assert cfg.pages[0].plots[0].curves == ["Press_G0"]
        assert cfg.plots[0].curves == ["Press_G0"]
        assert resolved is not cfg

    def test_idempotent(self):
        cfg = session(PlotConfig(
            curves=["Press_G0"],
            curve_settings={"Press_G0": {"factor": 2.0}},
            curve_identities={"Press_G0": mdf_identity("Press", "bar", "Dev1")},
        ))
        loader = FakeLoader({"Press_G1": mdf_info("Press", "bar", "Dev1")})
        resolved, remap = resolve_identities(cfg, loader)
        assert remap == {"Press_G0": "Press_G1"}
        again, remap2 = resolve_identities(resolved, loader)
        assert remap2 == {}
        assert again.pages[0].plots[0].curves == ["Press_G1"]
        assert again.pages[0].plots[0].curve_settings == {"Press_G1": {"factor": 2.0}}
        assert again.plots[0].curves == ["Press_G1"]

    def test_v1_top_level_only(self):
        cfg = session(PlotConfig(curves=["Press_G0"]), v2=False)
        loader = FakeLoader({"Press_G1": mdf_info("Press")})
        resolved, remap = resolve_identities(cfg, loader)
        assert remap == {"Press_G0": "Press_G1"}
        assert resolved.plots[0].curves == ["Press_G1"]


class TestGuards:
    def test_none_loader(self):
        cfg = session(PlotConfig(curves=["a"]))
        resolved, remap = resolve_identities(cfg, None)
        assert resolved is cfg
        assert remap == {}

    def test_csv_loader_without_gci(self):
        """D8：CSV/Excel 无 get_channel_info → 原样返回。"""
        cfg = session(PlotConfig(curves=["a"]))
        loader = SimpleNamespace(var_names=["a", "b"])
        resolved, remap = resolve_identities(cfg, loader)
        assert resolved is cfg
        assert remap == {}

    def test_loader_without_var_names(self):
        cfg = session(PlotConfig(curves=["a"]))
        loader = SimpleNamespace(get_channel_info=lambda n: {})
        resolved, remap = resolve_identities(cfg, loader)
        assert resolved is cfg
        assert remap == {}

    def test_gci_keyerror_tolerated(self):
        class Broken(FakeLoader):
            def get_channel_info(self, name):
                raise KeyError(name)

        cfg = session(PlotConfig(curves=["Press_G0"]))
        resolved, remap = resolve_identities(cfg, Broken({}))
        assert remap == {}
        assert resolved.pages[0].plots[0].curves == ["Press_G0"]


class TestIdentityFromInfo:
    def test_mdf_with_si(self):
        entry = identity_from_channel_info(mdf_info("Press", "bar", "Dev1", "ECU_A"))
        assert entry == {
            "original_name": "Press", "kind": "mdf",
            "device": "Dev1", "ecu": "ECU_A", "unit": "bar",
        }

    def test_asammdf_artifact_ecu_filtered(self):
        """MJ-9：合成 v3 被注入的伪影 source.name 视为空。"""
        entry = identity_from_channel_info(
            mdf_info("Press", "bar", None, "Channel inserted by Python Script")
        )
        assert entry == {"original_name": "Press", "kind": "mdf", "unit": "bar"}

    def test_dash_unit_omitted(self):
        entry = identity_from_channel_info(mdf_info("Press", "-"))
        assert "unit" not in entry

    def test_blf_full(self):
        entry = identity_from_channel_info(
            blf_info("EngineSpeed", "EngineData", 0x1A0, unit="rpm", bus=1)
        )
        assert entry == {
            "original_name": "EngineSpeed", "kind": "blf", "frame": "EngineData",
            "can_id": 0x1A0, "extended": False, "bus_channel": 1, "unit": "rpm",
        }

    def test_blf_missing_int_can_id_rejected(self):
        info = blf_info("EngineSpeed", "EngineData", 0x1A0)
        del info["can"]["can_id"]
        assert identity_from_channel_info(info) is None

    def test_unknown_kind_rejected(self):
        assert identity_from_channel_info(
            {"meta": SimpleNamespace(original_name="a", unit="")}
        ) is None

    def test_empty_original_name_rejected(self):
        assert identity_from_channel_info(mdf_info("", "bar")) is None