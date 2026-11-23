"""导出后处理身份采集单测（身份匹配 C2 / §3.1 / D2/D8 / MJ-7/MJ-9）。"""

from types import SimpleNamespace

from src.core.plot_config import PlotConfig, PlotPageConfig, PlotSessionConfig
from src.ui.plot_config_manager import PlotConfigManager

from tests.unit.core.test_template_identity import FakeLoader, blf_info, mdf_info


def _mw(loader):
    return SimpleNamespace(loader=loader)


class TestCollect:
    def test_mdf_si_collected(self):
        cfg = PlotSessionConfig(
            pages=[PlotPageConfig(plots=[PlotConfig(curves=["Press_G0"])])]
        )
        loader = FakeLoader({"Press_G0": mdf_info("Press", "bar", "Dev1", "ECU_A")})
        PlotConfigManager._collect_curve_identities(cfg, _mw(loader))
        assert cfg.pages[0].plots[0].curve_identities == {
            "Press_G0": {
                "original_name": "Press", "kind": "mdf",
                "device": "Dev1", "ecu": "ECU_A", "unit": "bar",
            }
        }

    def test_blf_quadruple_collected(self):
        cfg = PlotSessionConfig(
            pages=[PlotPageConfig(plots=[PlotConfig(curves=["EngineData.EngineSpeed_G0"])])]
        )
        loader = FakeLoader({
            "EngineData.EngineSpeed_G0": blf_info(
                "EngineSpeed", "EngineData", 0x1A0, unit="rpm", bus=1
            )
        })
        PlotConfigManager._collect_curve_identities(cfg, _mw(loader))
        assert cfg.pages[0].plots[0].curve_identities == {
            "EngineData.EngineSpeed_G0": {
                "original_name": "EngineSpeed", "kind": "blf",
                "frame": "EngineData", "can_id": 0x1A0,
                "extended": False, "bus_channel": 1, "unit": "rpm",
            }
        }

    def test_placeholder_name_skipped(self):
        """D2 披露：占位名不在 loader → KeyError 跳过，永远拿不到身份。"""
        cfg = PlotSessionConfig(
            pages=[PlotPageConfig(plots=[
                PlotConfig(curves=["Press_G0"], invalid_curves=["Ghost_G0"])
            ])]
        )
        loader = FakeLoader({"Press_G0": mdf_info("Press", "bar")})
        PlotConfigManager._collect_curve_identities(cfg, _mw(loader))
        p = cfg.pages[0].plots[0]
        assert set(p.curve_identities) == {"Press_G0"}

    def test_csv_loader_no_collection(self):
        """D8：CSV/Excel 无 get_channel_info → 不采集。"""
        cfg = PlotSessionConfig(
            pages=[PlotPageConfig(plots=[PlotConfig(curves=["speed"])])]
        )
        PlotConfigManager._collect_curve_identities(
            cfg, _mw(SimpleNamespace(var_names=["speed"]))
        )
        assert cfg.pages[0].plots[0].curve_identities == {}

    def test_none_loader_no_raise(self):
        cfg = PlotSessionConfig(plots=[PlotConfig(curves=["a"])])
        PlotConfigManager._collect_curve_identities(cfg, _mw(None))
        assert cfg.plots[0].curve_identities == {}

    def test_legacy_top_level_plots_covered(self):
        """MJ-7：后处理天然覆盖 legacy 替身分支（只有顶层 plots）。"""
        cfg = PlotSessionConfig(plots=[PlotConfig(curves=["Press_G0"])])
        loader = FakeLoader({"Press_G0": mdf_info("Press", "bar")})
        PlotConfigManager._collect_curve_identities(cfg, _mw(loader))
        assert "Press_G0" in cfg.plots[0].curve_identities

    def test_shared_object_dedup(self):
        """导出式构造顶层与 pages 共享对象：按身份去重，只采集一次。"""
        calls = []

        class Counting(FakeLoader):
            def get_channel_info(self, name):
                calls.append(name)
                return super().get_channel_info(name)

        shared = PlotConfig(curves=["Press_G0"])
        cfg = PlotSessionConfig(
            plots=[shared], pages=[PlotPageConfig(plots=[shared])]
        )
        loader = Counting({"Press_G0": mdf_info("Press", "bar")})
        PlotConfigManager._collect_curve_identities(cfg, _mw(loader))
        assert calls == ["Press_G0"]
        assert cfg.plots[0].curve_identities == cfg.pages[0].plots[0].curve_identities

    def test_asammdf_artifact_not_collected_as_ecu(self):
        cfg = PlotSessionConfig(plots=[PlotConfig(curves=["Press_G0"])])
        loader = FakeLoader({
            "Press_G0": mdf_info("Press", "bar", None, "Channel inserted by Python Script")
        })
        PlotConfigManager._collect_curve_identities(cfg, _mw(loader))
        entry = cfg.plots[0].curve_identities["Press_G0"]
        assert "ecu" not in entry

    def test_closed_loader_keyerror_skipped(self):
        class Closed:
            var_names = ["Press_G0"]

            def get_channel_info(self, name):
                raise KeyError("数据源已关闭")

        cfg = PlotSessionConfig(plots=[PlotConfig(curves=["Press_G0"])])
        PlotConfigManager._collect_curve_identities(cfg, _mw(Closed()))
        assert cfg.plots[0].curve_identities == {}