"""should_apply_auto_save 的 loader 可选参（身份匹配 C3 / D12）。

漂移名在当前数据中**不存在**时：传 loader → resolve 重映射计入 matched
放行；不传（旧调用口径）→ 交集 0% 拒放。夹具用合成 loader 替身。
"""

from types import SimpleNamespace

import pytest

from src.core.auto_save_manager import AutoSaveManager
from src.core.plot_config import PlotConfig, PlotSessionConfig


@pytest.fixture()
def asm(qapp, tmp_path, app_settings):
    manager = AutoSaveManager(storage_path=tmp_path / "cfg")
    manager.set_auto_save_enabled(True)
    return manager


def _save(asm, curves):
    asm.auto_save(PlotSessionConfig(plots=[PlotConfig(curves=list(curves))]))


def _loader(var_names):
    def gci(name):
        return {
            "meta": SimpleNamespace(original_name=name.split("_G")[0], unit="bar"),
            "source": {"path": None, "name": None},
        }

    return SimpleNamespace(var_names=list(var_names), get_channel_info=gci)


def test_drifted_name_absent_with_loader_applies(asm):
    _save(asm, ["Press_G0"])
    should, reason = asm.should_apply_auto_save(["Press_G1"], _loader(["Press_G1"]))
    assert should is True
    assert "100%" in reason


def test_drifted_name_absent_without_loader_rejects(asm):
    """旧调用口径零回归：单参调用行为与改造前逐字一致。"""
    _save(asm, ["Press_G0"])
    should, reason = asm.should_apply_auto_save(["Press_G1"])
    assert should is False
    assert reason == "No matching variables"


def test_csv_loader_without_gci_unchanged(asm):
    """D8：CSV 无 get_channel_info → resolve 原样返回，旧口径。"""
    _save(asm, ["speed"])
    csv = SimpleNamespace(var_names=["speed", "rpm"])
    should, _ = asm.should_apply_auto_save(["speed", "rpm"], csv)
    assert should is True
    should_miss, _ = asm.should_apply_auto_save(["rpm"], csv)
    assert should_miss is False


def test_none_loader_unchanged(asm):
    _save(asm, ["speed"])
    should, _ = asm.should_apply_auto_save(["speed"], None)
    assert should is True


# ---------------- M-1 方案乙：demoted 计入失配（0.6 阈值同口径） ----------------


def _loader_with(infos):
    """infos: {显示名: (original_name, unit)}，无 SI 的 MDF 替身。"""
    def gci(name):
        on, unit = infos[name]
        return {
            "meta": SimpleNamespace(original_name=on, unit=unit),
            "source": {"path": None, "name": None},
        }

    return SimpleNamespace(var_names=list(infos), get_channel_info=gci)


def _save_with_identity(asm, curves, identities):
    asm.auto_save(PlotSessionConfig(
        plots=[PlotConfig(curves=list(curves), curve_identities=dict(identities))]
    ))


def test_all_demoted_rejects(asm):
    """全降占位：ratio 0 → 拒放（不走纯占位捷径）。"""
    _save_with_identity(
        asm, ["Press_G0"], {"Press_G0": {"original_name": "Press", "kind": "mdf", "unit": "bar"}}
    )
    loader = _loader_with({"Press_G0": ("Press", "degC"), "Press_G1": ("Press", "bar")})
    should, reason = asm.should_apply_auto_save(list(loader.var_names), loader)
    assert should is False
    assert reason == "No matching variables"


def test_partial_demoted_below_threshold_rejects(asm):
    """1/2 匹配（另一条降占位）→ 50% < 60% 拒放。"""
    _save_with_identity(
        asm, ["Press_G0", "Speed"],
        {"Press_G0": {"original_name": "Press", "kind": "mdf", "unit": "bar"}},
    )
    loader = _loader_with({
        "Press_G0": ("Press", "degC"), "Press_G1": ("Press", "bar"),
        "Speed": ("Speed", "km/h"),
    })
    should, reason = asm.should_apply_auto_save(list(loader.var_names), loader)
    assert should is False
    assert "50%" in reason


def test_partial_demoted_above_threshold_applies(asm):
    """2/3 匹配（另一条降占位）→ 67% ≥ 60% 放行。"""
    _save_with_identity(
        asm, ["Press_G0", "Speed", "Rpm"],
        {"Press_G0": {"original_name": "Press", "kind": "mdf", "unit": "bar"}},
    )
    loader = _loader_with({
        "Press_G0": ("Press", "degC"), "Press_G1": ("Press", "bar"),
        "Speed": ("Speed", "km/h"), "Rpm": ("Rpm", "rpm"),
    })
    should, reason = asm.should_apply_auto_save(list(loader.var_names), loader)
    assert should is True
    assert "67%" in reason