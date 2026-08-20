"""core.auto_save_manager 单元测试：保存/加载往返、备份回退、应用判定。"""

from __future__ import annotations

import pytest

from src.core.auto_save_manager import AutoSaveManager
from src.core.plot_config import PlotConfig, PlotSessionConfig


def _make_config(curves: list[str]) -> PlotSessionConfig:
    return PlotSessionConfig(plots=[PlotConfig(curves=curves)])


@pytest.fixture
def auto_save(qapp, tmp_path):
    return AutoSaveManager(storage_path=tmp_path / "autosave")


class TestSaveLoadRoundtrip:
    def test_roundtrip(self, auto_save):
        cfg = _make_config(["a", "b"])
        auto_save.auto_save(cfg)
        loaded = auto_save.load_auto_save()
        assert loaded is not None
        assert loaded.plots[0].curves == ["a", "b"]

    def test_load_without_file_returns_none(self, auto_save):
        assert auto_save.load_auto_save() is None

    def test_backup_fallback_when_main_missing(self, auto_save):
        """主文件丢失时应从 backup 恢复"""
        auto_save.auto_save(_make_config(["v1"]))
        auto_save.auto_save(_make_config(["v2"]))  # 生成 backup
        auto_save._auto_save_file.unlink()
        loaded = auto_save.load_auto_save()
        assert loaded is not None
        assert loaded.plots[0].curves == ["v1"]


class TestShouldApply:
    def test_disabled_rejects(self, auto_save, app_settings):
        app_settings.set_auto_save_enabled(False)
        auto_save.auto_save(_make_config(["a", "b"]))
        should, reason = auto_save.should_apply_auto_save(["a", "b"])
        assert should is False
        assert "disabled" in reason

    def test_no_saved_config_rejects(self, auto_save, app_settings):
        app_settings.set_auto_save_enabled(True)
        should, _reason = auto_save.should_apply_auto_save(["a"])
        assert should is False

    def test_high_match_ratio_applies(self, auto_save, app_settings):
        app_settings.set_auto_save_enabled(True)
        auto_save.auto_save(_make_config(["a", "b", "c"]))
        should, reason = auto_save.should_apply_auto_save(["a", "b", "c", "x"])
        assert should is True
        assert "100%" in reason

    def test_low_match_ratio_rejects(self, auto_save, app_settings):
        app_settings.set_auto_save_enabled(True)
        auto_save.auto_save(_make_config(["a", "b", "c", "d", "e"]))
        # 仅 1/5 = 20% < 60%
        should, reason = auto_save.should_apply_auto_save(["a", "x", "y"])
        assert should is False
        assert "20%" in reason

    def test_empty_current_vars_applies(self, auto_save, app_settings):
        app_settings.set_auto_save_enabled(True)
        auto_save.auto_save(_make_config(["a"]))
        should, _ = auto_save.should_apply_auto_save([])
        assert should is True
