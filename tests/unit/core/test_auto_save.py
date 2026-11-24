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

    @pytest.mark.parametrize(
        "corrupt_payload",
        ["pages: [broken", "- not-a-config-mapping"],
        ids=["invalid-yaml", "invalid-config-shape"],
    )
    def test_r6_backup_fallback_when_main_exists_but_corrupt(
        self, auto_save, corrupt_payload
    ):
        """R6：主文件存在但 YAML/配置损坏时，也必须尝试可用 backup。"""
        auto_save.auto_save(_make_config(["v1"]))
        auto_save.auto_save(_make_config(["v2"]))  # backup 保留 v1
        auto_save._auto_save_file.write_text(corrupt_payload, encoding="utf-8")
        assert auto_save._auto_save_file.exists()
        assert auto_save._auto_save_backup_file.exists()

        loaded = auto_save.load_auto_save()

        assert loaded is not None
        assert loaded.plots[0].curves == ["v1"]

    def test_write_failure_returns_false_and_cleans_tmp(self, auto_save, monkeypatch):
        auto_save.auto_save(_make_config(["a"]))

        def fail_replace(*_args):
            raise OSError("synthetic replace failure")

        monkeypatch.setattr("src.core.auto_save_manager.os.replace", fail_replace)

        assert auto_save.auto_save(_make_config(["b"])) is False
        assert not list(auto_save._storage_path.glob("*.tmp"))
        assert auto_save.load_auto_save().plots[0].curves == ["a"]

    def test_concurrent_saves_keep_primary_and_backup_readable(self, auto_save, monkeypatch):
        import threading
        import yaml

        auto_save.auto_save(_make_config(["old"]))
        barrier = threading.Barrier(2)
        real_dump = yaml.dump

        def synchronized_dump(*args, **kwargs):
            barrier.wait(timeout=5)
            return real_dump(*args, **kwargs)

        monkeypatch.setattr("src.core.auto_save_manager.yaml.dump", synchronized_dump)
        results = []

        def save(tag):
            cfg = _make_config([f"{tag}-{i}" for i in range(6000)])
            results.append(auto_save.auto_save(cfg))

        threads = [threading.Thread(target=save, args=(tag,)) for tag in ("a", "b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert results == [True, True]
        for path in (auto_save._auto_save_file, auto_save._auto_save_backup_file):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            curves = data["plots"][0]["curves"]
            assert len(curves) in (1, 6000)
            if len(curves) == 1:
                assert curves == ["old"]
            else:
                assert curves[0] in {"a-0", "b-0"}
                assert curves == [f"{curves[0][0]}-{i}" for i in range(6000)]
        assert auto_save.load_auto_save() is not None
        assert not list(auto_save._storage_path.glob("*.tmp"))

    def test_prepare_reuses_one_read_and_identity_snapshot(
        self, auto_save, app_settings, monkeypatch
    ):
        app_settings.set_auto_save_enabled(True)
        config = _make_config(["a", "b"])
        loads: list[PlotSessionConfig] = []

        def load_once():
            loads.append(config)
            return config

        monkeypatch.setattr(auto_save, "load_auto_save", load_once)

        decision = auto_save.prepare_auto_save(["a", "b"])

        assert loads == [config]
        assert decision.config is config
        assert decision.identity_snapshot.config is config
        assert decision.should is True

    def test_prepare_marks_existing_unreadable_files_as_failed(
        self, auto_save, app_settings
    ):
        app_settings.set_auto_save_enabled(True)
        auto_save._auto_save_file.write_text("pages: [broken", encoding="utf-8")
        auto_save._auto_save_backup_file.write_text("pages: [broken", encoding="utf-8")

        decision = auto_save.prepare_auto_save(["a"])

        assert decision.should is False
        assert decision.failed is True
        assert decision.config is None


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

    def test_placeholder_only_config_applies(self, auto_save, app_settings):
        """纯占位配置（曲线全部失效）是布局资产，必须允许恢复。

        回归点：``extract_variables_from_config`` 只收 ``curves``，占位配置会被
        当成「无变量」直接拒掉，于是「失效占位进 auto-save」在恢复侧落空——
        用户所有曲线失效后退出，下次加载数据只剩空布局。
        """
        app_settings.set_auto_save_enabled(True)
        auto_save.auto_save(
            PlotSessionConfig(
                plots=[PlotConfig(curves=[], invalid_curves=["ghost_a", "ghost_b"])]
            )
        )
        should, reason = auto_save.should_apply_auto_save(["x", "y"])
        assert should is True
        assert "placeholder" in reason

    def test_config_without_curves_or_placeholders_still_rejects(
        self, auto_save, app_settings
    ):
        """既无曲线也无占位的空配置仍须拒绝（不得放宽成无条件恢复）。"""
        app_settings.set_auto_save_enabled(True)
        auto_save.auto_save(_make_config([]))
        should, reason = auto_save.should_apply_auto_save(["x"])
        assert should is False
        assert "no variables" in reason
