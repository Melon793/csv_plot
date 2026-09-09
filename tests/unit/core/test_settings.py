"""core.settings 单元测试：环境变量路径注入、配置读写与持久化。"""

from __future__ import annotations

import os

from src.core.settings import AppSettings, ConfigKey


class TestConfigDirInjection:
    def test_config_dir_respects_env_override(self, app_settings):
        """CSV_PLOT_CONFIG_DIR 注入生效，避免污染真实用户配置"""
        override = os.environ["CSV_PLOT_CONFIG_DIR"]
        assert str(app_settings.config_dir) == override
        assert (app_settings.config_dir / "app.ini").exists()


class TestGetSetRoundtrip:
    def test_font_name_roundtrip(self, app_settings):
        app_settings.set_font_name("Consolas")
        assert app_settings.get_font_name() == "Consolas"

    def test_font_cache_version_roundtrip(self, app_settings):
        app_settings.set_font_cache_version(7)
        assert app_settings.get_font_cache_version() == 7

    def test_mono_font_roundtrip(self, app_settings):
        app_settings.set_mono_font_name("Menlo")
        app_settings.set_mono_font_cache_version(3)
        assert app_settings.get_mono_font_name() == "Menlo"
        assert app_settings.get_mono_font_cache_version() == 3

    def test_auto_save_flag_roundtrip(self, app_settings):
        # 会话内 ini 持久化，先显式置 False 避免受其它用例影响
        app_settings.set_auto_save_enabled(False)
        assert app_settings.is_auto_save_enabled() is False
        app_settings.set_auto_save_enabled(True)
        assert app_settings.is_auto_save_enabled() is True

    def test_last_template_roundtrip(self, app_settings):
        app_settings.set_last_template_id("abcd1234")
        app_settings.set_last_template_name("tpl")
        assert app_settings.get_last_template_id() == "abcd1234"
        assert app_settings.get_last_template_name() == "tpl"

    def test_get_value_default(self, app_settings):
        assert app_settings.get_value("nonexistent/key", "fallback") == "fallback"


class TestPersistence:
    def test_values_survive_singleton_reset(self, qapp):
        """sync 后重置单例，新实例仍能从 ini 读回数据"""
        AppSettings._reset_for_tests()
        s1 = AppSettings()
        s1.set_value(ConfigKey.FONT_NAME, "Arial")
        s1.sync()

        AppSettings._reset_for_tests()
        s2 = AppSettings()
        assert s2.get_font_name() == "Arial"
        AppSettings._reset_for_tests()

    def test_singleton_returns_same_instance(self, app_settings):
        assert AppSettings() is app_settings
