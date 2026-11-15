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


class TestBlfDbNotes:
    """CAN 数据库备注：JSON dict 持久化，键为规范化路径。"""

    def test_note_roundtrip_and_normalization(self, app_settings, tmp_path):
        # 未归一化的相对写法也要能命中同一键（normalize_input_path 幂等）
        real = tmp_path / "powertrain.dbc"
        real.write_text("x", encoding="utf-8")
        app_settings.set_blf_db_note(str(real), "动力总成主库")
        assert app_settings.get_blf_db_note(str(real)) == "动力总成主库"
        assert app_settings.get_blf_db_note(str(real.resolve())) == "动力总成主库"
        assert (
            app_settings.get_blf_db_notes()[str(real)] == "动力总成主库"
        ), "两个写法必须命中同一条备注"

    def test_blank_note_deletes_the_entry(self, app_settings, tmp_path):
        # app_settings 只重置单例不删 ini，其他键会跨用例残留 —— 只断言本键
        p = tmp_path / "a.dbc"
        app_settings.set_blf_db_note(str(p), "temp")
        app_settings.set_blf_db_note(str(p), "   ")
        assert app_settings.get_blf_db_note(str(p)) == ""
        assert str(p) not in app_settings.get_blf_db_notes()

    def test_corrupt_json_degrades_to_empty(self, app_settings):
        app_settings.set_value(ConfigKey.BLF_DB_NOTES, "{not json")
        assert app_settings.get_blf_db_notes() == {}
        assert app_settings.get_blf_db_note("whatever.dbc") == ""

    def test_notes_survive_singleton_reset(self, tmp_path):
        p = tmp_path / "keep.dbc"
        AppSettings._reset_for_tests()
        AppSettings().set_blf_db_note(str(p), "keep")
        AppSettings().sync()
        AppSettings._reset_for_tests()
        assert AppSettings().get_blf_db_note(str(p)) == "keep"
        AppSettings._reset_for_tests()


class TestMarkStatsPersistence:
    """标记统计窗口持久化：列配置四级读取防御 + 表头状态 QByteArray。"""

    DEFAULT = ["dy", "slope", "y_avg"]
    VALID = ["x1", "x2", "dx", "dy", "slope", "y_avg", "y_max", "y_min", "count"]

    def _get(self, settings):
        return settings.get_mark_stats_columns(self.DEFAULT, self.VALID)

    def test_columns_roundtrip_preserves_order(self, app_settings):
        """存储顺序即显示顺序，读回不得重排。"""
        app_settings.set_mark_stats_columns(["y_max", "dy", "x1"])
        assert self._get(app_settings) == ["y_max", "dy", "x1"]

    def test_columns_roundtrip_survives_singleton_reset(self, qapp):
        from src.core.settings import AppSettings

        AppSettings._reset_for_tests()
        AppSettings().set_mark_stats_columns(["dy", "count"])
        AppSettings._reset_for_tests()
        assert self._get(AppSettings()) == ["dy", "count"]
        AppSettings._reset_for_tests()

    def test_unknown_keys_filtered(self, app_settings):
        """注册表外的 key（手改 ini / 未来版本删指标）静默过滤。"""
        app_settings.set_mark_stats_columns(["dy", "bogus"])
        assert self._get(app_settings) == ["dy"]

    def test_corrupt_json_falls_back_to_default(self, app_settings):
        app_settings.set_value(ConfigKey.MARK_STATS_COLUMNS, "{not json")
        assert self._get(app_settings) == self.DEFAULT

    def test_non_list_json_falls_back_to_default(self, app_settings):
        app_settings.set_value(ConfigKey.MARK_STATS_COLUMNS, '"dy"')
        assert self._get(app_settings) == self.DEFAULT

    def test_empty_list_falls_back_to_default(self, app_settings):
        """全部 key 非法 → 过滤后为空 → 默认集（③级防御）。"""
        app_settings.set_mark_stats_columns(["nope", "also_bad"])
        assert self._get(app_settings) == self.DEFAULT

    def test_duplicates_removed_keeping_first(self, app_settings):
        app_settings.set_mark_stats_columns(["dy", "dy", "slope", "dy"])
        assert self._get(app_settings) == ["dy", "slope"]

    def test_unset_returns_default(self, app_settings):
        # fixture 只重置单例不删 ini：上一用例写入的列配置会残留，先显式清键
        app_settings.set_value(ConfigKey.MARK_STATS_COLUMNS, "")
        assert self._get(app_settings) == self.DEFAULT

    def test_header_state_roundtrip(self, app_settings, qapp):
        """QByteArray 表头状态经 ini（base64）往返不变。"""
        from PySide6.QtCore import QByteArray

        state = QByteArray(b"\x00\x01header-state-bytes")
        app_settings.set_mark_stats_header_state(state)
        restored = app_settings.get_mark_stats_header_state()
        assert bytes(restored) == b"\x00\x01header-state-bytes"

    def test_header_state_unset_returns_none(self, app_settings):
        # 同上：上一用例写入的状态残留在 ini，先清键再断言
        app_settings.set_value(ConfigKey.MARK_STATS_HEADER_STATE, "")
        assert app_settings.get_mark_stats_header_state() is None


class TestPlotVariableEditorGeometry:
    def test_geometry_roundtrip(self, app_settings):
        """QByteArray 几何经 ini（base64）往返不变。"""
        from PySide6.QtCore import QByteArray

        geometry = QByteArray(b"\x00\x01editor-geometry-bytes")
        app_settings.set_plot_variable_editor_geometry(geometry)
        restored = app_settings.get_plot_variable_editor_geometry()
        assert bytes(restored) == b"\x00\x01editor-geometry-bytes"

    def test_geometry_unset_returns_none(self, app_settings):
        # 同上：上一用例写入的几何会残留在 ini，先显式清键
        app_settings.set_value(ConfigKey.PLOT_VARIABLE_EDITOR_GEOMETRY, "")
        assert app_settings.get_plot_variable_editor_geometry() is None


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


class TestBlfFileBindings:
    """按文件记忆的完整配置：路径作键、size/mtime 作校验、LRU 20 份。"""

    def test_remember_and_read_roundtrip(self, app_settings, tmp_path):
        blf = tmp_path / "demo.blf"
        blf.write_bytes(b"x")
        config = {
            "schema": "canview/1",
            "databases": ["p.dbc"],
            "channels": {"2": {"alias": "车身", "databases": ["p.dbc"]}},
        }
        app_settings.remember_blf_file_binding(
            str(blf), size=1, mtime=123.0, config=config
        )

        entry = app_settings.get_blf_file_bindings()[
            AppSettings.blf_memory_key(str(blf))
        ]
        assert entry["size"] == 1
        assert entry["mtime"] == 123.0
        assert entry["config"] == config
        assert entry["ts"] > 0

    def test_lru_keeps_only_the_newest_twenty(self, app_settings, tmp_path):
        for i in range(25):
            app_settings.remember_blf_file_binding(
                str(tmp_path / f"f{i}.blf"), size=0, mtime=0.0,
                config={"databases": []},
            )
        entries = app_settings.get_blf_file_bindings()
        assert len(entries) == AppSettings.BLF_FILE_BINDINGS_LIMIT == 20

    def test_bad_json_and_non_config_entries_are_ignored(self, app_settings):
        app_settings.set_value(ConfigKey.BLF_FILE_BINDINGS, "{not json")
        assert app_settings.get_blf_file_bindings() == {}

        app_settings.set_value(
            ConfigKey.BLF_FILE_BINDINGS,
            '{"a": {"size": 1}, "b": "x"}',   # 没有 config 的条目一律丢弃
        )
        assert app_settings.get_blf_file_bindings() == {}

    def test_forget_removes_only_that_file(self, app_settings, tmp_path):
        for name in ("a.blf", "b.blf"):
            app_settings.remember_blf_file_binding(
                str(tmp_path / name), size=0, mtime=0.0,
                config={"databases": []},
            )
        app_settings.forget_blf_file_binding(str(tmp_path / "a.blf"))
        entries = app_settings.get_blf_file_bindings()
        assert AppSettings.blf_memory_key(str(tmp_path / "a.blf")) not in entries
        assert AppSettings.blf_memory_key(str(tmp_path / "b.blf")) in entries

    def test_store_path_stays_inside_app_data_dir(self, tmp_path):
        path = AppSettings.blf_binding_store_path(str(tmp_path / "x.blf"))
        assert path.endswith(".canview.yaml")
        assert "can_bindings" in path.replace("\\", "/")
