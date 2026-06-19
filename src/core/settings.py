"""统一配置管理器 - 集中管理应用所有配置"""

from __future__ import annotations
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Type

from PySide6.QtCore import QSettings


class ConfigKey(StrEnum):
    SCHEMA_VERSION = "config/schema_version"
    FONT_CACHE_VERSION = "font/cache_version"
    FONT_NAME = "font/name"
    FONT_MONO_CACHE_VERSION = "font/mono_cache_version"
    FONT_MONO_NAME = "font/mono_name"
    AUTO_SAVE_ENABLED = "auto_save/enabled"
    LOG_WINDOW_GEOMETRY = "log_window/geometry"
    TEMPLATE_LAST_ID = "template/last_id"
    TEMPLATE_LAST_NAME = "template/last_name"


class AppSettings:
    _instance: AppSettings | None = None
    _lock: Lock = Lock()

    SCHEMA_VERSION = 1

    def __new__(cls) -> AppSettings:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        config_dir = self._get_config_dir()
        try:
            config_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            pass

        self._settings = QSettings(
            str(config_dir / "app.ini"),
            QSettings.Format.IniFormat,
        )
        self._config_dir = config_dir
        self._migrate_if_needed()

    @staticmethod
    def _get_config_dir() -> Path:
        import sys
        if sys.platform == "darwin":
            base = Path.home() / "Library" / "Application Support"
        elif sys.platform == "win32":
            import os
            base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
        else:
            import os
            base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
        return base / "CSVPlot"

    def _migrate_if_needed(self) -> None:
        current_version = self._settings.value(
            ConfigKey.SCHEMA_VERSION, 0, type=int
        )
        if current_version < 1:
            self._migrate_from_v0()
        self._settings.setValue(ConfigKey.SCHEMA_VERSION, self.SCHEMA_VERSION)

    def _migrate_from_v0(self) -> None:
        try:
            from PySide6.QtCore import QSettings as QS

            migrations: list[tuple[str, str, list[tuple[str, str]]]] = [
                ("csv_plot", "font_cache", [
                    ("cache_version", ConfigKey.FONT_CACHE_VERSION),
                    ("font_name", ConfigKey.FONT_NAME),
                    ("mono_cache_version", ConfigKey.FONT_MONO_CACHE_VERSION),
                    ("mono_font_name", ConfigKey.FONT_MONO_NAME),
                ]),
                ("CSVPlot", "AutoSave", [
                    ("auto_save_enabled", ConfigKey.AUTO_SAVE_ENABLED),
                ]),
                ("CSVPlot", "LogWindow", [
                    ("geometry", ConfigKey.LOG_WINDOW_GEOMETRY),
                ]),
                ("CSVPlot", "TemplateMenu", [
                    ("last_template_id", ConfigKey.TEMPLATE_LAST_ID),
                    ("last_template_name", ConfigKey.TEMPLATE_LAST_NAME),
                ]),
            ]

            for org, app, key_mappings in migrations:
                old_settings = QS(org, app)
                for old_key, new_key in key_mappings:
                    val = old_settings.value(old_key)
                    if val is not None:
                        self._settings.setValue(new_key, val)

            self._settings.sync()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("配置迁移失败: %s", e)

    @property
    def config_dir(self) -> Path:
        return self._config_dir

    def get_value(
        self, key: str | ConfigKey, default: Any = None, type: Optional[Type] = None
    ) -> Any:
        if type is not None:
            return self._settings.value(key, default, type)
        return self._settings.value(key, default)

    def set_value(self, key: str | ConfigKey, value: Any) -> None:
        self._settings.setValue(key, value)

    def sync(self) -> None:
        self._settings.sync()

    def get_font_cache_version(self) -> int:
        return self._settings.value(ConfigKey.FONT_CACHE_VERSION, 0, type=int)

    def set_font_cache_version(self, version: int) -> None:
        self._settings.setValue(ConfigKey.FONT_CACHE_VERSION, version)

    def get_font_name(self) -> str:
        return self._settings.value(ConfigKey.FONT_NAME, "", type=str)

    def set_font_name(self, name: str) -> None:
        self._settings.setValue(ConfigKey.FONT_NAME, name)

    def get_mono_font_cache_version(self) -> int:
        return self._settings.value(ConfigKey.FONT_MONO_CACHE_VERSION, 0, type=int)

    def set_mono_font_cache_version(self, version: int) -> None:
        self._settings.setValue(ConfigKey.FONT_MONO_CACHE_VERSION, version)

    def get_mono_font_name(self) -> str:
        return self._settings.value(ConfigKey.FONT_MONO_NAME, "", type=str)

    def set_mono_font_name(self, name: str) -> None:
        self._settings.setValue(ConfigKey.FONT_MONO_NAME, name)

    def is_auto_save_enabled(self) -> bool:
        return self._settings.value(ConfigKey.AUTO_SAVE_ENABLED, True, type=bool)

    def set_auto_save_enabled(self, enabled: bool) -> None:
        self._settings.setValue(ConfigKey.AUTO_SAVE_ENABLED, enabled)

    def get_log_window_geometry(self) -> Any:
        return self._settings.value(ConfigKey.LOG_WINDOW_GEOMETRY)

    def set_log_window_geometry(self, geometry: Any) -> None:
        self._settings.setValue(ConfigKey.LOG_WINDOW_GEOMETRY, geometry)

    def get_last_template_id(self) -> str | None:
        return self._settings.value(ConfigKey.TEMPLATE_LAST_ID, None, type=str)

    def set_last_template_id(self, template_id: str | None) -> None:
        self._settings.setValue(ConfigKey.TEMPLATE_LAST_ID, template_id)

    def get_last_template_name(self) -> str | None:
        return self._settings.value(ConfigKey.TEMPLATE_LAST_NAME, None, type=str)

    def set_last_template_name(self, name: str | None) -> None:
        self._settings.setValue(ConfigKey.TEMPLATE_LAST_NAME, name)
