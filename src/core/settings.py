"""统一配置管理器 - 集中管理应用所有配置"""

from __future__ import annotations
import json
import logging
import os
import sys
import tempfile
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Optional, Sequence, Type

from PySide6.QtCore import QSettings

from src.utils.paths import normalize_input_path

_logger = logging.getLogger(__name__)


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
    # BLF 只存原始报文，必须搭配 DBC / ARXML 才能解出信号。记住上次选中的
    # 数据库列表（os.pathsep 连接），下次导入时在弹窗里**预选**（不再静默
    # 自动导入），让连续分析同一台架的多份 BLF 不必每次从零重选。
    BLF_LAST_DB_PATHS = "blf/last_db_paths"
    # 数据库文件的用户备注：JSON dict {规范化路径: 备注文本}。备注是自由文本，
    # 不能用 BLF_LAST_DB_PATHS 的 pathsep 连接（值里可能含分隔符或空格），
    # JSON 单字符串在 ini 后端里最稳。
    BLF_DB_NOTES = "blf/db_notes"
    # 标记统计窗口的已选指标列（JSON 数组字符串，有序；顺序即显示顺序）。
    # 合法 key 集合由窗口侧的指标注册表定义，settings 只做存储与防御性解析，
    # 不反向依赖 ui 层。
    MARK_STATS_COLUMNS = "mark_stats/columns"
    # 标记统计窗口表头状态（QHeaderView.saveState() 的 QByteArray，ini 后端
    # 存 base64）：列宽 + 视觉列序 + 隐藏位。
    MARK_STATS_HEADER_STATE = "mark_stats/header_state"


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
        except (PermissionError, OSError) as e:
            # 降级到系统临时目录，保证程序可运行
            fallback = Path(tempfile.gettempdir()) / "CSVPlot"
            try:
                fallback.mkdir(parents=True, exist_ok=True)
                config_dir = fallback
                _logger.warning(
                    "配置目录创建失败(%s)，降级到临时目录: %s", e, config_dir
                )
            except OSError:
                config_dir = fallback
                _logger.error("配置目录和临时目录均不可写，设置将无法持久化")

        self._settings = QSettings(
            str(config_dir / "app.ini"),
            QSettings.Format.IniFormat,
        )
        self._config_dir = config_dir
        self._migrate_if_needed()

    @staticmethod
    def _get_config_dir() -> Path:
        # 环境变量注入优先（供自动化测试重定向到临时目录，避免污染真实用户配置）
        override = os.environ.get("CSV_PLOT_CONFIG_DIR")
        if override:
            return Path(override)
        if sys.platform == "darwin":
            base = Path.home() / "Library" / "Application Support"
        elif sys.platform == "win32":
            base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
        else:
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
                old_settings = QSettings(org, app)
                for old_key, new_key in key_mappings:
                    val = old_settings.value(old_key)
                    if val is not None:
                        self._settings.setValue(new_key, val)

            self._settings.sync()
        except Exception as e:
            logging.getLogger(__name__).warning("配置迁移失败: %s", e)

    @classmethod
    def _reset_for_tests(cls) -> None:
        """仅供自动化测试使用：重置单例，使下次实例化时重新读取配置目录"""
        with cls._lock:
            cls._instance = None

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
        return self._settings.value(ConfigKey.AUTO_SAVE_ENABLED, False, type=bool)

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

    # ------------------------------------------------------------------
    # BLF / CAN 数据库记忆
    #
    # 刻意以「单个字符串 + os.pathsep 连接」而非 QStringList 存储：QSettings 的
    # ini 后端对列表值会写成带计数的复合键，跨平台读取时的类型不稳定（实测
    # 单元素列表返回 str、多元素返回 list[str]）。pathsep 方案与 PATH 语义一致，
    # 且 Windows 下的盘符反斜杠不会破坏 ini。
    # ------------------------------------------------------------------

    def get_blf_last_db_paths(self) -> list[str]:
        raw = self._settings.value(ConfigKey.BLF_LAST_DB_PATHS, "", type=str) or ""
        return [p for p in raw.split(os.pathsep) if p.strip()]

    def set_blf_last_db_paths(self, paths: list[str]) -> None:
        self._settings.setValue(
            ConfigKey.BLF_LAST_DB_PATHS, os.pathsep.join(str(p) for p in paths)
        )

    # ------------------------------------------------------------------
    # CAN 数据库备注（用户在选库对话框里给 DBC / ARXML 写的说明）
    #
    # 键一律走 normalize_input_path：文件对话框 / 拖拽 / 记忆列表给到的写法
    # 可能不同（~ 展开、分隔符差异），归一后才能命中同一条备注。
    # ------------------------------------------------------------------

    def get_blf_db_notes(self) -> dict[str, str]:
        """全部数据库备注。坏 JSON / 类型不符一律返回空 dict，不让设置脏数据拖垮 UI。"""
        raw = self._settings.value(ConfigKey.BLF_DB_NOTES, "", type=str) or ""
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
        except ValueError:
            _logger.warning("CAN 数据库备注解析失败，已忽略: %.120s", raw)
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items() if str(v).strip()}

    def get_blf_db_note(self, path: str) -> str:
        """单个数据库文件的备注；无备注返回空串。"""
        return self.get_blf_db_notes().get(normalize_input_path(path), "")

    def set_blf_db_note(self, path: str, note: str) -> None:
        """写单条备注；清空（全空白）即删除条目，避免 ini 无限膨胀。"""
        key = normalize_input_path(path)
        if not key:
            return
        notes = self.get_blf_db_notes()
        text = str(note).strip()
        if text:
            notes[key] = text
        else:
            notes.pop(key, None)
        self._settings.setValue(
            ConfigKey.BLF_DB_NOTES, json.dumps(notes, ensure_ascii=False)
        )

    # ------------------------------------------------------------------
    # 标记统计窗口（MarkStatsWindow）持久化
    #
    # 列配置走 JSON-in-ini（同 BLF_DB_NOTES 模式）；表头状态直接存
    # QByteArray（QSettings 原生支持，ini 后端写 base64）。写入即 sync()，
    # 崩溃也不丢。读取防御四级：① JSON 解析失败 → 默认集；② 过滤注册表
    # 外的 key（防手改 ini / 未来版本删指标）；③ 结果为空 → 默认集；
    # ④ 去重且保持存储顺序。合法 key 集与默认集由调用方（窗口侧注册表）
    # 传入，settings 不依赖 ui 层。
    # ------------------------------------------------------------------

    def get_mark_stats_columns(
        self, default_keys: list[str], valid_keys: "Iterable[str]"
    ) -> list[str]:
        """读取已选指标列（有序）。任何损坏/非法输入都降级到默认集。"""
        raw = self._settings.value(ConfigKey.MARK_STATS_COLUMNS, "", type=str) or ""
        keys: list[str] = []
        if raw.strip():
            try:
                data = json.loads(raw)
            except ValueError:
                _logger.warning(
                    "标记统计列配置解析失败，回退默认集: %.120s", raw
                )
                data = None
            if isinstance(data, list):
                valid = set(valid_keys)
                seen: set[str] = set()
                for item in data:
                    if isinstance(item, str) and item in valid and item not in seen:
                        seen.add(item)
                        keys.append(item)
        return keys if keys else list(default_keys)

    def set_mark_stats_columns(self, keys: "Sequence[str]") -> None:
        """写入已选指标列（顺序即显示顺序），即写即持久化。"""
        self._settings.setValue(
            ConfigKey.MARK_STATS_COLUMNS,
            json.dumps(list(keys), ensure_ascii=False),
        )
        self._settings.sync()

    def get_mark_stats_header_state(self) -> Any:
        """读取表头状态；无存档/空值返回 None，由窗口侧静默走默认列宽。"""
        state = self._settings.value(ConfigKey.MARK_STATS_HEADER_STATE)
        # 键不存在 → None；被清成空串 / 空 QByteArray 同样视为无存档
        if not state:
            return None
        return state

    def set_mark_stats_header_state(self, state: Any) -> None:
        """写入表头状态（QHeaderView.saveState() 的 QByteArray），即写即持久化。"""
        self._settings.setValue(ConfigKey.MARK_STATS_HEADER_STATE, state)
        self._settings.sync()
