"""统一配置管理器 - 集中管理应用所有配置"""

from __future__ import annotations
import json
import logging
import os
import sys
import tempfile
import time
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
    # 绘图变量编辑器窗口几何（QByteArray）：位置 + 大小，失效屏幕
    # （多显示器拔插后位置落已不存在的屏幕）由窗口侧校验拦截并回退
    PLOT_VARIABLE_EDITOR_GEOMETRY = "plot_variable_editor/geometry"
    # 编辑器列宽（list[int]，6 列）与用户拖拽固定列（list[int]）：跨重启记忆，
    # 不进模板（列宽是个人视图偏好，D1/D10）。类型不符视为无存档。
    PLOT_VARIABLE_EDITOR_COLUMN_WIDTHS = "plot_variable_editor/column_widths"
    PLOT_VARIABLE_EDITOR_FIXED_COLS = "plot_variable_editor/fixed_cols"
    # 曲线暂存区浮窗几何（QByteArray）：跨重启记忆；失效屏幕由窗口侧守卫
    CURVE_SHELF_GEOMETRY = "curve_shelf/geometry"
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
    # 按日志文件记住的完整配置（JSON dict）：键 = 规范化路径，值 =
    # {size, mtime, ts, config}。config 是 CanBindings.to_yaml_dict() 的形状
    # （有序池 + 1 基通道绑定 + 别名）。**键用路径而不是路径+size+mtime**：
    # 现场常把日志从共享盘拷到本地，mtime 必变，三元组作键几乎不命中；
    # size/mtime 只用于「同一路径的文件换了」的确认提示（评审 F10）。
    # 只保留最近 20 份（按 ts），ini 不会无限膨胀。
    BLF_FILE_BINDINGS = "blf/file_bindings"
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

    def get_plot_variable_editor_geometry(self) -> Any:
        """读取编辑器窗口几何；无存档/空值/类型不符返回 None。

        类型防御：ini 里被写成纯字符串（非 @ByteArray 格式）时
        QSettings 返回 str，直接交给 restoreGeometry 会抛 TypeError，
        视为无存档由窗口侧走初始几何。
        """
        from PySide6.QtCore import QByteArray

        geometry = self._settings.value(ConfigKey.PLOT_VARIABLE_EDITOR_GEOMETRY)
        # 键不存在 → None；被清成空串 / 空 QByteArray 同样视为无存档
        if not geometry:
            return None
        if not isinstance(geometry, (QByteArray, bytes, bytearray)):
            return None
        return geometry

    def set_plot_variable_editor_geometry(self, geometry: Any) -> None:
        self._settings.setValue(ConfigKey.PLOT_VARIABLE_EDITOR_GEOMETRY, geometry)

    def get_plot_variable_editor_column_widths(self) -> Any:
        """读取编辑器列宽存档；无存档/类型不符/长度非法返回 None。

        防御口径同 geometry：ini 脏值（str/含非 int/负值）一律视为无存档，
        由窗口侧走默认初值，避免 restore 时抛异常或落极端窄列。
        """
        widths = self._settings.value(ConfigKey.PLOT_VARIABLE_EDITOR_COLUMN_WIDTHS)
        if not isinstance(widths, (list, tuple)) or not widths:
            return None
        out = []
        for w in widths:
            if isinstance(w, bool) or not isinstance(w, int) or w <= 0:
                return None
            out.append(int(w))
        return out

    def set_plot_variable_editor_column_widths(self, widths: Any) -> None:
        self._settings.setValue(
            ConfigKey.PLOT_VARIABLE_EDITOR_COLUMN_WIDTHS, [int(w) for w in widths]
        )

    def get_plot_variable_editor_fixed_cols(self) -> Any:
        """读取用户拖拽固定列集合存档；类型不符返回 None（同列宽口径）。"""
        cols = self._settings.value(ConfigKey.PLOT_VARIABLE_EDITOR_FIXED_COLS)
        if not isinstance(cols, (list, tuple)):
            return None
        out = []
        for c in cols:
            if isinstance(c, bool) or not isinstance(c, int) or c < 0:
                return None
            out.append(int(c))
        return out

    def set_plot_variable_editor_fixed_cols(self, cols: Any) -> None:
        self._settings.setValue(
            ConfigKey.PLOT_VARIABLE_EDITOR_FIXED_COLS, [int(c) for c in cols]
        )

    def get_curve_shelf_geometry(self) -> Any:
        """读取暂存区浮窗几何；无存档/空值/类型不符返回 None（同编辑器口径）。"""
        from PySide6.QtCore import QByteArray

        geometry = self._settings.value(ConfigKey.CURVE_SHELF_GEOMETRY)
        if not geometry:
            return None
        if not isinstance(geometry, (QByteArray, bytes, bytearray)):
            return None
        return geometry

    def set_curve_shelf_geometry(self, geometry: Any) -> None:
        self._settings.setValue(ConfigKey.CURVE_SHELF_GEOMETRY, geometry)

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
    # BLF 按文件记忆的完整配置（池 + 通道绑定 + 别名）
    # ------------------------------------------------------------------

    BLF_FILE_BINDINGS_LIMIT = 20

    @staticmethod
    def blf_memory_key(blf_path: str) -> str:
        """按文件记忆的键：与通道绑定同口径（normcase + 绝对路径）。

        评审 P2：此前用 ``normalize_input_path``（不做大小写归一），Windows /
        UNC 上同一文件的不同写法会漏命中，与 ``_path_key`` 口径分裂。
        """
        return os.path.normcase(os.path.abspath(normalize_input_path(blf_path)))

    def get_blf_file_bindings(self) -> dict[str, dict]:
        """``{规范化路径: {size, mtime, ts, config}}``。坏数据一律空 dict。"""
        raw = self._settings.value(ConfigKey.BLF_FILE_BINDINGS, "", type=str) or ""
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
        except ValueError:
            _logger.warning("BLF 文件配置记忆解析失败，已忽略: %.120s", raw)
            return {}
        if not isinstance(data, dict):
            return {}
        out: dict[str, dict] = {}
        for key, entry in data.items():
            if isinstance(entry, dict) and isinstance(entry.get("config"), dict):
                out[str(key)] = entry
        return out

    def remember_blf_file_binding(
        self, blf_path: str, *, size: int, mtime: float, config: dict
    ) -> None:
        """记住某份日志的配置；LRU（按 ts）保留最近 ``BLF_FILE_BINDINGS_LIMIT`` 份。"""
        key = self.blf_memory_key(blf_path)
        if not key or not isinstance(config, dict):
            return
        entries = self.get_blf_file_bindings()
        entries[key] = {
            "size": int(size),
            "mtime": float(mtime),
            "ts": time.time(),
            "config": config,
        }
        if len(entries) > self.BLF_FILE_BINDINGS_LIMIT:
            newest = sorted(
                entries.items(),
                key=lambda kv: float(kv[1].get("ts") or 0.0),
                reverse=True,
            )[: self.BLF_FILE_BINDINGS_LIMIT]
            entries = dict(newest)
        self._settings.setValue(
            ConfigKey.BLF_FILE_BINDINGS,
            json.dumps(entries, ensure_ascii=False),
        )

    def forget_blf_file_binding(self, blf_path: str) -> None:
        key = self.blf_memory_key(blf_path)
        entries = self.get_blf_file_bindings()
        if key in entries:
            entries.pop(key, None)
            self._settings.setValue(
                ConfigKey.BLF_FILE_BINDINGS,
                json.dumps(entries, ensure_ascii=False),
            )

    @staticmethod
    def blf_binding_store_path(blf_path: str) -> str:
        """默认落点：用户配置目录下按日志路径哈希命名的 YAML（跨只读盘可用）。"""
        import hashlib

        from PySide6.QtCore import QStandardPaths

        base = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation
        ) or tempfile.gettempdir()
        folder = os.path.join(base, "can_bindings")
        digest = hashlib.sha1(
            os.path.normcase(os.path.abspath(blf_path)).encode("utf-8", "replace")
        ).hexdigest()[:16]
        return os.path.join(folder, f"{digest}.canview.yaml")

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
