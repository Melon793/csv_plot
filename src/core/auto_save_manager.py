"""自动保存管理器 - 处理数据加载时的配置恢复"""

from __future__ import annotations
import tempfile
import os
import shutil
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from PySide6.QtCore import QObject, Signal
from src.core.plot_config import PlotSessionConfig
from src.core.settings import AppSettings
from src.core.logger import get_logger
from src.core.template_models import extract_variables_from_config


logger = get_logger(__name__)


@dataclass(frozen=True)
class PreparedAutoSave:
    should: bool
    reason: str
    config: PlotSessionConfig | None = None
    identity_snapshot: object | None = None
    failed: bool = False


def _config_has_placeholders(config) -> bool:
    """配置里是否存在失效占位（``invalid_curves`` 非空的子图）。

    纯占位配置（曲线全部失效）同样是布局资产，不应被当成「空配置」。
    """
    try:
        pages = config.normalized_pages()
    except AttributeError:
        return False
    for page in pages:
        for plot in getattr(page, "plots", None) or ():
            if getattr(plot, "invalid_curves", None):
                return True
    return False


class AutoSaveManager(QObject):
    """自动保存管理器"""

    # 信号
    match_result = Signal(bool, str)  # (是否匹配, 原因)
    config_applied = Signal()

    # 最小匹配比例（可配置）
    MIN_MATCH_RATIO: float = 0.6

    def __init__(self, storage_path: Optional[Path] = None):
        super().__init__()
        if storage_path is None:
            storage_path = AppSettings().config_dir
        self._storage_path = storage_path
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._auto_save_file = self._storage_path / "auto_save.yaml"
        self._auto_save_backup_file = self._storage_path / "auto_save.yaml.backup"

    def is_auto_save_enabled(self) -> bool:
        return AppSettings().is_auto_save_enabled()

    def set_auto_save_enabled(self, enabled: bool):
        settings = AppSettings()
        settings.set_auto_save_enabled(enabled)
        settings.sync()

    def auto_save(self, config: PlotSessionConfig) -> bool:
        """自动保存当前配置，返回是否写入成功。"""
        tmp_files: list[Path] = []
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self._storage_path,
                prefix=".auto-save-", suffix=".tmp", delete=False,
            ) as f:
                tmp_file = Path(f.name)
                tmp_files.append(tmp_file)
                yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True, indent=2)

            # 主档和备份都先写私有临时文件，避免并发保存交错覆盖同一 inode。
            if self._auto_save_file.exists():
                with tempfile.NamedTemporaryFile(
                    dir=self._storage_path, prefix=".auto-save-backup-",
                    suffix=".tmp", delete=False,
                ) as backup_tmp:
                    backup_file = Path(backup_tmp.name)
                tmp_files.append(backup_file)
                shutil.copy2(self._auto_save_file, backup_file)
                os.replace(backup_file, self._auto_save_backup_file)
                tmp_files.remove(backup_file)

            os.replace(tmp_file, self._auto_save_file)
            tmp_files.remove(tmp_file)
            logger.debug("Auto-saved config successfully")
            return True
        except Exception as e:
            logger.error("Failed to auto-save config: %s", e, exc_info=True)
            for tmp_file in tmp_files:
                try:
                    tmp_file.unlink(missing_ok=True)
                except OSError:
                    logger.debug("清理临时自动保存文件失败", exc_info=True)
            return False

    def load_auto_save(self) -> Optional[PlotSessionConfig]:
        """加载自动保存的配置；主文件存在但损坏时继续尝试 backup。"""
        for path in (self._auto_save_file, self._auto_save_backup_file):
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                config = PlotSessionConfig.from_dict(data)
                if path == self._auto_save_backup_file:
                    logger.warning("Auto-save primary unavailable, loaded backup")
                return config
            except Exception as e:
                logger.error("Failed to load auto-saved config %s: %s", path.name, e, exc_info=True)
        return None

    def prepare_auto_save(
        self, current_vars: list[str], loader=None
    ) -> PreparedAutoSave:
        """一次读盘并解析身份，返回放行决策与可复用快照。"""
        if not self.is_auto_save_enabled():
            return PreparedAutoSave(False, "Auto-save is disabled")

        try:
            had_saved_file = (
                self._auto_save_file.exists() or self._auto_save_backup_file.exists()
            )
            config = self.load_auto_save()
        except Exception as exc:
            logger.error("Failed to prepare auto-saved config: %s", exc, exc_info=True)
            return PreparedAutoSave(False, "Auto-save read failed", failed=True)
        if config is None:
            return PreparedAutoSave(
                False, "No auto-saved config", failed=had_saved_file
            )

        from src.core.template_identity import resolve_identity_snapshot

        snapshot = resolve_identity_snapshot(config, loader)
        config = snapshot.config
        demoted = snapshot.demoted
        config_vars = extract_variables_from_config(config)
        current_vars_set = set(current_vars)

        if not config_vars and not demoted:
            if not _config_has_placeholders(config):
                return PreparedAutoSave(
                    False, "Auto-saved config has no variables", config, snapshot
                )
            return PreparedAutoSave(
                True, "Auto-saved config has placeholders only, applying", config, snapshot
            )

        if not current_vars_set:
            return PreparedAutoSave(
                True, "No current variables, applying auto-saved config", config, snapshot
            )

        matched = len(config_vars & current_vars_set)
        total = len(config_vars | set(demoted))
        ratio = matched / total if total > 0 else 0.0
        if ratio >= self.MIN_MATCH_RATIO:
            return PreparedAutoSave(
                True, f"Match ratio {ratio:.0%}, applying auto-saved config", config, snapshot
            )
        if ratio > 0:
            return PreparedAutoSave(
                False,
                f"Match ratio only {ratio:.0%}, requires ≥{self.MIN_MATCH_RATIO:.0%}",
                config,
                snapshot,
            )
        return PreparedAutoSave(False, "No matching variables", config, snapshot)

    def should_apply_auto_save(self, current_vars: list[str], loader=None) -> Tuple[bool, str]:
        """兼容旧二元返回接口。"""
        decision = self.prepare_auto_save(current_vars, loader)
        return decision.should, decision.reason

