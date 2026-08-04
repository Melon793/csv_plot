"""自动保存管理器 - 处理数据加载时的配置恢复"""

from __future__ import annotations
import os
import shutil
import yaml
from pathlib import Path
from typing import Optional, Tuple
from PySide6.QtCore import QObject, Signal
from src.core.plot_config import PlotSessionConfig
from src.core.logger import get_logger


logger = get_logger(__name__)


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
            from src.core.settings import AppSettings
            storage_path = AppSettings().config_dir
        self._storage_path = storage_path
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._auto_save_file = self._storage_path / "auto_save.yaml"
        self._auto_save_backup_file = self._storage_path / "auto_save.yaml.backup"

    def is_auto_save_enabled(self) -> bool:
        from src.core.settings import AppSettings
        return AppSettings().is_auto_save_enabled()

    def set_auto_save_enabled(self, enabled: bool):
        from src.core.settings import AppSettings
        settings = AppSettings()
        settings.set_auto_save_enabled(enabled)
        settings.sync()

    def auto_save(self, config: PlotSessionConfig):
        """自动保存当前配置"""
        tmp_file = None
        try:
            tmp_file = self._auto_save_file.with_suffix(".yaml.tmp")
            with open(tmp_file, "w", encoding="utf-8") as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True, indent=2)

            # 用 copy2 备份主文件（主文件始终保留，避免 rename 造成的"主文件不存在"窗口），
            # 再原子替换主文件；即使中途崩溃，主文件与备份至少有一个可用
            if self._auto_save_file.exists():
                shutil.copy2(self._auto_save_file, self._auto_save_backup_file)

            os.replace(tmp_file, self._auto_save_file)
            logger.debug("Auto-saved config successfully")
        except Exception as e:
            logger.error(f"Failed to auto-save config: {e}")
            if tmp_file is not None and tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass

    def load_auto_save(self) -> Optional[PlotSessionConfig]:
        """加载自动保存的配置"""
        try:
            if self._auto_save_file.exists():
                with open(self._auto_save_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                return PlotSessionConfig.from_dict(data)
            elif self._auto_save_backup_file.exists():
                logger.warning("Auto-save file not found, trying backup")
                with open(self._auto_save_backup_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                return PlotSessionConfig.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load auto-saved config: {e}")
        return None

    def should_apply_auto_save(self, current_vars: list[str]) -> Tuple[bool, str]:
        """判断是否应应用自动保存的配置"""
        if not self.is_auto_save_enabled():
            return False, "Auto-save is disabled"

        config = self.load_auto_save()
        if not config:
            return False, "No auto-saved config"

        config_vars = self._extract_variables(config)
        current_vars_set = set(current_vars)

        if not config_vars:
            return False, "Auto-saved config has no variables"

        if not current_vars_set:
            return True, "No current variables, applying auto-saved config"

        # 计算匹配度：分母为配置中的变量数（与模板匹配语义一致）
        matched = len(config_vars & current_vars_set)
        total = len(config_vars)
        ratio = matched / total if total > 0 else 0.0

        if ratio >= self.MIN_MATCH_RATIO:
            return True, f"Match ratio {ratio:.0%}, applying auto-saved config"
        elif ratio > 0:
            return False, f"Match ratio only {ratio:.0%}, requires ≥{self.MIN_MATCH_RATIO:.0%}"
        else:
            return False, "No matching variables"

    @staticmethod
    def _extract_variables(config: PlotSessionConfig) -> set[str]:
        """从配置中提取所有变量名"""
        var_set = set()
        for plot in config.plots:
            for v in plot.curves:
                var_set.add(v)
        return var_set
