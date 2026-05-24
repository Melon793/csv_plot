"""模板存储层 - 负责文件系统的读写和监控"""

from __future__ import annotations
import os
import re
import yaml
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional
from PySide6.QtCore import QObject, QFileSystemWatcher, Signal
from src.core.template_models import PlotTemplate, TemplateMetadata
from src.core.plot_config import TemplateStorageError
from src.core.logger import get_logger


logger = get_logger(__name__)


class TemplateStorage(QObject):
    """存储层，负责模板文件的 I/O 和监控"""

    directory_changed = Signal()
    file_changed = Signal(str)  # template_id

    def __init__(self, storage_path: Optional[Path] = None):
        super().__init__()
        if storage_path is None:
            storage_path = self._get_default_storage_path()
        self._storage_path = storage_path
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, PlotTemplate] = {}
        self._watcher = QFileSystemWatcher()
        self._watcher.directoryChanged.connect(self._on_directory_changed)
        self._watcher.fileChanged.connect(self._on_file_changed)
        self._watcher.addPath(str(self._storage_path))
        self.scan_directory()

    @staticmethod
    def _get_default_storage_path() -> Path:
        from PySide6.QtCore import QStandardPaths
        config_dir = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation))
        return config_dir / "templates"

    @staticmethod
    def _make_filename(name: str) -> str:
        """将模板名称转为安全文件名"""
        safe = re.sub(r'[\\/:*?"<>|]', "_", name)
        safe = safe.strip().replace(" ", "_")
        safe = re.sub(r'_+', '_', safe)
        if not safe:
            safe = "untitled"
        return safe[:120]

    def _find_file_by_id(self, template_id: str) -> Optional[Path]:
        """在目录中查找属于指定 ID 的 yaml 文件"""
        for f in self._storage_path.glob("*.yaml"):
            try:
                template = self.read_template_from_file(f)
                if template and template.metadata.id == template_id:
                    return f
            except Exception:
                pass
        return None

    def _id_from_cached_name(self, name: str) -> Optional[str]:
        """根据名称从缓存中反查 ID"""
        for tid, tpl in self._cache.items():
            if tpl.metadata.name == name:
                return tid
        return None

    def scan_directory(self) -> list[str]:
        """扫描目录，返回所有模板 ID"""
        template_ids = []
        self._cache.clear()
        try:
            for file in self._storage_path.glob("*.yaml"):
                try:
                    template = self.read_template_from_file(file)
                    if template:
                        self._cache[template.metadata.id] = template
                        template_ids.append(template.metadata.id)
                except Exception as e:
                    logger.warning(f"Failed to read template file {file}: {e}")
        except Exception as e:
            logger.error(f"Failed to scan template directory: {e}")
        return template_ids

    def read_template(self, template_id: str) -> Optional[PlotTemplate]:
        """读取指定模板"""
        if template_id in self._cache:
            return self._cache[template_id]
        f = self._find_file_by_id(template_id)
        if f:
            template = self.read_template_from_file(f)
            if template:
                self._cache[template_id] = template
                return template
        return None

    def read_template_from_file(self, file: Path) -> Optional[PlotTemplate]:
        """从文件读取模板"""
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if self._validate_template_data(data):
                return PlotTemplate.from_dict(data)
        except Exception as e:
            logger.error(f"Error reading template from {file}: {e}")
        return None

    def write_template(self, template: PlotTemplate) -> bool:
        """写入模板到文件（文件名 = 安全化名称.yaml）"""
        try:
            name = template.metadata.name
            new_filename = self._make_filename(name) + ".yaml"

            if template.metadata.id in self._cache:
                old = self._cache[template.metadata.id]
                old_filename = self._make_filename(old.metadata.name) + ".yaml"
                if old_filename != new_filename:
                    old_file = self._storage_path / old_filename
                    if old_file.exists():
                        old_file.unlink()

            file = self._storage_path / new_filename
            tmp_file = file.with_suffix(".yaml.tmp")
            with open(tmp_file, "w", encoding="utf-8") as f:
                yaml.dump(template.to_dict(), f, default_flow_style=False, allow_unicode=True, indent=2)
            os.replace(tmp_file, file)
            self._cache[template.metadata.id] = template
            return True
        except Exception as e:
            logger.error(f"Error writing template {template.metadata.id}: {e}")
            raise TemplateStorageError(f"Failed to write template: {e}")

    def delete_template(self, template_id: str) -> bool:
        """删除模板文件"""
        try:
            if template_id in self._cache:
                name = self._cache[template_id].metadata.name
                file = self._storage_path / (self._make_filename(name) + ".yaml")
            else:
                file = self._find_file_by_id(template_id)
            if file and file.exists():
                file.unlink()
            if template_id in self._cache:
                del self._cache[template_id]
            return True
        except Exception as e:
            logger.error(f"Error deleting template {template_id}: {e}")
            raise TemplateStorageError(f"Failed to delete template: {e}")

    def template_exists(self, template_id: str) -> bool:
        """检查模板是否存在"""
        if template_id in self._cache:
            return True
        return self._find_file_by_id(template_id) is not None

    def import_from_external(self, external_path: Path) -> Optional[PlotTemplate]:
        """从外部文件导入"""
        try:
            template = self.read_template_from_file(external_path)
            if template:
                new_id = self._generate_template_id()
                template.metadata.id = new_id
                template.metadata.source_file = str(external_path)
                template.metadata.created_at = datetime.now().isoformat()
                template.metadata.updated_at = datetime.now().isoformat()
                self.write_template(template)
                return template
        except Exception as e:
            logger.error(f"Error importing template from {external_path}: {e}")
        return None

    def export_to_external(self, template_id: str, target_path: Path) -> bool:
        """导出到外部位置"""
        try:
            template = self.read_template(template_id)
            if template:
                tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    yaml.dump(template.to_dict(), f, default_flow_style=False, allow_unicode=True, indent=2)
                os.replace(tmp_path, target_path)
                return True
        except Exception as e:
            logger.error(f"Error exporting template {template_id} to {target_path}: {e}")
        return False

    @staticmethod
    def _generate_template_id() -> str:
        return uuid.uuid4().hex[:8]

    @staticmethod
    def _validate_template_data(data: dict) -> bool:
        if not isinstance(data, dict):
            return False
        if "metadata" not in data or "config" not in data:
            return False
        metadata = data["metadata"]
        if not isinstance(metadata, dict):
            return False
        if "id" not in metadata or "name" not in metadata:
            return False
        config = data["config"]
        if not isinstance(config, dict):
            return False
        return True

    def _on_directory_changed(self, path: str):
        logger.debug(f"Directory changed: {path}")
        old_ids = set(self._cache.keys())
        self.scan_directory()
        new_ids = set(self._cache.keys())
        for tid in old_ids - new_ids:
            self.file_changed.emit(tid)
        for tid in new_ids - old_ids:
            self.file_changed.emit(tid)
        self.directory_changed.emit()

    def _on_file_changed(self, path: str):
        logger.debug(f"File changed: {path}")
        file = Path(path)
        if file.exists():
            template = self.read_template_from_file(file)
            if template:
                self._cache[template.metadata.id] = template
                self.file_changed.emit(template.metadata.id)
        else:
            for tid, tpl in list(self._cache.items()):
                if self._make_filename(tpl.metadata.name) + ".yaml" == file.name:
                    del self._cache[tid]
                    self.file_changed.emit(tid)
                    break
        self.directory_changed.emit()

    def get_all_templates(self) -> list[PlotTemplate]:
        return list(self._cache.values())

    def get_cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self):
        self._cache.clear()
