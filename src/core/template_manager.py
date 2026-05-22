"""模板管理器 - 提供模板的完整业务逻辑"""

from __future__ import annotations
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from PySide6.QtCore import QObject, Signal
from src.core.storage import TemplateStorage
from src.core.template_models import PlotTemplate, TemplateMetadata
from src.core.plot_config import (
    PlotSessionConfig,
    TemplateError,
    TemplateNotFoundError,
    TemplateNameConflictError,
    TemplateValidationError,
    TemplateStorageError,
)
from src.core.logger import get_logger


logger = get_logger(__name__)


class TemplateManager(QObject):
    """模板管理器 - 提供模板的完整业务逻辑"""

    # 信号
    template_added = Signal(str)
    template_removed = Signal(str)
    template_updated = Signal(str)
    template_list_changed = Signal()

    def __init__(self, storage_path: Optional[Path] = None):
        super().__init__()
        self._storage = TemplateStorage(storage_path)
        self._storage.directory_changed.connect(self._on_directory_changed)
        self._storage.file_changed.connect(self._on_file_changed)

    def get_all_templates(self) -> list[PlotTemplate]:
        """获取所有模板（按更新时间倒序）"""
        templates = self._storage.get_all_templates()
        return sorted(
            templates,
            key=lambda t: t.metadata.updated_at,
            reverse=True,
        )

    def get_template(self, template_id: str) -> Optional[PlotTemplate]:
        """根据 ID 获取模板"""
        return self._storage.read_template(template_id)

    def search(self, keyword: str = "", min_variables: int = 0) -> list[PlotTemplate]:
        """搜索模板"""
        templates = self.get_all_templates()
        results = []
        keyword = keyword.lower()

        for template in templates:
            # 按名称或描述搜索
            name_match = keyword in template.metadata.name.lower()
            desc_match = keyword in template.metadata.description.lower()
            if keyword and not (name_match or desc_match):
                continue

            # 按变量数量筛选
            if min_variables > 0:
                var_count = self._count_variables(template.config)
                if var_count < min_variables:
                    continue

            results.append(template)
        return results

    def exists(self, name: str, exclude_id: Optional[str] = None) -> bool:
        """检查名称是否存在"""
        templates = self.get_all_templates()
        for template in templates:
            if template.metadata.name == name:
                if exclude_id is None or template.metadata.id != exclude_id:
                    return True
        return False

    def save_template(
        self,
        config: PlotSessionConfig,
        name: str,
        description: str = "",
        template_id: Optional[str] = None,
    ) -> PlotTemplate:
        """保存模板"""
        # 检查名称冲突
        if self.exists(name, exclude_id=template_id):
            raise TemplateNameConflictError(f"Template name '{name}' already exists")

        if template_id:
            # 更新现有模板
            existing = self._storage.read_template(template_id)
            if not existing:
                raise TemplateNotFoundError(f"Template {template_id} not found")
            existing.metadata.name = name
            existing.metadata.description = description
            existing.metadata.updated_at = datetime.now().isoformat()
            existing.config = config.to_dict()
            self._storage.write_template(existing)
            self.template_updated.emit(template_id)
            self.template_list_changed.emit()
            return existing
        else:
            # 创建新模板
            new_id = self._generate_template_id()
            metadata = TemplateMetadata(
                id=new_id,
                name=name,
                description=description,
            )
            template = PlotTemplate(
                metadata=metadata,
                config=config.to_dict(),
            )
            self._storage.write_template(template)
            self.template_added.emit(new_id)
            self.template_list_changed.emit()
            return template

    def rename_template(self, template_id: str, new_name: str) -> bool:
        """重命名模板"""
        template = self._storage.read_template(template_id)
        if not template:
            raise TemplateNotFoundError(f"Template {template_id} not found")

        if self.exists(new_name, exclude_id=template_id):
            raise TemplateNameConflictError(f"Template name '{new_name}' already exists")

        template.metadata.name = new_name
        template.metadata.updated_at = datetime.now().isoformat()
        self._storage.write_template(template)
        self.template_updated.emit(template_id)
        self.template_list_changed.emit()
        return True

    def delete_template(self, template_id: str) -> bool:
        """删除模板"""
        template = self._storage.read_template(template_id)
        if not template:
            raise TemplateNotFoundError(f"Template {template_id} not found")

        self._storage.delete_template(template_id)
        self.template_removed.emit(template_id)
        self.template_list_changed.emit()
        return True

    def duplicate_template(self, template_id: str, new_name: str) -> Optional[PlotTemplate]:
        """复制模板"""
        template = self._storage.read_template(template_id)
        if not template:
            raise TemplateNotFoundError(f"Template {template_id} not found")

        # 创建新模板，保留配置
        new_id = self._generate_template_id()
        new_metadata = TemplateMetadata(
            id=new_id,
            name=new_name,
            description=template.metadata.description,
            source_file=template.metadata.source_file,
        )
        new_template = PlotTemplate(
            metadata=new_metadata,
            config=template.config.copy(),
        )
        self._storage.write_template(new_template)
        self.template_added.emit(new_id)
        self.template_list_changed.emit()
        return new_template

    def import_template(self, external_path: Path) -> Optional[PlotTemplate]:
        """从外部文件导入"""
        template = self._storage.import_from_external(external_path)
        if template:
            # 检查名称是否冲突，必要时添加后缀
            base_name = template.metadata.name
            counter = 1
            while self.exists(template.metadata.name):
                template.metadata.name = f"{base_name} ({counter})"
                counter += 1
            self._storage.write_template(template)
            self.template_added.emit(template.metadata.id)
            self.template_list_changed.emit()
        return template

    def export_template(self, template_id: str, target_path: Path) -> bool:
        """导出到外部"""
        return self._storage.export_to_external(template_id, target_path)

    def _on_directory_changed(self):
        """目录变化处理"""
        self.template_list_changed.emit()

    def _on_file_changed(self, template_id: str):
        """单个文件变化处理"""
        template = self._storage.read_template(template_id)
        if template:
            self.template_updated.emit(template_id)
        else:
            self.template_removed.emit(template_id)
        self.template_list_changed.emit()

    @staticmethod
    def _generate_template_id() -> str:
        """生成 8 位 UUID 作为模板 ID"""
        return uuid.uuid4().hex[:8]

    @staticmethod
    def _count_variables(config: dict) -> int:
        """计算配置中的变量数量"""
        var_set = set()
        plots = config.get("plots", [])
        for plot in plots:
            curves = plot.get("curves", [])
            var_set.update(curves)
        return len(var_set)
