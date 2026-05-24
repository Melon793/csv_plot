"""模板数据模型"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from pathlib import Path


@dataclass
class TemplateMetadata:
    """模板元数据"""

    id: str  # 唯一标识 (8位 UUID)
    name: str  # 显示名称
    description: str = ""

    created_at: str = ""
    updated_at: str = ""

    source_file: Optional[str] = None

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_file": self.source_file,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TemplateMetadata":
        """从字典反序列化"""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            source_file=data.get("source_file"),
        )


@dataclass
class PlotTemplate:
    """完整的模板"""

    metadata: TemplateMetadata
    config: dict  # PlotSessionConfig 的字典形式

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "metadata": self.metadata.to_dict(),
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlotTemplate":
        return cls(
            metadata=TemplateMetadata.from_dict(data["metadata"]),
            config=data["config"],
        )


def count_template_variables(config: dict) -> int:
    var_set = set()
    plots = config.get("plots", []) or []
    for plot in plots:
        curves = (plot or {}).get("curves", []) or []
        var_set.update(curves)
    return len(var_set)
