"""模板数据模型"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from src.core.plot_config import PlotSessionConfig


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


def _iter_config_plots(config: dict):
    """按 v2 ``pages`` 优先、v1 顶层 ``plots`` 回退，逐个子图配置产出。"""
    pages = config.get("pages") if isinstance(config, dict) else None
    if isinstance(pages, list) and pages:
        for page in pages:
            for plot in (page or {}).get("plots", []) or []:
                yield plot or {}
        return
    for plot in config.get("plots", []) or []:
        yield plot or {}


def count_template_variables(config: dict) -> int:
    var_set = set()
    for plot in _iter_config_plots(config):
        curves = plot.get("curves", []) or []
        var_set.update(curves)
    return len(var_set)


def count_template_plots(config: dict) -> int:
    """模板包含的非空子图数（v2 按全部页统计）。"""
    return sum(1 for plot in _iter_config_plots(config) if plot.get("curves"))


def count_template_pages(config: dict) -> int:
    """模板包含的绘图页数；v1 档固定为 1。"""
    pages = config.get("pages") if isinstance(config, dict) else None
    if isinstance(pages, list) and pages:
        return len(pages)
    return 1


def extract_variables_from_config(config: "PlotSessionConfig | dict") -> set[str]:
    """从配置中提取所有变量名（统一实现，支持 PlotSessionConfig 和 dict 两种格式）。

    Args:
        config: PlotSessionConfig 对象或 dict 格式的配置

    Returns:
        变量名的集合
    """
    var_set: set[str] = set()

    if isinstance(config, dict):
        for plot in _iter_config_plots(config):
            var_set.update(plot.get("curves", []) or [])
    else:
        for page in config.normalized_pages():
            for plot in page.plots:
                var_set.update(plot.curves)

    return var_set
