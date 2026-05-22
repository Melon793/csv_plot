"""Plot 会话配置数据结构"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class PlotConfig:
    """单个 Plot 的配置"""
    curves: list[str] = field(default_factory=list)  # 曲线变量名列表

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {"curves": self.curves}

    @classmethod
    def from_dict(cls, data: dict) -> "PlotConfig":
        """从字典反序列化"""
        return cls(curves=data.get("curves", []))


@dataclass
class PlotSessionConfig:
    """完整的 Plot 会话配置"""
    created_at: str = ""

    # 全局布局
    layout_rows: int = 1
    layout_cols: int = 1

    # 全局设置
    time_factor: float = 1.0
    time_offset: float = 0.0

    # 各 Plot 配置（按 row-major 顺序）
    plots: list[PlotConfig] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "created_at": self.created_at,
            "layout_rows": self.layout_rows,
            "layout_cols": self.layout_cols,
            "time_factor": self.time_factor,
            "time_offset": self.time_offset,
            "plots": [p.to_dict() for p in self.plots],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlotSessionConfig":
        """从字典反序列化"""
        return cls(
            created_at=data.get("created_at", ""),
            layout_rows=data.get("layout_rows", 1),
            layout_cols=data.get("layout_cols", 1),
            time_factor=data.get("time_factor", 1.0),
            time_offset=data.get("time_offset", 0.0),
            plots=[PlotConfig.from_dict(p) for p in data.get("plots", [])],
        )


# 异常定义
class TemplateError(Exception):
    """模板相关异常基类"""
    pass


class TemplateNotFoundError(TemplateError):
    """模板不存在"""
    pass


class TemplateNameConflictError(TemplateError):
    """模板名称冲突"""
    pass


class TemplateValidationError(TemplateError):
    """模板验证失败"""
    pass


class TemplateStorageError(TemplateError):
    """存储操作失败"""
    pass
