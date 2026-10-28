"""Plot 会话配置数据结构"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class PlotConfig:
    """单个 Plot 的配置

    ``invalid_curves`` 是失效占位的变量名（方案 B2，§8.1）：仅存名字，颜色随
    ``InvalidCurveRef`` 在运行期保存、不进模板（与"模板不含颜色"现状一致）。
    可选字段——序列化时仅在非空才写键，旧模板/旧 auto-save 缺省读作空列表，
    旧版软件读新模板忽略未知键，往返零影响。
    """
    curves: list[str] = field(default_factory=list)  # 曲线变量名列表
    invalid_curves: list[str] = field(default_factory=list)  # 失效占位变量名列表

    def to_dict(self) -> dict:
        """序列化为字典（invalid_curves 仅非空才写，保持 v1 视图与旧档兼容）"""
        data = {"curves": self.curves}
        if self.invalid_curves:
            data["invalid_curves"] = list(self.invalid_curves)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PlotConfig":
        """从字典反序列化（缺省 invalid_curves 容错为空列表）"""
        return cls(
            curves=data.get("curves", []),
            invalid_curves=data.get("invalid_curves", []) or [],
        )


@dataclass
class PlotPageConfig:
    """单页配置（模板 schema v2，设计文档 §6.5）。"""
    title: str = ""
    layout_rows: int = 1
    layout_cols: int = 1
    #: 行高百分比；JSON/YAML 反序列化后键可能是字符串，由 from_dict 还原
    row_height_factors: dict[int, int] = field(default_factory=dict)
    plots: list[PlotConfig] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "layout_rows": self.layout_rows,
            "layout_cols": self.layout_cols,
            "row_height_factors": {
                int(row): int(value)
                for row, value in (self.row_height_factors or {}).items()
            },
            "plots": [p.to_dict() for p in self.plots],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlotPageConfig":
        # JSON 会把 int 键序列化成字符串；YAML 正常，但旧档可能被手工编辑过
        factors = {
            int(row): int(value)
            for row, value in (data.get("row_height_factors") or {}).items()
        }
        return cls(
            title=str(data.get("title", "") or ""),
            layout_rows=int(data.get("layout_rows", 1) or 1),
            layout_cols=int(data.get("layout_cols", 1) or 1),
            row_height_factors=factors,
            plots=[PlotConfig.from_dict(p) for p in data.get("plots", []) or []],
        )


@dataclass
class PlotSessionConfig:
    """完整的 Plot 会话配置。

    v1（旧模板）：只有顶层 ``layout_rows/cols + plots``；
    v2（多标签页）：``pages`` 非空即按页读取，顶层字段保留为第 1 页的
    回退视图，让旧版软件仍能打开新模板（设计文档 §6.5）。
    """
    created_at: str = ""

    # 全局布局（v1 字段；v2 时为第 1 页的回退视图）
    layout_rows: int = 1
    layout_cols: int = 1

    # 全局设置
    time_factor: float = 1.0
    time_offset: float = 0.0

    # 各 Plot 配置（v1 字段；v2 时与 pages[0].plots 保持一致）
    plots: list[PlotConfig] = field(default_factory=list)

    # v2：多页配置；为空即视为 v1
    pages: list[PlotPageConfig] = field(default_factory=list)
    active_page: int = 0

    # v2 可选视图态（缺省 None = 应用时保留当前值，不覆盖）
    cursor_mode: str | None = None
    saved_mark_range: list[float] | None = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    @classmethod
    def is_v2(cls, data: dict) -> bool:
        """只看键不看 version：旧版软件写出的 v1 档没有 ``pages``。"""
        return (
            isinstance(data, dict)
            and isinstance(data.get("pages"), list)
            and bool(data.get("pages"))
        )

    def normalized_pages(self) -> list[PlotPageConfig]:
        """v2 返回自身 pages；v1 把顶层字段包成单页。"""
        if self.pages:
            return list(self.pages)
        return [
            PlotPageConfig(
                title="图 1",
                layout_rows=int(self.layout_rows),
                layout_cols=int(self.layout_cols),
                plots=list(self.plots),
            )
        ]

    def to_dict(self) -> dict:
        """只写 v2 结构，同时保留顶层 v1 回退视图。"""
        pages = self.normalized_pages()
        first = pages[0]
        return {
            "created_at": self.created_at,
            "layout_rows": first.layout_rows,
            "layout_cols": first.layout_cols,
            "time_factor": self.time_factor,
            "time_offset": self.time_offset,
            # 顶层 plots 是旧版软件的回退视图（第 1 页）
            "plots": [p.to_dict() for p in first.plots],
            "pages": [page.to_dict() for page in pages],
            "active_page": int(self.active_page),
            "cursor_mode": self.cursor_mode,
            "saved_mark_range": (
                list(self.saved_mark_range)
                if self.saved_mark_range is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlotSessionConfig":
        """从字典反序列化；无 ``pages`` 键按 v1 读取。"""
        pages = [
            PlotPageConfig.from_dict(page)
            for page in (data.get("pages") or [])
            if isinstance(page, dict)
        ]
        plots = [PlotConfig.from_dict(p) for p in data.get("plots", []) or []]
        mark_range = data.get("saved_mark_range")
        return cls(
            created_at=data.get("created_at", ""),
            layout_rows=data.get("layout_rows", 1),
            layout_cols=data.get("layout_cols", 1),
            time_factor=data.get("time_factor", 1.0),
            time_offset=data.get("time_offset", 0.0),
            plots=plots,
            pages=pages,
            active_page=int(data.get("active_page", 0) or 0),
            cursor_mode=data.get("cursor_mode"),
            saved_mark_range=list(mark_range) if isinstance(mark_range, list) else None,
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
