"""Plot 会话配置数据结构"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class PlotConfig:
    """单个 Plot 的配置

    ``invalid_curves`` 是失效占位的变量名（方案 B2，§8.1）：仅存名字，
    占位运行期颜色随 InvalidCurveRef、模板来源占位固定灰（§8.3/§12.2）。
    可选字段——序列化时仅在非空才写键，旧模板/旧 auto-save 缺省读作空列表，
    旧版软件读新模板忽略未知键，往返零影响。

    ``curve_styles`` 是曲线外观束（2026-11-21 边界决策：外观升级为布局
    资产，取代原"模板不含颜色"声明）：变量名 → {color 恒写, line_preset /
    draw_mode / symbol_policy 仅偏离默认值才写}。模板与 auto-save 共用，
    跨会话/跨机器复现强调布局；缺键 = 色板轮转等默认行为（旧档零回归）。
    """
    curves: list[str] = field(default_factory=list)  # 曲线变量名列表
    invalid_curves: list[str] = field(default_factory=list)  # 失效占位变量名列表
    #: 曲线外观束（可选键，仅非空才写）：见类 docstring 与
    #: .qoder/plans/curve_styles_template_dev.md §2
    curve_styles: dict[str, dict] = field(default_factory=dict)
    #: P1-L（上轮 P1-10）：本格在页内的真实格子索引（row*grid_cols+col）。
    #: 空可见格子不进导出列表，套用端若按 enumerate 对位会把后面的曲线整体
    #: 前移错位。可选字段——仅非 None 才写键；旧档缺省 None，套用端回退
    #: enumerate 对位（旧行为，零回归）。
    cell_index: int | None = None
    #: P1-7（第 5 轮）：格子的 (行, 列) 坐标。cell_index 是导出端「已建网格」
    #: 的扁平索引，built stride 与可见列数不一致时（扩后缩、页 1 满容量建矩阵、
    #: 页重排）跨会话恢复会错位/越界丢弃。row/col 与 stride 无关，套用端按目标
    #: 页 grid_cols 换算。可选字段——仅非 None 才写键；旧档缺省回退 cell_index
    #: （旧行为，零回归）。
    cell_row: int | None = None
    cell_col: int | None = None

    def to_dict(self) -> dict:
        """序列化为字典（invalid_curves/cell_index/cell_row/cell_col/curve_styles 仅有值才写，保持旧档兼容）"""
        data = {"curves": self.curves}
        if self.invalid_curves:
            data["invalid_curves"] = list(self.invalid_curves)
        if self.cell_index is not None:
            data["cell_index"] = int(self.cell_index)
        if self.cell_row is not None and self.cell_col is not None:
            data["cell_row"] = int(self.cell_row)
            data["cell_col"] = int(self.cell_col)
        if self.curve_styles:
            data["curve_styles"] = {
                name: dict(styles) for name, styles in self.curve_styles.items()
            }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PlotConfig":
        """从字典反序列化（缺省 invalid_curves 容错为空列表；cell_index/cell_row/cell_col 容错 None）"""
        raw_cell = data.get("cell_index")
        try:
            cell_index = None if raw_cell is None else int(raw_cell)
        except (TypeError, ValueError):
            cell_index = None
        try:
            raw_row, raw_col = data.get("cell_row"), data.get("cell_col")
            cell_row = None if raw_row is None else int(raw_row)
            cell_col = None if raw_col is None else int(raw_col)
        except (TypeError, ValueError):
            cell_row = cell_col = None
        curves = data.get("curves") or []
        if not isinstance(curves, list):
            # 评审 B1：curves: null / 标量不得让下游 set.update 抛 TypeError
            curves = []
        # 复审 BL-3：元素级守卫——数字通道名（YAML 未加引号）str 化后仍能
        # 与 var_names 匹配；null 丢弃；unhashable（list/dict）str 化避免
        # check_template_match / _show_match_low_dialog 在入口 try 外抛 TypeError
        curves = [str(c) for c in curves if c is not None]
        invalid = data.get("invalid_curves") or []
        if not isinstance(invalid, list):
            invalid = []
        invalid = [str(c) for c in invalid if c is not None]
        return cls(
            curves=curves,
            invalid_curves=invalid,
            cell_index=cell_index,
            cell_row=cell_row,
            cell_col=cell_col,
            curve_styles=cls._parse_curve_styles(data.get("curve_styles")),
        )

    @staticmethod
    def _parse_curve_styles(raw: Any) -> dict[str, dict]:
        """外观束逐字段守卫（B1/BL-3 风格）：畸形手写档不得让异常逃出槽。

        preset/policy 的 auto 值丢（= 默认值，无承载意义）；draw_mode 的
        'auto' **保留**——它只可能来自 user 钉住的显式直线（枚举通道用户
        override），套用端 apply 入口钉 source='user' 恰好复现该意图；
        丢弃会造成模板链独丢、与 reload/搬运链不一致且往返不幂等。
        """
        if not isinstance(raw, dict):
            return {}
        from PySide6.QtGui import QColor

        from src.core.data_types import (
            DRAW_MODES,
            LINE_PRESET_AUTO,
            LINE_PRESETS,
            SYMBOL_POLICY_AUTO,
            SYMBOL_POLICIES,
        )

        styles: dict[str, dict] = {}
        for name, entry_raw in raw.items():
            if not isinstance(entry_raw, dict):
                continue
            entry: dict = {}
            color = entry_raw.get("color")
            if isinstance(color, str) and QColor(color).isValid():
                entry["color"] = color
            preset = entry_raw.get("line_preset")
            if preset in LINE_PRESETS and preset != LINE_PRESET_AUTO:
                entry["line_preset"] = preset
            mode = entry_raw.get("draw_mode")
            if mode in DRAW_MODES:
                entry["draw_mode"] = mode
            policy = entry_raw.get("symbol_policy")
            if policy in SYMBOL_POLICIES and policy != SYMBOL_POLICY_AUTO:
                entry["symbol_policy"] = policy
            if entry:
                styles[str(name)] = entry
        return styles


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
        factors_raw = data.get("row_height_factors") or {}
        if not isinstance(factors_raw, dict):
            # 评审 B1：手写档 factors 写成标量时 .items() 抛 AttributeError
            factors_raw = {}
        factors = {
            int(row): int(value)
            for row, value in factors_raw.items()
        }
        return cls(
            title=str(data.get("title", "") or ""),
            layout_rows=int(data.get("layout_rows", 1) or 1),
            layout_cols=int(data.get("layout_cols", 1) or 1),
            row_height_factors=factors,
            # 评审 B1：非 dict plot（如 plots: ["speed"]）对齐预览端
            # _iter_config_plots 的跳过口径，不得抛 AttributeError
            plots=[
                PlotConfig.from_dict(p)
                for p in (data.get("plots") or [])
                if isinstance(p, dict)
            ],
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
        plots = [
            PlotConfig.from_dict(p)
            for p in (data.get("plots") or [])
            if isinstance(p, dict)
        ]
        mark_range = data.get("saved_mark_range")
        return cls(
            created_at=data.get("created_at", ""),
            # 复审 BL-3：v1 顶层 layout 与页级同口径 int 收口——非法即抛
            # ValueError 落入入口 except，不得留到 try 外的 normalized_pages
            layout_rows=int(data.get("layout_rows", 1) or 1),
            layout_cols=int(data.get("layout_cols", 1) or 1),
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
