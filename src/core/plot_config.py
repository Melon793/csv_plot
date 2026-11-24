"""Plot 会话配置数据结构"""

from __future__ import annotations
import math
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
    #: 编辑器强化轮 D5/D6：逐曲线显示名/换算 {变量名: {display_name,factor,offset}}。
    #: 可选字段——仅非空才写键、条目仅非默认键；旧档缺省读作空 dict，
    #: 旧版软件读新模板忽略未知键，往返零影响（沿用 invalid_curves 惯例）。
    #: 失效占位的设置一并存（按名为键），复活时带回。
    curve_settings: dict = field(default_factory=dict)
    #: 模板身份匹配（方案 A，D2）：{曲线显示名: 身份条目}。身份条目记录
    #: 跨文件稳定的通道指纹（original_name + kind + MDF device/ecu 或
    #: BLF frame/can_id/extended/bus_channel），供套用端在后缀漂移时重映射。
    #: 可选字段——仅非空才写键；旧档缺省读作空 dict，旧版软件读新模板
    #: 忽略未知键，往返零影响（沿用 curve_settings 惯例）。
    #: 披露：占位名导出时不在 loader，永远拿不到身份条目；占位跨文件
    #: 回退只靠引擎的 L2 名字启发式（D10）。
    curve_identities: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化为字典（invalid_curves/cell_index/cell_row/cell_col/curve_settings/curve_identities 仅有值才写，保持旧档兼容）"""
        data = {"curves": self.curves}
        if self.invalid_curves:
            data["invalid_curves"] = list(self.invalid_curves)
        if self.cell_index is not None:
            data["cell_index"] = int(self.cell_index)
        if self.cell_row is not None and self.cell_col is not None:
            data["cell_row"] = int(self.cell_row)
            data["cell_col"] = int(self.cell_col)
        if self.curve_settings:
            data["curve_settings"] = {
                str(k): dict(v) for k, v in self.curve_settings.items()
            }
        if self.curve_identities:
            data["curve_identities"] = {
                str(k): dict(v) for k, v in self.curve_identities.items()
            }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PlotConfig":
        """从字典反序列化（缺省 invalid_curves 容错为空列表；cell_index/cell_row/cell_col 容错 None）"""
        raw_cell = data.get("cell_index")
        try:
            # 第 6 轮 P2-44：bool 拒收（int(True)→1 会静默落错格，
            # 与预览端排除 bool 的口径对齐）
            cell_index = (
                None if raw_cell is None or isinstance(raw_cell, bool)
                else int(raw_cell)
            )
        except (TypeError, ValueError):
            cell_index = None
        try:
            raw_row, raw_col = data.get("cell_row"), data.get("cell_col")
            cell_row = (
                None if raw_row is None or isinstance(raw_row, bool)
                else int(raw_row)
            )
            cell_col = (
                None if raw_col is None or isinstance(raw_col, bool)
                else int(raw_col)
            )
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
            curve_settings=parse_curve_settings(data.get("curve_settings")),
            curve_identities=parse_curve_identities(data.get("curve_identities")),
        )


def parse_curve_settings(raw) -> dict:
    """curve_settings 容错解析（D5，照 cell_index 第 6 轮 P2-44 守卫口径）。

    非 dict → 空；键 str 化；条目非 dict 丢弃；display_name 仅非空 str；
    factor/offset bool 拒收、float try/except、非有限数丢弃、仅留非默认值；
    未知键忽略。全默认条目不保留（与 to_dict 仅非默认写键同口径）。
    """
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for key, val in raw.items():
        if key is None or not isinstance(val, dict):
            continue
        entry: dict = {}
        dn = val.get("display_name")
        if isinstance(dn, str) and dn.strip():
            entry["display_name"] = dn
        for fld, dflt in (("factor", 1.0), ("offset", 0.0)):
            v = val.get(fld)
            if v is None or isinstance(v, bool):
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv) and fv != dflt:
                entry[fld] = fv
        if entry:
            out[str(key)] = entry
    return out


def parse_curve_identities(raw) -> dict:
    """curve_identities 容错解析（D2，照 parse_curve_settings 守卫口径）。

    非 dict → 空；键 str 化（None 键丢弃）；条目非 dict 丢弃；
    original_name 必须是非空 str，kind 必须是 'mdf'/'blf'，否则整条丢弃
    （无 original_name 的身份既不能做 D9 校验也不能做 L1 比对）。
    mdf：device/ecu 仅收非空 str（可缺省——真实 MDF3 常无 SI，L1 退 L2）；
    blf：frame/can_id/extended/bus_channel 四元组必须齐全且类型合法
    （int 拒收 bool，照 cell_index P2-44 口径），数值不得截断或越界，
    缺任一即整条丢弃——半截指纹按 D12 无法全等比对，留着只会误导。
    未知键忽略。
    """
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for key, val in raw.items():
        if key is None or not isinstance(val, dict):
            continue
        on = val.get("original_name")
        kind = val.get("kind")
        if not isinstance(on, str) or not on.strip():
            continue
        if kind not in ("mdf", "blf"):
            continue
        entry: dict = {"original_name": on, "kind": kind}
        # D9 一致性校验需要模板侧单位；"-" 是 loader 的无单位占位，视为空
        unit = val.get("unit")
        if isinstance(unit, str) and unit.strip() and unit.strip() != "-":
            entry["unit"] = unit
        if kind == "mdf":
            for fld in ("device", "ecu"):
                v = val.get(fld)
                if isinstance(v, str) and v.strip():
                    entry[fld] = v
        else:
            frame = val.get("frame")
            if not isinstance(frame, str) or not frame.strip():
                continue
            ext = val.get("extended")
            if not isinstance(ext, bool):
                continue
            nums: dict = {}
            valid = True
            for fld in ("can_id", "bus_channel"):
                v = val.get(fld)
                if v is None or isinstance(v, bool):
                    valid = False
                    break
                if isinstance(v, str):
                    try:
                        iv = int(v)
                    except ValueError:
                        valid = False
                        break
                elif isinstance(v, int):
                    iv = v
                elif isinstance(v, float) and math.isfinite(v) and v.is_integer():
                    iv = int(v)
                else:
                    valid = False
                    break
                if iv < 0:
                    valid = False
                    break
                if fld == "can_id" and iv > (0x1FFFFFFF if ext else 0x7FF):
                    valid = False
                    break
                nums[fld] = iv
            if not valid:
                continue
            entry.update(frame=frame, extended=ext, **nums)
        out[str(key)] = entry
    return out


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
        if mark_range is not None:
            if not isinstance(mark_range, list) or len(mark_range) != 2:
                raise ValueError("saved_mark_range must contain exactly two numbers")
            try:
                mark_range = [float(value) for value in mark_range]
            except (TypeError, ValueError) as exc:
                raise ValueError("saved_mark_range must contain numbers") from exc
            if not all(math.isfinite(value) for value in mark_range) or not (
                mark_range[0] < mark_range[1]
            ):
                raise ValueError("saved_mark_range must be finite and increasing")
        time_factor = data.get("time_factor", 1.0)
        time_offset = data.get("time_offset", 0.0)
        try:
            time_factor = float(time_factor)
            time_offset = float(time_offset)
        except (TypeError, ValueError) as exc:
            raise ValueError("time_factor and time_offset must be numbers") from exc
        if not math.isfinite(time_factor) or not math.isfinite(time_offset):
            raise ValueError("time_factor and time_offset must be finite")
        return cls(
            created_at=data.get("created_at", ""),
            # 复审 BL-3：v1 顶层 layout 与页级同口径 int 收口——非法即抛
            # ValueError 落入入口 except，不得留到 try 外的 normalized_pages
            layout_rows=int(data.get("layout_rows", 1) or 1),
            layout_cols=int(data.get("layout_cols", 1) or 1),
            time_factor=time_factor,
            time_offset=time_offset,
            plots=plots,
            pages=pages,
            active_page=int(data.get("active_page", 0) or 0),
            cursor_mode=data.get("cursor_mode"),
            saved_mark_range=mark_range,
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
