"""曲线标注的数据模型（零 Qt 依赖）

设计要点：
- 只做「数据 + 校验 + 序列化」，不引入任何 Qt 依赖，可直接在 unit 层测试。
- 几何统一用「数据坐标点列表」表达，由 kind 决定解释方式：
      text              -> 1 个锚点
      target            -> 1 个锚点（标靶：屏幕尺寸恒定的十字标记）
      rect / ellipse    -> 2 个对角点
      line / arrow      -> 2 个点（尾 -> 头）
      polyline          -> 2 个及以上点（折点可变长，上限 MAX_POLYLINE_POINTS）
- 非法数据一律「丢弃」而不是抛异常：标注是用户手工产物，一条坏记录不该让
  整份模板加载失败。from_dict 对坏输入返回 None，由调用方决定是否记日志。
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass

# 图元类型白名单：反序列化按它校验，未知 kind 直接丢弃
ANNOTATION_KINDS = ("text", "rect", "ellipse", "line", "arrow", "polyline", "target")

# 锚定方式（预留字段，MVP 只实现 data）：
#   data     —— 锚点是数据坐标，缩放平移时跟着数据走。语义正确，MVP 只用这个。
#   viewport —— 锚点固定在图窗上（水印 / 整图结论）。P3 实现。
# 当前版本不认识 viewport 值，读到就回退 data，避免新老版本互读时行为漂移。
ANNOTATION_ANCHORS = ("data", "viewport")
DEFAULT_ANCHOR = "data"

# 折线折点数上限：与 MAX_ANNOTATIONS_PER_PLOT 同一类保护 —— 折点可以由用户在
# 「段上点击」无限追加，模板文件里的 points 更是外部输入，不设上限等于让一份
# 坏模板把场景图元数撑爆。超出的点按顺序截断（保留前 N 个）。
MAX_POLYLINE_POINTS = 64

# 每个 kind 的点数下限；polyline 折点可变长，上限见 MAX_POLYLINE_POINTS
_MIN_POINTS = {
    "text": 1, "target": 1, "rect": 2, "ellipse": 2, "line": 2, "arrow": 2, "polyline": 2,
}
_MAX_POINTS: dict[str, int | None] = {
    "text": 1, "target": 1, "rect": 2, "ellipse": 2, "line": 2, "arrow": 2,
    "polyline": MAX_POLYLINE_POINTS,
}

# 图层 Z 序：夹在曲线（0，PlotDataItem 默认）与游标竖线（100）之间。
# 这样标注永远盖住曲线、但永远不遮挡游标 —— "显示游标"的既有视觉行为不变。
Z_ANNOTATION = 50
Z_ANNOTATION_SELECTED = 90

# 单个子图的标注数量上限：防误操作（例如脚本批量创建）把界面拖垮
MAX_ANNOTATIONS_PER_PLOT = 200
MAX_TEXT_LENGTH = 500

# 撤销栈深度：快照式撤销每步存一份完整 dump，单条标注约 200 字节，
# 200 条 × 50 步约 2MB —— 上限取 50 是"够用"与"不无界增长"的折中。
MAX_UNDO_STEPS = 50

DEFAULT_STROKE = "#E24B4A"
DEFAULT_STROKE_WIDTH = 2
DEFAULT_FILL_ALPHA = 40
DEFAULT_FONT_SIZE = 12
DEFAULT_VISIBLE = True

# 线宽下限 0：0 仅对文字类型有意义 = 无文字外边框（TextItem.paint 按
# border 样式为 NoPen 跳过绘制；Qt 里 width=0 的笔是 cosmetic 1px，不等价）。
# 其余类型的图元层仍钳制到 ≥1（直线/箭头宽 0 会隐形，标靶会只剩标签）。
STROKE_WIDTH_RANGE = (0, 8)
FONT_SIZE_RANGE = (8, 48)
FILL_ALPHA_RANGE = (0, 255)

# 只接受十六进制色值：#RGB / #RRGGBB / #AARRGGBB。
# 不接受颜色名：Qt 认识的颜色名集合随版本变化，且拼错时静默变黑更难排查，
# 统一在这里挡掉、回退到默认色，行为可预测也可测试。
_HEX_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


def new_annotation_id() -> str:
    """生成 8 位标注 id（够用且短，便于日志与模板肉眼核对）"""
    return uuid.uuid4().hex[:8]


def minimum_points(kind: str) -> int:
    """kind 需要的最少点数，未知 kind 按 1 处理

    给 UI 侧回写用：图元处在中间态（例如折线正在重建手柄）时会读到过少的点，
    那种读数不该写进模型 —— 一条点数不足的标注会让整份模板加载失败。
    """
    return _MIN_POINTS.get(kind, 1)


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


def _is_finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def safe_color(value: object, fallback: str = DEFAULT_STROKE) -> str:
    """校验色值，非法一律回退

    必须同时挡掉 NaN/inf：它们会被 Qt 当成未定义颜色，渲染结果不可预期。
    """
    if isinstance(value, str) and _HEX_COLOR_RE.match(value.strip()):
        return value.strip()
    return fallback


def safe_int(value: object, fallback: int, low: int, high: int) -> int:
    """把任意输入收拢成 [low, high] 内的整数，非法回退"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    if not _is_finite(value):
        return fallback
    return int(_clamp(int(value), low, high))


def safe_bool(value: object, fallback: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    return fallback


def _clean_points(raw: object) -> list[tuple[float, float]] | None:
    """把原始点列表收拢成 [(x, y), ...]，任一坐标非有限数直接判整体非法

    不做"只丢坏点"的降级：点的下标带语义（矩形 pos 与 size、直线的两端），
    悄悄丢一个点会把矩形变成另一个矩形，比整条丢弃更难排查。
    """
    if not isinstance(raw, (list, tuple)):
        return None
    points: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        x, y = item[0], item[1]
        if not (_is_finite(x) and _is_finite(y)):
            return None
        points.append((float(x), float(y)))
    return points


@dataclass
class AnnotationItem:
    """单条标注

    ``points`` 一律是数据坐标，单位与所在子图的 X/Y 轴一致。
    """

    id: str
    kind: str
    points: list[tuple[float, float]]

    text: str = ""                       # 仅 kind == "text" 使用
    stroke: str = DEFAULT_STROKE
    stroke_width: int = DEFAULT_STROKE_WIDTH
    fill: str | None = None              # None 表示不填充
    fill_alpha: int = DEFAULT_FILL_ALPHA
    font_size: int = DEFAULT_FONT_SIZE
    anchor: str = DEFAULT_ANCHOR         # 预留：data / viewport
    visible: bool = DEFAULT_VISIBLE

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """序列化为 JSON 友好的普通字典（不出现 Qt / numpy 类型）

        points 写成 [[x, y], ...] 而不是元组列表：YAML/JSON 往返后元组会退化成
        列表，统一成列表可让 to_dict(from_dict(x)) == x 严格成立。
        """
        return {
            "id": self.id,
            "kind": self.kind,
            "points": [[float(x), float(y)] for x, y in self.points],
            "text": self.text,
            "stroke": self.stroke,
            "stroke_width": int(self.stroke_width),
            "fill": self.fill,
            "fill_alpha": int(self.fill_alpha),
            "font_size": int(self.font_size),
            "anchor": self.anchor,
            "visible": bool(self.visible),
        }

    @classmethod
    def from_dict(cls, raw: object) -> "AnnotationItem | None":
        """从字典反序列化，非法输入返回 None（绝不抛异常）"""
        if not isinstance(raw, dict):
            return None

        kind = raw.get("kind")
        if kind not in ANNOTATION_KINDS:
            return None

        points = _clean_points(raw.get("points"))
        if points is None:
            return None
        low = _MIN_POINTS[kind]
        high = _MAX_POINTS[kind]
        if len(points) < low:
            return None
        if high is not None and len(points) > high:
            points = points[:high]

        raw_id = raw.get("id")
        annotation_id = raw_id.strip() if isinstance(raw_id, str) and raw_id.strip() else new_annotation_id()

        raw_text = raw.get("text")
        text = raw_text[:MAX_TEXT_LENGTH] if isinstance(raw_text, str) else ""

        raw_fill = raw.get("fill")
        fill = safe_color(raw_fill, DEFAULT_STROKE) if isinstance(raw_fill, str) and raw_fill.strip() else None

        raw_anchor = raw.get("anchor")
        anchor = raw_anchor if raw_anchor in ANNOTATION_ANCHORS else DEFAULT_ANCHOR

        return cls(
            id=annotation_id,
            kind=kind,
            points=points,
            text=text,
            stroke=safe_color(raw.get("stroke"), DEFAULT_STROKE),
            stroke_width=safe_int(raw.get("stroke_width"), DEFAULT_STROKE_WIDTH, *STROKE_WIDTH_RANGE),
            fill=fill,
            fill_alpha=safe_int(raw.get("fill_alpha"), DEFAULT_FILL_ALPHA, *FILL_ALPHA_RANGE),
            font_size=safe_int(raw.get("font_size"), DEFAULT_FONT_SIZE, *FONT_SIZE_RANGE),
            anchor=anchor,
            visible=safe_bool(raw.get("visible"), DEFAULT_VISIBLE),
        )

    # ------------------------------------------------------------------
    # 校验与几何辅助
    # ------------------------------------------------------------------
    def is_valid(self) -> bool:
        """校验当前字段（用于已构造对象自检，拒绝脏数据进模板）"""
        if self.kind not in ANNOTATION_KINDS:
            return False
        if not isinstance(self.id, str) or not self.id.strip():
            return False
        if _clean_points(self.points) is None:
            return False
        low = _MIN_POINTS[self.kind]
        high = _MAX_POINTS[self.kind]
        if len(self.points) < low:
            return False
        if high is not None and len(self.points) > high:
            return False
        return True

    def bounds(self) -> tuple[float, float, float, float] | None:
        """(xmin, ymin, xmax, ymax)；单点图元（text / target）返回 None

        单点图元没有"范围"可言 —— 硬算会得到 (x, y, x, y) 这种零面积矩形，
        调用方还得自己判断是不是退化，不如在这里就说清楚。
        """
        if self.kind in ("text", "target") or not self.points:
            return None
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    def move_by(self, dx: float, dy: float) -> None:
        """整体平移（数据坐标增量）"""
        self.points = [(x + dx, y + dy) for x, y in self.points]

    def summary(self) -> str:
        """一行摘要，供标注列表对话框显示"""
        names = {
            "text": "文字", "target": "标靶", "rect": "矩形", "ellipse": "椭圆",
            "line": "直线", "arrow": "箭头", "polyline": "折线",
        }
        label = names.get(self.kind, self.kind)
        if self.kind == "text":
            body = self.text.strip().replace("\n", " ") or "(空)"
            if len(body) > 24:
                body = body[:24] + "…"
            return f'{label}　"{body}"'
        box = self.bounds()
        if box is None:
            # 单点图元：给坐标而不是退化矩形；标靶若带标签文案一并显示
            if not self.points:
                return label
            x, y = self.points[0]
            text = f"　{label}　({x:.3g}, {y:.3g})"
            body = self.text.strip().replace("\n", " ")
            if self.kind == "target" and body:
                if len(body) > 16:
                    body = body[:16] + "…"
                text += f'　"{body}"'
            return text.strip()
        x0, y0, x1, y1 = box
        return f"{label}　({x0:.3g}, {y0:.3g}) → ({x1:.3g}, {y1:.3g})"
