"""数据结构和类型定义

包含项目共用的数据类、异常类型和枚举：
- FormatInfo: 文件格式自动检测结果
- CurveInfo: 单条曲线元数据
- InvalidCurveRef: 失效曲线占位（布局资产保留）
- MarkStatEntry: 标记区域统计条目
- AutoDetectError: 格式检测失败异常
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.config import XRANGE_THRESHOLD_FOR_SYMBOLS

if TYPE_CHECKING:
    import numpy as np
    import pyqtgraph as pg


class AutoDetectError(Exception):
    """自动检测文件格式失败，需要用户手动指定分隔符/标题行/单位行"""

    pass


@dataclass
class FormatInfo:
    """文件格式自动检测结果

    包含一次文件扫描得到的全部格式信息：编码、分隔符、标题行位置、是否包含单位行。
    供 auto_detect() 统一入口返回，也供未来的 ImportDialog 消费。
    """

    encoding: str | None  # 检测到的编码名称，None 表示检测失败
    sep: str | None  # 检测到的分隔符，None 表示检测失败
    header_row: int  # 标题行 0-based 行号
    has_unit: bool  # 是否包含单位行
    sheet_name: str | None = None  # 选中的 Sheet 名（仅 Excel 使用）


# ---- 曲线绘制方式（draw_mode）与 symbol 策略（symbol_policy）取值 ----
# draw_mode：'auto' = 直线连接（默认，与既有行为一致）；'step-post' = 阶梯
# 且跳变发生在采样时刻（采样时刻记录当前值，matplotlib steps-post 语义）；
# 'step-pre' = 阶梯且跳变提前一个采样间隔（采样时刻记录上一区间已生效的
# 值，matplotlib steps-pre 语义）。
# ⚠ 映射陷阱：pyqtgraph 的 stepMode='left'/'right' 指数据点在平台的哪一侧
# （'left' = 平台画在点左侧，点是右边缘 = steps-pre；'right' = 平台画在
# 点右侧，点是左边缘 = steps-post），与其 PlotDataItem docstring 的表述
# 恰好相反——以 PlotCurveItem._generateStepModeData 的实际几何为准
# （见 test_step_mode_semantics_geometry 回归测试）。
DRAW_MODE_AUTO = "auto"
DRAW_MODE_STEP_POST = "step-post"
DRAW_MODE_STEP_PRE = "step-pre"
DRAW_MODES = (DRAW_MODE_AUTO, DRAW_MODE_STEP_POST, DRAW_MODE_STEP_PRE)

# symbol_policy：'auto' = 按可见点数阈值自动切换（既有行为）；'on'/'off'
# = 用户显式钉住（始终显示 / 始终隐藏），与 draw_mode 正交。
SYMBOL_POLICY_AUTO = "auto"
SYMBOL_POLICY_ON = "on"
SYMBOL_POLICY_OFF = "off"
SYMBOL_POLICIES = (SYMBOL_POLICY_AUTO, SYMBOL_POLICY_ON, SYMBOL_POLICY_OFF)


def step_mode_for(draw_mode: str) -> str | None:
    """draw_mode → pyqtgraph PlotDataItem 的 stepMode 参数。

    'auto' → None（直线，pyqtgraph 默认）；'step-post' → 'right'（点是
    平台左边缘，跳变发生在采样点）；'step-pre' → 'left'（点是平台右边缘，
    跳变提前一个采样间隔）。未知值按 'auto' 处理（防御）。

    注意：pyqtgraph docstring 对 'left'/'right' 的描述与实现相反，
    此映射经 _generateStepModeData 顶点序列实测校准（见
    test_step_mode_semantics_geometry）。
    """
    if draw_mode == DRAW_MODE_STEP_POST:
        return "right"
    if draw_mode == DRAW_MODE_STEP_PRE:
        return "left"
    return None


def resolve_symbol_show(ci: "CurveInfo", x_width: float) -> bool:
    """单曲线 symbol 判定：symbol_policy 钉住优先，auto 按可见点数阈值。

    判据：估算可见点数 = x宽度 × 该曲线采样密度
    < XRANGE_THRESHOLD_FOR_SYMBOLS 时显示 symbol。逐曲线口径
    （update_plot_style 热路径与 apply_curve_symbol_policy 显式
    用户动作共用），x_width 取当前可见区间宽度。
    """
    policy = getattr(ci, "symbol_policy", SYMBOL_POLICY_AUTO)
    if policy == SYMBOL_POLICY_ON:
        return True
    if policy == SYMBOL_POLICY_OFF:
        return False
    return x_width * ci.point_density < XRANGE_THRESHOLD_FOR_SYMBOLS


@dataclass
class CurveInfo:
    """单条曲线的元数据。

    包含曲线的变量名、PlotDataItem 引用、数据数组、颜色、可见性等属性。
    缓存 x_min/x_max 和 point_density 以避免重复计算。
    """

    var_name: str
    curve: "pg.PlotDataItem"
    x_data: np.ndarray
    y_data: np.ndarray
    original_index: np.ndarray | None = None  # 原始索引/时间戳（用于时间修正重算，消除反算精度问题）
    color: str = "blue"
    y_format: str = ""
    visible: bool = True
    x_min: float = 0.0
    x_max: float = 0.0
    point_density: float = 0.0
    # 绘制方式与 symbol 策略（与线色同口径：会话内存活，不持久化）
    draw_mode: str = DRAW_MODE_AUTO       # 'auto' | 'step-post' | 'step-pre'
    symbol_policy: str = SYMBOL_POLICY_AUTO  # 'auto' | 'on' | 'off'

    def __post_init__(self):
        """自动从 x_data 计算缓存的 x_min / x_max 和 point_density"""
        self._refresh_x_cache()

    def _refresh_x_cache(self):
        """刷新 x_min/x_max/point_density 缓存"""
        import numpy as np
        if self.x_data is not None and len(self.x_data) > 1:
            self.x_min = float(np.min(self.x_data))
            self.x_max = float(np.max(self.x_data))
            span = self.x_max - self.x_min
            self.point_density = len(self.x_data) / span if span > 0 else 0.0
        elif self.x_data is not None and len(self.x_data) == 1:
            self.x_min = float(self.x_data[0])
            self.x_max = float(self.x_data[0])
            self.point_density = 0.0

    def update_x_range(self):
        """当 x_data 变更后调用，同步更新缓存"""
        self._refresh_x_cache()


@dataclass
class InvalidCurveRef:
    """失效曲线占位（方案 B2：布局是用户资产，数据只是输入）。

    加载新数据后，某变量不存在或 ``df_validity == INVALID`` 时，曲线不被删除，
    而是降级为一条轻量占位：保留原颜色与可见性偏好，等该变量再次出现时
    自动复活。占位与 ``CurveInfo`` 物理隔离（存放在 ``pw.invalid_curves``
    字典），不建 PlotDataItem、不写场景，因此所有遍历 ``pw.curves`` 的下游
    消费方（Y 轴 auto_range、密度、cursor、标记统计、导出）天然不受影响。

    不变量：同一变量名绝不同时出现在 ``pw.curves`` 与 ``pw.invalid_curves``。
    """

    var_name: str
    color: str = "gray"
    visible_preference: bool = True  # 复活时恢复用户上次显隐选择


@dataclass
class MarkStatEntry:
    """标记区域统计条目

    ``label`` 保留拼好的 ``"var (unit)"``（导出文本与既有测试兼容）；
    ``var_name`` / ``unit`` 是未拼接的原始字段，供统计表格独立成列
    （变量名本身可含括号，从 label 反解不可靠）。
    """
    x1: float
    x2: float
    y1: float
    y2: float
    dx: float
    dy: float
    slope: float
    label: str
    y_avg: float
    y_max: float
    y_min: float
    # ---- 新增（全部带默认值，向后兼容旧构造点）----
    var_name: str = ""                 # 变量名（未拼单位的原始名，表格"曲线"列）
    unit: str = ""                     # 单位（独立成列）
    x_at_y_max: float = float("nan")   # y 最大时的 x 值
    x_at_y_min: float = float("nan")   # y 最小时的 x 值
    y_pp: float = float("nan")         # 峰峰值 y_max - y_min
    count: int = 0                     # 区间内有效样本数
