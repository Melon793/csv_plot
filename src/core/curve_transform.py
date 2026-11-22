"""逐曲线线性变换投影（编辑器强化轮 D3）。

y' = factor·y + offset 统一投影进 ``ci.y_data``；``ci.y_raw`` 持原始值，
重投影只从 y_raw 出发（幂等，防叠乘）。非数值格式原样返回：枚举码值与
Unix 时间戳（'s' 纯时间 / 'date' 日期）不得参与数值变换，与游标格式化
同口径。计算用 float64，结果 cast 回原始 dtype（保 float32 性能路径）。
"""

from __future__ import annotations

import numpy as np

#: 禁用线性变换的 y_format 集合（枚举码值 / 时间戳）
NON_NUMERIC_FORMATS = frozenset({"enum", "date", "s"})


def project_y(
    raw: np.ndarray | None,
    factor: float,
    offset: float,
    y_format: str = "",
) -> np.ndarray | None:
    """对原始 y 做线性投影；默认变换或非数值格式原样返回。"""
    if raw is None:
        return None
    if y_format in NON_NUMERIC_FORMATS or (factor == 1.0 and offset == 0.0):
        return raw
    src = np.asarray(raw)
    out = src.astype(np.float64) * float(factor) + float(offset)
    return out.astype(src.dtype, copy=False)


def reproject(ci) -> None:
    """从 ci.y_raw 重投影写回 ci.y_data（改 factor/offset 后唯一入口）。"""
    ci.y_data = project_y(ci.y_raw, ci.factor, ci.offset, ci.y_format)