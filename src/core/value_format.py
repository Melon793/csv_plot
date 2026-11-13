"""数值显示精度工具（游标 X 标签与标记统计窗口共用）

背景：X 标签的小数位过去由 ``factor`` 的字面小数位决定，而 ``factor`` 在
MDF/BLF 这类非均匀时间轴上只是 1.0 的无效倍数，导致大面积显示整数。这里
提供两个纯函数，把「按步长推导精度」与「定点格式化并去尾零」收敛成唯一
口径，供 ``CurveInfo``、``CursorManager``、``mark_stats`` 共用。
"""

from __future__ import annotations

import math

# 小数位上限：防止异常小步长（如重复时间戳被压缩到 1e-12）产生超长文本
MAX_X_DECIMALS = 9


def decimals_for_step(step: float, cap: int = MAX_X_DECIMALS) -> int:
    """相邻样本步长 -> 能区分相邻样本的小数位数。

    1 -> 0、0.1 -> 1、0.001 -> 3、1e-7 -> 7；非有限/非正步长回退 0。
    ``cap`` 用于封顶，避免异常数据产生超长标签。
    """
    if not math.isfinite(step) or step <= 0:
        return 0
    exponent = -math.log10(step)
    # 浮点误差保护：1370 s 附近的 1 ms 间隔算出的 log10 会略大于 3
    # （如 3.00000000001），直接 ceil 会凭空多一位小数
    if abs(exponent - round(exponent)) < 1e-9:
        exponent = round(exponent)
    return max(0, min(cap, math.ceil(exponent)))


def format_fixed_trim(value: float, decimals: int) -> str:
    """定点格式化并去掉无意义尾零；``decimals<=0`` 时取整。"""
    if value == 0:  # 同时覆盖 -0.0，避免显示 "-0"
        return "0"
    if decimals <= 0:
        return str(int(round(value)))
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".") or "0"
