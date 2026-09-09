"""合成测试数据工厂：确定性小数据生成，避免依赖 data/ 下的大文件。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def make_timeseries(
    n: int = 10_000, seed: int = 42, inject_extremes: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """生成固定种子的正弦时序数据。

    与 tmp/verify_freeze_restore_e2e.py 的数据构造方式保持一致：
    主体范围 40~60（sin*10+50），可选在可见窗口外注入极值点。
    """
    x = np.linspace(0, 100, n)
    y = np.sin(x / 5.0) * 10 + 50
    if inject_extremes and n > 9500:
        y[100] = 500.0
        y[9500] = -300.0
    return x, y


def write_csv(
    path: Path | str,
    *,
    header: list[str],
    units: list[str] | None = None,
    rows: list[list],
    sep: str = ",",
    encoding: str = "utf-8",
    desc_rows: list[str] | None = None,
) -> Path:
    """写入合成 CSV 文件。

    Args:
        path: 目标路径
        header: 变量名行
        units: 单位行（None 表示不写单位行）
        rows: 数据行
        sep: 分隔符
        encoding: 文件编码
        desc_rows: 可选的元数据描述行（置于文件开头）

    Returns:
        写入的 Path
    """
    path = Path(path)
    lines: list[str] = []
    if desc_rows:
        lines.extend(desc_rows)
    lines.append(sep.join(str(c) for c in header))
    if units is not None:
        lines.append(sep.join(str(u) for u in units))
    for row in rows:
        lines.append(sep.join(str(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding=encoding)
    return path


def make_simple_rows(n: int = 20) -> list[list]:
    """生成 n 行简单数值数据：[time, speed, rpm, flag]。

    speed 递增（非常量）、rpm 递增（非常量）、flag 恒为 1（常量列）。
    """
    rows = []
    for i in range(n):
        rows.append([f"{i * 0.1:.1f}", f"{10.0 + i * 0.5:.2f}", 800 + i * 10, 1])
    return rows
