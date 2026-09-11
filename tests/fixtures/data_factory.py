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


# ---------------------------------------------------------------------------
# MDF 合成
# ---------------------------------------------------------------------------

#: 合成的枚举文本表：{码值: 标签}
ENUM_TEXTS: dict[int, str] = {0: "off", 1: "on", 2: "err"}


def write_mdf(
    path: Path | str,
    *,
    version: str = "4.10",
    n: int = 10,
    with_enum: bool = True,
    with_string: bool = True,
    with_dup_group: bool = True,
    with_single_shot_group: bool = False,
    with_empty_group: bool = False,
) -> Path:
    """写入合成 MDF 文件，返回 **asammdf 实际写出的路径**。

    必须使用返回值：``MDF.save()`` 会无条件把扩展名改写为 ``.mdf``
    （实测传入 ``syn_330.dat`` 得到 ``syn_330.mdf``），调用方若继续用
    原路径会得到 FileNotFoundError。

    Args:
        path: 目标路径（扩展名会被 asammdf 改写）
        version: ``'3.30'`` 生成 MDF3，``'4.10'`` 生成 MDF4。
            这是本工厂的核心价值 —— 同一个枚举通道在两版下落地为
            **不同的 conversion_type**（实测 MDF3 为 ct=12 RTABX、
            MDF4 为 ct=7 TABX），因此可确定性地验证版本感知判定。
        n: 主组采样点数
        with_enum: 是否写入枚举通道 ``State``（uint8 + 文本表）
        with_string: 是否写入字符串通道 ``Label``。两版的存储方式不同
            （MDF3 的 dtype_fmt 即 |S1；MDF4 的 dtype_fmt 是 uint64 索引，
            data_type=7），用于验证 is_numeric 的跨版本判定
        with_dup_group: 是否再写一个含同名通道 ``Press`` 的组，
            触发跨组重名的聚合改名（``Press_G1``）
        with_single_shot_group: 是否写入 comment 含 ``SingleShotGroup`` 的组。
            MDFLazyLoader 会跳过这类组，用于验证过滤行为
        with_empty_group: 是否写入零长度组（cycles_nr=0 且无数据块），
            实测真实 MDF3 文件中约 12.5% 的通道属于此类预留组

    Returns:
        asammdf 实际写出的 Path
    """
    from asammdf import MDF, Signal

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mdf = MDF(version=version)
    t = np.arange(n, dtype=np.float64) * 0.1

    signals = [
        Signal(
            np.linspace(1.0, 5.0, n).astype(np.float32), t, unit="bar", name="Press"
        )
    ]
    if with_enum:
        # asammdf 见到 val_0 + text_0 即自动推断为文本表转换，
        # 无需手工构造 ChannelConversion 块（跨 v3/v4 的块类型不同）
        conv: dict = {}
        for i, (code, text) in enumerate(sorted(ENUM_TEXTS.items())):
            conv[f"val_{i}"] = code
            conv[f"text_{i}"] = text
        signals.append(
            Signal(
                np.array([i % len(ENUM_TEXTS) for i in range(n)], dtype=np.uint8),
                t,
                unit="st",
                name="State",
                conversion=conv,
            )
        )
    if with_string:
        # MDF3 不支持 <U1 dtype（抛 MdfException: Unknown type），必须给 bytes；
        # MDF4 则要求显式 encoding，否则抛 'wrong encoding "None" for string signal'
        signals.append(
            Signal(
                np.array([chr(ord("a") + i % 26).encode() for i in range(n)]),
                t,
                unit="",
                name="Label",
                encoding="utf-8",
            )
        )

    mdf.append(
        signals,
        acq_name="NormalGroup",
        comment="synthetic test group",
        units={"Press": "bar", "State": "st"},
    )

    if with_dup_group:
        n2 = max(1, n // 2)
        t2 = np.arange(n2, dtype=np.float64) * 0.2
        mdf.append(
            [Signal(np.arange(n2, dtype=np.float32), t2, unit="degC", name="Press")],
            acq_name="DupGroup",
            comment="duplicate channel name group",
            units={"Press": "degC"},
        )

    if with_single_shot_group:
        t3 = np.arange(2, dtype=np.float64)
        mdf.append(
            [
                Signal(
                    np.array([1.5, 2.5], dtype=np.float32), t3, unit="V", name="SingleCh"
                )
            ],
            acq_name="SingleShotGroup",
            # loader 靠 comment 子串识别并跳过，注释文案不可改
            comment="SingleShotGroup",
        )

    if with_empty_group:
        mdf.append(
            [
                Signal(
                    np.array([], dtype=np.float32),
                    np.array([], dtype=np.float64),
                    unit="-",
                    name="EmptyCh",
                )
            ],
            acq_name="EmptyGroup",
            comment="empty reserved group",
        )

    out = mdf.save(str(path), overwrite=True)
    mdf.close()
    return Path(out)
