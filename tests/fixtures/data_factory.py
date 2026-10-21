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


def write_xlsx(
    path: Path | str,
    *,
    header: list[str],
    units: list[str] | None = None,
    rows: list[list],
    sheet_name: str = "Sheet1",
    extra_sheet_names: list[str] | None = None,
) -> Path:
    """写入合成 xlsx：标题行 + 可选单位行 + 数据行。

    Args:
        path: 目标路径
        header: 变量名行（工作表第一行）
        units: 单位行（None 表示不写单位行）
        rows: 数据行。允许塞真实的 ``date`` / ``time`` / ``datetime`` 对象：
            Excel 原生时间单元格是 loader 的一条独立分支，用字符串写不出那种
            单元格类型
        sheet_name: 数据所在工作表的表名
        extra_sheet_names: 额外工作表的表名。多表用例关心的是"选中了哪张表"，
            这些表统一写一份最小两列表（time/speed 各一行），内容不参与断言

    Returns:
        写入的 Path
    """
    # 懒导入：本工厂只被少数 Excel 用例使用，不必把 openpyxl 的导入成本
    # 摊到每个 import 本模块的测试上（与 write_mdf 的处理一致）
    import openpyxl

    path = Path(path)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append(list(header))
    if units is not None:
        ws.append(list(units))
    for row in rows:
        ws.append(list(row))
    for name in extra_sheet_names or ():
        extra = wb.create_sheet(name)
        extra.append(["time", "speed"])
        extra.append([0.0, 1.0])
    wb.save(str(path))
    wb.close()
    return path


# ---------------------------------------------------------------------------
# MDF 合成
# ---------------------------------------------------------------------------

#: 合成的枚举文本表：{码值: 标签}
ENUM_TEXTS: dict[int, str] = {0: "off", 1: "on", 2: "err"}

# ---------------------------------------------------------------------------
# 归属信息（var_info 的「归属信息」块）用的合成样本
# ---------------------------------------------------------------------------

#: HD 注释正文：实测 5/5 个 CANape/INCA 文件都是这个 ``Key: Value`` 格式。
#: **必须给纯文本**：asammdf 会自己套 ``<HDcomment><TX>``，预先包 XML 会被
#: 递归转义成 ``&amp;lt;HDcomment&amp;gt;``（实测）。
ATTRIBUTION_HEADER_COMMENT = (
    "Database: SYN_DB\n"
    "Experiment: SYN_EXP\n"
    "Workspace: SYN_WS\n"
    "Devices: XCP:1,CAN-Monitoring:1,CalcDev\n"
    "Program Description: SYN_PROG\n"
    "WP: SYN_WP\n"
    "RP: SYN_RP\n"
    "Date: 05/09/2026"
)

#: 通道名内嵌设备、注释为函数样式（v3/v4 通用）的组注释
ATTRIBUTION_RASTER_GROUP = "XcpEvent_100ms_Raster"
#: 设备 / ECU 节点 / 报文组名（取自实测 mf4）
ATTRIBUTION_DEVICE = "CAN-Monitoring:1"
ATTRIBUTION_ECU = "BMCe"
ATTRIBUTION_MESSAGE_GROUP = "BMS_CellVoltInfo"
#: 中文注释 + 来源数据库标注；变量名与组名不相关，故不会误判为函数
ATTRIBUTION_SIGNAL_NAME = "BMS_CellVolt082"
ATTRIBUTION_SIGNAL_COMMENT = "82号单体电压 created from: CANDB: BMS_CellVolt082"
#: 两条走不同回退规则的通道名（'\Device' 取设备、'Fkt/Var\Device' 取函数）
ATTRIBUTION_CHANNEL_NAME = "WaterTemp\\XCP:1"
ATTRIBUTION_CHANNEL_COMMENT = "RBArithmeticElement"
ATTRIBUTION_HIERARCHY_NAME = "EpmCaS_phiSegOfs_CA/isx\\XCP:1"
ATTRIBUTION_HIERARCHY_COMMENT = "#Index"  # 非标识符，必须被丢弃


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
    with_attribution: bool = False,
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
        with_attribution: 是否追加带归属信息的组与 HD 注释（见本文件的
            ``ATTRIBUTION_*`` 常量）。v4 额外写一个带 SI 源块与 display_names
            的报文组（v3 无 SI 块概念，asammdf 只会给它注入一个
            ``'Channel inserted by Python Script'`` 的噪声源）

    Returns:
        asammdf 实际写出的 Path
    """
    from asammdf import MDF, Signal, Source

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

    if with_attribution:
        # 组 A：通道名内嵌设备 + 两条走不同函数回退规则的通道
        mdf.append(
            [
                Signal(
                    np.zeros(n, dtype=np.float32),
                    t,
                    unit="degC",
                    name=ATTRIBUTION_CHANNEL_NAME,
                    comment=ATTRIBUTION_CHANNEL_COMMENT,
                ),
                Signal(
                    np.zeros(n, dtype=np.float32),
                    t,
                    unit="%",
                    name=ATTRIBUTION_HIERARCHY_NAME,
                    comment=ATTRIBUTION_HIERARCHY_COMMENT,
                ),
            ],
            acq_name="RasterGroup",
            comment=ATTRIBUTION_RASTER_GROUP,
        )
        # 组 B：仅 v4 —— SI 源块（设备/ECU/总线类型）+ 层级显示名 + 中文注释
        if version.startswith("4"):
            src = Source(
                name=ATTRIBUTION_ECU,
                path=ATTRIBUTION_DEVICE,
                comment="",
                source_type=1,
                bus_type=2,
            )
            mdf.append(
                [
                    Signal(
                        np.zeros(n, dtype=np.uint16),
                        t,
                        unit="V",
                        name=ATTRIBUTION_SIGNAL_NAME,
                        comment=ATTRIBUTION_SIGNAL_COMMENT,
                        source=src,
                        display_names={
                            f"{ATTRIBUTION_DEVICE}.{ATTRIBUTION_SIGNAL_NAME}": (
                                "source_path"
                            )
                        },
                    )
                ],
                acq_name=ATTRIBUTION_MESSAGE_GROUP,
                acq_source=src,
                comment=ATTRIBUTION_MESSAGE_GROUP,
            )
        mdf.header.comment = ATTRIBUTION_HEADER_COMMENT

    out = mdf.save(str(path), overwrite=True)
    mdf.close()
    return Path(out)


# ---------------------------------------------------------------------------
# CAN 数据库（DBC / ARXML）与 BLF 合成
# ---------------------------------------------------------------------------
#
# 报文定义统一用 dict，字段名与 DBC 文本一一对应，便于手工核对：
#   name/size/cycle_ms/sender/comment             → BO_ 行
#   signals[]: name/start/size/order/signed/factor/offset/minimum/maximum/
#              unit/values/comment/receivers/float_bits/mux
#
# **start 采用 DBC 文本编号约定**（不是 canmatrix 的内部位序）：Intel 为
# LSB 线性位号，Motorola 为 MSB 编号。canmatrix 导入时对 Motorola 会再做一次
# ``set_startbit(..., bitNumbering=1)`` 换算，因此夹具作者写的就是 CANoe/DBC
# 编辑器里看到的数字。
# ``mux`` 取 "M"（多路复用选择器）、int（该路分支编号）或 None。

#: 覆盖解码器全部分支的默认报文定义：Intel / Motorola / 有符号 / float32 /
#: VAL_ 枚举 / 多路复用 / 跨报文重名信号 / 两条总线。
#:
#: 位区间经过实测核对（tmp/probe_can_factory_roundtrip.py），同帧内互不重叠：
#: ``EngineData`` 的 4 个 Intel 信号占满 64 bit；``ChassisData`` 的两个
#: Motorola 信号按 DBC 的 MSB 编号分别从 bit7 / bit11 起算，合占 byte0..3。

#: 每个信号刻意只写必要字段，其余由 :func:`_dbc_signals` 补默认值
CAN_FRAMES: tuple[dict, ...] = (
    {
        "name": "EngineData",
        "can_id": 0x1A0,
        "size": 8,
        "cycle_ms": 20,
        "sender": "EngineECU",
        "comment": "发动机数据",
        "channel": 0,
        "signals": (
            {
                "name": "EngineSpeed",
                "start": 0,
                "size": 16,
                "order": "intel",
                "factor": 0.125,
                "offset": 0.0,
                "minimum": 0.0,
                "maximum": 8031.875,
                "unit": "rpm",
                "comment": "发动机转速",
                "sweep": 3000,
            },
            {
                "name": "CoolantTemp",
                "start": 16,
                "size": 8,
                "order": "intel",
                "signed": True,
                "factor": 1.0,
                "offset": -40.0,
                "minimum": -40.0,
                "maximum": 215.0,
                "unit": "degC",
                "sweep": 30,
            },
            {
                "name": "StarterMode",
                "start": 24,
                "size": 2,
                "order": "intel",
                "factor": 1.0,
                "offset": 0.0,
                "minimum": 0.0,
                "maximum": 3.0,
                "unit": "st",
                "values": {0: "off", 1: "run", 2: "error"},
                "sweep": 3,
            },
            {
                "name": "TorqueLimit",
                "start": 32,
                "size": 32,
                "order": "intel",
                "float_bits": 32,
                "factor": 1.0,
                "offset": 0.0,
                "minimum": 0.0,
                "maximum": 0.0,
                "unit": "Nm",
                "sweep": 12,
            },
        ),
    },
    {
        "name": "DiagMux",
        "can_id": 0x3B0,
        "size": 8,
        "cycle_ms": 100,
        "sender": "Gateway",
        "signals": (
            {
                "name": "MuxSel",
                "start": 0,
                "size": 8,
                "order": "intel",
                "factor": 1.0,
                "offset": 0.0,
                "minimum": 0.0,
                "maximum": 255.0,
                "unit": "",
                "mux": "M",
            },
            {
                "name": "BranchA",
                "start": 8,
                "size": 16,
                "order": "intel",
                "factor": 1.0,
                "offset": 0.0,
                "minimum": 0.0,
                "maximum": 65535.0,
                "unit": "V",
                "mux": 0,
                "sweep": 3000,
            },
            {
                "name": "BranchB",
                "start": 8,
                "size": 16,
                "order": "intel",
                "signed": True,
                "factor": 0.5,
                "offset": -100.0,
                "minimum": -100.0,
                "maximum": 32667.5,
                "unit": "A",
                "mux": 1,
                "sweep": 40,
            },
        ),
    },
    {
        # Motorola（大端）专用报文：DBC 里 Motorola 的 start 是 **MSB 编号**，
        # 与 Intel 的线性位号不可混用，故单独占一帧避免与上面的信号重叠
        "name": "ChassisData",
        "can_id": 0x2C0,
        "size": 8,
        "cycle_ms": 10,
        "sender": "RSU",
        "channel": 0,
        "signals": (
            {
                "name": "WheelSpeedFL",
                "start": 7,
                "size": 12,
                "order": "motorola",
                "factor": 0.05625,
                "offset": 0.0,
                "minimum": 0.0,
                "maximum": 230.0625,
                "unit": "km/h",
                "receivers": ("Cluster", "DiagTool"),
                "sweep": 90,
            },
            {
                "name": "WheelSpeedFR",
                "start": 11,
                "size": 12,
                "order": "motorola",
                "signed": True,
                "factor": 0.05625,
                "offset": 0.0,
                "minimum": -115.0,
                "maximum": 115.0,
                "unit": "km/h",
                "sweep": 40,
            },
        ),
    },
    {
        # 与 EngineData 有同名信号 ``EngineSpeed`` → 触发「报文名.信号名」改名
        "name": "BrakeData",
        "can_id": 0x1A1,
        "size": 8,
        "cycle_ms": 10,
        "sender": "BrakeECU",
        "channel": 0,
        "signals": (
            {
                "name": "EngineSpeed",
                "start": 0,
                "size": 16,
                "order": "intel",
                "factor": 1.0,
                "offset": 0.0,
                "minimum": 0.0,
                "maximum": 65535.0,
                "unit": "rpm",
                "sweep": 200,
            },
            {
                "name": "BrakePressure",
                "start": 32,
                "size": 16,
                "order": "intel",
                "factor": 0.01,
                "offset": 0.0,
                "minimum": 0.0,
                "maximum": 655.35,
                "unit": "bar",
                "sweep": 60,
            },
        ),
    },
)

#: 默认 BLF 里出现、但**任何数据库都没有定义**的报文 ID（未匹配上报路径用）
CAN_UNMATCHED_ID = 0x7FF


def _dbc_number(value) -> str:
    r"""把 factor/offset/min/max 写成 DBC 可解析的十进制文本。

    刻意避免科学计数法与 ``1e-05`` 这类写法：canmatrix 的正则
    ``[0-9.+\-eE]+`` 虽能解析，但 cantools 的 DBC 词法对个别形式会报错，
    交叉验证要求同一份文本两个解析器都能读。
    """
    from decimal import Decimal

    d = Decimal(str(value))
    text = format(d, "f")
    if text in ("-0", "-0.0"):
        text = text.lstrip("-")
    return text


def _dbc_signals(frame: dict) -> list[dict]:
    """补全信号默认字段，返回可直接渲染成 DBC 行的 dict 列表。"""
    out = []
    for sig in frame.get("signals", ()):
        row = dict(sig)
        row.setdefault("size", 8)
        row.setdefault("order", "intel")
        row.setdefault("signed", False)
        row.setdefault("factor", 1.0)
        row.setdefault("offset", 0.0)
        row.setdefault("minimum", 0.0)
        row.setdefault("maximum", 0.0)
        row.setdefault("unit", "")
        row.setdefault("receivers", ("Vector__XXX",))
        row.setdefault("values", None)
        row.setdefault("comment", "")
        row.setdefault("float_bits", 0)
        row.setdefault("mux", None)
        out.append(row)
    return out


def write_dbc(
    path: Path | str,
    frames: list[dict] | tuple[dict, ...] | None = None,
    *,
    encoding: str = "utf-8",
) -> Path:
    """写入合成 DBC 文件（纯文本模板，不经任何第三方库生成）。

    刻意手写文本而非调用 ``canmatrix.formats.dump``：夹具必须独立于被测
    的解析路径，否则「canmatrix 写 + canmatrix 读」只能证明它自己没坏。

    Args:
        path: 目标路径（扩展名由调用方决定，一般是 ``.dbc``）
        frames: 报文定义，默认 :data:`CAN_FRAMES`
        encoding: 文件编码；传 ``cp1252`` 可复现真实 DBC 的欧洲注释

    Returns:
        写入的 Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = tuple(frames) if frames is not None else CAN_FRAMES

    ecus: list[str] = []
    for fr in frames:
        sender = fr.get("sender", "Vector__XXX")
        if sender not in ecus:
            ecus.append(sender)
        for sig in _dbc_signals(fr):
            for recv in sig["receivers"]:
                if recv != "Vector__XXX" and recv not in ecus:
                    ecus.append(recv)

    lines: list[str] = [
        'VERSION ""',
        "",
        "NS_ : ",
        "\tNS_DESC_",
        "\tCM_",
        "\tBA_DEF_",
        "\tBA_",
        "\tVAL_",
        "\tEV_DATA_",
        "\tENVVAR_DATA_",
        "\tSGTYPE_",
        "\tSGTYPE_VAL_",
        "\tSIG_TYPE_REF_",
        "\tVAL_TABLE_",
        "\tSIG_GROUP_",
        "\tSIG_VALTYPE_",
        "\tBU_BO_REL_",
        "\tSG_MUL_VAL_",
        "",
        "BS_: ",
        "",
        "BU_: " + " ".join(ecus) if ecus else "BU_: ",
        "",
    ]

    val_rows: list[str] = []
    sig_valtype_rows: list[str] = []
    comment_rows: list[str] = []
    cycle_rows: list[str] = []
    has_cycle = any(fr.get("cycle_ms") for fr in frames)

    for fr in frames:
        can_id = int(fr["can_id"])
        sender = fr.get("sender", "Vector__XXX")
        lines.append(
            f"BO_ {can_id} {fr['name']}: {int(fr.get('size', 8))} {sender}"
        )
        for sig in _dbc_signals(fr):
            order = 1 if sig["order"] == "intel" else 0
            sign = "-" if sig["signed"] else "+"
            mux_text = ""
            if sig["mux"] == "M":
                mux_text = "M "
            elif sig["mux"] is not None:
                mux_text = f"m{int(sig['mux'])} "
            receivers = ", ".join(sig["receivers"])
            lines.append(
                " SG_ {name} {mux}: {start}|{size}@{order}{sign} "
                "({factor},{offset}) [{minimum}|{maximum}] "
                '\"{unit}\" {receivers}'.format(
                    name=sig["name"],
                    mux=mux_text,
                    start=int(sig["start"]),
                    size=int(sig["size"]),
                    order=order,
                    sign=sign,
                    factor=_dbc_number(sig["factor"]),
                    offset=_dbc_number(sig["offset"]),
                    minimum=_dbc_number(sig["minimum"]),
                    maximum=_dbc_number(sig["maximum"]),
                    unit=sig["unit"],
                    receivers=receivers,
                )
            )
            if sig["values"]:
                pairs = " ".join(
                    f'{int(code)} "{text}"' for code, text in sorted(sig["values"].items())
                )
                val_rows.append(f"VAL_ {can_id} {sig['name']} {pairs} ;")
            if sig["float_bits"] == 32:
                sig_valtype_rows.append(f"SIG_VALTYPE_ {can_id} {sig['name']} : 1;")
            elif sig["float_bits"] == 64:
                sig_valtype_rows.append(f"SIG_VALTYPE_ {can_id} {sig['name']} : 2;")
            if sig["comment"]:
                comment_rows.append(
                    f'CM_ SG_ {can_id} {sig["name"]} "{sig["comment"]}";'
                )
        if fr.get("comment"):
            comment_rows.append(f'CM_ BO_ {can_id} "{fr["comment"]}";')
        if fr.get("cycle_ms"):
            cycle_rows.append(
                f'BA_ "GenMsgCycleTime" BO_ {can_id} {int(fr["cycle_ms"])};'
            )
        lines.append("")

    lines += comment_rows + [""]
    if has_cycle:
        lines.append('BA_DEF_ BO_ "GenMsgCycleTime" INT 0 65535;')
        lines += cycle_rows + [""]
    lines += sig_valtype_rows + val_rows + [""]

    path.write_text("\n".join(lines), encoding=encoding)
    return path


def write_arxml(
    path: Path | str,
    frames: list[dict] | tuple[dict, ...] | None = None,
    *,
    cluster: str = "SynCluster",
) -> Path:
    """写入合成 AUTOSAR 4.x ARXML（由 canmatrix 导出）。

    与 :func:`write_dbc` 不同，这里**借用 canmatrix 的导出器**：ARXML 手写
    成本过高（ISignal/IPdu/IFrame/COMPU-METHOD 四层引用）。代价是数据库
    层无法与 cantools 交叉验证（实测 cantools 读不懂 canmatrix 导出的
    ARXML：``BASE-TYPE-ENCODING in base type "sint16" does not exist``），
    因此 ARXML 夹具只用于验证 **cluster flatten、多库合并、UI 链路**，
    位段解码的正确性一律由 DBC 夹具与 canmatrix 逐位对照承担。

    已知导出损失：``cycle_time`` 不落盘、帧名被加 ``FRAME_`` 前缀。
    """
    import canmatrix
    from canmatrix.formats import dump

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = tuple(frames) if frames is not None else CAN_FRAMES

    matrix = canmatrix.CanMatrix()
    for fr in frames:
        frame = canmatrix.Frame(
            name=fr["name"],
            arbitration_id=canmatrix.ArbitrationId(
                int(fr["can_id"]), bool(fr.get("extended", False))
            ),
            size=int(fr.get("size", 8)),
        )
        for sig in _dbc_signals(fr):
            signal = canmatrix.Signal(
                name=sig["name"],
                start_bit=int(sig["start"]),
                size=int(sig["size"]),
                is_little_endian=sig["order"] == "intel",
                is_signed=bool(sig["signed"]),
                factor=float(sig["factor"]),
                offset=float(sig["offset"]),
                min=float(sig["minimum"]),
                max=float(sig["maximum"]),
                unit=sig["unit"],
                receivers=list(sig["receivers"]),
            )
            for code, text in sorted((sig["values"] or {}).items()):
                signal.add_values(int(code), str(text))
            if sig["mux"] == "M":
                signal.multiplex = "Multiplexor"
            elif sig["mux"] is not None:
                signal.mux_val = int(sig["mux"])
            frame.add_signal(signal)
        matrix.add_frame(frame)
    matrix.update_ecu_list()

    class _Sink:
        def __init__(self):
            self.chunks: list[bytes] = []

        def write(self, data):
            self.chunks.append(data if isinstance(data, bytes) else data.encode())

    sink = _Sink()
    dump({cluster: matrix}, sink, "arxml")
    path.write_bytes(b"".join(sink.chunks))
    return path


def can_frame_spec(frames, name: str) -> dict:
    """按报文名取定义，供测试构造 payload 时引用。"""
    frames = tuple(frames) if frames is not None else CAN_FRAMES
    for fr in frames:
        if fr["name"] == name:
            return fr
    raise KeyError(name)


def encode_frame_payload_raw(size: int, data: bytes) -> bytes:
    """右侧补零 / 截断到 dlc，直接给出 payload（不依赖任何数据库）。"""
    size = int(size)
    return (bytes(data) + b"\x00" * size)[:size]


def make_blf_messages(
    t0: float = 1_700_000_000.0,
    n: int = 50,
    *,
    frames=None,
    buses: tuple[int, ...] = (0, 1),
    with_unmatched: bool = True,
    with_remote: bool = False,
    with_error: bool = False,
    fd: bool = False,
) -> list[dict]:
    """生成写 BLF 用的报文描述列表（:func:`write_blf` 的入参）。

    时间戳是**绝对 unix 秒**（python-can 的约定），BLFLazyLoader 负责减去
    首帧归零。``buses`` 里的每条总线都会收到全量报文 —— 这正是真实多总线
    日志里「同一 CAN ID 出现在多个通道」的形态，也是验证
    ``(channel, can_id)`` 分桶与重名改名的最小样本。

    每个信号的物理值由该信号的 ``sweep``（峰值）与序号决定，因而
    **期望值无需第三方库即可算出**，测试可直接断言解码结果。
    """
    frames = tuple(frames) if frames is not None else CAN_FRAMES
    buses = tuple(buses) or (0,)
    rows: list[dict] = []
    step_s = 0.01
    seq = 0
    mux_round = 0
    for i in range(n):
        fr = frames[i % len(frames)]
        sigs = _dbc_signals(fr)
        # mux 选路按「多路复用帧自己出现的次数」交替，且同一轮的全部总线共用一个
        # 选择值。两点都不能省：用全局 seq 的话，4 帧轮转下 DiagMux 只落在奇数
        # seq，``seq % 2`` 恒为 1，另一个分支永远全 NaN；按 (i, bus) 计数的话，
        # 每条总线各自只看到偶数轮次，同样只有一条分支有数据 —— 两种写法都会让
        # mux 掩码路径没被走到却照样绿灯。
        mux_sel = mux_round % 2
        if any(s["mux"] == "M" for s in sigs):
            mux_round += 1
        for bus in buses:
            values = {}
            for sig in sigs:
                if sig["mux"] == "M":
                    values[sig["name"]] = float(mux_sel)
                    continue
                if sig["mux"] is not None and int(sig["mux"]) != mux_sel:
                    # 非当前 mux 值的分支：不写，解码后应为 NaN
                    continue
                top = float(sig.get("sweep", 1) or 1)
                base = float(sig["minimum"])
                values[sig["name"]] = base + top * (seq % 32) / 31.0
            payload = encode_frame_payload(fr, values)
            rows.append(
                {
                    "timestamp": t0 + seq * step_s,
                    "channel": bus,
                    "can_id": int(fr["can_id"]),
                    "extended": bool(fr.get("extended", False)),
                    "data": payload,
                    "fd": fd,
                    "_frame": fr["name"],
                }
            )
            if with_unmatched and seq % 10 == 5:
                rows.append(
                    {
                        "timestamp": t0 + seq * step_s + 0.001,
                        "channel": bus,
                        "can_id": CAN_UNMATCHED_ID,
                        "data": b"\x11\x22",
                    }
                )
            if with_remote and seq % 10 == 6:
                rows.append(
                    {
                        "timestamp": t0 + seq * step_s + 0.002,
                        "channel": bus,
                        "can_id": 0x500 + seq,
                        "remote": True,
                        "data": b"",
                    }
                )
            if with_error and seq % 20 == 19:
                rows.append(
                    {
                        "timestamp": t0 + seq * step_s + 0.003,
                        "channel": bus,
                        "can_id": 0,
                        "error": True,
                        "data": b"",
                    }
                )
            seq += 1
    rows.sort(key=lambda row: row["timestamp"])
    return rows


def encode_frame_payload(frame: dict, values: dict) -> bytes:
    """把物理值编成 payload 字节（:func:`_encode_bits` 的公开入口）。

    位拼装用的是本文件自己的一份标量实现，**不走** :mod:`src.data.can_decode`
    的向量化路径；它自己的正确性由
    ``tests/unit/data/test_blf_loader.py::test_decoded_series_matches_cantools_reference_decode``
    拿 cantools（独立实现）逐信号交叉验证 —— 那一条同时约束了解码器与本工厂，
    两边任何一方写错位序都会报。
    不用 cantools 直接编码是因为大样本（百万帧）下临时建库会成为瓶颈。
    """
    return _encode_bits(frame, values)


def _encode_bits(frame: dict, values: dict) -> bytes:
    """逐位拼装 payload：raw = (phys - offset) / factor，再按位序写位。

    位号约定与 canmatrix 的内部表示一致（已与 :mod:`src.data.can_decode`
    的向量化解码器双向对过）：

    * Intel：DBC 的 ``start`` 就是信号 **LSB** 的线性位号
      ``byte*8 + 字节内位号（0 = LSB）``，位号越高越重要；
    * Motorola：DBC 的 ``start`` 是 **MSB** 的网络位号，先按
      ``p - p%8 + 7 - p%8`` 换算成 canmatrix 内部位号，再取
      ``[内部位号, 内部位号+size)`` 作为从 MSB 到 LSB 的位区间
      （字节内位号此时从 MSB 起算）。
    """
    import struct

    buf = bytearray(int(frame.get("size", 8)))
    for sig in _dbc_signals(frame):
        if sig["name"] not in values:
            continue
        phys = float(values[sig["name"]])
        bits = int(sig["size"])
        if sig["float_bits"] == 32:
            pattern = struct.pack(">f", phys)
        elif sig["float_bits"] == 64:
            pattern = struct.pack(">d", phys)
        else:
            factor = float(sig["factor"]) or 1.0
            raw = (phys - float(sig["offset"])) / factor
            if sig["signed"] and raw < 0:
                raw = int(round(raw)) + (1 << bits)
            raw = int(round(raw)) & ((1 << bits) - 1)
            pattern = raw.to_bytes((bits + 7) // 8 or 1, "big")
        # 裁成信号自身的 bits 位，保留从 MSB 到 LSB 的序列
        base = len(pattern) * 8 - bits
        seq = [(pattern[(base + i) >> 3] >> (7 - ((base + i) & 7))) & 1 for i in range(bits)]
        start = int(sig["start"])
        if sig["order"] == "intel":
            # positions[0] 对应 LSB，故用反序（LSB 优先）的 bit 序列
            pairs = [
                (seq[bits - 1 - k], (start + k) >> 3, (start + k) & 7)
                for k in range(bits)
            ]
        else:
            internal = start - (start % 8) + 7 - (start % 8)
            pairs = [
                (seq[k], (internal + k) >> 3, 7 - ((internal + k) & 7))
                for k in range(bits)
            ]
        for value_bit, byte_index, shift in pairs:
            buf[byte_index] |= value_bit << shift
    return bytes(buf)


def write_blf(
    path: Path | str,
    messages: list[dict],
    *,
    compressed: bool = True,
) -> Path:
    """写入合成 BLF（python-can 的 ``BLFWriter``）。

    Args:
        path: 目标路径
        messages: dict 列表，键 ``timestamp/channel/can_id/data``，可选
            ``extended/remote/error/fd/bit_rate_switch/dlc``
        compressed: True 用 python-can 默认压缩级别（复现真实 BLF 的
            LOG_CONTAINER + zlib 路径），False 走非压缩对象流

    Returns:
        写入的 Path

    Note:
        python-can 写出的 channel 是 1 基（``channel + 1``），读回时减 1；
        因此本工厂入参的 ``channel`` 与读到的 ``channel`` **数值相同**。
        交叉验证请用 ``vblf.BlfReader``，它保留 1 基通道号。
    """
    import can

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    writer = can.BLFWriter(
        str(path), compression_level=6 if compressed else 0
    )
    try:
        for row in messages:
            kwargs = {
                "timestamp": float(row.get("timestamp", 0.0)),
                "channel": int(row.get("channel", 0)),
                "is_error_frame": bool(row.get("error", False)),
                "is_remote_frame": bool(row.get("remote", False)),
                "is_extended_id": bool(row.get("extended", False)),
                "is_fd": bool(row.get("fd", False)),
                "bitrate_switch": bool(row.get("bit_rate_switch", False)),
            }
            data = bytes(row.get("data", b""))
            if kwargs["is_fd"]:
                kwargs["arbitration_id"] = int(row.get("can_id", 0))
                kwargs["data"] = data
                kwargs["dlc"] = int(row.get("dlc", max(8, len(data))))
            elif kwargs["is_error_frame"]:
                kwargs["arbitration_id"] = int(row.get("can_id", 0))
                kwargs["data"] = data
            else:
                kwargs["arbitration_id"] = int(row.get("can_id", 0))
                kwargs["data"] = data
            writer.on_message_received(can.Message(**kwargs))
    finally:
        writer.stop()
    return path


DEFAULT_BLF_NAME = "syn_can.blf"
DEFAULT_DBC_NAME = "syn_can.dbc"


def write_blf_pair(
    directory: Path | str,
    *,
    n: int = 50,
    basename: str = "syn_can",
    buses: tuple[int, ...] = (0, 1),
    with_unmatched: bool = True,
    with_remote: bool = False,
    with_error: bool = False,
    fd: bool = False,
    t0: float = 1_700_000_000.0,
    compressed: bool = True,
    frames=None,
) -> tuple[Path, Path]:
    """一次产出配对的 ``*.blf`` + ``*.dbc``，返回 ``(blf_path, dbc_path)``。"""
    directory = Path(directory)
    frames = tuple(frames) if frames is not None else CAN_FRAMES
    dbc = write_dbc(directory / f"{basename}.dbc", frames)
    msgs = make_blf_messages(
        t0, n=n, frames=frames, buses=buses, with_unmatched=with_unmatched,
        with_remote=with_remote, with_error=with_error, fd=fd,
    )
    blf = write_blf(directory / f"{basename}.blf", msgs, compressed=compressed)
    return blf, dbc
