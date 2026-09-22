"""MDF 变量归属信息提取（纯函数，零 Qt / 零 asammdf 依赖）。

输入是 ``MDFLazyLoader.get_channel_info()`` 产出的**基础类型 dict**，输出是
「属性 - 值」有序行，供 ``var_info`` 直接放进快照的 sections。

为什么需要"推断"：MDF 标准里**没有**函数层级块（实测 5 个真实文件无
function hierarchy 块、mf4 ``attachments=0``，A2L 未内嵌），ETAS MDA 展示的
function 来自它自己加载的 A2L。我们能做的只是从通道名 / 通道注释 / 通道组
注释 / SI 源块里把等价信息还原出来，因此界面文案一律标注「（推断）」。

三条纪律：

1. **不 import asammdf**（项目有启动性能与打包体积的历史约束），也不 import
   Qt —— 全部规则可脱离文件做参数化单测。
2. **永不抛异常**：对 ``None`` / 空串 / 缺键一律返回空。归属信息是锦上添花，
   任何解析失败都不能让变量信息窗口打不开。
3. **不做缓存**：实测单次调用在 10 μs 级，无缓存价值；缓存反而要处理 reload
   失效。

规则的实测依据见 ``tmp/mdf_function_meta_extraction_report.md``。
"""

from __future__ import annotations

import html
import re

from src.core.config import MDF_ATTRIBUTION_ENABLED, MDF_GBK_TEXT_REPAIR

# ---------------------------------------------------------------------------
# 行标签（渲染顺序即此处的常量顺序，见 build_attribution_rows）
# ---------------------------------------------------------------------------

LABEL_DEVICE = "设备 / 总线"
LABEL_ECU = "ECU / 源节点"
LABEL_BUS_TYPE = "总线类型"
LABEL_SOURCE_TYPE = "采集源类型"
LABEL_GROUP = "测量组 / 报文"
LABEL_FUNCTION = "所属函数（推断）"
LABEL_FUNCTION_BASIS = "函数推断依据"
LABEL_SIGNAL_IN_GROUP = "报文内信号名"
LABEL_DB_SOURCE = "来源数据库"
LABEL_LONG_NAME = "层级显示名"
LABEL_HINT = "提示"

BASIS_NAME = "通道名层级结构"
BASIS_CG_COMMENT = "通道组注释(SingleShotGroup)"
BASIS_AUX_TEXT = "通道注释字段(推断)"

HINT_INFERRED = "归属由 MDF 通道名/注释推断，非 A2L 定义"

# ---------------------------------------------------------------------------
# 码值表（对齐 asammdf blocks/v4_constants.py 的 BUS_TYPE_TO_STRING /
# SOURCE_TYPE_TO_STRING；0 与 1 在实测中均为"未指定/其他"，不展示）
# ---------------------------------------------------------------------------

_BUS_TYPE_LABELS = {
    2: "CAN",
    3: "LIN",
    4: "MOST",
    5: "FLEXRAY",
    6: "K_LINE",
    7: "ETHERNET",
    8: "USB",
}

_SOURCE_TYPE_LABELS = {
    1: "ECU",
    2: "BUS",
    3: "IO",
    4: "TOOL",
    5: "USER",
    6: "VIDEO",
    7: "RADAR",
    8: "LIDAR",
    9: "PROTOCOL",
}

#: asammdf 为合成 v3 文件自动注入的源块名（真实 CANape v3 无 SI 块），
#: 直接展示给用户只会被当成垃圾信息
_NOISE_SOURCE_NAMES = frozenset({"channel inserted by python script"})

# ---------------------------------------------------------------------------
# 正则
# ---------------------------------------------------------------------------

#: v4 的 CN/HD 注释是 XML，正文包在 <TX> 内；实测真实文件为单行、合成文件带换行
_TX_RE = re.compile(r"<TX>(.*?)</TX>", re.S)

#: 通道名内嵌设备：'WaterTemp\XCP:1' / 'x\XCP:1#RAMCal'
_DEVICE_IN_NAME_RE = re.compile(r"^(?P<base>[^\\]+)\\(?P<dev>[^\\]*[:#][^\\]*)$")

#: 通道组注释 'KFMSNWDKVP\x\XCP:1\SingleShotGroup' → 函数 'KFMSNWDKVP\x'
_SINGLE_SHOT_GROUP_RE = re.compile(r"^(?P<fkt>.+?)\\[^\\]*\\SingleShotGroup$")

#: 'created from: CANDB: BMC_Spannung'（实测 mf4 1281 条 / .dat 45 条）
_DB_SOURCE_RE = re.compile(r"created from:\s*([A-Za-z0-9]+)\s*:\s*(.+)", re.I)

#: "标识符样式"判定：**不含 '.'** —— 实测描述性缩写 't.b.d.' / 'T.B.D.' 若允许
#: 点号会被误判成函数名，这是本规则唯一的真实误判来源
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")

#: HD 注释正文里的 'Key: Value' 行；键不含冒号（'Devices: XCP:1' 只在首个冒号切）
_HD_LINE_RE = re.compile(r"^(?P<k>[A-Za-z][A-Za-z ()/_-]{0,39})\s*:\s*(?P<v>.+)$")

#: HD 正文尾部的 CANape 占位噪声行
_HD_NOISE_LINES = frozenset({"", "§@", "§", "@"})

#: ``extract_tx`` 最多递进抽取几层（实测 asammdf 写 v3 header 注释时套了两层）
_TX_NESTING_LIMIT = 3

#: 试验级归属键 → 中文标签（顺序即展示顺序）
HD_ATTRIBUTION_KEYS: tuple[tuple[str, str], ...] = (
    ("Database", "数据库"),
    ("Experiment", "试验"),
    ("Workspace", "工作空间"),
    ("Devices", "设备清单"),
    ("Program Description", "程序描述"),
    # WP / RP 是 CANape 的参数页概念：working page（工作页，本次测量实际生效的
    # 参数集）与 reference page（参考页，出厂/烧录基线）。该译法经 5 个真实文件
    # 核实：两者总是成对出现，且 WP 基本就是「RP 基名 + 改动标记」（
    # 05I..._EGR → 05I..._EGR_1；…COOL FLOW → …COOL FLOW_EGR DCM）。
    # 曾误译作“写保护参数集 / 运行参数集”——“写保护”在 CANape 里是另一个
    # 完全不相关的功能（参数集读写保护状态），会把用户带偏。
    ("WP", "工作页"),
    ("RP", "参考页"),
)


# ---------------------------------------------------------------------------
# 文本归一
# ---------------------------------------------------------------------------


def _has_cjk(s: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in s)


def _ascii_skeleton(s: str) -> str:
    """只保留 ASCII 字符的序列，用于比对回转前后的“可读骨干”。"""
    return "".join(c for c in s if ord(c) < 0x80)


def repair_text(s) -> str:
    """回转被 latin-1/cp1252 误读的 GBK 中文（D5 保守触发）。

    实测 CANape 以 GBK 写入中文注释，asammdf 按单字节编码解码，得到
    ``'82ºÅµ¥ÌåµçÑ¹'``；再编码回单字节并按 GBK 解码即还原 ``'82号单体电压'``。

    四条保守条件保证不会把正确文本改坏：

    1. 原串**不含 CJK** —— 已正确解码的中文直接原样返回；
    2. 回转后必须含 CJK；
    3. 回转前后的 **ASCII 骨架必须相同** —— 这一条挡住真实存在的误修场景：
       法文 ``'Théorie du moteur'`` 里 'é'(0xE9) 会把后面的 'o' 当作 GBK 尾
       字节吞掉，骨架由 'Thorie…' 变成 'Thrie…' → 拒绝回转。
       （仅靠“长度相等”或“ASCII 位置对齐”都不行：GBK 两字节必塌成一
       字符，长度必然变；而骨架比对能同时挡住奇数个高位字节的偏移。）
    4. 全局开关 ``MDF_GBK_TEXT_REPAIR`` 可一键关停。

    cp1252 优先于 latin-1：asammdf 用 cp1252 解码时 0x80-0x9F 落在 €‚ƒ„ 等
    字符上，这些字符不在 latin-1 值域内，只用 latin-1 会漏修；反之遇到
    cp1252 未定义的 0x81/0x8D 等，编码失败后自动降到 latin-1。
    """
    if not s:
        return ""
    if not isinstance(s, str):
        s = str(s)
    if not MDF_GBK_TEXT_REPAIR:
        return s
    if _has_cjk(s):
        return s
    skeleton = _ascii_skeleton(s)
    for enc in ("cp1252", "latin-1"):
        try:
            raw = s.encode(enc)
        except UnicodeEncodeError:
            continue
        try:
            fixed = raw.decode("gbk")
        except (UnicodeDecodeError, LookupError):
            continue
        if (
            fixed != s
            and _has_cjk(fixed)
            and _ascii_skeleton(fixed) == skeleton
        ):
            return fixed
    return s


def extract_tx(text) -> str:
    """从块注释里取出人类可读正文。

    实测三种落点，互为兜底：

    - v4 真实文件：``<CNcomment xmlns="…"><TX>…</TX><names>…</raster></CNcomment>``
      （单行）；
    - v4 合成文件 / HD 注释：带换行的 ``<TX>`` 段；
    - v3：不是 XML，原样返回。

    可重入：asammdf 写 MDF3 的 header 注释时会对正文**再包一层并转义**（实测
    合成 v3 文件读回为 ``<HDcomment><TX>&lt;HDcomment&gt;&lt;TX&gt;Database: …``），
    所以抽出后若仍是 ``<TX>`` 结构就再抽一层，最多 3 层以防死循环。
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    body = text
    unescaped = False
    for _ in range(_TX_NESTING_LIMIT):
        match = _TX_RE.search(body)
        if match is None:
            break
        body = html.unescape(match.group(1))
        unescaped = True
    if not unescaped:
        body = html.unescape(body)
    lines = [ln.strip() for ln in body.splitlines()]
    # 去掉尾部噪声行（实测 CANape 在 HD 正文末尾写了一行孤立的 '§@'）
    while lines and lines[-1] in _HD_NOISE_LINES:
        lines.pop()
    return "\n".join(lines).strip()


def channel_aux_text(channel: dict) -> str:
    """通道级附加文本（v3/v4 落点不同，必须互为兜底）。

    实测：真实 v3 的 ``comment`` 与 ``description`` 95% 相同； asammdf 合成的
    v3 只有 ``description``（``comment`` 为空）；v4 只有 ``comment``。
    """
    if not isinstance(channel, dict):
        return ""
    for key in ("comment", "description"):
        text = repair_text(extract_tx(channel.get(key)))
        if text:
            return text
    return ""


# ---------------------------------------------------------------------------
# 名称与 HD 注释解析
# ---------------------------------------------------------------------------


def split_name_device(name, devices: frozenset[str] = frozenset()):
    """把 ``'WaterTemp\\XCP:1'`` 拆成 ``(基名, 设备)``；无设备时设备为空串。

    判据取"形态 + 白名单"两条：尾部含 ``:``/``#``（XCP/CAN 通道写法），或
    整体命中 HD 注释 ``Devices:`` 列出的设备名。仅凭"名字里有反斜杠"不够 ——
    实测真实文件里 ``CalcDev`` 这类计算设备不带冒号。
    """
    if not name:
        return "", ""
    if not isinstance(name, str):
        name = str(name)
    if "\\" not in name:
        return name, ""
    match = _DEVICE_IN_NAME_RE.match(name)
    if match:
        return match.group("base"), match.group("dev")
    base, _, tail = name.rpartition("\\")
    if base and tail and tail in devices:
        return base, tail
    return name, ""


def parse_hd_comment(text) -> dict[str, str]:
    """把 HD 注释正文解析成 ``{Key: Value}``（同键取首次出现）。

    实测 5/5 文件都是 CANape/INCA 通用格式：``<HDcomment><TX>Database: X ⏎
    Experiment: Y …</TX><common_properties>…``，故先抽 TX 再逐行切分。
    """
    body = extract_tx(text)
    out: dict[str, str] = {}
    for line in body.splitlines():
        match = _HD_LINE_RE.match(line.strip())
        if match:
            key = match.group("k").strip()
            out.setdefault(key, match.group("v").strip())
    return out


def parse_hd_devices(header_comment) -> frozenset[str]:
    """HD 注释里的 ``Devices:`` 清单，用作通道名设备后缀的白名单。"""
    listed = parse_hd_comment(header_comment).get("Devices", "")
    return frozenset(part.strip() for part in listed.split(",") if part.strip())


def extract_db_source(aux: str) -> tuple[str, str]:
    """从附加文本里剥出「来源数据库」，返回 ``(数据库标注, 剩余文本)``。

    实测 mf4 有 1281 条形如 ``created from: CANDB: BMC_Spannung``。必须**先剥离
    再分类** —— 这串文本含空格，若不剥离就会被判成"描述"，而它其实是最可靠的
    数据库出处信息。
    """
    aux = (aux or "").strip()
    if not aux:
        return "", ""
    match = _DB_SOURCE_RE.search(aux)
    if not match:
        return "", aux
    db, value = match.group(1).strip(), match.group(2).strip()
    rest = (aux[: match.start()] + " " + aux[match.end():]).strip()
    return f"{db}: {value}", rest


def classify_aux_text(aux: str, *, base_name: str = "", group: str = "") -> str:
    """判定通道附加文本的语义，返回 ``function`` / ``signal`` / ``description`` / ``''``。

    只有"长得像标识符"的文本才可能是 A2L 函数名；实测另两类必须排除：

    - 含空格的自然语言描述（``'Lambda actual value sensor 1 bank 1'``）；
    - 缩写占位（``'t.b.d.'`` / ``'T.B.D.'`` —— 点号不在标识符字符集内）。

    与组名同前缀的是**报文内信号名**（``group='AI50'`` + ``aux='AI50_3'``），
    与组名完全相同则是零信息，都不当函数。
    """
    aux = (aux or "").strip()
    if not aux:
        return ""
    if not _IDENTIFIER_RE.match(aux):
        return "description"
    if aux == base_name:
        return ""
    if group:
        if aux == group:
            return ""
        if aux.startswith(group):
            return "signal"
    return "function"


def _clean_source_name(value) -> str:
    """过滤 asammdf 注入的噪声源名（D7）。"""
    name = str(value or "").strip()
    if not name or name.lower() in _NOISE_SOURCE_NAMES:
        return ""
    return name


def _clean_value(value) -> str:
    """行值的统一清洗：只去首尾空白，**不做长度截断**。

    截断曾按 120 字符实现，理由是“不截断会把「值」列撑宽”—— 实测不成立
    （值列是 ``QHeaderView.Stretch``，3000 字符也不改变列宽，已由
    ``tests/component/test_variable_info_dialog_rendering.py::
    TestMdfAttributionPresentation`` 固化），而截断会把真值从 tooltip、
    行尾复制按钮与 Markdown 导出里一并削掉 —— 实测真实 mf4 有 141/4704
    个通道注释超 120 字符，截了就是丢数据。
    """
    return str(value or "").strip()


# ---------------------------------------------------------------------------
# 聚合器
# ---------------------------------------------------------------------------


def build_attribution_rows(info: dict) -> list[tuple[str, str]]:
    """从 ``get_channel_info()`` 的返回构造「归属信息」行。

    返回 ``[]`` 表示该变量没有任何可展示的归属信息 —— 调用方据此**不创建该
    section**（D3），避免空块。

    Args:
        info: ``MDFLazyLoader.get_channel_info()`` 的返回值。

    Note:
        两个开关（``MDF_ATTRIBUTION_ENABLED`` / ``MDF_GBK_TEXT_REPAIR``）在调用
        时读取，因此测试可用 monkeypatch 改写本模块的名字。
    """
    if not MDF_ATTRIBUTION_ENABLED or not isinstance(info, dict):
        return []

    channel = info.get("channel") or {}
    channel_group = info.get("channel_group") or {}
    source = info.get("source") or {}
    header = info.get("header") or {}

    devices = parse_hd_devices(header.get("comment"))
    base_name, name_device = split_name_device(channel.get("name"), devices)

    # 设备：v3 从通道名后缀拿（实测 4 个 .dat 99~100%），v4 从 SI 源块的 path
    # 拿（实测 mf4 100% 填充）。两条都试、先到先得，因为合成 v3 文件也会带
    # SI 块、而部分 v4 文件的通道名同样内嵌了设备。
    device = name_device or str(source.get("path") or "").strip()
    ecu = _clean_source_name(source.get("name"))

    group = str(channel_group.get("acq_name") or channel_group.get("comment") or "").strip()

    aux = channel_aux_text(channel)
    db_source, aux_rest = extract_db_source(aux)

    function, basis, signal = "", "", ""
    pure_name = base_name
    if "/" in base_name:  # 规则 ①：'EpmCaS_phiSegOfs_CA/isx'
        pure_name, _, inner = base_name.partition("/")
        if pure_name:
            function, basis = pure_name, BASIS_NAME
    if not function and group:  # 规则 ②：通道组注释的 SingleShotGroup 形态
        match = _SINGLE_SHOT_GROUP_RE.match(group)
        if match:
            function, basis = match.group("fkt"), BASIS_CG_COMMENT
    kind = classify_aux_text(aux_rest, base_name=pure_name, group=group)
    if kind == "function" and not function:  # 规则 ③：标识符样式的注释
        function, basis = aux_rest, BASIS_AUX_TEXT
    elif kind == "signal":
        signal = aux_rest

    long_name = ""
    display_names = channel.get("display_names")
    if isinstance(display_names, dict):
        # 实测 v3 恒为 {} 或 {'': 'display_name'}（空键无意义），v4 真实文件的
        # 键形如 'CAN-Monitoring:1.BMS_CellVolt082'、合成文件为 {'…': 'display'}
        for key in display_names:
            if str(key or "").strip():
                long_name = str(key).strip()
                break

    bus_type = _BUS_TYPE_LABELS.get(source.get("bus_type"), "")
    source_type = _SOURCE_TYPE_LABELS.get(source.get("source_type"), "")

    candidates = (
        (LABEL_DEVICE, device),
        (LABEL_ECU, ecu),
        (LABEL_BUS_TYPE, bus_type),
        (LABEL_SOURCE_TYPE, source_type),
        (LABEL_GROUP, group),
        (LABEL_FUNCTION, function),
        (LABEL_FUNCTION_BASIS, basis),
        (LABEL_SIGNAL_IN_GROUP, signal),
        (LABEL_DB_SOURCE, db_source),
        (LABEL_LONG_NAME, long_name),
        (LABEL_HINT, HINT_INFERRED if basis == BASIS_AUX_TEXT else ""),
    )
    rows = [(label, _clean_value(value)) for label, value in candidates if value]
    return rows
