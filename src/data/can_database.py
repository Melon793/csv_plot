"""
CAN 数据库（DBC / ARXML）语义层

把 Vector 的 BLF（只有原始报文）与 CAN 数据库文件结合，产出本工具绘图所需的
帧 / 信号元数据。语义全部来自 canmatrix，本模块只做**结构收敛**：

- 把 ``canmatrix.formats.loadp()`` 的多形态返回值（单对象 / ARXML cluster dict /
  空 dict）统一 flatten 成 ``list[CanMatrix]``；
- 把 canmatrix 的 ``Frame`` / ``Signal`` 富对象抽取成只含基础类型的 dataclass，
  使解码层与绘图层不持有 canmatrix 对象引用（canmatrix 的对象图很大且非线程安全）；
- 提供 first-match-wins 的 ``find_frame()``：数据库列表的**顺序即优先级**。

本模块只依赖 canmatrix 与标准库，**不 import numpy / Qt**（启动性能与打包体积约束）。
"""

from __future__ import annotations

import logging
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional

from src.core.logger import get_logger

logger = get_logger(__name__)

# canmatrix 按文件后缀决定 import_type；本项目只支持这两类数据库文件
SUPPORTED_DB_EXTS = (".dbc", ".arxml")
_DB_IMPORT_TYPE = {".dbc": "dbc", ".arxml": "arxml"}

# 多路复用选择器在 canmatrix Signal.multiplex 中的标记值
_MUX_SELECTOR = "Multiplexor"

# canmatrix 对「isignal 未挂有效 compu method」的告警逐信号 logger.error（实测
# ARXML 演示文件 5 个信号 → 5 条 ERROR），而一次 BLF 导入流程会把数据库解析
# 2~4 遍（预扫命中率 + 正式解析 + 重绑），同一批告警会成刷屏 ERROR。解析窗口内
# 按此模式捕获汇总，其余 canmatrix 记录原样放行。
_COMPU_METHOD_RE = re.compile(
    r"No valid compu method found for isignal/systemsignal (\S+)/(\S+)"
)
# 解析可能发生在后台线程（预扫 / loader），接管全局 logger 的窗口必须串行化
_CM_CAPTURE_LOCK = threading.Lock()


class _CanmatrixCapture(logging.Handler):
    """解析窗口内接管 canmatrix 日志的 handler。

    命中 compu method 缺失模式的记录就地收集（按 isignal 名保序去重）；其余
    记录手动转发 root，保持原有可见性。handler 挂在 ``canmatrix`` 祖先 logger
    上，覆盖其全部格式子模块（arxml / dbc / ...）。
    """

    def __init__(self):
        super().__init__(level=0)  # 全量接住，级别过滤交给 root 的 handler
        self.missing_signals: list[str] = []
        self._seen: set[str] = set()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            match = _COMPU_METHOD_RE.search(record.getMessage())
            if match is None:
                logging.getLogger().handle(record)  # 原样放行
                return
            isignal = match.group(1)
            if isignal not in self._seen:
                self._seen.add(isignal)
                self.missing_signals.append(isignal)
        except Exception:  # noqa: BLE001 - 日志路径绝不能反向拖垮解析
            self.handleError(record)


@contextmanager
def _capture_canmatrix_noise() -> Iterator[_CanmatrixCapture]:
    """临时切断 canmatrix → root 的日志传播，由 collector 决定吞或放行。

    退出时无条件恢复 ``propagate`` 并摘除 handler；全局锁保证多线程解析不会
    互相偷走对方的日志窗口。
    """
    collector = _CanmatrixCapture()
    cm_logger = logging.getLogger("canmatrix")
    with _CM_CAPTURE_LOCK:
        cm_logger.addHandler(collector)
        prev_propagate = cm_logger.propagate
        cm_logger.propagate = False
        try:
            yield collector
        finally:
            cm_logger.propagate = prev_propagate
            cm_logger.removeHandler(collector)


# DBC 的导入编码候选。canmatrix 的 ``dbc.load`` 默认
# ``dbcImportEncoding='iso-8859-1'``，而 iso-8859-1 **逐字节可解、永不抛异常**，
# 于是 UTF-8 保存的 DBC 里的中文注释会被静默地解码成 mojibake（实测
# ``CM_ BO_ 416 "发动机数据"`` 读回来是 ``åå¨æºæ°æ®``）。所以只能把
# iso-8859-1 当最后兜底。
#
# gb18030 与 cp1252 都能“成功”解码对方的字节串，不能按顺序硬猜：
# cp1252 会把 GBK 中文解成 ``·¢¶¯»ú``，gb18030 也会把 ``Größe`` 的
# 0xF6 0xDF 当成一个汉字。判据见 :func:`_sniff_dbc_encoding`。
_DBC_ENCODING_FALLBACK = "iso-8859-1"

# CJK 统一表意文字（含扩展 A 与兼容区）。GBK 中文解出来是**连续**的多字 run；
# 而 cp1252 欧洲文本被 gb18030 误读时，每个重音字母只换来一个孤立汉字。
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
# cp1252 文本的“自然度”字符集：小写重音字母是正常欧洲语言，大写重音字母与
# 符号则是 GBK 字节被按 cp1252 解码后的典型乱码（``ÊÇ`` / ``·¢`` / ``¿ª``）。
_CP1252_LOWER = frozenset("àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ")
_CP1252_UPPER = frozenset("ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ×")
_CP1252_SYMBOLS = frozenset("°±²³µ·¸¹º»¼½¾¿¡¢£¤¥¦§¨ª«¬¯´®")


def _cp1252_naturalness(text: str) -> int:
    """cp1252 解码结果像不像自然欧洲语言；越低越像 GBK 乱码。"""
    score = 0
    for ch in text:
        if ch in _CP1252_LOWER:
            score += 1
        elif ch in _CP1252_UPPER or ch in _CP1252_SYMBOLS:
            score -= 1
        elif ord(ch) < 32 and ch not in "\t\r\n":
            score -= 3
    return score


def _sniff_dbc_encoding(path: str) -> str:
    """猜一份 DBC 的文本编码（ARXML 自带 XML 声明，不需要这套）。

    顺序：UTF-8 → gb18030 / cp1252 二选一 → iso-8859-1 兜底。

    gb18030 的判据是“解码结果里存在连续两个及以上的汉字 run”。真实中文
    注释 / 枚举标签几乎总是多字词；欧洲 cp1252 文本被 gb18030 误读时只会
    产生**孤立**汉字（每个重音字节吃掉了后面的 ASCII 字母）。单字标签
    （``开`` / ``否``）另走一条窄判据：孤立汉字两侧必须是非字母数字边界
    （通常就是 DBC 的引号），且 cp1252 结果里没有小写重音字母、符号占多数
    （``¿ª`` / ``ÊÇ``）。``°C`` 这类符号后跟 ASCII 也会被 gb18030 读成一个
    汉字，但它两侧不是“被引号包住的独立词”，且 cp1252 自然度不够低。
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return _DBC_ENCODING_FALLBACK
    try:
        data.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    try:
        gb_text = data.decode("gb18030")
    except UnicodeDecodeError:
        gb_text = None
    try:
        cp_text = data.decode("cp1252")
    except UnicodeDecodeError:
        cp_text = None

    if gb_text is None:
        return "cp1252" if cp_text is not None else _DBC_ENCODING_FALLBACK
    if cp_text is None:
        return "gb18030"
    runs = _CJK_RUN_RE.findall(gb_text)
    if not runs:
        return "cp1252"
    if max(len(run) for run in runs) >= 2:
        return "gb18030"
    if (
        _has_standalone_cjk(gb_text)
        and _cp1252_naturalness(cp_text) <= -2
    ):
        return "gb18030"
    return "cp1252"


def _has_standalone_cjk(text: str) -> bool:
    """是否存在“两侧都不是字母数字”的孤立汉字（DBC 里即被引号包住的标签）。"""
    for match in _CJK_RUN_RE.finditer(text):
        if match.end() - match.start() > 1:
            return True
        before = text[match.start() - 1] if match.start() else ""
        after = text[match.end()] if match.end() < len(text) else ""
        if not before.isalnum() and not after.isalnum():
            return True
    return False


def import_type_for(path: str) -> Optional[str]:
    """按文件后缀返回 canmatrix 的 import_type，不支持的后缀返回 None。"""
    return _DB_IMPORT_TYPE.get(os.path.splitext(path)[1].lower())


@dataclass(slots=True)
class CanSignalSpec:
    """单个 CAN 信号的解码配方（全部为基础类型，可安全跨线程读取）。"""

    name: str
    start_bit: int
    size: int
    little_endian: bool
    signed: bool
    is_float: bool
    factor: float
    offset: float
    unit: str
    minimum: Optional[float]
    maximum: Optional[float]
    values: Optional[dict[int, str]]
    comment: str
    mux: Optional[Any]  # None / "Multiplexor" / int（mux 组号）
    mux_value: Optional[int]
    receivers: tuple[str, ...]
    cycle_time_s: Optional[float]
    db_path: str
    frame_name: str
    can_id: int
    extended: bool
    # DBC 文本里的位号（Intel = LSB 线性位号，Motorola = MSB 网络位号）。
    # ``start_bit`` 是 canmatrix 的**内部**位号（解码器用它），两者只在
    # Motorola 上不同；给用户看的必须是他 in CANoe / DBC 编辑器里看到的
    # 那个数字，故两个都存。
    dbc_start_bit: int = 0

    @property
    def is_enum(self) -> bool:
        return bool(self.values)

    @property
    def is_mux_selector(self) -> bool:
        return self.mux == _MUX_SELECTOR


@dataclass(slots=True)
class CanFrameSpec:
    """单条报文（帧）的元数据 + 其信号列表。"""

    name: str
    can_id: int
    extended: bool
    dlc: int
    cycle_time_s: Optional[float] = None
    signals: list[CanSignalSpec] = field(default_factory=list)
    mux_signal: Optional[str] = None
    is_multiplexed: bool = False
    is_complex_multiplexed: bool = False
    is_pdu_container: bool = False
    is_fd: bool = False
    comment: str = ""
    db_path: str = ""
    # canmatrix 的原始 Frame 对象，**只**在向量化解码不适用（PDU container、
    # 复杂多路复用）时由 ``decode_rows_via_canmatrix`` 调用 ``decode()``；位宽
    # > 64 bit 的普通帧走大整数位提取，不需要它。持有它的代价是几千个信号级
    # dataclass，换来的是「与 Vector 工具链口径一致」的兜底语义。
    cm_frame: Any = None

    @property
    def is_vectorizable(self) -> bool:
        """能否走 numpy 向量化位段解码。

        canmatrix 对容器帧（PDU container）与安全帧（SecOC）的位段布局有自己的
        特殊处理（逐 PDU 偏移、认证码截断），自研向量化路径复刻这些分支的收益极
        低且极易出错，因此这类帧整体回落到 ``can_decode.decode_rows_via_canmatrix``
        （真正调 :attr:`cm_frame` 的 ``decode()``，解不出的信号给 NaN）。
        """
        return not (self.is_pdu_container or self.is_complex_multiplexed)

    def signal_by_name(self, name: str) -> Optional[CanSignalSpec]:
        for sig in self.signals:
            if sig.name == name:
                return sig
        return None


@dataclass(slots=True)
class DbFileReport:
    """单个数据库文件的加载结果，供 :class:`CanDatabaseSelectorDialog` 状态列展示。"""

    path: str
    ok: bool
    frame_count: int = 0
    signal_count: int = 0
    error: str = ""
    warning: str = ""


def _as_bool(value: Any) -> bool:
    """canmatrix 的部分标记在不同版本里可能是 property 或方法，统一取值。"""
    if callable(value):
        try:
            value = value()
        except Exception:
            logger.debug("canmatrix 布尔属性调用失败", exc_info=True)
            return False
    return bool(value)


def _as_float(value: Any) -> Optional[float]:
    """canmatrix 的 factor/offset/min/max 是 Decimal，None/异常统一转 None。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cycle_ms_to_s(value: Any) -> Optional[float]:
    """canmatrix 的 cycle_time 单位是毫秒，0 / None 均表示「未定义」。"""
    ms = _as_float(value)
    if ms is None or ms <= 0:
        return None
    return ms / 1000.0


def _arbitration(frame: Any) -> tuple[int, bool]:
    """从 canmatrix Frame 取 (裸 ID, 是否扩展帧)。

    ``frame.arbitration_id`` 是 ``ArbitrationId``（含 ``.id`` / ``.extended``），
    但部分导出路径会直接给 int，两者都要能吃下。
    """
    aid = getattr(frame, "arbitration_id", None)
    if aid is None:
        return (-1, False)
    if isinstance(aid, int):
        # canmatrix 的复合整数把扩展位编在 bit31
        return (aid & 0x1FFFFFFF, bool(aid & 0x80000000))
    can_id = getattr(aid, "id", None)
    if can_id is None:
        return (-1, False)
    return (int(can_id), bool(getattr(aid, "extended", False)))


def _convert_values(values: Any) -> Optional[dict[int, str]]:
    """VAL_ / 状态表：{raw: text}。键转 int，值转 str，异常条目整体放弃。"""
    if not values:
        return None
    out: dict[int, str] = {}
    try:
        items = values.items()
    except AttributeError:
        return None
    for raw, text in items:
        try:
            out[int(raw)] = str(text)
        except (TypeError, ValueError):
            continue
    return out or None


def _mux_value_of(sig: Any) -> Optional[int]:
    """取信号所属的 mux 组号。

    canmatrix 1.2 的 ``Signal`` **没有** ``.mux`` 属性，也没有用声明式的
    ``mux_value``（它恒为 None）：真正的组号由 ``multiplex_setter`` 写进
    ``mux_val``。而 ``multiplex`` 本身是 ``'Multiplexor' | int | None``。
    三个来源按优先级读，保证 canmatrix 换实现时不至于静默丢信息。
    """
    for attr in ("mux_val", "multiplex", "mux_value"):
        value = getattr(sig, attr, None)
        if isinstance(value, bool) or value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _convert_signal(sig: Any, frame: Any, can_id: int, extended: bool,
                    db_path: str) -> CanSignalSpec:
    receivers = getattr(sig, "receivers", None) or ()
    return CanSignalSpec(
        name=str(sig.name),
        start_bit=int(sig.start_bit),
        size=int(sig.size),
        little_endian=bool(sig.is_little_endian),
        signed=bool(sig.is_signed),
        is_float=bool(getattr(sig, "is_float", False)),
        factor=_as_float(getattr(sig, "factor", None)) if getattr(sig, "factor", None) is not None else 1.0,
        offset=_as_float(getattr(sig, "offset", None)) or 0.0,
        unit=(str(sig.unit).strip() if getattr(sig, "unit", None) else "-") or "-",
        minimum=_as_float(getattr(sig, "min", None)),
        maximum=_as_float(getattr(sig, "max", None)),
        values=_convert_values(getattr(sig, "values", None)),
        comment=str(getattr(sig, "comment", None) or ""),
        mux=getattr(sig, "multiplex", None),
        mux_value=_mux_value_of(sig) if str(getattr(sig, "multiplex", None)) != _MUX_SELECTOR else None,
        receivers=tuple(str(r) for r in receivers),
        cycle_time_s=_cycle_ms_to_s(getattr(sig, "cycle_time", None)),
        db_path=db_path,
        frame_name=str(frame.name),
        can_id=can_id,
        extended=extended,
        dbc_start_bit=_dbc_start_bit(sig),
    )


def _dbc_start_bit(sig: Any) -> int:
    """取 DBC 文本位号（canmatrix 的 ``get_startbit(bit_numbering=1)``）。"""
    try:
        return int(sig.get_startbit(bit_numbering=1))
    except Exception:  # noqa: BLE001 - 旧版 canmatrix 无此参数时退回内部位号
        return int(getattr(sig, "start_bit", 0) or 0)


def _convert_frame(frame: Any, db_path: str) -> CanFrameSpec:
    can_id, extended = _arbitration(frame)
    dlc = int(getattr(frame, "size", 0) or 0)
    signals: list[CanSignalSpec] = []
    kept_raw_signals: list[Any] = []
    for s in (frame.signals or []):
        try:
            spec = _convert_signal(s, frame, can_id, extended, db_path)
        except Exception:
            logger.debug("跳过无法转换的信号 %s", getattr(s, "name", "?"), exc_info=True)
            continue
        # 零宽信号（实测 tmpb.dbc 的 zeroSignal size=0）与越界信号都无法解码：
        # canmatrix 自己对前者会抛 int('', 2) 异常，后者会切出空串。与其在绘图
        # 时给出一列常数值，不如在建模阶段直接丢弃（日志 debug 可查）。
        if spec.size <= 0:
            logger.debug("丢弃零宽信号 %s.%s", frame.name, spec.name)
            continue
        # Intel 与 Motorola 的 start_bit 都在各自的扁平位序里从 0 起算，两者统一
        # 要求位区间 [start_bit, start_bit+size) 落在 dlc*8 之内，否则切片为空。
        if dlc > 0 and spec.start_bit + spec.size > dlc * 8:
            logger.debug(
                "丢弃超出 dlc 的信号 %s.%s (start=%d size=%d dlc=%d)",
                frame.name, spec.name, spec.start_bit, spec.size, dlc,
            )
            continue
        signals.append(spec)
        kept_raw_signals.append(s)

    # 同步剔除 cm_frame 里的退化信号：canmatrix 的 ``Frame.decode()`` 会遍历
    # **全部** signals，零宽信号抛 ``int('', 2)``、越界信号切出空串 —— 不剪枝的
    # 话，同帧里其他正常信号的回落解码也会被一个垃圾条目带崩。
    try:
        frame.signals = kept_raw_signals
    except Exception:  # pragma: no cover - canmatrix 字段改名时的兼容兜底
        logger.debug("canmatrix Frame.signals 剪枝失败", exc_info=True)
    mux_name = None
    # 先从**已转换的** spec 里找选择器：canmatrix 的 ``Frame.get_multiplexer``
    # 在 1.2 里是个返回 ``Signal`` 对象的 property（不是名字、也不是方法），
    # 直接 ``str()`` 会把整个 Signal 的 repr 塞进 mux_signal。自己推等价且不
    # 依赖 canmatrix 版本；只在该帧的选择器信号被剪枝丢掉时才回落过去。
    for spec_sig in signals:
        if spec_sig.is_mux_selector:
            mux_name = spec_sig.name
            break
    if mux_name is None:
        try:
            muxer = getattr(frame, "get_multiplexer", None)
            if callable(muxer):
                muxer = muxer()
            if muxer is not None:
                mux_name = str(getattr(muxer, "name", muxer) or "") or None
        except Exception:
            logger.debug("get_multiplexer 提取失败", exc_info=True)
            mux_name = None
    spec = CanFrameSpec(
        name=str(frame.name),
        can_id=can_id,
        extended=extended,
        dlc=int(getattr(frame, "size", 0) or 0),
        cycle_time_s=_cycle_ms_to_s(getattr(frame, "cycle_time", None)),
        mux_signal=(str(mux_name) if mux_name else None),
        is_multiplexed=_as_bool(getattr(frame, "is_multiplexed", False)),
        is_complex_multiplexed=_as_bool(getattr(frame, "is_complex_multiplexed", False)),
        is_pdu_container=_as_bool(getattr(frame, "is_pdu_container", False)),
        is_fd=_as_bool(getattr(frame, "is_fd", False)),
        comment=str(getattr(frame, "comment", None) or ""),
        db_path=db_path,
        cm_frame=frame,
    )
    spec.signals = signals
    return spec


class CanDatabaseSet:
    """多个 DBC / ARXML 文件合并后的只读查询视图（顺序即优先级）。"""

    def __init__(self, paths: Iterable[str] = ()):
        self._paths: list[str] = []
        self._frames: list[CanFrameSpec] = []
        # {(can_id, extended): [CanFrameSpec, ...]}，按数据库加载顺序追加
        self._index: dict[tuple[int, bool], list[CanFrameSpec]] = {}
        self._reports: list[DbFileReport] = []
        paths = list(paths or [])
        if paths:
            self.load(paths)

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def load(self, paths: Iterable[str]) -> "CanDatabaseSet":
        """按列表顺序加载全部数据库文件。

        **单个文件失败只记 warning 不中断**：用户常常一次选中整个工程目录下的
        所有 dbc，其中个别老文件 canmatrix 读不动，此时其余文件的信号仍应可用。
        """
        self._paths = []
        self._frames = []
        self._index = {}
        self._reports = []

        for path in paths:
            self._paths.append(path)
            report = DbFileReport(path=path, ok=False)
            try:
                with _capture_canmatrix_noise() as cm_noise:
                    frames = self._load_matrix_list(path)
            except Exception as e:  # 解析失败 / 编码错误 / canmatrix 内部异常
                report.error = f"{type(e).__name__}: {e}"
                self._reports.append(report)
                logger.warning("CAN 数据库解析失败 %s: %s", path, e)
                continue

            for frame in frames:
                self._append_frame(frame)
            report.ok = True
            report.frame_count = sum(1 for f in self._frames if f.db_path == path)
            report.signal_count = sum(
                len(f.signals) for f in self._frames if f.db_path == path
            )
            if report.frame_count == 0:
                report.warning = "无 CAN 报文"
            self._append_compu_method_note(report, cm_noise.missing_signals)
            self._reports.append(report)

        logger.info(
            "CAN 数据库加载完成: %d 个文件, %d 报文 / %d 信号",
            len(self._reports), self.frame_count, self.signal_count,
        )
        return self

    @staticmethod
    def _append_compu_method_note(report: DbFileReport, missing: list[str]) -> None:
        """把 compu method 缺失汇总成一条 WARNING + 状态列 tooltip 提示。

        逐条 ERROR 由 :func:`_capture_canmatrix_noise` 吞掉；这里只发一条
        WARNING（预览前几个信号名，完整列表进 debug），并追加到 ``report.warning``
        —— 状态列的三态判定只看 ``ok`` / ``frame_count``，warning 字符串仅用于
        tooltip，不会改变 ✔ / ⚠ / ✖ 语义。
        """
        if not missing:
            return
        preview = ", ".join(missing[:8])
        if len(missing) > 8:
            preview += f" …等 {len(missing)} 个"
        logger.warning(
            "数据库 %s: %d 个信号缺少有效换算方法（compu method 缺失），"
            "这些信号将按原始码值显示: %s",
            report.path, len(missing), preview,
        )
        logger.debug(
            "缺少 compu method 的完整信号列表 %s: %s",
            report.path, ", ".join(missing),
        )
        note = f"{len(missing)} 个信号缺少换算方法，按原始码值显示"
        report.warning = f"{report.warning}; {note}" if report.warning else note

    @staticmethod
    def _load_matrix_list(path: str) -> list[CanFrameSpec]:
        """加载单个数据库文件，返回扁平化后的帧列表。

        ``canmatrix.formats.loadp()`` 的返回值形态按格式不同：

        - DBC: ``{"db": CanMatrix}``
        - ARXML: ``{cluster 名: CanMatrix}``（**多 cluster**），实测可能是 ``{}``
          （如仅含 CompuMethod 定义、无帧的文件）或以 ``""`` 为键
        全部统一 flatten 合并，cluster 名不参与匹配（同 ID 冲突仍按文件顺序决定）。
        """
        import canmatrix.formats

        itype = import_type_for(path)
        if itype is None:
            raise ValueError(f"不支持的数据库类型: {os.path.splitext(path)[1]}")

        options: dict[str, str] = {}
        if itype == "dbc":
            enc = _sniff_dbc_encoding(path)
            if enc != "utf-8":
                # 非 UTF-8 属于“猜”的结果：用户遇到标签乱码时先看这条日志
                logger.info("DBC %s 按 %s 解码", os.path.basename(path), enc)
            options = {"dbcImportEncoding": enc, "dbcImportCommentEncoding": enc}

        loaded = canmatrix.formats.loadp(path, import_type=itype, key="db", **options)
        if loaded is None:
            raise ValueError("canmatrix 未返回任何数据库内容")
        if isinstance(loaded, dict):
            matrices = list(loaded.values())
        else:  # 某些版本 / 某些格式直接返回 CanMatrix 单对象
            matrices = [loaded]

        frames: list[CanFrameSpec] = []
        for matrix in matrices:
            for frame in (getattr(matrix, "frames", None) or []):
                try:
                    spec = _convert_frame(frame, path)
                except Exception:
                    # 单帧属性异常（如 size=0 的占位帧）不应拖垮整个数据库
                    logger.debug("跳过无法转换的报文 %s", getattr(frame, "name", "?"),
                                 exc_info=True)
                    continue
                if spec.can_id < 0 or not spec.signals:
                    continue
                frames.append(spec)
        return frames

    def _append_frame(self, frame: CanFrameSpec):
        self._frames.append(frame)
        self._index.setdefault((frame.can_id, frame.extended), []).append(frame)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def find_frame(self, channel: int, can_id: int,
                   extended: bool = False) -> Optional[CanFrameSpec]:
        """按 (can_id, extended) 返回**第一个**命中的帧定义。

        ``channel`` 当前不参与匹配（DBC/ARXML 里没有总线通道概念），保留形参是
        为将来的「按总线绑定不同数据库」留口子。同一 ID 在多库中重复时，
        列表里靠前的库生效 —— 这正是选择对话框允许调序的意义。
        """
        hits = self._index.get((can_id, bool(extended)))
        return hits[0] if hits else None

    def has_frame(self, can_id: int, extended: bool = False) -> bool:
        return (int(can_id), bool(extended)) in self._index

    @property
    def frames(self) -> list[CanFrameSpec]:
        return self._frames

    @property
    def source_files(self) -> list[str]:
        return list(self._paths)

    @property
    def reports(self) -> list[DbFileReport]:
        return list(self._reports)

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def signal_count(self) -> int:
        return sum(len(f.signals) for f in self._frames)

    @property
    def load_errors(self) -> list[str]:
        """**真正的解析失败**，不含「无 CAN 报文」这类 warning。

        §5.2 状态列的三态各有归宿：``✖`` 读 ``error``、``⚠`` 读 ``warning``。
        把 warning 混进本列表会让「N 个错误」的提示与 banner 状态③说谎。
        """
        return [f"{r.path}: {r.error}" for r in self._reports if not r.ok]

    @property
    def loaded_files(self) -> list[str]:
        return [r.path for r in self._reports if r.ok]

    def __len__(self) -> int:
        return len(self._frames)
