"""
CAN 信号的 numpy 向量化位段解码

解码公式与 canmatrix 的 ``Frame.decode()`` **逐位等价**（由
``tests/unit/data/test_can_decode.py`` 的随机对照用例锁定），实现依据是 canmatrix
``canmatrix.py`` 中的三段源码：

- ``bytes_to_bitstrings()``：``little`` = 各字节 8 位二进制串**按字节逆序**拼接、
  串内仍是 MSB→LSB；``big`` = 按字节正序拼接。
- ``bitstring_to_signal_list()``：Intel 取 ``little[size-total ... size-start_bit]``
  的等价形式，Motorola 取 ``big[start_bit : start_bit+size]``。
- ``unpack_bitstring()``：``int(bits, 2)`` → 有符号时 ``-= 1 << len(bits)``；
  float 走 ``'>f'`` / ``'>d'``（**切片本身就是 MSB 优先的字节流**）。

由此推出两条只依赖移位与掩码的规则（``nb`` = 参与解码的字节数）：

- Intel（``is_little_endian``）: ``raw = LE_int(payload) >> start_bit & mask``
- Motorola: ``raw = BE_int(payload) >> (nb*8 - start_bit - size) & mask``

整数值放不进 uint64（``size > 64``，常见于 CAN FD 的大信号）或帧布局特殊
（PDU container / 复杂多路复用）时，向量化路径不适用，回落逐帧解码；
:func:`decode_single` / :func:`decode_rows` 用 Python 大整数处理**任意**位宽。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from src.core.logger import get_logger
from src.data.can_database import CanFrameSpec, CanSignalSpec

logger = get_logger(__name__)

_UINT64 = np.uint64
_MASK64 = (1 << 64) - 1
# 逐帧回落是纯 Python 循环：超过该样本数时把日志提到 WARNING，让「打开一条
# 曲线卡了几秒」在日志里对得上号（评审 M1 的短期收敛；分块解码另行跟进）。
_ROW_FALLBACK_WARN_ROWS = 100_000


def _mask_for(size: int) -> int:
    return (1 << size) - 1


def _byte_window_le(payload: np.ndarray, byte0: int, n_rows: int) -> np.ndarray:
    """把 payload[byte0 : byte0+8] 拼成 uint64 小端整数（缺失字节按 0 处理）。"""
    acc = np.zeros(n_rows, dtype=_UINT64)
    nb = payload.shape[1]
    for k in range(8):
        b = byte0 + k
        if b >= nb:
            break
        acc |= payload[:, b].astype(_UINT64) << _UINT64(8 * k)
    return acc


def _byte_window_be(payload: np.ndarray, byte0: int, n_rows: int) -> np.ndarray:
    """把 payload[byte0 : byte0+8] 拼成 uint64 大端整数（缺失字节按 0 处理）。"""
    acc = np.zeros(n_rows, dtype=_UINT64)
    nb = payload.shape[1]
    for k in range(8):
        b = byte0 + k
        if b >= nb:
            break
        acc |= payload[:, b].astype(_UINT64) << _UINT64(8 * (7 - k))
    return acc


def _shift_right(values: np.ndarray, shift: int) -> np.ndarray:
    return values >> _UINT64(shift) if shift else values


def extract_raw(payload: np.ndarray, sig: CanSignalSpec) -> np.ndarray:
    """向量化提取原始码值（未做 factor/offset 变换）。

    Args:
        payload: ``(N, stride)`` uint8 报文数据区
        sig: 信号解码配方

    Returns:
        ``(N,)`` 数组。非 float 有符号信号为 int64，无符号为 float64（便于与
        物理值统一做 NaN 掩码），float32/64 信号为 float64。

        无符号走 float64 的代价是 **> 2^53 的码值会丢精度**（64 bit 无符号计数
        器读不回原值）。物理值路径 ``raw * factor + offset`` 本身就是 float64，
        绘图中也不可能区分这两个量级；而唯一直接取 raw 的枚举信号位宽只到几位。
        因此不为此单独开 uint64 分支（那会把 uint64 数组送进 pyqtgraph）。

    Raises:
        ValueError: 位段宽度 > 64 bit，向量化路径不适用（调用方回落逐帧解码）
    """
    if payload.ndim != 2 or payload.dtype != np.uint8:
        raise ValueError("payload 必须是 (N, stride) 的 uint8 数组")
    if sig.size <= 0 or sig.size > 64:
        raise ValueError(f"信号 {sig.name} 位宽 {sig.size} 不适用于 uint64 向量化")

    n_rows = payload.shape[0]
    mask = _UINT64(_mask_for(sig.size) & _MASK64)

    if sig.little_endian:
        byte0 = sig.start_bit // 8
        bit0 = sig.start_bit % 8
        acc = _byte_window_le(payload, byte0, n_rows)
        raw = _shift_right(acc, bit0)
        if bit0:
            # acc 的低 64-bit 装不下 bit0 + size 位，缺口来自第 9 字节的高位
            needed = bit0 + sig.size - 64
            if needed > 0:
                b8 = byte0 + 8
                if b8 < payload.shape[1]:
                    extra = (payload[:, b8].astype(_UINT64)
                             >> _UINT64(8 - needed)) & _UINT64((1 << needed) - 1)
                    raw |= extra << _UINT64(64 - bit0)
        raw &= mask
    else:
        byte0 = sig.start_bit // 8
        bit0 = sig.start_bit % 8
        acc = _byte_window_be(payload, byte0, n_rows)
        if bit0 + sig.size <= 64:
            raw = _shift_right(acc, 64 - bit0 - sig.size)
        else:
            needed = bit0 + sig.size - 64
            b8 = byte0 + 8
            high = np.zeros(n_rows, dtype=_UINT64)
            if b8 < payload.shape[1]:
                high = payload[:, b8].astype(_UINT64) >> _UINT64(8 - needed)
            raw = ((acc << _UINT64(needed)) | high) & mask
            return _finalize(raw, sig, n_rows)
        raw &= mask

    return _finalize(raw, sig, n_rows)


def _finalize(raw: np.ndarray, sig: CanSignalSpec, n_rows: int) -> np.ndarray:
    """把 uint64 码值转成可参与物理换算的数值（补码 / IEEE float）。"""
    if sig.is_float and sig.size in (32, 64):
        int_dtype = np.dtype(">u4") if sig.size == 32 else np.dtype(">u8")
        flt_dtype = np.dtype(">f4") if sig.size == 32 else np.dtype(">f8")
        values = raw.astype(np.uint64).astype(int_dtype).view(flt_dtype)
        # 报文的未初始化区域常包含 signaling NaN 位模式（实测 0xff8b0af1），
        # float32 → float64 的静默转换会按 IEEE 754 抛 invalid 告警。这是
        # 数据本身的状态而不是解码缺陷，与 MDF 侧对 inf/nan 的处理一致。
        with np.errstate(invalid="ignore"):
            return values.astype(np.float64)

    if sig.signed and sig.size < 64:
        half = 1 << (sig.size - 1)
        return (raw.astype(np.int64) ^ np.int64(half)) - np.int64(half)

    if sig.signed and sig.size == 64:
        return raw.astype(np.int64)

    return raw.astype(np.float64)


def row_validity(lengths: Optional[np.ndarray],
                 sig: CanSignalSpec) -> Optional[np.ndarray]:
    """逐行判断信号占用的位段是否落在该行**真实**有效字节内。

    BLF 桶按桶内最大 stride 对齐存储，短行尾部是补出来的 0。补 0 只适合
    让数组形状一致，**不能参与解码**：CAN FD 的 ``valid_bytes < DLC`` 与
    经典 CAN 的 ``DLC < 8`` 都是常态，把缺失字节当 0 会静默给出
    ``0 * factor + offset`` 这类看似合理的错值。

    Args:
        lengths: ``(N,)`` 每行有效字节数；None 表示调用方没有行级信息
            （兼容独立使用，返回 None，不改变旧行为）
        sig: 信号配方；canmatrix 内部位号下，Intel/Motorola 都满足
            「占用位区间 = [start_bit, start_bit + size)」

    Returns:
        ``(N,)`` bool；True 表示该行包含完整信号位段
    """
    if lengths is None:
        return None
    need_bits = int(sig.start_bit) + int(sig.size)
    return lengths.astype(np.uint16) * 8 >= need_bits


def mux_validity(payload: np.ndarray, frame: CanFrameSpec,
                 target: CanSignalSpec,
                 lengths: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """计算目标信号在本帧中的 mux 有效性掩码。

    多路复用帧里，非本 mux 值的采样点**没有数据**（报文的这一段是上一 mux 组
    的残留值）。不掩掉就会把残留画成台阶，因此掩不到的位置必须由调用方填 NaN。

    传入 ``lengths`` 时还会排除选择器本身不完整的短行（补出来的 0 会被当成
    组 0）。目标信号自身的行级有效性是另一个维度，仍由调用方用
    :func:`row_validity` 合并 —— 这里只负责 mux。

    Returns:
        ``(N,)`` bool；None 表示该信号不参与多路复用（全部有效）
    """
    if not frame.is_multiplexed or frame.mux_signal is None:
        return None
    if target.is_mux_selector:
        return None  # 选择器自身对所有行都有效
    # canmatrix 1.2 用 ``multiplex`` 存组号；未来版本若只填 ``mux_val``
    # （解析层已兜到 ``mux_value``），这里必须跟着回退，否则会把非本组采样点
    # 当公共信号全放行（评审 L2）。
    group = target.mux if target.mux is not None else target.mux_value
    if group is None or isinstance(group, str):
        return None  # 公共信号 / 未知组：保持全部有效

    selector = frame.signal_by_name(frame.mux_signal)
    if selector is None:
        return None
    try:
        sel_raw = extract_raw(payload, selector)
    except ValueError:
        return None

    group = int(group)
    valid = sel_raw == group
    if lengths is not None:
        sel_ok = row_validity(lengths, selector)
        if sel_ok is not None:
            valid = valid & sel_ok
    return valid


def decode_physical(payload: np.ndarray, sig: CanSignalSpec,
                    mask: Optional[Sequence[bool] | np.ndarray] = None,
                    frame: Optional[CanFrameSpec] = None) -> np.ndarray:
    """向量化解码物理值：``raw * factor + offset``，mux 之外的采样点填 NaN。

    Raises:
        ValueError: 位宽不适用向量化（调用方回落 :func:`decode_rows`）
    """
    raw = extract_raw(payload, sig)
    factor = sig.factor if sig.factor is not None else 1.0
    offset = sig.offset or 0.0
    values = np.asarray(raw, dtype=np.float64) * factor + offset
    if mask is not None:
        values = values.copy()
        valid = np.asarray(mask, dtype=bool)
        if valid.shape[0] == values.shape[0]:
            values[~valid] = np.nan
        else:
            # 长度不一致时静默跳过，等于把 mux / 短行的掩码失效变成「全有效」，
            # 残留值会被画成曲线（评审 L7）。至少留一条可追的证据。
            logger.warning(
                "信号 %s 的有效性掩码长度 %d 与采样点数 %d 不一致，已忽略掩码",
                sig.name, valid.shape[0], values.shape[0],
            )
    return values


def decode_single(payload_row: bytes | memoryview, sig: CanSignalSpec,
                  frame: Optional[CanFrameSpec] = None) -> float:
    """任意位宽的标量解码（Python 大整数，与 canmatrix 的切片语义一致）。

    用于 ``size > 64`` 的信号与 payload 短于 dlc 的边界帧。比 canmatrix 的
    ``str(int, 2)`` 路径快一个数量级，且无需持有 canmatrix 对象。

    注意：PDU container / 复杂多路复用帧**不**走这里 —— 那类帧整帧交给
    :func:`decode_rows` 里的 canmatrix 分支（子报文偏移与扩展复用组表由
    canmatrix 自己处理，逐位切片会读到别的子报文的数据）。
    """
    if sig.size > 64 and frame is not None and frame.cm_frame is not None:
        # 优先用数据库自带的解码器，保证与 Vector 工具链口径一致
        try:
            decoded = frame.cm_frame.decode(bytearray(payload_row))
            entry = decoded.get(sig.name)
            if entry is not None:
                return float(getattr(entry, "phys_value", entry))
        except Exception:
            logger.debug("canmatrix 回落解码失败: %s", sig.name, exc_info=True)

    nb = len(payload_row)
    total_bits = nb * 8
    if sig.little_endian:
        width = sig.start_bit + sig.size
        if width > total_bits:
            return float("nan")
        value = int.from_bytes(payload_row, "little") >> sig.start_bit
    else:
        end = sig.start_bit + sig.size
        if end > total_bits:
            return float("nan")
        value = int.from_bytes(payload_row, "big") >> (total_bits - end)

    value &= _mask_for(sig.size)
    if sig.is_float and sig.size in (32, 64):
        import struct

        fmt = ">f" if sig.size == 32 else ">d"
        packed = value.to_bytes(sig.size // 8, "big")
        return float(struct.unpack(fmt, packed)[0])
    if sig.signed and (value >> (sig.size - 1)) & 1:
        value -= 1 << sig.size
    factor = sig.factor if sig.factor is not None else 1.0
    return float(value) * factor + (sig.offset or 0.0)


def _log_row_fallback(kind: str, n_rows: int, frame_name: str,
                      sig_name: str) -> None:
    """记录回落解码的规模（M1）：大样本时提到 WARNING，便于对上 UI 卡顿。"""
    if n_rows >= _ROW_FALLBACK_WARN_ROWS:
        logger.warning(
            "%s 逐帧解码 %d 个采样点（报文 %s，信号 %s），可能阻塞数秒",
            kind, n_rows, frame_name, sig_name,
        )
    else:
        logger.debug(
            "%s 逐帧解码 %d 个采样点（报文 %s，信号 %s）",
            kind, n_rows, frame_name, sig_name,
        )


def _rows_view(payload: np.ndarray, use: int) -> memoryview:
    """把 ``(N, use)`` 的字节窗口摊平成可按行偏移切的 memoryview。

    多维 memoryview 既不支持和 ``mv[i, :n]`` 下标，也不支持 ``mv[i]`` 子视图
    （``NotImplementedError: multi-dimensional sub-views are not implemented``），
    所以只能按行偏移在平直视图上切。``use == stride`` 且数组连续时窗口就是
    整行：直接 ravel，避免 30M 行 × stride 的瞬时副本；只有切列（``use <
    stride``）才必须 ascontiguousarray。
    """
    if use >= payload.shape[1] and payload.flags.c_contiguous:
        return memoryview(payload.ravel())
    window = np.ascontiguousarray(payload[:, :use])
    return memoryview(window.ravel())


def decode_rows_via_canmatrix(payload: np.ndarray, sig: CanSignalSpec,
                              frame: CanFrameSpec,
                              lengths: Optional[np.ndarray] = None) -> np.ndarray:
    """容器帧 / 复杂多路复用帧的整帧回落解码：解不出该信号即 **NaN**。

    这两类帧的位段布局由 canmatrix 自己处理（逐 PDU 的子报文偏移、扩展复用组
    表、SecOC 认证码截断），自研位提取会照着父帧的连续位序去切 —— 结果不是报错
    而是**给出一条看起来很像样的错曲线**，因此宁可空：NaN 在绘图里是断线，错值
    是会被人拿去下结论的数据。同理这里不再叠加向量化的 mux 掩码：canmatrix
    对未选中的组根本不会产出键，缺失本身就是正确的掩码。

    ``cm_frame`` 缺失或逐帧解码抛异常时一律填 NaN 并告警一次，不静默退回位提取。
    ``lengths`` 只用来跳过 0 字节行：canmatrix 的 ``Frame.decode()`` 要求整帧
    长度，喂短行会抛 ``DecodingFrameLength``。这里仍喂对齐后的整行（缺失字节
    为 0），有效段内信号的解码结果不受影响；跨越缺失字节的信号由调用方的
    ``row_ok`` 掩码置 NaN，不会把补出来的 0 当数据。
    """
    n_rows = payload.shape[0]
    _log_row_fallback("容器/复杂多路复用帧", n_rows, frame.name, sig.name)
    out = np.full(n_rows, np.nan, dtype=np.float64)
    cm_frame = getattr(frame, "cm_frame", None)
    if cm_frame is None:
        logger.warning(
            "报文 %s 不适用向量化解码且无 canmatrix 原始帧，信号 %s 置空",
            frame.name, sig.name,
        )
        return out

    stride = payload.shape[1]
    dlc = frame.dlc or stride
    use = min(stride, dlc)
    view = _rows_view(payload, use)
    warned = False
    for i in range(n_rows):
        if lengths is not None and int(lengths[i]) <= 0:
            continue
        row = bytes(view[i * use:(i + 1) * use])
        try:
            entry = cm_frame.decode(bytearray(row)).get(sig.name)
        except Exception:
            if not warned:
                logger.warning(
                    "canmatrix 解码报文 %s 失败，信号 %s 当前采样点置空",
                    frame.name, sig.name, exc_info=True,
                )
                warned = True
            continue
        if entry is not None:
            out[i] = float(getattr(entry, "phys_value", entry))
    return out


def decode_rows(payload: np.ndarray, sig: CanSignalSpec,
                frame: Optional[CanFrameSpec] = None,
                mask: Optional[np.ndarray] = None,
                lengths: Optional[np.ndarray] = None) -> np.ndarray:
    """逐帧标量解码（向量化失败时的回落路径）。

    按帧布局分流：容器帧 / 复杂多路复用帧走 :func:`decode_rows_via_canmatrix`
    （掩码由 canmatrix 的缺键语义代替），其余（仅 ``size > 64`` 触发的回落）仍用
    大整数位提取 —— 那条路径与 canmatrix 逐位等价且快一个数量级。

    ``lengths`` 给出时按每行真实有效字节解码；``decode_single`` 本身对
    位段越界返回 NaN，这里再按 ``mask``（含行级有效性）统一收口。
    """
    if frame is not None and not frame.is_vectorizable:
        out = decode_rows_via_canmatrix(payload, sig, frame, lengths)
    else:
        n_rows = payload.shape[0]
        _log_row_fallback(
            "逐帧位提取", n_rows,
            frame.name if frame is not None else "-", sig.name,
        )
        out = np.full(n_rows, np.nan, dtype=np.float64)
        stride = payload.shape[1]
        dlc = frame.dlc if frame is not None else stride
        use = min(stride, dlc) if dlc else stride
        view = _rows_view(payload, use)
        for i in range(n_rows):
            row_len = min(int(lengths[i]), use) if lengths is not None else use
            if row_len <= 0:
                continue
            out[i] = decode_single(view[i * use:i * use + row_len], sig, frame)

    if mask is not None:
        valid = np.asarray(mask, dtype=bool)
        if valid.shape[0] == out.shape[0]:
            out[~valid] = np.nan
        else:
            logger.warning(
                "信号 %s 的有效性掩码长度 %d 与采样点数 %d 不一致，已忽略掩码",
                sig.name, valid.shape[0], out.shape[0],
            )
    return out
