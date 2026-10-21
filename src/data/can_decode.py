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


def mux_validity(payload: np.ndarray, frame: CanFrameSpec,
                 target: CanSignalSpec) -> Optional[np.ndarray]:
    """计算目标信号在本帧中的 mux 有效性掩码。

    多路复用帧里，非本 mux 值的采样点**没有数据**（报文的这一段是上一 mux 组
    的残留值）。不掩掉就会把残留画成台阶，因此掩不到的位置必须由调用方填 NaN。

    Returns:
        ``(N,)`` bool；None 表示该信号不参与多路复用（全部有效）
    """
    if not frame.is_multiplexed or frame.mux_signal is None:
        return None
    if target.mux is None or target.is_mux_selector:
        return None  # mux 为 None 的公共信号始终有效

    selector = frame.signal_by_name(frame.mux_signal)
    if selector is None:
        return None
    try:
        sel_raw = extract_raw(payload, selector)
    except ValueError:
        return None

    group = int(target.mux) if not isinstance(target.mux, str) else None
    if group is None:
        return None
    return sel_raw == group


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


def _rows_view(payload: np.ndarray, use: int) -> memoryview:
    """把 ``(N, use)`` 的字节窗口摊平成可按行偏移切的 memoryview。

    多维 memoryview 既不支持和 ``mv[i, :n]`` 下标，也不支持 ``mv[i]`` 子视图
    （``NotImplementedError: multi-dimensional sub-views are not implemented``），
    所以只能按行偏移在平直视图上切。切列后的视图本就非连续，ascontiguousarray
    这一步省不掉。
    """
    window = np.ascontiguousarray(payload[:, :use])
    return memoryview(window.ravel())


def decode_rows_via_canmatrix(payload: np.ndarray, sig: CanSignalSpec,
                              frame: CanFrameSpec) -> np.ndarray:
    """容器帧 / 复杂多路复用帧的整帧回落解码：解不出该信号即 **NaN**。

    这两类帧的位段布局由 canmatrix 自己处理（逐 PDU 的子报文偏移、扩展复用组
    表、SecOC 认证码截断），自研位提取会照着父帧的连续位序去切 —— 结果不是报错
    而是**给出一条看起来很像样的错曲线**，因此宁可空：NaN 在绘图里是断线，错值
    是会被人拿去下结论的数据。同理这里不再叠加向量化的 mux 掩码：canmatrix
    对未选中的组根本不会产出键，缺失本身就是正确的掩码。

    ``cm_frame`` 缺失或逐帧解码抛异常时一律填 NaN 并告警一次，不静默退回位提取。
    """
    n_rows = payload.shape[0]
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
        row = bytes(view[i * use:(i + 1) * use])
        try:
            entry = cm_frame.decode(bytearray(row)).get(sig.name)
        except Exception:
            if not warned:
                logger.warning(
                    "canmatrix 解码报文 %s 失败，信号 %s 及其后采样点置空",
                    frame.name, sig.name, exc_info=True,
                )
                warned = True
            continue
        if entry is not None:
            out[i] = float(getattr(entry, "phys_value", entry))
    return out


def decode_rows(payload: np.ndarray, sig: CanSignalSpec,
                frame: Optional[CanFrameSpec] = None,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
    """逐帧标量解码（向量化失败时的回落路径）。

    按帧布局分流：容器帧 / 复杂多路复用帧走 :func:`decode_rows_via_canmatrix`
    （掩码由 canmatrix 的缺键语义代替），其余（仅 ``size > 64`` 触发的回落）仍用
    大整数位提取 —— 那条路径与 canmatrix 逐位等价且快一个数量级。
    """
    if frame is not None and not frame.is_vectorizable:
        return decode_rows_via_canmatrix(payload, sig, frame)

    n_rows = payload.shape[0]
    out = np.empty(n_rows, dtype=np.float64)
    stride = payload.shape[1]
    dlc = frame.dlc if frame is not None else stride
    use = min(stride, dlc) if dlc else stride
    view = _rows_view(payload, use)
    for i in range(n_rows):
        out[i] = decode_single(view[i * use:(i + 1) * use], sig, frame)
    if mask is not None:
        valid = np.asarray(mask, dtype=bool)
        if valid.shape[0] == n_rows:
            out[~valid] = np.nan
    return out
