"""
BLFLazyLoader - Vector BLF（CAN / CAN-FD 报文日志）惰性数据源

BLF 里只有**原始报文**（时间戳 + CAN ID + 数据字节），没有任何信号名，因此
必须搭配 DBC / ARXML 数据库才能还原出可绘图的信号。本模块的职责：

1. **打开时只扫报文、不解码信号**：一次线性扫描把每条报文按
   ``(总线通道, CAN ID)`` 分桶，原始 payload 常驻内存；
2. 把桶 ∩ 数据库帧定义映射成与 :class:`MDFLazyLoader` 同构的
   「组 / 变量」元数据，使 UI 的分组 tab、游标同步、X 轴 limits 全部零改动复用；
3. 绘图时才做 numpy 向量化解码，并按 LRU 缓存结果。

设计后果（也是「重选数据库」不需重扫报文的原因）：换数据库**不需要重扫报文**，
只重建元数据 + 清缓存，见 :meth:`bind_databases`（数据库解析仍是同步的）。

生命周期、并发与缓存纪律全部照抄 MDFLazyLoader 已验证的写法
（``RLock`` / ``_closed`` 先置位 / ``_ensure_open()`` 抛 ``KeyError`` /
``__del__ → close()`` 逐属性兜底）。
"""

from __future__ import annotations

import itertools
import os
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.core.config import (
    BLF_MAX_MESSAGES,
    BLF_MAX_PAYLOAD_BYTES,
    BLF_PENDING_ROWS_BUDGET,
    BLF_SCAN_CHUNK_MESSAGES,
    BLF_UNMATCHED_REPORT_LIMIT,
)
from src.data.can_database import CanDatabaseSet, CanFrameSpec, CanSignalSpec
from src.data.can_decode import (
    decode_physical,
    decode_rows,
    extract_raw,
    mux_validity,
    row_validity,
)
from src.data.metadata import UNKNOWN, VarMetadata
from src.core.logger import get_logger

logger = get_logger("data.blf")

# 每个桶的攒块阈值：太小会让 np.array 的调用次数与固定开销占比上升，
# 太大会让「一次 flush」在取消路径上成为不可中断的长尾。
_BUCKET_BLOCK_ROWS = 8192


def _safe_tell(handle, fallback: int) -> int:
    """读取句柄当前位置，句柄已被关闭时返回 ``fallback``。

    python-can 的 ``BLFReader.__iter__`` 在正常走到 EOF 时会调用 ``self.stop()``
    并**关闭传入的句柄**（实测 4.6.1），于是「读完最后一块」之后的那次
    ``tell()`` 必抛 ``ValueError: I/O operation on closed file``。此时进度本就
    该是 100%，直接回报文件大小即可，无需与第三方库的所有权语义纠缠。
    """
    try:
        return int(handle.tell())
    except (ValueError, OSError):
        return int(fallback)


def prescan_blf_frames(path: str, *, should_cancel=None):
    """快速预扫 BLF：只统计 ``(通道, CAN ID, 扩展位)`` 的帧数，不存任何 payload。

    供选库对话框在后台估算「每个数据库能解出多少帧」（命中列）：BLF 没有索引，
    报文 ID 集合只有完整读一遍才知道，而完整读一遍就是 ``_scan`` 的耗时 ——
    所以预扫必须是后台线程里的可取消轻量循环。刻意与 :meth:`BLFLazyLoader._scan`
    保持同一套过滤口径（跳过错误帧 / 远程帧、通道号 -1 归 0、同一上限截断），
    否则命中率与实际解码结果会对不上。

    Returns:
        ``Counter[(channel, can_id, extended), 帧数]``；帧数总和即参与解码的
        总帧数。文件损坏 / 无报文 / python-can 缺失时抛异常，由调用方兜底。

    Args:
        should_cancel: 轮询式取消谓词，返回 True 时抛 :class:`InterruptedError`。
    """
    from collections import Counter

    try:
        from can import BLFReader
    except ImportError as e:
        raise ImportError(f"{BLFLazyLoader._CAN_IMPORT_ERROR}\n原始错误: {e}") from e

    counts: Counter = Counter()
    total = 0
    total_bytes = 0
    handle = open(path, "rb")
    try:
        reader = BLFReader(handle)
        iterator = iter(reader)
        while True:
            chunk = list(itertools.islice(iterator, BLF_SCAN_CHUNK_MESSAGES))
            if not chunk:
                break
            for msg in chunk:
                if msg.is_error_frame or msg.is_remote_frame:
                    continue  # 与 _scan 同口径：这两类帧不产生采样点
                channel = int(msg.channel) if msg.channel is not None else 0
                counts[(max(0, channel), int(msg.arbitration_id),
                        bool(msg.is_extended_id))] += 1
                total_bytes += len(msg.data) if msg.data else 0
            total = sum(counts.values())
            if total >= BLF_MAX_MESSAGES or total_bytes >= BLF_MAX_PAYLOAD_BYTES:
                # 与 _scan 的截断一致：超限部分不会生成信号，命中率按会解码的部分算
                logger.warning(
                    "BLF 预扫达到上限（%s 帧 / %s MB），其余报文不计入命中率",
                    f"{BLF_MAX_MESSAGES:,}",
                    f"{BLF_MAX_PAYLOAD_BYTES // (1024 * 1024)}",
                )
                break
            if should_cancel is not None and should_cancel():
                raise InterruptedError("预扫已取消")
            time.sleep(0)  # 让出 GIL，别在 UI 线程忙等时挤占事件循环
    finally:
        try:
            handle.close()
        except Exception:
            logger.debug("关闭 BLF 预扫句柄失败", exc_info=True)

    if not counts:
        raise ValueError("BLF 文件未包含任何 CAN 报文")
    return counts


class _MessageBucket:
    """单个 ``(通道, CAN ID)`` 报文桶的块式列存储。

    刻意不用 ``list[float]`` + ``list[bytes]`` 常驻：实测 Python float 约 24 B、
    bytes 对象约 33 B + 数据，500 万帧会白吃 300 MB 以上。改为每攒满
    ``_BUCKET_BLOCK_ROWS`` 条就压成 numpy 块，最终再合并成连续数组。
    """

    __slots__ = (
        "channel", "can_id", "extended", "stride", "count", "t_min", "t_max",
        "frame", "_t_blocks", "_p_blocks", "_l_blocks", "_pending_t",
        "_pending_p", "_pending_l", "_payload", "_lengths", "_stored_bytes",
    )

    def __init__(self, channel: int, can_id: int, extended: bool):
        self.channel = channel
        self.can_id = can_id
        self.extended = extended
        self.stride = 0
        self.count = 0
        self.t_min = float("inf")
        self.t_max = float("-inf")
        self.frame: Optional[CanFrameSpec] = None
        self._t_blocks: list[np.ndarray] = []
        self._p_blocks: list[np.ndarray] = []
        # 每行**实际**有效字节数。payload 仍按桶内最大 stride 对齐存储（列式
        # 数组不能变宽），但短行尾部是补出来的 0，不能当真实数据解码。
        # CAN FD 上限 64 B，uint8 足够；30M 行也只需 30 MB。
        self._l_blocks: list[np.ndarray] = []
        self._pending_t: list[float] = []
        self._pending_p: list[bytes] = []
        self._pending_l: list[int] = []
        self._payload: Optional[np.ndarray] = None
        self._lengths: Optional[np.ndarray] = None
        # 已占用的存储字节数（payload 块 + 行长度数组，含按 stride 补出来的
        # 那部分）。扫描期用它兑现 ``BLF_MAX_PAYLOAD_BYTES``；pending 阶段先
        # 按实际长度记，压块时补上对齐差额。
        self._stored_bytes = 0

    def add(self, timestamp: float, payload: bytes) -> int:
        """追加一帧，返回本次调用造成的存储字节增量（供全局预算统计）。"""
        before = self._stored_bytes
        n = len(payload)
        if n > self.stride:
            self.stride = n
        self.count += 1
        if timestamp < self.t_min:
            self.t_min = timestamp
        if timestamp > self.t_max:
            self.t_max = timestamp
        self._pending_t.append(timestamp)
        self._pending_p.append(payload)
        self._pending_l.append(min(n, 255))
        self._stored_bytes += n
        if len(self._pending_t) >= _BUCKET_BLOCK_ROWS:
            self._flush()
        return self._stored_bytes - before

    def pending_rows(self) -> int:
        return len(self._pending_t)

    def flush(self):
        """把 pending 压成 numpy 块（全局 pending 预算触发时调用）。"""
        self._flush()

    def _flush(self):
        if not self._pending_t:
            return
        n = len(self._pending_t)
        stride = self.stride
        buf = bytearray(n * stride)
        off = 0
        data_bytes = 0
        for row in self._pending_p:
            buf[off:off + len(row)] = row
            off += stride
            data_bytes += len(row)
        self._t_blocks.append(np.asarray(self._pending_t, dtype=np.float64))
        self._p_blocks.append(
            np.frombuffer(bytes(buf), dtype=np.uint8).reshape(n, stride)
        )
        self._l_blocks.append(np.asarray(self._pending_l, dtype=np.uint8))
        # pending 按实际长度记的账，压块后变为 n×stride（含补位）+ n（长度）
        self._stored_bytes += n * stride + n - data_bytes
        self._pending_t = []
        self._pending_p = []
        self._pending_l = []

    def finalize(self, should_cancel: Optional[Callable[[], bool]] = None):
        """收尾并把分块合并成单条连续 payload 数组 + 行级有效长度。

        Args:
            should_cancel: 轮询式取消谓词。合并是逐块拷贝，块数可达数百，
                每块之间检查一次，否则大文件拖到 90% 再点取消没有效果。
        """
        self._flush()
        if not self._p_blocks:
            self._payload = np.zeros((0, 0), dtype=np.uint8)
            self._lengths = np.zeros(0, dtype=np.uint8)
            return
        if len(self._p_blocks) == 1:
            self._payload = self._p_blocks[0]
        else:
            merged = np.zeros((self.count, self.stride), dtype=np.uint8)
            row = 0
            for block in self._p_blocks:
                if should_cancel is not None and should_cancel():
                    raise InterruptedError("用户取消了 BLF 扫描")
                # 各块的列数可以不同：``stride`` 是「到目前为止的最大 dlc」，
                # 所以同一个桶里「前几帧短、后几帧长」（实测 CAN FD 与
                # issue_1905 那一类 valid_bytes 偏小的帧）会留下窄历史块。
                # 直接 ``merged[a:b] = block`` 会
                # ``ValueError: could not broadcast (n, 6) into (n, 8)``；窄列
                # 尾部留 0，但真实行长度记录在 ``_lengths`` 里，解码时会把
                # 该行相关信号置 NaN，不会把补出来的 0 当数据。
                width = min(block.shape[1], self.stride)
                merged[row:row + block.shape[0], :width] = block[:, :width]
                row += block.shape[0]
            self._payload = merged
            self._p_blocks = []
        self._lengths = np.concatenate(self._l_blocks) if self._l_blocks else np.zeros(
            0, dtype=np.uint8
        )

    @property
    def timestamps(self) -> np.ndarray:
        blocks = self._t_blocks
        if len(blocks) == 1:
            return blocks[0]
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float64)

    @property
    def payload(self) -> np.ndarray:
        return self._payload

    @property
    def lengths(self) -> Optional[np.ndarray]:
        """每行实际有效字节数；``finalize()`` 之前为 None。"""
        return self._lengths

    @property
    def mean_cycle_s(self) -> Optional[float]:
        if self.count < 2:
            return None
        span = self.t_max - self.t_min
        return span / (self.count - 1) if span > 0 else None


class BLFLazyLoader:
    """BLF + CAN 数据库的惰性数据源（与 MDFLazyLoader 同名同义的契约）。"""

    LOADER_TYPE = "blf"
    # 组语义的说明性标记：BLF 的组 = 一条报文（同通道同 ID），非 MDF 的 channel group
    GROUP_MODE = "can_message"

    MAX_CACHE_SIZE = 256
    # 信号缓存字节预算：单条就是整条信号的 float64 数组。BLF 允许 30M 帧，
    # 单个高频 ID 可达百万级（8 MB/条），只限 256 条时理论峰值是 GB 级。
    MAX_CACHE_BYTES = 128 * 1024 * 1024
    MAX_TIME_CACHE_SIZE = 64
    # 时间戳缓存同样按字节压一道上限（每条约 8 B/帧）。
    MAX_TIME_CACHE_BYTES = 64 * 1024 * 1024

    _CAN_IMPORT_ERROR = "python-can 库未安装。请运行: uv add python-can>=4.4"

    def __init__(
        self,
        path: str,
        *,
        db_paths: Optional[list[str]] = None,
        _progress: Optional[Callable[[int], None]] = None,
        _should_cancel: Optional[Callable[[], bool]] = None,
        _stats: Optional[dict] = None,
    ):
        self._path = path
        self._progress = _progress
        self._should_cancel = _should_cancel
        # 扫描过程中的实时计数（单键 dict 赋值受 GIL 保护）。进度对话框需要
        # 「已读 N 帧」这种 loader 内部状态，而 loader 实例本身在扫描完成前
        # 无法从 UI 拿到（它还活在后台线程里），故用一个共享 dict 透出。
        self._stats = _stats if _stats is not None else {}

        # 并发保护与关闭标志必须先于任何可能抛异常的步骤创建：__del__ 会调用
        # close()，而 _validate_file() / import can / 扫描均可能抛异常。
        self._access_lock = threading.RLock()
        self._closed = False

        self._signal_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._time_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._signal_cache_bytes = 0
        self._time_cache_bytes = 0
        self._metadata: list[VarMetadata] = []
        self._var_to_meta: dict[str, VarMetadata] = {}
        self._var_to_sig: dict[str, CanSignalSpec] = {}
        self._buckets: list[_MessageBucket] = []
        self._raw_groups: dict[int, list[VarMetadata]] = {}
        self._group_labels: dict[int, str] = {}
        self._current_group_index = 0

        self.db_paths: list[str] = list(db_paths or [])
        self.total_messages = 0
        #: 扫描期 payload 实际占用的字节数（含对齐补位与行长度数组）
        self.payload_bytes = 0
        self.truncated = False
        #: ``"messages"`` / ``"payload"``；未截断时为 None
        self.truncated_reason: Optional[str] = None
        self.unmatched_ids: list[tuple[int, int, bool, int]] = []
        self.load_errors: list[str] = []
        self._t0 = 0.0
        self._database = CanDatabaseSet()

        self._validate_file()
        self._scan()
        self._bind_database(self.db_paths)

        logger.info(
            "BLF 加载完成: %s (%.1f MB) · %d 帧 / %d 报文桶 · %d 信号 · %d 未匹配 ID",
            path, self._file_size / 1024 / 1024, self.total_messages,
            len(self._buckets), len(self._metadata), len(self.unmatched_ids),
        )

    # ------------------------------------------------------------------
    # 构造期辅助
    # ------------------------------------------------------------------

    def __del__(self):
        self.close()

    def _notify_progress(self, value: int):
        if self._progress:
            try:
                self._progress(value)
            except Exception:
                logger.debug("进度通知回调失败", exc_info=True)

    def _validate_file(self):
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"BLF 文件不存在: {self._path}")
        file_size = os.path.getsize(self._path)
        if file_size == 0:
            raise ValueError("BLF 文件为空")
        self._file_size = file_size

    # ------------------------------------------------------------------
    # 扫描：只读报文，不做任何信号解码
    # ------------------------------------------------------------------

    def _scan(self):
        # 惰性 import：python-can 的包初始化会扫描硬件后端 entry point，启动期
        # 代价与打包体积都不小；CSV/MDF 用户完全不该为 BLF 付这笔钱。
        try:
            from can import BLFReader
        except ImportError as e:
            raise ImportError(f"{self._CAN_IMPORT_ERROR}\n原始错误: {e}") from e

        buckets: dict[tuple[int, int, bool], _MessageBucket] = {}
        total = 0
        payload_bytes = 0
        pending_rows = 0
        t_min = float("inf")
        t_max = float("-inf")

        # 自持二进制句柄：BLFReader 对 str 路径会自己 open，我们就拿不到
        # tell() 来做可量化的进度百分比（扫描可达分钟级，摆动条不可接受）。
        handle = open(self._path, "rb")
        try:
            reader = BLFReader(handle)
            iterator = iter(reader)
            while True:
                chunk = list(itertools.islice(iterator, BLF_SCAN_CHUNK_MESSAGES))
                if not chunk:
                    break
                for msg in chunk:
                    if msg.is_error_frame or msg.is_remote_frame:
                        # 错误帧没有可信的数据区，远程帧只有 ID 没有 payload，
                        # 两者都不该产生采样点（画出来只会是误导性的台阶）。
                        continue
                    aid = int(msg.arbitration_id)
                    extended = bool(msg.is_extended_id)
                    # python-can 把 BLF 的 1 基通道号减 1，老文件里的 0 会变成 -1
                    channel = int(msg.channel) if msg.channel is not None else 0
                    if channel < 0:
                        channel = 0
                    key = (channel, aid, extended)
                    bucket = buckets.get(key)
                    if bucket is None:
                        bucket = buckets[key] = _MessageBucket(channel, aid, extended)
                    payload = bytes(msg.data) if msg.data else b""
                    ts = float(msg.timestamp)
                    before_pending = bucket.pending_rows()
                    payload_bytes += bucket.add(ts, payload)
                    pending_rows += bucket.pending_rows() - before_pending
                    if ts < t_min:
                        t_min = ts
                    if ts > t_max:
                        t_max = ts
                    total += 1

                # 全局 pending 预算：单桶 8192 在 ID 基数大的日志里等于没上限
                # （4000 个 ID × 7500 帧会全部以 Python 对象驻留）。超预算就
                # 把所有桶一起压块，代价是块数变多，换来 pending 有确定上界。
                if pending_rows >= BLF_PENDING_ROWS_BUDGET:
                    for b in buckets.values():
                        b.flush()
                    pending_rows = 0
                    # 压块时补上了对齐差额，重算一次让字节预算的账保持真实
                    payload_bytes = sum(b._stored_bytes for b in buckets.values())

                self._stats["messages"] = total
                self._stats["buckets"] = len(buckets)
                self._notify_progress(
                    10 + min(80, int(_safe_tell(handle, self._file_size) / max(1, self._file_size) * 80))
                )
                if self._should_cancel is not None and self._should_cancel():
                    raise InterruptedError("用户取消了 BLF 扫描")
                # 让出 GIL：后台扫描线程在纯 Python 循环里会挤占 UI 线程，
                # sleep(0) 是当前唯一在 Windows 上也有效的让步手段。
                time.sleep(0)

                if total >= BLF_MAX_MESSAGES:
                    self.truncated = True
                    self.truncated_reason = "messages"
                    logger.warning(
                        "BLF 报文数超过上限 %s，已停止扫描（其余报文不生成信号）",
                        f"{BLF_MAX_MESSAGES:,}",
                    )
                    break
                if payload_bytes >= BLF_MAX_PAYLOAD_BYTES:
                    self.truncated = True
                    self.truncated_reason = "payload"
                    logger.warning(
                        "BLF 报文数据超过上限 %s MB，已停止扫描（其余报文不生成信号）",
                        f"{BLF_MAX_PAYLOAD_BYTES // (1024 * 1024)}",
                    )
                    break
        finally:
            try:
                handle.close()
            except Exception:
                logger.debug("关闭 BLF 句柄失败", exc_info=True)

        if total == 0:
            raise ValueError("BLF 文件未包含任何 CAN 报文")

        ordered = sorted(buckets.values(), key=lambda b: (b.channel, b.extended, b.can_id))
        for bucket in ordered:
            # 合并是逐块拷贝，一个百万帧文件可能在 finalize 里再花数秒。
            # 每个桶前后都查一次取消，避免「进度到 90% 后取消没反应」。
            if self._should_cancel is not None and self._should_cancel():
                raise InterruptedError("用户取消了 BLF 扫描")
            bucket.finalize(self._should_cancel)
        self._buckets = ordered
        self.total_messages = total
        # 以最终数组为准重算：合并时窄历史块会被补到 stride，扫描期的
        # 增量记账到不了这部分补位。
        self.payload_bytes = sum(
            b.payload.nbytes + b.lengths.nbytes for b in ordered
        )
        self._t0 = t_min if np.isfinite(t_min) else 0.0
        self._scan_span = (t_min, t_max) if np.isfinite(t_min) else (0.0, 1.0)
        self._notify_progress(90)

    # ------------------------------------------------------------------
    # 数据库绑定与元数据构建
    # ------------------------------------------------------------------

    def bind_databases(self, paths: list[str], *,
                       rollback_if_empty: bool = False) -> dict:
        """重新绑定 DBC / ARXML：**不重扫报文**，只重建元数据并清缓存。

        ``CanDatabaseSet(paths)`` 会在调用线程**同步解析**全部数据库，耗时取
        决于库的数量/体积（几十 MB 的 ARXML 可达数百毫秒到数秒），并非
        「毫秒级」；调用方必须自行给出忙碌反馈。省掉的只是整份 BLF 的重扫
        —— 原始 payload 与数据库解耦，换库只改变「怎么解释这些字节」。

        Args:
            rollback_if_empty: 新库与当前报文完全对不上（``var_count == 0``）
                时恢复旧库、旧元数据，并在返回值里带 ``rolled_back=True``。
                供 UI 的「重选数据库」使用：一次误选不能先把已画曲线清掉再
                告诉用户「没有信号」。默认 False 保持既有调用语义。
        """
        # ``db_paths`` 与元数据是同一份状态：写它必须跟重建共用一把锁，否则
        # reload 路径（读 loader.db_paths 直接复用）会读到半新半旧的组合。
        with self._access_lock:
            old_paths = list(self.db_paths)
            old_database = self._database
            old_errors = list(self.load_errors)
            self.db_paths = list(paths or [])
            try:
                stats = self._bind_database(self.db_paths)
            except Exception:
                # 解析/建元数据中途失败：恢复到调用前的可用状态，再把异常抛回
                self.db_paths = old_paths
                self._database = old_database
                self.load_errors = old_errors
                self._build_metadata()
                raise
            if rollback_if_empty and not stats.get("var_count"):
                self.db_paths = old_paths
                self._database = old_database
                self.load_errors = old_errors
                self._build_metadata()
                return {
                    "var_count": len(self._metadata),
                    "unmatched": len(self.unmatched_ids),
                    "frame_count": old_database.frame_count,
                    "signal_count": old_database.signal_count,
                    "rolled_back": True,
                }
            return stats

    def _bind_database(self, paths: list[str]) -> dict:
        self._database = CanDatabaseSet(paths)
        self.load_errors = list(self._database.load_errors)
        # 90→99 是「正在生成信号列表…」的区间（§5.4），必须排在 _build_metadata()
        # 末尾那记 100 之前；顺序写反会让进度条从 100 退回 99。
        self._notify_progress(99)
        self._build_metadata()
        self._clear_lru_caches()
        return {
            "var_count": len(self._metadata),
            "unmatched": len(self.unmatched_ids),
            "frame_count": self._database.frame_count,
            "signal_count": self._database.signal_count,
        }

    def _build_metadata(self):
        """报文桶 ∩ 数据库 → VarMetadata（组 = 一个桶）。"""
        self._metadata = []
        self._var_to_meta = {}
        self._var_to_sig = {}
        self._raw_groups = {}
        self._group_labels = {}
        unmatched: list[tuple[int, int, bool, int]] = []

        candidates: list[tuple[int, _MessageBucket, CanSignalSpec, str]] = []
        for gi, bucket in enumerate(self._buckets):
            frame = self._database.find_frame(bucket.channel, bucket.can_id,
                                              bucket.extended)
            bucket.frame = frame
            if frame is None:
                # 扩展帧的标志必须一起存：11 位与 29 位 ID 会重叠（一个
                # 0x123 的**标准**帧与 0x123 的**扩展**帧在库里是两条报文），
                # 只给 ID 会让用户按这张表去补 DBC 时找错报文。
                unmatched.append(
                    (bucket.channel, bucket.can_id, bucket.extended, bucket.count)
                )
                continue
            for sig in frame.signals:
                candidates.append((gi, bucket, sig, sig.name))

        # 命名策略（三轮退化）：裸信号名全局唯一时直接用；重名时用「报文名.信号名」
        # 保证跨报文可辨；仍重名（同一信号出现在多个通道/多份库的同 ID 报文）时
        # 追加组序号，与 MDFLazyLoader 的 _G{gi} 约定一致。
        bare_counts: dict[str, int] = {}
        for _, _, _, bare in candidates:
            bare_counts[bare] = bare_counts.get(bare, 0) + 1
        qualified_counts: dict[str, int] = {}
        for _, _, sig, bare in candidates:
            if bare_counts[bare] > 1:
                q = f"{sig.frame_name}.{bare}"
                qualified_counts[q] = qualified_counts.get(q, 0) + 1

        group_metas: dict[int, list[VarMetadata]] = {}
        for gi, bucket, sig, bare in candidates:
            # 标称周期优先用数据库声明值：DBC 的周期是**报文**属性
            # （GenMsgCycleTime），canmatrix 不会把它下发到 Signal 上，所以
            # 只看 ``sig.cycle_time_s`` 会永远为空，导致「标称采样间隔」退化成
            # 「实测平均间隔」——两个读数本该能互相揭示偏差。
            declared = sig.cycle_time_s or (
                bucket.frame.cycle_time_s if bucket.frame is not None else None
            )
            display = bare if bare_counts[bare] == 1 else f"{sig.frame_name}.{bare}"
            if qualified_counts.get(display, 0) > 1:
                display = f"{display}_G{gi}"
            if display in self._var_to_meta:
                # 同组内报文名+信号名也撞了（多库合并的同 ID 帧），最后一道兜底
                display = f"{display}_G{gi}"

            meta = VarMetadata(
                name=display,
                unit=sig.unit or "-",
                group_index=gi,
                channel_index=len(group_metas.setdefault(gi, [])),
                time_min=(bucket.t_min - self._t0) if bucket.count else 0.0,
                time_max=(bucket.t_max - self._t0) if bucket.count else 0.0,
                sample_count=bucket.count,
                nominal_raster_s=declared or bucket.mean_cycle_s,
                effective_rate_hz=(
                    1.0 / bucket.mean_cycle_s if bucket.mean_cycle_s else None
                ),
                original_name=bare,
                is_enum=sig.is_enum,
                is_time_channel=False,
                validity=UNKNOWN,
                enum_map=dict(sig.values) if sig.values else None,
            )
            group_metas[gi].append(meta)
            self._metadata.append(meta)
            self._var_to_meta[display] = meta
            self._var_to_sig[display] = sig

        self._raw_groups = group_metas
        for gi, bucket in enumerate(self._buckets):
            frame = bucket.frame
            label_id = f"0x{bucket.can_id:X}" + ("E" if bucket.extended else "")
            name_part = f" {frame.name}" if frame is not None else ""
            self._group_labels[gi] = (
                f"Ch{bucket.channel} {label_id}{name_part} (G{gi})"
            )

        self.unmatched_ids = sorted(unmatched, key=lambda x: -x[3])
        if unmatched:
            logger.info(
                "BLF 中 %d 个报文 ID 未在已选数据库中定义（前 %d 个见详情对话框/日志）",
                len(unmatched), min(len(unmatched), BLF_UNMATCHED_REPORT_LIMIT),
            )
        for ch, cid, ext, cnt in self.unmatched_ids:
            logger.debug("未匹配报文: Ch%d 0x%X%s (%d 帧)",
                         ch, cid, "（扩展帧）" if ext else "", cnt)

        self._compute_global_time_range()
        self._notify_progress(100)

    def _compute_global_time_range(self):
        t_min, t_max = getattr(self, "_scan_span", (0.0, 1.0))
        start = t_min - self._t0
        end = t_max - self._t0
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            end = start + 1.0
        self._cached_global_time_range = (float(start), float(end))
        self._cached_max_samples = max(
            (b.count for b in self._buckets if b.frame is not None), default=0
        ) or max((b.count for b in self._buckets), default=0)

    # ------------------------------------------------------------------
    # LRU 缓存与并发纪律（与 MDFLazyLoader 一致）
    # ------------------------------------------------------------------

    def _cache_get(self, name: str) -> Optional[np.ndarray]:
        if name in self._signal_cache:
            self._signal_cache.move_to_end(name)
            return self._signal_cache[name]
        return None

    def _enforce_cache_budget(
        self,
        cache: OrderedDict,
        byte_total: int,
        *,
        max_entries: int,
        max_bytes: int,
    ) -> int:
        """LRU 逐出到「条数 + 字节」双双达标，返回逐出后的字节总量。

        调用方必须已持有 ``_access_lock``。至少保留一条：单条数组本身就可能
        超过预算，把它也逐出等于刚算出来就丢，下次同一信号还得再解一遍。
        """
        while len(cache) > 1 and (
            len(cache) > max_entries or byte_total > max_bytes
        ):
            _, dropped = cache.popitem(last=False)
            byte_total -= dropped.nbytes
        return byte_total

    def _cache_put(self, name: str, data: np.ndarray):
        if name in self._signal_cache:
            self._signal_cache.move_to_end(name)
        else:
            self._signal_cache[name] = data
            self._signal_cache_bytes += data.nbytes
            self._signal_cache_bytes = self._enforce_cache_budget(
                self._signal_cache,
                self._signal_cache_bytes,
                max_entries=self.MAX_CACHE_SIZE,
                max_bytes=self.MAX_CACHE_BYTES,
            )

    def _cache_put_time(self, group_index: int, timestamps: np.ndarray) -> np.ndarray:
        """写入时间轴 LRU（调用方持锁），返回入缓存的那份数组。"""
        if group_index in self._time_cache:
            self._time_cache.move_to_end(group_index)
            return self._time_cache[group_index]
        self._time_cache[group_index] = timestamps
        self._time_cache_bytes += timestamps.nbytes
        self._time_cache_bytes = self._enforce_cache_budget(
            self._time_cache,
            self._time_cache_bytes,
            max_entries=self.MAX_TIME_CACHE_SIZE,
            max_bytes=self.MAX_TIME_CACHE_BYTES,
        )
        return timestamps

    def _ensure_open(self):
        """调用方必须已持有 self._access_lock。"""
        if self._closed:
            raise KeyError("BLF 数据源已关闭")

    def _clear_lru_caches(self):
        """清空两份 LRU 并归零字节账（调用方持锁）。"""
        self._signal_cache.clear()
        self._time_cache.clear()
        self._signal_cache_bytes = 0
        self._time_cache_bytes = 0

    def clear_cache(self):
        with self._access_lock:
            self._clear_lru_caches()

    def release_memory(self):
        """清空解码缓存（原始 payload 仍需保留，故不动桶数据）。"""
        self.clear_cache()

    def close(self):
        lock = getattr(self, "_access_lock", None)
        if lock is None:
            return
        with lock:
            self._closed = True
            for attr in ("_signal_cache", "_time_cache", "_metadata",
                         "_var_to_meta", "_var_to_sig", "_raw_groups",
                         "_group_labels"):
                container = getattr(self, attr, None)
                if container is not None:
                    container.clear()
            self._signal_cache_bytes = 0
            self._time_cache_bytes = 0
            self._buckets = []
            # canmatrix 的整棵 Frame/Signal 对象图跟着数据库走：loader 关闭后
            # 仍可能被「最近文件」/pending 列表引用，留着等于继续常驻一份 DBC。
            # 置空集而非 None，``database`` 属性对已关闭 loader 仍可安全读取。
            self._database = CanDatabaseSet()
            self._cached_max_samples = 0
            self._cached_global_time_range = (0.0, 1.0)

    # ------------------------------------------------------------------
    # 解码
    # ------------------------------------------------------------------

    def _decode(self, display_name: str) -> tuple[np.ndarray, VarMetadata]:
        meta = self._var_to_meta.get(display_name)
        if meta is None:
            raise KeyError(f"变量 '{display_name}' 不存在")
        cached = self._cache_get(display_name)
        if cached is not None:
            return cached, meta

        bucket = self._buckets[meta.group_index]
        sig = self._var_to_sig[display_name]
        frame = bucket.frame
        payload = bucket.payload
        lengths = bucket.lengths
        # 行级有效性是 mux 掩码之外的独立维度：短行的补 0 既不能参与物理值，
        # 也不能让选择器被误当成组 0。
        row_ok = row_validity(lengths, sig)
        mask = mux_validity(payload, frame, sig, lengths) if frame is not None else None
        if mask is None:
            mask = row_ok
        elif row_ok is not None:
            mask = mask & row_ok
        try:
            if frame is not None and not frame.is_vectorizable:
                raise ValueError("该报文布局不适用向量化解码")
            values = decode_physical(payload, sig, mask, frame)
        except ValueError:
            # 位宽 > 64 bit（CAN FD 大信号）或布局特殊（PDU container / 复杂
            # 多路复用）：回落逐帧解码。这类信号在真实工程里极少，但画不出来
            # 比画得慢更糟。
            logger.debug("信号 %s 回落逐帧解码", display_name, exc_info=True)
            values = decode_rows(payload, sig, frame, mask, lengths)

        if sig.is_enum:
            # 与 MDF 一致：枚举变量绘图走**原始码值**，文本标签交给 enum_map，
            # 避免字符串数组进入 pyqtgraph 的绘图路径。直接取 raw 而不做
            # “物理值回乘”的逆变换：factor 为小数时除法再四舍五入会漂一个码值。
            try:
                values = extract_raw(payload, sig)
                if mask is not None:
                    valid = np.asarray(mask, dtype=bool)
                    if valid.shape[0] == values.shape[0]:
                        values = values.astype(np.float64).copy()
                        values[~valid] = np.nan
                    else:
                        # 静默忽略掩码 = 保留补 0 解出的残留值，正是 mux / 短行
                        # 最该避免的失败模式；长度不一致必须留下证据（评审 L7）。
                        logger.warning(
                            "信号 %s 的有效性掩码长度 %d 与采样点数 %d 不一致，"
                            "已忽略掩码",
                            display_name, valid.shape[0], values.shape[0],
                        )
            except ValueError:
                logger.debug("枚举码值提取失败，回落物理值: %s", display_name,
                             exc_info=True)

        self._cache_put(display_name, values)
        return values, meta

    def get_series(self, display_name: str) -> pd.Series:
        with self._access_lock:
            self._ensure_open()
            values, meta = self._decode(display_name)
        return pd.Series(values, name=display_name)

    def get_value_from_name(self, display_name: str):
        with self._access_lock:
            self._ensure_open()
            values, meta = self._decode(display_name)
            x = self._group_time_array(meta.group_index)
            enum_map = meta.enum_map if meta.is_enum else None
            unit = meta.unit
        return x, values, unit, enum_map or {}

    # ------------------------------------------------------------------
    # 组级访问
    # ------------------------------------------------------------------

    def _group_time_array(self, group_index: int) -> np.ndarray:
        """相对秒时间轴。缓存的是**已归零**的数组，避免每个消费方各减一次。"""
        if group_index in self._time_cache:
            self._time_cache.move_to_end(group_index)
            return self._time_cache[group_index]
        if group_index >= len(self._buckets):
            return np.zeros(0, dtype=np.float64)
        bucket = self._buckets[group_index]
        abs_ts = bucket.timestamps
        rel = (abs_ts - self._t0) if abs_ts.size else abs_ts
        return self._cache_put_time(group_index, rel)

    def get_var_group_index(self, display_name: str) -> int:
        with self._access_lock:
            self._ensure_open()
            meta = self._var_to_meta.get(display_name)
            if meta is None:
                raise KeyError(f"变量 '{display_name}' 不存在")
            return meta.group_index

    def get_group_time_array(self, group_index: int) -> np.ndarray:
        with self._access_lock:
            self._ensure_open()
            return self._group_time_array(group_index)

    def get_group_label(self, group_index: int) -> str:
        with self._access_lock:
            self._ensure_open()
            return self._group_labels.get(group_index, f"G{group_index}")

    def get_group_variables(self, group_index: int) -> list[str]:
        with self._access_lock:
            self._ensure_open()
            return [m.name for m in self._metadata if m.group_index == group_index]

    def search_variables(self, keyword: str, limit: int = 50) -> list[tuple[str, int, str]]:
        keyword_lower = keyword.lower()
        results: list[tuple[str, int, str]] = []
        with self._access_lock:
            self._ensure_open()
            label_cache: dict[int, str] = {}
            for meta in self._metadata:
                if keyword_lower in meta.name.lower():
                    gi = meta.group_index
                    if gi not in label_cache:
                        label_cache[gi] = self._group_labels.get(gi, f"G{gi}")
                    results.append((meta.name, gi, label_cache[gi]))
                    if len(results) >= limit:
                        break
        return results

    # ------------------------------------------------------------------
    # 元数据访问（供变量信息窗口）
    # ------------------------------------------------------------------

    def get_metadata(self, display_name: str) -> Optional[VarMetadata]:
        with self._access_lock:
            self._ensure_open()
            return self._var_to_meta.get(display_name)

    def get_channel_info(self, display_name: str) -> dict:
        """返回与 MDFLazyLoader 同构的 dict，另加一个 ``can`` 子 dict。

        ``channel`` / ``channel_group`` 两组键刻意保持 MDF 侧已有的名字
        （bit_count / lower_limit / cycles_nr / comment…），使
        ``var_info._from_blf`` 能复用 ``_from_mdf`` 的骨架而不必重造轮子。
        """
        with self._access_lock:
            self._ensure_open()
            meta = self._var_to_meta.get(display_name)
            if meta is None:
                raise KeyError(f"变量 '{display_name}' 不存在")
            sig = self._var_to_sig[display_name]
            bucket = self._buckets[meta.group_index]
            frame = bucket.frame

            can_dict = {
                "message_name": frame.name if frame else "",
                "can_id_hex": f"0x{bucket.can_id:X}"
                + ("（扩展帧）" if bucket.extended else ""),
                "is_extended": bucket.extended,
                "dlc": frame.dlc if frame is not None else (bucket.stride or 0),
                # DBC 声明的是「最多多少字节」，桶的 stride 是「日志里实际观察到
                # 的最大 payload」。CAN FD 的 valid_bytes < DLC 时两者不同，把
                # 观察值报成 DLC 会掩盖高字节全 0 的真实原因（评审 L1）。
                "observed_max_bytes": bucket.stride,
                "bus_channel": bucket.channel,
                "start_bit": sig.dbc_start_bit,
                # 解码器真正用的 canmatrix 内部位号（Intel 与 DBC 相同，
                # Motorola 不同），放在副键里供排错时比对
                "start_bit_internal": sig.start_bit,
                "bit_size": sig.size,
                "byte_order": "Intel（小端）" if sig.little_endian else "Motorola（大端）",
                "is_signed": sig.signed,
                "is_float": sig.is_float,
                "factor": sig.factor,
                "offset": sig.offset,
                "cycle_time_s": (
                    bucket.mean_cycle_s or sig.cycle_time_s
                    or (frame.cycle_time_s if frame is not None else None)
                ),
                # 单帧桶（偶发/触发型报文）算不出均值，上面的读数会退成数据库
                # 声明值。不带这个标志，「周期（实测均值）」就会在声明值上撒谎。
                "cycle_time_measured": bool(bucket.mean_cycle_s),
                "receivers": ", ".join(sig.receivers),
                "db_file": sig.db_path,
                "mux_text": self._mux_text(frame, sig),
            }

            return {
                "meta": meta,
                "version": "BLF",
                "dtype": "float64",
                "is_numeric": True,
                "channel": {
                    "name": meta.original_name,
                    "unit": meta.unit,
                    "bit_count": sig.size,
                    "lower_limit": sig.minimum,
                    "upper_limit": sig.maximum,
                    "comment": sig.comment,
                },
                "channel_group": {
                    "cycles_nr": bucket.count,
                    "samples_byte_nr": bucket.stride,
                    "acq_name": frame.name if frame else "",
                    "comment": (frame.comment if frame else "") or "",
                    "channel_count": len(self._raw_groups.get(meta.group_index, [])),
                },
                "conversion": {
                    "conversion_type": None,
                    "unit": meta.unit,
                    "a": sig.factor,
                    "b": sig.offset,
                    "formula": "",
                    "ref_param_nr": len(sig.values) if sig.values else 0,
                    "comment": "",
                },
                "header": {
                    "author": "",
                    "department": "",
                    "project": "",
                    "subject": "",
                    "start_time_string": self._start_time_string(),
                    "comment": "",
                },
                "time_base": {
                    "master_name": "CAN 报文时间戳",
                    "nominal_raster_s": meta.nominal_raster_s,
                    "effective_rate_hz": meta.effective_rate_hz,
                    "time_min": meta.time_min,
                    "time_max": meta.time_max,
                    "sample_count": meta.sample_count,
                },
                "file": {
                    "path": self._path,
                    "size": self._file_size,
                    "group_count": len(self._raw_groups),
                    "var_count": len(self._metadata),
                },
                "enum_map": meta.enum_map,
                "can": can_dict,
            }

    @staticmethod
    def _mux_text(frame: Optional[CanFrameSpec], sig: CanSignalSpec) -> str:
        if frame is None or not frame.is_multiplexed:
            return ""
        if sig.is_mux_selector:
            return f"多路复用选择器（{frame.mux_signal}）"
        if sig.mux is None:
            return "公共信号（不依赖 mux 值）"
        return f"仅在 {frame.mux_signal} == {sig.mux} 时有效"

    def _start_time_string(self) -> str:
        """BLF 的绝对起始时间（python-can 给的是 unix 秒）→ 可读字符串。"""
        t0 = getattr(self, "_t0", 0.0)
        if not t0:
            return ""
        try:
            return datetime.fromtimestamp(t0, tz=timezone.utc).astimezone().strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )
        except (OverflowError, OSError, ValueError):
            logger.debug("BLF 起始时间转换失败 t0=%r", t0, exc_info=True)
            return ""

    def get_samples_chunked(self, display_name: str, offset: int,
                            count: int) -> np.ndarray:
        """分块读取物理值，供后台统计流式累加。

        与 :meth:`get_series` 的两点差异（与 MDF 侧口径一致）：
        1. 枚举信号在此**不做码值化**，统计必须基于物理值；
        2. 不写 ``_signal_cache``，避免整条长信号挤占绘图的 LRU 空间。
        """
        with self._access_lock:
            self._ensure_open()
            meta = self._var_to_meta.get(display_name)
            if meta is None:
                raise KeyError(f"变量 '{display_name}' 不存在")
            bucket = self._buckets[meta.group_index]
            sig = self._var_to_sig[display_name]
            frame = bucket.frame
            payload = bucket.payload
            total = payload.shape[0]
            if count is None or count <= 0:
                count = total
            stop = min(total, offset + count)
            if offset >= total or stop <= offset:
                return np.zeros(0, dtype=np.float64)
            window = payload[offset:stop]
            window_lengths = (
                bucket.lengths[offset:stop] if bucket.lengths is not None else None
            )
            row_ok = row_validity(window_lengths, sig)
            mask = (
                mux_validity(window, frame, sig, window_lengths)
                if frame is not None else None
            )
            if mask is None:
                mask = row_ok
            elif row_ok is not None:
                mask = mask & row_ok
            try:
                if frame is not None and not frame.is_vectorizable:
                    raise ValueError("该报文布局不适用向量化解码")
                return decode_physical(window, sig, mask, frame)
            except ValueError:
                logger.debug("信号 %s 统计路径回落逐帧解码", display_name,
                             exc_info=True)
                return decode_rows(window, sig, frame, mask, window_lengths)

    # ------------------------------------------------------------------
    # 属性（与 FastDataLoader / MDFLazyLoader 对齐）
    # ------------------------------------------------------------------

    @property
    def var_names(self) -> list[str]:
        with self._access_lock:
            return [m.name for m in self._metadata]

    @property
    def units(self) -> dict[str, str]:
        with self._access_lock:
            return {m.name: m.unit for m in self._metadata}

    @property
    def df(self):
        return None

    @property
    def df_validity(self) -> dict[str, int]:
        with self._access_lock:
            return {m.name: UNKNOWN for m in self._metadata}

    @property
    def datalength(self) -> int:
        return getattr(self, "_cached_max_samples", 0)

    @property
    def max_row_count(self) -> int:
        return self.datalength

    @property
    def row_count(self) -> int:
        return self.datalength

    @property
    def column_count(self) -> int:
        with self._access_lock:
            return len(self._metadata)

    @property
    def global_time_range(self) -> tuple[float, float]:
        return getattr(self, "_cached_global_time_range", (0.0, 1.0))

    @property
    def baseline_density(self) -> float:
        t_min, t_max = self.global_time_range
        span = t_max - t_min
        if span > 0:
            return float(self.datalength) / span
        return 0.0

    @property
    def time_column_name(self) -> Optional[str]:
        return "time"

    @property
    def time_axis_label(self) -> str:
        return "time (s)"

    @property
    def path(self) -> str:
        return self._path

    @property
    def file_size(self) -> int:
        return self._file_size

    @property
    def time_values(self):
        gi = self._current_group_index
        with self._access_lock:
            if self._closed:
                # 关闭后不再伪造 ``[0]`` 索引：x 与窗口里缓存的 y 长度、语义
                # 都对不上，空序列至少能让调用方/绘图发现「没有时间轴」。
                return pd.Series(dtype=np.float64, name="time")
            if gi < len(self._buckets):
                arr = self._group_time_array(gi)
                if arr.size:
                    return pd.Series(arr, name="time")
        return pd.Series(np.arange(max(1, self.datalength)), name="index")

    @property
    def time_channels_info(self) -> dict[str, str]:
        return {}

    @property
    def groups(self) -> list:
        with self._access_lock:
            result = []
            for gi in sorted(self._raw_groups.keys()):
                metas = self._raw_groups.get(gi, [])
                if not metas:
                    continue
                result.append({"index": gi, "var_names": [m.name for m in metas]})
            return result

    @property
    def group_count(self) -> int:
        # 与 MDFLazyLoader 对齐：只数**生成了变量**的组。未匹配的报文桶仍然占
        # 一个 gi（label/索引稳定），但不能出现在 range(group_count) 里，
        # 否则 UI 会画出永远为空的组 tab。
        with self._access_lock:
            return len(self._raw_groups)

    @property
    def current_group_index(self) -> int:
        return self._current_group_index

    @current_group_index.setter
    def current_group_index(self, value: int):
        self._current_group_index = int(value)

    @property
    def database(self) -> CanDatabaseSet:
        return self._database

    @property
    def message_count(self) -> int:
        return self.total_messages

    def unmatched_report(self, limit: int = BLF_UNMATCHED_REPORT_LIMIT) -> list:
        """未匹配 ID 的前 ``limit`` 行（按帧数降序），供详情对话框展示。"""
        return self.unmatched_ids[:limit]

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return (
            f"<BLFLazyLoader {os.path.basename(self._path)} "
            f"msgs={self.total_messages} vars={len(self._metadata)} "
            f"unmatched={len(self.unmatched_ids)}>"
        )
