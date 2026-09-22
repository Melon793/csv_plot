"""通用列级 LRU 缓存（条数 + 字节双预算）。

新 ParquetLazyLoader（P4）先用；MDF 的缓存迁移留到 P8（届时以实例级
预算传入 256MB，而不是改这里的默认值，见 D7/Q4）。

设计要点：
- 双预算：条数上限（默认 256）+ 字节上限（默认 64MB）任一超限即逐出
  最久未用条目。64MB 的依据：256MB 单独就能击穿 P6 的「常驻 < 150MB」
  验收，两个数字必须自洽；parquet 单列远小于 MDF 单通道（109k 行 × 4B
  ≈ 0.44MB），64MB 约可容纳 145 列，默认 12 子图布局绰绰有余。
- ndarray / pandas Series 用 ``nbytes``；polars Series 没有 ``nbytes``，
  用 ``estimated_size()``。不能退到 ``sys.getsizeof``，否则缓存会严重低估。
- 超预算的代价只是逐出后重读（实测整列 3–5ms），不是错误。
- 单条超过 max_bytes 的值不缓存（缓存它等于立刻逐出全部其他列，
  得不偿失；调用方直接走慢路径重读即可）。
- 线程安全：与 MDF 同模式，RLock 串行化绘图线程与统计 worker。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

# 新 loader 的默认预算（模块常量，刻意不做配置项：多一个开关就多一份
# 需要解释的文档；LAZY_CONVERT_ENABLED / LAZY_CONVERT_MIN_MB 已经够）。
DEFAULT_MAX_ENTRIES = 256
DEFAULT_MAX_BYTES = 64 * 1024 * 1024


def _nbytes(value: Any) -> int:
    """估算缓存值的内存占用（字节）。

    ndarray / pandas Series 用 .nbytes（底层数组字节数，不含索引开销）；
    polars Series 用 estimated_size()，它没有 .nbytes 属性。兜底
    sys.getsizeof 只给小对象用，不能用于数据列。索引开销（RangeIndex 每
    entry 数十字节）相对列数据可忽略，不为此引入 pandas 依赖。
    """
    estimated_size = getattr(value, "estimated_size", None)
    if callable(estimated_size):
        return int(estimated_size())
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int) and nbytes >= 0:
        return nbytes
    import sys

    return sys.getsizeof(value)


class ColumnCache:
    """列级 LRU：get 命中即提升新鲜度，put 超预算逐出最旧条目。"""

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ):
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._lock = threading.RLock()
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._bytes = 0
        # 命中统计（诊断用；不做淘汰策略输入）
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # ---- 查询 ----
    def get(self, key: str) -> Any | None:
        """取列缓存；未命中返回 None（调用方自行走慢路径后 put）。"""
        with self._lock:
            value = self._store.get(key)
            if value is None:
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return value

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._store

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    @property
    def nbytes_total(self) -> int:
        with self._lock:
            return self._bytes

    # ---- 写入 ----
    def put(self, key: str, value: Any) -> None:
        """放入/覆盖一列。同名覆盖时先回收旧值的字节计数。"""
        size = _nbytes(value)
        if size > self._max_bytes:
            # 单条超预算：不缓存。缓存它的唯一效果是逐出其他全部列，
            # 而这条自己迟早也要被逐出——直接让调用方每次重读更可预测。
            return
        with self._lock:
            old = self._store.pop(key, None)
            if old is not None:
                self._bytes -= _nbytes(old)
            self._store[key] = value
            self._bytes += size
            self._evict_locked()

    def _evict_locked(self) -> None:
        """逐出最久未用条目直到双预算都满足（调用方须持锁）。"""
        while (
            self._store
            and (len(self._store) > self._max_entries or self._bytes > self._max_bytes)
        ):
            _, victim = self._store.popitem(last=False)
            self._bytes -= _nbytes(victim)
            self.evictions += 1

    def clear(self) -> None:
        """清零（幂等）。release_memory() 调用。"""
        with self._lock:
            self._store.clear()
            self._bytes = 0
