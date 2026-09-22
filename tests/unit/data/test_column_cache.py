"""ColumnCache（列级 LRU，双预算）单元测试（P2）。"""

from __future__ import annotations

import numpy as np
import pytest

from src.data._column_cache import ColumnCache, _nbytes


def _arr(n: int = 1000) -> np.ndarray:
    return np.zeros(n, dtype=np.float64)  # 1000 * 8 = 8000 bytes


class TestEvictionOrder:
    def test_entry_budget_evicts_lru(self):
        cache = ColumnCache(max_entries=3, max_bytes=10**9)
        for i in range(3):
            cache.put(f"c{i}", _arr())
        # 访问 c0 → c1 变成最旧
        cache.get("c0")
        cache.put("c3", _arr())  # 逐出 c1
        assert "c1" not in cache
        assert cache.get("c0") is not None
        assert cache.get("c2") is not None
        assert cache.get("c3") is not None

    def test_get_promotes_freshness(self):
        cache = ColumnCache(max_entries=2, max_bytes=10**9)
        cache.put("a", _arr())
        cache.put("b", _arr())
        cache.get("a")
        cache.put("c", _arr())
        # b 最旧被逐出；a 因刚访问而保留
        assert "b" not in cache
        assert cache.get("a") is not None

    def test_byte_budget_evicts(self):
        # 每条 8000B，预算 20000B → 最多 2 条 + 第 3 条触发逐出
        cache = ColumnCache(max_entries=100, max_bytes=20_000)
        cache.put("a", _arr())
        cache.put("b", _arr())
        cache.put("c", _arr())  # 24000 > 20000 → 逐出 a
        assert "a" not in cache
        assert len(cache) == 2
        assert cache.nbytes_total <= 20_000

    def test_eviction_counted(self):
        cache = ColumnCache(max_entries=2, max_bytes=10**9)
        for i in range(5):
            cache.put(f"c{i}", _arr())
        assert cache.evictions == 3


class TestOversizedEntry:
    def test_single_entry_over_byte_budget_not_cached(self):
        cache = ColumnCache(max_entries=10, max_bytes=4_000)
        big = _arr(1000)  # 8000B > 4000B
        cache.put("big", big)
        assert len(cache) == 0
        assert cache.get("big") is None

    def test_oversized_put_does_not_evict_existing(self):
        cache = ColumnCache(max_entries=10, max_bytes=8_000)
        cache.put("keep", _arr())
        cache.put("big", _arr(2000))  # 16000B 超预算，不入缓存
        assert cache.get("keep") is not None
        assert cache.evictions == 0


class TestOverwriteAndClear:
    def test_overwrite_updates_byte_accounting(self):
        cache = ColumnCache(max_entries=10, max_bytes=8_000)
        cache.put("k", _arr())            # 8000B
        assert cache.nbytes_total == 8_000
        cache.put("k", np.zeros(500, dtype=np.float64))  # 4000B 覆盖
        assert cache.nbytes_total == 4_000

    def test_clear_is_idempotent_and_resets_bytes(self):
        cache = ColumnCache()
        cache.put("a", _arr())
        cache.clear()
        cache.clear()  # 幂等
        assert len(cache) == 0
        assert cache.nbytes_total == 0
        assert cache.get("a") is None


class TestStatsAndNbytes:
    def test_hit_miss_counters(self):
        cache = ColumnCache()
        cache.put("a", _arr())
        cache.get("a")
        cache.get("a")
        cache.get("missing")
        assert cache.hits == 2
        assert cache.misses == 1

    def test_budgets_instance_level(self):
        # 预算实例级可传（P8 MDF 迁移用 256MB 的依据）
        # small：5 条 × 8000B = 40000B > 32000B → 逐出最旧留 4 条
        small = ColumnCache(max_entries=5, max_bytes=32_000)
        big = ColumnCache(max_entries=5, max_bytes=10**9)
        for i in range(4):
            small.put(f"c{i}", _arr())
            big.put(f"c{i}", _arr())
        small.put("c4", _arr())  # small 逐出；big 不逐出
        big.put("c4", _arr())
        assert len(small) == 4
        assert "c0" not in small  # 逐出的是最旧的 c0
        assert len(big) == 5

    def test_nbytes_helper_variants(self):
        assert _nbytes(_arr(10)) == 80
        assert _nbytes("plain-string") > 0  # 无 nbytes 属性走 getsizeof

    def test_polars_series_nbytes_uses_estimated_size(self):
        """polars Series 没有 .nbytes；必须用 estimated_size 计入字节预算。"""
        import polars as pl

        ps = pl.Series("x", [1.5] * 1000)
        assert _nbytes(ps) == ps.estimated_size()

        cache = ColumnCache(max_entries=10, max_bytes=4_000)
        cache.put("polars", ps)
        assert len(cache) == 0  # 8000B 超预算，不能因 getsizeof 误判为小而缓存


class TestThreadSafety:
    def test_concurrent_put_get_smoke(self):
        import threading

        cache = ColumnCache(max_entries=8, max_bytes=64_000)
        errors: list[Exception] = []

        def worker(seed: int):
            try:
                for i in range(200):
                    k = f"c{(seed + i) % 10}"
                    cache.put(k, _arr(10))
                    cache.get(k)
            except Exception as ex:  # noqa: BLE001
                errors.append(ex)

        threads = [threading.Thread(target=worker, args=(s,)) for s in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert cache.nbytes_total >= 0


if __name__ == "__main__":
    pytest.main([__file__])
