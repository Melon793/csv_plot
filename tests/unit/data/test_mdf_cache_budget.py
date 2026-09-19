"""MDF LRU 缓存的字节预算（V6.0 P1-17 防回归）。

单条缓存是整通道数组：776 MB 级 .mf4 的单通道可达数十至上百 MB，只限 256 条
时理论峰值是 GB 级。这里绕过 asammdf 句柄，直接对着缓存层做进出账验证。
"""

from collections import OrderedDict
from threading import RLock

import numpy as np

from src.data.mdf_lazy_loader import MDFLazyLoader

MB = 1024 * 1024


def _bare_loader() -> MDFLazyLoader:
    """只搭缓存层所需的最小状态（不打开文件、不构造 asammdf 句柄）"""
    loader = MDFLazyLoader.__new__(MDFLazyLoader)
    loader._access_lock = RLock()
    loader._signal_cache = OrderedDict()
    loader._time_cache = OrderedDict()
    loader._signal_cache_bytes = 0
    loader._time_cache_bytes = 0
    return loader


def _arr(mb: float) -> np.ndarray:
    return np.zeros(int(mb * MB / 8), dtype=np.float64)


class TestSignalCacheByteBudget:
    def test_oldest_evicted_until_under_budget(self, monkeypatch):
        monkeypatch.setattr(MDFLazyLoader, "MAX_CACHE_BYTES", 2 * MB)
        monkeypatch.setattr(MDFLazyLoader, "MAX_CACHE_SIZE", 256)
        loader = _bare_loader()

        for i in range(5):
            loader._cache_put(f"ch{i}", _arr(1))

        assert list(loader._signal_cache) == ["ch3", "ch4"]
        assert loader._signal_cache_bytes == 2 * MB

    def test_oversized_single_entry_survives(self, monkeypatch):
        """刚读出来的通道不能被自己挤掉，否则同一通道反复重读"""
        monkeypatch.setattr(MDFLazyLoader, "MAX_CACHE_BYTES", 1 * MB)
        loader = _bare_loader()

        loader._cache_put("big", _arr(4))

        assert list(loader._signal_cache) == ["big"]
        assert loader._signal_cache_bytes == 4 * MB

        # 但超预算的旧条目会被新条目挤掉，不让单条例外变成常驻泄漏
        loader._cache_put("bigger", _arr(5))

        assert list(loader._signal_cache) == ["bigger"]
        assert loader._signal_cache_bytes == 5 * MB

    def test_counter_matches_content_after_mixed_traffic(self, monkeypatch):
        monkeypatch.setattr(MDFLazyLoader, "MAX_CACHE_BYTES", 6 * MB)
        loader = _bare_loader()

        for i in range(20):
            loader._cache_put(f"ch{i % 7}", _arr(1))
            loader._cache_get("ch3")

        assert loader._signal_cache_bytes == sum(
            a.nbytes for a in loader._signal_cache.values()
        )
        assert loader._signal_cache_bytes <= 6 * MB

    def test_re_put_of_same_name_does_not_double_count(self, monkeypatch):
        monkeypatch.setattr(MDFLazyLoader, "MAX_CACHE_BYTES", 10 * MB)
        loader = _bare_loader()
        arr = _arr(1)

        loader._cache_put("a", arr)
        loader._cache_put("a", arr)

        assert loader._signal_cache_bytes == arr.nbytes

    def test_entry_count_cap_still_applies(self, monkeypatch):
        monkeypatch.setattr(MDFLazyLoader, "MAX_CACHE_BYTES", 100 * MB)
        monkeypatch.setattr(MDFLazyLoader, "MAX_CACHE_SIZE", 3)
        loader = _bare_loader()

        for i in range(10):
            loader._cache_put(f"ch{i}", _arr(1))

        assert len(loader._signal_cache) == 3


class TestTimeCacheByteBudget:
    def test_byte_budget_and_reuse(self, monkeypatch):
        monkeypatch.setattr(MDFLazyLoader, "MAX_TIME_CACHE_BYTES", 2 * MB)
        monkeypatch.setattr(MDFLazyLoader, "MAX_TIME_CACHE_SIZE", 64)
        loader = _bare_loader()

        first = loader._cache_put_time(0, _arr(1))
        again = loader._cache_put_time(0, _arr(1))
        assert again is first
        assert loader._time_cache_bytes == 1 * MB

        loader._cache_put_time(1, _arr(1))
        loader._cache_put_time(2, _arr(1))

        assert list(loader._time_cache) == [1, 2]
        assert loader._time_cache_bytes == 2 * MB


class TestCacheReset:
    def test_release_memory_zeroes_the_ledger(self, monkeypatch):
        monkeypatch.setattr(MDFLazyLoader, "MAX_CACHE_BYTES", 100 * MB)
        monkeypatch.setattr(MDFLazyLoader, "MAX_TIME_CACHE_BYTES", 100 * MB)
        loader = _bare_loader()
        loader._cache_put("a", _arr(1))
        loader._cache_put_time(0, _arr(1))

        loader.release_memory()

        assert loader._signal_cache == {}
        assert loader._time_cache == {}
        assert loader._signal_cache_bytes == 0
        assert loader._time_cache_bytes == 0

    def test_clear_cache_zeroes_the_ledger(self, monkeypatch):
        monkeypatch.setattr(MDFLazyLoader, "MAX_CACHE_BYTES", 100 * MB)
        loader = _bare_loader()
        loader._cache_put("a", _arr(1))

        loader.clear_cache()

        assert loader._signal_cache_bytes == 0
