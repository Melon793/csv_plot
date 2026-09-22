"""惰性加载访问成本护栏（P7，marker perf，默认集排除）。

目的不是复现 P4 DoD 的真实大文件预算（12 列 ≤10ms 等，那需要 555MB 级
文件，只能人工实测，见 tmp/ 报告与 §6.3 冒烟清单），而是在 CI 可承受的
~13MB 合成夹具上钉住**量级**：
- 转换吞吐不塌方（row_group/压缩参数被改坏时最先红）
- 12 列冷读不出现数量级回归（如 row_group_size 被调回 1024 → 慢 4.7×）
- 缓存命中接近零成本（ColumnCache 失效时红）
- 枚举列文本还原不物化爆炸

运行方式（默认 `uv run pytest -q` 不含本目录，见 pyproject addopts）：
    uv run pytest -m perf -q
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from src.data.parquet_converter import convert_to_parquet
from src.data.parquet_lazy_loader import ParquetLazyLoader
from src.data.temp_cache_dir import TempCacheDir
from tests.fixtures.data_factory import make_timeseries

N_ROWS = 100_000
N_COLS = 12
STATES = ("ON", "OFF", "STANDBY")


@pytest.fixture(scope="module")
def lazy_env(tmp_path_factory):
    """~13MB 合成 CSV（12 数值列 + 1 低基数枚举列）→ parquet 惰性 loader。"""
    tmp = tmp_path_factory.mktemp("perf_lazy")
    cols = [f"v{i:02d}" for i in range(1, N_COLS + 1)] + ["state"]
    data = {}
    for i, name in enumerate(cols[:-1], start=1):
        y = make_timeseries(N_ROWS, seed=i)[1]
        # make_timeseries 无随机分量：给每列不同的缩放/偏置，避免 12 列
        # 全同时 zstd 压成近似空文件、护栏失真
        data[name] = y * (1.0 + 0.137 * i) + i
    df = pd.DataFrame(data)
    df["state"] = [STATES[i % 3] for i in range(N_ROWS)]

    csv_path = tmp / "perf_lazy.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        f.write(",".join(["m"] * N_COLS + ["-"]) + "\n")
        df.to_csv(f, index=False, header=False, float_format="%.6f")
    del df

    size_mb = csv_path.stat().st_size / 1024 / 1024
    temp = TempCacheDir.create()
    t0 = time.perf_counter()
    convert_to_parquet(str(csv_path), outdir=temp.path(), has_unit=True, sep=",")
    conv_s = time.perf_counter() - t0
    loader = ParquetLazyLoader(str(csv_path), temp)
    yield loader, cols, size_mb, conv_s
    loader.close()


def test_conversion_throughput_floor(lazy_env):
    """转换吞吐护栏：~13MB 必须在 10s 内完成（本机实测约 0.5s 量级）。

    上限刻意宽松（约 20× 实测值）：护栏针对的是数量级塌方（如 row_group
    改小、压缩改坏、意外走转码重试），不是与 CI 机器比快。
    """
    _, _, size_mb, conv_s = lazy_env
    assert conv_s < 10.0, f"{size_mb:.1f}MB 转换耗时 {conv_s:.2f}s，吞吐塌方"
    assert size_mb > 8.0, f"夹具 {size_mb:.1f}MB 过小，护栏失去意义"


def test_cold_read_12_columns(lazy_env):
    """12 列冷读（UI 首屏形态）：无数量级回归。

    真实 555MB 文件的预算是 ≤10ms（P4 DoD 6，人工实测口径）；本夹具小
    两个数量级，上限 300ms 足以捕获 row_group_size=1024 那类 4.7× 劣化
    叠加 I/O 放大的回归。
    """
    loader, cols, _, _ = lazy_env
    loader.release_memory()  # 清空 ColumnCache，强制冷读
    t0 = time.perf_counter()
    for name in cols[:12]:
        x, y, unit, _ = loader.get_value_from_name(name)
        assert len(x) == len(y) == loader.datalength
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 300.0, f"12 列冷读 {elapsed_ms:.1f}ms，超护栏"


def test_cache_hit_is_near_free(lazy_env):
    """缓存命中平均 ≤0.5ms（验收实测 ~0.001ms；ColumnCache 失效时红）。"""
    loader, cols, _, _ = lazy_env
    loader.get_value_from_name(cols[0])  # 预热
    t0 = time.perf_counter()
    for _ in range(100):
        loader.get_value_from_name(cols[0])
    avg_ms = (time.perf_counter() - t0) * 1000 / 100
    assert avg_ms < 0.5, f"命中平均 {avg_ms:.3f}ms，缓存可能失效"


def test_enum_text_restore_budget(lazy_env):
    """枚举列 get_series 文本还原（10 万行 from_codes）不物化爆炸。"""
    loader, _, _, _ = lazy_env
    loader.release_memory()
    t0 = time.perf_counter()
    s = loader.get_series("state")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert str(s.dtype) == "category"
    assert s.tolist()[:3] == ["ON", "OFF", "STANDBY"]
    assert elapsed_ms < 500.0, f"枚举文本还原 {elapsed_ms:.1f}ms，超护栏"
