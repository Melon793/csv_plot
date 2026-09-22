"""ParquetLazyLoader - 临时 parquet 列式文件的惰性加载器（P4，§4.4 契约）。

数据不在内存：单列按需从 data.parquet 读取（ColumnCache LRU），元数据
全部来自转换期一次性读入的 meta.json。与 MDFLazyLoader 同模式的线程
安全（RLock + _closed 先置位）。

dtype 还原契约（get_series 与今天的 pandas 内存 loader 逐字相同）：
- float32 / float64 / int64 / bool：polars 同型 to_numpy 直接回 pandas
- CSV 文本列（meta dtype_str="category"，含全空列）：astype("category")
  —— 今天 read_csv(dtype={"...": "category"}) 的口径
- 枚举列（meta is_enum）：pd.Categorical.from_codes(codes, labels) 还原
  **文本**（不是码值！D8 双入口：get_series 给文本，get_value_from_name
  给码值）。polars Int32 含 null 时 to_numpy() 会升 float64——先
  fill_null(-1) 再转，-1 恰是 from_codes 的 NaN 官方编码
- CSV 日期列（dtype_str="object"）/ Excel 文本列（dtype_str="str"）：
  今天分别是 pandas 自动推断的 StringDtype；astype("str") 还原
- Excel 日期列（dtype_str="datetime64[s]" 等）：to_numpy 后
  astype(dtype_str) 还原原分辨率
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.logger import get_logger
from src.data._column_cache import ColumnCache
from src.data.temp_cache_dir import TempCacheDir

logger = get_logger("data.parquet_lazy")

# 转换器（parquet_converter.convert_to_parquet）写死的产物文件名
_PARQUET_NAME = "data.parquet"
_META_NAME = "meta.json"


class ParquetLazyLoader:

    LOADER_TYPE = "parquet"

    # 能力谓词（D5）：数据不在内存，取数走 get_series / get_value_from_name
    IS_LAZY = True

    def __init__(self, src_path: str, cache_dir: TempCacheDir):
        """Args:
        src_path: 源 CSV/Excel 路径（path/file_size/文件信息块都指向它，
            不是临时目录）
        cache_dir: 转换产物所在的 TempCacheDir（含 data.parquet + meta.json）；
            close() 时整目录删除
        """
        self._path = src_path

        # 并发保护与关闭标志必须在任何可能抛异常的步骤之前初始化：
        # __del__ 会调用 close()，而下面的 stat/读 meta 都可能抛异常
        # （对齐 mdf_lazy_loader.py:69-99 的顺序论证）。
        self._access_lock = threading.RLock()
        self._closed = False
        self._cache = ColumnCache()  # D7：默认 64MB 双预算
        self._cache_dir = cache_dir

        import os

        parquet_path = Path(cache_dir.path()) / _PARQUET_NAME
        meta_path = Path(cache_dir.path()) / _META_NAME
        if not parquet_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"临时目录缺少转换产物（{parquet_path} / {meta_path}）"
            )
        try:
            self._file_size = os.stat(src_path).st_size
        except OSError as e:
            raise FileNotFoundError(f"源文件不存在: {src_path}") from e

        # meta.json 一次读入并立刻关闭句柄（GC 契约：不持有打开的 fd，
        # 否则 Windows 上 close() 删不掉临时目录）
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        try:
            self._rows = int(meta["rows"])
            self._var_names = list(meta["var_names"])  # 已排除时间列
            self._var_names_all = list(meta["var_names_all"])
            self._units = dict(meta["units"])
            self._date_formats = dict(meta["date_formats"])
            self._columns = dict(meta["columns"])
            self._enum_labels = dict(meta["enum_labels"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"meta.json 结构不完整或损坏: {e}") from e
        logger.info(
            "ParquetLazyLoader 就绪: %d 行 %d 列（源 %.1f MB）",
            self._rows,
            len(self._var_names_all),
            self._file_size / 1024 / 1024,
        )

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _ensure_open(self):
        """调用方必须已持有 self._access_lock（语义同 MDFLazyLoader）。"""
        if self._closed:
            raise KeyError("parquet 数据源已关闭")

    def _read_column(self, name: str):
        """从 parquet 读单列（调用方持锁；返回 polars Series）。"""
        import polars as pl

        frame = pl.read_parquet(
            str(Path(self._cache_dir.path()) / _PARQUET_NAME), columns=[name]
        )
        return frame.get_column(name)

    def _get_or_read(self, name: str):
        """缓存命中的 polars 单列；未命中读 parquet 并入缓存（调用方持锁）。

        缓存键带 "r:" 前缀——ColumnCache 里同时存着 pandas 形态（"s:"）
        与枚举码值（"v:"），见 _get_or_build_series。
        """
        key = f"r:{name}"
        cached = self._cache.get(key)
        if cached is None:
            cached = self._read_column(name)
            self._cache.put(key, cached)
        return cached

    def _get_or_build_series(self, name: str) -> pd.Series:
        """pandas 形态单列（缓存键 "s:"；调用方持锁）。

        dtype 还原（astype("category") / from_codes / astype(dtype_str)）
        是一次性成本：高基数文本列的 category 化实测 200k unique 约
        70ms，不缓存的话每次 get_series 都重付（P4 DoD 6 实测发现）。
        """
        key = f"s:{name}"
        s = self._cache.get(key)
        if s is None:
            s = self._to_pandas_series(name, self._get_or_read(name))
            self._cache.put(key, s)
        return s

    def _to_pandas_series(self, name: str, ps) -> pd.Series:
        """polars 单列 → 与今天 pandas 内存 loader 逐字相同 dtype 的 Series。

        纯内存操作，无需占用锁（对齐 mdf_lazy_loader.get_series 的锁外构造）。
        """
        entry = self._columns[name]
        dtype_str = entry.get("dtype_str", "object")

        if entry.get("is_enum"):
            labels = self._enum_labels.get(name) or []
            # null → -1（from_codes 的 NaN 官方编码）；避免 Int32 含 null
            # 时 to_numpy() 升 float64 的坑
            codes = ps.fill_null(-1).to_numpy()
            return pd.Series(
                pd.Categorical.from_codes(codes, categories=labels), name=name
            )
        if dtype_str == "category":
            # CSV 文本列（高基数 / 全空）：今天 read_csv(dtype="category")
            return pd.Series(ps.to_numpy(), name=name).astype("category")
        if dtype_str.startswith("datetime64"):
            # Excel 日期列：还原原分辨率（datetime64[s] 等来自 str(s.dtype)）
            return pd.Series(ps.to_numpy(), name=name).astype(dtype_str)
        if dtype_str in ("object", "str"):
            # CSV 日期列（pandas 自动推断 StringDtype）/ Excel 文本列
            return pd.Series(ps.to_numpy(), name=name).astype("str")
        # float32 / float64 / int64 / bool
        return pd.Series(ps.to_numpy(), name=name)

    # ------------------------------------------------------------------
    # 数据访问（§4.4 契约）
    # ------------------------------------------------------------------

    def get_series(self, name: str) -> pd.Series:
        with self._access_lock:
            self._ensure_open()
            if name not in self._columns:
                raise KeyError(f"变量 '{name}' 不存在")
            return self._get_or_build_series(name)

    def get_value_from_name(self, name: str):
        """绘图四元组 (index, values, unit, text_map)。

        枚举列（D8）：values 是**码值** ndarray（null 位为 NaN），
        text_map = {码: 标签}；其余列 values 是 pandas Series、text_map 为 {}。
        """
        with self._access_lock:
            self._ensure_open()
            if name not in self._columns:
                raise KeyError(f"变量 '{name}' 不存在")
            entry = self._columns[name]
            unit = entry.get("unit", "-")
            x = np.arange(1, self._rows + 1, dtype=np.float64)
            if entry.get("is_enum"):
                key = f"v:{name}"
                codes = self._cache.get(key)
                if codes is None:
                    # null → NaN（polars Int32 null 经 to_numpy 自动升
                    # float64），绘图侧 NaN 即断线语义
                    codes = self._get_or_read(name).to_numpy()
                    self._cache.put(key, codes)
                labels = self._enum_labels.get(name) or []
                text_map = {i: lab for i, lab in enumerate(labels)}
                return x, codes, unit, text_map
            return x, self._get_or_build_series(name), unit, {}

    def meta(self, name: str) -> dict:
        """单列转换期元数据（dtype_str/all_empty/validity/unit/is_enum/n_unique）。

        O(1) 只读入口，供 var_info._from_parquet 使用（D15：不读 parquet、
        不构造 Series）。返回浅拷贝，调用方的改写不污染内部状态。
        """
        with self._access_lock:
            self._ensure_open()
            entry = self._columns.get(name)
            if entry is None:
                raise KeyError(f"变量 '{name}' 不存在")
            return dict(entry)

    # ------------------------------------------------------------------
    # Properties（对齐 §4.4 契约表；元数据全部来自 meta.json，不读 parquet）
    # ------------------------------------------------------------------

    @property
    def path(self) -> str:
        return self._path

    @property
    def file_size(self) -> int:
        return self._file_size

    @property
    def df(self):
        return None  # 硬性契约：数据不在内存

    @property
    def datalength(self) -> int:
        return self._rows

    @property
    def max_row_count(self) -> int:
        return self._rows

    @property
    def row_count(self) -> int:
        return self._rows

    @property
    def column_count(self) -> int:
        return len(self._var_names_all)

    @property
    def var_names(self) -> list[str]:
        return list(self._var_names)  # 已排除时间列（转换期口径）

    @property
    def units(self) -> dict[str, str]:
        return dict(self._units)

    @property
    def df_validity(self) -> dict[str, int]:
        # 转换期全量真值（枚举列记 1，D8/D12）；var_names 口径已排除时间列
        return {n: self._columns[n].get("validity", -1) for n in self._var_names}

    @property
    def time_column_name(self):
        return None  # CSV/Excel 语义：无时间列

    @property
    def time_axis_label(self) -> str:
        return "Index"

    @property
    def time_channels_info(self) -> dict[str, str]:
        return dict(self._date_formats)

    @property
    def global_time_range(self) -> tuple[float, float]:
        return (1.0, float(self._rows))

    @property
    def baseline_density(self) -> float:
        return 1.0

    @property
    def time_values(self) -> pd.Series:
        # 无时间列 → 与 base_loader.default_time_values 同口径
        return pd.Series(np.arange(1, self._rows + 1), name="index")

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def release_memory(self):
        """清空列缓存（数据可按需重读）；不动临时目录。"""
        with self._access_lock:
            self._cache.clear()

    def close(self):
        """关闭并删除临时目录（幂等）。

        先置 _closed 再清理（顺序不可颠倒）：并发访问方在同一把锁内
        首行检查该标志，拿到可预期的 KeyError 而非 AttributeError。
        """
        # getattr 兜底：__init__ 极早期失败时 __del__ 仍可能调用到这里
        lock = getattr(self, "_access_lock", None)
        if lock is None:
            return
        with lock:
            if self._closed:
                return  # 幂等
            self._closed = True
            cache = getattr(self, "_cache", None)
            if cache is not None:
                cache.clear()
            cache_dir = getattr(self, "_cache_dir", None)
            if cache_dir is not None:
                cache_dir.cleanup()  # TempCacheDir.cleanup 自身幂等
