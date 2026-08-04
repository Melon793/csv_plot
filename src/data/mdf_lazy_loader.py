"""
MDFLazyLoader - memory='low' + LRU cache based MDF file loader

Replaces the old MDFDataLoader (full in-memory loading) with a lazy-loading
approach that only loads metadata at initialization and reads signal data
on demand with an LRU cache.

Supported formats: .mf4 (MDF 4.x), .mdf (MDF 3.x), .dat (INCA export)
"""

import os
import traceback
from collections import OrderedDict
from typing import Optional, Callable

import numpy as np
import pandas as pd

from src.data.metadata import (
    VarMetadata,
    UNKNOWN,
    is_enum_conversion,
    extract_enum_map,
)
from src.core.logger import get_logger

logger = get_logger("data.mdf")


class MDFLazyLoader:

    LOADER_TYPE = "mdf"

    MAX_CACHE_SIZE = 256

    _ASAMMDF_IMPORT_ERROR = "asammdf 库未安装。请运行: pip install asammdf>=7.4.0"

    def __init__(self, path: str, *, _progress: Callable[[int], None] = None):
        self._path = path
        self._progress = _progress
        self._validate_file()
        logger.info("开始加载 MDF 文件: %s (%.1f MB)", path, self._file_size / 1024 / 1024)

        self._signal_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._time_cache: dict[int, np.ndarray] = {}
        self._enum_cache: dict[str, dict[int, str]] = {}
        self._metadata: list[VarMetadata] = []
        self._var_to_meta: dict[str, VarMetadata] = {}
        self._original_to_aggregated: dict[tuple[int, str], str] = {}
        self._group_master_ci: dict[int, int] = {}
        self._current_group_index: int = 0

        self._notify_progress(0)

        try:
            import asammdf
        except ImportError as e:
            raise ImportError(f"{self._ASAMMDF_IMPORT_ERROR}\n原始错误: {e}") from e

        try:
            self._mdf = asammdf.MDF(path)
        except Exception as e:
            raise RuntimeError(
                f"MDF 文件无法以惰性模式打开（文件可能已损坏或版本不兼容）: {e}"
            ) from e

        self._notify_progress(5)

        self._load_metadata()
        self._build_aggregated_properties()

        self._notify_progress(100)
        logger.info(
            "MDF 加载完成: %d 个信号, %d 组",
            len(self._metadata),
            self._current_group_index + 1,
        )

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
            raise FileNotFoundError(f"MDF 文件不存在: {self._path}")

        file_size = os.path.getsize(self._path)
        if file_size == 0:
            raise ValueError("MDF 文件为空")
        self._file_size = file_size

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------

    def _load_metadata(self):
        groups = self._mdf.groups
        total_groups = len(groups)

        if total_groups == 0:
            raise ValueError("MDF 文件未包含任何 Channel Group")

        raw_metadata: dict[int, list[VarMetadata]] = {}

        mdf3_time_channels = {
            "time",
            "t",
            "timestamps",
            "timestamp",
            "zeit",
            "tmod",
            "Time",
        }

        for gi in range(total_groups):
            group_metas: list[VarMetadata] = []
            group_obj = groups[gi]

            cg = getattr(group_obj, "channel_group", None)
            cg_comment = getattr(cg, "comment", "") or ""
            if "SingleShotGroup" in cg_comment:
                raw_metadata[gi] = group_metas
                continue

            if not hasattr(group_obj, "channels") or not group_obj.channels:
                raw_metadata[gi] = group_metas
                continue

            master_ci = self._mdf.masters_db.get(gi)

            for ci, ch in enumerate(group_obj.channels):
                ch_name = ch.name

                is_master = (
                    master_ci is not None and ci == master_ci
                ) or ch_name.lower() in mdf3_time_channels

                if is_master:
                    self._group_master_ci[gi] = ci

                unit = self._extract_channel_unit(ch)
                conversion = getattr(ch, "conversion", None)
                enum_flag = is_enum_conversion(conversion)
                enum_map = extract_enum_map(conversion) if enum_flag else None

                sampling_rate_hz = None
                if hasattr(ch, "sampling_rate"):
                    sr = ch.sampling_rate
                    if sr is not None and sr > 0:
                        sampling_rate_hz = float(sr)

                is_time = is_master

                is_date = False
                is_time_of_day = False
                if conversion is not None:
                    if hasattr(conversion, "unit"):
                        cu = (conversion.unit or "").lower()
                        if cu == "date":
                            is_date = True
                        elif cu == "timeofday":
                            is_time_of_day = True

                meta = VarMetadata(
                    name=ch_name,
                    unit=unit,
                    group_index=gi,
                    channel_index=ci,
                    time_min=0.0,
                    time_max=0.0,
                    sample_count=0,
                    sampling_rate_hz=sampling_rate_hz,
                    is_enum=enum_flag,
                    is_time_channel=is_time,
                    is_date=is_date,
                    is_time_of_day=is_time_of_day,
                    validity=UNKNOWN,
                    enum_map=enum_map,
                )
                group_metas.append(meta)

                if enum_map and ch_name not in self._enum_cache:
                    self._enum_cache[ch_name] = enum_map

            raw_metadata[gi] = group_metas

        self._raw_metadata = raw_metadata

    @staticmethod
    def _extract_channel_unit(ch) -> str:
        if ch.unit and ch.unit.strip():
            return ch.unit.strip()
        conversion = getattr(ch, "conversion", None)
        if conversion is not None:
            conv_unit = getattr(conversion, "unit", None)
            if conv_unit and conv_unit.strip():
                return conv_unit.strip()
        return "-"

    # ------------------------------------------------------------------
    # Aggregation & conflict resolution
    # ------------------------------------------------------------------

    def _build_aggregated_properties(self):
        flat_metas: list[VarMetadata] = []
        name_counts: dict[str, int] = {}
        name_groups: dict[str, list[int]] = {}

        for gi, group_metas in self._raw_metadata.items():
            for meta in group_metas:
                if meta.is_time_channel:
                    continue
                flat_metas.append(meta)
                name_counts[meta.name] = name_counts.get(meta.name, 0) + 1
                if meta.name not in name_groups:
                    name_groups[meta.name] = []
                name_groups[meta.name].append(gi)

        conflict_names = {k for k, v in name_counts.items() if v > 1}

        self._metadata = []
        self._var_to_meta = {}
        self._original_to_aggregated = {}

        for gi, group_metas in self._raw_metadata.items():
            for meta in group_metas:
                if meta.is_time_channel:
                    continue
                pure_name = meta.name
                if pure_name in conflict_names:
                    display_name = f"{pure_name}_G{gi}"
                else:
                    display_name = pure_name

                aggregated_meta = VarMetadata(
                    name=display_name,
                    unit=meta.unit,
                    group_index=meta.group_index,
                    channel_index=meta.channel_index,
                    time_min=meta.time_min,
                    time_max=meta.time_max,
                    sample_count=meta.sample_count,
                    sampling_rate_hz=meta.sampling_rate_hz,
                    is_enum=meta.is_enum,
                    is_time_channel=meta.is_time_channel,
                    is_date=meta.is_date,
                    is_time_of_day=meta.is_time_of_day,
                    validity=UNKNOWN,
                    enum_map=meta.enum_map,
                )
                self._metadata.append(aggregated_meta)
                self._var_to_meta[display_name] = aggregated_meta
                self._original_to_aggregated[(gi, pure_name)] = display_name

        self._compute_global_time_range()

    def _compute_global_time_range(self):
        all_mins = []
        all_maxs = []
        total_samples = 0
        total_groups = len(self._raw_metadata)

        for idx, gi in enumerate(sorted(self._raw_metadata.keys())):
            if gi not in self._group_master_ci:
                continue

            master_ci = self._group_master_ci[gi]
            cg = self._mdf.groups[gi].channel_group
            cycles = cg.cycles_nr

            if cycles <= 0:
                continue

            try:
                sig_first = self._mdf.get(
                    name=None, group=gi, index=master_ci, record_count=1
                )
                if len(sig_first.timestamps) == 0:
                    continue

                t_min = float(sig_first.timestamps[0])

                if cycles > 1:
                    sig_last = self._mdf.get(
                        name=None,
                        group=gi,
                        index=master_ci,
                        record_offset=cycles - 1,
                        record_count=1,
                    )
                    t_max = (
                        float(sig_last.timestamps[0])
                        if len(sig_last.timestamps) > 0
                        else t_min
                    )
                else:
                    t_max = t_min

                all_mins.append(t_min)
                all_maxs.append(t_max)
                total_samples = max(total_samples, cycles)

            except Exception as e:
                logger.debug("汇总信号 gi=%d 时间范围时异常，跳过\n%s", gi, traceback.format_exc())

            if self._progress and total_groups > 0:
                progress = 50 + int((idx + 1) / total_groups * 50)
                self._notify_progress(min(progress, 99))

        if all_mins:
            self._cached_global_time_range = (min(all_mins), max(all_maxs))
        else:
            self._cached_global_time_range = (0.0, 1.0)

        self._cached_max_samples = total_samples

    # ------------------------------------------------------------------
    # LRU cache layer
    # ------------------------------------------------------------------

    def _cache_get(self, name: str) -> Optional[np.ndarray]:
        if name in self._signal_cache:
            self._signal_cache.move_to_end(name)
            return self._signal_cache[name]
        return None

    def _cache_put(self, name: str, data: np.ndarray):
        if name in self._signal_cache:
            self._signal_cache.move_to_end(name)
        else:
            if len(self._signal_cache) >= self.MAX_CACHE_SIZE:
                self._signal_cache.popitem(last=False)
            self._signal_cache[name] = data

    def clear_cache(self):
        self._signal_cache.clear()
        self._time_cache.clear()

    def release_memory(self):
        """清空 LRU 缓存（信号数据可以按需重新加载）。"""
        self._signal_cache.clear()
        self._time_cache.clear()

    def close(self):
        self._signal_cache.clear()
        self._time_cache.clear()
        self._enum_cache.clear()
        self._metadata.clear()
        self._var_to_meta.clear()
        self._cached_max_samples = 0
        self._cached_global_time_range = (0.0, 1.0)
        if hasattr(self, "_mdf") and self._mdf is not None:
            try:
                self._mdf.close()
            except Exception as e:
                logger.debug("关闭 MDF 文件时异常\n%s", traceback.format_exc())
            del self._mdf
            self._mdf = None

    # ------------------------------------------------------------------
    # Core data access
    # ------------------------------------------------------------------

    def get_series(self, display_name: str) -> pd.Series:
        meta = self._var_to_meta.get(display_name)
        if meta is None:
            raise KeyError(f"变量 '{display_name}' 不存在")

        y = self._cache_get(display_name)
        if y is None:
            signal = self._mdf.get(
                name=None,
                group=meta.group_index,
                index=meta.channel_index,
                raw=meta.is_enum,
            )
            y = signal.samples
            self._cache_put(display_name, y)

        return pd.Series(y, name=display_name)

    def get_value_from_name(self, display_name: str):
        meta = self._var_to_meta.get(display_name)
        if meta is None:
            raise KeyError(f"变量 '{display_name}' 不存在")

        gi = meta.group_index
        if gi not in self._time_cache:
            master_ci = self._group_master_ci.get(gi, 0)
            master_signal = self._mdf.get(
                name=None,
                group=gi,
                index=master_ci,
            )
            self._time_cache[gi] = master_signal.timestamps.astype(np.float64)

        x = self._time_cache[gi]

        y = self._cache_get(display_name)
        if y is None:
            signal = self._mdf.get(
                name=None,
                group=gi,
                index=meta.channel_index,
                raw=meta.is_enum,
            )
            y = signal.samples
            self._cache_put(display_name, y)

        enum_map = None
        if meta.is_enum:
            if display_name not in self._enum_cache and meta.enum_map:
                self._enum_cache[display_name] = meta.enum_map
            enum_map = self._enum_cache.get(display_name)

        return x, y, meta.unit, enum_map or {}

    # ------------------------------------------------------------------
    # Properties (aligned with FastDataLoader interface)
    # ------------------------------------------------------------------

    @property
    def var_names(self) -> list[str]:
        return [m.name for m in self._metadata]

    @property
    def units(self) -> dict[str, str]:
        return {m.name: m.unit for m in self._metadata}

    @property
    def df(self):
        return None

    @property
    def df_validity(self) -> dict[str, int]:
        return {m.name: UNKNOWN for m in self._metadata}

    @property
    def datalength(self) -> int:
        return getattr(self, "_cached_max_samples", 0)

    @property
    def max_row_count(self) -> int:
        return self.datalength

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
        for meta in self._metadata:
            if meta.is_time_channel and meta.group_index == self._current_group_index:
                return meta.name
        return "time"

    @property
    def time_axis_label(self) -> str:
        name = self.time_column_name or "time"
        unit = self.units.get(name, "-")
        if unit and unit != "-":
            return f"{name} ({unit})"
        return name

    @property
    def path(self) -> str:
        return self._path

    @property
    def file_size(self) -> int:
        return self._file_size

    @property
    def time_values(self):
        gi = self._current_group_index
        if gi in self._time_cache:
            return pd.Series(self._time_cache[gi], name=self.time_column_name or "time")
        if not self._metadata:
            return pd.Series(np.arange(1), name="index")
        return pd.Series(np.arange(self.datalength), name="index")

    @property
    def time_channels_info(self) -> dict[str, str]:
        # 与基类契约一致返回 dict[str, str]（消费方按 key 检查/取值）；
        # MDF 时间通道为数值型，无日期格式串，值统一为空字符串
        return {m.name: "" for m in self._metadata if m.is_time_channel}

    @property
    def groups(self) -> list:
        result = []
        for gi in sorted(self._raw_metadata.keys()):
            group_metas = self._raw_metadata.get(gi, [])
            if not group_metas:
                continue
            result.append(
                {
                    "index": gi,
                    "var_names": [m.name for m in group_metas],
                }
            )
        return result

    @property
    def group_count(self) -> int:
        return len(self._raw_metadata)

    @property
    def current_group_index(self) -> int:
        return self._current_group_index

    @property
    def row_count(self) -> int:
        return self.datalength

    @property
    def column_count(self) -> int:
        return len(self._metadata)
