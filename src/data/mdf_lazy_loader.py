"""
MDFLazyLoader - memory='low' + LRU cache based MDF file loader

Replaces the old MDFDataLoader (full in-memory loading) with a lazy-loading
approach that only loads metadata at initialization and reads signal data
on demand with an LRU cache.

Supported formats: .mf4 (MDF 4.x), .mdf (MDF 3.x), .dat (INCA export)
"""

import os
import re
import threading
import traceback
from collections import OrderedDict
from typing import Optional, Callable

import numpy as np
import pandas as pd

from src.data.metadata import (
    VarMetadata,
    UNKNOWN,
    is_enum_conversion,
    is_mdf3_version,
    extract_enum_map,
)
from src.core.logger import get_logger

logger = get_logger("data.mdf")

# MDF4 的标称采样间隔藏在 CN comment 的 XML 内（实测 <raster>0.001</raster>）。
# 仅在 comment 含 "<raster>" 子串时才走正则，避免数千通道的无谓开销。
_RASTER_RE = re.compile(r"<raster>([^<]+)</raster>")

# 字符串/字节流类 data_type，这类通道不可数值统计与绘图。
# 为何不能只看 dtype_fmt：MDF4 的可变长字符串在记录里存的是指向
# SDblock 的索引，asammdf 因此把 dtype_fmt 报为 uint64（实测合成文件：
# data_type=7(STRING_UTF_8) 而 dtype_fmt=uint64，get() 之后才是 |S1），
# 单看 dtype_fmt 会把 MDF4 字符串通道误报为数值。MDF3 则相反，其
# dtype_fmt 就是真实的 |S1，两者并查才能跨版本正确。
# 取值来自 asammdf 8.8.9 的 DATA_TYPE_* 常量表。
_STRING_DT_MDF3 = frozenset({7, 8})  # STRING / BYTEARRAY
_STRING_DT_MDF4 = frozenset({6, 7, 8, 9, 10, 11, 12, 17})
# 上述集合对应 STRING_LATIN_1 / STRING_UTF_8 / STRING_UTF_16_LE /
# STRING_UTF_16_BE / BYTEARRAY / MIME_SAMPLE / MIME_STREAM / STRING_WITH_BOM


class MDFLazyLoader:

    LOADER_TYPE = "mdf"

    MAX_CACHE_SIZE = 256
    # 信号缓存字节预算。单条就是整通道数组：776 MB 级 .mf4 的单通道可达数十至上百
    # MB，只限 256 条时理论峰值是 GB 级，故再按 nbytes 压一道上限。
    MAX_CACHE_BYTES = 256 * 1024 * 1024
    # 时间戳缓存上限。每个通道组一条时间轴，超限时淘汰最久未用的条目。
    # Trade-off: >64 组的 MDF 文件在频繁切换绘图时会产生时间戳重读（asammdf 元数据级 I/O），
    # 但避免了无界内存增长（每条时间轴可达数 MB）。实测 64 组覆盖绝大多数 MDF 文件。
    MAX_TIME_CACHE_SIZE = 64
    MAX_TIME_CACHE_BYTES = 64 * 1024 * 1024

    _ASAMMDF_IMPORT_ERROR = "asammdf 库未安装。请运行: pip install asammdf>=7.4.0"

    def __init__(self, path: str, *, _progress: Callable[[int], None] = None):
        self._path = path
        self._progress = _progress

        # 并发保护与关闭标志必须在任何可能抛异常的步骤之前初始化：
        # __del__ 会调用 close()，而 _validate_file() 与 asammdf.MDF() 均可能抛异常，
        # 此时若锁尚未创建，close() 自身会再抛 AttributeError。
        #
        # 为何共享 handle 而不为后台统计开独立 handle：实测 776 MB .mf4 重开需
        # 737 ms 且元数据内存翻倍。UI 线程绘图与统计 worker 共用同一 handle，
        # 由 RLock 串行化；CPython 无竞争 RLock 约 50-80 ns，相对 18 ms 级的
        # 通道读取可忽略。
        self._access_lock = threading.RLock()
        # close() 先置位再关句柄；所有数据访问在同一把锁内首行检查此标志，
        # 使并发方在 loader 关闭后拿到可预期的 KeyError 而非 AttributeError。
        self._closed = False
        self._mdf = None

        # close() 会清空下列所有容器，因此它们也必须先于任何可能抛异常的
        # 步骤创建。实测：_validate_file() 对不存在/零字节文件抛异常后，
        # __del__ → close() 会因 _signal_cache 缺失再抛
        # AttributeError（在 GC 路径上表现为 "Exception ignored in __del__"）。
        self._signal_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._time_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        # 两份缓存各自的 nbytes 总量，与容器同步维护（插入累加、逐出递减、清空归零）
        self._signal_cache_bytes = 0
        self._time_cache_bytes = 0
        self._enum_cache: dict[str, dict[int, str]] = {}
        self._metadata: list[VarMetadata] = []
        self._var_to_meta: dict[str, VarMetadata] = {}
        self._original_to_aggregated: dict[tuple[int, str], str] = {}
        self._group_master_ci: dict[int, int] = {}
        self._current_group_index: int = 0

        self._validate_file()
        logger.info("开始加载 MDF 文件: %s (%.1f MB)", path, self._file_size / 1024 / 1024)

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
        # 旧写法打印 self._current_group_index + 1（当前查看的组序号+1），未加载
        # 任何组之前恒为 1，日志里那句“1 组”与真实组数无关，排查 tab 问题时
        # 严重误导（实测 776MB 文件 317 组被记成 1 组）。
        logger.info(
            "MDF 加载完成: %d 个信号, %d 个 channel group",
            len(self._metadata),
            self.group_count,
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
        # 版本仅取一次：conversion_type 语义与 raster 提取方式均依赖它，
        # 避免在数千通道的循环里重复访问。
        mdf_version = self._mdf.version

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
                enum_flag = is_enum_conversion(conversion, mdf_version)
                enum_map = extract_enum_map(conversion) if enum_flag else None

                nominal_raster_s = self._extract_nominal_raster(ch, mdf_version)

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
                    nominal_raster_s=nominal_raster_s,
                    original_name=ch_name,
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

    @staticmethod
    def _extract_nominal_raster(ch, mdf_version: Optional[str]) -> Optional[float]:
        """提取标称采样间隔（秒）。

        MDF3：ch.sampling_rate 本身就是以秒为单位的间隔（asammdf 文档明确
              "sampling rate in 's'"），不能当作 Hz。
        MDF4：v4 Channel 无 sampling_rate 属性，raster 藏在 CN comment 的 XML
              <raster> 标签内。

        注意标称值不等于有效采样率：实测某 .mf4 标称 0.001 s（1 kHz），
        而 master 时间戳实测间隔约 0.01 s（100 Hz），两者必须分开展示。
        """
        if is_mdf3_version(mdf_version):
            sr = getattr(ch, "sampling_rate", None)
            if sr is not None and sr > 0:
                return float(sr)
            return None

        comment = getattr(ch, "comment", "") or ""
        if "<raster>" not in comment:
            return None
        match = _RASTER_RE.search(comment)
        if not match:
            return None
        try:
            val = float(match.group(1))
        except (TypeError, ValueError):
            return None
        return val if val > 0 else None

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
                    nominal_raster_s=meta.nominal_raster_s,
                    original_name=pure_name,
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

        # 按 group_index 预建索引，使下方回填从 O(组数 × 变量数) 降为 O(变量数)：
        # 4704 变量 × 317 组的场景由 149 万次比较降为 4704 次。
        # 注意索引的是 self._metadata（聚合后对象），_var_to_meta 指向同一批对象，
        # 且本方法在 _build_aggregated_properties 末尾调用，此时聚合已完成。
        metas_by_group: dict[int, list[VarMetadata]] = {}
        for meta in self._metadata:
            metas_by_group.setdefault(meta.group_index, []).append(meta)

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

                # 回填本组全部变量的时间基准与点数：这些数据此处已经取得，
                # 不产生任何额外磁盘读取。effective_rate_hz 是平均值，
                # 对变速/事件型采样会失真，展示时需加注说明。
                rate = (
                    (cycles - 1) / (t_max - t_min)
                    if (cycles > 1 and t_max > t_min)
                    else None
                )
                for meta in metas_by_group.get(gi, ()):
                    meta.time_min = t_min
                    meta.time_max = t_max
                    meta.sample_count = cycles
                    meta.effective_rate_hz = rate

            except Exception:
                # 时间范围汇总属于"有则更好"的附加信息（改进 B），失败不得
                # 影响加载主流程；记完整 traceback 比只记异常文本更易定位。
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

    def _enforce_cache_budget(
        self,
        cache: OrderedDict,
        byte_total: int,
        *,
        max_entries: int,
        max_bytes: int,
    ) -> int:
        """LRU 逐出到「条数 + 字节」双双达标，返回逐出后的字节总量。

        调用方必须已持有 self._access_lock。至少保留一条：单通道数组本身就可能超过
        预算，把它也逐出等于刚读出来就丢，下次同一通道再读一遍。
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
        """写入时间轴 LRU 缓存（调用方持锁），返回入缓存的那份数组"""
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
        """调用方必须已持有 self._access_lock。

        close() 在同一把锁内置位 _closed 并关闭句柄，因此 RLock 互斥保证
        并发读取与 close 不可能交叉：进入临界区时 close 要么已完成（此处抛
        KeyError，调用方可预期地降级），要么尚未开始（本次读取完成后 close
        才拿到锁）。这消除了"在已关闭的 asammdf 句柄上继续读取"导致的
        C 扩展层崩溃风险。
        """
        if self._closed or getattr(self, "_mdf", None) is None:
            raise KeyError("MDF 数据源已关闭")

    def _clear_lru_caches(self):
        """清空两份 LRU 缓存并归零字节账（调用方持锁）"""
        self._signal_cache.clear()
        self._time_cache.clear()
        self._signal_cache_bytes = 0
        self._time_cache_bytes = 0

    def clear_cache(self):
        with self._access_lock:
            self._clear_lru_caches()

    def release_memory(self):
        """清空 LRU 缓存（信号数据可以按需重新加载）。"""
        with self._access_lock:
            self._clear_lru_caches()

    def close(self):
        # getattr 兜底：__init__ 极端早期失败时 __del__ 仍可能调用到这里。
        lock = getattr(self, "_access_lock", None)
        if lock is None:
            return
        with lock:
            # 先置位再关句柄，顺序不可颠倒。
            self._closed = True
            # 逐个 getattr 兜底：__del__ 可能在任意构造阶段被触发（包括锁已
            # 建但容器未建的中间态），此处的 AttributeError 会变成难以诊断的
            # "Exception ignored in __del__" 噪声，而且会跳过后续的 _mdf.close()。
            for attr in (
                "_signal_cache",
                "_time_cache",
                "_enum_cache",
                "_metadata",
                "_var_to_meta",
            ):
                container = getattr(self, attr, None)
                if container is not None:
                    container.clear()
            # 上面只清容器，两份 LRU 的字节账要单独归零（直接赋值：__del__ 走到
            # 早期构造失败态时这两个属性可能还不存在）
            self._signal_cache_bytes = 0
            self._time_cache_bytes = 0
            self._cached_max_samples = 0
            self._cached_global_time_range = (0.0, 1.0)
            if getattr(self, "_mdf", None) is not None:
                try:
                    self._mdf.close()
                except Exception:
                    logger.debug("关闭 MDF 文件时异常\n%s", traceback.format_exc())
                # 不再 del：del 后紧接赋值语义混乱，且会让并发方拿到
                # AttributeError('NoneType' object has no attribute 'get')
                # 而非 _ensure_open 抛出的可预期 KeyError。
                self._mdf = None

    # ------------------------------------------------------------------
    # Core data access
    # ------------------------------------------------------------------

    def get_series(self, display_name: str) -> pd.Series:
        with self._access_lock:
            self._ensure_open()
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

        # 锁外构造：纯内存操作，无需占用共享 handle
        return pd.Series(y, name=display_name)

    def get_value_from_name(self, display_name: str):
        with self._access_lock:
            self._ensure_open()
            meta = self._var_to_meta.get(display_name)
            if meta is None:
                raise KeyError(f"变量 '{display_name}' 不存在")

            gi = meta.group_index
            x = self._time_cache.get(gi)
            if x is None:
                master_ci = self._group_master_ci.get(gi, 0)
                master_signal = self._mdf.get(
                    name=None,
                    group=gi,
                    index=master_ci,
                )
                x = self._cache_put_time(
                    gi, master_signal.timestamps.astype(np.float64)
                )
            else:
                self._time_cache.move_to_end(gi)

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
            unit = meta.unit

        return x, y, unit, enum_map or {}

    # ------------------------------------------------------------------
    # Group-level access (供变量数值表 tab 模式使用)
    #
    # 这组接口为 UI 层提供 channel group 维度的数据访问能力，
    # 使 DataTableDialog 可按 group 分 tab 展示不同时间轴的变量。
    # ------------------------------------------------------------------

    def get_var_group_index(self, display_name: str) -> int:
        """返回变量所属的 channel group 索引。

        Args:
            display_name: 变量的显示名称（可能含 _G{gi} 后缀）

        Returns:
            int: channel group 索引

        Raises:
            KeyError: 变量不存在
        """
        meta = self._var_to_meta.get(display_name)
        if meta is None:
            raise KeyError(f"变量 '{display_name}' 不存在")
        return meta.group_index

    def get_group_time_array(self, group_index: int) -> np.ndarray:
        """返回指定 group 的 master 时间戳数组。

        复用已有 _time_cache（LRU 淘汰），与 get_value_from_name 共享缓存。

        Args:
            group_index: channel group 索引

        Returns:
            np.ndarray: float64 时间戳数组
        """
        with self._access_lock:
            self._ensure_open()
            timestamps = self._time_cache.get(group_index)
            if timestamps is None:
                master_ci = self._group_master_ci.get(group_index, 0)
                master_signal = self._mdf.get(
                    name=None,
                    group=group_index,
                    index=master_ci,
                )
                timestamps = self._cache_put_time(
                    group_index, master_signal.timestamps.astype(np.float64)
                )
            else:
                self._time_cache.move_to_end(group_index)
            return timestamps

    def get_group_label(self, group_index: int) -> str:
        """返回 group 的可读标签。

        优先使用 acq_name，其次 comment 前 20 字符，兜底 "G{index}"。
        用于 tooltip 和搜索栏候选列表。

        Args:
            group_index: channel group 索引

        Returns:
            str: 可读标签（如 "ECU1 (G0)" 或 "G5"）
        """
        with self._access_lock:
            self._ensure_open()
            if group_index not in self._raw_metadata:
                return f"G{group_index}"
            group_obj = self._mdf.groups[group_index]
            cg = getattr(group_obj, "channel_group", None)
            acq_name = getattr(cg, "acq_name", "") or ""
            if acq_name.strip():
                return f"{acq_name.strip()} (G{group_index})"
            comment = getattr(cg, "comment", "") or ""
            if comment.strip():
                short = comment.strip()[:20]
                return f"{short} (G{group_index})"
            return f"G{group_index}"

    def get_group_variables(self, group_index: int) -> list[str]:
        """列出指定 channel group 内的全部变量名（聚合后显示名，按通道顺序）。

        必须读 self._metadata（聚合后）而不是 _raw_metadata：后者存的是文件里的
        原始通道名，跨组重名时表格列名是 Press_G0/Press_G1（见
        _build_aggregated_properties 的 conflict_names 分支），用原始名去
        get_series 会 KeyError。时间通道在聚合时已被排除，无需再过滤。

        未知 group 返回空列表（与 get_group_time_array 对越界 group 的宽容处理
        一致）。但注意：数据源已 close 时 _ensure_open 仍抛 KeyError，"不必
        try/except"仅指越界组这一种情况；UI 侧调用方（_pending_group_var_count
        / 批量添加入口）按"会抛"口径做降级，口径以测试
        test_closed_loader_raises_keyerror 为准。

        Args:
            group_index: channel group 索引

        Returns:
            list[str]: 该组的变量显示名，顺序与文件内通道顺序一致
        """
        with self._access_lock:
            self._ensure_open()
            return [m.name for m in self._metadata if m.group_index == group_index]

    def search_variables(
        self, keyword: str, limit: int = 50
    ) -> list[tuple[str, int, str]]:
        """跨所有 group 搜索变量名（大小写不敏感子串匹配）。

        当前无 UI 调用方：变量数值表定位框只列“已加入表格”的变量（见
        DataTableDialog._refresh_var_locator_items），本方法作为 loader 的
        通用元数据接口保留，供后续全文件搜索类需求直接使用。

        Args:
            keyword: 搜索关键词
            limit: 最大返回数量

        Returns:
            list[tuple[str, int, str]]: [(display_name, group_index, group_label), ...]
        """
        keyword_lower = keyword.lower()
        results: list[tuple[str, int, str]] = []
        # 加锁统一元数据读取纪律（_access_lock 是 RLock，内部 get_group_label
        # 可重入）；_ensure_open 要求调用方已持锁，且 close() 后还能给出
        # 明确报错而不是拿着已释放的句柄报意外异常
        with self._access_lock:
            self._ensure_open()
            # 缓存已查过的 group_label，避免重复加锁访问
            label_cache: dict[int, str] = {}
            for meta in self._metadata:
                if keyword_lower in meta.name.lower():
                    gi = meta.group_index
                    if gi not in label_cache:
                        label_cache[gi] = self.get_group_label(gi)
                    results.append((meta.name, gi, label_cache[gi]))
                    if len(results) >= limit:
                        break
        return results

    # ------------------------------------------------------------------
    # Read-only metadata access (供变量信息窗口使用)
    #
    # 这组接口的存在意义：不让 UI 层直接触碰 _mdf 与块对象，使信息提取
    # 逻辑与具体 Loader 解耦。返回值均为已提取好的基础类型（str/int/float/
    # dict），UI 层只需组织展示，不需感知 MDF3/MDF4 的块结构差异。
    # ------------------------------------------------------------------

    _CHANNEL_ATTRS = (
        "name", "unit", "channel_type", "sync_type", "data_type",
        "bit_count", "byte_offset", "bit_offset", "precision",
        "lower_limit", "upper_limit", "address", "comment",
        # description: v3 的 CN 长文本落点（合成文件只写 description、comment 为空）；
        # display_names: v4 的层级显示名映射。两者供归属信息提取使用，v3/v4 互缺
        # 一侧时由 getattr 置 None，调用方按缺省处理。
        "description", "display_names",
    )
    _CG_ATTRS = (
        "cycles_nr", "samples_byte_nr", "record_id",
        "acq_name", "acq_source", "comment",
    )
    _SOURCE_ATTRS = ("name", "path", "bus_type", "source_type", "comment", "address")
    _CONV_ATTRS = (
        "conversion_type", "unit", "name", "a", "b", "formula",
        "ref_param_nr", "comment",
        "P1", "P2", "P3", "P4", "P5", "P6", "P7",
    )
    _HEADER_ATTRS = (
        "author", "department", "project", "subject",
        "start_time_string", "comment",
    )

    @staticmethod
    def _block_attrs(obj, names: tuple) -> dict:
        """按名从 asammdf 块对象提取属性；对象为 None 时返回空 dict。

        bytes 统一解码为 str，缺失属性置 None，保证 UI 层拿到的都是可直接
        渲染的类型。不同 MDF 版本的块属性集不同（如 v3 的 RAT 转换用 P1..P4、
        v4 用 a/b），用 getattr 兼容两者而不做版本分支。

        可调用属性（如 v3/v4 HeaderBlock.start_time_string 均为**方法**而非数据）
        一律无参调用取返回值，调用失败置 None —— 宁可少一行，也不能把 bound
        method 的 repr 原样显示给用户。
        """
        if obj is None:
            return {}
        out = {}
        for n in names:
            v = getattr(obj, n, None)
            if callable(v):
                try:
                    v = v()
                except Exception:
                    logger.debug("块属性 %s 调用失败", n, exc_info=True)
                    v = None
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace").rstrip("\x00")
            out[n] = v
        return out

    def get_metadata(self, display_name: str) -> Optional[VarMetadata]:
        """返回聚合后的变量元数据对象（不存在时返回 None）。

        与 get_channel_info / get_samples_chunked 保持一致：数据源已关闭时
        抛 KeyError。若此处静默返回 None，调用方就无法区分"变量不存在"与
        "数据源已关闭"，改进 I 的统一降级契约也就失效了。
        """
        with self._access_lock:
            self._ensure_open()
            return self._var_to_meta.get(display_name)

    def get_channel_info(self, display_name: str) -> dict:
        """返回单通道的完整块结构信息。

        全部来自加载期已解析的内存对象，**零磁盘 I/O**（实测六组全属性
        访问约 99 μs），因此可在 UI 线程同步调用。

        dtype 取自 ch.dtype_fmt：实测 v3/v4 通用且与实际读取 dtype 一致
        （v3 字符串通道 -> dtype('S256')、v4 数值通道 -> dtype('uint16')），
        因此字符串通道识别无需 get(record_count=1) 探测。

        channel 组额外携带 description / display_names 两个文本字段，
        供 src.data.mdf_attribution 提取变量归属信息（设备 / ECU / 函数）。
        """
        with self._access_lock:
            self._ensure_open()
            meta = self._var_to_meta.get(display_name)
            if meta is None:
                raise KeyError(f"变量 '{display_name}' 不存在")

            gi, ci = meta.group_index, meta.channel_index
            group = self._mdf.groups[gi]
            channels = group.channels
            ch = channels[ci]
            cg = group.channel_group

            dtype_fmt = getattr(ch, "dtype_fmt", None)
            try:
                np_dtype = np.dtype(dtype_fmt) if dtype_fmt is not None else None
            except TypeError:
                np_dtype = None

            version = str(getattr(self._mdf, "version", "") or "")
            data_type = getattr(ch, "data_type", None)
            string_dts = _STRING_DT_MDF3 if is_mdf3_version(version) else _STRING_DT_MDF4
            is_string_by_dt = data_type in string_dts

            # S=字节串 / U=Unicode 串 / O=对象，均不可绘图与统计；
            # 再并查 data_type 以覆盖 MDF4 字符串通道的 dtype_fmt 失真
            is_numeric = not is_string_by_dt and (
                np_dtype.kind not in "SUO" if np_dtype is not None else True
            )
            dtype_str = str(np_dtype) if np_dtype is not None else ""
            if (
                is_string_by_dt
                and np_dtype is not None
                and np_dtype.kind not in "SUO"
            ):
                # 如实标注，避免用户看到 uint64 却读到文本而困惑
                dtype_str = f"{dtype_str}（记录内为字符串索引，读取后解引用为文本）"

            master_ci = self._group_master_ci.get(gi)
            master_name = (
                channels[master_ci].name
                if master_ci is not None and master_ci < len(channels)
                else ""
            )

            cg_info = self._block_attrs(cg, self._CG_ATTRS)
            cg_info["channel_count"] = len(channels)

            return {
                "meta": meta,
                "version": version,
                "dtype": dtype_str,
                "is_numeric": bool(is_numeric),
                "channel": self._block_attrs(ch, self._CHANNEL_ATTRS),
                "channel_group": cg_info,
                "source": self._block_attrs(getattr(ch, "source", None), self._SOURCE_ATTRS),
                "conversion": self._block_attrs(
                    getattr(ch, "conversion", None), self._CONV_ATTRS
                ),
                "header": self._block_attrs(
                    getattr(self._mdf, "header", None), self._HEADER_ATTRS
                ),
                "time_base": {
                    "master_name": master_name,
                    "nominal_raster_s": meta.nominal_raster_s,
                    "effective_rate_hz": meta.effective_rate_hz,
                    "time_min": meta.time_min,
                    "time_max": meta.time_max,
                    "sample_count": meta.sample_count,
                },
                "file": {
                    "path": self._path,
                    "size": self._file_size,
                    "group_count": len(self._raw_metadata),
                    "var_count": len(self._metadata),
                },
                "enum_map": meta.enum_map,
            }

    def get_samples_chunked(
        self, display_name: str, offset: int, count: int
    ) -> np.ndarray:
        """分块读取**物理值**样本，供后台统计流式累加。

        与 get_series() 的两个关键差异：
        1. 恒定 raw=False。绘图路径用 raw=meta.is_enum（枚举取码值配合文本
           标签），而统计必须基于物理值，否则 min/max/mean 得到的是无意义
           的枚举码。
        2. 不写入 _signal_cache。统计是块级流式累加，缓存整块会挤占绘图的
           LRU 空间（单条 428k float32 约 1.7 MB，上限 256 条）。

        调用方按 count 分块循环，可使单次锁持有时间控制在 ≤20 ms，
        UI 线程并发绘图无可感知停顿。

        ``count <= 0`` 表示"读到末尾"（供 sample_count 未回填的通道降级使用）。
        """
        with self._access_lock:
            self._ensure_open()
            meta = self._var_to_meta.get(display_name)
            if meta is None:
                raise KeyError(f"变量 '{display_name}' 不存在")
            # 必须把"读到末尾"规范化为 None 再交给 asammdf：实测 MDF4 会把
            # -1 当真实点数用于计算缓冲区大小，抛
            # ValueError: negative count（mdf_v4.py 的 bytearray(split_size)）；
            # 而 MDF3 恰好容忍 -1。两版行为不一致，故在此统一。
            record_count = count if count and count > 0 else None
            signal = self._mdf.get(
                name=None,
                group=meta.group_index,
                index=meta.channel_index,
                raw=False,
                record_offset=offset,
                record_count=record_count,
            )
            return signal.samples

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
