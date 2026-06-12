"""loader"""

from __future__ import annotations
import os
import gc
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal
from src.core.config import (
    _UNIT_KEYWORDS,
    UNIT_KEYWORD_RATIO_THRESHOLD,
    VALID_NUMERIC_RATIO_THRESHOLD,
    _evaluate_float32_safety,
)
from src.core.data_types import FormatInfo, AutoDetectError
from src.core.logger import get_logger
from src.data.base_loader import BaseDataLoader

logger = get_logger("data.loader")


class DataLoadThread(QThread):
    """
    数据加载线程类
    在后台线程中异步加载CSV数据文件，避免阻塞主界面
    通过信号机制向主线程发送加载进度和结果
    """

    # 信号：发送进度 0-100，或直接发 DataFrame
    progress = Signal(int)  # 百分比
    finished = Signal(object)  # FastDataLoader 实例
    error = Signal(str)

    def __init__(
        self,
        file_path: str,
        parent=None,
        desc_rows: int = 0,
        sep: str = ",",
        has_unit: bool | None = True,
        encoding: str | None = None,
        sheet_name: str | None = None,
    ):
        super().__init__(parent)
        self.file_path = file_path
        self.desc_rows = desc_rows
        self.sep = sep
        self.has_unit = has_unit
        self.encoding = encoding
        self.sheet_name = sheet_name

    def run(self):
        """
        线程执行方法
        在后台线程中执行数据加载操作，避免阻塞主界面
        通过信号机制向主线程发送进度更新和结果
        """
        try:
            def _progress_cb(progress: int):
                self.progress.emit(progress)

            if not os.path.exists(self.file_path):
                logger.error("文件不存在: %s", self.file_path)
                self.error.emit("文件不存在或已被删除")
                return

            ext = os.path.splitext(self.file_path)[1].lower()
            if ext in (".xlsx", ".xlsm"):
                from src.data.excel_loader import ExcelDataLoader
                loader = ExcelDataLoader(
                    self.file_path,
                    sheet_name=self.sheet_name or 0,
                    desc_rows=self.desc_rows,
                    has_unit=self.has_unit,
                    _progress=_progress_cb,
                )
            elif ext in (".mf4", ".mdf", ".dat"):
                from src.data.mdf_lazy_loader import MDFLazyLoader

                loader = MDFLazyLoader(self.file_path, _progress=_progress_cb)
            else:
                loader = FastDataLoader(
                    self.file_path,
                    desc_rows=self.desc_rows,
                    sep=self.sep,
                    has_unit=self.has_unit,
                    encoding=self.encoding,
                    chunksize=3600,
                    _progress=_progress_cb,
                )
            self.finished.emit(loader)
        except MemoryError:
            logger.critical("DataLoadThread 内存不足: %s", self.file_path)
            self.error.emit("内存不足，无法加载此文件。请尝试加载较小的文件。")
        except OSError as e:
            logger.error("DataLoadThread 文件访问错误: %s", e)
            self.error.emit(f"文件访问错误: {e}")
        except Exception as e:
            logger.error("DataLoadThread 加载文件失败: %s", e, exc_info=True)
            self.error.emit(f"加载文件时发生未知错误: {str(e)}")


class FastDataLoader(BaseDataLoader):
    """
    快速数据加载器类
    高效加载和处理大型CSV文件，支持分块读取、数据类型推断、编码检测等功能
    专门为大数据文件优化，提供进度回调和内存管理
    """

    LOADER_TYPE = "csv"

    @staticmethod
    def _detect_sep_from_lines(lines: list[str]) -> str | None:
        """从已读取的行列表中检测分隔符（方差法，无 I/O）

        借鉴 Excel "分列"功能原理：统计候选分隔符在各行出现的次数，
        选择出现次数最一致（方差最小）且出现次数>0的作为分隔符。

        Args:
            lines: 已读取的非空行列表

        Returns:
            检测到的分隔符，无法检测时返回 None
        """
        candidates = [",", ";", "\t"]
        counts = {c: [] for c in candidates}
        for line in lines:
            for c in candidates:
                counts[c].append(line.count(c))
        scores = {}
        for c in candidates:
            vals = counts[c]
            if not vals or all(v == 0 for v in vals):
                scores[c] = float("inf")
                continue
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            scores[c] = variance
        best = min(scores, key=lambda c: (scores[c], {";": 1, "\t": 2}.get(c, 0)))
        if scores[best] == float("inf"):
            return None
        return best

    @staticmethod
    def _detect_header_from_lines(lines: list[str], sep: str) -> int:
        """从已读取的行列表中定位标题行（非数值占比法，无 I/O）

        扫描行列表，找到第一个包含分隔符且非数值占比 > 50% 的行作为标题行。

        Args:
            lines: 已读取的非空行列表
            sep: 已检测到的分隔符

        Returns:
            标题行在 lines 列表中的索引 (0-based)，未找到时返回 0
        """
        for idx, line in enumerate(lines):
            if sep not in line:
                continue
            parts = line.split(sep)
            if len(parts) < 2:
                continue
            non_numeric_count = 0
            for cell in parts:
                cell_stripped = cell.strip().strip('"').strip("'")
                if not cell_stripped:
                    continue
                try:
                    float(cell_stripped)
                except (ValueError, TypeError):
                    non_numeric_count += 1
            total = len(parts)
            if total > 0 and non_numeric_count / total > 0.5:
                return idx
        return 0

    @staticmethod
    def _detect_has_unit_from_lines(
        lines: list[str], sep: str, header_row: int
    ) -> bool:
        """从已读取的行列表中判断标题行下一行是否为单位行（无 I/O）

        检查 header_row+1 行，统计其中包含 _UNIT_KEYWORDS 中关键字的列比例。
        比例超过 UNIT_KEYWORD_RATIO_THRESHOLD 时判定为单位行。

        Args:
            lines: 已读取的非空行列表
            sep: 分隔符
            header_row: 标题行在 lines 中的索引

        Returns:
            True 表示包含单位行
        """
        if header_row + 1 >= len(lines):
            return False
        parts = lines[header_row + 1].split(sep)
        if len(parts) < 2:
            return False
        unit_hit = 0
        total = 0
        for cell in parts:
            cell_stripped = cell.strip().strip('"').strip("'")
            cell_lower = cell_stripped.lower()
            if not cell_stripped:
                continue
            total += 1
            for keyword in _UNIT_KEYWORDS:
                if keyword.lower() in cell_lower:
                    unit_hit += 1
                    break
        if total == 0:
            return False
        return (unit_hit / total) > UNIT_KEYWORD_RATIO_THRESHOLD

    @staticmethod
    def _auto_detect_format(file_path: str) -> FormatInfo:
        """一次文件扫描完成编码探测 + 分隔符检测 + 标题行定位 + 单位行探测

        1. 用 charset_normalizer 读取文件前 2000 字节推断编码
        2. 根据置信度选择回退策略（高置信度精简链、低置信度完整链）
        3. 用正确编码打开文件，一次读取前 50 行
        4. 在同一批行数据上依次检测分隔符→标题行→单位行

        Args:
            file_path: 文件路径

        Returns:
            FormatInfo(encoding, sep, header_row, has_unit)

        Raises:
            AutoDetectError: 文件内容不足（< 2 行）无法检测
        """
        from charset_normalizer import from_bytes

        sample_size = 2000
        with Path(file_path).open("rb") as f:
            raw_sample = f.read(sample_size)

        result = from_bytes(raw_sample).best()
        if result and result.encoding:
            detected_encoding = result.encoding
            coherence = result.coherence
        else:
            detected_encoding = "utf-8"
            coherence = 0.0

        if coherence > 0.8:
            encodings_to_try = list(dict.fromkeys([detected_encoding, "utf-8"]))
        else:
            encodings_to_try = list(
                dict.fromkeys(
                    [detected_encoding, "utf-8", "gb18030", "cp1252", "latin-1"]
                )
            )

        lines = []
        final_encoding = None
        for enc in encodings_to_try:
            try:
                with open(file_path, "r", encoding=enc, errors="replace") as f:
                    for _ in range(50):
                        line = f.readline()
                        if not line:
                            break
                        stripped = line.rstrip("\n\r")
                        if stripped.strip():
                            lines.append(stripped)
                        if len(lines) >= 40:
                            break
                final_encoding = enc
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if final_encoding is None:
            return FormatInfo(encoding=None, sep=None, header_row=0, has_unit=False)

        if len(lines) < 2:
            raise AutoDetectError(
                f"文件内容不足（仅 {len(lines)} 行），无法自动检测格式: {file_path}"
            )

        sep = FastDataLoader._detect_sep_from_lines(lines)
        if sep is not None:
            header_row = FastDataLoader._detect_header_from_lines(lines, sep)
            has_unit = FastDataLoader._detect_has_unit_from_lines(
                lines, sep, header_row
            )
        else:
            header_row = 0
            has_unit = False

        return FormatInfo(
            encoding=final_encoding, sep=sep, header_row=header_row, has_unit=has_unit
        )

    @staticmethod
    def auto_detect(file_path: str) -> FormatInfo:
        """自动检测文件格式的统一入口

        调用 _auto_detect_format() 完成一次文件扫描，返回 FormatInfo。

        Args:
            file_path: 文件路径

        Returns:
            FormatInfo(encoding, sep, header_row, has_unit)

        Raises:
            AutoDetectError: 无法自动检测文件分隔符或文件内容不足
        """
        fmt = FastDataLoader._auto_detect_format(file_path)
        logger.debug(
            "auto_detect: enc=%s, sep=%s, header=%d, unit=%s",
            fmt.encoding, fmt.sep, fmt.header_row, fmt.has_unit,
        )
        if fmt.sep is None:
            raise AutoDetectError(f"无法自动检测文件分隔符: {file_path}")
        return fmt

    def __init__(
        self,
        csv_path: str,
        *,
        max_rows_infer: int = 200,
        chunksize: int | None = None,
        usecols: list[str] | None = None,
        drop_empty: bool = False,
        downcast_float: bool = True,
        desc_rows: int = 0,
        sep: str = ",",
        _progress: Callable | None = None,
        do_parse_date: bool = False,
        has_unit: bool = True,
        encoding: str | None = None,
    ):
        """初始化快速数据加载器

        配置数据加载参数，包括文件路径、数据类型推断、分块大小等。
        当 encoding 不为 None 时跳过编码自动检测直接使用传入编码。

        Args:
            csv_path: CSV文件路径
            max_rows_infer: 用于推断数据类型的最大行数
            chunksize: 分块读取大小
            usecols: 要读取的列名列表
            drop_empty: 是否删除空行
            downcast_float: 是否下转换浮点数类型
            desc_rows: 描述行数量
            sep: 分隔符
            _progress: 进度回调函数
            do_parse_date: 是否解析日期
            has_unit: 是否包含单位行
            encoding: 预检测的文件编码，为 None 时内部自动检测
        """
        super().__init__()
        self._path = csv_path
        self.file_size = os.path.getsize(csv_path)
        logger.info(
            "开始加载 CSV: %s (%.1f MB, sep=%s, desc=%d, has_unit=%s)",
            csv_path, self.file_size / 1024 / 1024, sep, desc_rows, has_unit,
        )
        self.max_rows_infer = max_rows_infer
        self.usecols = usecols
        self.drop_empty = drop_empty
        self.downcast_float = downcast_float
        self.sep = sep
        self.desc_rows = desc_rows
        self._progress_cb = _progress
        self.do_parse_date = do_parse_date
        self.has_unit = has_unit

        self._var_names, self._units, self.encoding_used, self.has_unit = (
            self._load_header_units(
                self._path,
                desc_rows=self.desc_rows,
                usecols=self.usecols,
                sep=self.sep,
                has_unit=self.has_unit,
                encoding=encoding,
            )
        )

        if self._progress_cb:
            self._progress_cb(5)

        # 推断 dtype
        sample = pd.read_csv(
            self._path,
            skiprows=(2 + self.desc_rows) if self.has_unit else (1 + self.desc_rows),
            nrows=self.max_rows_infer,
            names=self._var_names,
            encoding=self.encoding_used,
            usecols=self.usecols,
            low_memory=False,
            sep=self.sep,
            na_values=self._NA_VALUES,
            keep_default_na=True,
        )

        # 推断schema（包含时间格式）
        dtype_map, parse_dates, date_formats, downcast_ratio = self._infer_schema(
            sample
        )
        self.date_formats = date_formats

        self.sample_mem_size = sample.memory_usage(deep=True).sum()
        self.byte_per_line = (0.6 * self.sample_mem_size) / sample.shape[0]
        self.estimated_lines = int(self.file_size / (self.byte_per_line))

        del sample
        gc.collect()
        if self._progress_cb:
            self._progress_cb(15)

        # 计算 chunk 大小
        if chunksize is None:
            chunksize = 3600

        # 正式读取数据
        self._df = self._read_chunks(
            self._path,
            dtype_map,
            parse_dates,
            int(chunksize),
            sep=self.sep,
            desc_rows=self.desc_rows,
            has_unit=self.has_unit,
        )

        # 后处理
        if drop_empty:
            self._df = self._df.dropna(axis=1, how="all")
        if downcast_float:
            self._downcast_numeric()

        self._df_validity = self._check_df_validity()

        # 强制垃圾回收
        gc.collect()

        if self._progress_cb:
            self._progress_cb(100)

        logger.info(
            "CSV 加载完成: %s (%d 行, %d 列)",
            csv_path, len(self._df), len(self._var_names),
        )

    @staticmethod
    def _load_header_units(
        path: str,
        desc_rows: int = 0,
        usecols: list[str] | None = None,
        sep: str = ",",
        has_unit: bool = True,
        encoding: str | None = None,
    ) -> tuple[list[str], dict[str, str], str, bool]:
        """加载CSV文件的表头和单位信息

        根据传入编码或自动检测编码读取文件头部，提取变量名和单位信息。
        当 encoding 不为 None 时跳过编码检测直接使用传入值（保留回退链兜底）。
        单位行检测仅作为对照校验输出 debug 日志，不再覆盖传入的 has_unit。

        Args:
            path: CSV文件路径
            desc_rows: 描述行数量
            usecols: 要读取的列名列表
            sep: 分隔符
            has_unit: 是否包含单位行
            encoding: 预检测的文件编码，为 None 时内部自动检测

        Returns:
            tuple: (变量名列表, {变量名: 单位}, 最终编码, 实际使用的has_unit)
        """
        nrows_read = 5 if has_unit else 1

        if encoding is not None:
            # 上游已检测编码，直接使用并追加回退链兜底
            encodings_to_try = list(dict.fromkeys([encoding, "utf-8", "cp1252"]))
        else:
            # 自行检测编码
            from charset_normalizer import from_bytes

            sample_size = 2000

            with Path(path).open("rb") as f:
                raw_sample = f.read(sample_size)

            result = from_bytes(raw_sample).best()
            if result and result.encoding:
                detected_enc = result.encoding
            else:
                detected_enc = "utf-8"

            encodings_to_try = list(dict.fromkeys([detected_enc, "utf-8", "cp1252"]))

        for enc in encodings_to_try:
            try:
                df = pd.read_csv(
                    path,
                    skiprows=desc_rows,
                    nrows=nrows_read,
                    header=None,
                    usecols=usecols,
                    sep=sep,
                    encoding=enc,
                    engine="python",
                )
                break
            except UnicodeDecodeError:
                continue
        else:
            raise RuntimeError("无法以任何可用编码读取文件")

        # 单位行对照校验（不覆盖传入的 has_unit，仅输出 debug 日志）
        actual_has_unit = has_unit
        if has_unit and df.shape[0] >= 2:
            row2 = df.iloc[1].fillna("").astype(str).tolist()

            unit_keyword_count = 0
            total_cols = len(row2)

            for cell in row2:
                cell_lower = str(cell).lower().strip()
                for keyword in _UNIT_KEYWORDS:
                    keyword_lower = keyword.lower()
                    if keyword_lower in cell_lower:
                        unit_keyword_count += 1
                        break

            unit_keyword_ratio = (
                unit_keyword_count / total_cols if total_cols > 0 else 0
            )

            if unit_keyword_ratio > UNIT_KEYWORD_RATIO_THRESHOLD:
                pass
            else:
                numeric_count = 0
                valid_numeric_count = 0
                total_cols = len(row2)

                for cell in row2:
                    cell_str = str(cell).strip()
                    try:
                        val = float(cell_str)
                        numeric_count += 1
                        if abs(val - 1.0) > 1e-9 and abs(val + 1.0) > 1e-9:
                            valid_numeric_count += 1
                    except (ValueError, TypeError):
                        continue

                if total_cols > 0:
                    valid_numeric_ratio = valid_numeric_count / total_cols
                    if valid_numeric_ratio > VALID_NUMERIC_RATIO_THRESHOLD:
                        pass
                    else:
                        pass

        min_required_rows = 2 if actual_has_unit else 1
        if df.shape[0] < min_required_rows:
            raise ValueError(f"文件至少需要{min_required_rows}行")

        var_names = df.iloc[0].astype(str).tolist()
        var_names = FastDataLoader._make_unique(var_names)
        if actual_has_unit:
            units = dict(zip(var_names, df.iloc[1].fillna("").astype(str).tolist()))
        else:
            units = dict(zip(var_names, ["-"] * len(var_names)))
        return var_names, units, enc, actual_has_unit

    @staticmethod
    def _infer_schema(
        sample: pd.DataFrame,
    ) -> tuple[dict[str, str], list[str], dict[str, str], float]:
        """推断数据类型和时间格式

        优化策略：
        1. 只对列名包含时间相关关键字的列进行时间格式推断
        2. 使用更精简的日期格式候选列表
        3. 对每列只采样前10行进行格式推断
        """
        dtype_map: dict[str, str] = {}
        parse_dates: list[str] = []
        date_formats: dict[str, str] = {}

        # 日期格式候选列表（按优先级排序）
        date_candidates = [
            "%H:%M:%S.%f",  # 带微秒的时间格式（支持毫秒和微秒）
            "%H:%M:%S",  # 时间格式
            "%d/%m/%Y",  # 欧洲日期格式 (例: 18/11/2017)
            "%Y/%m/%d",  # 日期格式 (例: 2024/10/31)
            "%Y-%m-%d",  # ISO日期格式 (例: 2024-10-31)
        ]

        # 时间列的关键字（不区分大小写）
        time_keywords = ["time", "date", "datetime", "timestamp", "zeit", "tmod"]

        float_cols = sample.select_dtypes(include=["float", "float64", "category"])
        downcast_ratio_est = (
            float_cols.shape[1] / sample.shape[1] if sample.shape[1] > 0 else 0.000001
        )

        for col in sample.columns:
            s = sample[col]
            if s.isna().all():
                dtype_map[col] = "category"
                continue

            # 只对列名包含时间关键字的列进行时间格式推断
            col_lower = col.lower()
            is_time_candidate = any(keyword in col_lower for keyword in time_keywords)

            if is_time_candidate:
                # 采样前10行进行格式推断
                s_sample = s.head(10).dropna()
                if len(s_sample) > 0:
                    for fmt in date_candidates:
                        try:
                            pd.to_datetime(s_sample, format=fmt, errors="raise")
                            parse_dates.append(col)
                            date_formats[col] = fmt
                            break
                        except (ValueError, TypeError):
                            continue
                    else:
                        # 不是时间格式，按数值处理
                        if pd.api.types.is_numeric_dtype(s):
                            is_safe, _ = _evaluate_float32_safety(s)
                            dtype_map[col] = "float32" if is_safe else "float64"
                        else:
                            dtype_map[col] = "category"
                else:
                    dtype_map[col] = "category"
            else:
                # 非时间列，直接判断数值类型
                if pd.api.types.is_numeric_dtype(s):
                    is_safe, _ = _evaluate_float32_safety(s)
                    dtype_map[col] = "float32" if is_safe else "float64"
                else:
                    dtype_map[col] = "category"

        return dtype_map, parse_dates, date_formats, downcast_ratio_est

    def _read_chunks(
        self,
        path: str,
        dtype_map,
        parse_dates: list[str],
        chunksize: int,
        sep: None | str = ",",
        desc_rows: int = 0,
        has_unit: bool = True,
    ) -> pd.DataFrame:
        chunks: list[pd.DataFrame] = []
        # do not parse date
        if not self.do_parse_date:
            parse_dates = []
        total_chunks_est = max(
            1,
            self.estimated_lines // chunksize
            + (1 if self.estimated_lines % chunksize else 0),
        )
        increment = 80 / total_chunks_est

        # 使用更小的chunk size来减少内存峰值
        optimized_chunksize = min(chunksize, 2000)  # 限制最大chunk size

        for idx, chunk in enumerate(
            pd.read_csv(
                path,
                skiprows=(2 + desc_rows) if has_unit else (1 + desc_rows),
                names=self._var_names,
                dtype=dtype_map,
                parse_dates=parse_dates,
                encoding=self.encoding_used,
                chunksize=optimized_chunksize,
                usecols=self.usecols,
                low_memory=False,
                memory_map=True,
                sep=sep,
                na_values=self._NA_VALUES,
                keep_default_na=True,
                on_bad_lines="skip",
            )
        ):
            # print(f"chunksize is {chunksize}, full size {self.file_size/(1024**2):2f}Mb")
            if self._progress_cb:
                chunk_progress = min(80, (idx + 1) * increment)
                self._progress_cb(15 + int(chunk_progress))
                # print (f"progress {idx} is {bytes_read}")
            chunks.append(chunk)

            # 每处理几个chunk就进行一次垃圾回收
            if idx % 5 == 0:
                import gc

                gc.collect()
        return pd.concat(chunks, ignore_index=True)
