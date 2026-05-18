"""loader"""
from __future__ import annotations
import os, gc
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread,pyqtSignal
from src.core.config import DEBUG_LOG_ENABLED,debug_log,_UNIT_KEYWORDS,UNIT_KEYWORD_RATIO_THRESHOLD,VALID_NUMERIC_RATIO_THRESHOLD,_evaluate_float32_safety

class DataLoadThread(QThread):
    """
    数据加载线程类
    在后台线程中异步加载CSV数据文件，避免阻塞主界面
    通过信号机制向主线程发送加载进度和结果
    """
    # 信号：发送进度 0-100，或直接发 DataFrame
    progress = pyqtSignal(int)        # 百分比
    finished = pyqtSignal(object)     # FastDataLoader 实例
    error = pyqtSignal(str)

    def __init__(self, file_path: str, parent=None, descRows: int = 0, sep: str = ',',
                 hasunit: bool = True, encoding: str | None = None):
        super().__init__(parent)
        self.file_path = file_path
        self.descRows = descRows
        self.sep = sep
        self.hasunit = hasunit
        self.encoding = encoding
    def run(self):
        """
        线程执行方法
        在后台线程中执行数据加载操作，避免阻塞主界面
        通过信号机制向主线程发送进度更新和结果
        """
        debug_log("DataLoadThread.start path=%s descRows=%s sep=%s hasunit=%s",
                  self.file_path, self.descRows, self.sep, self.hasunit)
        try:
            last_logged = {"value": -10}
            def _progress_cb(progress: int):
                if DEBUG_LOG_ENABLED:
                    prev = last_logged["value"]
                    if progress in (0, 100) or progress - prev >= 10:
                        debug_log("DataLoadThread.progress path=%s value=%s",
                                  self.file_path, progress)
                        last_logged["value"] = progress
                self.progress.emit(progress)

            if not os.path.exists(self.file_path):
                self.error.emit("文件不存在或已被删除")
                return

            ext = os.path.splitext(self.file_path)[1].lower()
            if ext in ('.mf4', '.mdf', '.dat'):
                from mdf_loader import MDFDataLoader
                loader = MDFDataLoader(self.file_path, _progress=_progress_cb)
            else:
                loader = FastDataLoader(
                    self.file_path,
                    descRows=self.descRows,
                    sep=self.sep,
                    hasunit=self.hasunit,
                    encoding=self.encoding,
                    chunksize=3600,
                    _progress=_progress_cb,
                )
            debug_log("DataLoadThread.finish path=%s datalength=%s columns=%s",
                      self.file_path,
                      getattr(loader, "datalength", None),
                      len(getattr(loader, "var_names", []) or []))
            self.finished.emit(loader)
        except MemoryError:
            debug_log("DataLoadThread.memory_error path=%s", self.file_path)
            self.error.emit("内存不足，无法加载此文件。请尝试加载较小的文件。")
        except OSError as e:
            debug_log("DataLoadThread.os_error path=%s err=%s", self.file_path, e)
            self.error.emit(f"文件访问错误: {e}")
        except Exception as e:
            debug_log("DataLoadThread.exception path=%s err=%r", self.file_path, e)
            self.error.emit(f"加载文件时发生未知错误: {str(e)}")

class FastDataLoader:
    """
    快速数据加载器类
    高效加载和处理大型CSV文件，支持分块读取、数据类型推断、编码检测等功能
    专门为大数据文件优化，提供进度回调和内存管理
    """
    # 脏数据清单
    _NA_VALUES = [
        # 空缺 / 空字符串
        "", "NULL", "None", "NA", "N/A", "n/a", "null ",
        # Excel / 统计缺失标记
        "#N/A", "#N/A N/A", "#NA",
        # IEEE NaN / 非法数值
        "NaN", "nan", "-NaN", "-nan", "1.#IND", "-1.#IND", "1.#QNAN", "-1.#QNAN",
        # 无穷值表示
        "Infinity", "Inf", "inf", "plus infinity", "minus infinity", "1.#INF", "-1.#INF",
        # 其他脏数据字符串
        "data err", "* *", "----", "no value"
    ]

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
        candidates = [',', ';', '\t']
        counts = {c: [] for c in candidates}
        for line in lines:
            for c in candidates:
                counts[c].append(line.count(c))
        scores = {}
        for c in candidates:
            vals = counts[c]
            if not vals or all(v == 0 for v in vals):
                scores[c] = float('inf')
                continue
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            scores[c] = variance
        best = min(scores, key=lambda c: (scores[c], {';': 1, '\t': 2}.get(c, 0)))
        if scores[best] == float('inf'):
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
    def _detect_hasunit_from_lines(lines: list[str], sep: str, header_row: int) -> bool:
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
            FormatInfo(encoding, sep, header_row, hasunit)
            
        Raises:
            AutoDetectError: 文件内容不足（< 2 行）无法检测
        """
        from charset_normalizer import from_bytes
        
        sample_size = 2000
        with Path(file_path).open('rb') as f:
            raw_sample = f.read(sample_size)

        result = from_bytes(raw_sample).best()
        if result and result.encoding:
            detected_encoding = result.encoding
            coherence = result.coherence
        else:
            detected_encoding = 'utf-8'
            coherence = 0.0

        if coherence > 0.8:
            encodings_to_try = list(dict.fromkeys([detected_encoding, 'utf-8']))
        else:
            encodings_to_try = list(dict.fromkeys(
                [detected_encoding, 'utf-8', 'gb18030', 'cp1252', 'latin-1']
            ))

        lines = []
        final_encoding = None
        for enc in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=enc, errors='replace') as f:
                    for _ in range(50):
                        line = f.readline()
                        if not line:
                            break
                        stripped = line.rstrip('\n\r')
                        if stripped.strip():
                            lines.append(stripped)
                        if len(lines) >= 40:
                            break
                final_encoding = enc
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if final_encoding is None:
            return FormatInfo(encoding=None, sep=None, header_row=0, hasunit=False)

        if len(lines) < 2:
            raise AutoDetectError(f"文件内容不足（仅 {len(lines)} 行），无法自动检测格式: {file_path}")

        sep = FastDataLoader._detect_sep_from_lines(lines)
        if sep is not None:
            header_row = FastDataLoader._detect_header_from_lines(lines, sep)
            hasunit = FastDataLoader._detect_hasunit_from_lines(lines, sep, header_row)
        else:
            header_row = 0
            hasunit = False

        return FormatInfo(encoding=final_encoding, sep=sep, header_row=header_row, hasunit=hasunit)

    @staticmethod
    def auto_detect(file_path: str) -> FormatInfo:
        """自动检测文件格式的统一入口
        
        调用 _auto_detect_format() 完成一次文件扫描，返回 FormatInfo。
        
        Args:
            file_path: 文件路径
            
        Returns:
            FormatInfo(encoding, sep, header_row, hasunit)
            
        Raises:
            AutoDetectError: 无法自动检测文件分隔符或文件内容不足
        """
        fmt = FastDataLoader._auto_detect_format(file_path)
        if fmt.sep is None:
            raise AutoDetectError(f"无法自动检测文件分隔符: {file_path}")
        return fmt

    from typing import Callable
    def __init__(
        self,
        csv_path: str ,
        *,
        max_rows_infer: int = 200,
        chunksize: int | None = None,
        usecols: list[str] | None = None,
        drop_empty: bool = False,
        downcast_float: bool = True,
        descRows: int = 0,
        sep: str = ",",
        _progress: Callable | None = None,
        do_parse_date: bool =False,
        hasunit:bool = True,
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
            descRows: 描述行数量
            sep: 分隔符
            _progress: 进度回调函数
            do_parse_date: 是否解析日期
            hasunit: 是否包含单位行
            encoding: 预检测的文件编码，为 None 时内部自动检测
        """
        self._path = csv_path
        self.file_size = os.path.getsize(csv_path) 
        self.max_rows_infer = max_rows_infer
        self.usecols = usecols
        self.drop_empty = drop_empty
        self.downcast_float = downcast_float
        self.sep = sep
        self.descRows = descRows
        self._progress_cb = _progress
        self.do_parse_date=do_parse_date
        self.hasunit=hasunit

        self.time_column_name: str | None = None

        self._var_names, self._units, self.encoding_used, self.hasunit = self._load_header_units(
            self._path, desc_rows=self.descRows, usecols=self.usecols, sep=self.sep,
            hasunit=self.hasunit, encoding=encoding
        )
        
        if self._progress_cb:
            self._progress_cb(5)

        # 推断 dtype
        sample = pd.read_csv(
            self._path,
            skiprows=(2 + self.descRows) if self.hasunit else (1+self.descRows),
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
        dtype_map, parse_dates, date_formats,downcast_ratio = self._infer_schema(sample)
        self.date_formats = date_formats
        
        self.sample_mem_size = sample.memory_usage(deep=True).sum()
        self.byte_per_line = (0.6*self.sample_mem_size)/sample.shape[0]
        self.estimated_lines = int(self.file_size/(self.byte_per_line ))
        
        import gc
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
            descRows=self.descRows,
            hasunit=self.hasunit
        )
        
        # 后处理
        if drop_empty:
            self._df = self._df.dropna(axis=1, how="all")
        if downcast_float:
            self._downcast_numeric()
        
        self._df_validity=self._check_df_validity()
        
        # 强制垃圾回收
        gc.collect()
        
        if self._progress_cb:
            self._progress_cb(100)

    @staticmethod
    def _load_header_units(
        path: str,
        desc_rows: int = 0,
        usecols: list[str] | None = None,
        sep: str = ",",
        hasunit: bool = True,
        encoding: str | None = None,
    ) -> tuple[list[str], dict[str, str], str, bool]:
        """加载CSV文件的表头和单位信息
        
        根据传入编码或自动检测编码读取文件头部，提取变量名和单位信息。
        当 encoding 不为 None 时跳过编码检测直接使用传入值（保留回退链兜底）。
        单位行检测仅作为对照校验输出 debug 日志，不再覆盖传入的 hasunit。
        
        Args:
            path: CSV文件路径
            desc_rows: 描述行数量
            usecols: 要读取的列名列表
            sep: 分隔符
            hasunit: 是否包含单位行
            encoding: 预检测的文件编码，为 None 时内部自动检测
            
        Returns:
            tuple: (变量名列表, {变量名: 单位}, 最终编码, 实际使用的hasunit)
        """
        nrows_read = 5 if hasunit else 1

        if encoding is not None:
            # 上游已检测编码，直接使用并追加回退链兜底
            encodings_to_try = list(dict.fromkeys([encoding, 'utf-8', 'cp1252']))
        else:
            # 自行检测编码
            from charset_normalizer import from_bytes

            sample_size = 2000

            with Path(path).open('rb') as f:
                raw_sample = f.read(sample_size)

            result = from_bytes(raw_sample).best()
            if result and result.encoding:
                detected_enc = result.encoding
            else:
                detected_enc = 'utf-8'

            encodings_to_try = list(dict.fromkeys([detected_enc, 'utf-8', 'cp1252']))

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

        # 单位行对照校验（不覆盖传入的 hasunit，仅输出 debug 日志）
        actual_hasunit = hasunit
        if hasunit and df.shape[0] >= 2:
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

            unit_keyword_ratio = unit_keyword_count / total_cols if total_cols > 0 else 0

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
                        debug_log(
                            "_load_header_units: 上游检测 hasunit=True，但本行单位关键字占比仅 %.1f%%、"
                            "有效数值列比例 %.1f%%，疑似为数据行。检查文件: %s",
                            unit_keyword_ratio * 100, valid_numeric_ratio * 100, path
                        )
                    else:
                        debug_log(
                            "_load_header_units: 上游检测 hasunit=True，但本行单位关键字占比仅 %.1f%%，"
                            "疑似误判。检查文件: %s",
                            unit_keyword_ratio * 100, path
                        )

        min_required_rows = 2 if actual_hasunit else 1
        if df.shape[0] < min_required_rows:
            raise ValueError(f"文件至少需要{min_required_rows}行")

        var_names = df.iloc[0].astype(str).tolist()
        var_names = FastDataLoader._make_unique(var_names)
        if actual_hasunit:
            units = dict(zip(var_names, df.iloc[1].fillna("").astype(str).tolist()))
        else:
            units = dict(zip(var_names, ['-'] * len(var_names)))
        return var_names, units, enc, actual_hasunit

    @staticmethod
    def _infer_schema(sample: pd.DataFrame) -> tuple[dict[str, str], list[str], dict[str, str],float]:
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
            "%H:%M:%S.%f",   # 带微秒的时间格式（支持毫秒和微秒）
            "%H:%M:%S",      # 时间格式
            "%d/%m/%Y",      # 欧洲日期格式 (例: 18/11/2017)
            "%Y/%m/%d",      # 日期格式 (例: 2024/10/31)
            "%Y-%m-%d",      # ISO日期格式 (例: 2024-10-31)
        ]
        
        # 时间列的关键字（不区分大小写）
        time_keywords = ['time', 'date', 'datetime', 'timestamp', 'zeit', 'tmod']
        
        float_cols = sample.select_dtypes(include=['float', 'float64','category'])
        downcast_ratio_est = float_cols.shape[1] / sample.shape[1] if sample.shape[1] > 0 else 0.000001
        
        # 【NumPy优化】批量识别numeric列和非numeric列（用于后续优化）
        numeric_cols = sample.select_dtypes(include=['float32', 'float64', 'int', 'int32', 'int64']).columns
        non_numeric_cols = [col for col in sample.columns if col not in numeric_cols]
        
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
        
        return dtype_map, parse_dates, date_formats,downcast_ratio_est

    def _read_chunks(
        self,
        path: str,
        dtype_map,
        parse_dates: list[str],
        chunksize: int,
        sep: None | str = ",",
        descRows: int = 0,
        hasunit:bool = True,
    ) -> pd.DataFrame:
        chunks: list[pd.DataFrame] = []
        # do not parse date
        if not self.do_parse_date:
            parse_dates=[]
        total_chunks_est = max(1, self.estimated_lines // chunksize + (1 if self.estimated_lines % chunksize else 0))
        increment = 80 / total_chunks_est
        
        # 使用更小的chunk size来减少内存峰值
        optimized_chunksize = min(chunksize, 2000)  # 限制最大chunk size
        
        for idx,chunk in enumerate(pd.read_csv(
            path,
            skiprows=(2 + descRows) if hasunit else (1+descRows),
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
            on_bad_lines='skip'
        )):
            #print(f"chunksize is {chunksize}, full size {self.file_size/(1024**2):2f}Mb")
            if self._progress_cb:
                chunk_progress = min(80, (idx + 1) * increment)
                self._progress_cb(15 + int(chunk_progress))
                #print (f"progress {idx} is {bytes_read}")
            chunks.append(chunk)
            
            # 每处理几个chunk就进行一次垃圾回收
            if idx % 5 == 0:
                import gc
                gc.collect()
        return pd.concat(chunks, ignore_index=True)

    def _downcast_numeric(self) -> None:
        float_cols = self._df.select_dtypes(include=["float32", "float64"]).columns
        for col in float_cols:
            cleaned = pd.to_numeric(self._df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            is_safe, _ = _evaluate_float32_safety(cleaned)
            if is_safe:
                self._df[col] = cleaned.astype("float32")
            else:
                # 当 float32 会溢出时改用 float64 保留精度
                self._df[col] = cleaned.astype("float64", copy=False)


    def _check_df_validity(self) -> dict:
        validity : dict = {}
        for col in self._df.columns:
            # 传入列名和date_formats参数
            validity[col] = self._classify_column(self._df[col], col, self.date_formats)
        
        return validity

    @staticmethod
    def _make_unique(names: list[str]) -> list[str]:
        seen = {}
        unique_names = []
        for name in names:
            if name in seen:
                seen[name] += 1
                new_name = f"{name}_{seen[name]}"
            else:
                seen[name] = 0
                new_name = name
            unique_names.append(new_name)
        return unique_names
    
    @staticmethod
    def _classify_column(series: pd.Series, col_name: str, date_formats: dict) -> int:
        """
        1: （全部可转数字，且 ≥2 个不同有效值） 或 （该列是日期格式） 或 （数据长度为1，且可转换为数字）
        0: 数据长度>=2, 且全部可转数字，且唯一有效值
        -1: 存在非数字（不含日期格式） 或 全部 NaN
        
        【NumPy优化】使用NumPy直接检查唯一值，避免Pandas的循环操作
        """
        # 如果该列是日期格式，则直接返回1（有效）   
        if col_name in date_formats:
            return 1

        # 1) 先尝试整列转 float，失败直接 -1
        try:
            numeric = pd.to_numeric(series, errors="raise").values  # 转为NumPy array
        except (ValueError, TypeError):
            return -1

        # 2) 【NumPy优化】用NumPy过滤NaN（兼容整数类型）
        # 先转换为浮点类型以支持NaN检查，避免整数类型的NaN检查错误
        if numeric.dtype.kind in 'iu':  # 整数类型
            # 整数类型没有NaN，直接使用
            valid = numeric
        else:
            # 浮点类型，需要过滤NaN
            valid = numeric[~np.isnan(numeric)]
        
        if len(valid) == 0:          # 全 NaN 或空数组
            return -1

        # 数据长度为1且可转数字 → 返回1
        if len(series) == 1:
            return 1

        # 【NumPy优化】用np.unique直接计算唯一值数量，比Pandas更快
        unique_count = np.unique(valid).size
        if unique_count == 1:
            return 0
        else:
            return 1
    
    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def units(self) -> dict[str, str]:
        return self._units
    
    @property
    def path(self) -> str:
        return str(self._path)

    @property
    def datalength(self) -> int:
        return self._df.shape[0]

    @property
    def default_time_values(self) -> pd.Series:
        return pd.Series(np.arange(1, len(self._df) + 1), name='index')

    @property
    def time_values(self) -> pd.Series:
        if self.time_column_name and self.time_column_name in self._df.columns:
            return self._df[self.time_column_name]
        return self.default_time_values

    @property
    def time_axis_label(self) -> str:
        if self.time_column_name:
            unit = self._units.get(self.time_column_name, '')
            if unit and unit != '-':
                return f"{self.time_column_name} ({unit})"
            return self.time_column_name
        return "Index"

    @property
    def var_names(self) -> list[str]:
        cols = self._df.columns.tolist()
        if self.time_column_name and self.time_column_name in cols:
            cols = [c for c in cols if c != self.time_column_name]
        return cols
    
    @property
    def row_count(self) -> int:
        return len(self._df)
    
    @property
    def column_count(self) -> int:
        return len(self._df.columns)
    
    @property
    def time_channels_info(self) -> dict[str, str]:
        return self.date_formats
    
    @property
    def df_validity(self) -> dict:
        validity = dict(self._df_validity)
        if self.time_column_name and self.time_column_name in validity:
            del validity[self.time_column_name]
        return validity
    
