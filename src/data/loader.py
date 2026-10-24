"""loader"""

from __future__ import annotations
import os
import gc
import time
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal
from src.core.config import (
    _UNIT_KEYWORDS,
    UNIT_KEYWORD_RATIO_THRESHOLD,
    VALID_NUMERIC_RATIO_THRESHOLD,
    CSV_PERF_LOG_THRESHOLD_S,
    _evaluate_float32_safety,
)
from src.core.gc_guard import no_autogc
from src.core.data_types import FormatInfo, AutoDetectError
from src.core.logger import get_logger
from src.data.base_loader import BaseDataLoader

logger = get_logger("data.loader")

# 编码回退链（格式探测与表头/单位读取共用，顺序即优先级）。
# gb18030 覆盖中文 Windows 导出文件；cp1252/latin-1 收尾保证链末位必然可解码。
ENCODING_FALLBACKS: tuple[str, ...] = ("utf-8", "gb18030", "cp1252", "latin-1")


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
    # 用户取消（目前只有 BLF 扫描会走到）：单列一个信号而不是复用 error，
    # 是为了让 UI 能静默恢复原数据而不弹错误框。
    cancelled = Signal(str)  # 被取消的文件路径

    def __init__(
        self,
        file_path: str,
        parent=None,
        desc_rows: int | None = 0,
        sep: str = ",",
        has_unit: bool | None = True,
        encoding: str | None = None,
        sheet_name: str | None = None,
        db_paths: list[str] | None = None,
    ):
        super().__init__(parent)
        self.file_path = file_path
        self.desc_rows = desc_rows
        self.sep = sep
        self.has_unit = has_unit
        self.encoding = encoding
        self.sheet_name = sheet_name
        # BLF 专有：DBC / ARXML 数据库路径列表。其他格式一律忽略，
        # 但 reload 路径会无条件读 ``loader.db_paths``，故统一存下来。
        self.db_paths = list(db_paths) if db_paths else []
        self._cancel_requested = False
        # BLF 扫描的实时计数（{"messages": int, "buckets": int}），供进度框
        # 显示「已读 N 帧」。非 BLF 格式始终为空 dict。
        self.scan_stats: dict = {}

    def request_cancel(self):
        """请求取消加载（目前只对 BLF 扫描生效）。

        刻意不叫 ``cancel()``：``QThread`` 没有 cancel，但本方法要与
        ``InterruptedError`` 抛出后的静默恢复路径配套，命名上显式带
        "request" 强调它只是置标志位，真正的中断发生在下个分块边界。
        """
        self._cancel_requested = True

    def run(self):
        """
        线程执行方法
        在后台线程中执行数据加载操作，避免阻塞主界面
        通过信号机制向主线程发送进度更新和结果
        """
        # 整段加载都在 no_autogc() 窗口内：自动分代收集若落在本 worker 线程，
        # 会把主线程创建的 QObject 包装器拿到这里析构（详见 src/core/gc_guard.py）。
        with no_autogc():
            self._load()

    def _load(self):
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
            elif ext == ".blf":
                from src.data.blf_loader import BLFLazyLoader

                loader = BLFLazyLoader(
                    self.file_path,
                    db_paths=self.db_paths,
                    _progress=_progress_cb,
                    _should_cancel=lambda: self._cancel_requested,
                    _stats=self.scan_stats,
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
                    # 本类是 worker 线程：全量 GC 会把主线程的 QObject 包装器拿到
                    # 本线程析构，与 Shiboken 主线程延迟删除互锁 → 整窗死锁
                    allow_gc=False,
                )
            self.finished.emit(loader)
        except InterruptedError:
            # 用户主动取消：不是错误，不发 error 信号，也不发 finished。
            # 调用方收到 cancelled 后按「已取消」恢复按钮与原数据。
            logger.info("已取消后台加载: %s", self.file_path)
            self.cancelled.emit(self.file_path)
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

    # 列数跳跃检测阈值：当候选标题行列数超过此前候选列数的 N 倍时，
    # 判定前面的行为元数据描述区，当前行为真正的标题行。
    _COLUMN_JUMP_FACTOR = 3

    @staticmethod
    def _detect_header_from_lines(lines: list[str], sep: str) -> int:
        """从已读取的行列表中定位标题行（非数值占比 + 列数跳跃检测）

        CSV 文件在导出时可能带有元数据描述区（如 UniPlot 的 key=value 对），
        这些行通常仅含少量字段（2-3 列），而真实标题行会包含全部变量列。
        因此不采用简单的「首命中即返回」，而是追踪候选行列数：
        当后续候选行的列数发生数量级跳跃（≥ COLUMN_JUMP_FACTOR 倍），
        说明前面的候选在元数据区，跳跃点才是真正标题行。

        Args:
            lines: 已读取的非空行列表
            sep: 已检测到的分隔符

        Returns:
            标题行在 lines 列表中的索引 (0-based)，未找到时返回 0
        """
        candidate_idx = -1
        candidate_cols = 0

        for idx, line in enumerate(lines):
            if sep not in line:
                continue
            parts = line.split(sep)
            col_count = len(parts)
            if col_count < 2:
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
            total = col_count
            if not (total > 0 and non_numeric_count / total > 0.5):
                continue

            # 找到第一个候选
            if candidate_idx == -1:
                candidate_idx = idx
                candidate_cols = col_count
                continue

            # 列数发生数量级跳跃 → 前面的候选在元数据区，当前行才是标题
            if col_count >= candidate_cols * FastDataLoader._COLUMN_JUMP_FACTOR:
                return idx

            # 列数更多但不足够跳跃 → 更新为更优候选
            if col_count > candidate_cols:
                candidate_idx = idx
                candidate_cols = col_count

        return candidate_idx if candidate_idx != -1 else 0

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
        detected_encoding = (
            result.encoding if (result and result.encoding) else "utf-8"
        )
        # 回退链一律用严格解码筛选（见下方 P0-5 注释），因此把候选全部列出
        # 比按 coherence 截断更安全：错误候选会自己失败并让位。
        encodings_to_try = list(dict.fromkeys([detected_encoding, *ENCODING_FALLBACKS]))

        lines: list[str] = []
        physical_lines: list[int] = []
        final_encoding = None
        for enc in encodings_to_try:
            # P0-5: 必须 errors="strict"。errors="replace" 下解码永不抛
            # UnicodeDecodeError → 首轮必然 break，整条回退链沦为死代码，
            # 误判的编码被静默采纳，后续分隔符/标题行/单位行检测全在
            # U+FFFD 乱码上进行。
            try:
                decoded: list[str] = []
                # P1-5: decoded 是**过滤掉空行**的列表，它的下标不等于物理行号。
                # 平行记录每个非空行的物理行号，返回值一律用物理口径 —— 下游
                # `_load_header_units` 用 `skiprows=desc_rows`、正式读取用
                # `skiprows=desc_rows(+1~2)`，pandas 的 skiprows 数的是物理行；
                # 拿过滤下标当物理行号，文件里只要有空行（表头之前的空行很常见）
                # 就会整段错位：少跳一行 → 把单位行读成表头，多跳一行 → 丢首行数据。
                physical_of: list[int] = []
                physical_index = 0
                with open(file_path, "r", encoding=enc, errors="strict") as f:
                    for _ in range(50):
                        line = f.readline()
                        if not line:
                            break
                        stripped = line.rstrip("\n\r")
                        if stripped.strip():
                            decoded.append(stripped)
                            physical_of.append(physical_index)
                        physical_index += 1
                        if len(decoded) >= 40:
                            break
            except (UnicodeDecodeError, UnicodeError, LookupError):
                continue
            lines = decoded
            physical_lines = physical_of
            final_encoding = enc
            break

        if final_encoding is None:
            return FormatInfo(encoding=None, sep=None, header_row=0, has_unit=False)

        if len(lines) < 2:
            raise AutoDetectError(
                f"文件内容不足（仅 {len(lines)} 行），无法自动检测格式: {file_path}"
            )

        sep = FastDataLoader._detect_sep_from_lines(lines)
        if sep is not None:
            # 探测函数一律按「过滤后的行列表」取索引，最后才换算成物理行号
            header_pos = FastDataLoader._detect_header_from_lines(lines, sep)
            has_unit = FastDataLoader._detect_has_unit_from_lines(
                lines, sep, header_pos
            )
        else:
            header_pos = 0
            has_unit = False

        header_row = (
            physical_lines[header_pos]
            if header_pos < len(physical_lines)
            else header_pos
        )
        if header_row != header_pos:
            logger.debug(
                "auto_detect: 表头前有空行，过滤下标 %d → 物理行号 %d",
                header_pos, header_row,
            )

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
        allow_gc: bool = True,
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
            allow_gc: 构造过程是否执行全量 GC。主线程同步加载保持 True；
                worker 线程（DataLoadThread）必须传 False——GC 在哪个线程执行就
                在哪个线程析构对象，在 worker 里全量回收会把主线程的 QObject
                包装器拿到 worker 去销毁，与 Shiboken 主线程延迟删除互锁成死锁
        """
        super().__init__()
        self._path = csv_path
        self.file_size = os.path.getsize(csv_path)
        logger.info(
            "开始加载 CSV: %s (%.1f MB, sep=%s, desc=%d, has_unit=%s)",
            csv_path, self.file_size / 1024 / 1024, sep, desc_rows, has_unit,
        )
        # [PERF] 阶段计时：慢阶段立刻落 WARNING。这样即使中途抛错，日志里也
        # 已经有「跑到哪一步、各花了多久」，不至于只剩首尾两条 INFO 定不了位。
        _perf_started = time.perf_counter()
        _perf_prev = [_perf_started]

        def _perf_mark(stage: str) -> None:
            now = time.perf_counter()
            elapsed = now - _perf_prev[0]
            _perf_prev[0] = now
            if elapsed >= CSV_PERF_LOG_THRESHOLD_S:
                logger.warning(
                    "[PERF] CSV 阶段「%s」耗时 %.3fs（阈值 %.1fs）｜%s",
                    stage, elapsed, CSV_PERF_LOG_THRESHOLD_S,
                    os.path.basename(csv_path),
                )
            else:
                logger.debug("[PERF] CSV 阶段「%s」耗时 %.3fs", stage, elapsed)
        self.max_rows_infer = max_rows_infer
        self.usecols = usecols
        self.drop_empty = drop_empty
        self.downcast_float = downcast_float
        self.sep = sep
        self.desc_rows = desc_rows
        self._progress_cb = _progress
        self.do_parse_date = do_parse_date
        self.has_unit = has_unit
        self._allow_gc = allow_gc

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
        _perf_mark("表头/单位")

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
            # 与 _read_chunks 同口径：坏行跳过而不是整次加载失败。正式读取本来就
            # 会跳过它们，样本这里卡住只会让文件「打不开」而非「少一行」
            on_bad_lines="skip",
        )

        # 推断schema（包含时间格式）
        dtype_map, parse_dates, date_formats, _ = self._infer_schema(
            sample
        )
        self.date_formats = date_formats
        _perf_mark("采样读取+schema 推断")

        del sample
        if self._allow_gc:
            gc.collect()
        if self._progress_cb:
            self._progress_cb(15)
        _perf_mark("采样后 GC")

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
        _perf_mark("正式读取")

        # 后处理：合并 downcast + validity 为单次遍历
        self._df_validity = self._postprocess_columns(downcast=downcast_float)
        _perf_mark("后处理(downcast+validity)")

        # 强制垃圾回收（worker 线程路径由 allow_gc 关闭）
        if self._allow_gc:
            gc.collect()

        if self._progress_cb:
            self._progress_cb(100)
        _perf_mark("收尾 GC")

        logger.info(
            "CSV 加载完成: %s (%d 行, %d 列)｜总耗时 %.3fs",
            csv_path, len(self._df), len(self._var_names),
            time.perf_counter() - _perf_started,
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
            # P0-5: 与探测链共用 ENCODING_FALLBACKS（旧链缺 gb18030，
            # 中文 Windows 文件在两条链上会得出不同结论）
            encodings_to_try = list(dict.fromkeys([encoding, *ENCODING_FALLBACKS]))
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

            encodings_to_try = list(
                dict.fromkeys([detected_enc, *ENCODING_FALLBACKS])
            )

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

        # 单位行对照校验（不修改 actual_has_unit，仅输出诊断日志）
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
                logger.debug(
                    "单位行校验通过: unit_keyword_ratio=%.2f > %.2f",
                    unit_keyword_ratio, UNIT_KEYWORD_RATIO_THRESHOLD,
                )
            else:
                valid_numeric_count = 0
                total_cols = len(row2)

                for cell in row2:
                    cell_str = str(cell).strip()
                    try:
                        val = float(cell_str)
                        if abs(val - 1.0) > 1e-9 and abs(val + 1.0) > 1e-9:
                            valid_numeric_count += 1
                    except (ValueError, TypeError):
                        continue

                if total_cols > 0:
                    valid_numeric_ratio = valid_numeric_count / total_cols
                    if valid_numeric_ratio > VALID_NUMERIC_RATIO_THRESHOLD:
                        logger.warning(
                            "单位行校验: 第2行疑似数据行 (numeric_ratio=%.2f > %.2f)，"
                            "但保持 has_unit 不变",
                            valid_numeric_ratio, VALID_NUMERIC_RATIO_THRESHOLD,
                        )
                    else:
                        logger.debug(
                            "单位行校验: 未达数值阈值 (ratio=%.2f)，保持 has_unit=%s",
                            valid_numeric_ratio, has_unit,
                        )

        min_required_rows = 2 if actual_has_unit else 1
        if df.shape[0] < min_required_rows:
            raise ValueError(f"文件至少需要{min_required_rows}行")

        var_names = df.iloc[0].astype(str).tolist()
        # pandas 3.0 的 astype(str) 会把缺失值保留为 float NaN（而非 "nan" 字符串）。
        # CSV/TXT 表头末尾的空单元格因此会变成 float 列名，后续 _infer_schema 的
        # col.lower()、UI 的排序/筛选/绘制都会在它上面抛异常。
        # 按列位置归一为 Unnamed: N（0-based），与 pandas 自身命名习惯一致。
        var_names = [
            name if isinstance(name, str) else f"Unnamed: {i}"
            for i, name in enumerate(var_names)
        ]
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

    #: 「采样窗口的 dtype 太窄」在 pandas 侧的报错特征串（P1-6）
    _DTYPE_CONVERSION_MARKERS = (
        "could not convert string to float",
        "cannot cast array data from dtype('o')",
    )

    @classmethod
    def _is_dtype_conversion_error(cls, exc: ValueError) -> bool:
        """判定 ValueError 是否由「推断出的数值 dtype 撞上后段文本」引起。

        只认这两类特征串。真损坏文件、编码问题等 ValueError 照旧上抛 ——
        否则容错路径会把「文件确实打不开」也吞成「打开但少数据」，那才是更大的坑。
        """
        text = str(exc).lower()
        return any(marker in text for marker in cls._DTYPE_CONVERSION_MARKERS)

    @staticmethod
    def _expected_numeric_columns(dtype_map: dict) -> list[str]:
        """采样阶段被判成数值列（float32/float64）的那些列名。"""
        return [c for c, dt in dtype_map.items() if str(dt) in ("float32", "float64")]

    def _coerce_expected_numeric(
        self, df: pd.DataFrame, expected: list[str], source: str
    ) -> None:
        """容错重读后把「本该数值却变成文本」的列转回数值。

        重读不带 dtype，pandas 自己按整列判断：干净的列照样是 float64，只有真的
        混进文本的列才落成 object —— 所以这里只处理落在 expected 里、却不再是
        数值 dtype 的列，转不动的单元格置为缺失并计数（图上表现为断点）。
        """
        repaired = getattr(self, "dtype_repairs", None)
        if repaired is None:
            repaired = self.dtype_repairs = {}
        for col in expected:
            if col not in df.columns:
                continue
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                continue
            numeric = pd.to_numeric(series, errors="coerce")
            bad = int((numeric.isna() & series.notna()).sum())
            df[col] = numeric
            if bad:
                logger.warning(
                    "列 %s 有 %d 个非数值单元格（采样窗口外出现的文本），已置为缺失: %s",
                    col, bad, source,
                )
            repaired[col] = bad

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
        if not self.do_parse_date:
            parse_dates = []

        if self._progress_cb:
            self._progress_cb(20)

        _LARGE_FILE_THRESHOLD = 2 * 1024 * 1024 * 1024  # 2GB
        try:
            file_size = os.path.getsize(path)
        except OSError:
            file_size = 0

        small_file = 0 < file_size <= _LARGE_FILE_THRESHOLD
        self.dtype_repairs = {}
        common_kwargs = {
            "skiprows": (2 + desc_rows) if has_unit else (1 + desc_rows),
            "names": self._var_names,
            "encoding": self.encoding_used,
            "usecols": self.usecols,
            "memory_map": True,
            "sep": sep,
            "na_values": self._NA_VALUES,
            "keep_default_na": True,
            "on_bad_lines": "skip",
        }

        def _read(dtype: dict | None, dates: list[str]) -> pd.DataFrame:
            """按给定 dtype/日期列读一次；``dtype=None`` 让 pandas 自己按整列推断。"""
            kwargs = dict(common_kwargs, dtype=dtype, parse_dates=dates)
            if small_file:
                # pandas 内部已是 C 级流式解析，Python 层 chunksize + concat 反而
                # 额外占用 ~1× 内存。
                kwargs["low_memory"] = False
                return pd.read_csv(path, **kwargs)
            # —— 超大文件（>2GB 或无法获取大小）：保留 chunksize 分批读取防止 OOM
            kwargs["low_memory"] = True  # 超大文件分列推断，减少峰值内存
            chunks: list[pd.DataFrame] = []
            for idx, chunk in enumerate(
                pd.read_csv(path, chunksize=chunksize, **kwargs)
            ):
                chunks.append(chunk)
                if self._progress_cb and file_size > 0:
                    # 粗略进度估算（基于已读 chunk 数 × chunksize / 文件大小）
                    progress = min(80, int((idx + 1) * chunksize * 100 / file_size))
                    self._progress_cb(15 + progress)
            df = pd.concat(chunks, ignore_index=True)
            del chunks
            return df

        try:
            df = _read(dtype_map, parse_dates)
        except ValueError as e:
            # P1-6：dtype 是拿前 max_rows_infer 行推断的，文件后段（现场很常见：
            # 结尾几行状态文本、传感器故障行的字符串标记）一旦出现非数值文本，
            # 带 dtype 的读取直接抛 ValueError → 整个文件「打不开」。
            # 这里改成不带 dtype 重读一次（pandas 按整列推断，干净的列照旧是
            # 数值，只有混入文本的列落成文本），再把本该数值的列逐格 coerce，
            # 让文件照常打开，只有那几个单元格变成缺失。
            # 判定见 _is_dtype_conversion_error —— 真损坏文件照旧上抛。
            expected = self._expected_numeric_columns(dtype_map)
            if not expected or not self._is_dtype_conversion_error(e):
                raise
            logger.warning(
                "按采样推断的 dtype 读取失败（文件后段出现非数值文本），"
                "改用不带 dtype 重读: %s",
                e,
            )
            df = _read(None, parse_dates)
            self._coerce_expected_numeric(df, expected, path)

        if self._progress_cb:
            self._progress_cb(85)

        return df
