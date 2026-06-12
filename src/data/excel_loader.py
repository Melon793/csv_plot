"""ExcelDataLoader - Excel 数据加载器

使用 calamine (Rust) 引擎快速读取，fallback 到 openpyxl iter_rows。
"""

from __future__ import annotations
import os
import gc
from typing import Callable
import numpy as np
import pandas as pd
from src.data.base_loader import BaseDataLoader
from src.core.config import _UNIT_KEYWORDS, UNIT_KEYWORD_RATIO_THRESHOLD
from src.core.logger import get_logger

logger = get_logger("data.excel_loader")


class ExcelDataLoader(BaseDataLoader):
    """Excel 数据加载器，继承自 BaseDataLoader。

    优先使用 calamine (Rust) 引擎读取，不可用时回退到优化后的 openpyxl。
    """

    LOADER_TYPE = "excel"

    # 分块大小：每次读取的行数
    CHUNK_SIZE = 2000

    def __init__(
        self,
        file_path: str,
        *,
        sheet_name: str | int = 0,
        downcast_float: bool = True,
        desc_rows: int = 0,
        has_unit: bool | None = None,
        _progress: Callable | None = None,
    ):
        """初始化 Excel 数据加载器

        Args:
            file_path: Excel 文件路径
            sheet_name: Sheet 名称或索引（0-based）
            downcast_float: 是否下转换浮点数类型
            desc_rows: 描述行数量
            has_unit: 是否包含单位行，None 表示自动检测
            _progress: 进度回调函数
        """
        super().__init__()

        self._path = file_path
        self.file_size = os.path.getsize(file_path)
        self._sheet_name = sheet_name
        self.downcast_float = downcast_float
        self.desc_rows = desc_rows
        self._progress_cb = _progress

        logger.info("开始加载 Excel: %s (%.1f MB)", file_path, self.file_size / 1024 / 1024)

        import openpyxl
        self._wb = openpyxl.load_workbook(
            file_path, read_only=True, data_only=True
        )

        # 解析 sheet_name（支持字符串名或索引）
        if isinstance(sheet_name, int):
            ws_name = self._wb.sheetnames[sheet_name]
        else:
            ws_name = sheet_name
        self._ws = self._wb[ws_name]

        if self._progress_cb:
            self._progress_cb(5)

        # 读取表头 + 自动检测 has_unit
        self._var_names, self._units, self.has_unit = self._load_header_units(has_unit)

        if self._progress_cb:
            self._progress_cb(10)

        # 读取数据：优先使用 calamine (Rust)，fallback 到优化后的 openpyxl
        self._df = self._read_data()

        # 时间列推断
        self._infer_time_columns()

        # 后处理：合并 downcast + validity 为单次遍历
        self._df_validity = self._postprocess_columns(
            downcast=downcast_float
        )

        # 关闭 workbook 释放资源
        self._wb.close()

        if self._progress_cb:
            self._progress_cb(100)

        logger.info(
            "Excel 加载完成: %s (sheet=%s, %d 行, %d 列)",
            file_path, ws_name, len(self._df), len(self._var_names),
        )

    def _load_header_units(self, has_unit_hint: bool | None):
        """加载表头和单位信息，支持 has_unit 自动检测"""
        header_row_idx = self.desc_rows + 1  # openpyxl 1-based 行号
        rows = []
        for row in self._ws.iter_rows(
            min_row=header_row_idx, max_row=header_row_idx + 1, values_only=True
        ):
            rows.append(row)

        if not rows:
            raise ValueError("Excel 文件标题行为空")

        # 解析列名（清理前后空格和换行符）
        var_names = [
            str(cell).strip().replace("\n", " ").replace("\r", "")
            if cell is not None else f"Column_{i}"
            for i, cell in enumerate(rows[0])
        ]
        var_names = self._make_unique(var_names)

        # has_unit 自动检测
        if has_unit_hint is not None:
            actual_has_unit = has_unit_hint
        elif len(rows) >= 2:
            unit_row = rows[1]
            unit_hit = 0
            total = 0
            for cell in unit_row:
                if cell is None:
                    continue
                cell_str = str(cell).strip()
                if not cell_str:
                    continue
                total += 1
                cell_lower = cell_str.lower()
                for keyword in _UNIT_KEYWORDS:
                    if keyword.lower() in cell_lower:
                        unit_hit += 1
                        break
            actual_has_unit = (
                total > 0 and (unit_hit / total) > UNIT_KEYWORD_RATIO_THRESHOLD
            )
        else:
            actual_has_unit = False

        if actual_has_unit and len(rows) >= 2:
            units = {
                name: (str(cell).strip() if cell is not None else "-")
                for name, cell in zip(var_names, rows[1])
            }
        else:
            units = {name: "-" for name in var_names}

        return var_names, units, actual_has_unit

    def _read_data(self) -> pd.DataFrame:
        """读取数据：优先 calamine (Rust)，不可用时回退到优化后的 openpyxl"""
        try:
            return self._read_with_calamine()
        except ImportError:
            logger.info("calamine 不可用，使用优化后的 openpyxl 读取")
        except Exception as e:
            logger.warning("calamine 读取失败 (%s)，fallback 到 openpyxl", e)

        return self._read_chunks()

    def _read_with_calamine(self) -> pd.DataFrame:
        """使用 calamine (Rust) 引擎快速读取 Excel 数据。

        calamine 读取缓存值（与 openpyxl data_only=True 行为一致），
        不支持公式重新求值，但对正常保存的 Excel 文件无影响。
        """
        data_start = self.desc_rows + (3 if self.has_unit else 2)  # 1-based

        df = pd.read_excel(
            self._path,
            sheet_name=self._sheet_name,
            engine='calamine',
            header=None,
            skiprows=data_start - 1,
            names=self._var_names,
        )

        if self._progress_cb:
            self._progress_cb(90)

        # 后处理：类型转换（对象列→数值，跳过 calamine 已推断的类型）
        datetime_cols = self._detect_datetime_cols(df)
        obj_cols = [
            c for c in df.columns
            if c not in datetime_cols and df[c].dtype == object
        ]
        if obj_cols:
            df[obj_cols] = df[obj_cols].apply(
                pd.to_numeric, errors='coerce'
            )
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        return df

    def _read_chunks(self) -> pd.DataFrame:
        """优化后的 openpyxl 读取：一次性 iter_rows + 内存分块。"""
        data_start = self.desc_rows + (3 if self.has_unit else 2)  # 1-based
        max_row = self._ws.max_row or 0
        if max_row < data_start:
            return pd.DataFrame(columns=self._var_names)

        # 一次性读取所有数据行，消除重复 XML 解析
        all_rows = list(self._ws.iter_rows(
            min_row=data_start, max_row=max_row, values_only=True
        ))

        total_rows = len(all_rows)
        total_chunks = max(1, (total_rows + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE)
        increment = 80 / total_chunks

        chunks: list[pd.DataFrame] = []
        datetime_cols: set[str] | None = None

        for chunk_idx in range(total_chunks):
            start = chunk_idx * self.CHUNK_SIZE
            end = min(start + self.CHUNK_SIZE, total_rows)
            chunk_data = all_rows[start:end]

            if not chunk_data:
                break

            df_chunk = pd.DataFrame(chunk_data, columns=self._var_names)

            # 首块：检测 datetime 列
            if datetime_cols is None:
                datetime_cols = self._detect_datetime_cols(df_chunk)

            # 所有数值列一次性批量转换
            non_dt_cols = [c for c in df_chunk.columns if c not in datetime_cols]
            if non_dt_cols:
                df_chunk[non_dt_cols] = df_chunk[non_dt_cols].apply(
                    pd.to_numeric, errors='coerce'
                )

            # datetime 列单独处理
            for col in datetime_cols:
                if col in df_chunk.columns:
                    df_chunk[col] = pd.to_datetime(df_chunk[col], errors="coerce")

            chunks.append(df_chunk)

            if self._progress_cb:
                chunk_progress = min(80, (chunk_idx + 1) * increment)
                self._progress_cb(15 + int(chunk_progress))

            if chunk_idx % 3 == 0:
                gc.collect()

        if not chunks:
            return pd.DataFrame(columns=self._var_names)

        return pd.concat(chunks, ignore_index=True)

    @staticmethod
    def _detect_datetime_cols(df_chunk: pd.DataFrame) -> set[str]:
        """从首块数据中检测 datetime 类型列"""
        import datetime as _dt
        dt_cols: set[str] = set()
        for col in df_chunk.columns:
            sample = df_chunk[col].dropna()
            if len(sample) == 0:
                continue
            first_val = sample.iloc[0]
            if isinstance(first_val, (_dt.datetime, _dt.date, _dt.time)):
                dt_cols.add(col)
        return dt_cols

    def _infer_time_columns(self):
        """推断时间列"""
        import datetime as _dt

        time_keywords = ["time", "date", "datetime", "timestamp", "zeit", "tmod"]
        date_candidates = [
            "%H:%M:%S.%f", "%H:%M:%S",
            "%d/%m/%Y", "%Y/%m/%d", "%Y-%m-%d",
        ]

        for col in self._var_names:
            col_lower = col.lower()
            if not any(kw in col_lower for kw in time_keywords):
                continue
            if col not in self._df.columns:
                continue

            s = self._df[col]
            s_sample = s.head(10).dropna()
            if len(s_sample) == 0:
                continue

            # 情况1: 已解析为 datetime 类型
            if pd.api.types.is_datetime64_any_dtype(s):
                self.date_formats[col] = "%Y-%m-%d %H:%M:%S"
                if self.time_column_name is None:
                    self.time_column_name = col
                continue

            # 情况2: 列元素是 Python datetime 对象（object dtype）
            first_val = s_sample.iloc[0]
            if isinstance(first_val, (_dt.datetime, _dt.date)):
                self._df[col] = pd.to_datetime(self._df[col], errors="coerce")
                self.date_formats[col] = "%Y-%m-%d %H:%M:%S"
                if self.time_column_name is None:
                    self.time_column_name = col
                continue

            # 情况3: 仍为字符串，使用格式候选列表匹配
            for fmt in date_candidates:
                try:
                    pd.to_datetime(s_sample, format=fmt, errors="raise")
                    self.date_formats[col] = fmt
                    if self.time_column_name is None:
                        self.time_column_name = col
                    break
                except (ValueError, TypeError):
                    continue

    @staticmethod
    def get_sheet_info(file_path: str) -> list[dict]:
        """获取所有 Sheet 的基本信息（名称、行数、列数）

        使用 openpyxl 只读模式，仅获取元数据，不解析数据内容。
        """
        import openpyxl
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        result = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            result.append({
                'name': sheet_name,
                'rows': ws.max_row or 0,
                'cols': ws.max_column or 0,
            })
        wb.close()
        return result
