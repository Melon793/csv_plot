from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import gc
from typing import Dict, List, Tuple, Callable
import time
from PyQt6.QtCore import QThread, pyqtSignal

# some change 
class DataLoadThread(QThread):
    # 信号：发送进度 0-100，或直接发 DataFrame
    progress = pyqtSignal(int)        # 百分比
    finished = pyqtSignal(object)     # FastDataLoader 实例
    error = pyqtSignal(str)

    def __init__(self, file_path: str, parent=None,descRows:int=0,sep:str=',',hasunit:bool=True):
        super().__init__(parent)
        self.file_path = file_path
        self.descRows=descRows
        self.sep=sep
        self.hasunit=hasunit
    def run(self):
        try:
            # 将 FastDataLoader 的读取过程按块拆进度
            # 这里用文件大小估算百分比，够简单
            from pathlib import Path
            total_bytes = Path(self.file_path).stat().st_size

            def _progress_cb(bytes_read: int):
                if total_bytes > 0:
                    self.progress.emit(int(bytes_read / total_bytes * 100))

            # print("Calling FastDataLoader with _progress:", _progress_cb) 

            # 给 FastDataLoader 打补丁：加一个回调
            loader = FastDataLoader(
                self.file_path,
                # 其他参数照抄
                descRows=self.descRows,
                sep=self.sep,
                hasunit=self.hasunit,
                chunksize=10_000,          
                _progress= _progress_cb,
            )
            self.finished.emit(loader)
        except Exception as e:
            self.error.emit(str(e))

class FastDataLoader:
    # 脏数据清单
    _NA_VALUES = [
        "", "nan", "NaN", "NAN", "NULL", "null", "None",
        "Inf", "inf", "-inf", "-Inf", "1.#INF", "-1.#INF", "data err"
    ]

    def __init__(
        self,
        csv_path: str | Path,
        *,
        max_rows_infer: int = 1000,
        chunksize: int | None = None,
        usecols: list[str] | None = None,
        drop_empty: bool = False,
        downcast_float: bool = True,
        descRows: int = 0,
        sep: str = ",",
        _progress: callable | None = None,
        do_parse_date: bool =False,
        hasunit:bool = True
    ):
        #print("Calling inside FastDataLoader with _progress:", _progress) 
        self._path = Path(csv_path)
        self.file_size = Path(csv_path).stat().st_size 
        self.max_rows_infer = max_rows_infer
        self.usecols = usecols
        self.drop_empty = drop_empty
        self.downcast_float = downcast_float
        self.sep = sep
        self.descRows = descRows
        self._progress_cb = _progress
        self.do_parse_date=do_parse_date
        self.hasunit=hasunit
        # 一次性读取 header + 单位行，并回退编码
        self._var_names, self._units, self.encoding_used = self._load_header_units(
            self._path, desc_rows=self.descRows, usecols=self.usecols, sep=self.sep,hasunit=self.hasunit,
        )

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
        dtype_map, parse_dates, date_formats,downcast_ratio = self._infer_schema(sample)
        self.date_formats = date_formats
        self.sample_mem_size = sample.memory_usage(deep=True).sum()
        # print(f"the estimated downcast ratio is {downcast_ratio*100:2f} %, the compression ratio estimated {(0.5*downcast_ratio+1*(1-downcast_ratio))}")
        # print(f"sample of {sample.shape[0]} lines has costed memory {self.sample_mem_size/(1024**2):2f}Mb")
        self.byte_per_line = ((0.5*downcast_ratio+1*(1-downcast_ratio))*self.sample_mem_size)/sample.shape[0]

        self.estimated_lines = int(self.file_size/(self.byte_per_line ))
        # print(f"this file might have lines of {self.estimated_lines}")
        # 计算 chunk 大小
        if chunksize is None:
            chunksize = 10_000
        
        #print(f"chunk size is {chunksize}")
        # 正式读取
        self._df = self._read_chunks(
            self._path,
            dtype_map,
            parse_dates,
            int(chunksize),
            sep=self.sep,
            descRows=self.descRows,
            hasunit=self.hasunit
        )
        #print(f"actual lines of data files is {self.row_count}")
        # 后处理
        if drop_empty:
            self._df = self._df.dropna(axis=1, how="all")
        if downcast_float:
            self._downcast_numeric()
        self._df_validity=self._check_df_validity()
        gc.collect()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _load_header_units(
        path: Path,
        desc_rows: int = 0,
        usecols: list[str] | None = None,
        sep: str = ",",
        hasunit:bool=True
    ) -> tuple[list[str], dict[str, str], str]:
        """
        返回 (变量名列表, {变量名: 单位}, 最终编码)
        """
        nrows = 2 if hasunit else 1
        encodings = ["utf-8", "cp1252"]
        for enc in encodings:
            try:
                df = pd.read_csv(
                    path,
                    skiprows=desc_rows,
                    nrows=nrows,
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

        if df.shape[0] < nrows:
            raise ValueError("文件至少需要两行（变量名 + 单位）")

        var_names = df.iloc[0].astype(str).tolist()
        var_names = FastDataLoader._make_unique(var_names) 
        if hasunit:
            units = dict(zip(var_names, df.iloc[1].fillna("").astype(str).tolist()))
        else:
            units = dict(zip(var_names, ['-'] * len(var_names)))
        return var_names, units, enc

    @staticmethod
    def _infer_schema(sample: pd.DataFrame) -> tuple[dict[str, str], list[str], dict[str, str],float]:
        dtype_map: dict[str, str] = {}
        parse_dates: list[str] = []
        date_formats: dict[str, str] = {}

        date_candidates = [
            "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
            "%H:%M:%S", "%H:%M:%S.%f",
        ]
        float_cols = sample.select_dtypes(include=['float', 'float64','category'])
        downcast_ratio_est = float_cols.shape[1] / sample.shape[1] if sample.shape[1] > 0 else 0.000001
        for col in sample.columns:
            s = sample[col]
            if s.isna().all():
                dtype_map[col] = "category"
                continue
            for fmt in date_candidates:
                try:
                    pd.to_datetime(s, format=fmt, errors="raise")
                    parse_dates.append(col)
                    date_formats[col] = fmt
                    break
                except (ValueError, TypeError):
                    continue
            else:
                if pd.api.types.is_numeric_dtype(s):
                    dtype_map[col] = "float32"
                else:
                    dtype_map[col] = "category"
        return dtype_map, parse_dates, date_formats,downcast_ratio_est

    def _read_chunks(
        self,
        path: str | Path,
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
        for idx,chunk in enumerate(pd.read_csv(
            path,
            skiprows=(2 + descRows) if hasunit else (1+descRows),
            names=self._var_names,
            dtype=dtype_map,
            parse_dates=parse_dates,
            encoding=self.encoding_used,
            chunksize=chunksize,
            usecols=self.usecols,
            low_memory=False,
            memory_map=True,
            sep=sep,
            na_values=self._NA_VALUES,
            keep_default_na=True,
        )):
            #print(f"chunksize is {chunksize}, full size {self.file_size/(1024**2):2f}Mb")
            if self._progress_cb:
                bytes_read = (idx + 1) * chunksize * self.byte_per_line   # 粗略估算
                self._progress_cb(min(bytes_read, self.file_size))
                #print (f"progress {idx} is {bytes_read}")
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)

    def _downcast_numeric(self) -> None:
        float_cols = self._df.select_dtypes(include=["float32", "float64"]).columns
        for col in float_cols:
            self._df[col] = (
                pd.to_numeric(self._df[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .astype("float32")
            )

    # ------------------------------------------------------------------
    # 对外 API
    # ------------------------------------------------------------------

    def _check_df_validity(self) -> pd.DataFrame:
        time_start = time.perf_counter()
        validity : Dict = {}
        for col in self._df.columns:
            validity[col] = self._classify_column(self._df[col])

        validity_result = (pd.concat(
                    [#pd.Series(self._var_names, name='name'),
                    pd.Series(self._units, name='unit'),
                    pd.Series(validity, name='validity')],
                    axis=1
                )
        .rename_axis('name')
        .reset_index()
        )
        print(f"used time: {(time.perf_counter()-time_start):3f}")
        return validity_result

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
    def _classify_column(series: pd.Series) -> int:
        """
        1: 全部可转数字，且 ≥2 个不同有效值
        0: 全部可转数字，且唯一有效值
        -1: 存在非数字 或 全部 NaN
        """
        # 1) 先尝试整列转 float，失败直接 C
        try:
            numeric = pd.to_numeric(series, errors="raise")
        except (ValueError, TypeError):
            return (-1)

        # 2) 去掉 NaN 后看有效值
        valid = numeric.dropna()
        if valid.empty:          # 全 NaN
            return (-1)

        unique_vals = valid.unique()
        if len(unique_vals) == 1:
            return (0)
        return (1)
    
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
    def var_names(self) -> list[str]:
        return self._df.columns.tolist()
    
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
    def df_validity(self) -> pd.DataFrame:
        return self._df_validity
