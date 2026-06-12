"""BaseDataLoader - 数据加载器基类

为内存加载型加载器（FastDataLoader、ExcelDataLoader）提供公共功能。
MDFLazyLoader 因惰性加载设计差异不纳入此体系。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from src.core.config import _evaluate_float32_safety, FLOAT32_SAFE_MAX


class BaseDataLoader:
    """数据加载器基类，提供通用功能。

    仅适用于内存加载型加载器（FastDataLoader、ExcelDataLoader）。
    MDFLazyLoader 因惰性加载设计差异不纳入此体系。
    """

    LOADER_TYPE = "base"

    # 公共的脏数据清单（CSV 读取时作为 na_values 传入；
    # Excel 场景仅用于日志标记，openpyxl 读取的是原生类型）
    _NA_VALUES = [
        "", "NULL", "None", "NA", "N/A", "n/a", "null ",
        "#N/A", "#N/A N/A", "#NA",
        "NaN", "nan", "-NaN", "-nan",
        "1.#IND", "-1.#IND", "1.#QNAN", "-1.#QNAN",
        "Infinity", "Inf", "inf", "plus infinity", "minus infinity",
        "1.#INF", "-1.#INF",
        "data err", "* *", "**", "----", "no value",
    ]

    def __init__(self):
        self._df: pd.DataFrame = pd.DataFrame()
        self._var_names: list[str] = []
        self._units: dict[str, str] = {}
        self._df_validity: dict[str, int] = {}
        self.time_column_name: str | None = None  # 公开属性
        self.date_formats: dict[str, str] = {}
        self._path: str = ""

    # ---- 公共静态方法 ----
    @staticmethod
    def _make_unique(names: list[str]) -> list[str]:
        """确保列名唯一"""
        seen: dict[str, int] = {}
        unique_names: list[str] = []
        for name in names:
            if name in seen:
                seen[name] += 1
                unique_names.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 0
                unique_names.append(name)
        return unique_names

    @staticmethod
    def _classify_column(series: pd.Series, col_name: str, date_formats: dict) -> int:
        """列有效性分类：1=可绘图, 0=常数列, -1=无效"""
        if col_name in date_formats:
            return 1
        try:
            numeric = pd.to_numeric(series, errors="raise").values
        except (ValueError, TypeError):
            return -1
        if numeric.dtype.kind in "iu":
            valid = numeric
        else:
            valid = numeric[~np.isnan(numeric)]
        if len(valid) == 0:
            return -1
        if len(series) == 1:
            return 1
        unique_count = np.unique(valid).size
        return 0 if unique_count == 1 else 1

    def _downcast_numeric(self) -> None:
        """下转换数值类型以节省内存"""
        float_cols = self._df.select_dtypes(include=["float32", "float64"]).columns
        for col in float_cols:
            cleaned = pd.to_numeric(self._df[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            is_safe, _ = _evaluate_float32_safety(cleaned)
            if is_safe:
                self._df[col] = cleaned.astype("float32")
            else:
                self._df[col] = cleaned.astype("float64", copy=False)

    def _check_df_validity(self) -> dict[str, int]:
        """检查数据有效性"""
        validity: dict[str, int] = {}
        for col in self._df.columns:
            validity[col] = self._classify_column(self._df[col], col, self.date_formats)
        return validity

    def _postprocess_columns(self, downcast: bool = True) -> dict[str, int]:
        """合并 downcast + validity 检查为单次遍历，减少冗余数组创建

        对每列在一次遍历中完成：
        1. 数值类型的 inf 清理和 float32 下转换
        2. 列有效性分类（1=可绘图, 0=常数列, -1=无效）

        Args:
            downcast: 是否执行 float64 -> float32 下转换

        Returns:
            {列名: 有效性} 字典
        """
        validity: dict[str, int] = {}

        for col in self._df.columns:
            # 时间格式列直接标记为可绘图
            if col in self.date_formats:
                validity[col] = 1
                continue

            series = self._df[col]

            if pd.api.types.is_float_dtype(series):
                arr = series.to_numpy()

                if downcast:
                    # 一次性清理 inf -> nan
                    has_inf = not np.all(np.isfinite(arr))
                    if has_inf:
                        arr = np.where(np.isfinite(arr), arr, np.nan)
                        self._df[col] = arr

                    # float32 安全检查（直接操作 numpy 数组，无冗余 pd.to_numeric）
                    finite_vals = arr[np.isfinite(arr)]
                    if finite_vals.size > 0:
                        abs_max = float(np.max(np.abs(finite_vals)))
                        if abs_max <= FLOAT32_SAFE_MAX and arr.dtype == np.float64:
                            arr = arr.astype(np.float32)
                            self._df[col] = arr

                # 有效性检查（复用已有的 arr）
                if arr.dtype.kind == 'f':
                    valid = arr[~np.isnan(arr)]
                else:
                    valid = arr
                if valid.size == 0:
                    validity[col] = -1
                elif np.unique(valid).size == 1:
                    validity[col] = 0
                else:
                    validity[col] = 1

            elif pd.api.types.is_integer_dtype(series):
                arr = series.to_numpy()
                if arr.size == 0:
                    validity[col] = -1
                elif np.unique(arr).size == 1:
                    validity[col] = 0
                else:
                    validity[col] = 1

            else:
                # 非数值列（category/object/datetime 等）：尝试数值转换
                try:
                    numeric = pd.to_numeric(series, errors="raise").to_numpy()
                    if numeric.dtype.kind in "iu":
                        valid = numeric
                    else:
                        valid = numeric[~np.isnan(numeric)]
                    if valid.size == 0:
                        validity[col] = -1
                    elif np.unique(valid).size == 1:
                        validity[col] = 0
                    else:
                        validity[col] = 1
                except (ValueError, TypeError):
                    validity[col] = -1

        return validity

    # ---- 公共属性接口 ----
    @property
    def path(self) -> str:
        return self._path

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def units(self) -> dict[str, str]:
        return self._units

    @property
    def datalength(self) -> int:
        return self._df.shape[0]

    @property
    def var_names(self) -> list[str]:
        cols = self._df.columns.tolist()
        if self.time_column_name and self.time_column_name in cols:
            cols = [c for c in cols if c != self.time_column_name]
        return cols

    @property
    def df_validity(self) -> dict[str, int]:
        validity = dict(self._df_validity)
        if self.time_column_name and self.time_column_name in validity:
            del validity[self.time_column_name]
        return validity

    @property
    def time_channels_info(self) -> dict[str, str]:
        return self.date_formats

    @property
    def time_axis_label(self) -> str:
        if self.time_column_name:
            unit = self._units.get(self.time_column_name, "")
            if unit and unit != "-":
                return f"{self.time_column_name} ({unit})"
            return self.time_column_name
        return "Index"

    @property
    def default_time_values(self) -> pd.Series:
        return pd.Series(np.arange(1, len(self._df) + 1), name="index")

    @property
    def time_values(self) -> pd.Series:
        if self.time_column_name and self.time_column_name in self._df.columns:
            return self._df[self.time_column_name]
        return self.default_time_values

    @property
    def global_time_range(self) -> tuple[float, float]:
        return (1.0, float(len(self._df)))

    @property
    def baseline_density(self) -> float:
        return 1.0

    @property
    def max_row_count(self) -> int:
        return len(self._df)

    @property
    def row_count(self) -> int:
        return len(self._df)

    @property
    def column_count(self) -> int:
        return len(self._df.columns)

    # ---- 数据访问接口 ----
    def get_series(self, name: str) -> pd.Series:
        """返回单列 Series"""
        return self._df[name]

    def get_value_from_name(self, name: str):
        """返回绘图所需的四元组 (index, values, unit, enum_map)"""
        index = np.arange(1, len(self._df) + 1, dtype=np.float64)
        values = self._df[name]
        unit = self._units.get(name, "-")
        return index, values, unit, {}
