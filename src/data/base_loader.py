"""BaseDataLoader - 数据加载器基类

为内存加载型加载器（FastDataLoader、ExcelDataLoader）提供公共功能。
MDFLazyLoader 因惰性加载设计差异不纳入此体系。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from src.core.config import FLOAT32_REPRESENTABLE_MAX
from src.core.logger import get_logger

logger = get_logger(__name__)


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

    def _postprocess_columns(self, downcast: bool = True) -> dict[str, int]:
        """合并 downcast + validity 检查为单次遍历，减少冗余数组创建。

        优化点：
        1. float 列：isfinite 检测 + inf 清理 + float32 下转换合并为单次操作
        2. 整数列：首尾比较替代 np.unique 的全量排序
        3. 有效性判断：nanmax/nanmin 替代布尔索引拷贝
        """
        validity: dict[str, int] = {}

        for col in self._df.columns:
            if col in self.date_formats:
                validity[col] = 1
                continue

            series = self._df[col]

            if pd.api.types.is_float_dtype(series):
                arr = series.to_numpy(copy=True)  # 必须可写（memory_map 产生的数组可能是只读的）

                if downcast:
                    finite_mask = np.isfinite(arr)
                    if finite_mask.all() and arr.dtype == np.float64:
                        # 无 inf，直接判断是否能下转换
                        max_abs = float(np.nanmax(np.abs(arr))) if arr.size > 0 else 0.0
                        if max_abs <= FLOAT32_REPRESENTABLE_MAX:
                            arr = arr.astype(np.float32)  # 拷贝 2（如果需要）
                            self._df[col] = arr
                    elif not finite_mask.all():
                        # 有 inf，清理后判断是否下转换
                        if arr.dtype == np.float64:
                            finite_vals = arr[finite_mask]
                            if finite_vals.size > 0:
                                max_abs = float(np.max(np.abs(finite_vals)))
                            else:
                                max_abs = 0.0
                            if max_abs <= FLOAT32_REPRESENTABLE_MAX:
                                target = np.empty(arr.shape, dtype=np.float32)
                                np.copyto(target, arr, where=finite_mask)
                                target[~finite_mask] = np.float32(np.nan)
                                self._df[col] = target
                                arr = target
                            else:
                                arr[~finite_mask] = np.nan
                                self._df[col] = arr
                        else:
                            arr[~finite_mask] = np.nan
                            self._df[col] = arr

                # 有效性判断（复用已处理的 arr，无额外分配）
                if arr.size == 0:
                    validity[col] = -1
                else:
                    try:
                        min_v = np.nanmin(arr)
                        max_v = np.nanmax(arr)
                        if np.isnan(min_v) and np.isnan(max_v):
                            validity[col] = -1
                        elif min_v == max_v:
                            validity[col] = 0
                        else:
                            validity[col] = 1
                    except (ValueError, TypeError):
                        validity[col] = -1

            elif pd.api.types.is_integer_dtype(series):
                arr = series.to_numpy()
                if arr.size == 0:
                    validity[col] = -1
                else:
                    # 快速判断：首尾比较 → 若不同则一定非常量
                    # 若相同再用 np.all 确认（避免 np.unique 的全量排序开销）
                    if arr[0] != arr[-1]:
                        validity[col] = 1
                    elif np.all(arr == arr[0]):
                        validity[col] = 0
                    else:
                        validity[col] = 1

            else:
                # 非数值列：尝试一次性转为数值
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

    def release_memory(self):
        """显式释放内存（供重载流程调用）。

        直接将 _df 替换为空 DataFrame，让旧 DataFrame 被 Python 引用计数回收。
        """
        if hasattr(self, "_df") and self._df is not None:
            self._df = pd.DataFrame()

        # 清理其他可能占内存的属性
        for attr in ("_var_names", "_units", "_df_validity"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    logger.debug("清理属性 '%s' 失败", attr)

        # Excel loader: 关闭 workbook（如果存在）
        if hasattr(self, "_wb") and self._wb is not None:
            try:
                self._wb.close()
            except Exception:
                logger.debug("关闭 workbook 失败")
        if hasattr(self, "_ws"):
            try:
                self._ws = None
            except Exception:
                logger.debug("清理 worksheet 引用失败")

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
