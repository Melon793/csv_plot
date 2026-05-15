"""
MDF 文件加载器模块 —— 多 Group 模式

提供 MDF（Measurement Data Format）文件的完整解析与加载功能，
对外暴露与 FastDataLoader 对齐的属性接口，使 UI 层无需感知文件格式差异。

支持格式: .mf4（MDF 4.x）、.mdf（MDF 3.x）、.dat（INCA 导出）

作者: SOLO
创建日期: 2024
"""

import os
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class GroupData:
    """单个 MDF Channel Group 的完整数据封装

    Attributes:
        index: Group 编号（0-based）
        time_values: 时间轴数组（float64），取自 master channel
        time_channel_name: 主通道名称（如 "time", "t"）
        signals: 信号名 → 信号值数组的映射
        units: 信号名 → 单位的映射
    """
    index: int
    time_values: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    time_channel_name: str = ""
    signals: dict[str, np.ndarray] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)

    @property
    def var_names(self) -> list[str]:
        """返回当前分组中所有信号变量的名称列表"""
        return list(self.signals.keys())

    @property
    def datalength(self) -> int:
        """返回当前分组的数据长度（采样点数）"""
        return len(self.time_values)

    def to_dataframe(self) -> pd.DataFrame:
        """将分组数据转换为 DataFrame，index=时间值，columns=信号名"""
        data = {}
        if self.time_channel_name:
            data[self.time_channel_name] = self.time_values
        else:
            data["time"] = self.time_values
        data.update(self.signals)
        df = pd.DataFrame(data)
        if self.time_channel_name:
            df.index = df[self.time_channel_name]
            df.index.name = self.time_channel_name
        return df


class MDFDataLoader:
    """INCA MDF/DAT 文件加载器 —— 多 Group 模式。

    接口与 FastDataLoader 对齐，上层代码无需区分文件格式。

    典型用法::

        loader = MDFDataLoader("measurement.dat")
        print(loader.var_names)       # 聚合后的变量名列表
        print(loader.group_count)     # Group 数量
        loader.select_group(1)        # 切换 Group
        df = loader.df                # 当前 Group DataFrame
    """

    _ASAMMDF_IMPORT_ERROR = (
        "asammdf 库未安装。请运行: pip install asammdf>=7.4.0"
    )

    _CONFLICT_SUFFIX_PATTERN = re.compile(r'_(G\d+)$')

    def __init__(self, path: str, *, _progress: Callable[[int], None] = None):
        """初始化 MDFDataLoader，加载并解析 MDF 文件。

        Args:
            path: MDF 文件路径（.mf4 / .mdf / .dat）
            _progress: 可选进度回调，参数为 0-100 的整数百分比

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件为空或格式不兼容
            ImportError: asammdf 库未安装
        """
        self._path = path
        self._progress = _progress
        self._groups: list[GroupData] = []
        self._current_group_index: int = 0
        self._var_to_group: dict[str, int] = {}
        self._aggregated_var_names: list[str] = []
        self._aggregated_units: dict[str, str] = {}
        self._file_size: int = 0

        self._validate_file()
        self._load_groups()
        self._build_aggregated_properties()

        if self._groups:
            self._current_group_index = 0
        else:
            raise ValueError("MDF 文件未包含任何 Channel Group")

    def _notify_progress(self, value: int):
        if self._progress:
            try:
                self._progress(value)
            except Exception:
                pass

    def _validate_file(self):
        """校验文件存在性、大小及格式"""
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"MDF 文件不存在: {self._path}")

        file_size = os.path.getsize(self._path)
        if file_size == 0:
            raise ValueError("MDF 文件为空")
        self._file_size = file_size

    @staticmethod
    def _safe_to_numpy(series: pd.Series) -> np.ndarray:
        """安全地将 pandas Series 转换为 numpy 数组，优先使用 float64"""
        if np.issubdtype(series.dtype, np.number):
            return series.to_numpy(dtype=np.float64).copy()
        try:
            return series.to_numpy(dtype=np.float64).copy()
        except (ValueError, TypeError):
            return series.to_numpy(dtype=object).copy()

    def _load_groups(self):
        """使用 asammdf 解析 MDF 文件，填充 self._groups"""
        try:
            import asammdf
        except ImportError:
            raise ImportError(self._ASAMMDF_IMPORT_ERROR)

        self._notify_progress(0)

        try:
            mdf = asammdf.MDF(self._path)
        except Exception as e:
            raise ValueError(f"无法打开 MDF 文件（文件可能已损坏或不兼容）: {e}")

        self._notify_progress(5)

        total_groups = len(mdf.groups)
        if total_groups == 0:
            raise ValueError("MDF 文件未包含任何 Channel Group")

        for gi in range(total_groups):
            try:
                group_df = mdf.get_group(gi)
                group_obj = mdf.groups[gi]

                time_values = group_df.index.to_numpy(dtype=np.float64).copy()
                time_channel_name = group_df.index.name or "time"

                signals: dict[str, np.ndarray] = {}
                units: dict[str, str] = {}

                for col in group_df.columns:
                    arr = self._safe_to_numpy(group_df[col])
                    signals[col] = arr

                if hasattr(group_obj, 'channels') and group_obj.channels:
                    for ch in group_obj.channels:
                        unit_val = None
                        if ch.unit and ch.unit.strip():
                            unit_val = ch.unit.strip()
                        elif hasattr(ch, 'conversion') and ch.conversion is not None:
                            conv_unit = getattr(ch.conversion, 'unit', None)
                            if conv_unit and conv_unit.strip():
                                unit_val = conv_unit.strip()
                        if unit_val is None:
                            unit_val = "-"
                        is_time_channel = (
                            ch.name == time_channel_name
                            or ch.name == "time"
                            and time_channel_name in ("time", "timestamps")
                        )
                        if ch.name in signals or is_time_channel:
                            units[ch.name] = unit_val
                            if is_time_channel and ch.name != time_channel_name:
                                units[time_channel_name] = unit_val

                for col in group_df.columns:
                    if col not in units:
                        units[col] = "-"

                gd = GroupData(
                    index=gi,
                    time_values=time_values,
                    time_channel_name=time_channel_name,
                    signals=signals,
                    units=units,
                )
                self._groups.append(gd)

            except Exception as e:
                raise RuntimeError(f"加载 Group {gi} 时出错: {e}")

            progress = int(5 + (gi + 1) / total_groups * 90)
            self._notify_progress(progress)

        self._notify_progress(100)

    def _build_aggregated_properties(self):
        """构建跨 Group 的聚合属性列表与映射"""
        pure_name_counts: dict[str, int] = {}
        pure_name_groups: dict[str, list[int]] = {}

        for gd in self._groups:
            for var in gd.var_names:
                pure_name_counts[var] = pure_name_counts.get(var, 0) + 1
                if var not in pure_name_groups:
                    pure_name_groups[var] = []
                pure_name_groups[var].append(gd.index)

        conflict_names = {k for k, v in pure_name_counts.items() if v > 1}

        self._aggregated_var_names = []
        self._aggregated_units = {}
        self._var_to_group = {}
        self._original_to_aggregated: dict[tuple[int, str], str] = {}

        for gd in self._groups:
            for var in gd.var_names:
                if var in conflict_names:
                    display_name = f"{var}_G{gd.index}"
                else:
                    display_name = var

                self._aggregated_var_names.append(display_name)
                self._aggregated_units[display_name] = gd.units.get(var, "-")
                self._var_to_group[display_name] = gd.index
                self._original_to_aggregated[(gd.index, var)] = display_name

        self._aggregated_validity: dict[str, int] = {}
        for display_name in self._aggregated_var_names:
            group_index = self._var_to_group[display_name]
            gd = self._groups[group_index]
            pure_name = self._resolve_pure_column_name(display_name)
            values = gd.signals.get(pure_name)
            if values is None:
                self._aggregated_validity[display_name] = -1
                continue
            self._aggregated_validity[display_name] = self._classify_validity(values)

    @staticmethod
    def _classify_validity(values: np.ndarray) -> int:
        """对信号值进行有效性分类。

        Returns:
            1=有效, 0=常量, -1=无效
        """
        if values.size == 0:
            return -1
        if not np.issubdtype(values.dtype, np.number):
            return 1
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            return -1
        finite_vals = values[finite_mask]
        if np.allclose(finite_vals, finite_vals[0]):
            return 0
        return 1

    def select_group(self, index: int):
        """切换到指定 Group

        Args:
            index: Group 索引（0-based）

        Raises:
            IndexError: Group 索引超出范围
        """
        if index < 0 or index >= len(self._groups):
            raise IndexError(
                f"Group 索引 {index} 超出范围，有效范围: 0-{len(self._groups) - 1}"
            )
        self._current_group_index = index

    def get_var_metadata(self, display_name: str) -> tuple[int, np.ndarray, str]:
        """获取变量的元数据。

        Args:
            display_name: 显示变量名（可能含 _G{index} 冲突后缀）

        Returns:
            (group_index, time_values_array, unit_string)
        """
        group_index = self._var_to_group.get(display_name)
        if group_index is None:
            raise KeyError(f"变量 '{display_name}' 不存在")
        gd = self._groups[group_index]
        time_values = gd.time_values.copy()
        unit = self._aggregated_units.get(display_name, "-")
        return group_index, time_values, unit

    @staticmethod
    def _resolve_pure_column_name(display_name: str) -> str:
        """去掉 _G{index} 冲突后缀，还原纯变量名

        Args:
            display_name: 可能带 _G{index} 后缀的变量名

        Returns:
            纯变量名
        """
        return MDFDataLoader._CONFLICT_SUFFIX_PATTERN.sub('', display_name)

    @property
    def groups(self) -> list[GroupData]:
        """所有 Group 数据列表"""
        return self._groups

    @property
    def group_count(self) -> int:
        """Group 总数"""
        return len(self._groups)

    @property
    def current_group_index(self) -> int:
        """当前激活的 Group 索引"""
        return self._current_group_index

    @property
    def _current_group(self) -> Optional[GroupData]:
        """返回当前选中分组的 GroupData 对象"""
        if 0 <= self._current_group_index < len(self._groups):
            return self._groups[self._current_group_index]
        return None

    @property
    def var_names(self) -> list[str]:
        """所有 Group 聚合后的显示变量名列表"""
        return self._aggregated_var_names

    @property
    def units(self) -> dict[str, str]:
        """变量名 → 单位的映射（聚合所有 Group）"""
        return dict(self._aggregated_units)

    @property
    def df(self) -> pd.DataFrame:
        """当前 Group 的 DataFrame，列名使用聚合变量名"""
        group = self._current_group
        if group:
            df = group.to_dataframe()
            rename_map = {}
            for col in df.columns:
                key = (group.index, col)
                agg_name = self._original_to_aggregated.get(key, col)
                if agg_name != col:
                    rename_map[col] = agg_name
            if rename_map:
                df = df.rename(columns=rename_map)
            return df
        return pd.DataFrame()

    @property
    def df_validity(self) -> dict[str, int]:
        """所有 Group 聚合后的变量有效性字典（key=聚合变量名，value=1有效/0常量/-1无效）"""
        return dict(self._aggregated_validity)

    def get_value_from_name(self, display_name: str):
        """通过聚合变量名获取变量的 (x_data, y_data, unit)。

        支持跨 Group 查找，无需手动切换 Group。

        Args:
            display_name: 聚合变量名（可能含 _G{index} 冲突后缀）

        Returns:
            (x_data: np.ndarray, y_data: np.ndarray, unit: str)

        Raises:
            KeyError: 变量不存在
        """
        group_index, time_values, unit = self.get_var_metadata(display_name)
        pure_name = self._resolve_pure_column_name(display_name)
        gd = self._groups[group_index]
        y_data = gd.signals[pure_name]
        return time_values, y_data, unit

    def get_series(self, display_name: str) -> pd.Series:
        """通过聚合变量名获取变量的 pandas Series。

        自动解析跨 Group 的变量位置。

        Args:
            display_name: 聚合变量名（可能含 _G{index} 冲突后缀）

        Returns:
            包含该变量所有值的 pd.Series

        Raises:
            KeyError: 变量不存在
        """
        group_index, time_values, unit = self.get_var_metadata(display_name)
        pure_name = self._resolve_pure_column_name(display_name)
        gd = self._groups[group_index]
        y_data = gd.signals[pure_name]
        return pd.Series(y_data, name=display_name)

    @property
    def time_values(self) -> pd.Series:
        """当前 Group 的时间轴 Series"""
        group = self._current_group
        if group and len(group.time_values) > 0:
            return pd.Series(
                group.time_values,
                name=group.time_channel_name or "time",
                dtype=np.float64,
            )
        return pd.Series(np.arange(1), name="index")

    @property
    def time_column_name(self) -> Optional[str]:
        """当前 Group 的主通道名称"""
        group = self._current_group
        if group:
            return group.time_channel_name or "time"
        return None

    @property
    def time_axis_label(self) -> str:
        """X 轴标签文本（含时间单位）"""
        group = self._current_group
        if group:
            name = group.time_channel_name or "time"
            time_unit = "-"
            if group.time_channel_name:
                time_unit = group.units.get(group.time_channel_name, "-")
            if time_unit and time_unit != "-":
                return f"{name} ({time_unit})"
            return name
        return "Index"

    @property
    def datalength(self) -> int:
        """当前 Group 的数据行数"""
        group = self._current_group
        if group:
            return group.datalength
        return 0

    @property
    def row_count(self) -> int:
        """当前 Group 的数据行数（与 datalength 相同）"""
        return self.datalength

    @property
    def column_count(self) -> int:
        """当前 Group 的变量数"""
        group = self._current_group
        if group:
            return len(group.var_names)
        return 0

    @property
    def time_channels_info(self) -> list:
        """时间通道信息（MDF 场景下返回空列表）"""
        return []

    @property
    def path(self) -> str:
        """文件路径"""
        return self._path

    @property
    def file_size(self) -> int:
        """文件大小（字节）"""
        return self._file_size
