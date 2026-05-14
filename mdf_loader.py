"""
MDF 文件加载器模块

为 MDF 文件导入功能提供核心架构支持，包括：
- GroupData: MDF 数据分组的封装数据结构
- MDFDataLoader: MDF 文件加载器（第二阶段实现加载逻辑）

本模块设计遵循与 FastDataLoader 一致的接口约定，确保与现有 UI 层无缝对接。

作者: SOLO
创建日期: 2024
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GroupData:
    """MDF 数据分组

    封装 MDF 文件中单个分组（channel group）的数据，
    包含时间轴、信号数据和元信息。

    Attributes:
        name: 分组名称
        time_values: 时间轴数值（相对时间，秒）
        time_channel_name: 时间通道名称
        signals: 信号名 -> 信号数值的映射
        units: 信号名 -> 单位的映射
    """
    name: str
    time_values: np.ndarray = field(default_factory=lambda: np.array([]))
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
        """将分组数据转换为 pandas DataFrame"""
        data = {self.time_channel_name or "time": self.time_values}
        data.update(self.signals)
        return pd.DataFrame(data)


class MDFDataLoader:
    """MDF 文件加载器（第一阶段：架构骨架）

    提供与 FastDataLoader 一致的属性接口，确保 UI 层无需感知文件格式差异。
    第二阶段将实现完整的 MDF 文件解析和分组加载功能。

    统一接口属性:
        - var_names: 当前分组中的变量名列表
        - units: 当前分组中变量名 -> 单位的映射
        - df: 当前分组数据的 DataFrame
        - df_validity: 数据有效性检查结果
        - time_values: 当前分组的时间轴 Series
        - time_column_name: 时间通道的名称
        - time_axis_label: X 轴标签文本
        - datalength: 数据长度
        - groups: 所有分组的字典
        - group_names: 所有分组的名称列表

    MDF 特有属性:
        - current_group_name: 当前选中分组的名称
    """

    def __init__(self, path: str):
        """初始化 MDFDataLoader

        Args:
            path: MDF 文件路径（.mf4 或 .dat）
        """
        self._path = path
        self._groups: dict[str, GroupData] = {}
        self._current_group_name: Optional[str] = None
        self._load_groups()

    def _load_groups(self):
        """加载 MDF 文件中的所有分组数据（第二阶段实现）

        当前为桩方法（stub），仅验证文件存在性并创建占位分组。
        第二阶段将实现完整的 asammdf 解析逻辑。
        """
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"MDF 文件不存在: {self._path}")

    def select_group(self, group_name: str):
        """切换当前激活的分组

        Args:
            group_name: 目标分组名称

        Raises:
            KeyError: 分组名称不存在
        """
        if group_name not in self._groups:
            raise KeyError(f"分组 '{group_name}' 不存在，可用分组: {list(self._groups.keys())}")
        self._current_group_name = group_name

    @property
    def groups(self) -> dict[str, GroupData]:
        """返回所有分组的字典 {group_name: GroupData}"""
        return self._groups

    @property
    def group_names(self) -> list[str]:
        """返回所有分组的名称列表"""
        return list(self._groups.keys())

    @property
    def current_group_name(self) -> Optional[str]:
        """返回当前选中分组的名称"""
        return self._current_group_name

    @property
    def _current_group(self) -> Optional[GroupData]:
        """返回当前选中分组的 GroupData 对象"""
        if self._current_group_name:
            return self._groups.get(self._current_group_name)
        return None

    @property
    def var_names(self) -> list[str]:
        """返回当前分组中的变量名列表"""
        group = self._current_group
        if group:
            return group.var_names
        return []

    @property
    def units(self) -> dict[str, str]:
        """返回当前分组中变量名 -> 单位的映射"""
        group = self._current_group
        if group:
            return dict(group.units)
        return {}

    @property
    def df(self) -> pd.DataFrame:
        """返回当前分组数据的 DataFrame"""
        group = self._current_group
        if group:
            return group.to_dataframe()
        return pd.DataFrame()

    @property
    def df_validity(self) -> dict:
        """返回数据有效性检查结果

        MDF 数据通常为完整的数值数据，默认全部有效（值为 1）。
        """
        validity = {}
        group = self._current_group
        if group:
            for name in group.signals:
                validity[name] = 1
        return validity

    @property
    def time_values(self) -> pd.Series:
        """返回当前分组的时间轴 Series"""
        group = self._current_group
        if group and len(group.time_values) > 0:
            return pd.Series(group.time_values, name=group.time_channel_name or "time")
        return pd.Series(np.arange(1), name="index")

    @property
    def time_column_name(self) -> Optional[str]:
        """返回时间通道的名称"""
        group = self._current_group
        if group:
            return group.time_channel_name or "time"
        return None

    @property
    def time_axis_label(self) -> str:
        """返回 X 轴标签文本"""
        name = self.time_column_name
        if name:
            return f"{name} (s)"
        return "Index"

    @property
    def datalength(self) -> int:
        """返回当前分组的数据长度"""
        group = self._current_group
        if group:
            return group.datalength
        return 0

    @property
    def time_channels_info(self):
        """返回时间通道信息（保持与 FastDataLoader 接口一致）"""
        return []
