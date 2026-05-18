# X轴真实时间架构改造计划

## 〇、核心设计原则

本次架构改造围绕两个核心目标展开：

1. **CSV 默认时间轴机制**：为 CSV 文件输入自动生成从 1 开始、步长为 1、长度与数据点数相等的序列（`1:1:n`）作为默认时间轴，无需依赖数据中已有时间通道。
2. **MDF 数据结构兼容**：重构内部数据表示，使其与 `mdf.get_group().to_dataframe()`（不进行重采样）产生的 DataFrame 结构保持高度一致，确保未来添加 INCA MDF/DAT 文件支持时实现最大兼容性。

---

## 一、问题诊断

### 1.1 根因分析

当前 X 轴系统基于以下核心公式构建：

```
x_display = offset + factor * np.arange(1, datalength + 1)
```

即所有曲线的 X 轴值均为**基于行号的线性序列**，`factor` 和 `offset` 作为全局缩放/平移参数。该设计的根本假设是"所有数据行等间距采样"，并未将时间轴作为一等公民处理。

**核心问题**：时间轴概念未在数据结构层面显式表达。每次需要 X 坐标时，通过 `offset + factor * index` 临时计算，导致：

- 无时间通道的 CSV 文件的行为与有时间通道的文件无法统一
- 无法自然映射 MDF 的 `DataFrame.index`（即 master channel 时间值）
- `factor`/`offset` 承载了过多语义负担（既表示缩放/平移，又隐含表示时间步长）

### 1.2 当前时间通道处理现状

时间通道（如 `Tmod`, `Time`）被当作 **Y 轴数据**处理而非 **X 轴源**：

```python
# get_value_from_name:L5945 - 时间通道被当作 Y 值（Unix时间戳）
elif var_name in self.time_channels_info:
    dt_values = pd.to_datetime(raw_values, format=fmt, errors='coerce')
    y_values = self.datetime_to_unix_seconds(dt_values)  # 转为 Unix 秒
```

时间通道的 Y 值存储为 Unix 时间戳 float64，在绘图时被当作普通变量绘制在 Y 轴上。

---

## 二、MDF 数据结构分析

### 2.1 `mdf.get_group().to_dataframe()` 输出结构

`asammdf` 库中，`mdf.get_group()` 返回一个 channel group，调用 `.to_dataframe()` 后产生如下结构：

```python
import asammdf

mdf = asammdf.MDF("measurement.mf4")
# 获取指定 channel group
group = mdf.get_group(channel_group_index)
df = group.to_dataframe()  # 不进行重采样
```

**返回的 DataFrame 结构**：

| DataFrame 属性 | 内容 | 说明 |
|---------------|------|------|
| `df.index` | 主通道（master channel）时间值 | 通常是 `np.float64`，单位取决于 MDF 文件（秒/毫秒） |
| `df.columns` | 通道名称列表 | 如 `['EngineSpeed', 'CoolantTemp', 'VehicleSpeed']` |
| `df[col]` | 各通道的采样数据 | `np.ndarray` 或 `pd.Series`，与主通道同步采样 |
| `df.index.name` | 主通道名称 | 如 `'Time'`, `'t'` |

**关键特征**：
- **时间轴天然存在于 `df.index` 中**，不需要额外识别
- **所有通道共享同一个时间轴**（同一 channel group 内等步长采样）
- **不同 channel group 可能具有不同的采样频率和长度**

### 2.2 与当前 CSV 加载结果的对比

| 属性 | 当前 FastDataLoader（CSV） | MDF get_group().to_dataframe() |
|------|---------------------------|-------------------------------|
| `df.index` | 默认整数索引 `0, 1, 2, ...` | 主通道时间值（float64） |
| `df.columns` | CSV 列名 | 通道名称 |
| `df[col]` | 列数据 | 通道采样数据 |
| 时间轴来源 | 需通过 `time_channels_info` 从某列提取 | 直接来自 `df.index` |
| 单位信息 | `self.units` 字典（从 CSV 单位行解析） | 需从 channel metadata 单独提取 |
| 多采样率 | 不支持（单一 DataFrame） | 原生支持多 Group，跨 Group 绘图 |

### 2.3 统一数据结构设计

核心思路：**将时间轴作为 DataFrame 的 index 表达**，使 CSV 和 MDF 数据加载器产出结构一致的 DataFrame。

```python
# 目标：统一的数据模型
# CSV 加载后 → DataFrame(index=默认时间轴 1:n, columns=变量名)
# MDF 加载后 → DataFrame(index=主通道时间值, columns=通道名)
```

**统一后的 `_df` 结构**：

```
                Channel_A   Channel_B   Channel_C   ...
index (时间轴)
1.0             0.123       -0.456      1.789       ...   ← CSV 默认时间轴
2.0             0.234       -0.567      1.890       ...
3.0             0.345       -0.678      1.901       ...
...
```

或 MDF 时间轴：

```
                  EngineSpeed  CoolantTemp  VehicleSpeed  ...
index (时间/s)
0.000000          800.0        25.3         0.0           ...
0.010000          810.5        25.3         0.5           ...
0.020000          815.2        25.4         1.2           ...
...
```

---

## 三、系统架构重构方案

### 3.1 新架构总览

```
┌───────────────────────────────────────────────────────────────────┐
│                       DataModel (MainWindow)                       │
│  time_values: np.ndarray  ← 统一时间轴（默认1:n或真实时间值）        │
│  time_column_name: str | None  ← 选作时间轴的列名（None=默认行号）  │
│  time_axis_label: str  ← X轴标签显示文本                           │
│  df_validity: dict  ← 列有效性（不变）                              │
│  units: dict  ← 单位（不变）                                       │
└────────────────────────────┬──────────────────────────────────────┘
                             │ 注入到各 DraggableGraphicsLayoutWidget
                             ▼
┌───────────────────────────────────────────────────────────────────┐
│              DraggableGraphicsLayoutWidget（每个子图）               │
│                                                                    │
│  time_values: np.ndarray | None  ← 共享的时间轴值                   │
│  time_column_name: str | None    ← 时间轴来源列名                   │
│  time_axis_label: str            ← X轴标签                         │
│                                                                    │
│  _get_x_data_for_variable() → 统一返回 X 轴数据                     │
│  _get_effective_x_range()   → 基于时间数组的 min/max                │
└───────────────────────────────────────────────────────────────────┘
```

### 3.2 核心数据模型变更

#### 3.2.1 FastDataLoader 新增/变更属性和方法

| 属性/方法 | 类型 | 说明 |
|----------|------|------|
| `_df` (变更) | `pd.DataFrame` | **Index 改为默认时间轴** `1.0, 2.0, 3.0, ...` 或真实时间值 |
| `default_time_values` (新增) | `np.ndarray` | 默认时间轴：`np.arange(1, n+1, dtype=np.float64)` |
| `time_values` (新增 property) | `np.ndarray` | 当前有效的时间值数组（默认或用户指定） |
| `time_column_name` (新增) | `str \| None` | 被选为时间轴的列名（`None` = 使用默认索引） |
| `time_axis_label` (新增 property) | `str \| None` | **可选**。X轴标签文本，默认 `None`（不显示），仅在用户指定时间列时返回列名 |
| `var_names` (调整) | `list[str]` | 排除时间列后的变量名列表（若某列被指定为时间轴，则从变量列表中移除） |
| `df_validity` (调整) | `dict` | 排除时间列后的有效性字典 |

#### 3.2.2 DraggableGraphicsLayoutWidget 新增/变更属性

| 属性 | 类型 | 说明 | 迁移策略 |
|------|------|------|---------|
| `time_values` (新增) | `np.ndarray \| None` | 真实时间值数组（共享引用） | 初始 None = fallback 到 1:n 索引 |
| `time_column_name` (新增) | `str \| None` | 时间轴来源列名 | None = 默认行号 |
| `time_axis_label` (新增) | `str \| None` | X轴标签（可选） | 默认 `None`（不显示），仅用户指定时间列时返回列名 |
| `factor` (保留) | `float` | 缩放因子 | 默认 `1.0`，时间模式下可用于单位转换 |
| `offset` (保留) | `float` | 偏移量 | 默认 `0.0`，时间模式下可用于零点偏移 |
| `original_index_x` (废弃标记) | `np.ndarray \| None` | 保留向后兼容，逐渐迁移到 time_values | 不再作为主要 X 数据源 |

#### 3.2.3 MainWindow 新增/变更属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `time_column_name` (新增) | `str \| None` | 当前选为时间轴的列名 |
| `time_values` (新增) | `np.ndarray \| None` | 时间轴值数组 |
| `time_axis_label` (新增) | `str` | X轴标签 |

---

## 四、分步实施计划

### Phase 1：CSV 默认时间轴生成（数据模型层）

#### Step 1.1：FastDataLoader 新增 `default_time_values` 属性

**文件**：[csv_plot_pyqt6.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py) — `FastDataLoader` 类

**改动内容**：在 `FastDataLoader` 类中新增属性，生成默认的 `1:1:n` 时间轴。

```python
# FastDataLoader 新增属性和方法

@property
def default_time_values(self) -> np.ndarray:
    """默认时间轴：从 1 开始、步长为 1、长度等于数据点数量的序列（1:1:n）
    
    这是 CSV 文件加载后默认使用的 X 轴值。
    当用户未指定时间通道时，所有曲线共享此时间轴。
    """
    return np.arange(1, self.datalength + 1, dtype=np.float64)

@property
def time_values(self) -> np.ndarray:
    """当前有效的时间轴值
    
    优先级：
    1. 如果用户指定了 time_column_name，返回该列的值（转换为 float64）
    2. 否则返回 default_time_values（1:1:n）
    """
    if self.time_column_name and self.time_column_name in self._df.columns:
        values = pd.to_numeric(self._df[self.time_column_name], errors='coerce')
        return values.to_numpy(dtype=np.float64)
    return self.default_time_values

@property
def time_axis_label(self) -> str:
    """X轴标签文本"""
    if self.time_column_name:
        unit = self._units.get(self.time_column_name, '')
        if unit and unit.strip() and unit != '-':
            return f"{self.time_column_name} ({unit})"
        return self.time_column_name
    return None  # 默认不显示 X 轴标签
```

**初始化时新增属性**：

```python
# FastDataLoader.__init__ 中新增
self.time_column_name: str | None = None  # 默认不指定时间列，使用 1:n
```

#### Step 1.2：调整 `var_names` 属性排除时间列

**改动**：当 `time_column_name` 被设置时，`var_names` 应排除该列，使其不作为 Y 轴变量出现在变量列表中。

```python
@property
def var_names(self) -> list[str]:
    """变量名列表（排除时间列）"""
    cols = self._df.columns.tolist()
    if self.time_column_name and self.time_column_name in cols:
        cols.remove(self.time_column_name)
    return cols
```

#### Step 1.3：调整 `df_validity` 属性排除时间列

```python
@property
def df_validity(self) -> dict:
    """列有效性字典（排除时间列）"""
    validity = dict(self._df_validity)
    if self.time_column_name and self.time_column_name in validity:
        del validity[self.time_column_name]
    return validity
```

---

### Phase 2：`_apply_loader()` 传递时间轴数据

**文件**：[csv_plot_pyqt6.py:L8917-L8963](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L8917-L8963)

**改动**：在 `widget.data = self.loader.df` 基础上增加时间轴注入：

```python
def _apply_loader(self):
    """把 loader 的内容同步到 UI"""
    self.var_names = self.loader.var_names
    self.units = self.loader.units
    self.time_channels_infos = self.loader.time_channels_info
    self.data_validity = self.loader.df_validity
    self.data = self.loader.df
    self.list_widget.populate(self.var_names, self.units, self.data_validity)

    # 移除占位符
    if self.placeholder_label.parent():
        self.placeholder_label.setParent(None)

    # 如果尚未创建子图矩阵，则创建
    if not self.plot_widgets:
        self.create_subplots_matrix(self._plot_row_max_default, self._plot_col_max_default)
        self.set_plots_visible(self._plot_row_current, self._plot_col_current)

    # 更新所有 plot_widgets 的数据
    for container in self.plot_widgets:
        widget = container.plot_widget
        widget.data = self.loader.df
        widget.units = self.loader.units
        widget.time_channels_info = self.loader.time_channels_info
        # ===== 新增：注入时间轴数据 =====
        widget.time_values = self.loader.time_values       # 默认1:n 或用户指定的时间列
        widget.time_column_name = self.loader.time_column_name
        widget.time_axis_label = self.loader.time_axis_label
        # ================================

    self.replots_after_loading()
    # ... 其余代码不变
```

---

### Phase 3：X 轴值计算核心重构（绘图层）

#### Step 3.1：新增 `_get_x_data_for_variable()` 方法

**文件**：[csv_plot_pyqt6.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py) — `DraggableGraphicsLayoutWidget` 类

**算法设计**：

```python
def _get_x_data_for_variable(self, y_data_length: int) -> np.ndarray:
    """
    获取变量的 X 轴数据。

    策略优先级：
    1. 如果 self.time_values 存在且长度 >= y_data_length，截取前 y_data_length 个值
    2. 如果 time_values 不存在或长度不足，fallback：
       offset + factor * np.arange(1, y_data_length + 1)

    这保证：
    - CSV 默认模式：time_values = [1, 2, 3, ..., n]，factor=1.0, offset=0.0
    - MDF 时间轴模式：time_values = df.index，factor=1.0, offset=0.0
    - 缩放偏移后：factor 和 offset 作用于 time_values 实现平移缩放

    Args:
        y_data_length: Y 数据长度，用于截取或生成匹配的 X 数组

    Returns:
        np.ndarray: X 轴数据
    """
    if self.time_values is not None and len(self.time_values) > 0:
        # 时间轴模式：截取与 y 数据匹配的长度
        if len(self.time_values) >= y_data_length:
            return self.time_values[:y_data_length].astype(np.float64)
        else:
            # y 数据比时间轴长（不应出现），填充
            dt = np.median(np.diff(self.time_values)) if len(self.time_values) > 1 else 1.0
            extra = np.arange(len(self.time_values), y_data_length) * dt + self.time_values[-1]
            return np.concatenate([self.time_values, extra]).astype(np.float64)

    # Fallback：纯索引模式（原有行为）
    return self.offset + self.factor * np.arange(1, y_data_length + 1, dtype=np.float64)
```

**关键设计说明**：
- `time_values` 已包含 `factor`/`offset` 的变换结果。当用户调整 factor/offset 时，`time_values` 被整体更新，`_get_x_data_for_variable` 只需直接返回即可。
- 这意味着 `factor`/`offset` 的操作在 `time_values` 层面完成，而非在每个 X 数组生成时计算。
- CSV 默认模式下，`time_values = [1, 2, ..., n]`，完全符合现有的 `factor=1.0, offset=0.0` 行为。

#### Step 3.2：重构 `_prepare_plot_data()` 生成 X 数组

**文件**：[csv_plot_pyqt6.py:L6194-L6256](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L6194-L6256)

**改动**：将 L6250-L6251 的 `np.arange(1, len(y_array)+1)` 替换为调用 `_get_x_data_for_variable()`。

```python
def _prepare_plot_data(self, var_name: str) -> tuple[bool, str, np.ndarray, np.ndarray, str]:
    try:
        y_values, y_format = self.get_value_from_name(var_name=var_name)

        if y_values is None or len(y_values) == 0:
            return False, f"变量 {var_name} 没有有效数据", None, None, ""

        # ... 数据类型转换逻辑不变 ...

        y_array = ...  # 现有逻辑

        if np.all(np.isnan(y_array)):
            return False, f"变量 {var_name} 的数据全为无效值", None, None, ""

        # ===== 替换原有代码 =====
        # 旧：x_array = np.arange(1, len(y_array) + 1, dtype=np.float32)
        # 新：统一通过 _get_x_data_for_variable 获取 X 轴数据
        x_array = self._get_x_data_for_variable(len(y_array))
        # ========================

        return True, "", x_array, y_array, y_format

    except Exception as e:
        return False, f"处理数据时出错: {str(e)}", None, None, ""
```

#### Step 3.3：重构 `plot_variable()` 中 X 值的生成

**文件**：[csv_plot_pyqt6.py:L6258-L6293](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L6258)

**改动**：多曲线模式下不再用 `offset + factor * x_array` 二次变换 X 值（因为 `_get_x_data_for_variable` 已经返回最终 X 值）。

```python
def plot_variable(self, var_name: str, show_duplicate_warning: bool = True) -> bool:
    # ... 验证逻辑 ...

    success, error_msg, x_array, y_array, y_format = self._prepare_plot_data(var_name)
    if not success:
        QMessageBox.warning(self, "错误", error_msg)
        return False

    try:
        if self.is_multi_curve_mode:
            # ===== 变更 =====
            # 旧：x_values = self.offset + self.factor * x_array
            # 新：x_array 已经通过 _get_x_data_for_variable 包含了 factor/offset 变换
            return self.add_variable_to_plot(var_name, x_array, y_array, y_format,
                                             show_duplicate_warning=show_duplicate_warning)
            # ================

        # 单曲线模式
        self.y_format = y_format
        self.y_name = var_name
        # ... 其余代码不变 ...
```

#### Step 3.4：重构 `_sync_data_range()` 的 X 轴范围计算

**文件**：[csv_plot_pyqt6.py:L4072-L4092](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L4072-L4092)

**改动**：统一通过曲线存储的 `x_data` 获取范围。

```python
def _sync_data_range(self):
    # ... 清轴逻辑 ...

    # 获取 x_values
    if self.is_multi_curve_mode and self.curves:
        # 多曲线模式：合并所有曲线的 x_data
        x_arrays = self._collect_visible_curve_arrays('x_data')
        if x_arrays:
            x_values = np.concatenate(x_arrays)
        else:
            return False
    else:
        # 单曲线模式：优先从 curve 获取数据
        if self.curve:
            x_data, _ = self.curve.getData()
            if x_data is not None:
                x_values = x_data
            else:
                return False
        elif self.original_index_x is not None:
            # 向后兼容
            x_values = self.offset + self.factor * self.original_index_x
        else:
            return False

    # ... 其余范围计算逻辑不变 ...
```

**注意**：原来的 `self.offset + self.factor * self.original_index_x` 分支保留作为 fallback，因为 `original_index_x` 可能在初始状态下尚未被 time_values 替换。

#### Step 3.5：重构 `update_time_correction()` 支持时间偏移

**文件**：[csv_plot_pyqt6.py:L5987-L6039](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L5987-L6039)

**改动**：当处于时间轴模式时，`factor`/`offset` 的操作语义从"索引缩放"变为"时间缩放+偏移"。

```python
def update_time_correction(self, new_factor, new_offset):
    self._suppress_pin_update = True
    try:
        old_factor = self.factor
        old_offset = self.offset
        self.factor = new_factor
        self.offset = new_offset

        if self.is_multi_curve_mode:
            for var_name, curve_info in self.curves.items():
                if 'curve' in curve_info and 'x_data' in curve_info:
                    curve = curve_info['curve']
                    old_x = curve_info['x_data']

                    if old_factor != 0:
                        # 反算原始索引
                        original_index = (old_x - old_offset) / old_factor
                    else:
                        original_index = np.arange(1, len(old_x) + 1)

                    new_x = new_offset + new_factor * original_index
                    curve.setData(new_x, curve_info['y_data'])
                    curve_info['x_data'] = new_x
        else:
            if self.original_index_x is not None:
                new_x = new_offset + new_factor * self.original_index_x
                self.curve.setData(new_x, self.original_y)

        # ===== 新增：同步更新 time_values =====
        if self.time_values is not None and len(self.time_values) > 0:
            if old_factor != 0:
                original_index = (self.time_values - old_offset) / old_factor
            else:
                original_index = np.arange(1, len(self.time_values) + 1)
            self.time_values = (new_offset + new_factor * original_index).astype(np.float64)
        # =====================================

        # ... bounds 计算逻辑 ...
    finally:
        # ... 恢复逻辑 ...
```

**关键设计**：`time_values` 同步更新，确保后续通过 `_get_x_data_for_variable()` 获取的 X 值自动反映最新的 factor/offset 变换。

#### Step 3.6：重构 `_update_vline_bounds_from_data()` 使用曲线 x_data

**文件**：[csv_plot_pyqt6.py:L5830-L5887](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L5830-L5887)

**改动**：多曲线模式下优先使用曲线中存储的 `x_data`。

```python
def _update_vline_bounds_from_data(self):
    try:
        # 策略1：多曲线模式 — 从所有曲线合并 x_data 范围
        if self.is_multi_curve_mode and self.curves:
            x_arrays = self._collect_visible_curve_arrays('x_data')
            if x_arrays:
                combined = np.concatenate(x_arrays)
                min_x, max_x = np.nanmin(combined), np.nanmax(combined)
                self._set_vline_bounds([min_x, max_x])
                return min_x, max_x

        # 策略2：单曲线模式 — 使用 original_index_x + factor/offset
        if self.original_index_x is not None and len(self.original_index_x) > 0:
            min_index = np.min(self.original_index_x)
            max_index = np.max(self.original_index_x)
            min_x = self.offset + self.factor * min_index
            max_x = self.offset + self.factor * max_index
            self._set_vline_bounds([min_x, max_x])
            return min_x, max_x

        # Fallback：从实际 curve 数据读取
        # ... 保持不变 ...
    except Exception as e:
        print(f"Warning: Error updating vline bounds: {e}")
        self._set_vline_bounds([None, None])
        return None, None
```

---

### Phase 4：MDF 多 Group 数据结构兼容性保障

#### Step 4.1：GroupData 类定义

**新增类**：`GroupData` — 单个 MDF Channel Group 的完整数据封装。

```python
@dataclass
class GroupData:
    """单个 MDF Channel Group 的完整数据"""
    index: int
    df: pd.DataFrame               # index=时间, columns=通道名
    master_channel_name: str | None # 主通道名称
    
    @property
    def time_values(self) -> np.ndarray:
        return self.df.index.to_numpy(dtype=np.float64)
    
    @property
    def var_names(self) -> list[str]:
        return self.df.columns.tolist()
    
    @property
    def datalength(self) -> int:
        return self.df.shape[0]
    
    @property
    def time_min(self) -> float:
        return float(self.df.index[0])
    
    @property
    def time_max(self) -> float:
        return float(self.df.index[-1])
```

#### Step 4.2：MDFDataLoader 多 Group 实现

**新增文件**：`mdf_loader.py`

**设计**：`MDFDataLoader` 初始化时**全量加载所有 Channel Group**，通过 `_groups: list[GroupData]` 存储。接口与 `FastDataLoader` 对齐，同时提供多 Group 查询能力。

```python
class MDFDataLoader:
    """
    INCA MDF/DAT 文件加载器 —— 多 Group 模式。
    
    构造时加载所有 Channel Group 的数据。
    接口与 FastDataLoader 对齐，上层代码无需区分文件格式。
    """
    
    def __init__(self, file_path: str, *, _progress: callable = None):
        self._path = file_path
        self.file_size = os.path.getsize(file_path)
        
        import asammdf
        mdf = asammdf.MDF(file_path)
        
        if _progress:
            _progress(5)
        
        # ========== 加载所有 Group ==========
        self._groups: list[GroupData] = []
        
        for gi in range(len(mdf.groups)):
            group = mdf.get_group(gi)
            df = group.to_dataframe()  # 不重采样
            
            master_name = df.index.name
            
            self._groups.append(GroupData(
                index=gi,
                df=df,
                master_channel_name=master_name,
            ))
            
            if _progress:
                _progress(5 + (gi + 1) * 80 / len(mdf.groups))
        
        # ========== 构建聚合属性 ==========
        self._build_aggregated_properties()
        
        self.byte_per_line = self._estimate_memory()
        
        if _progress:
            _progress(100)
    
    def _build_aggregated_properties(self):
        """
        构建跨 group 的聚合属性（供 UI 变量列表使用）。
        
        命名规则：仅冲突时追加 _G{group_index}，否则保持纯变量名。
        例如：EngineSpeed 在所有 group 中唯一 → 保持 "EngineSpeed"
             CoolantTemp 出现在 Group0 和 Group1 → "CoolantTemp_G0", "CoolantTemp_G1"
        """
        from collections import Counter
        
        pure_name_occurrences: dict[str, int] = Counter()
        for g in self._groups:
            for col in g.df.columns:
                pure_name_occurrences[col] += 1
        
        all_var_names = []
        all_units = {}
        all_validity = {}
        self._var_to_group: dict[str, int] = {}
        
        for g in self._groups:
            for col in g.df.columns:
                if pure_name_occurrences[col] > 1:
                    # 冲突：加 _G{group_index} 后缀
                    display_name = f"{col}_G{g.index}"
                else:
                    # 无冲突：保持原名
                    display_name = col
                
                all_var_names.append(display_name)
                all_units[display_name] = self._get_unit_for_column(g, col)
                all_validity[display_name] = self._get_validity_for_column(g, col)
                self._var_to_group[display_name] = g.index
        
        self._var_names = all_var_names
        self._units = all_units
        self._df_validity = all_validity
        self._time_channels_info = {}
    
    def _resolve_pure_column_name(self, display_name: str) -> str:
        """
        从显示名中提取纯变量名（去掉 _Gxxx 后缀）。
        用于：statistics 面板标题、legend 标签、复制变量名等场景。
        
        例如 "CoolantTemp_G2" → "CoolantTemp"
             "EngineSpeed"     → "EngineSpeed"
        """
        import re
        m = re.match(r'^(.+)_G\d+$', display_name)
        if m:
            return m.group(1)
        return display_name
    
    def get_group(self, index: int) -> GroupData:
        return self._groups[index]
    
    def get_var_metadata(self, display_name: str) -> tuple[int, np.ndarray, str]:
        """
        获取变量的元数据。
        
        返回:
            (group_index, time_values, unit)
        """
        gi = self._var_to_group[display_name]
        g = self._groups[gi]
        pure_col = self._resolve_pure_column_name(display_name)
        unit = self._get_unit_for_column(g, pure_col)
        return gi, g.time_values, unit
    
    # ===== 聚合属性（与 FastDataLoader 对齐） =====
    
    @property
    def df(self) -> pd.DataFrame | None:
        """MDF 多 Group 无单一 DataFrame，返回 None"""
        return None
    
    @property
    def var_names(self) -> list[str]:
        return self._var_names
    
    @property
    def units(self) -> dict[str, str]:
        return self._units
    
    @property
    def df_validity(self) -> dict:
        return self._df_validity
    
    @property
    def time_channels_info(self) -> dict:
        return self._time_channels_info
    
    @property
    def datalength(self) -> int:
        return max((g.datalength for g in self._groups), default=0)
    
    @property
    def group_count(self) -> int:
        return len(self._groups)
    
    @property
    def time_values(self) -> None:
        """MDF 多 group 模式下不存在单一的 time_values，始终返回 None"""
        return None
    
    @property
    def time_axis_label(self) -> str | None:
        return None  # 默认不显示
    
    @property
    def time_column_name(self) -> None:
        return None
    
    @property
    def path(self) -> str:
        return str(self._path)
    
    @property
    def row_count(self) -> int:
        return self.datalength
    
    @property
    def column_count(self) -> int:
        return len(self._var_names)
    
    def _get_unit_for_column(self, group: GroupData, col: str) -> str:
        try:
            return self._units.get(col, '-')
        except Exception:
            return '-'
    
    def _get_validity_for_column(self, group: GroupData, col: str) -> int:
        return MDFDataLoader._classify_column(group.df[col], col, {})
    
    def _estimate_memory(self) -> float:
        total = sum(g.df.memory_usage(deep=True).sum() for g in self._groups)
        total_rows = sum(g.datalength for g in self._groups)
        return total / max(total_rows, 1)
    
    @staticmethod
    def _extract_units_from_mdf(mdf) -> dict[str, str]:
        units = {}
        try:
            for channel in mdf.channels_db.values():
                name = channel.get('name', '')
                unit = channel.get('unit', '-')
                if name:
                    units[name] = unit or '-'
        except Exception:
            pass
        return units
    
    @staticmethod
    def _classify_column(series, col_name, date_formats) -> int:
        try:
            numeric = pd.to_numeric(series, errors="raise").values
        except (ValueError, TypeError):
            return -1
        if numeric.dtype.kind in 'iu':
            valid = numeric
        else:
            valid = numeric[~np.isnan(numeric)]
        if len(valid) == 0:
            return -1
        if len(series) == 1:
            return 1
        if np.unique(valid).size == 1:
            return 0
        return 1
```

#### Step 4.3：MDF 变量命名规范

**仅冲突时加 `_G{group_index}` 后缀，无冲突则保持纯变量名**。这使大多数通道名称与 INCA/MDA 中看到的一致。

```
假设 MDF 文件结构：
  Group0 (100Hz): EngineSpeed, ThrottlePosition, CoolantTemp
  Group1 (10Hz):  CoolantTemp, OilPressure               ← CoolantTemp 与 Group0 冲突
  Group2 (1Hz):   AmbientTemp                            ← 全部唯一

生成的变量列表:
├── EngineSpeed             ← 唯一，不加后缀
├── ThrottlePosition        ← 唯一
├── CoolantTemp_G0          ← 冲突，加 _G0（Group0 的 CoolantTemp）
├── CoolantTemp_G1          ← 冲突，加 _G1（Group1 的 CoolantTemp）
├── OilPressure             ← 唯一
└── AmbientTemp             ← 唯一
```

**复制变量名规则**：复制操作（如 Statistics 面板标题、legend、export 等）时，通过 `_resolve_pure_column_name()` 去掉 `_Gxxx` 后缀，只展示纯变量名：

```
UI 显示: CoolantTemp_G0 → 复制/导出: CoolantTemp
UI 显示: EngineSpeed    → 复制/导出: EngineSpeed
```

#### Step 4.4：文件格式自动分发

**文件**：[csv_plot_pyqt6.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py) — `MainWindow._load_file` / `_load_sync`

```python
def _load_sync(self, file_path, ...):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in ('.mdf', '.mf4', '.dat'):
        from mdf_loader import MDFDataLoader
        self.loader = MDFDataLoader(file_path, _progress=progress_callback)
    else:
        # 现有 CSV 加载逻辑
        self.loader = FastDataLoader(file_path, ...)
```

#### Step 4.5：MDF 依赖声明

`asammdf` 作为可选依赖。运行时检查导入：

```python
try:
    import asammdf
    MDF_AVAILABLE = True
except ImportError:
    MDF_AVAILABLE = False
```

#### Step 4.6：FastDataLoader 数据对齐保障

CSV 模式下 `time_values = [1, 2, ..., n]`（默认）或用户指定的时间列值。`_get_x_data_for_variable()` 在 CSV 模式下返回统一时间轴数组，与现有行为完全一致。上层代码无需区分 CSV/MDF 即可工作。

---

### Phase 3（续）：跨 Group 绘图支持

#### Step 3.6：重构 `_get_x_data_for_variable()` 支持曲线级 x_data

**文件**：[csv_plot_pyqt6.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py) — `DraggableGraphicsLayoutWidget`

**改动**：使该方法支持两种模式：
- **CSV 模式**：使用统一的 `time_values`（默认 1:n 或用户指定时间列）
- **MDF 多 Group 模式**：每条曲线携带各自 Group 的时间轴

```python
def _get_x_data_for_variable(self, y_data_or_len, var_metadata=None):
    """
    统一 X 轴数据获取入口。

    参数:
        y_data_or_len: Y 值数组或其长度
        var_metadata: 变量元数据（MDF 模式下包含 time_values）

    返回:
        np.ndarray: X 轴数据
    """
    length = len(y_data_or_len) if hasattr(y_data_or_len, '__len__') else y_data_or_len

    if var_metadata is not None and 'time_values' in var_metadata:
        # MDF 多 Group 模式：使用该变量专属的时间轴
        time_vals = var_metadata['time_values']
        return time_vals[:length]

    if self.time_values is not None and len(self.time_values) > 0:
        # CSV 模式（有统一时间轴）
        return self.time_values[:length].astype(np.float64)

    # Fallback：纯索引序列
    return self.offset + self.factor * np.arange(1, length + 1, dtype=np.float64)
```

#### Step 3.7：重构 `get_value_from_name()` 支持 MDF 多 Group

```python
def get_value_from_name(self, var_name: str):
    """
    增强：对 MDF 多 Group 变量返回 dict（含 y_values, group_index, time_values）
    对 CSV 变量保持原有行为（返回 y_values 数组）
    """
    # MDF 模式
    if hasattr(self, 'loader') and isinstance(self.loader, MDFDataLoader):
        gi, time_vals, unit = self.loader.get_var_metadata(var_name)
        pure_col = self.loader._resolve_pure_column_name(var_name)
        y_values = self.loader.get_group(gi).df[pure_col].values
        return {
            'y_values': y_values,
            'group_index': gi,
            'time_values': time_vals,
            'unit': unit,
        }
    
    # CSV 模式（原有逻辑）
    # ... 现有代码 ...
```

#### Step 3.8：重构 `add_variable_to_plot()` 适配多 Group

```python
def add_variable_to_plot(self, var_name):
    """添加变量到绘图区（适配多 Group）"""
    result = self.get_value_from_name(var_name)
    
    if isinstance(result, dict):
        # MDF 多 Group 模式
        y_values = result['y_values']
        x_values = result['time_values'][:len(y_values)]
        unit = result.get('unit', '-')
    else:
        # CSV 模式（原有逻辑）
        y_values = result
        x_values = self._get_x_data_for_variable(y_values)
        unit = self.loader.units.get(var_name, '-')
    
    curve = self.plot_item.plot(
        x=x_values,
        y=y_values,
        pen=color,
        name=var_name,
    )
    
    curve_info = {
        'curve': curve,
        'x_data': x_values,
        'y_data': y_values,
        'var_name': var_name,
        'unit': unit,
    }
    self.curves.append(curve_info)
    
    self._sync_data_range()
    self._update_vline_bounds_from_data()
```

#### Step 3.9：重构 `update_time_correction()` 批量作用于所有曲线

当 plot 中包含多个 Group 的曲线时，`factor` 和 `offset` 的变换应作用于**所有曲线的 x_data**：

```python
def update_time_correction(self, new_factor, new_offset):
    old_factor = self.factor
    old_offset = self.offset
    self.factor = new_factor
    self.offset = new_offset

    if self.is_multi_curve_mode:
        for curve_info in self.curves:
            old_x = curve_info['x_data']
            if old_factor != 0:
                original_x = (old_x - old_offset) / old_factor
            else:
                original_x = np.arange(1, len(old_x) + 1)
            new_x = new_offset + new_factor * original_x
            curve_info['curve'].setData(x=new_x, y=curve_info['y_data'])
            curve_info['x_data'] = new_x
    else:
        if self.original_index_x is not None:
            new_x = new_offset + new_factor * self.original_index_x
            self.curve.setData(new_x, self.original_y)

    # 同步更新 time_values（CSV 模式）
    if self.time_values is not None and len(self.time_values) > 0:
        if old_factor != 0:
            original_index = (self.time_values - old_offset) / old_factor
        else:
            original_index = np.arange(1, len(self.time_values) + 1)
        self.time_values = (new_offset + new_factor * original_index).astype(np.float64)

    self._sync_data_range()
    self._update_vline_bounds_from_data()
```

#### Step 3.10：跨 Group 数据流图解

```
MDF 文件
    ↓ MDFDataLoader.__init__
加载所有 Group:
    GroupData[0].df (index=时间值_100Hz, columns=通道名)
    GroupData[1].df (index=时间值_10Hz,  columns=通道名)
    GroupData[2].df (index=时间值_1Hz,   columns=通道名)
    ↓ _build_aggregated_properties
_var_names = ["EngineSpeed", "CoolantTemp_G0", "CoolantTemp_G1", "AmbientTemp", ...]

用户在同一个 Plot 中添加两条来自不同 Group 的曲线：

添加 "EngineSpeed"（Group0）:
    ↓ loader.get_var_metadata("EngineSpeed")
      → (group_index=0, time_values=[0.00, 0.01, ..., 60.00], unit="rpm")
    ↓ pure_col = "EngineSpeed"
    ↓ y = loader._groups[0].df["EngineSpeed"].values
    ↓ x = time_values[:len(y)]   → [0.00, 0.01, ..., 60.00]
    ↓ curve_0.setData(x=x, y=y)

添加 "CoolantTemp_G1"（Group1）:
    ↓ loader.get_var_metadata("CoolantTemp_G1")
      → (group_index=1, time_values=[0.00, 0.10, ..., 60.00], unit="°C")
    ↓ pure_col = "CoolantTemp"
    ↓ y = loader._groups[1].df["CoolantTemp"].values
    ↓ x = time_values[:len(y)]   → [0.00, 0.10, ..., 60.00]
    ↓ curve_1.setData(x=x, y=y)

结果:
    Plot 上有两条曲线，各有独立的 x_data
    X 轴范围 = [0.00, 60.00]（自动并集）
    两条曲线对齐在同一时间轴上
```

#### Step 3.11：`_update_vline_bounds_from_data()` 多曲线兼容

**文件**：[csv_plot_pyqt6.py:L5830-L5887](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L5830-L5887)

该方法仅处理当前 widget 中 `self.curves` 里的数据，这些数据来自已加载的 Group。不遍历所有 Group，每次调用 O(n)（n = 当前可见曲线数据点总数）。

| 场景 | 数据量 | 耗时 |
|------|--------|------|
| 单曲线 60K 点 | 60K float64 | < 1ms |
| 20 条曲线各 6K 点 | 120K float64 | < 2ms |
| 最坏：20 条曲线各 100K 点 | 2M float64 | ~10ms |

**性能完全可接受，无需修改核心逻辑**。

---

### Phase 5：UI 层适配

#### Step 5.1：X 轴标签显示

**文件**：[csv_plot_pyqt6.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py) — `DraggableGraphicsLayoutWidget.setup_plot_area`

**改动**：使用 `time_axis_label` 设置 X 轴标签。

```python
def setup_plot_area(self):
    # ... 现有初始化 ...

    # 设置 X 轴标签（可选参数，默认不显示以节省空间）
    if hasattr(self, 'time_axis_label') and self.time_axis_label:
        self.plot_item.setLabel('bottom', self.time_axis_label)
    # label 为 None/空 → 不调用 setLabel → pyqtgraph 不分配标签空间
```

#### Step 5.2：时间分辨率自适应 axis label

**功能**：当切换到真实时间轴时，根据数据范围自动选择合适的 X 轴刻度格式。

```python
def _update_axis_label(self):
    """根据 time_values 的范围自动选择时间标签（仅在用户指定时间列时调用）"""
    if self.time_values is None or len(self.time_values) < 2:
        return  # 不设置 label，保持不显示

    total_range = self.time_values[-1] - self.time_values[0]
    
    if total_range <= 0:
        return  # 不设置 label
    elif total_range < 1e-3:
        self.plot_item.setLabel('bottom', '时间 (μs)')
    elif total_range < 1:
        self.plot_item.setLabel('bottom', '时间 (ms)')
    elif total_range < 3600:
        self.plot_item.setLabel('bottom', '时间 (s)')
    elif total_range < 86400:
        self.plot_item.setLabel('bottom', '时间 (min)')
    else:
        self.plot_item.setLabel('bottom', '时间 (h)')
```

---

### Phase 6：兼容性与稳定性保障

#### Step 6.1：默认时间轴完全向后兼容

**验证标准**：加载 CSV 后（无时间通道），所有行为与现有实现完全一致：
- `time_values = [1.0, 2.0, 3.0, ..., n]`
- `factor = 1.0`, `offset = 0.0`
- `_prepare_plot_data` 中的 `x_array = _get_x_data_for_variable(n)` → 返回 `[1, 2, ..., n]`
- `_sync_data_range` → `min_x=1, max_x=n`
- `update_time_correction` → 行为不变（只是公式中的 index 现在是 time_values）

#### Step 6.2：factor/offset 参数语义

| 场景 | factor 语义 | offset 语义 |
|------|-----------|------------|
| CSV 默认（1:n 时间轴） | 索引缩放（现有行为） | 索引平移（现有行为） |
| CSV 时间通道 | 时间单位转换因子 | 零点偏移 |
| MDF 时间轴 | 时间单位转换因子 | 零点偏移 |

用户通过 UI 调整 factor/offset 时：
- **CSV 默认模式下**：与现有行为完全一致（`new_x = offset + factor * index`）
- **时间轴模式下**：对时间值做缩放和偏移

#### Step 6.3：replots_after_loading 兼容

**文件**：[csv_plot_pyqt6.py:L9781-L9963](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L9781-L9963)

**改动**：重绘时保留时间轴上下文。

```python
def replots_after_loading(self):
    # ... 现有初始化 ...

    # 确保所有 widget 拥有最新的 time_values
    if hasattr(self.loader, 'time_values'):
        for container in self.plot_widgets:
            widget = container.plot_widget
            widget.time_values = self.loader.time_values
            widget.time_column_name = self.loader.time_column_name
            widget.time_axis_label = self.loader.time_axis_label
```

#### Step 6.4：XLink 同步兼容性

**问题**：不同子图可能加载了不同长度的数据，XLink 强制它们共享相同的 X 轴范围。

**解决方案**：
- XLink 保持为 **X 轴视图范围同步**（不变）
- 当用户切换到时间轴模式时，全局 X 轴范围 = 所有子图数据的 `x_min`/`x_max` 的并集
- 某个子图的数据超出全局 X 范围时，数据被裁剪显示但可以滚入视野
- 由于所有子图共享同一 `time_values`，不同子图之间的 XLink 天然一致

#### Step 6.5：DataTableDialog 兼容

**文件**：[csv_plot_pyqt6.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py) — `DataTableDialog` 相关代码

**改动**：时间列在表格中正常显示为数值列，不做特殊处理。用户可通过 UI 选择器将其指定为时间轴。当某列被选为时间轴后，`var_names` 自动排除该列，使其不出现在变量列表中。

#### Step 6.6：索引模式完全向后兼容

所有改动必须保证 **当 `time_values` 为默认 1:n 时，行为与现有实现完全一致**。验证标准：
- 加载 CSV 后，不选择任何时间通道 → `time_values = [1, 2, ..., n]`，`factor=1.0`, `offset=0.0`
- `_prepare_plot_data` → `x_array = _get_x_data_for_variable(n)` → 返回 `[1, 2, ..., n]`
- `_sync_data_range` → `min_x=1, max_x=n`，与当前行为一致
- `update_time_correction` → `new_x = offset + factor * index`，与当前行为一致

---

## 五、数据结构对比总览

### 5.1 统一前后对比

| 层面 | 当前实现 | 改造后 |
|------|---------|--------|
| **X 轴值来源** | `offset + factor * np.arange(1, n+1)` 每次现算 | `self.time_values` (默认 1:n 或真实时间值) 集中存储 |
| **FastDataLoader._df.index** | 默认整数 0, 1, 2, ... | 不变（仍为默认整数索引，性能考虑不做修改） |
| **时间轴表达** | 无显式表达 | `FastDataLoader.time_values` 属性 |
| **MDF 兼容性** | 不支持 | `MDFDataLoader.time_values` = `df.index.to_numpy()` |
| **变量列表** | 所有列（含时间列） | 自动排除时间列（`time_column_name`） |
| **factor/offset 语义** | 全局索引缩放/平移 | CSV 默认：索引缩放/平移；时间模式：时间缩放/偏移 |
| **代码复杂度** | 散落在各处的 `offset + factor * index` | 统一归一到 `_get_x_data_for_variable()` |

### 5.2 MDF 与 CSV 数据流对比

```
CSV 文件
    ↓ FastDataLoader
DataFrame(index=0:n-1, columns=变量名)
    ↓ .time_values → np.arange(1, n+1)   ← 默认时间轴
    ↓ _apply_loader 注入所有 widget
    ↓ _get_x_data_for_variable(len) → time_values[:len]
    绘图


MDF 文件（多 Group）
    ↓ MDFDataLoader.__init__
加载所有 Group:
    GroupData[0].df (index=时间值_100Hz, columns=通道名)
    GroupData[1].df (index=时间值_10Hz,  columns=通道名)
    ↓ _build_aggregated_properties
var_names = ["EngineSpeed", "CoolantTemp_G0", "CoolantTemp_G1", ...]
    ↓ MDFDataLoader.time_values → None  （多 Group 无统一时间轴）
    ↓ get_var_metadata("EngineSpeed") → (gi=0, time_vals=[0.00,0.01,...], unit="rpm")
    ↓ Widget._get_x_data_for_variable(y_len, var_metadata) → time_vals[:y_len]
    绘图（每条曲线携带独立 x_data）

关键对齐点：
- CSV: loader.time_values 返回统一的 np.ndarray，所有曲线共享
- MDF: loader.time_values 返回 None，每条曲线通过 get_var_metadata() 获取独立 x_data
- 上层 add_variable_to_plot() 通过 get_value_from_name() 统一分发
```

---

## 六、测试计划

### 6.1 单元测试

| 测试场景 | 输入 | 期望输出 | 验证点 |
|---------|------|---------|--------|
| T1: CSV 默认时间轴 | 任意 CSV，n 行 | `time_values = [1.0, 2.0, ..., n]` | 1:1:n 序列正确 |
| T2: CSV 指定时间列 | CSV 包含 `Tmod` 列，值 `0.0, 10.0, 20.0...` | `time_values = [0.0, 10.0, 20.0, ...]` | 用户指定列作为时间轴 |
| T3: var_names 排除时间列 | `time_column_name = 'Tmod'` | `var_names` 不含 `'Tmod'` | 时间列不出现在变量列表中 |
| T4: X 轴值生成（默认） | `time_values = [1, 2, 3]`, `len(y)=3` | `_get_x_data_for_variable(3) = [1, 2, 3]` | 与现有行为一致 |
| T5: X 轴值生成（时间模式） | `time_values = [0.0, 10.0, 20.0]`, `len(y)=3` | `_get_x_data_for_variable(3) = [0.0, 10.0, 20.0]` | X 值为真实时间 |
| T6: factor/offset 变换 | `time_values = [1, 2, 3]`, `factor=2.0, offset=0` | `update_time_correction` 后 `time_values = [2, 4, 6]` | time_values 同步更新 |
| T7: MDF 变量命名（无冲突） | MDF 单 Group 唯一变量名 `EngineSpeed` | `var_names` 含 `"EngineSpeed"`（无后缀） | 纯变量名保持一致 |
| T8: MDF 变量命名（冲突） | MDF 两个 Group 均有 `CoolantTemp` | `var_names` 含 `"CoolantTemp_G0"`, `"CoolantTemp_G1"` | 冲突时追加 `_G{index}` 后缀 |
| T9: `_resolve_pure_column_name` | 输入 `"EngineSpeed_G2"` | 输出 `"EngineSpeed"` | 去掉 `_Gxxx` 后缀还原纯变量名 |

### 6.2 集成测试

| 测试场景 | 步骤 | 期望结果 |
|---------|------|---------|
| I1: CSV 加载默认时间轴 | 加载 CSV，添加变量 | X 轴显示 1:n 序号，**无 X 轴标签** |
| I2: 选择时间列作为 X 轴 | 1) 加载 CSV 2) 选择 `Tmod` 列作为时间轴 3) 添加变量 | X 轴变为 Tmod 的值，标签显示 `Tmod`（带单位），`Tmod` 从变量列表消失 |
| I3: 时间轴模式下的缩放 | 1) 时间轴模式 2) 滚轮缩放 | X 轴以时间值为单位缩放 |
| I4: factor/offset 调整 | 1) 时间轴模式 2) 输入 factor=2.0 | X 轴值变为原来的 2 倍 |
| I5: 切回默认行号模式 | 1) 取消时间列选择 | X 轴恢复为 1:n，标签消失，行为与旧版一致 |
| I6: MDF 多 Group 加载 | 1) 加载 .mdf 文件（含 3 个 Group） 2) 检查变量列表 | 变量名含冲突后缀 `_G{index}`；`time_values` 为 `None`；get_var_metadata 返回各自时间轴 |
| I7: MDF 跨 Group 绘图 | 1) 加载 MDF 2) 同一 plot 添加 Group0 和 Group1 的变量 | 两条曲线各有独立 x_data，X 轴范围自动并集 |
| I8: MDF conflict suffix 复制 | 1) 选中 `"CoolantTemp_G0"` 2) 执行复制名称操作 | 复制结果为 `"CoolantTemp"`（纯变量名） |
| I9: 多子图 + 时间轴 | 1) 启用时间轴 2) 每个子图添加不同变量 | 所有子图使用同一时间轴，XLink 正常工作 |

### 6.3 回归测试

| 测试场景 | 输入 | 验证点 |
|---------|------|--------|
| R1: 小 CSV 加载 | 5行×2列 | var_names, units, datalength 正确 |
| R2: 大 CSV 加载 | 733列, 18K行 CSV | 加载成功，无 OOM，默认 time_values 正确 |
| R3: 时间格式 CSV | 包含 `%d/%m/%Y` 列 | date_formats 正确，时间列可被选为时间轴 |
| R4: 多曲线添加/删除 | 添加 3 条曲线，删除 1 条 | 剩余 2 条 X 轴正确显示 |
| R5: 标记区域统计 | 添加标记区域 | 统计窗口正常打开，X 值为时间轴值 |
| R6: Cursor 锁定 | 2-cursor 模式 | 锁定位置在时间轴值上正确 |
| R7: Plot 重加载 | 加载新文件 | 旧 time_values 清理，新 time_values 正确注入 |

---

## 七、风险点与缓解措施

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 时间通道与 Y 值数组长度不一致 | 个别曲线 X/Y 长度不匹配 | `_get_x_data_for_variable` 中使用 `min(len(time_values), y_length)` 截取 |
| DataFrame index 重设性能 | 大文件 index 赋值耗时 | **不修改 `_df.index`**，仅通过 `time_values` 属性按需返回 |
| 默认时间轴 float64 内存 | n=1M 时约 8MB | 仅为属性值，与现有 `np.arange(1, n+1)` 内存开销相当 |
| MDF 文件时间通道为相对时间（ms）非绝对时间（ns） | 时间显示不正确 | 检测时间值域，通过 factor 自动调整；用户可手动调整 |
| asammdf 库可用性 | 用户未安装 asammdf | 运行时检查导入，若缺失则跳过 MDF 支持 + UI 提示 |
| XLink 同步下不同子图 X 轴冲突 | 主/从视图范围不同步 | 所有子图共享统一的 `time_values` + 相同的 `time_column_name` |
| 多 Group 全量加载内存占用 | Group 过多时总 DataFrame 内存可能较大 | 全量加载所有 Group；按需延迟加载列数据；监控内存使用 |

---

## 八、实施优先级

```
优先级 1（必须）：Phase 1 Step 1.1 → 默认时间轴 1:n 属性
优先级 2（必须）：Phase 1 Step 1.2-1.3 → var_names/df_validity 排除时间列
优先级 3（必须）：Phase 3 Step 3.1-3.11 → X轴值计算核心重构 + 多 Group 绘图支持
优先级 4（必须）：Phase 2 + Phase 6 → _apply_loader 注入 + 向后兼容保障
优先级 5（核心）：Phase 4 Step 4.1-4.3 → MDFDataLoader 多 Group 全量加载 + GroupData + 变量命名
优先级 6（核心）：Phase 3 Step 3.6-3.11 → 跨 Group 绘图（get_value_from_name / add_variable_to_plot / update_time_correction）
优先级 7（重要）：Phase 5 → X轴标签（可选）与 UI 适配
优先级 8（重要）：Phase 4 Step 4.4 → 文件格式自动分发

最小可行实施路径（MVP）：
  Phase 1 → Phase 3 → Phase 2 → 即获得 CSV 默认时间轴 + 统一 X 数据获取
  Phase 4 为 MDF 多 Group 核心支持，Phase 5 为 UI 打磨
```

---

## 九、代码变更清单

### 9.1 FastDataLoader 变更

| 位置 | 变更类型 | 说明 |
|------|---------|------|
| `__init__` | 新增属性 | `self.time_column_name: str \| None = None` |
| 新增 property | `default_time_values` | 返回 `np.arange(1, n+1, dtype=np.float64)` |
| 新增 property | `time_values` | 返回用户指定时间列或默认 1:n |
| 新增 property | `time_axis_label` | 返回 X 轴标签文本 |
| `var_names` (property) | 修改 | 当 `time_column_name` 被设置时排除该列 |
| `df_validity` (property) | 修改 | 当 `time_column_name` 被设置时排除该列 |

### 9.2 DraggableGraphicsLayoutWidget 变更

| 位置 | 变更类型 | 说明 |
|------|---------|------|
| `__init__` / `setup_ui` | 新增属性 | `self.time_values`, `self.time_column_name`, `self.time_axis_label` |
| 新增方法 | `_get_x_data_for_variable(y_data_length)` | 统一 X 轴数据获取入口 |
| `_prepare_plot_data` | 修改 L6251 | `x_array = self._get_x_data_for_variable(len(y_array))` |
| `plot_variable` | 修改 L6288 | 多曲线模式下直接使用 x_array（不再 `offset + factor * x_array`） |
| `_sync_data_range` | 修改 | 优先从曲线 `x_data` 获取范围 |
| `update_time_correction` | 修改 | 同步更新 `self.time_values` |
| `_update_vline_bounds_from_data` | 修改 | 多曲线优先使用曲线 `x_data` |

### 9.3 MainWindow 变更

| 位置 | 变更类型 | 说明 |
|------|---------|------|
| `_apply_loader` | 新增注入 | `widget.time_values`, `widget.time_column_name`, `widget.time_axis_label` |
| `replots_after_loading` | 新增保留 | 重绘时同步 `time_values` 到所有 widget |

### 9.4 MDFDataLoader 多 Group 变更（新增）

| 位置 | 变更类型 | 说明 |
|------|---------|------|
| `_groups` | 新增属性 | `list[GroupData]`，初始化时加载所有 Channel Group |
| `get_group(index)` | 新增方法 | 返回指定 Group 的 `GroupData` |
| `get_var_metadata(name)` | 新增方法 | 返回 `(group_index, time_values, unit)` 三元组 |
| `_build_aggregated_properties()` | 新增方法 | 跨 Group 变量名去重（冲突时追加 `_G{index}`） |
| `_resolve_pure_column_name(name)` | 新增方法 | 去掉 `_Gxxx` 后缀，还原纯变量名 |
| `_var_to_group` | 新增属性 | `dict[str, int]`，display_name → group_index 映射 |
| `time_values` | = `None` | 多 Group 无统一时间轴，始终返回 None |
| `time_axis_label` | = `None` | 默认不显示 X 轴标签 |
| `group_count` | 新增属性 | Group 总数 |

### 9.5 新增类

| 类 | 文件 | 说明 |
|------|------|------|
| `GroupData` | `mdf_loader.py` | 单个 MDF Channel Group 的完整数据封装（df, time_values, var_names, units, datalength, master_channel_name） |
| `MDFDataLoader` | `mdf_loader.py` | 多 Group MDF 加载器，接口与 `FastDataLoader` 对齐，额外提供 `get_var_metadata` 等 |
