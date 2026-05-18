# MDF 枚举类型通道（ValueToTextConversion）支持方案

> **状态**: 待 Review  
> **日期**: 2026-05-17  

---

## 一、背景与问题

MDF 文件中某些通道使用 `ValueToTextConversion`（枚举/文本映射），例如原始值 `0→"Idle"`, `1→"Running"`。当前代码存在三个问题：

| 问题 | 现象 | 根因 |
|---|---|---|
| **P1: 无法绘图** | 枚举通道无法绘制曲线 | `mdf.get_group()` 返回 converted 后的 bytes/str 值，`get_value_from_name` 中 `pd.to_numeric` 全部转为 NaN |
| **P2: 无原始值** | 只存储了 converted 值，丢失了数值形式的原始信号 | `GroupData.signals` 只存 `group_df` 的列值 |
| **P3: 显示 b'xxx'** | cursor label 显示 `b'True'`, `b'speed control'` 而非 `True`, `speed control` | MDF 字符串以 bytes 存储，asammdf 未自动 decode |

---

## 二、推荐方案总览

**核心思路**：在加载阶段区分 conversion 类型，数值通道保持现有逻辑；枚举通道同时存储 **raw 整数**（用于绘图）和 **text labels**（用于 cursor 显示）。

```
┌─────────────────────────────────────────────────────┐
│                   mdf_loader.py                      │
│                                                      │
│  _load_groups():                                     │
│    for each channel:                                 │
│      if conversion is ValueToTextConversion:         │
│        signals[ch] ← ch.raw_samples (float64)  ←绘图 │
│        text_labels[ch] ← decode(ch.samples)    ←显示 │
│        conversions[ch] ← ch.conversion          ←备用│
│      else:                                           │
│        signals[ch] ← group_df[col]       (不变)      │
│                                                      │
│  get_value_from_name() 新增返回 text_labels          │
│  get_var_metadata() 新增返回 has_enum_conversion     │
├─────────────────────────────────────────────────────┤
│                csv_plot_pyqt6.py                     │
│                                                      │
│  PlotWidget.get_value_from_name():                   │
│    检测 is_mdf + conversion → y_format='enum'        │
│    → 返回 (int_array, 'enum', text_labels)           │
│                                                      │
│  _update_multi_curve_cursor_label():                 │
│    y_format == 'enum' → 查表 y_val→label             │
│  _update_single_curve_cursor_label():                │
│    同上                                              │
└─────────────────────────────────────────────────────┘
```

---

## 三、详细代码改动

### 3.1 `mdf_loader.py` — 数据结构层

#### 3.1.1 `GroupData` 新增字段

文件：[mdf_loader.py:L32-L43](file:///Users/xiaolin/Documents/python_repo/csv_plot/mdf_loader.py#L32-L43)

```python
@dataclass
class GroupData:
    index: int
    time_values: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    time_channel_name: str = ""
    signals: dict[str, np.ndarray] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    # ========== 新增 ==========
    text_labels: dict[str, list[str]] = field(default_factory=dict)
    #   channel_name → 转换后的文本标签列表（仅 ValueToTextConversion 通道）
    #   例: {"status": ["Idle", "Running", "Running", "Fault", ...]}
    # ==========================
```

**说明**：
- `conversions` 对象不建议直接存储在 `GroupData` 中（asammdf 内部对象跨线程/跨 pickle 存在问题），改为只存储 `text_labels` 字典，其本质是 `{raw_int: text_label}` 的列表形式。
- 但实际上 cursor label 查表需要 `{int_value → text_label}` 的映射（O(1) 查找），所以 `_load_groups()` 内部构建 `val_to_text` 映射，最终存两种形态：
  - `text_labels: dict[str, dict[int, str]]` — 映射表，用于 cursor label 查表

最终设计：

```python
text_labels: dict[str, dict[int, str]] = field(default_factory=dict)
# channel_name → {0: "Idle", 1: "Running", 2: "Fault", ...}
```

#### 3.1.2 `_load_groups()` — 区分 conversion 类型

文件：[mdf_loader.py:L140-L220](file:///Users/xiaolin/Documents/python_repo/csv_plot/mdf_loader.py#L140-L220)

**改动区域**：`for gi in range(total_groups)` 循环内部

当前代码（L162-L174）：

```python
group_df = mdf.get_group(gi)
# ...
for col in group_df.columns:
    arr = self._safe_to_numpy(group_df[col])
    signals[col] = arr
```

改为：

```python
group_df = mdf.get_group(gi)
group_obj = mdf.groups[gi]

time_values = group_df.index.to_numpy(dtype=np.float64).copy()
time_channel_name = group_df.index.name or "time"

signals: dict[str, np.ndarray] = {}
units: dict[str, str] = {}
text_labels: dict[str, dict[int, str]] = {}        # 新增

# --- 先遍历 channels，区分 conversion 类型 ---
channel_conversion_map: dict[str, bool] = {}
# channel_name → True 表示 ValueToTextConversion

if hasattr(group_obj, 'channels') and group_obj.channels:
    for ch in group_obj.channels:
        is_v2t = (
            ch.conversion is not None
            and hasattr(ch.conversion, 'val_to_text')
        )
        channel_conversion_map[ch.name] = is_v2t

        # --- 构建 text_labels 映射表 ---
        if is_v2t and ch.conversion.val_to_text:
            # ch.conversion.val_to_text 是 {int: bytes|str} 或 {int: str}
            label_map: dict[int, str] = {}
            for int_key, label_val in ch.conversion.val_to_text.items():
                label_map[int_key] = _decode_bytes(label_val)
            text_labels[ch.name] = label_map

        # --- units 提取（不变） ---
        unit_val = None
        if ch.unit and ch.unit.strip():
            unit_val = ch.unit.strip()
        elif ch.conversion is not None:
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
        if ch.name in group_df.columns or is_time_channel:
            units[ch.name] = unit_val
            if is_time_channel and ch.name != time_channel_name:
                units[time_channel_name] = unit_val

# --- 数据提取：枚举通道用 raw_samples，普通通道用 group_df ---
for col in group_df.columns:
    if channel_conversion_map.get(col):
        # ValueToTextConversion: 取原始整数值用于绘图
        ch = None
        if hasattr(group_obj, 'channels') and group_obj.channels:
            for c in group_obj.channels:
                if c.name == col:
                    ch = c
                    break
        if ch is not None and hasattr(ch, 'raw_samples') and ch.raw_samples is not None:
            raw_arr = np.asarray(ch.raw_samples, dtype=np.float64)
            signals[col] = raw_arr
        else:
            # fallback: 从 group_df 尝试转为数值（可能失败）
            arr = self._safe_to_numpy(group_df[col])
            signals[col] = arr
    else:
        arr = self._safe_to_numpy(group_df[col])
        signals[col] = arr

# --- 补全 units（不变） ---
for col in group_df.columns:
    if col not in units:
        units[col] = "-"

gd = GroupData(
    index=gi,
    time_values=time_values,
    time_channel_name=time_channel_name,
    signals=signals,
    units=units,
    text_labels=text_labels,          # 新增
)
self._groups.append(gd)
```

**新增工具函数**：

```python
def _decode_bytes(value):
    """递归解码 bytes 为 str"""
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, (list, tuple)):
        return type(value)(_decode_bytes(v) for v in value)
    return value
```

位置：`_safe_to_numpy` 之前或 `_load_groups` 之前，作为 `MDFDataLoader` 的静态方法或模块级函数。

#### 3.1.3 `get_value_from_name()` — 新增返回 text_labels

文件：[mdf_loader.py:L349-L371](file:///Users/xiaolin/Documents/python_repo/csv_plot/mdf_loader.py#L349-L371)

当前签名：

```python
def get_value_from_name(self, display_name: str):
    # ...
    return time_values, y_data, unit
```

改为同时返回 text_labels 映射表（如果有枚举类型）：

```python
def get_value_from_name(self, display_name: str):
    group_index, time_values, unit = self.get_var_metadata(display_name)
    pure_name = self._resolve_pure_column_name(display_name)
    gd = self._groups[group_index]
    y_data = gd.signals[pure_name]
    text_map = gd.text_labels.get(pure_name, {})
    return time_values, y_data, unit, text_map
```

**兼容性说明**：所有调用 `get_value_from_name` 的地方需要同步适配（见 3.2 节）。

---

### 3.2 `csv_plot_pyqt6.py` — 显示层

#### 3.2.1 `PlotWidget.get_value_from_name()` — 新增 `'enum'` y_format

文件：[csv_plot_pyqt6.py:L6117-L6170](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L6117-L6170)

该方法判断 `dtype_kind` 的逻辑中，新增对 MDF loader 的检测：

```python
def get_value_from_name(self, var_name) -> tuple | None:
    main_window = self.window()
    if var_name in main_window.value_cache:
        return main_window.value_cache[var_name]

    if main_window and hasattr(main_window, 'loader') and main_window.loader is not None:
        loader = main_window.loader
        if hasattr(loader, 'get_series'):
            raw_values = loader.get_series(var_name)
        else:
            raw_values = self.data[var_name]
    else:
        raw_values = self.data[var_name]

    dtype_kind = raw_values.dtype.kind
    y_values = None
    y_format = 'number'

    # ========== 新增：检测 MDF 枚举类型 ==========
    if (
        hasattr(loader, 'get_value_from_name')
        and hasattr(loader, '_groups')
    ):
        full_result = loader.get_value_from_name(var_name)
        text_map = full_result[3] if len(full_result) > 3 else {}
        if text_map:
            # 这是一个 ValueToTextConversion 通道
            # full_result[1] 已经是 raw 整数（float64）
            y_values = full_result[1]  # type: np.ndarray
            y_format = 'enum'
            # 需要把 text_map 带出该方法，让调用方使用
            # 方法：存到 value_cache 时扩展 tuple
            main_window.value_cache[var_name] = (y_values, y_format, text_map)
            return y_values, y_format, text_map
    # =============================================

    if dtype_kind in "iuf":
        y_values = raw_values
    elif dtype_kind == "b":
        y_values = raw_values.astype(np.int32)
    # ... 后续不变 ...
```

**重要问题**：返回值签名会变（从 `(values, format)` 变成有时是 `(values, format, text_map)`），这会影响所有调用方。

**推荐方案**：不改变返回值签名，而是把 `text_map` 缓存到 `main_window` 的一个新属性：

```python
# 在 get_value_from_name 中：
if text_map:
    y_values = full_result[1]
    y_format = 'enum'
    if not hasattr(main_window, '_enum_text_maps'):
        main_window._enum_text_maps = {}
    main_window._enum_text_maps[var_name] = text_map
```

这样调用方只需检查 `y_format == 'enum'`，然后从 `main_window._enum_text_maps[var_name]` 取映射表。

#### 3.2.2 `PlotWidget._prepare_plot_data()` — 传递 y_format

文件：[csv_plot_pyqt6.py:L6429-L6543](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L6429-L6543)

```python
def _prepare_plot_data(self, var_name: str) -> tuple[bool, str, np.ndarray, np.ndarray, str]:
    try:
        result = self.get_value_from_name(var_name=var_name)
        
        # 兼容新格式（enum 类型的 y_format 通过缓存传递，返回值不变）
        y_values, y_format = result

        if y_values is None or len(y_values) == 0:
            return False, f"变量 {var_name} 没有有效数据", None, None, ""
        
        # enum 类型的 y_values 已经是 float64 整数，可以直接绘图
        # 不需要 dtype 转换分支
        
        # ... 后续 dtype 转换逻辑（对 enum 类型，target_dtype 保持 float64 即可）...
```

**说明**：枚举类型的 y_values 是 `np.float64` 整数数组（从 `raw_samples` 提取），可以直接通过现有的 dtype 转换逻辑。

#### 3.2.3 `_update_multi_curve_cursor_label()` — enum 查表转换

文件：[csv_plot_pyqt6.py:L5469-L5639](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L5469-L5639)

在构建 `curves_to_process` 时，传递 enum 映射表：

```python
curves_to_process.append({
    "var_name": var_name,
    "x_data": ci.x_data,
    "y_data": ci.y_data,
    "color": ci.color,
    "y_format": ci.y_format,
    "unit": self.units.get(var_name, ""),
    # ========== 新增 ==========
    "enum_map": main_window._enum_text_maps.get(var_name, {}) if (
        ci.y_format == 'enum' and main_window
    ) else {},
    # ==========================
})
```

在 L5584 的 format 分支中新增：

```python
if y_format == "enum":
    enum_map = curve_data.get("enum_map", {})
    raw_int = int(y_val)
    y_str = enum_map.get(raw_int, str(raw_int))
elif y_format == "s":
    y_str = self.sInt_to_fmtStr(y_val)
elif y_format == "date":
    y_str = self.dateInt_to_fmtStr(y_val)
else:
    y_str = f"{y_val:.5g}"
```

#### 3.2.4 `_update_single_curve_cursor_label()` — 同上

文件：[csv_plot_pyqt6.py:L5108-L5128](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L5108-L5128)

```python
if self.y_format == 'enum':
    enum_map = getattr(window, '_enum_text_maps', {}).get(self.y_name, {})
    y_str = enum_map.get(int(y_val), str(y_val))
elif self.y_format == 's':
    y_str = self.sInt_to_fmtStr(y_val)
elif self.y_format == 'date':
    y_str = self.dateInt_to_fmtStr(y_val)
else:
    y_str = f"{y_val:.5g}"
```

#### 3.2.5 其他需要适配 `y_format == 'enum'` 的位置

| 位置 | 文件行号 | 改动 |
|---|---|---|
| 多曲线添加 `add_variable_to_plot` | ~L6335, ~L6871, ~L6906 | 传递 `y_format='enum'` 到 CurveInfo |
| 单曲线 `plot_selected_var` | ~L6543 | `self.y_format = y_format`（自动处理） |
| 浮点数安全检测 | ~L6550 | enum 的 y_values 是 float64 整数，安全；`is_time_data` 检测不受影响 |

---

## 四、改动文件清单

| 文件 | 改动点 | 行数（估） | 风险 |
|---|---|---|---|
| `mdf_loader.py` | ① `GroupData` 新增 `text_labels` 字段 | +3 | 低 |
| | ② 新增 `_decode_bytes` 静态方法 | +10 | 低 |
| | ③ `_load_groups()` 区分 conversion 类型 + 填充 `text_labels` | ~40 行重构 | **中** |
| | ④ `get_value_from_name()` 返回 text_map | +5 | 低（向后兼容，tuple 扩展） |
| `csv_plot_pyqt6.py` | ⑤ `PlotWidget.get_value_from_name()` 检测 enum + 缓存 text_map | +15 | **中** |
| | ⑥ `_update_multi_curve_cursor_label()` 新增 `'enum'` 分支 | +6 | 低 |
| | ⑦ `_update_single_curve_cursor_label()` 新增 `'enum'` 分支 | +4 | 低 |
| | ⑧ `curves_to_process` 传递 `enum_map` | +5 | 低 |

**总计改动**：约 85 行，集中在 2 个文件的 8 个位置。

---

## 五、兼容性与边界情况

### 5.1 向后兼容

| 场景 | 行为 |
|---|---|
| 普通 linear conversion 通道 | **完全不变**，走现有分支 |
| 无 conversion 的通道 | **完全不变** |
| CSV/DAT 文件（非 MDF） | **完全不变** |
| 已缓存的 `var_name` | 若 `y_format != 'enum'`，走现有逻辑；若 `y_format == 'enum'`，需确保 `text_map` 也在缓存中 |
| `get_value_from_name` 调用方 | 返回值后 2 元组不变，仅内部可能扩展为 4 元组（在 mdf_loader 中） |

### 5.2 边界情况

| 场景 | 处理 |
|---|---|
| `val_to_text` 为空字典 | 不会标记为 enum 通道，走普通逻辑 |
| `raw_samples` 为 None | fallback 到 `group_df` 取值 |
| raw value 超出映射范围（如固件升级后新增枚举值） | `enum_map.get(raw_int, str(raw_int))` 显示原始数值 |
| `val_to_text` 的 key 不是连续整数 | `int(y_val)` 取整后查表，匹配原始 key |
| bytes 编码非 UTF-8 | `decode('utf-8', errors='replace')` 安全处理 |
| 同一通道名跨 Group | `_build_aggregated_properties` 已处理冲突后缀，text_labels 也按聚合名索引 |
| 多个 Group 中同一通道名但不同 conversion | 每个 Group 独立存储 text_labels，切换 Group 时不冲突 |

### 5.3 未覆盖场景

- **`RangeConversion` / `PolynomialConversion`**：这些转换的 converted 值仍是数值，asammdf 的 `mdf.get_group()` 返回的结果可以直接绘图。不需要特殊处理。
- **`ValueToValueConversion`**：同上，结果是数值，不受影响。

---

## 六、测试建议

1. **单元测试**：使用已知有 `ValueToTextConversion` 的 .mf4 文件，验证：
   - `signals[ch]` 是 float64 数组（而非 object 数组）
   - `text_labels[ch]` 中所有字符串不含 `b''` 前缀
   - `get_value_from_name(ch)` 返回正确的 `(time, values, unit, text_map)`

2. **绘图测试**：
   - 枚举通道能否正常绘制曲线（x=时间, y=枚举整数）
   - cursor label 显示的是 `"Idle"` 而非 `0` 或 `b'Idle'`
   - 切换 Group 后枚举通道正常

3. **回归测试**：
   - 普通 CSV 文件加载、绘图、cursor 显示不受影响
   - MDF 普通（非枚举）通道正常
   - 多曲线模式下同时包含枚举通道和普通通道
   - 时间校正（factor/offset）对枚举通道的 x 轴正确

---

## 七、实施顺序

| 步骤 | 内容 | 预计工作量 |
|---|---|---|
| Step 1 | `GroupData` 新增字段 + `_decode_bytes` 工具函数 | 10 min |
| Step 2 | `_load_groups()` 重构 conversion 区分逻辑 | 20 min |
| Step 3 | `get_value_from_name()` 扩展返回值 | 10 min |
| Step 4 | `PlotWidget.get_value_from_name()` 检测 enum + 缓存 | 15 min |
| Step 5 | 双 cursor label 更新方法新增 `'enum'` 分支 | 15 min |
| Step 6 | 多曲线模式 `enum_map` 传递 | 10 min |
| Step 7 | 回归测试 | 20 min |
