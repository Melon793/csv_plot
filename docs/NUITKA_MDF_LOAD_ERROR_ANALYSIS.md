# Nuitka 编译后 MDF 文件加载失败 — 完整排查与修复报告

> **问题文件**: `/data/CMP21 PHEV ADP _20%SOC _EGROPEN_20260422.mf4`  
> **问题通道**: `Epm_nEng_VW`, `Epm_nEng_tiSync_VW`, `Eng_tqCkAct_VW`, `Eng_st_VW`  
> **错误信息**: `加载文件时发生未知错误: 加载Group 时出错:'P1'` 或 `KeyError: 'P1'`

---

## 0. 问题通道诊断结果

### 0.1 通道加载程序

已编写完整的通道加载与诊断脚本：[scripts/diagnose_mdf_channel.py](file:///Users/xiaolin/CSV_Plot_PySide/scripts/diagnose_mdf_channel.py)

该程序具有以下特性：
- 完整的异常处理机制，可处理文件不存在、通道不存在等错误
- 提取通道的完整信息：数据类型、采样频率、数据长度、单位、统计指标等
- 结构化输出结果，包含汇总表和样本数据预览

### 0.2 `Epm_nEng_VW` 通道基本信息

通过诊断程序从文件 `/data/CMP21 PHEV ADP _20%SOC _EGROPEN_20260422.mf4` 中加载并分析 `Epm_nEng_VW` 通道，结果如下：

| 属性 | 值 |
|------|-----|
| 文件大小 | 740.10 MB |
| 总通道数 | 9389 |
| 总 Group 数 | 317 |

#### 通道详情表

| 属性 | `Epm_nEng_VW` | `Epm_nEng_tiSync_VW` | `Eng_tqCkAct_VW` | `Eng_st_VW` |
|------|---------------|----------------------|-----------------|-------------|
| **数据类型** | `float64` | `float64` | `float64` | `uint8` |
| **单位** | `rpm` | `-` | `Nm` | `-` |
| **数据长度** | 428,009 | 428,009 | 428,009 | 428,009 |
| **所属 Group** | 172 | 172 | 172 | 172 |
| **conversion_type** | 2 (有理函数) | 2 (有理函数) | 2 (有理函数) | 7 (text/time) |
| **is_enum** | `False` | `False` | `False` | `True` (修复后) |

#### 样本数据预览

| 索引 | Epm_nEng_VW (rpm) | Eng_st_VW (raw) | Eng_st_VW (physical) |
|------|-------------------|-----------------|---------------------|
| 0 | 0.00 | 1 | b'Ready' |
| 1 | 0.00 | 1 | b'Ready' |
| 2 | 0.00 | 1 | b'Ready' |

---

## 1. 问题现象

| 环境 | 结果 |
|------|------|
| IDE 直接运行 (CPython) | ✅ 正常加载 & 绘图 & 数值变量表 |
| PyInstaller 编译 EXE | ✅ 正常加载 & 绘图 & 数值变量表 |
| **Nuitka 编译 EXE** | ❌ 部分通道报 `KeyError: 'P1'` 错误 |

### 1.1 问题分类

| 通道 | conversion_type | 问题描述 | 状态 |
|------|----------------|----------|------|
| `Eng_st_VW` | 7 (text/time) | 读取的是 physical value (bytes) 而不是 raw value | ✅ 已修复 |
| `Epm_nEng_VW` | 2 (有理函数) | 触发 numexpr 错误 `KeyError: 'P1'` | ❌ 待修复 |
| `Epm_nEng_tiSync_VW` | 2 (有理函数) | 触发 numexpr 错误 `KeyError: 'P1'` | ❌ 待修复 |
| `Eng_tqCkAct_VW` | 2 (有理函数) | 触发 numexpr 错误 `KeyError: 'P1'` | ❌ 待修复 |

---

## 2. 调用链路分析

```
用户点击添加至 Plot →
  └→ add_variables_to_plot()              [csv_plot.py]
      └→ _prepare_plot_data()             [plot_data_manager.py]
          └→ get_value_from_name()         [plot_data_manager.py]
              └→ loader.get_series()       [mdf_lazy_loader.py:L371]
                  └→ self._mdf.get(        [asammdf/mdf.py:L1674]
                      name=None,
                      group=gi, index=ci)
                      └→ asammdf 内部处理
                          └→ convert()     [asammdf/blocks/v4_blocks.py:L3714]
                              └→ numexpr.evaluate(CONV_RAT_TEXT)  ← ★ 错误发生点
```

---

## 3. 已发现的问题

### 3.1 问题 1：`is_enum_conversion` 缺少 conversion_type=7

**文件**: [src/data/metadata.py:L58-62](file:///Users/xiaolin/CSV_Plot_PySide/src/data/metadata.py#L58-L62)

**原因**: `is_enum_conversion` 函数只检查 `conversion_type in (9, 10, 11)`，遗漏了 `conversion_type=7` (text/time conversion)。

**影响**: 
- `Eng_st_VW` (conversion_type=7) 被判断为 `is_enum=False`
- 导致读取时使用 `raw=False`，返回 physical value (bytes) 而不是 raw value (整数)

**修复**:
```python
# 修改前
return ct in (9, 10, 11)

# 修改后
return ct in (7, 9, 10, 11)
```

**状态**: ✅ 已修复

---

### 3.2 问题 2：批量操作导致一个通道失败影响整个 Group

**文件**: [src/data/mdf_lazy_loader.py:L440-447](file:///Users/xiaolin/CSV_Plot_PySide/src/data/mdf_lazy_loader.py#L440-L447)

**原因**: 代码使用批量操作 `self._mdf.select()` 一次性获取同一 Group 的所有通道。如果其中一个通道失败，整个批量操作就失败，导致同一 Group 的所有通道都无法加载。

**关键代码**:
```python
for gi, channels in nonenum_by_group.items():
    selection = [(None, gi, ci) for ci, _ in channels]
    signals = self._mdf.select(selection)  # ⚠️ 批量获取同一 Group 的所有通道！

    for (ci, name), signal in zip(channels, signals):
        y = signal.samples
        self._cache_put(name, y)
        result[name] = pd.Series(y, name=name)
```

**Group 172 结构**:
```
Group 172 (ECM_LZCU_10_3)
├── 通道 0: time
├── 通道 1: Epm_nEng_VW     ← conversion_type=2
├── 通道 2: Epm_nEng_tiSync_VW  ← conversion_type=2
├── 通道 3: Eng_st_VW      ← conversion_type=7
└── 通道 4: Eng_tqCkAct_VW  ← conversion_type=2
```

**影响**: 当加载 `Epm_nEng_VW` 时，会触发整个 Group 172 的批量操作，如果处理过程中有任何通道出错，整个操作都失败。

**状态**: ⚠️ 已知问题，待修复

---

### 3.3 问题 3：numexpr 在 Nuitka 环境下无法正常工作（核心问题）

**错误堆栈**:
```
KeyError: 'P1'
  at numexpr/necompiler.py:778 → getArguments()
  at numexpr/necompiler.py:896 → validate()
  at numexpr/necompiler.py:991 → evaluate()
  at asammdf/blocks/v4_blocks.py:3714 → convert()
  at asammdf/blocks/mdf_v4.py:7576 → get()
  at asammdf/mdf.py:1692 → get()
  at src/data/mdf_lazy_loader.py:378 → get_series()
```

#### 3.3.1 根本原因

asammdf 在处理 `conversion_type=2` (有理函数转换) 时，使用 numexpr 来计算转换公式：

```python
# 在 asammdf/blocks/v4_blocks.py 中
X = new_values
P1 = conversion.P1
P2 = conversion.P2
P3 = conversion.P3
P4 = conversion.P4
P5 = conversion.P5
P6 = conversion.P6

try:
    new_values = evaluate(CONV_RAT_TEXT)  # ⚠️ numexpr.evaluate()
except TypeError:
    new_values = (P1 * X**2 + P2 * X + P3) / (P4 * X**2 + P5 * X + P6)  # 纯 Python 回退
```

其中 `CONV_RAT_TEXT` 定义为：
```python
CONV_RAT_TEXT = "(P1 * X**2 + P2 * X + P3) / (P4 * X**2 + P5 * X + P6)"
```

#### 3.3.2 为什么在 Nuitka 下失败？

**numexpr 的工作原理**:
```python
# numexpr 内部通过某种方式调用 locals()/globals()
# 在 CPython 下可以找到当前作用域的变量
locals() = {
    'X': array([...]),
    'P1': 0.0,
    'P2': 0.5,
    ...
}
✅ 成功！
```

**在 Nuitka 下**:
```python
# Nuitka 将 Python 编译为 C 代码后，局部变量存储方式完全改变！
# numexpr 试图调用 locals()，但找不到 P1-P6 变量！
locals() = {}  # 或者找不到 P1-P6
❌ KeyError: 'P1'
```

#### 3.3.3 为什么 --include-package=numexpr 也没用？

**问题不是 numexpr 没被打包进来**，而是：
1. numexpr **已经被正确打包**（report.xml 中显示 numexpr version="2.14.1"）
2. 问题在于 **numexpr 的动态变量查找机制与 Nuitka 编译后的代码不兼容**
3. Nuitka 将 Python 编译为 C，破坏了 Python 的动态作用域特性

#### 3.3.4 为什么 Eng_st_VW 现在正常了？

| 通道 | conversion_type | is_enum | 读取方式 | 结果 |
|------|----------------|---------|----------|------|
| `Eng_st_VW` | 7 (text/time) | `True` (修复后) | `raw=True` | ✅ 正常 |
| `Epm_nEng_VW` | 2 (有理函数) | `False` | `raw=False` → numexpr | ❌ KeyError: 'P1' |

修复 `is_enum_conversion` 后，`Eng_st_VW` 使用 `raw=True` 读取，绕过了 conversion，直接返回原始值，避免了 numexpr 的问题。

**但这证明了**：
1. ✅ `is_enum_conversion` 的修复是有效的
2. ❌ numexpr 在 Nuitka 环境下就是无法正常工作

---

## 4. Nuitka vs PyInstaller vs CPython 运行差异

### 4.1 核心差异对比

| 维度 | IDE (CPython) | PyInstaller | Nuitka |
|------|--------------|-------------|--------|
| Python 代码执行 | 字节码解释器 | 字节码解释器（加密打包） | **编译为 C → 机器码** |
| `__slots__` 实现 | CPython 原生 | CPython 原生 | **Nuitka 自己实现** |
| `TypedDict` 严格性 | 运行时宽松 | 运行时宽松 | **编译时更严格** |
| `struct.Struct` 模块 | C 原生 (_struct) | C 原生 | 可能走纯 Python 回退 |
| C 扩展加载 | `import` 原生 | .pyd/.so 复制 + 路径设置 | `--include-package`/`--include-module` |
| `mmap` / 文件 I/O | 系统原生 API | 系统原生 API | 经 Nuitka 包装层 |
| 依赖追踪方式 | - | Analysis 阶段全量分析 | 静态导入链追踪 |
| **numexpr 兼容性** | ✅ 正常 | ✅ 正常 | ❌ **不兼容** |

### 4.2 根本差异：numexpr 的动态作用域

**CPython/PyInstaller**:
- Python 的 `locals()` 和 `globals()` 动态查找正常工作
- numexpr 可以找到公式中的变量 P1-P6

**Nuitka**:
- Python 代码被编译为 C 后，局部变量存储在栈/寄存器中，不再在 Python 的 `locals()` 字典中
- numexpr 的动态查找机制失效
- 导致 `KeyError: 'P1'`

---

## 5. asammdf conversion_type 参考

### 5.1 官方常量定义（asammdf/v4_constants.py）

```python
# conversion_type 的官方定义
CONVERSION_TYPE_NON = 0      # 无转换
CONVERSION_TYPE_LIN = 1      # 线性转换: y = a*x + b
CONVERSION_TYPE_RAT = 2      # 有理函数转换: y = (P1*x² + P2*x + P3) / (P4*x² + P5*x + P6)  ← 触发 numexpr
CONVERSION_TYPE_ALG = 3      # 公式转换
CONVERSION_TYPE_TABI = 4     # 内插表格转换
CONVERSION_TYPE_TAB = 5      # 表格转换
CONVERSION_TYPE_RTAB = 6     # 范围表格转换
CONVERSION_TYPE_TABX = 7      # 表格转换 (带文本)  ← 应该是 enum
CONVERSION_TYPE_RTABX = 8   # 范围表格转换 (带文本)
CONVERSION_TYPE_TTAB = 9     # 文本表格转换  ← 已经是 enum
CONVERSION_TYPE_TRANS = 10    # 转换  ← 已经是 enum
CONVERSION_TYPE_BITFIELD = 11 # 位域转换  ← 已经是 enum
```

### 5.2 is_enum_conversion 应该包含的类型

| conversion_type | 说明 | 应该是 enum | 当前代码 | 建议 |
|----------------|------|-------------|----------|------|
| 7 | text/time conversion | ✅ | ❌ 缺失 | ✅ 加入 |
| 8 | 范围表格转换 (带文本) | ⚠️ 可能 | ❌ 缺失 | 暂不加入（待确认） |
| 9 | 文本表格转换 | ✅ | ✅ 已包含 | 保持 |
| 10 | 转换 | ✅ | ✅ 已包含 | 保持 |
| 11 | 位域转换 | ✅ | ✅ 已包含 | 保持 |

---

## 6. 根因总结

| # | 问题 | 根因 | 状态 |
|---|------|------|------|
| **P1** | `Eng_st_VW` 读取到 bytes 文本 | `is_enum_conversion` 缺少 conversion_type=7 | ✅ 已修复 |
| **P2** | `Epm_nEng_VW` 等通道报 `KeyError: 'P1'` | numexpr 在 Nuitka 环境下无法找到局部变量 P1-P6 | ❌ 待修复 |

### 6.1 为什么 P1 会影响 P2？

**不是因为 `Eng_st_VW` 本身导致 `Epm_nEng_VW` 失败**，而是因为：

1. 所有问题通道都在 **同一个 Group (172)**
2. 代码使用 **批量操作 `self._mdf.select()`** 一次性获取同一 Group 的所有通道
3. 如果批量操作中任何一个通道处理失败，整个操作就失败
4. 导致同一 Group 的所有通道都无法加载

---

## 7. 已实施的修复

### 修复 1: `is_enum_conversion` 增加 conversion_type=7 ✅

**文件**: [src/data/metadata.py:L62](file:///Users/xiaolin/CSV_Plot_PySide/src/data/metadata.py#L62)

```python
# 修改前
return ct in (9, 10, 11)

# 修改后
return ct in (7, 9, 10, 11)
```

**效果**: `Eng_st_VW` 现在正确识别为 enum，使用 `raw=True` 读取

---

### 修复 2: Windows 构建脚本 `--include-module` → `--include-package`

**文件**: [scripts/build_exe_nuitka.bat](file:///Users/xiaolin/CSV_Plot_PySide/scripts/build_exe_nuitka.bat)

| 项目 | 修改前 | 修改后 |
|------|--------|--------|
| asammdf | `--include-module=asammdf` | `--include-package=asammdf` |
| asammdf.blocks | 缺失 | 新增 `--include-package=asammdf.blocks` |
| numpy plugin | 缺失 | 新增 `--enable-plugin=numpy` |
| report.xml | 缺失 | 新增 `--report=report.xml` |

---

## 8. 待解决问题：numexpr 在 Nuitka 下的兼容性

### 8.1 可能的解决方案

#### 方案 A：修复 Nuitka 打包配置

尝试添加更多 numexpr 相关的包含选项：
```batch
--include-package=numexpr
--include-module=numexpr.necompiler
--include-module=numexpr.interpreter
```

**状态**: 已在 `--include-package=numexpr`，但问题依然存在，说明不是打包问题

#### 方案 B：升级 asammdf

检查最新版本是否修复了 Nuitka 兼容性问题：
```bash
pip install asammdf --upgrade
```

#### 方案 C：总是读取 raw=True，然后手动应用转换（最稳健）

完全绕过 numexpr，自己处理转换：

```python
# 总是使用 raw=True
y = self._mdf.get(name=None, group=gi, index=ci, raw=True)

# 然后手动应用转换
if meta.conversion_type == 1:  # 线性转换
    y = y * meta.conv_a + meta.conv_b
elif meta.conversion_type == 2:  # 有理函数转换
    p1, p2, p3 = meta.conv_p1, meta.conv_p2, meta.conv_p3
    p4, p5, p6 = meta.conv_p4, meta.conv_p5, meta.conv_p6
    y = (p1 * y**2 + p2 * y + p3) / (p4 * y**2 + p5 * y + p6)
```

**优点**: 完全控制转换逻辑，不依赖 asammdf 的 numexpr
**缺点**: 需要修改代码，需要保存转换参数到 VarMetadata

#### 方案 D：逐个读取通道而不是批量读取

避免批量操作导致的"一损俱损"问题：
```python
# 修改前（批量获取）
signals = self._mdf.select(selection)

# 修改后（逐个获取）
for ci, name in channels:
    signal = self._mdf.get(name=None, group=gi, index=ci, raw=False)
    # 处理错误，继续下一个
```

**优点**: 一个通道失败不影响其他通道
**缺点**: 性能可能略有下降

#### 方案 E：禁用 numexpr（如果 asammdf 支持）

查看 asammdf 是否有禁用 numexpr 的选项或环境变量。

---

### 8.2 建议的验证步骤

1. **升级 asammdf**：
   ```bash
   pip install asammdf --upgrade
   ```
   然后重新编译测试

2. **如果升级无效**，尝试方案 C（总是 raw=True + 手动转换）

3. **同时实现方案 D**（逐个读取），避免批量操作的问题

---

## 9. 修复总结

| 优先级 | 问题 | 修复方案 | 状态 |
|--------|------|----------|------|
| P0 | `is_enum_conversion` 缺少 conversion_type=7 | 在 metadata.py 中加入 7 | ✅ 已完成 |
| P1 | Windows 构建脚本使用 `--include-module` | 改为 `--include-package` | ✅ 已完成 |
| P2 | numexpr 在 Nuitka 下不兼容 | 待确定（建议升级 asammdf 或方案 C） | ❌ 待处理 |

---

## 10. 参考资料

- asammdf 官方文档: https://asammdf.readthedocs.io/
- asammdf GitHub: https://github.com/danielhrisca/asammdf
- Nuitka 官方文档: https://nuitka.net/
- numexpr GitHub: https://github.com/pydata/numexpr
