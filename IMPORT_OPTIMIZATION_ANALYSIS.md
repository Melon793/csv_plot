# 导入优化分析报告

> 日期：2026-05-17  
> 目标：优化应用启动速度，减少 `import csv_plot_pyqt6` 的耗时

---

## 1. 现状分析

### 1.1 各模块导入耗时

使用 `subprocess` 独立进程测量，消除缓存影响：

| 导入项 | 耗时 | 占比 |
|--------|------|------|
| `numpy` | 0.180 s | 7.7% |
| `pandas` | 1.103 s | **47.5%** |
| `pyqtgraph` | 0.636 s | **27.4%** |
| `PyQt6.QtWidgets`（全部组件） | 0.046 s | 2.0% |
| `PyQt6.QtGui`（全部组件） | 0.038 s | 1.6% |
| `csv_plot_pyqt6`（完整导入） | 0.333 s * | 14.1% |
| **合计（file-level import）** | **~2.3 s** | 100% |

> *注：`csv_plot_pyqt6` 的 0.333s 是因为 numpy/pandas/pyqtgraph 已在 venv 环境中被缓存，实际冷启动更接近各组件独立耗时之和。

### 1.2 模块依赖图

```
csv_plot_pyqt6.py (主入口, 6693行)
├── import numpy                                    [0.18s, 必需]
├── import pandas                                   [1.10s, 必需]
├── import pyqtgraph                                [0.64s, 必需]
├── from PyQt6.QtCore import ...                    [~0.04s]
├── from PyQt6.QtGui import ...                     [~0.04s]
├── from PyQt6.QtWidgets import ... (32个组件)       [~0.05s]
├── from src.ui.drag_drop import ...                [叶子模块, 轻量]
├── from src.ui.widgets.custom_viewbox import ...    [叶子模块, 轻量]
├── from src.core.config import ...                 [import numpy + pandas, 重]
├── from src.core.types import ...                   [import numpy, 轻]
├── from src.core.scheduler import ...              [import PyQt6, 轻]
├── from src.data.loader import FastDataLoader...   [import numpy + pandas + PyQt6, 重]
├── from src.ui.table_dialog import ...             [import numpy + pandas + pyqtgraph + PyQt6, 重]
├── from src.ui.variable_list import ...            [import PyQt6, 中]
├── from src.ui.mark_stats import ...               [import numpy + PyQt6 + loader, 中]
├── from src.ui.plot_variable_editor import ...     [import pyqtgraph + numpy + PyQt6, 重]
├── from src.ui.dialogs.help import ...              [叶子模块, 轻量]
├── from src.ui.dialogs.layout_input import ...      [叶子模块, 轻量]
├── from src.ui.dialogs.axis import ...              [叶子模块, 轻量]
├── from src.ui.dialogs.time_correction import ...   [叶子模块, 轻量]
└── from src.app.plot_context import ...             [叶子模块, 轻量]
```

### 1.3 重/中/轻分类

| 级别 | 模块 | import 链中的重依赖 |
|------|------|-------------------|
| 🔴 重 | `csv_plot_pyqt6.py` | numpy + pandas + pyqtgraph + 全部PyQt6 |
| 🔴 重 | `src/data/loader.py` | numpy + pandas + PyQt6.QtCore |
| 🔴 重 | `src/ui/table_dialog.py` | numpy + pandas + pyqtgraph + PyQt6(全) |
| 🔴 重 | `src/ui/plot_variable_editor.py` | numpy + pyqtgraph + PyQt6(多) |
| 🟡 中 | `src/ui/variable_list.py` | PyQt6(多) |
| 🟡 中 | `src/ui/mark_stats.py` | numpy + PyQt6 + 引用 loader |
| 🟡 中 | `src/core/config.py` | numpy + pandas |
| 🟢 轻 | `src/core/types.py` | numpy only |
| 🟢 轻 | `src/core/scheduler.py` | PyQt6.QtCore only |
| 🟢 轻 | `src/ui/drag_drop.py` | PyQt6(少) |
| 🟢 轻 | `src/ui/widgets/custom_viewbox.py` | pyqtgraph + PyQt6(少) |
| 🟢 轻 | `src/ui/dialogs/*.py` (4个) | PyQt6(少) |
| 🟢 轻 | `src/app/plot_context.py` | typing only |

---

## 2. 问题定位

### 2.1 🔴 核心问题：主入口一次性加载全部模块

```python
# csv_plot_pyqt6.py L10-L35
from src.ui.table_dialog import DataTableDialog, DropOverlay     # 🔴 触发 pandas
from src.ui.variable_list import MyTableWidget                   # 🟡
from src.ui.mark_stats import MarkStatsWindow                    # 🟡 触发 loader → pandas
from src.ui.plot_variable_editor import PlotVariableEditorDialog  # 🔴 触发 pyqtgraph
from src.ui.dialogs.help import HelpDialog                       # 🟢
from src.ui.dialogs.layout_input import LayoutInputDialog         # 🟢
from src.ui.dialogs.axis import AxisDialog                       # 🟢
from src.ui.dialogs.time_correction import TimeCorrectionDialog   # 🟢
```

**后果**：即使用户只想快速打开应用看到主窗口，也必须等待 `pandas`(1.1s) + `pyqtgraph`(0.64s) 全部加载完成。

### 2.2 🔴 pandas 是最大瓶颈

`pandas` 占 47.5% 的导入时间。它被以下模块需要：
- `src/data/loader.py`（数据加载）— 启动后**立即**需要
- `src/core/config.py`（`_evaluate_float32_safety` 使用 `pd.Series`）
- `src/ui/table_dialog.py`（数据表格）— 用户点击后才需要

但实际上 `config.py` 中对 `pandas` 的使用仅限于 `_evaluate_float32_safety` 函数体内的类型检查，可以在函数内延迟导入。

### 2.3 🔴 pyqtgraph 在 4 个模块中重复导入

- `csv_plot_pyqt6.py`（主入口 — 必须）
- `src/ui/table_dialog.py`（仅用于 `XYScatterPlotDialog` 中的 `pg.plot()`）
- `src/ui/plot_variable_editor.py`（变量编辑器 — 用户操作后才需要）
- `src/ui/widgets/custom_viewbox.py`（右键菜单 — 始终需要）

其中 `table_dialog.py` 和 `plot_variable_editor.py` 的 pyqtgraph 导入可以延迟。

### 2.4 🟢 PyQt6 导入已足够高效

PyQt6.QtWidgets 的全部 32 个组件导入仅需 0.05s，不是主要瓶颈。

### 2.5 ⚠️ `config.py` 模块级副作用

```python
# src/core/config.py L14-L17
_FAULTHANDLER_FILE = None
_ORIGINAL_EXCEPTHOOK = sys.excepthook        # 访问 sys
_ORIGINAL_THREADING_EXCEPTHOOK = getattr(threading, "excepthook", None)
_QT_MESSAGE_HANDLER_INSTALLED = False
```

这些是轻量赋值操作，无实际 I/O 或计算，影响可忽略。

---

## 3. 优化建议

### 策略 A：延迟导入 — 对话框/编辑器模块（推荐，低风险）

**原理**：`DataTableDialog`、`MarkStatsWindow`、`PlotVariableEditorDialog` 等对话框类在用户实际点击按钮前不需要被导入。

**实现**：将主文件中这 5 个对话框类的 import 改为懒加载函数：

```python
# 替换前
from src.ui.table_dialog import DataTableDialog, DropOverlay
from src.ui.mark_stats import MarkStatsWindow
from src.ui.plot_variable_editor import PlotVariableEditorDialog

# 替换后（在主文件底部添加）
def _get_DataTableDialog():
    from src.ui.table_dialog import DataTableDialog
    return DataTableDialog

def _get_MarkStatsWindow():
    from src.ui.mark_stats import MarkStatsWindow
    return MarkStatsWindow
```

**涉及模块**：
| 模块 | 首次使用时机 | 延迟收益 |
|------|------------|---------|
| `src/ui/table_dialog.py` | 双击变量打开数据表 | 🔴 高（跳过 pandas + pyqtgraph） |
| `src/ui/plot_variable_editor.py` | 右键 "Plot Variable Editor" | 🔴 高（跳过 pyqtgraph） |
| `src/ui/mark_stats.py` | 点击统计按钮 | 🟡 中（跳过 loader → pandas） |
| `src/ui/dialogs/help.py` | 点击帮助按钮 | 🟢 低（本身很轻） |
| `src/ui/dialogs/layout_input.py` | 修改布局 | 🟢 低 |
| `src/ui/dialogs/axis.py` | 修改坐标轴 | 🟢 低 |
| `src/ui/dialogs/time_correction.py` | 时间校正 | 🟢 低 |

### 策略 B：config.py 中 pandas 的延迟导入（推荐，低风险）

`src/core/config.py` 导入 `pandas` 仅为了 `_evaluate_float32_safety` 中的类型检查：

```python
# 当前
import pandas as pd

def _evaluate_float32_safety(values: Any) -> tuple[bool, float | None]:
    if isinstance(values, pd.Series):  # 需要 pandas
        ...

# 优化后
def _evaluate_float32_safety(values: Any) -> tuple[bool, float | None]:
    import pandas as pd  # 延迟导入
    if isinstance(values, pd.Series):
        ...
```

同时移除 `config.py` 顶部的 `import pandas as pd`。

**注意**：需要确认哪些模块从 `config.py` import 了来自 pandas 的类型。当前 `config.py` 的 `pd` 仅用于函数体内的 `pd.Series` / `pd.to_numeric`，无模块级引用。

### 策略 C：pyqtgraph 在 table_dialog 中的延迟导入（推荐，低风险）

`src/ui/table_dialog.py` 中 `XYScatterPlotDialog` 使用 `pg.plot()` 创建散点图：

```python
# 优化后：在 XYScatterPlotDialog.__init__ 内部
import pyqtgraph as pg
```

移除模块顶部的 `import pyqtgraph as pg`。

### 策略 D：variable_list 与 table_dialog 循环引用打破（可选，低优先级）

`src/ui/variable_list.py` 导入 `DataTableDialog` 用于：
```python
DataTableDialog.add_variables(var_list, parent=main_window)
dlg = DataTableDialog.popup(var_name, series, parent=main_window)
```

这两处可以改为局部延迟导入。

---

## 4. 预期效果

### 按优先级排序的实施方案

| 阶段 | 措施 | 预期减少 | 风险 | 涉及文件 |
|------|------|---------|------|---------|
| ⭐ P0 | `table_dialog` / `plot_variable_editor` / `mark_stats` 延迟导入 | **~1.0s** | 低 | `csv_plot_pyqt6.py` |
| ⭐ P1 | `config.py` 中 pandas 延迟导入 | **~0.3s** | 低 | `src/core/config.py` |
| ⭐ P2 | `table_dialog.py` 中 pyqtgraph 延迟导入 | —（已在P0覆盖） | 低 | `src/ui/table_dialog.py` |
| P3 | `variable_list.py` 中 `DataTableDialog` 局部导入 | — | 低 | `src/ui/variable_list.py` |
| P4 | 4 个 dialog 文件延迟导入 | ~0.05s | 极低 | `csv_plot_pyqt6.py` |

### 总体预期

| 指标 | 优化前 | 优化后（P0+P1） |
|------|--------|----------------|
| 冷启动 `import csv_plot_pyqt6` | ~2.3s | **~1.1s** |
| 减少 | — | **约 52%** |
| 主窗口显示速度 | 需等待全部模块 | numpy + pyqtgraph + PyQt6 即可 |

---

## 5. 实施风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| 延迟导入导致 `isinstance` 检查失败（类引用不同） | 低 | 使用 `@lru_cache` 缓存导入结果，确保同一函数返回同一类对象 |
| 循环导入（variable_list ↔ table_dialog） | 低 | 已经在 P3 方案中改为函数内局部导入 |
| 用户首次点击按钮时短暂卡顿 | 极低 | pandas/pyqtgraph 在首次使用时加载，但 ~1s 的延迟在用户点击后难以察觉 |
| `DropOverlay` 类在 `create_subplots_matrix` 前就需要 | 中 | `DropOverlay` 在 `MainWindow.__init__` 中直接实例化（L4652），必须保留在顶部 import |

---

## 6. 建议执行顺序

```
1. 审核本文档
2. 执行 P0: 主文件延迟导入 table_dialog / plot_variable_editor / mark_stats
3. 执行 P1: config.py pandas 延迟导入
4. 编译 + 完整功能回归测试
5. 测量实际启动时间改善
6. （可选）执行 P2-P4
```
