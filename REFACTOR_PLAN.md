# csv_plot 模块拆分与目录重组方案

> 当前状态：`csv_plot_pyqt6.py` 共 **10,749 行**，包含 **22 个类**，已初步抽出 4 个模块到 `src/`。

---

## 目 录

1. [现状分析](#1-现状分析)
2. [目标目录结构](#2-目标目录结构)
3. [阶段一：数据层 + 核心工具](#3-阶段一数据层--核心工具-预计净减-800-行)
4. [阶段二：UI 组件模块化](#4-阶段二ui-组件模块化-预计净减-2500-行)
5. [阶段三：主文件瘦身 + 收尾](#5-阶段三主文件瘦身--收尾-预计净减-200-行)
6. [风险与注意事项](#6-风险与注意事项)

---

## 1. 现状分析

```
csv_plot_pyqt6.py (10,749 行)
├── Top matter (1–255)        导入、常量、工具函数
├── AutoDetectError           256
├── FormatInfo                262
├── CurveInfo                 275
├── UnifiedUpdateScheduler    354
├── HelpDialog                420
├── DataLoadThread            459
├── FastDataLoader            530   ← 718 行，纯数据层
├── DropOverlay               1249
├── PandasTableModel          1303
├── CustomDelegate            1346
├── XYScatterPlotDialog       1395
├── DataTableDialog           1451  ← 1,286 行，独立窗口
├── LayoutInputDialog         2738
├── AxisDialog                2781
├── NoHoverDelegate           2865
├── MyTableWidget             3017  ← 597 行，变量列表
├── DraggableGraphicsLayoutWidget  3462  ← 4,122 行 ⚠️ 最大类
├── MarkStatsWindow           7586
├── TimeCorrectionDialog      7687
├── PlotContainerWidget       7734
├── MainWindow                7801  ← 2,229 行
├── PlotVariableEditorDialog  10031 ← 655 行
└── main guard                10688
```

| 已抽取模块 | 位置 |
|-----------|------|
| CursorManager (混入，已弃用) | `src/ui/cursor_manager.py` |
| DragDropHandler 工具函数 | `src/ui/drag_drop.py` |
| CustomViewBox（信号化版本） | `src/ui/widgets/custom_viewbox.py` |
| PlotContext / PlotServices | `src/app/plot_context.py` |

### 拆分原则

1. **不留空壳**：只移走与主类（`MainWindow`、`DraggableGraphicsLayoutWidget`）无明显耦合的独立类
2. **不碰 MRO**：`DraggableGraphicsLayoutWidget` 不再使用混入类（已有教训）
3. **import 集中管理**：每个新模块统一用 `import` 语句在 `csv_plot_pyqt6.py` 顶部引入
4. **最小变更**：只移动代码位置和添加 import，不修改任何业务逻辑

---

## 2. 目标目录结构

```
csv_plot/
├── csv_plot_pyqt6.py          ← 瘦身后 ~2,000 行 (仅 MainWindow + DGWidget + 主程序入口)
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/                   ← 【阶段一新建】
│   │   ├── __init__.py
│   │   ├── config.py           ← 常量、日志配置、_UNIT_KEYWORDS、_evaluate_float32_safety
│   │   ├── types.py            ← AutoDetectError, FormatInfo, CurveInfo
│   │   └── scheduler.py        ← UnifiedUpdateScheduler
│   │
│   ├── data/                   ← 【阶段一新建】
│   │   ├── __init__.py
│   │   └── loader.py           ← FastDataLoader, DataLoadThread
│   │
│   ├── ui/                     ← 已有 + 【阶段二扩展】
│   │   ├── __init__.py
│   │   ├── drag_drop.py        ← ✅ 已有
│   │   ├── cursor_manager.py   ← ✅ 已有（保留备用）
│   │   │
│   │   ├── dialogs/            ← 【阶段二新建】
│   │   │   ├── __init__.py
│   │   │   ├── help.py         ← HelpDialog
│   │   │   ├── xy_scatter.py   ← XYScatterPlotDialog
│   │   │   ├── layout_input.py ← LayoutInputDialog
│   │   │   ├── axis.py         ← AxisDialog
│   │   │   └── time_correction.py ← TimeCorrectionDialog
│   │   │
│   │   ├── table_dialog.py     ← DataTableDialog + PandasTableModel + CustomDelegate + DropOverlay
│   │   ├── variable_list.py    ← MyTableWidget + NoHoverDelegate
│   │   ├── mark_stats.py       ← MarkStatsWindow
│   │   ├── plot_variable_editor.py ← PlotVariableEditorDialog
│   │   │
│   │   └── widgets/            ← ✅ 已有
│   │       ├── __init__.py
│   │       └── custom_viewbox.py
│   │
│   └── app/                    ← ✅ 已有
│       ├── __init__.py
│       └── plot_context.py
```

---

## 3. 阶段一：数据层 + 核心工具（预计净减 ~800 行）

### 3.1 `src/core/config.py` — 常量与工具函数

| 从 csv_plot_pyqt6.py 移出 | 行数 |
|---|---|
| 所有模块级常量（`DEFAULT_PADDING_VAL_X`, `FILE_SIZE_LIMIT_BACKGROUND_LOADING`, …） | ~150 |
| `_UNIT_KEYWORDS` 列表 | ~12 |
| `_evaluate_float32_safety()` 函数 | ~45 |
| `DEBUG_LOG_ENABLED` + `debug_log()` | ~15 |

### 3.2 `src/core/types.py` — 轻量数据类型

| 类 | 行数 |
|---|---|
| `AutoDetectError` | ~6 |
| `FormatInfo` (dataclass) | ~13 |
| `CurveInfo` (dataclass) | ~60 |

### 3.3 `src/core/scheduler.py` — 防抖调度器

| 类 | 行数 |
|---|---|
| `UnifiedUpdateScheduler` (QObject) | ~75 |

### 3.4 `src/data/loader.py` — CSV 数据加载

| 类 | 行数 | 说明 |
|---|---|---|
| `FastDataLoader` | ~718 | 核心：编码检测、schema 推断、chunk 读取 |
| `DataLoadThread` | ~75 | 后台加载线程 |

> **注意**：`FastDataLoader` 被 `MainWindow`、`DataTableDialog`、`DraggableGraphicsLayoutWidget` 多处引用，移出后只需在各自文件中 `from src.data.loader import FastDataLoader`。

### 阶段一影响

- `csv_plot_pyqt6.py`：移除约 800 行，新增 5 行 import
- 新增文件：4 个（`config.py`, `types.py`, `scheduler.py`, `loader.py`）

---

## 4. 阶段二：UI 组件模块化（预计净减 ~2,500 行）

### 4.1 `src/ui/table_dialog.py` — 数据表格系统

| 类 | 行数 |
|---|---|
| `DropOverlay` | ~55 |
| `PandasTableModel` | ~45 |
| `CustomDelegate` | ~50 |
| `DataTableDialog` | ~1,286 |

**依赖**：`FastDataLoader`（已移至 `src.data.loader`）

### 4.2 `src/ui/variable_list.py` — 变量列表面板

| 类 | 行数 |
|---|---|
| `NoHoverDelegate` | ~10 |
| `MyTableWidget` | ~597 |

**依赖**：`parse_var_names_from_mimedata`, `build_var_mimedata`, `create_drag_pixmap`（已在 `src.ui.drag_drop`）

### 4.3 `src/ui/dialogs/` — 独立对话框（5 个）

| 文件 | 类 | 行数 |
|---|---|---|
| `help.py` | `HelpDialog` | ~38 |
| `xy_scatter.py` | `XYScatterPlotDialog` | ~55 |
| `layout_input.py` | `LayoutInputDialog` | ~42 |
| `axis.py` | `AxisDialog` | ~82 |
| `time_correction.py` | `TimeCorrectionDialog` | ~100 |

这 5 个类与主窗口**完全解耦**，只通过构造函数接收参数，是最安全的提取目标。

### 4.4 `src/ui/mark_stats.py` — 标记统计窗口

| 类 | 行数 |
|---|---|
| `MarkStatsWindow` | ~100 |

### 4.5 `src/ui/plot_variable_editor.py` — 变量编辑器

| 类 | 行数 |
|---|---|
| `PlotVariableEditorDialog` | ~655 |

依赖 `DraggableGraphicsLayoutWidget`——通过类型导入即可解决。

### 阶段二影响

- `csv_plot_pyqt6.py`：移除约 2,500 行，新增约 10 行 import
- 新增文件：9 个

---

## 5. 阶段三：主文件瘦身 + 收尾（预计净减 ~200 行）

### 5.1 `csv_plot_pyqt6.py` 最终保留清单

| 内容 | 行数 |
|---|---|
| `from __future__` + 标准库 import + PyQt6 import + src 模块 import | ~30 |
| `resource_path()` + 图标加载 | ~20 |
| `PlotContainerWidget`（轻量 QWidget 包装器） | ~70 |
| `DraggableGraphicsLayoutWidget` 类 | ~1,200 |
| `MainWindow` 类 + 所有方法 | ~600 |
| `main` 函数 + `if __name__` 入口 + pyinstaller 注释 | ~60 |
| **总计** | **~2,000 行** |

### 5.2 收尾检查

| 检查项 | 方法 |
|---|---|
| 无循环导入 | 所有 import 路径为单向：`core` → `data` → `ui` → `app`，`csv_plot_pyqt6.py` 只消费 |
| 语法编译通过 | `python3 -c "import py_compile; py_compile.compile(…)"` 批量验证 |
| 运行时功能完整 | `uv run csv_plot_pyqt6.py` 加载 CSV/MDF 各一次 |
| 无 `get_main_window()` / `get_main_loader()` 残留 | grep 验证 |
| `_UNIT_KEYWORDS` / `_evaluate_float32_safety` 无重复 | grep 验证 |

---

## 6. 风险与注意事项

| 风险 | 缓解措施 |
|------|---------|
| **循环导入** | `FastDataLoader` 不 import MainWindow；所有 UI 模块只被主文件 import，互相之间无交叉依赖 |
| **全局变量的模块级引用** | `DEBUG_LOG_ENABLED` 等常量在 `config.py` 定义，其他模块 `from src.core.config import DEBUG_LOG_ENABLED` |
| **混入类 MRO 问题** | 不再使用混入继承，保持 `DraggableGraphicsLayoutWidget(pg.GraphicsLayoutWidget)` 单继承 |
| **`resource_path()` 保持原位** | 该函数被 `icon.ico` / `icon.icns` 加载使用，留在主文件不动 |
| **pyinstaller 打包兼容** | `--add-data` 和 `--hidden-import` 需补充新模块路径 |

### 不建议在本次提取的内容

| 类/方法 | 原因 |
|---------|------|
| `DraggableGraphicsLayoutWidget` 的方法拆分 | 前面已尝试混入提取失败，MRO 冲突风险极高 |
| `MainWindow` 的方法拆分 | 约 600 行核心，与 DGWidget 紧密耦合（`create_subplots_matrix`、`replots_after_loading` 等），改动风险大于收益 |
| `resource_path` / 图标逻辑 | PyInstaller 依赖，牵一发动全身 |

---

## 执行总结

| 阶段 | 新增文件数 | csv_plot_pyqt6.py 减少行数 | 风险 |
|------|----------|--------------------------|------|
| 一 | 4 | ~800 | 低 |
| 二 | 9 | ~2,500 | 中低 |
| 三 | 0 | ~200 | 低 |
| **合计** | **13** | **~3,500** | — |

最终主文件从 **10,749 行 → ~2,000 行**，同时保留全部功能和打包兼容性。
