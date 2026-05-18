# 重构阶段零（Phase 0）执行总结

> 执行日期：2026-05-18  
> 对应计划：DraggableGraphicsLayoutWidget_MainWindow_Refactor_Plan.md v3.0 §五～§六

---

## 一、目标

完成「前置准备」阶段的所有步骤，为后续管理器拆分建立基础设施：

1. 扩展 `PlotServices` 协议，覆盖所有 `self.window()` 跨对象访问
2. 将 `plot_context` 从后置注入改为构造时注入
3. 大规模替换 `self.window()` / `pw.window()` → `plot_context` 调用
4. 创建管理器基类（含 weakref + 生命周期钩子）
5. 迁移 `PlotContainerWidget` 到独立模块
6. 标记旧 Mixin 为弃用

---

## 二、完成步骤

### Step 0a — 扩展 PlotServices 协议

**文件**：[src/app/plot_context.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/app/plot_context.py)

| 变更 | 详情 |
|------|------|
| 新增协议属性 | `_global_max_density`, `value_cache`, `_enum_text_maps` |
| 新增协议方法 | `_sync_min_xrange()`, `_get_plot_container()`, `_show_drag_indicator_for_plot()`, `_hide_drag_indicator_for_plot()`, `auto_y_in_x_range()`, `collect_global_x_range()`, `set_cursor_enabled()`, `is_cursor_enabled()` |
| 协议签名修正 | `request_mark_stats_refresh(*, immediate: bool = False)` — 对齐 `MainWindow` 实际签名 |
| PlotContext 代理 | 全部新增属性/方法均已实现代理 |

### Step 0b — 构造时注入 plot_context

**文件**：[csv_plot_pyqt6.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py)

| 变更 | 位置 |
|------|------|
| `DraggableGraphicsLayoutWidget.__init__` 增加 `plot_context=None` 参数 | L96 |
| `self.plot_context = plot_context` 在构造时赋值 | L98 |
| `MainWindow.create_subplots_matrix` 改为构造时传入 `PlotContext(self)` | L6277 |

### Step 0c — 替换 self.window() / pw.window() 调用

**文件**：[csv_plot_pyqt6.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py)

| 类别 | 替换前 | 替换后 | 数量 |
|------|--------|--------|------|
| loader 访问 | `self.window().loader` | `self.plot_context.loader` | ~8 |
| cursor_btn 访问 | `self.window().cursor_btn` | `self.plot_context.cursor_btn` | ~6 |
| plot_widgets 遍历 | `self.window().plot_widgets` | `self.plot_context.plot_widgets` | ~8 |
| 跨图同步 | `self.window().sync_crosshair()` | `self.plot_context.sync_crosshair()` | ~2 |
| mark stats | `self.window().request_mark_stats_refresh()` | `self.plot_context.request_mark_stats_refresh()` | ~6 |
| 全局密度/范围 | `self.window()._global_max_density` | `self.plot_context._global_max_density` | ~3 |
| ViewBox 回调 | `pw.window().xxx` | `pw.plot_context.xxx` | ~12 |
| pinned_x_values | `self.window().pinned_x_values = ...` | `self.plot_context.pinned_x_values = ...` | ~2 |
| 拖拽通知 | `main_window._show_drag_indicator_for_plot()` | `self.plot_context._show_drag_indicator_for_plot()` | ~4 |
| 时间修正 | `self.window().loader.datalength` | `self.plot_context.loader.datalength` | ~2 |
| 全局范围 | `main_window.collect_global_x_range()` | `self.plot_context.collect_global_x_range()` | ~1 |
| sync_mark_regions | `self.window().sync_mark_regions()` | `self.plot_context.sync_mark_regions()` | ~2 |
| **合计替换** | | | **~56 处** |

**保留的 12 处 self.window() 调用**（均为合法使用）：

| 用途 | 行号 | 说明 |
|------|------|------|
| Dialog parent | L336, L3682, L4221 | Qt 对话框需要 QWidget 作为 parent，不能通过 plot_context |
| 拖拽坐标转换 | L886, L901 | `_should_hide_drag_indicator` 需要真实窗口坐标 |
| \_enum\_text\_maps 初始化 | L2352-2353 | 字典初始化需要直接操作 MainWindow 对象 |
| findChildren | L3670-3672 | QWidget 标准 API，遍历子控件 |
| 注释掉的旧代码 | L488, L3603 | 无影响 |

### Step 0d — 旧 Mixin 加弃用警告

**文件**：[src/ui/cursor_manager.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/ui/cursor_manager.py)

- 文件头部添加 `[已弃用]` 标记和详细的弃用说明
- 添加 `warnings.warn()` 在模块导入时触发 `DeprecationWarning`
- 保留原有代码供过渡期兼容

### Step 0e — 创建管理器基类

**文件**：[src/ui/widgets/base_manager.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/ui/widgets/base_manager.py)（新建，63 行）

```python
class BasePlotManager:
    def __init__(self, plot_widget):    # weakref.ref 防止循环引用
    @property
    def pw(self):                        # 安全检查关联对象是否存活
    def initialize(self): ...            # 生命周期：初始化后
    def cleanup(self): ...               # 生命周期：销毁前
    def reset(self): ...                 # 生命周期：数据重置后
```

### Step 0g — 迁移 PlotContainerWidget

| 操作 | 文件 |
|------|------|
| **新建** | [src/ui/widgets/plot_container.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/ui/widgets/plot_container.py)（82 行） |
| **移除** | csv_plot_pyqt6.py 中原 `class PlotContainerWidget` 定义（~68 行） |
| **新增导入** | csv_plot_pyqt6.py L12: `from src.ui.widgets.plot_container import PlotContainerWidget` |
| **导出注册** | [src/ui/widgets/__init__.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/ui/widgets/__init__.py)：导出 `PlotContainerWidget`, `CustomViewBox`, `BasePlotManager` |

### Step 0h — MainWindow 新增方法

**文件**：[csv_plot_pyqt6.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py)

| 新增方法 | 行号 | 说明 |
|----------|------|------|
| `MainWindow.set_cursor_enabled(enabled)` | L5877 | 光标启用/禁用 |
| `MainWindow.is_cursor_enabled()` | L5884 | 查询光标状态 |

---

## 三、文件变更统计

| 文件 | 操作 | 行数变化 |
|------|------|----------|
| [csv_plot_pyqt6.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py) | 修改 | 6715 → 6650（-65 行） |
| [src/app/plot_context.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/app/plot_context.py) | 修改 | +50 行（协议扩展 + 代理方法） |
| [src/ui/widgets/base_manager.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/ui/widgets/base_manager.py) | **新建** | 63 行 |
| [src/ui/widgets/plot_container.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/ui/widgets/plot_container.py) | **新建** | 82 行 |
| [src/ui/widgets/\_\_init\_\_.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/ui/widgets/__init__.py) | 修改 | +17 行（导出列表） |
| [src/ui/cursor_manager.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/ui/cursor_manager.py) | 修改 | +10 行（弃用警告） |
| [DraggableGraphicsLayoutWidget_MainWindow_Refactor_Plan.md](file:///Users/xiaolin/Documents/python_repo/csv_plot/DraggableGraphicsLayoutWidget_MainWindow_Refactor_Plan.md) | 修改 | v2.0 → v3.0 |

### 备份

| 位置 | 内容 |
|------|------|
| `.phase_v3_backup/csv_plot_pyqt6.py.bak` | 6715 行原始文件 |
| `.phase_v3_backup/src.bak/` | 原始 `src/` 目录完整快照 |

---

## 四、问题修复记录

### 🔴 问题 1：`request_mark_stats_refresh` 参数传递错误

**症状**：运行时持续报 `TypeError: MainWindow.request_mark_stats_refresh() takes 1 positional argument but 2 were given`

**根因**：`MainWindow.request_mark_stats_refresh(self, *, immediate: bool = False)` 中 `*` 将 `immediate` 设为仅限关键字参数，但 `PlotContext` 将它作为位置参数传递。

**修复**（[plot_context.py:L47](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/app/plot_context.py#L47)、[L129-L130](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/app/plot_context.py#L129-L130)）：

```python
# Protocol
def request_mark_stats_refresh(self, *, immediate: bool = False) -> None: ...
# PlotContext
def request_mark_stats_refresh(self, *, immediate: bool = False) -> None:
    self._services.request_mark_stats_refresh(immediate=immediate)
```

---

## 五、架构验证

### 当前依赖关系

```
MainWindow (QMainWindow)
│  实现 PlotServices 协议
│  持有: loader, plot_widgets, cursor_btn, cursor_mode...
│
├── PlotContext ── 代理 MainWindow 的 PlotServices 接口
│   ✅ 协议已覆盖所有 self.window() 访问模式
│   ✅ 构造时注入到 DraggableGraphicsLayoutWidget
│
├── DraggableGraphicsLayoutWidget
│   │  ✅ self.plot_context 替换了 ~56 处 self.window() 调用
│   │  ⚠️ 保留了 12 处合理 self.window()（Widget API / Dialog parent）
│   │
│   ├── CustomViewBox  ✅ 通过 self.plot_widget.plot_context 访问
│   └── PlotContainerWidget  ✅ 已迁移到 src/ui/widgets/plot_container.py
│
└── 管理器基础设施
    ├── BasePlotManager  ✅ weakref + lifecycle hooks 就绪
    └── src/ui/widgets/__init__.py  ✅ 公共接口导出就绪
```

### 验证结果

- ✅ 所有 Python 文件通过 `py_compile` 语法检查
- ✅ 所有模块导入路径有效
- ✅ `PlotServices` 协议与 `MainWindow` 实际接口签名一致
- ✅ 运行时无 `self.window()` 相关 TypeError

---

## 六、遗留事项

| 事项 | 优先级 | 计划处理阶段 |
|------|--------|-------------|
| `_enum_text_maps` 初始化需通过 `self.window()` 写回 MainWindow | 低 | 阶段四-五（MainWindow 拆解后自然解决） |
| `_should_hide_drag_indicator` 需要真实窗口坐标 | 低 | 阶段一（AxisManager 分离后评估） |
| `findChildren` 需要 QWidget API | 低 | 无法通过 plot_context 替代，保留 |
| Dialog parent 需要 QWidget 对象 | 低 | 无法通过 plot_context 替代，保留 |
| 性能基线脚本 | 中 | 阶段一开始前补充 |

---

## 七、下一步（阶段一）

按照重构计划 v3.0 §六，阶段一将迁移 **DraggableGraphicsLayoutWidget 底层管理器**：

1. **Step 1**：创建 `PlotUIManager`，迁移 UI 初始化 + 刷新协调方法
2. **Step 2**：创建 `AxisManager`，迁移坐标轴管理方法

**预计影响**：csv_plot_pyqt6.py 将减少约 500 行代码。

---

**文档版本**：1.0  
**创建日期**：2026-05-18
