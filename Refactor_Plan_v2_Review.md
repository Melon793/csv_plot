# 重构计划 v2.0 深度评审意见

> 评审对象: [DraggableGraphicsLayoutWidget_MainWindow_Refactor_Plan.md](file:///Users/xiaolin/Documents/python_repo/csv_plot/DraggableGraphicsLayoutWidget_MainWindow_Refactor_Plan.md) (v2.0)  
> 评审日期: 2026-05-17  
> 评审方法: 结合源码逐行审查，涵盖架构、实现细节、边缘情况  

---

## 一、总体评价

v2.0 计划较 v1.0 有了**质的提升**：管理器从 17 个精简为 11 个，新增了前置步骤消除 `self.window()` 依赖，明确了状态属性归属表和依赖图。方向正确，大部分关键问题已得到响应。

经过对 [csv_plot_pyqt6.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py)、[CustomViewBox](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/ui/widgets/custom_viewbox.py) 和 [PlotContext](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/app/plot_context.py) 的深度源码审查，发现以下 **6 个需要修正的问题**和 **4 个优化建议**。

---

## 二、需要修正的问题

### 2.1 🔴 信号连接拆分存在架构矛盾

**问题描述**:

v2.0 计划将 `_connect_viewbox_signals` 划归 `EventHandler`（[L235](file:///Users/xiaolin/Documents/python_repo/csv_plot/DraggableGraphicsLayoutWidget_MainWindow_Refactor_Plan.md#L235)），但将光标相关的信号处理器划归 `CursorManager`（`_on_vb_set_cursor_mode`、`_on_vb_show_cursor`、`_on_vb_hide_cursor`）。

源码中 `_connect_viewbox_signals` 一次性连接了所有 10 个信号（[L4160-L4172](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L4160-L4172)）：

```python
def _connect_viewbox_signals(self):
    vb.signals.request_jump_to_data.connect(self._on_vb_jump)        # → EventHandler
    vb.signals.request_clear_plot.connect(self._on_vb_clear)          # → EventHandler
    vb.signals.request_auto_y.connect(self._on_vb_auto_y)             # → EventHandler
    vb.signals.request_set_cursor_mode.connect(self._on_vb_set_cursor_mode)  # → CursorManager
    vb.signals.request_show_cursor_value.connect(self._on_vb_show_cursor)    # → CursorManager
    vb.signals.request_hide_cursor_value.connect(self._on_vb_hide_cursor)    # → CursorManager
    vb.signals.request_set_row_height.connect(...)                    # → EventHandler
    vb.signals.request_set_all_row_height.connect(...)                # → EventHandler
    vb.signals.request_copy_name.connect(...)                        # → EventHandler
    vb.signals.request_variable_editor.connect(...)                   # → EventHandler
```

如果 `_connect_viewbox_signals` 在 EventHandler 中，它需要引用 `CursorManager` 的方法来连接光标信号——这直接违反了"管理器间禁止直接引用"的规则。

**解决方案**:

将 `_connect_viewbox_signals` **保留在主类中**（而非 EventHandler），主类负责所有信号连接，内部路由到对应管理器：

```python
class DraggableGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    def _connect_viewbox_signals(self):
        vb = self.view_box
        vb.plot_widget = self
        # 路由到 EventHandler
        vb.signals.request_jump_to_data.connect(self.event_handler._on_vb_jump)
        vb.signals.request_clear_plot.connect(self.event_handler._on_vb_clear)
        vb.signals.request_auto_y.connect(self.event_handler._on_vb_auto_y)
        vb.signals.request_set_row_height.connect(self.event_handler._on_vb_set_row_height)
        vb.signals.request_set_all_row_height.connect(self.event_handler._on_vb_set_all_row_height)
        vb.signals.request_copy_name.connect(self.event_handler._on_vb_copy_name)
        vb.signals.request_variable_editor.connect(self.event_handler._on_vb_var_editor)
        # 路由到 CursorManager
        vb.signals.request_set_cursor_mode.connect(self.cursor_manager._on_vb_set_cursor_mode)
        vb.signals.request_show_cursor_value.connect(self.cursor_manager._on_vb_show_cursor)
        vb.signals.request_hide_cursor_value.connect(self.cursor_manager._on_vb_hide_cursor)
```

**计划更新**:
- EventHandler 从职责中移除 `_connect_viewbox_signals`
- 新增说明：信号连接保留在主类作为路由层

### 2.2 🟡 `_on_range_changed` 性能路径归属需明确

**问题描述**:

[L284](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L284) 中 `sigRangeChanged` 直接连接到 `self._on_range_changed`，这是整个应用中**调用频率最高的路径**（每次缩放/平移都触发）。计划将其划归 EventHandler（[L235](file:///Users/xiaolin/Documents/python_repo/csv_plot/DraggableGraphicsLayoutWidget_MainWindow_Refactor_Plan.md#L235)），但该连接是在 `setup_plot_area`（PlotUIManager）中建立的。

另外，`setup_plot_area` 中实际代码（L284）使用的是 `self.view_box.sigRangeChanged.connect(self._on_range_changed)`，这里的 `self` 是 `DraggableGraphicsLayoutWidget` 实例。如果 `_on_range_changed` 迁移到 EventHandler，连接也需要更新。

**解决方案**:

`_on_range_changed` **保留在主类中**作为薄委托（一条 line），不经 EventHandler 中转，避免高频路径的额外开销：

```python
class DraggableGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    def _on_range_changed(self):
        # 高频路径：直接委托，最快
        self.cursor_manager.update_cursor_label()
        # 如果还有其他 range_changed 逻辑，在这里路由
```

**如果** `_on_range_changed` 内部逻辑很重（需要查源码确认），则主类中保留连接，但业务逻辑委托给 EventHandler：

```python
    def _on_range_changed(self):
        self.event_handler._on_range_changed()
```

### 2.3 🟡 `_on_vb_*` 方法对 `plot_context` 的依赖不完整

**问题描述**:

v2.0 计划在 Step 0 中将 `self.window()` / `pw.window()` 替换为 `self.plot_context.xxx`。但当前 `_on_vb_*` 方法（[L4174-L4237](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L4174-L4237)）中有多个调用需要在 PlotServices 协议中补充：

| 源码调用 | 需要的 PlotServices 方法 | 当前协议中是否存在 |
|---------|------------------------|-----------------|
| `pw.window().auto_y_in_x_range()` | `auto_y_in_x_range()` | ✅ 已在计划中列出 |
| `pw.window().set_cursor_mode(...)` | `set_cursor_mode(...)` | ✅ 已存在 |
| `pw.window().cursor_values_hidden` | `cursor_values_hidden` | ✅ 已存在 |
| `pw.window().cursor_btn.isChecked()` | 已有 `is_cursor_enabled()` | ✅ 已存在 |
| `pw.window().plot_widgets` | `plot_widgets` | ✅ 已存在 |
| `pw.window()._plot_col_max_default` | `_plot_col_max_default` | ✅ 已存在 |
| `pw.window().set_row_height(row, pct)` | `set_row_height(row, pct)` | ✅ 已存在 |
| `w.set_all_row_height(pct)` | `set_all_row_height(pct)` | ✅ 已存在 |
| `pw.window().request_mark_stats_refresh(immediate=True)` | `request_mark_stats_refresh(immediate)` | ✅ 已存在 |
| `pw.window().loader` (in `_on_vb_var_editor`) | `loader` | ✅ 已存在 |

实际上，现有 `PlotServices` 协议（[L14-L47](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/app/plot_context.py#L14-L47)）+ v2.0 计划扩展（[L494-L512](file:///Users/xiaolin/Documents/python_repo/csv_plot/DraggableGraphicsLayoutWidget_MainWindow_Refactor_Plan.md#L494-L512)）已经覆盖了所有调用。但**缺少一个关键的验证：`_on_vb_var_editor` 中的 `_lazy_PlotVariableEditorDialog()` 调用**。

[L4233-L4234](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L4233-L4234):
```python
dialog = _lazy_PlotVariableEditorDialog()(pw, pw.window() if pw.window() and hasattr(pw.window(), "loader") else None)
```

迁移后变为：
```python
dialog = _lazy_PlotVariableEditorDialog()(pw, pw.plot_context if pw.plot_context else None)
```

`PlotVariableEditorDialog` 接收的是 `MainWindow` 实例，迁移后接收的是 `PlotContext` 实例。需要验证 `PlotVariableEditorDialog` 内部是否兼容 `PlotContext`，或者需要在 `PlotContext` 中添加 `loader` 属性的透传。

**解决方案**:
1. 在 Step 0 迁移清单中显式列出所有 `_on_vb_*` 方法的替换项
2. 验证 `PlotVariableEditorDialog` 接收 `PlotContext` 而非 `MainWindow` 时的兼容性
3. 如有不兼容，要么修改 `PlotVariableEditorDialog` 使用 `PlotServices` 协议，要么在 `PlotContext` 中添加完整代理

### 2.4 🟡 状态属性归属表缺少部分属性

**问题描述**:

v2.0 计划的状态属性归属表（[L287-L312](file:///Users/xiaolin/Documents/python_repo/csv_plot/DraggableGraphicsLayoutWidget_MainWindow_Refactor_Plan.md#L287-L312)）漏掉了以下在源码中广泛使用的属性：

| 遗漏属性 | 使用位置 | 建议归属 |
|---------|---------|---------|
| `y_format` / `x_format` | 多处 cursor label 格式化 | PlotDataManager ✅ 已列入 |
| `show_values_only` | cursor label 显示模式 | CursorManager |
| `_is_interacting` | 交互状态标志 | 主类保留（被 view_box 信号连接使用） |
| `_is_syncing_range` | 同步缩放标志 | AxisManager |
| `_cursor_label_busy` / `_cursor_label_dirty` | cursor 更新防抖 | CursorManager |
| `_pending_delete_items` / `_cleanup_timer` | 安全删除队列 | CursorManager |
| `_drag_indicator_source` / `_drag_indicator_guard` | 拖拽指示器 | MultiCurveManager |
| `_cached_data_version` | 数据版本缓存 | PlotDataManager |
| `_suppress_pin_update` | pin 更新抑制 | CursorManager |

**建议**: 补充这些遗漏属性到归属表，特别是 `show_values_only`、`_cursor_label_busy`、`_cursor_label_dirty` 属于 CursorManager 高耦合属性。

### 2.5 🟡 `create_subplots_matrix` 中的 plot_context 注入需同步更新

**问题描述**:

v2.0 计划 Step 0b 要求在 `DraggableGraphicsLayoutWidget.__init__` 中注入 `plot_context`（[L520-L525](file:///Users/xiaolin/Documents/python_repo/csv_plot/DraggableGraphicsLayoutWidget_MainWindow_Refactor_Plan.md#L520-L525)）。当前实际代码在 [L6276-L6277](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L6276-L6277) 是后置注入：

```python
plot_widget = DraggableGraphicsLayoutWidget(self.units, self.data, self.time_channels_infos)
plot_widget.plot_context = PlotContext(self)
```

改为构造注入后，应为：

```python
plot_widget = DraggableGraphicsLayoutWidget(
    self.units, self.data, self.time_channels_infos,
    plot_context=PlotContext(self)
)
```

同时，`setup_ui()` 内部开始使用 `self.plot_context` 的方法（如 `toggle_cursor`），需要确保 `plot_context` 在 `setup_ui` 调用前已设置。当前源码在 `__init__` 末尾调用 `self.setup_ui(...)`（[L122](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L122)），构造注入能保证这一点。

**但有一个时序问题**: `create_subplots_matrix` 在构造后又调用了：

```python
plot_widget.toggle_cursor(cursor_enabled)        # line 6281-6283
plot_widget.apply_cursor_mode(self.cursor_mode, ...)  # line 6285
```

这些调用在 setup_ui 完成后执行，此时 plot_context 已就绪，没有问题。

**建议**: 在计划中明确说明 `create_subplots_matrix` 的调用变更。

### 2.6 🟡 `jump_to_data_impl` 尚未纳入迁移清单

**问题描述**:

[Jump_to_data_impl](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L298) 方法（约 50 行）严重依赖 `self.window()`：

- [L317](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L317): `main_window = self.window()`
- [L318](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L318): `if not hasattr(main_window, 'loader') or main_window.loader is None`
- [L322](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L322): `is_mdf_loader = hasattr(main_window.loader, 'get_series')`
- [L334](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L334): `DataTableDialog.popup(var_name, series, parent=main_window)`

该方法未出现在 v2.0 计划的任何管理器方法列表中。它是通过 `_on_vb_jump` 触发的（[L4174-L4176](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L4174-L4176)）。

**建议**: 
1. 将 `jump_to_data_impl` 归入 `EventHandler`（与 `_on_vb_jump` 相同归属）
2. Step 0 中将其 `self.window()` 调用迁移到 `self.plot_context.xxx`
3. 注意 `DataTableDialog.popup` 的 `parent` 参数需要的是 QWidget，`PlotContext` 不是 QWidget。需要通过 `plot_context` 获取 MainWindow 引用或提供替代方案

---

## 三、优化建议

### 3.1 CustomViewBox 已使用 plot_context — 应统一检查

**现状**: [CustomViewBox](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/ui/widgets/custom_viewbox.py) 的 `_get_cursor_enabled`、`_get_current_cursor_mode`、`_get_cursor_values_hidden`、`_get_current_row_height`、`_has_data`、`_get_plot_row_index` 方法已经使用 `self.plot_widget.plot_context` 模式（[L248-L287](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/ui/widgets/custom_viewbox.py#L248-L287)）。这些方法在 v2.0 后仍能正常工作，因为 `plot_context` 属性仍存在且协议兼容。

**但需注意**: `_has_data` 直接访问 `self.plot_widget.curve` 和 `self.plot_widget.curves`（[L271-L276](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/ui/widgets/custom_viewbox.py#L271-L276)）。这些属性通过 `@property` 委托后仍可访问，无兼容问题。

**建议**: 在 Step 0 完成后，做一次全局搜索 `self.plot_widget.` 确保所有访问路径仍有效。

### 3.2 `_lazy_*` 惰性导入模式需保留

**现状**: [L6624-L6649](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L6624-L6649) 有 6 个 `_lazy_*` 函数避免循环导入。重构后管理器间可能产生新的循环导入（尤其是 `EventHandler` → `PlotVariableEditorDialog`、`LayoutManager` → `LayoutInputDialog` 等）。

**建议**: 
1. 在管理器文件中也采用惰性导入模式
2. 或统一将所有对话框导入放在主类中，通过依赖注入传递给管理器
3. 推荐方案：对话框导入保留在主类中，管理器接收工厂函数或直接接收已导入的类

### 3.3 建议增加 `PlotContainerWidget` → 管理器的迁移

**现状**: [PlotContainerWidget](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L4263) 位于 csv_plot_pyqt6.py 中（约 70 行），功能简单（绘制指示器、管理拖拽提示文本）。计划在 Step 13 迁移至 `src/ui/widgets/plot_container.py`。

**但注意**: `PlotContainerWidget._build_indicator_text`（[L4302-L4304](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py#L4302-L4304)）访问 `self.plot_widget.curve` 和 `self.plot_widget.curves`。迁移后这些访问通过 `@property` 委托，无问题。

### 3.4 建议增加 `eventFilter` 迁移提示

**现状**: MainWindow 的 `eventFilter` 方法处理全局拖拽事件和窗口调整。计划中 `MainWindowUIManager.handle_resize` 覆盖了 `resizeEvent`，但 `eventFilter` 未显式列出。

**建议**: 在 `MainWindowUIManager` 中增加 `eventFilter` 相关的业务逻辑方法，主类 `eventFilter` 保留为薄委托。

---

## 四、实施可行性评估（修正后）

| 维度 | v2.0 计划评分 | 修正后评分 | 说明 |
|------|------------|----------|------|
| MRO 防护 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 无需修正 |
| 依赖管理 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 修正信号连接方案后更清晰 |
| 管理器粒度 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 无需修正 |
| 事件处理 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | `_on_range_changed` 高频路径保留主类 |
| 状态归属 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 补充遗漏属性 |
| 实施步骤 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 明确 `create_subplots_matrix` 和 `jump_to_data_impl` 变更 |
| 兼容性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 补充 CustomViewBox 兼容性验证 |

**总体可行性**: ✅ 可行，需应用上述 6 个修正。

---

## 五、v2.0 修正清单

| 编号 | 严重度 | 修正内容 | 涉及章节 |
|------|--------|---------|---------|
| F1 | 🔴 | `_connect_viewbox_signals` 保留在主类而非 EventHandler | 2.2.3 → (7) EventHandler |
| F2 | 🟡 | `_on_range_changed` 保留在主类作为薄委托 | 2.2.3 → (7) EventHandler |
| F3 | 🟡 | 补充 `PlotServices` 协议中 `_on_vb_var_editor` 的兼容性验证 | 五、前置步骤 → 5.1 |
| F4 | 🟡 | 补充遗漏属性到状态归属表（show_values_only, _is_interacting, _cursor_label_busy 等） | 2.4 状态属性归属表 |
| F5 | 🟡 | 明确 `create_subplots_matrix` 中 plot_context 构造注入的调用变更 | 五、前置步骤 → 5.2 |
| F6 | 🟡 | 将 `jump_to_data_impl` 纳入 EventHandler 并列入 Step 0 迁移清单 | 2.2.3 → (7) EventHandler; 五 → 5.3 |

---

## 六、修正后的管理器职责终稿

### DraggableGraphicsLayoutWidget（7 个管理器）

| 管理器 | 职责 | 修正 |
|--------|------|------|
| **PlotUIManager** | UI 初始化 + 刷新协调 | 无修正 |
| **AxisManager** | 坐标轴管理、自动缩放 | 无修正 |
| **PlotDataManager** | 单曲线绘图 + 时间修正 + 数据清除 | 新增遗漏属性 |
| **MultiCurveManager** | 多曲线 + 拖拽 + 样式 | 新增遗漏属性（拖拽指示器） |
| **CursorManager** | 光标 + 光标信号 + 对象池 | 新增遗漏属性（show_values_only, _cursor_label_busy/dirty, _pending_delete_items） |
| **MarkRegionManager** | 标记区域 | 无修正 |
| **EventHandler** | 事件路由 + 非光标信号 + jump_to_data | **移除 `_connect_viewbox_signals`**；**移除 `_on_range_changed`**（保留主类）；**新增 `jump_to_data_impl`** |
| **主类（壳）** | `_connect_viewbox_signals`、`_on_range_changed`、Qt 事件薄委托、`@property` 委托 | **新增保留信号连接**；**新增 `_on_range_changed` 薄委托** |

### MainWindow（4 个管理器）

| 管理器 | 职责 | 修正 |
|--------|------|------|
| **MainWindowUIManager** | 主窗口 UI + eventFilter 业务 | 新增 eventFilter 业务逻辑 |
| **FileLoaderManager** | 文件加载 | 无修正 |
| **LayoutManager** | 布局 + 标记区域同步 | 无修正 |
| **CursorSyncManager** | 光标同步 + 绘图同步 | 无修正 |

---

## 七、总结

v2.0 计划已非常成熟，核心架构决策正确。上述 6 个修正均为**实现层面**的细化调整，不涉及架构方向变更。应用修正后，计划可进入实施阶段。

**实施前必须完成的检查**:
1. [ ] `_connect_viewbox_signals` 保留在主类
2. [ ] `_on_range_changed` 保留在主类作为薄委托
3. [ ] 验证 `PlotVariableEditorDialog` 接收 `PlotContext` 的兼容性
4. [ ] 补充遗漏属性到归属表
5. [ ] 更新 `create_subplots_matrix` 调用点
6. [ ] `jump_to_data_impl` 纳入 EventHandler + Step 0 迁移清单
7. [ ] 全局搜索 `self.window()` 确认零残留
8. [ ] 验证 CustomViewBox 所有 `self.plot_widget.xxx` 访问路径在 `@property` 委托后有效

---

**文档版本**: 1.0  
**评审日期**: 2026-05-17
