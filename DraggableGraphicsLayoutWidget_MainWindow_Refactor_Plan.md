# DraggableGraphicsLayoutWidget 与 MainWindow 重构计划 v3.0

> **变更摘要 (v2.0 → v3.0)**  
> - **架构修正**: `_connect_viewbox_signals` 回退至主类作为路由层，`_on_range_changed` 高频路径保留在主类薄委托  
> - **状态属性补充**: 补全归属表中遗漏的 `_is_interacting`、`_cursor_label_busy`、`_pending_delete_items` 等 7 个关键属性  
> - **前置步骤增强**: 新增性能基线建立、`pw.window()` 变体迁移、`PlotContainerWidget` 提前迁移  
> - **生命周期管理**: 为管理器基类增加 `initialize()`、`cleanup()`、`reset()` 统一钩子  
> - **光标功能整合**: 将 [cursor_mode_modification_plan.md](file:///Users/xiaolin/Documents/python_repo/csv_plot/cursor_mode_modification_plan.md) 的 "off" 模式功能整合入 CursorManager  
> - **风险评估更新**: 补充延迟导入路径失效、`PlotVariableEditorDialog` 兼容性两项风险  
> - **测试计划增强**: 新增管理器单元测试框架说明  

---

## 一、现有代码结构分析

### 1.1 项目概况
- **主文件**: [csv_plot_pyqt6.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py) - 共 6691 行
- **主要问题类**: 
  - `DraggableGraphicsLayoutWidget` (约 4174 行)
  - `MainWindow` (约 2356 行)
- **已完成部分重构**: 其他组件已迁移至 `src/` 目录下

### 1.2 DraggableGraphicsLayoutWidget 类功能分析

该类继承自 `pg.GraphicsLayoutWidget`，负责单个绘图区域的所有功能，按功能模块划分如下：

| 功能模块 | 关键方法 | 行数 | 说明 |
|---------|---------|------|------|
| UI 初始化 | `__init__`, `setup_ui`, `setup_header`, `setup_plot_area`, `setup_axes`, `setup_interaction` | ~200 行 | 初始化绘图组件 |
| 坐标轴管理 | `update_x_axis_label`, `set_xrange_with_link_handling`, `_get_safe_x_range`, `_get_min_x_range_value`, `_set_x_limits_with_min_range`, `_set_min_x_range`, `_set_safe_y_range`, `_setup_plot_axes`, `_reset_plot_limits` | ~180 行 | X/Y 轴范围和标签管理 |
| 自动缩放 | `auto_range`, `auto_y_in_x_range`, `_calculate_visible_points` | ~150 行 | 自动适配数据范围 |
| 光标管理 | `_get_cursor_mode`, `_get_cursor_x_positions`, `_set_vline_visibility_for_mode`, `_set_vline_bounds`, `apply_cursor_mode`, `update_cursor_label`, `_update_single_curve_cursor_label`, `_update_multi_curve_cursor_label`, `_position_labels_avoid_overlap`, `toggle_cursor`, `_show_x_position_only`, `_has_visible_curve_data`, `pin_cursor`, `free_cursor`, `reset_pin_state`, `_update_vline_bounds_from_data`, `_update_cursor_after_plot`, `_get_circle_from_pool`, `_get_label_from_pool`, `_get_x_label_from_pool`, `_clear_cursor_items`, `_queue_item_for_deletion`, `_process_pending_deletes` | ~800 行 | 光标位置、标签、模式管理 |
| 单曲线绘图 | `plot_variable`, `_validate_plot_data`, `_get_x_data_for_variable`, `_prepare_plot_data`, `_compute_valid_min_max`, `_get_y_range_in_x_window`, `handle_single_point_limits` | ~300 行 | 绘制单条曲线 |
| 多曲线管理 | `add_variables_to_plot`, `add_variable_to_plot`, `update_multi_curve_mode`, `update_legend`, `toggle_curve_visibility_by_name`, `_recreate_curve`, `_on_legend_clicked`, `_update_axes_for_multi_curve`, `_update_x_limits_for_plot`, `_collect_visible_curve_arrays`, `_collect_visible_curve_pairs`, `get_curve_x_limits` | ~900 行 | 多条曲线绘制、图例、可见性管理 |
| 数据清除 | `_clear_plot_data`, `clear_plot_item`, `reset_plot`, `clear_value_cache` | ~100 行 | 清除绘图和数据缓存 |
| 时间修正 | `datetime_to_unix_seconds`, `update_time_correction`, `get_value_from_name` | ~150 行 | 时间轴缩放和偏移 |
| 拖拽放置 | `dragEnterEvent`, `dragMoveEvent`, `dragLeaveEvent`, `dropEvent`, `_extract_var_names_from_text`, `_should_hide_drag_indicator`, `_enforce_drag_indicator_visibility`, `_notify_drag_indicator` | ~150 行 | 变量拖拽到绘图区域 |
| 标记区域 | `add_mark_region`, `remove_mark_region`, `update_mark_region`, `get_mark_stats` | ~150 行 | 区域选择和统计 |
| 样式管理 | `_apply_plot_style`, `update_plot_style` | ~150 行 | 曲线粗细、符号显示 |
| 事件处理 | `wheelEvent`, `mouse_moved`, `_is_cursor_update_locked`, `resizeEvent`, `mouseDoubleClickEvent`, `mousePressEvent`, `mouseMoveEvent`, `mouseReleaseEvent` | ~150 行 | 鼠标和键盘事件 |
| UI 刷新协调 | `_init_ui_refresh_coordinator`, `_queue_ui_refresh`, `_cancel_ui_refresh`, `_run_style_refresh`, `_run_cursor_refresh`, `_run_stats_refresh` | ~100 行 | 防抖和批量更新 |
| ViewBox 信号 | `_start_interaction`, `_end_interaction`, `_schedule_cursor_geometry_update`, `_refresh_cursor_geometry`, `_connect_viewbox_signals`, `_on_vb_jump`, `_on_vb_clear`, `_on_vb_auto_y`, `_on_vb_set_cursor_mode`, `_on_vb_show_cursor`, `_on_vb_hide_cursor`, `_on_vb_set_row_height`, `_on_vb_set_all_row_height`, `_on_vb_copy_name`, `_on_vb_var_editor` | ~200 行 | 处理 ViewBox 事件 |
| 安全清理 | `_safe_clear_plot_items` | ~50 行 | 安全清除 PyQtGraph 项 |
| **总计** |  | **~4174 行** |  |

### 1.3 MainWindow 类功能分析

该类继承自 `QMainWindow`，是应用程序的主界面，按功能模块划分如下：

| 功能模块 | 关键方法 | 行数 | 说明 |
|---------|---------|------|------|
| 初始化与 UI | `__init__`, `_on_splitter_moved`, `_ensure_splitter_ready`, `_apply_fixed_splitter_width`, `resizeEvent`, `toggle_plot_area`, `show_help`, `_get_plot_container`, `_show_drag_indicator_for_plot`, `_hide_drag_indicator_for_plot`, `spawn_clone_window` | ~700 行 | 主窗口和布局初始化 |
| 文件加载 | `load_btn_click`, `_validate_file_path`, `_check_file_size`, `_begin_data_reload`, `_end_data_reload`, `_post_reload_ui_refresh`, `load_csv_file`, `set_button_status`, `reload_data`, `_load_file`, `_has_valid_loader`, `_has_valid_data`, `_current_data_length`, `_cleanup_old_data`, `_post_load_actions`, `_remember_last_open_dir`, `_get_dialog_initial_directory`, `_default_system_directory`, `load_dict`, `_extract_file_extension`, `_validate_load_parameters`, `_load_sync`, `_on_load_done`, `_on_load_error`, `_apply_loader` | ~800 行 | 文件对话框、数据验证、同步/异步加载 |
| 变量过滤 | `filter_variables` | ~50 行 | 搜索和过滤变量 |
| 标记区域 | `toggle_mark_region`, `sync_mark_regions`, `request_mark_stats_refresh`, `_flush_mark_stats_refresh`, `update_mark_stats` | ~150 行 | 跨绘图区域同步标记区域 |
| 布局管理 | `open_layout_dialog`, `open_time_correction_dialog`, `update_mark_regions_on_layout_change`, `create_subplots_matrix`, `set_row_height`, `set_all_row_height`, `get_row_height`, `set_plots_visible` | ~400 行 | 子图网格布局、行高调整 |
| 光标管理 | `reset_plots_after_loading`, `_get_cursor_source_plot`, `_get_cursor_view_range`, `_clamp_value`, `_calc_second_cursor_position`, `_select_farthest_cursor_index`, `_apply_cursor_mode_to_plots`, `set_cursor_mode`, `toggle_cursor_all`, `_realign_pinned_cursor_after_time_correction`, `sync_crosshair`, `_flush_crosshair_updates`, `reset_all_pin_states` | ~400 行 | 跨绘图区域同步光标 |
| 清除与缩放 | `clear_all_plots`, `collect_global_x_range`, `_compute_baseline_density`, `_sync_min_xrange`, `auto_range_all_plots`, `auto_y_in_x_range` | ~150 行 | 全局范围计算和同步 |
| 重载后重绘 | `replots_after_loading` | ~200 行 | 数据重载后恢复绘图状态 |
| **总计** |  | **~2356 行** |  |

### 1.4 现有 `src/` 目录结构

```
src/
├── __init__.py
├── app/
│   ├── __init__.py
│   └── plot_context.py     # PlotServices 协议、PlotContext 类
├── core/
│   ├── __init__.py
│   ├── config.py           # 常量配置、debug_log、safe_callback
│   ├── scheduler.py        # UnifiedUpdateScheduler
│   └── types.py            # AutoDetectError, FormatInfo, CurveInfo
├── ui/
│   ├── __init__.py
│   ├── cursor_manager.py   # ⚠️ 旧 Mixin，待弃用
│   ├── drag_drop.py        # VAR_SEPARATOR, 解析/构建变量名, 创建拖拽 Pixmap
│   ├── mark_stats.py       # MarkStatsWindow
│   ├── plot_variable_editor.py  # PlotVariableEditorDialog
│   ├── table_dialog.py     # DataTableDialog, DropOverlay, PandasTableModel, CustomDelegate
│   ├── variable_list.py    # MyTableWidget, NoHoverDelegate
│   ├── dialogs/
│   │   ├── __init__.py
│   │   ├── axis.py         # AxisDialog
│   │   ├── help.py         # HelpDialog
│   │   ├── layout_input.py # LayoutInputDialog
│   │   └── time_correction.py  # TimeCorrectionDialog
│   └── widgets/
│       ├── __init__.py
│       └── custom_viewbox.py   # CustomViewBox
└── data/
    ├── __init__.py
    └── loader.py           # FastDataLoader, DataLoadThread
```

### 1.5 关键技术债务：`self.window()` 跨对象引用

代码中存在 **53 处** `self.window()` 调用、**19 处** ViewBox 信号回调中的 `pw.window()` 调用，以及 **53 处** `main_window.xxx` 直接属性访问，散布在几乎所有功能模块中。项目已有 [PlotContext](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/app/plot_context.py) 解耦层，但当前几乎未使用（仅在 `create_subplots_matrix` 中注入，`CustomViewBox` 中少量使用）。

**这是比 MRO 更紧迫的架构债务，必须在管理器拆分之前解决。**

典型模式：
```python
# 模式 1: 访问 loader
main_window = self.window()
if main_window and hasattr(main_window, 'loader') and main_window.loader is not None:
    loader = main_window.loader

# 模式 2: 访问全局状态
if self.window() and hasattr(self.window(), "cursor_btn"):
    self.window().cursor_btn.setChecked(True)

# 模式 3: 跨 plot 同步
for container in self.window().plot_widgets:
    container.plot_widget.xxx()

# 模式 4: ViewBox 回调中 (v3.0 新增确认)
def _on_vb_clear(self, pw):
    if pw and pw.window():
        pw.window().request_mark_stats_refresh(immediate=True)
```

---

## 二、具体拆解方案

### 2.1 核心原则
1. **不使用多重继承和 Mixin** — 完全避免 MRO 问题
2. **使用组合 (Composition) 优于继承** — 将功能封装为独立的管理器类
3. **保持主类精简** — 主类只负责协调，具体逻辑委托给管理器
4. **向后兼容** — 保持现有公有 API 不变
5. **渐进式重构** — 每步均可测试和回滚
6. **先解耦再拆分** — 先消除 `self.window()` 硬依赖，再拆分管理器
7. **合并高耦合模块** — 避免管理器碎片化和网状依赖
8. **v3.0 新增**: **高频路径不走委托** — `_on_range_changed` 这类每秒调用数十次的方法保留在主类薄委托层

### 2.2 DraggableGraphicsLayoutWidget 拆解方案

#### 2.2.1 管理器依赖图

```
                    ┌─────────────┐
                    │  主类 (壳)   │
                    │  路由/协调   │ ← 信号连接、高频路径在此
                    └──────┬──────┘
                           │ 委托
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
   │ PlotUIManager│  │ AxisManager │  │MarkRegionMgr │
   │ (含刷新协调) │  │  (被多方依赖)│  │  (独立)      │
   └─────────────┘  └──────┬──────┘  └──────────────┘
                           │ 被依赖
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌──────────────┐ ┌───────────────┐ ┌──────────────┐
   │PlotDataManager│ │MultiCurveMgr  │ │ CursorManager│
   │(含时间修正)   │ │(含拖拽+样式)  │ │(含光标信号)  │
   └──────┬───────┘ └───────┬───────┘ └──────┬───────┘
          │                 │                 │
          └────────┬────────┘                 │
                   ▼                          │
            (互相调用通过                      │
             主类委托)                         │
                   │                          │
                   └──────────┬───────────────┘
                              ▼
                      ┌──────────────┐
                      │ EventHandler │
                      │ (事件逻辑)    │
                      └──────────────┘
```

**管理器间通信规则**：
- 管理器**禁止**直接引用其他管理器（`self.pw.other_manager.xxx` ❌）
- 跨管理器调用**必须通过主类委托**（`self.pw.some_method()` ✅）
- 主类方法决定路由到哪个管理器
- **v3.0 新增**: 信号连接和高频路径 (`_on_range_changed`) 保留在主类

#### 2.2.2 新的类结构（7 个管理器）

```
DraggableGraphicsLayoutWidget (继承 pg.GraphicsLayoutWidget)
├── 引用以下管理器类
│   ├── PlotUIManager          (UI 初始化 + 刷新协调)
│   ├── AxisManager            (坐标轴管理)
│   ├── PlotDataManager        (单曲线绘图 + 时间修正)
│   ├── MultiCurveManager      (多曲线 + 拖拽 + 样式)
│   ├── CursorManager          (光标 + 光标相关信号 + "off" 模式)
│   ├── MarkRegionManager      (标记区域)
│   └── EventHandler           (事件业务逻辑，不含信号连接)
├── 保留在主类的方法 (v3.0 明确)
│   ├── __init__               (创建管理器、初始化)
│   ├── setup_ui               (委托 PlotUIManager)
│   ├── _connect_viewbox_signals (信号连接路由层)
│   ├── _on_range_changed      (高频路径薄委托)
│   ├── Qt 事件方法            (薄委托层，分发到 EventHandler)
│   └── 所有公有 API           (委托给对应管理器)
└── 属性委托                   (通过 @property 保持 API 兼容)
```

#### 2.2.3 各管理器详细设计

##### (1) `PlotUIManager` — UI 初始化和刷新协调
- **职责**: 初始化绘图区域、头部标签、坐标轴布局、防抖和批量更新
- **包含方法**: `setup_ui`, `setup_header`, `setup_plot_area`, `setup_axes`, `setup_interaction`, `_init_ui_refresh_coordinator`, `_queue_ui_refresh`, `_cancel_ui_refresh`, `_run_style_refresh`, `_run_cursor_refresh`, `_run_stats_refresh`
- **位置**: `src/ui/widgets/plot_ui_manager.py`
- **v2.0 变更**: 合并了原 `UIRefreshCoordinator`，因为刷新协调是 UI 生命周期的一部分

##### (2) `AxisManager` — 坐标轴管理
- **职责**: X/Y 轴范围、边界、标签管理
- **包含方法**: `update_x_axis_label`, `set_xrange_with_link_handling`, `_get_safe_x_range`, `_get_min_x_range_value`, `_set_x_limits_with_min_range`, `_set_min_x_range`, `_set_safe_y_range`, `_setup_plot_axes`, `_reset_plot_limits`, `_recalc_max_point_density`, `auto_range`, `auto_y_in_x_range`
- **位置**: `src/ui/widgets/axis_manager.py`
- **v2.0 变更**: 无合并，职责清晰且被多方依赖，保持独立

##### (3) `PlotDataManager` — 单曲线绘图和数据管理
- **职责**: 单曲线绘制、数据准备、验证、清除、时间修正、数据变换
- **包含方法**: `plot_variable`, `_validate_plot_data`, `_get_x_data_for_variable`, `_prepare_plot_data`, `_compute_valid_min_max`, `_get_y_range_in_x_window`, `handle_single_point_limits`, `_clear_plot_data`, `clear_plot_item`, `reset_plot`, `clear_value_cache`, `_safe_clear_plot_items`, `datetime_to_unix_seconds`, `update_time_correction`, `get_value_from_name`
- **位置**: `src/ui/widgets/plot_data_manager.py`
- **v2.0 变更**: 合并了原 `TimeCorrectionManager`，因为时间修正本质是数据变换，且 `get_value_from_name()` 被两者共用

##### (4) `MultiCurveManager` — 多曲线管理（含拖拽和样式）
- **职责**: 多曲线添加、图例、可见性切换、拖拽放置、曲线样式
- **包含方法**: `add_variables_to_plot`, `add_variable_to_plot`, `update_multi_curve_mode`, `update_legend`, `toggle_curve_visibility_by_name`, `_recreate_curve`, `_on_legend_clicked`, `_update_axes_for_multi_curve`, `_update_x_limits_for_plot`, `_collect_visible_curve_arrays`, `_collect_visible_curve_pairs`, `get_curve_x_limits`, `handle_drag_enter`, `handle_drag_move`, `handle_drag_leave`, `handle_drop`, `_extract_var_names_from_text`, `_should_hide_drag_indicator`, `_enforce_drag_indicator_visibility`, `_notify_drag_indicator`, `_apply_plot_style`, `update_plot_style`, `_calculate_visible_points`
- **位置**: `src/ui/widgets/multi_curve_manager.py`
- **v2.0 变更**: 合并了原 `DragDropManager` + `StyleManager`，因为拖拽→添加曲线→样式更新是一条完整链路，样式直接操作 curves 字典

##### (5) `CursorManager` — 光标管理（含光标相关信号 + "off" 模式）
- **职责**: 光标位置、标签、模式、对象池管理、光标相关 ViewBox 信号处理、光标模式 "off" 选项
- **包含方法**: `_get_cursor_mode`, `_get_cursor_x_positions`, `_set_vline_visibility_for_mode`, `_set_vline_bounds`, `apply_cursor_mode`, `update_cursor_label`, `_update_single_curve_cursor_label`, `_update_multi_curve_cursor_label`, `_position_labels_avoid_overlap`, `toggle_cursor`, `_show_x_position_only`, `_has_visible_curve_data`, `pin_cursor`, `free_cursor`, `reset_pin_state`, `_update_vline_bounds_from_data`, `_update_cursor_after_plot`, `_get_circle_from_pool`, `_get_label_from_pool`, `_get_x_label_from_pool`, `_clear_cursor_items`, `_queue_item_for_deletion`, `_process_pending_deletes`, `_on_vb_set_cursor_mode`, `_on_vb_show_cursor`, `_on_vb_hide_cursor`, `_start_interaction`, `_end_interaction`, `_schedule_cursor_geometry_update`, `_refresh_cursor_geometry`, `_is_cursor_update_locked`
- **位置**: `src/ui/widgets/cursor_manager.py`
- **v2.0 变更**: 合并了原 `ViewBoxSignalHandler` 中的光标相关信号处理，因为光标操作和光标信号处理高度耦合
- **v3.0 变更**: 整合 [cursor_mode_modification_plan.md](file:///Users/xiaolin/Documents/python_repo/csv_plot/cursor_mode_modification_plan.md) 的 "off" 模式功能，包括 `last_valid_cursor_mode` 状态管理
- **注意**: 与旧 `src/ui/cursor_manager.py`（Mixin）区分，此为新的组合式管理器

##### (6) `MarkRegionManager` — 标记区域管理
- **职责**: 区域选择和统计计算
- **包含方法**: `add_mark_region`, `remove_mark_region`, `update_mark_region`, `get_mark_stats`
- **位置**: `src/ui/widgets/mark_region_manager.py`
- **v2.0 变更**: 无合并，职责清晰且相对独立

##### (7) `EventHandler` — 事件路由和非光标信号
- **职责**: Qt 事件分发业务逻辑、非光标 ViewBox 信号处理
- **包含方法**: `handle_double_click`, `handle_wheel`, `handle_mouse_moved`, `handle_resize`, `handle_mouse_press`, `handle_mouse_move`, `handle_mouse_release`, `_on_vb_jump`, `_on_vb_clear`, `_on_vb_auto_y`, `_on_vb_set_row_height`, `_on_vb_set_all_row_height`, `_on_vb_copy_name`, `_on_vb_var_editor`
- **位置**: `src/ui/widgets/event_handler.py`
- **v2.0 变更**: Qt 事件方法**保留在主类**作为薄委托层，EventHandler 只提供业务逻辑方法
- **v3.0 修正**: **移除** `_connect_viewbox_signals` 和 `_on_range_changed`；信号连接保留在主类，高频路径保留在主类薄委托

---

### 2.3 MainWindow 拆解方案

#### 2.3.1 新的类结构（4 个管理器）

```
MainWindow (继承 QMainWindow)
├── 引用以下管理器类
│   ├── MainWindowUIManager    (主窗口 UI + eventFilter 业务逻辑)
│   ├── FileLoaderManager      (文件加载)
│   ├── LayoutManager          (布局 + 标记区域同步)
│   └── CursorSyncManager      (光标同步 + 绘图同步)
├── PlotContainerWidget        (提前迁移至 src/ui/widgets/plot_container.py)
└── 保留的核心公有方法
    ├── __init__
    ├── 所有对外暴露的公有接口 (委托给管理器)
    └── eventFilter            (保留在主类，业务逻辑委托)
```

#### 2.3.2 各管理器详细设计

##### (1) `MainWindowUIManager` — 主窗口 UI 管理
- **职责**: 主窗口布局、左侧变量面板、右侧绘图区域、分隔条、eventFilter 业务逻辑
- **包含方法**: `setup_ui`, `_on_splitter_moved`, `_ensure_splitter_ready`, `_apply_fixed_splitter_width`, `handle_resize`, `toggle_plot_area`, `show_help`, `_get_plot_container`, `_show_drag_indicator_for_plot`, `_hide_drag_indicator_for_plot`, `spawn_clone_window`, `set_button_status`
- **位置**: `src/ui/main_window_ui_manager.py`
- **v3.0 变更**: 明确包含 eventFilter 业务逻辑

##### (2) `FileLoaderManager` — 文件加载管理
- **职责**: 文件对话框、验证、同步/异步加载
- **包含方法**: `load_btn_click`, `_validate_file_path`, `_check_file_size`, `_begin_data_reload`, `_end_data_reload`, `_post_reload_ui_refresh`, `load_csv_file`, `reload_data`, `_load_file`, `_has_valid_loader`, `_has_valid_data`, `_current_data_length`, `_cleanup_old_data`, `_post_load_actions`, `_remember_last_open_dir`, `_get_dialog_initial_directory`, `_default_system_directory`, `load_dict`, `_extract_file_extension`, `_validate_load_parameters`, `_load_sync`, `_on_load_done`, `_on_load_error`, `_apply_loader`
- **位置**: `src/ui/file_loader_manager.py`

##### (3) `LayoutManager` — 布局管理（含标记区域同步）
- **职责**: 子图网格、行高调整、可见性控制、跨绘图区域标记同步
- **包含方法**: `open_layout_dialog`, `open_time_correction_dialog`, `update_mark_regions_on_layout_change`, `create_subplots_matrix`, `set_row_height`, `set_all_row_height`, `get_row_height`, `set_plots_visible`, `toggle_mark_region`, `sync_mark_regions`, `request_mark_stats_refresh`, `_flush_mark_stats_refresh`, `update_mark_stats`
- **位置**: `src/ui/layout_manager.py`
- **v2.0 变更**: 合并了原 `MarkRegionSyncManager`，因为标记区域同步是布局管理的一部分

##### (4) `CursorSyncManager` — 光标同步管理（含绘图同步）
- **职责**: 跨绘图区域同步光标、全局缩放、范围同步、重载后重绘
- **包含方法**: `reset_plots_after_loading`, `_get_cursor_source_plot`, `_get_cursor_view_range`, `_clamp_value`, `_calc_second_cursor_position`, `_select_farthest_cursor_index`, `_apply_cursor_mode_to_plots`, `set_cursor_mode`, `toggle_cursor_all`, `_realign_pinned_cursor_after_time_correction`, `sync_crosshair`, `_flush_crosshair_updates`, `reset_all_pin_states`, `clear_all_plots`, `collect_global_x_range`, `_compute_baseline_density`, `_sync_min_xrange`, `auto_range_all_plots`, `auto_y_in_x_range`, `replots_after_loading`, `filter_variables`
- **位置**: `src/ui/cursor_sync_manager.py`
- **v2.0 变更**: 合并了原 `PlotSyncManager`，因为光标同步和绘图同步共享 `plot_widgets` 遍历逻辑
- **v3.0 变更**: 明确包含 cursor_mode "off" 模式下的 `last_valid_cursor_mode` 同步逻辑

---

### 2.4 状态属性归属表

| 属性 | 归属管理器 | 主类 @property 委托 | 说明 |
|------|-----------|---------------------|------|
| `curve` | PlotDataManager | ✅ | 单曲线对象 |
| `curves` | MultiCurveManager | ✅ | 多曲线字典 |
| `y_name` / `x_name` | PlotDataManager | ✅ | 变量名 |
| `y_format` / `x_format` | PlotDataManager | ✅ | 格式字符串 |
| `factor` / `offset` | PlotDataManager | ✅ | 时间修正系数 |
| `original_index_x` / `original_y` | PlotDataManager | ✅ | 原始数据 |
| `data` / `units` / `time_channels_info` | PlotDataManager | ✅ | 数据引用 |
| `is_multi_curve_mode` | MultiCurveManager | ✅ | 多曲线模式标志 |
| `current_color_index` / `curve_colors` | MultiCurveManager | ✅ | 颜色管理 |
| `_batch_adding` | MultiCurveManager | ✅ | 批量添加标志 |
| `_drag_indicator_hidden_by` | MultiCurveManager | ✅ | 拖拽指示器状态 (v3.0 新增) |
| `vline` | CursorManager | ✅ | 竖直光标线 |
| `is_cursor_pinned` | CursorManager | ✅ | 固定光标标志 |
| `pinned_x_value` / `pinned_x_values` | CursorManager | ✅ | 固定光标位置 |
| `pinned_index_values` | CursorManager | ✅ | 固定索引值 |
| `_is_interacting` | CursorManager | ✅ | 交互状态标志 (v3.0 新增) |
| `_cursor_label_busy` | CursorManager | ✅ | 光标标签更新标志 (v3.0 新增) |
| `_cursor_label_dirty` | CursorManager | ✅ | 光标标签需要刷新标志 (v3.0 新增) |
| `_pending_delete_items` | CursorManager | ✅ | 待删除对象池 (v3.0 新增) |
| `_cleanup_timer` | CursorManager | ✅ | 清理定时器 (v3.0 新增) |
| `show_values_only` | CursorManager | ✅ | 仅显示数值标志 (v3.0 新增) |
| `last_valid_cursor_mode` | CursorManager | ✅ | 非 "off" 的最后模式 (v3.0 新增) |
| `mark_region` | MarkRegionManager | ✅ | 标记区域对象 |
| `plot_item` / `view_box` | 主类保留 | — | PyQtGraph 核心对象 |
| `axis_x` / `axis_y` | 主类保留 | — | 坐标轴对象 |
| `label_left` / `label_right` | 主类保留 | — | 头部标签 |
| `_is_updating_data` | 主类保留 | — | 全局安全标志 |
| `_is_being_destroyed` | 主类保留 | — | 全局安全标志 |
| `_max_point_density` | AxisManager | ✅ | 点密度 |
| `synchronizer` | 主类保留 | — | 同步器引用 |
| `rubberBand` / `origin` | 主类保留 | — | 框选功能 |
| `time_values` / `time_column_name` / `time_axis_label` | PlotDataManager | ✅ | 时间轴信息 |

---

## 三、文件夹组织结构设计

### 3.1 目标目录结构

```
src/
├── __init__.py
│
├── app/
│   ├── __init__.py
│   └── plot_context.py          # ✏️ 扩展 PlotServices 协议 (含 set_cursor_enabled)
│
├── core/
│   ├── __init__.py
│   ├── config.py                # (保持不变)
│   ├── scheduler.py             # (保持不变)
│   └── types.py                 # (保持不变)
│
├── data/
│   ├── __init__.py
│   └── loader.py                # (保持不变)
│
└── ui/
    ├── __init__.py
    ├── cursor_manager.py        # ⚠️ 添加 DeprecationWarning，过渡期保留
    ├── drag_drop.py             # (保持不变)
    ├── mark_stats.py            # (保持不变)
    ├── plot_variable_editor.py  # (保持不变)
    ├── table_dialog.py          # (保持不变)
    ├── variable_list.py         # (保持不变)
    │
    ├── dialogs/                 # (保持不变)
    │   ├── __init__.py
    │   ├── axis.py
    │   ├── help.py
    │   ├── layout_input.py
    │   └── time_correction.py
    │
    ├── widgets/                 # (新增绘图管理器)
    │   ├── __init__.py
    │   ├── custom_viewbox.py    # (保持不变)
    │   ├── base_manager.py      # 🆕 管理器基类 (含 weakref + 生命周期钩子)
    │   ├── plot_container.py    # 🆕 PlotContainerWidget 迁移 (v3.0 提前)
    │   │
    │   ├── plot_ui_manager.py
    │   ├── axis_manager.py
    │   ├── cursor_manager.py    # 🆕 新的组合式管理器，非 Mixin (含 off 模式)
    │   ├── plot_data_manager.py
    │   ├── multi_curve_manager.py
    │   ├── mark_region_manager.py
    │   └── event_handler.py
    │
    ├── main_window_ui_manager.py
    ├── file_loader_manager.py
    ├── layout_manager.py
    └── cursor_sync_manager.py
```

### 3.2 `csv_plot_pyqt6.py` 瘦身目标

重构后，[csv_plot_pyqt6.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/csv_plot_pyqt6.py) 将只包含：
1. 导入语句
2. 顶部工具函数 (`resource_path`)
3. `DraggableGraphicsLayoutWidget` 精简主类 (~400-500 行，含 @property 委托、信号连接、高频路径薄委托)
4. `MainWindow` 精简主类 (~300-400 行)
5. 主程序入口 (`if __name__ == "__main__":`)

**目标总行数**: 从 6691 行减少到约 900-1100 行

---

## 四、MRO 问题的预防与解决方案

### 4.1 之前 MRO 问题的根本原因

现有的 [cursor_manager.py](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/ui/cursor_manager.py) 是一个 Mixin 类，文档中明确写了 `class MyPlotWidget(CursorManager, pg.GraphicsLayoutWidget)` 的用法。这正是之前导致 MRO 问题的根源。当前 `DraggableGraphicsLayoutWidget` 已不使用此 Mixin，但文件仍存在。

### 4.2 本次重构的 MRO 防护策略

#### 4.2.1 彻底的组合模式 (No Inheritance, Only Composition)

**完全避免使用多重继承**，所有功能模块都设计为独立的管理器类，通过组合方式接入主类：

```python
# ✅ 正确：组合模式
class AxisManager:
    def __init__(self, plot_widget):
        self._pw_ref = weakref.ref(plot_widget)
    
    def update_x_axis_label(self):
        pw = self.pw
        # 实现逻辑
        pass

class DraggableGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    def __init__(self, ...):
        super().__init__()
        self.axis_manager = AxisManager(self)
    
    def update_x_axis_label(self):
        return self.axis_manager.update_x_axis_label()
```

#### 4.2.2 管理器基类 — 统一 weakref 和生命周期 (v3.0 更新)

```python
# src/ui/widgets/base_manager.py
import weakref
from typing import Any

class BasePlotManager:
    """所有绘图管理器的基类，提供统一的弱引用和生命周期管理。"""
    def __init__(self, plot_widget: Any):
        self._pw_ref = weakref.ref(plot_widget)
    
    @property
    def pw(self) -> Any:
        """获取关联的 plot_widget 引用，安全检查是否已被销毁。"""
        pw = self._pw_ref()
        if pw is None:
            raise RuntimeError(f"{type(self).__name__}: PlotWidget has been garbage collected")
        return pw
    
    def initialize(self) -> None:
        """在主类 setup_ui 完成后调用，用于执行初始化后的额外设置。
        
        子类可重写此方法以初始化计时器、连接信号等。
        """
        pass
    
    def cleanup(self) -> None:
        """在 plot_widget 销毁前调用，用于释放资源、断开信号等。
        
        子类可重写此方法以执行清理工作。
        """
        pass
    
    def reset(self) -> None:
        """在 reset_plot 或 clear_plot_item 后调用，用于重置管理器特有状态。
        
        子类可重写此方法以清空内部缓存、重置标志位等。
        """
        pass
```

**关键约束**：
- 管理器基类**禁止调用 `super().__init__()`**，避免 Qt 基类重复初始化
- 管理器尽量**不继承 QObject**，将 QTimer 等挂载到 `plot_widget` 上
- 如必须继承 QObject（如需要信号），确保 `parent` 参数正确传递

#### 4.2.3 `__init_subclass__` 运行时防护

```python
class DraggableGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    def __init_subclass__(cls, **kwargs):
        bases = cls.__bases__
        qt_bases = [b for b in bases if issubclass(b, (pg.GraphicsLayoutWidget, QWidget))]
        if len(qt_bases) > 1:
            raise TypeError(
                f"{cls.__name__} 继承了多个 Qt/PyQtGraph 基类，"
                f"这会导致 MRO 问题。请使用组合模式替代。"
            )
        super().__init_subclass__(**kwargs)
```

#### 4.2.4 管理器间通信规则

- **禁止**管理器间直接引用（`self.pw.other_manager.xxx` ❌）
- 跨管理器调用**必须通过主类委托**（`self.pw.some_method()` ✅）
- 主类方法决定路由到哪个管理器

#### 4.2.5 旧 Mixin 文件处理

1. 在 `src/ui/cursor_manager.py` 中添加 `DeprecationWarning`
2. 新的组合式 `CursorManager` 放在 `src/ui/widgets/cursor_manager.py`
3. 过渡期结束后删除旧文件

---

## 五、前置步骤：消除 `self.window()` 依赖

> **这是整个重构的前提条件**，必须在管理器拆分之前完成。

### 5.1 扩展 PlotServices 协议 (v3.0 更新)

当前 [PlotServices](file:///Users/xiaolin/Documents/python_repo/csv_plot/src/app/plot_context.py#L14-L47) 协议仅覆盖了部分接口。需要扩展以覆盖所有 `self.window()` 和 `pw.window()` 访问：

```python
class PlotServices(Protocol):
    # 现有属性 (保持不变)
    @property
    def loader(self) -> Any: ...
    @property
    def plot_widgets(self) -> list[Any]: ...
    @property
    def cursor_mode(self) -> str: ...
    # ... (现有属性)
    
    # 🆕 需要新增的属性和方法
    @property
    def _global_max_density(self) -> float: ...
    @property
    def value_cache(self) -> dict: ...
    @property
    def _is_loading_new_data(self) -> bool: ...
    @property
    def _is_time_correction_active(self) -> bool: ...
    @property
    def _enum_text_maps(self) -> dict: ...
    
    def _sync_min_xrange(self) -> None: ...
    def _get_plot_container(self, plot_widget: Any) -> Any: ...
    def _show_drag_indicator_for_plot(self, plot_widget: Any, var_names: list[str], text_override: str | None = None) -> None: ...
    def _hide_drag_indicator_for_plot(self, plot_widget: Any) -> None: ...
    def request_mark_stats_refresh(self, immediate: bool = False) -> None: ...
    def auto_y_in_x_range(self) -> None: ...
    def collect_global_x_range(self, curves_filter: str = "visible") -> tuple[float | None, float | None]: ...
    def set_cursor_enabled(self, enabled: bool) -> None: ...  # v3.0 新增，来自 cursor_mode 计划
    def is_cursor_enabled(self) -> bool: ...  # v3.0 新增
```

### 5.2 注入时机调整

当前 `plot_context` 在 `create_subplots_matrix` 中后置注入。需要改为在 `DraggableGraphicsLayoutWidget.__init__` 中注入：

```python
class DraggableGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    def __init__(self, units_dict, dataframe, time_channels_info=None, synchronizer=None, plot_context=None):
        super().__init__()
        self.plot_context = plot_context  # 🆕 构造时注入
        # ...

# 在 MainWindow.create_subplots_matrix 中同步调整：
plot_widget = DraggableGraphicsLayoutWidget(
    units_dict, dataframe, time_channels_info,
    synchronizer=synchronizer,
    plot_context=PlotContext(self)  # 构造时注入而非后置赋值
)
```

### 5.3 迁移策略 (v3.0 更新)

逐个替换 `self.window()`、`pw.window()`、`plot_widget.window()` 调用为对应的 `plot_context` 调用：

| 原调用 | 替换为 |
|--------|--------|
| `self.window().loader` | `self.plot_context.loader` |
| `self.window().cursor_btn` | `self.plot_context.cursor_btn` |
| `self.window().plot_widgets` | `self.plot_context.plot_widgets` |
| `self.window()._sync_min_xrange()` | `self.plot_context._sync_min_xrange()` |
| `self.window().sync_crosshair(x, self)` | `self.plot_context.sync_crosshair(x, self)` |
| `self.window().request_mark_stats_refresh()` | `self.plot_context.request_mark_stats_refresh()` |
| `main_window.value_cache` | `self.plot_context.value_cache` |
| `main_window._global_max_density` | `self.plot_context._global_max_density` |
| `pw.window().xxx` | `pw.plot_context.xxx` |
| `plot_widget.window().xxx` | `plot_widget.plot_context.xxx` |

---

## 六、重构实施步骤 (v3.0 重排)

### 阶段零：前置准备 (Step 0)

| 步骤 | 任务 | 测试重点 |
|-----|------|---------|
| Step 0a | 扩展 PlotServices 协议，覆盖所有 `self.window()` 和 `pw.window()` 接口，新增 `set_cursor_enabled`/`is_cursor_enabled` | 协议完整性 |
| Step 0b | 在 `DraggableGraphicsLayoutWidget.__init__` 中注入 `plot_context`，修改 `create_subplots_matrix` | 构造时注入正常 |
| Step 0c | 逐个替换 `self.window()`、`pw.window()`、`plot_widget.window()` → `plot_context` (53+19+53处) | 每替换 10 处做一次功能测试 |
| Step 0d | 在旧 `cursor_manager.py` 添加 DeprecationWarning | 导入时显示警告 |
| Step 0e | 创建管理器基类 `BasePlotManager` (含 weakref + 生命周期钩子) | 基类可正常使用 |
| Step 0f | 创建所有新模块的空文件，在管理器模块中应用延迟导入模式 | 导入不报错 |
| Step 0g | **新增 (v3.0)**: 建立性能基线脚本，测试 `_on_range_changed` 耗时、100万点加载帧率 | 性能基线数据完整 |
| Step 0h | **新增 (v3.0)**: 迁移 `PlotContainerWidget` 至 `src/ui/widgets/plot_container.py` | 拖拽指示器功能正常 |

### 阶段一：DraggableGraphicsLayoutWidget 底层管理器 (Step 1 - Step 2)

| 步骤 | 任务 | 测试重点 |
|-----|------|---------|
| Step 1 | 创建 `PlotUIManager`，迁移 UI 初始化 + 刷新协调 | 绘图区域正确显示、防抖正常 |
| Step 2 | 创建 `AxisManager`，迁移坐标轴管理 | 缩放、平移、范围限制正常 |

### 阶段二：DraggableGraphicsLayoutWidget 数据层管理器 (Step 3 - Step 4)

| 步骤 | 任务 | 测试重点 |
|-----|------|---------|
| Step 3 | 创建 `PlotDataManager`，迁移单曲线绘图 + 时间修正 | 单变量绘图、时间修正正常 |
| Step 4 | 创建 `MultiCurveManager`，迁移多曲线 + 拖拽 + 样式 | 多变量、图例、拖拽、样式切换正常 |

### 阶段三：DraggableGraphicsLayoutWidget 上层管理器 (Step 5 - Step 7)

| 步骤 | 任务 | 测试重点 |
|-----|------|---------|
| Step 5 | 创建 `CursorManager`，迁移光标 + 光标信号处理 + 整合 "off" 模式功能 | 光标模式、标签、固定功能、"off" 模式正常 |
| Step 6 | 创建 `MarkRegionManager`，迁移标记区域 | 区域选择、统计计算正常 |
| Step 7 | 创建 `EventHandler`，迁移事件业务逻辑；主类 Qt 事件方法改为薄委托；**信号连接保留在主类** | 鼠标、滚轮、双击事件正常 |

### 阶段四：DraggableGraphicsLayoutWidget 集成 (Step 8)

| 步骤 | 任务 | 测试重点 |
|-----|------|---------|
| Step 8 | 集成所有管理器到主类，添加 @property 委托，清理主类代码，实现管理器生命周期钩子调用 | 整体功能回归测试、性能对比测试 |

### 阶段五：MainWindow 重构 (Step 9 - Step 12)

| 步骤 | 任务 | 测试重点 |
|-----|------|---------|
| Step 9 | 创建 `MainWindowUIManager`，迁移 UI 初始化 + eventFilter 业务逻辑 | 主窗口布局正确 |
| Step 10 | 创建 `FileLoaderManager`，迁移文件加载方法 | 文件打开、重载、进度显示正常 |
| Step 11 | 创建 `LayoutManager`，迁移布局 + 标记同步 | 子图网格、行高、标记区域同步正常 |
| Step 12 | 创建 `CursorSyncManager`，迁移光标同步 + 绘图同步 | 光标同步、全局缩放、重载后恢复正常 |

### 阶段六：收尾与验证 (Step 13 - Step 15)

| 步骤 | 任务 |
|-----|------|
| Step 13 | 在 `src/ui/widgets/__init__.py` 中统一导出公共接口 |
| Step 14 | 清理导入，删除旧 `cursor_manager.py` (如过渡期已过)，优化模块结构 |
| Step 15 | 全面回归测试 + 性能基准对比测试 + 代码审查 |

### 验证检查点

每个步骤完成后，执行以下自动化检查：

1. **MRO 检查**: 确认 `DraggableGraphicsLayoutWidget.mro()` 和 `MainWindow.mro()` 无冲突
2. **API 兼容性检查**: 对比 `dir()` 输出，确保公有方法未丢失
3. **内存泄漏检查**: 使用 `tracemalloc` 检查对象引用计数
4. **功能回归测试**: 执行手动测试清单
5. **v3.0 新增**: **性能回归检查**: 对比性能基线，无 >5% 退化

---

## 七、测试验证计划 (v3.0 增强)

### 7.1 手动测试清单

#### DraggableGraphicsLayoutWidget 功能测试
- [ ] 单变量拖拽绘图
- [ ] 多变量拖拽绘图
- [ ] 图例点击切换可见性
- [ ] X 轴缩放、平移
- [ ] Y 轴自动缩放
- [ ] 光标显示/隐藏
- [ ] 自由光标模式
- [ ] 固定光标模式
- [ ] 双固定光标模式
- [ ] **v3.0 新增**: 光标 "off" 模式
- [ ] 标记区域选择
- [ ] 标记区域统计显示
- [ ] 时间修正功能
- [ ] 滚轮缩放
- [ ] 鼠标平移
- [ ] 曲线样式切换 (细线+符号/粗线无符号)
- [ ] 数据重载后恢复
- [ ] 右键菜单功能
- [ ] 框选缩放
- [ ] 双击坐标轴弹出轴设置
- [ ] 双击绘图区弹出变量编辑器

#### MainWindow 功能测试
- [ ] 应用程序启动
- [ ] 打开 CSV 文件
- [ ] 打开 MDF 文件
- [ ] 文件拖拽到窗口
- [ ] 大文件异步加载
- [ ] 变量搜索过滤
- [ ] 隐藏/显示绘图区域
- [ ] 分身窗口
- [ ] 帮助对话框
- [ ] 布局对话框 (修改行列)
- [ ] 时间修正对话框
- [ ] 轴设置对话框
- [ ] 自动缩放所有绘图
- [ ] 仅 Y 轴自动缩放
- [ ] 显示/隐藏所有光标
- [ ] **v3.0 新增**: 顶部按钮与光标模式菜单双向联动
- [ ] 标记区域显示/隐藏
- [ ] 标记区域跨绘图同步
- [ ] 光标跨绘图同步
- [ ] 行高调整
- [ ] 清除所有绘图
- [ ] 数据重载
- [ ] 窗口大小调整

### 7.2 性能测试
- 加载 100 万点数据文件，检查内存占用
- 加载 100 万点数据文件，检查缩放流畅度
- 对比重构前后的性能差异
- **高频路径性能**: 确保 `_on_range_changed`、`mouse_moved` 等高频调用无明显退化
- **v3.0 新增**: `_on_range_changed` 单次调用耗时 <1ms 目标

### 7.3 回归测试
- 重点检查边缘情况 (如单数据点、空数据、NaN 值)
- 对比重构前后 `dir(DraggableGraphicsLayoutWidget)` 和 `dir(MainWindow)` 输出

### 7.4 v3.0 新增: 管理器单元测试框架

为每个管理器创建独立的单元测试，测试时可以提供 mock 的 plot_widget：

| 测试文件 | 覆盖管理器 | 测试重点 |
|---------|-----------|---------|
| `test_axis_manager.py` | AxisManager | 范围计算、标签更新 |
| `test_plot_data_manager.py` | PlotDataManager | 数据验证、时间修正 |
| `test_cursor_manager.py` | CursorManager | 模式切换、对象池、"off" 模式 |
| `test_mark_region_manager.py` | MarkRegionManager | 统计计算、区域更新 |

---

## 八、风险评估与应对 (v3.0 更新)

| 风险 | 可能性 | 影响 | 应对措施 |
|-----|-------|------|---------|
| `self.window()`/`pw.window()` 迁移遗漏 | 中 | 高 | 代码搜索确认零残留；添加 lint 规则禁止 `.window()` 调用（含变体） |
| 管理器间循环依赖 | 低 | 高 | 通信规则：仅通过主类委托；依赖图审查 |
| 性能下降 | 中 | 中 | 高频路径不走委托；性能基准测试；性能回归检查点 |
| 公有 API 意外变更 | 低 | 高 | `dir()` 对比；`@property` 委托保持兼容 |
| 内存泄漏 | 中 | 中 | weakref 防循环引用；`tracemalloc` 检查；管理器 `cleanup()` 钩子 |
| Qt 事件分发错误 | 中 | 高 | 事件方法保留在主类；逐步迁移业务逻辑 |
| `plot_context` 注入时机问题 | 中 | 高 | 构造时注入而非后置注入；None 检查和降级 |
| **v3.0 新增**: `_lazy_*` 延迟导入路径在管理器拆分后失效 | 中 | 中 | 对话框从主类通过依赖注入传递给管理器；主类保留所有 `_lazy_*` 函数 |
| **v3.0 新增**: `PlotVariableEditorDialog(pw, pw.window())` 兼容性问题 | 中 | 高 | 对话框构造时改为接收 `plot_context`；验证兼容性，必要时扩展协议 |

---

## 九、附录

### A.1 管理器基类 (v3.0 更新)

```python
# src/ui/widgets/base_manager.py
import weakref
from typing import Any

class BasePlotManager:
    """所有绘图管理器的基类，提供统一的弱引用和生命周期管理。"""
    def __init__(self, plot_widget: Any):
        self._pw_ref = weakref.ref(plot_widget)
    
    @property
    def pw(self) -> Any:
        """获取关联的 plot_widget 引用，安全检查是否已被销毁。"""
        pw = self._pw_ref()
        if pw is None:
            raise RuntimeError(f"{type(self).__name__}: PlotWidget has been garbage collected")
        return pw
    
    def initialize(self) -> None:
        """在主类 setup_ui 完成后调用，用于执行初始化后的额外设置。
        
        子类可重写此方法以初始化计时器、连接信号等。
        """
        pass
    
    def cleanup(self) -> None:
        """在 plot_widget 销毁前调用，用于释放资源、断开信号等。
        
        子类可重写此方法以执行清理工作。
        """
        pass
    
    def reset(self) -> None:
        """在 reset_plot 或 clear_plot_item 后调用，用于重置管理器特有状态。
        
        子类可重写此方法以清空内部缓存、重置标志位等。
        """
        pass
```

### A.2 主类使用管理器的模板 (v3.0 更新)

```python
# csv_plot_pyqt6.py (重构后)
from src.ui.widgets.plot_ui_manager import PlotUIManager
from src.ui.widgets.axis_manager import AxisManager
from src.ui.widgets.plot_data_manager import PlotDataManager
from src.ui.widgets.multi_curve_manager import MultiCurveManager
from src.ui.widgets.cursor_manager import CursorManager
from src.ui.widgets.mark_region_manager import MarkRegionManager
from src.ui.widgets.event_handler import EventHandler

class DraggableGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    def __init_subclass__(cls, **kwargs):
        bases = cls.__bases__
        qt_bases = [b for b in bases if issubclass(b, (pg.GraphicsLayoutWidget, QWidget))]
        if len(qt_bases) > 1:
            raise TypeError(
                f"{cls.__name__} 继承了多个 Qt/PyQtGraph 基类，"
                f"这会导致 MRO 问题。请使用组合模式替代。"
            )
        super().__init_subclass__(**kwargs)
    
    def __init__(self, units_dict, dataframe, time_channels_info=None,
                 synchronizer=None, plot_context=None):
        super().__init__()
        self.plot_context = plot_context
        
        # 1. 底层管理器 (无依赖)
        self.ui_manager = PlotUIManager(self)
        self.axis_manager = AxisManager(self)
        # 2. 数据层管理器 (依赖 AxisManager)
        self.plot_data_manager = PlotDataManager(self)
        self.multi_curve_manager = MultiCurveManager(self)
        # 3. 上层管理器 (依赖前面所有)
        self.cursor_manager = CursorManager(self)
        self.mark_region_manager = MarkRegionManager(self)
        self.event_handler = EventHandler(self)
        
        # 初始化 UI
        self.ui_manager.setup_ui(units_dict, dataframe, time_channels_info, synchronizer)
        
        # v3.0: 连接 ViewBox 信号 (保留在主类作为路由层)
        self._connect_viewbox_signals()
        
        # v3.0: 调用管理器 initialize 钩子
        for manager in [self.ui_manager, self.axis_manager, self.plot_data_manager,
                       self.multi_curve_manager, self.cursor_manager,
                       self.mark_region_manager, self.event_handler]:
            manager.initialize()
    
    # --- v3.0: 信号连接保留在主类 ---
    def _connect_viewbox_signals(self):
        vb = self.view_box
        vb.plot_widget = self
        vb.signals.request_jump_to_data.connect(self.event_handler._on_vb_jump)
        vb.signals.request_clear_plot.connect(self.event_handler._on_vb_clear)
        vb.signals.request_auto_y.connect(self.event_handler._on_vb_auto_y)
        vb.signals.request_set_cursor_mode.connect(self.cursor_manager._on_vb_set_cursor_mode)
        vb.signals.request_show_cursor_value.connect(self.cursor_manager._on_vb_show_cursor)
        vb.signals.request_hide_cursor_value.connect(self.cursor_manager._on_vb_hide_cursor)
        vb.signals.request_set_row_height.connect(self.event_handler._on_vb_set_row_height)
        vb.signals.request_set_all_row_height.connect(self.event_handler._on_vb_set_all_row_height)
        vb.signals.request_copy_name.connect(self.event_handler._on_vb_copy_name)
        vb.signals.request_variable_editor.connect(self.event_handler._on_vb_var_editor)
    
    # --- v3.0: 高频路径 _on_range_changed 保留在主类薄委托 ---
    def _on_range_changed(self, view_box, range, changed=None):
        try:
            if getattr(self, '_is_updating_data', False) or getattr(self, '_is_being_destroyed', False):
                self.ui_manager._cancel_ui_refresh()
                return
            
            if getattr(self, '_is_syncing_range', False):
                return
            
            if not self.cursor_manager._is_interacting:
                self.ui_manager._queue_ui_refresh()
            
            self.cursor_manager._schedule_cursor_geometry_update()
        except Exception as e:
            from src.core.config import debug_log
            debug_log(f"_on_range_changed error: {e}")
    
    # --- 属性委托 (保持 API 兼容) ---
    @property
    def curve(self):
        return self.plot_data_manager.curve
    @curve.setter
    def curve(self, value):
        self.plot_data_manager.curve = value
    
    @property
    def curves(self):
        return self.multi_curve_manager.curves
    @curves.setter
    def curves(self, value):
        self.multi_curve_manager.curves = value
    
    # ... 其他 @property 委托
    
    # --- Qt 事件方法 (薄委托层) ---
    def mouseDoubleClickEvent(self, event):
        self.event_handler.handle_double_click(event)
    
    def wheelEvent(self, ev):
        self.event_handler.handle_wheel(ev)
    
    # ... 其他事件方法
    
    # --- 公有 API 委托 ---
    def auto_range(self, external_xmin=None, external_xmax=None):
        return self.axis_manager.auto_range(external_xmin, external_xmax)
    
    def plot_variable(self, var_name, show_duplicate_warning=True):
        return self.plot_data_manager.plot_variable(var_name, show_duplicate_warning)
    
    # ... 其他公有 API
    
    # --- v3.0: 生命周期钩子调用 ---
    def closeEvent(self, event):
        """在销毁前调用所有管理器的 cleanup 钩子。"""
        for manager in [self.event_handler, self.mark_region_manager,
                       self.cursor_manager, self.multi_curve_manager,
                       self.plot_data_manager, self.axis_manager,
                       self.ui_manager]:
            try:
                manager.cleanup()
            except Exception as e:
                from src.core.config import debug_log
                debug_log(f"Error cleaning up {type(manager).__name__}: {e}")
        super().closeEvent(event)
```

### A.3 旧 Mixin 弃用处理

```python
# src/ui/cursor_manager.py (添加弃用警告)
import warnings

warnings.warn(
    "src.ui.cursor_manager.CursorManager (Mixin) is deprecated. "
    "Use src.ui.widgets.cursor_manager.CursorManager (composition) instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# ... 保留原有代码，供过渡期使用
```

---

**文档版本**: 3.0  
**创建日期**: 2026-05-17  
**最后更新**: 2026-05-18  
**变更记录**:  
- v1.0 → v2.0 — 整合评审意见，合并管理器，增加前置步骤，强化 MRO 防护  
- v2.0 → v3.0 — 修正架构设计（信号连接回主类、高频路径保留），补充状态属性，增强前置步骤，增加生命周期管理，整合光标 "off" 模式，更新风险评估
