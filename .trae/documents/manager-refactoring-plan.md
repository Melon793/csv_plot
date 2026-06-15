# 重构计划：激活 Manager 层，消除 plot\_widget.py 重复代码

> **版本**: v8.0 (2026-06-15)
> **v1.0**: 初始计划
> **v2.0**: 新增代码审查报告（第六节），修正委托目标错误，补充遗漏方法清单
> **v3.0**: 第一阶段（AxisManager）完成，新增执行记录（第七节）
> **v4.0**: 第二阶段（CursorManager）完成
> **v5.0**: 第二阶段问题修复完成，新增代码一致性验证规范（第八节）
> **v6.0**: 第三阶段（PlotDataManager + MultiCurveManager）完成
> **v7.0**: v6.0 代码审查 + `_on_legend_clicked` fallback 修复，commit `de108c4`
> **v7.1**: Step 7 (MarkRegionManager) 完成，commit `efb16fa`
> **v8.0**: Steps 1-3 (EventHandler + PlotUIManager + 初始化管线) 完成，commit `876f1cc` → **全部 7/7 Manager 已激活**

---

## 一、当前状态总结

重构已完成！全部 7 个 Manager 已激活。

`DraggableGraphicsLayoutWidget` (plot\_widget.py) 从原始 ~4100+ 行减少到当前 1538 行。

* 所有 7 个 Manager 均已通过 `_init_manager_chain()` 创建
* `_init_manager_chain()` 已在 `__init__` 中的 `setup_ui()` 之前调用
* UI 初始化信号（sigRangeChanged、timer）通过 EventHandler 路由
* 所有 Manager 方法通过 widget 委托层调用

***

## 二、重构步骤

### 步骤 1：修复初始化管线（最关键）

**问题：** `__init__` 先调用 `self.setup_ui()`（line 96），在其中运行时调用 `_init_manager_chain()`（line 160），但此时 `setup_header`、`setup_plot_area`、`setup_axes`、`setup_interaction` 等已在 widget 自身方法中执行完毕。Manager 创建后没有任何作用。

**修改** **`plot_widget.py`** **的** **`__init__`：**

1. 将 `_init_manager_chain()` 调用从 `setup_ui()` 末尾（line 160）**移至** `__init__` 的最前面，在 `super().__init__()` 之后、`self.setup_ui()` 之前
2. 将 `setup_ui()` 方法体替换为对 `self._plot_ui_manager.setup_ui(units_dict, dataframe, ...)` 的委托调用

源码变更位置：`plot_widget.py` line 68-160

**修改** **`plot_ui_manager.py`** **的** **`PlotUIManager.setup_ui()`：**

当前 `_setup_interaction()` (line 259-310) 会创建 `pw.vline`、`pw.vline2`、`pw.cursor_label`、`pw._cursor_item_pool` 等光标基础设施。但按职责划分，这些应属于 CursorManager。

修复方案：

* `PlotUIManager._setup_interaction()` 中移除光标相关元素创建（line 264-293）

* 这些光标基础设施的创建保留在 widget 的 `setup_interaction()` 中，因为它们是 CursorManager 通过 `self.pw` 引用的下层属性，属于基础设施

* `PlotUIManager._setup_interaction()` 中保留的信号连接代码（line 296-310）与实际运行代码一致即可

**关键风险控制：**

* `plot_ui_manager.py:300-301` 的 `pw.vline.sigPositionChanged.connect(pw.on_vline_position_changed)` 会与 widget 自身的连接重复。需确保 **只保留一个连接源**。策略：widget 的 `setup_interaction()` 被移除后，只有 `PlotUIManager` 连接，不会重复。

* 验证 `super().__init__()` 的 `GraphicsLayoutWidget.__init__()` 在 Manager 创建之前执行不会导致问题（Manager 只存储 weakref，不依赖 widget 的 GraphicsLayoutWidget 初始化状态）

***

### 步骤 2：激活 EventHandler（信号路由层）

**文件：** `src/ui/widgets/event_handler.py`（247 行）
**文件：** `src/ui/widgets/plot_widget.py`

**目标：** 将 ViewBox 信号和 timer 信号全部通过 EventHandler 路由。

**2.1 信号连接迁移**

`plot_widget.py` 中以下连接需改为连接到 `self._event_handler` 的对应方法：

| widget 当前连接 (plot\_widget.py)                                                    | 应改为                                              |
| -------------------------------------------------------------------------------- | ------------------------------------------------ |
| line 283: `view_box.sigRangeChanged.connect(self._on_range_changed)`             | `→ self._event_handler._on_range_changed`        |
| line 254: `_interaction_timer.timeout.connect(self._end_interaction)`            | `→ self._event_handler._end_interaction`         |
| line 720: `_cursor_refresh_timer.timeout.connect(self._refresh_cursor_geometry)` | `→ self._event_handler._refresh_cursor_geometry` |
| line 4080-4092: `_connect_viewbox_signals()` 中全部 `self._on_vb_*`                 | `→ self._event_handler._on_vb_*`                 |
| line 709: `pg.SignalProxy(...).slot=self.mouse_moved`                            | 保留（mouse\_moved 属于 widget 级事件）                   |
| line 710-711: `vline.sigPositionChanged.connect(self.on_vline_position_changed)` | 保留（on\_vline\_position\_changed 操作 widget 状态）    |

**2.2 异步/定时器调度逻辑**

`EventHandler` 有自己的 `_schedule_cursor_geometry_update()`、`_start_interaction()`、`_end_interaction()` 方法，其实现与 widget 中的同名方法基本一致。激活后：

* widget 中保留定时器对象（`_cursor_refresh_timer`、`_interaction_timer`）作为成员变量

* widget 的 `_start_interaction()`、`_end_interaction()`、`_schedule_cursor_geometry_update()`、`_refresh_cursor_geometry()` 四个方法改为委托到 `self._event_handler` 对应方法

* widget 的 `_on_range_changed()` 方法改为委托到 `self._event_handler._on_range_changed()`

**2.3 ViewBox 信号处理方法**

`EventHandler` 有完整的一套 `_on_vb_*` 方法（`_on_vb_jump`、`_on_vb_clear`、`_on_vb_auto_y`、`_on_vb_set_cursor_mode`、`_on_vb_show_cursor`、`_on_vb_hide_cursor`、`_on_vb_set_row_height`、`_on_vb_set_all_row_height`、`_on_vb_copy_name`、`_on_vb_var_editor`），与 widget 中 line 4094-4157 的方法一一对应。

widget 中的这些方法改为委托到 `self._event_handler` 的对应方法。

**注意：** `EventHandler._cancel_ui_refresh()` 和 `_queue_ui_refresh()` 通过 `self.pw._cancel_ui_refresh()` / `self.pw._queue_ui_refresh()` 回调 widget，这意味着 widget 上需要保留这两个方法（它们将在步骤 3 中通过 PlotUIManager 实现）。

***

### 步骤 3：激活 PlotUIManager（UI 初始化 + 刷新协调）

**文件：** `src/ui/widgets/plot_ui_manager.py`
**文件：** `src/ui/widgets/plot_widget.py`

**已完成（步骤 1）：** `setup_ui()` 委托已建立。

**3.1 刷新协调方法委托**

widget 中的以下方法改为委托到 `self._plot_ui_manager`：

* `_queue_ui_refresh()` → `self._plot_ui_manager._queue_ui_refresh(self, ...)`

* `_cancel_ui_refresh()` → `self._plot_ui_manager._cancel_ui_refresh(self, ...)`

* `_run_style_refresh()` → `self._plot_ui_manager._run_style_refresh()`

* `_run_cursor_refresh()` → `self._plot_ui_manager._run_cursor_refresh()`

* `_run_stats_refresh()` → `self._plot_ui_manager._run_stats_refresh()`

* `_init_ui_refresh_coordinator()` → `self._plot_ui_manager._init_ui_refresh_coordinator(self)`

**3.2 Header/Plot Area/Axes 设置方法**

`setup_header()`、`setup_plot_area()`、`setup_axes()` 这些方法在 `PlotUIManager` 中已有对应（`_setup_header`、`_setup_plot_area`、`_setup_axes`），但他们是内部方法（`_` 前缀），通过 `setup_ui()` 编排调用。

由于步骤 1 已经将 `setup_ui()` 委托出去，这些单独的 setup 方法在 widget 中将不再被调用，可删除其实现体，保留空壳（标记为 deprecated）或直接移除。

**3.3** **`update_x_axis_label()`** **委托**

此方法在 `AxisManager` 中有对应实现。widget 中的 `update_x_axis_label()` → 委托到 `self._axis_manager.update_x_axis_label()`。

***

### 步骤 4：激活 CursorManager（光标逻辑，最大重复块 \~1100 行）

**文件：** `src/ui/widgets/cursor_manager.py`
**文件：** `src/ui/widgets/plot_widget.py`

**4.1 方法委托映射**

`CursorManager` 中以下方法与 widget 完全重复。widget 中的这些方法改为一句委托调用：

| widget 方法 (plot\_widget.py)                 | 委托到                                                        |
| ------------------------------------------- | ---------------------------------------------------------- |
| `update_cursor_label()` L1226               | `self._cursor_manager.update_cursor_label()`               |
| `_update_multi_curve_cursor_label()` L1630  | `self._cursor_manager._update_multi_curve_cursor_label()`  |
| `_update_single_curve_cursor_label()` L1265 | `self._cursor_manager._update_single_curve_cursor_label()` |
| `_position_labels_avoid_overlap()` L1811    | `self._cursor_manager._position_labels_avoid_overlap()`    |
| `_show_x_position_only()` L1998             | `self._cursor_manager._show_x_position_only()`             |
| `_get_circle_from_pool()` L1300             | `self._cursor_manager._get_circle_from_pool()`             |
| `_get_label_from_pool()` L1327              | `self._cursor_manager._get_label_from_pool()`              |
| `_get_x_label_from_pool()` L1360            | `self._cursor_manager._get_x_label_from_pool()`            |
| `_clear_cursor_items()` L1380               | `self._cursor_manager._clear_cursor_items()`               |
| `_queue_item_for_deletion()` L1457          | `self._cursor_manager._queue_item_for_deletion()`          |
| `_process_pending_deletes()` L1466          | `self._cursor_manager._process_pending_deletes()`          |
| `_is_cursor_update_locked()` L959           | `self._cursor_manager._is_cursor_update_locked()`          |
| `_has_visible_curve_data()` L2039           | `self._cursor_manager._has_visible_curve_data()`           |
| `toggle_cursor()` L1972                     | `self._cursor_manager.toggle_cursor()`                     |
| `apply_cursor_mode()` L1153                 | `self._cursor_manager.apply_cursor_mode()`                 |
| `pin_cursor()` L2057                        | `self._cursor_manager.pin_cursor()`                        |
| `free_cursor()` L2132                       | `self._cursor_manager.free_cursor()`                       |
| `reset_pin_state()` L2160                   | `self._cursor_manager.reset_pin_state()`                   |
| `_get_cursor_mode()` L1109                  | `self._cursor_manager._get_cursor_mode()`                  |
| `_get_cursor_x_positions()` L1115           | `self._cursor_manager._get_cursor_x_positions()`           |
| `_set_vline_visibility_for_mode()` L1135    | `self._cursor_manager._set_vline_visibility_for_mode()`    |
| `_set_vline_bounds()` L1147                 | ~~`self._cursor_manager._set_vline_bounds()`~~ → **修正：CursorManager 无此方法，应委托到 AxisManager（见步骤 5）** |
| `_update_view_range_from_data()` (L约2220)   | `self._cursor_manager._update_view_range_from_data()`      |
| `_update_cursor_after_plot()` L2235         | `self._cursor_manager._update_cursor_after_plot()`         |

**4.2** **`on_vline_position_changed()`** **处理**

此方法在 widget (line 989) 和 CursorManager (line 1020) 中均有实现。方法是 widget 的核心交互逻辑，需要**保留在 widget 中**（因为涉及跨 widget 同步 `self.window().plot_widgets` 循环）。

CursorManager 中的版本不会独立使用，改为 widget 调用 `self._cursor_manager.on_vline_position_changed(line_obj)` 时，CursorManager 只处理光标内部状态更新（pinned\_x\_values、pinned\_index\_values），而跨 widget 同步逻辑由 widget 自身处理。

**实际做法：** widget 的 `on_vline_position_changed()` 保留跨 widget 同步逻辑，内部光标状态更新部分委托到 `self._cursor_manager`。

**4.3 高级版** **`_position_labels_avoid_overlap`** **替换**

`cursor_manager.py` 中的简化版需要替换为 `plot_widget.py` 中的高级版（含动态字体尺寸计算、三级边缘避让、场景坐标钳制）。

* 将 `plot_widget.py` L1811-1970 的完整实现覆盖到 `cursor_manager.py` L726-811

* widget 中的方法体改为 `return self._cursor_manager._position_labels_avoid_overlap(cursor_values, x_min, x_max, y_min, y_max)`

***

### 步骤 5：激活 AxisManager

**文件：** `src/ui/widgets/axis_manager.py`
**文件：** `src/ui/widgets/plot_widget.py`

`AxisManager` 的方法目前只被死代码 `PlotDataManager` 调用。激活后，widget 需要委托以下方法：

检查 `plot_widget.py` 中与 AxisManager 方法对应的 widget 方法：

* `_get_safe_x_range()` → `self._axis_manager._get_safe_x_range()`

* `_set_x_limits_with_min_range()` → `self._axis_manager._set_x_limits_with_min_range()`

* `_set_safe_y_range()` → `self._axis_manager._set_safe_y_range()`

* `_set_vline_bounds()` → `self._axis_manager._set_vline_bounds()` **（修正：原计划误标为委托到 CursorManager，实际 CursorManager 无此方法）**

* `_recalc_max_point_density()` → `self._axis_manager._recalc_max_point_density()`

* `_setup_plot_axes()` → `self._axis_manager._setup_plot_axes()`

* `_reset_plot_limits()` → `self._axis_manager._reset_plot_limits()`

* `auto_range()` → `self._axis_manager.auto_range()`

* `auto_y_in_x_range()` → `self._axis_manager.auto_y_in_x_range()`

***

### 步骤 6：激活 PlotDataManager 和 MultiCurveManager

**文件：** `src/ui/widgets/plot_data_manager.py`、`multi_curve_manager.py`

这些 Manager 包含数据操作和多曲线逻辑。widget 中的对应方法改为委托：

**PlotDataManager：**

* `plot_variable()` → `self._plot_data_manager.plot_variable()`

* `clear_plot_item()` → `self._plot_data_manager.clear_plot_item()`

* `get_value_from_name()` → `self._plot_data_manager.get_value_from_name()`

* `handle_single_point_limits()` → `self._plot_data_manager.handle_single_point_limits()`

* `update_time_correction()` → `self._plot_data_manager.update_time_correction()`

* `datetime_to_unix_seconds()` → `self._plot_data_manager.datetime_to_unix_seconds()`

* `clear_value_cache()` → `self._plot_data_manager.clear_value_cache()`

**MultiCurveManager：**

* `update_multi_curve_mode()` → `self._multi_curve_manager.update_multi_curve_mode()`

* `update_legend()` → `self._multi_curve_manager.update_legend()`

* `toggle_curve_visibility_by_name()` → `self._multi_curve_manager.toggle_curve_visibility_by_name()`

* `get_curve_x_limits()` → `self._multi_curve_manager.get_curve_x_limits()`

* `_on_legend_clicked()` → `self._multi_curve_manager._on_legend_clicked()`

***

### 步骤 7：激活 MarkRegionManager

**文件：** `src/ui/widgets/mark_region_manager.py`

* `add_mark_region()` → `self._mark_region_manager.add_mark_region()`

* `remove_mark_region()` → `self._mark_region_manager.remove_mark_region()`

* `update_mark_region()` → `self._mark_region_manager.update_mark_region()`

* `get_mark_stats()` → `self._mark_region_manager.get_mark_stats()`

***

### 步骤 8：清理和验证

1. 确认 `plot_widget.py` 中所有被委托的方法体已替换为单句委托调用
2. 检查所有外部调用点 — 确保外部代码通过 `widget.xxx()` 调用时仍然有效（因为 widget 保留了方法签名，只是内部委托）
3. 运行 lint 检查：`ruff check src/ui/widgets/`
4. 运行类型检查（如有）：`mypy src/`
5. 手动测试：启动应用，验证光标移动、标签定位、ViewBox 交互等功能正常
6. 确认无方法重复后，可清理 `plot_widget.py` 中完全被委托的方法实现（在后续迭代中逐步移除）

***

## 三、文件变更清单

| 文件                       | 变更类型  | 说明                                            |
| ------------------------ | ----- | --------------------------------------------- |
| `plot_widget.py`         | 大量修改  | 方法体改为委托，调整初始化顺序                               |
| `plot_ui_manager.py`     | 少量修改  | `_setup_interaction()` 移除重复的光标元素创建代码          |
| `event_handler.py`       | 少量修改  | 确保 `_refresh_cursor_geometry` 与 widget 版本逻辑一致 |
| `cursor_manager.py`      | 中等修改  | `_position_labels_avoid_overlap` 替换为高级版       |
| `axis_manager.py`        | 可能无修改 | 方法已就绪，只需建立委托                                  |
| `plot_data_manager.py`   | 可能无修改 | 同上                                            |
| `multi_curve_manager.py` | 可能无修改 | 同上                                            |
| `mark_region_manager.py` | 可能无修改 | 同上                                            |
| `__init__.py`            | 无修改   | 导出保持不变                                        |

***

## 四、假设与决策

1. **`DraggableGraphicsLayoutWidget`** **保留为统一入口**：外部代码仍通过 `widget.method()` 调用，方法内部委托到对应 Manager。不引入 `__getattr__` 自动代理，保持调用链显式可追踪。

2. **光标基础设施（vline、vline2、cursor\_label、\_cursor\_item\_pool）保留为 widget 属性**：虽然 CursorManager 通过 `self.pw` 访问它们，但它们是 pyqtgraph 的图形元素，生命周期由 scene 管理，放在 widget 上更自然。

3. **`on_vline_position_changed()`** **保留在 widget**：此方法涉及跨 widget 同步逻辑（`self.window().plot_widgets`），不适合完全下放。

4. **`cursor_manager.py`** **的高级版标签定位算法来自** **`plot_widget.py`** **L1811-1970**：它比 cursor\_manager 当前版本多了动态字体尺寸计算、三级边缘避让和场景坐标钳制。

5. **重构后不改变任何外部 API**：`DraggableGraphicsLayoutWidget` 的公开方法签名不变。

***

## 五、验证清单

* [ ] `ruff check src/ui/widgets/` 无新增错误

* [ ] 应用启动正常，无崩溃

* [ ] 光标移动时 y 值标签位置正确

* [ ] 光标靠近视图边缘时标签不会超出边界

* [ ] ViewBox 右键菜单各项功能正常

* [ ] 多曲线模式下光标标签不重叠

* [ ] 框选 (rubberBand) 功能正常

* [ ] 拖拽重排功能正常

* [ ] 单曲线/多曲线切换正常

***

## 六、代码审查报告（v2.0 新增）

> 以下为基于实际代码逐一比对后得出的审查结论，已在上方步骤中内联修正了明确的错误（如 `_set_vline_bounds` 委托目标）。本节保留完整的审查分析供实施时参考。

### 6.1 总体评价

| 维度 | 评分 | 说明 |
|---|---|---|
| 目标明确性 | **8/10** | 目标清晰：激活死代码 Manager，消除重复 |
| 技术方案合理性 | **7/10** | 委托模式可行，但有信号连接顺序和状态管理风险 |
| 步骤清晰度 | **7/10** | 步骤划分合理，但步骤 1 和 2 有隐式依赖未标注 |
| 完整性 | **5/10** | 遗漏约 25 个方法的委托方案，跨 Manager 重复未处理 |
| 风险评估 | **6/10** | 提到了部分风险，但遗漏了双重连接、状态冲突等问题 |
| 收益预期 | **7/10** | 能显著减少 plot_widget.py 体量，但委托层增加间接性 |

***

### 6.2 重大问题

#### 问题 1：约 25 个 widget 方法未被任何 Manager 覆盖（最大遗漏）

`plot_widget.py` 共 100 个方法，计划只覆盖了约 60 个。以下方法在 Manager 中有对应实现但计划未列出委托，或在任何 Manager 中都没有对应实现：

##### 1.1 Manager 有对应实现但计划遗漏委托

**AxisManager 遗漏：**

| widget 方法 | 行号 | AxisManager 行号 |
|---|---|---|
| `_get_min_x_range_value` | 509 | L236 |
| `_set_min_x_range` | 536 | L267 |
| `set_xrange_with_link_handling` | 1092 | L192 |

**PlotDataManager 遗漏：**

| widget 方法 | 行号 | PlotDataManager 行号 |
|---|---|---|
| `_validate_plot_data` | 2568 | L151 |
| `_prepare_plot_data` | 2604 | L188 |
| `_compute_valid_min_max` | 2806 | L259 |
| `_get_y_range_in_x_window` | 2832 | L287 |
| `_safe_clear_plot_items` | 1567 | L550 |
| `_clear_plot_data` | 2915 | L584 |
| `reset_plot` | 594 | L634 |

**MultiCurveManager 遗漏：**

| widget 方法 | 行号 | MultiCurveManager 行号 |
|---|---|---|
| `_recreate_curve` | 3329 | L93 |
| `_collect_visible_curve_arrays` | 1500 | L124 |
| `_collect_visible_curve_pairs` | 1516 | L135 |
| `_update_axes_for_multi_curve` | 3473 | L166 |

##### 1.2 任何 Manager 都没有对应实现的方法

**数据操作核心（~350 行）：**

| widget 方法 | 行号 | 行数 | 说明 |
|---|---|---|---|
| `add_variable_to_plot` | 2978 | **250** | 多曲线添加核心，单曲线→多曲线迁移 |
| `add_variables_to_plot` | 2466 | **101** | 批量添加入口 |

**拖拽指示器（~100 行）：**

| widget 方法 | 行号 | 行数 |
|---|---|---|
| `_should_hide_drag_indicator` | 791 | 34 |
| `_enforce_drag_indicator_visibility` | 826 | 25 |
| `_notify_drag_indicator` | 852 | 25 |
| `_extract_var_names_from_text` | 779 | 11 |

**光标边界更新：**

| widget 方法 | 行号 | 行数 | 说明 |
|---|---|---|---|
| `_update_vline_bounds_from_data` | 2176 | 58 | PlotDataManager L692 有此方法，但 CursorManager 中没有 |

**轴更新：**

| widget 方法 | 行号 | 行数 |
|---|---|---|
| `_update_x_limits_for_plot` | 3552 | 24 |
| `update_plot_style` | 3954 | 37 |

**Qt 事件处理器（必须保留在 widget 中）：**

`wheelEvent`(915)、`dragEnterEvent`(2436)、`dragMoveEvent`(2445)、`dragLeaveEvent`(2454)、`dropEvent`(2458)、`mouseDoubleClickEvent`(3578)、`mousePressEvent`(3631)、`mouseMoveEvent`(3640)、`mouseReleaseEvent`(3647)、`resizeEvent`(984)

**格式化工具函数（可提取为模块级函数）：**

`sInt_to_fmtStr`(1049)、`dateInt_to_fmtStr`(1058)、`_significant_decimal_format_str`(1068)

**建议处理方案：**
- 6.2.1.1 中 Manager 有对应实现的方法 → 补充到各步骤的委托列表中
- `add_variable_to_plot`(250行) 和 `add_variables_to_plot`(101行) → 保留在 widget 中（需单独重构计划）
- 拖拽指示器方法 → 保留在 widget 中（涉及跨组件交互）
- Qt 事件处理器 → 必须保留在 widget 中（Qt 框架要求）
- 格式化函数 → 可提取为模块级纯函数

***

#### 问题 2：`_set_vline_bounds` 委托目标错误 ✅ 已修正

~~计划步骤 4 声称委托到 CursorManager，但 CursorManager 中没有此方法。实际存在于 AxisManager L376。~~

已在上方步骤 4 和步骤 5 中内联修正。

***

#### 问题 3：`_is_interacting` 状态存在双重管理冲突

`EventHandler`（L64-71）和 `CursorManager`（L41-47）都定义了 `_is_interacting` 的 property getter/setter，均代理到 `self.pw._is_interacting`。

**风险：** 当前不会冲突（都代理到 widget），但如果将来某个 Manager 改为存储在自己实例上会导致状态不同步。

**建议：** 明确 `_is_interacting` 的单一所有者为 widget，所有 Manager 通过 `self.pw._is_interacting` 访问。

***

#### 问题 4：步骤 1 和步骤 2 有隐式依赖，必须同步实施

`PlotUIManager._setup_plot_area()` (L175) 内部调用了 `pw._connect_viewbox_signals()`（L182），该方法将 ViewBox 信号连接到 widget 自身的 `_on_vb_*` 方法。步骤 2 要将这些连接改为 EventHandler。

如果仅实施步骤 1（委托 `setup_ui` 给 PlotUIManager）而未同步实施步骤 2（改信号连接目标），`PlotUIManager._setup_plot_area()` 会调用 widget 的 `_connect_viewbox_signals()` 连接到 widget 方法——功能正常但与步骤 2 的目标矛盾。

**建议：** 将步骤 1 和步骤 2 合并为一个原子操作，或在计划中明确标注"步骤 1 和 2 必须同时实施"。

***

#### 问题 5：PlotDataManager 中的死代码委托方法

`plot_data_manager.py` L670-692 包含 6 个纯委托方法（转发到 AxisManager）：
- `_recalc_max_point_density` → `self._axis_manager._recalc_max_point_density()`
- `_get_safe_x_range` → `self._axis_manager._get_safe_x_range()`
- `_set_safe_y_range` → `self._axis_manager._set_safe_y_range()`
- `_set_x_limits_with_min_range` → `self._axis_manager._set_x_limits_with_min_range()`
- `_set_vline_bounds` → `self._axis_manager._set_vline_bounds()`
- `_update_vline_bounds_from_data` → `self._axis_manager._update_vline_bounds_from_data()`

激活后会产生混乱的双层间接路径：widget → PlotDataManager → AxisManager，而非 widget → AxisManager。

**建议：** widget 直接委托到 AxisManager，PlotDataManager 中的这些委托方法标记为废弃或删除。

***

#### 问题 6：EventHandler `_on_vb_*` 方法签名与 widget 不一致

| 方法 | widget 签名 | EventHandler 签名 |
|---|---|---|
| `_on_vb_jump` | `(self)` | `(self, pw, ctx_x)` |
| `_on_vb_clear` | `(self)` | `(self, pw)` |
| `_on_vb_auto_y` | `(self)` | `(self, pw)` |
| `_on_vb_set_cursor_mode` | `(self, mode)` | `(self, mode, pw, ctx_x)` |

EventHandler 版本需要显式传入 `pw` 参数，不能简单委托。但 EventHandler 有自己的 `_connect_viewbox_signals()`（L224-237），用 lambda 传递 `pw`，可直接建立信号连接。

**建议：** 直接使用 `EventHandler._connect_viewbox_signals()` 建立信号连接，替代 widget 中的 `_connect_viewbox_signals()`，无需经过 widget 中转。

***

### 6.3 跨 Manager 方法重复

以下方法在 **多个 Manager** 中都有实现，计划未说明清理策略：

| 方法 | CursorManager | EventHandler | 建议归属 |
|---|---|---|---|
| `_start_interaction` | L979 | L102 | EventHandler |
| `_end_interaction` | L993 | L113 | EventHandler |
| `_schedule_cursor_geometry_update` | L1003 | L124 | EventHandler |
| `_refresh_cursor_geometry` | L1012 | L136 | EventHandler |
| `_on_vb_set_cursor_mode` | L1093 | L165 | EventHandler |
| `_on_vb_show_cursor` | L1101 | L170 | EventHandler |
| `_on_vb_hide_cursor` | L1107 | L178 | EventHandler |

**建议：** CursorManager 中的这 7 个方法应标记为废弃，统一由 EventHandler 管理。

***

### 6.4 架构兼容性评估

#### 责任链引用一致性

当前链式结构：
```
EventHandler → MarkRegionManager → CursorManager → MultiCurveManager → PlotDataManager → AxisManager → PlotUIManager → widget
```

**问题：** 只有 `PlotUIManager` 继承 `BasePlotManager` 使用 `weakref`，其他 Manager 通过强引用链式访问。如果 widget 被销毁，仅 `PlotUIManager.pw` 安全返回 None，其他 Manager 的 `.pw` 可能访问到已被 Qt 销毁的对象。

#### 委托模式可行性

委托模式整体可行。Manager 方法中通过 `self.pw` 访问 widget，委托后调用路径不变。但增加了调试时的调用栈深度（widget → Manager → pw.xxx → 另一个 Manager）。

***

### 6.5 `_position_labels_avoid_overlap` 替换注意事项

计划步骤 4.3 提到将 widget 高级版替换 cursor_manager 简化版，但两个版本的 API 有差异：

| 对比维度 | widget 版本 | cursor_manager 版本 |
|---|---|---|
| 字体度量 | `QFontMetrics` 动态计算标签像素宽高 | **无**，不考虑标签尺寸 |
| 定位策略 | 基于场景坐标的精确偏移，4 方向候选 | 固定像素偏移，视图角落定位 |
| 边界检查 | 检查标签四边是否在视图内 | 仅检查候选点坐标 |
| 边缘避让 | 完整的三级边缘避让 + soft margin | **无** |
| 超出兜底 | 强制 clamp 到边界内 | 直接使用第一个候选位置 |
| addItem 方式 | `plot_item.addItem(text_item, ignoreBounds=True)` | `scene.addItem(label)` |

替换时需确保：
1. CursorManager 能通过 `self.pw.plot_item` 访问 plot_item
2. 字体度量需要 `QFontMetrics` 实例
3. 替换后 `_show_x_position_only` 等调用方兼容

***

### 6.6 建议的步骤执行顺序

原计划按 Manager 编号顺序执行。建议按风险从低到高调整：

1. **步骤 5**（AxisManager）——最独立，风险最低，适合试点验证委托模式
2. **步骤 4**（CursorManager）——最大重复块（~1100 行），收益最高
3. **步骤 1 + 2 合并**（PlotUIManager + EventHandler）——必须同步实施
4. **步骤 3**（PlotUIManager 刷新协调）
5. **步骤 6**（PlotDataManager + MultiCurveManager）——需补充遗漏方法
6. **步骤 7**（MarkRegionManager）
7. **步骤 8**（清理验证 + 删除跨 Manager 重复方法）

***

### 6.7 补充验证清单

在原第五节验证清单基础上，增加：

* [ ] 确认 CursorManager 中 7 个与 EventHandler 重复的方法已标记废弃
* [ ] 确认 PlotDataManager L670-692 的 6 个委托方法已标记废弃
* [ ] 确认 `_set_vline_bounds` 委托到 AxisManager 而非 CursorManager
* [ ] 确认 `_is_interacting` 状态仅由 widget 持有，Manager 通过 `self.pw` 访问
* [ ] 确认步骤 1 和 2 同步实施后无信号双重连接
* [ ] 确认约 25 个遗漏方法已按处理方案分类（委托/保留/废弃）

---

## 七、执行记录

### 7.1 第一阶段：激活 AxisManager（步骤 5）

**执行时间**: 2026-06-15
**状态**: ✅ 完成，用户测试通过

#### 变更文件

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `plot_widget.py` | 修改 | 13 个方法改为委托调用，新增 `self.plot_context = None` 初始化 |
| `axis_manager.py` | 无修改 | 方法已就绪 |

#### 委托方法清单（13 个）

| widget 方法 | 委托目标 | 减少行数 |
|-------------|----------|----------|
| `update_x_axis_label` | `AxisManager.update_x_axis_label` | -7 |
| `auto_range` | `AxisManager.auto_range` | -114 |
| `auto_y_in_x_range` | `AxisManager.auto_y_in_x_range` | -2 |
| `_get_safe_x_range` | `AxisManager._get_safe_x_range` | -7 |
| `_get_min_x_range_value` | `AxisManager._get_min_x_range_value` | -17 |
| `_set_x_limits_with_min_range` | `AxisManager._set_x_limits_with_min_range` | -4 |
| `_set_min_x_range` | `AxisManager._set_min_x_range` | -1 |
| `_recalc_max_point_density` | `AxisManager._recalc_max_point_density` | -11 |
| `_set_safe_y_range` | `AxisManager._set_safe_y_range` | -38 |
| `set_xrange_with_link_handling` | `AxisManager.set_xrange_with_link_handling` | -13 |
| `_setup_plot_axes` | `AxisManager._setup_plot_axes` | -36 |
| `_reset_plot_limits` | `AxisManager._reset_plot_limits` | -6 |
| `_set_vline_bounds` | `AxisManager._set_vline_bounds` | -3 |

**总计减少约 260 行实现代码**

#### 遇到的问题及修复

1. **`AttributeError: no attribute '_axis_manager'`**
   - **原因**: `setup_plot_area()` 内部调用 `update_x_axis_label()`（L285），此时 `_init_manager_chain()` 尚未执行（L160），`_axis_manager` 不存在
   - **修复**: `update_x_axis_label()` 添加 `hasattr(self, '_axis_manager')` 安全检查，在 Manager 链初始化前使用内联实现作为 fallback

2. **`AttributeError: property 'plot_context' has no setter`**
   - **原因**: 初始方案将 `plot_context` 设为只读 property（返回 `self.window()`），但 `layout_manager.py` L488 需要给 `plot_widget.plot_context` 赋值 `PlotContext` 对象
   - **修复**: 移除只读 property，改为在 `__init__` 中 `self.plot_context = None`，由 `layout_manager` 在后续流程中赋值为 `PlotContext` 实例

#### 备份位置

`.trae/backup/v2.0-pre-refactor/`（包含全部 9 个 widget 相关文件的原始副本）

#### 验证结果

- [x] IDE 诊断零错误
- [x] 语法检查通过（Python ast.parse）
- [x] 用户功能测试通过：加载数据、自动缩放、光标交互均正常

### 7.2 第二阶段：激活 CursorManager（步骤 4）

**执行时间**: 2026-06-15
**状态**: ✅ 完成，待用户测试验证

#### 变更文件

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `plot_widget.py` | 修改 | 22 个方法改为委托调用 |
| `cursor_manager.py` | 修改 | 替换 `_position_labels_avoid_overlap` 为高级版；标记 7 个废弃方法 |

#### 委托方法清单（22 个）

| widget 方法 | 委托目标 | 减少行数 |
|-------------|----------|----------|
| `_is_cursor_update_locked` | `CursorManager._is_cursor_update_locked` | -21 |
| `_get_cursor_mode` | `CursorManager._get_cursor_mode` | -4 |
| `_get_cursor_x_positions` | `CursorManager._get_cursor_x_positions` | -18 |
| `_set_vline_visibility_for_mode` | `CursorManager._set_vline_visibility_for_mode` | -10 |
| `apply_cursor_mode` | `CursorManager.apply_cursor_mode` | -71 |
| `update_cursor_label` | `CursorManager.update_cursor_label` | -36 |
| `_update_single_curve_cursor_label` | `CursorManager._update_single_curve_cursor_label` | -33 |
| `_get_circle_from_pool` | `CursorManager._get_circle_from_pool` | -25 |
| `_get_label_from_pool` | `CursorManager._get_label_from_pool` | -31 |
| `_get_x_label_from_pool` | `CursorManager._get_x_label_from_pool` | -18 |
| `_clear_cursor_items` | `CursorManager._clear_cursor_items` | -75 |
| `_queue_item_for_deletion` | `CursorManager._queue_item_for_deletion` | -7 |
| `_process_pending_deletes` | `CursorManager._process_pending_deletes` | -32 |
| `_update_multi_curve_cursor_label` | `CursorManager._update_multi_curve_cursor_label` | -179 |
| `_position_labels_avoid_overlap` | `CursorManager._position_labels_avoid_overlap` | -159 |
| `toggle_cursor` | `CursorManager.toggle_cursor` | -24 |
| `_show_x_position_only` | `CursorManager._show_x_position_only` | -39 |
| `_has_visible_curve_data` | `CursorManager._has_visible_curve_data` | -16 |
| `pin_cursor` | `CursorManager.pin_cursor` | -73 |
| `free_cursor` | `CursorManager.free_cursor` | -27 |
| `reset_pin_state` | `CursorManager.reset_pin_state` | -14 |
| `_update_cursor_after_plot` | `CursorManager._update_cursor_after_plot` | -26 |

**总计减少约 938 行实现代码**

#### CursorManager 修改

1. **`_position_labels_avoid_overlap` 替换为高级版**
   - 新增 `from PySide6.QtGui import QFontMetrics` 和 `from PySide6.QtCore import QPointF`
   - 使用 QFontMetrics 动态计算标签宽度（替代固定宽度）
   - 4 候选位置策略（右上、左上、右下、左下）在场景坐标系中计算
   - 三级边缘避让逻辑（严格边缘、软边缘、中心区域）
   - 所有坐标转换使用 `mapViewToScene` 和 `mapSceneToView`

2. **7 个方法标记为 `[DEPRECATED]`**
   - `_start_interaction` — 已由 EventHandler 接管
   - `_end_interaction` — 已由 EventHandler 接管
   - `_schedule_cursor_geometry_update` — 已由 EventHandler 接管
   - `_refresh_cursor_geometry` — 已由 EventHandler 接管
   - `_on_vb_set_cursor_mode` — 已由 EventHandler 接管
   - `_on_vb_show_cursor` — 已由 EventHandler 接管
   - `_on_vb_hide_cursor` — 已由 EventHandler 接管

#### 保留在 widget 中的方法

- `on_vline_position_changed` — 跨 widget 同步逻辑（涉及 `self.window().plot_widgets` 循环），不适合下放
- `_set_vline_bounds` — 已在第一阶段委托到 AxisManager

#### 遇到的问题及修复

1. **`ModuleNotFoundError: No module named 'PyQt6'`**
   - **原因**: 添加的 imports 使用了 `PyQt6`，但项目已迁移到 `PySide6`
   - **修复**: 将 `from PyQt6.QtGui import QFontMetrics` 和 `from PyQt6.QtCore import QPointF` 改为 `PySide6`

2. **重载数据后 cursor 状态丢失（v5.0 修复）**
   - **现象**: 重载数据后 vline 同步失效、y 值文本框消失、右侧光标"消失"
   - **根因**: `PlotContext` 缺少 `_data_version` 属性，导致 `_is_cursor_update_locked()` 中的版本检查永远返回 True（`current_version` 始终为 0，而 `my_version` 在 reload 后递增）
   - **影响**: `update_cursor_label` 直接 return，标签永远不会被创建
   - **修复**: 在 `plot_context.py` 中添加 `_data_version` property，委托到 `mw._data_version`

3. **`toggle_cursor` 与原始实现不一致（v5.0 修复）**
   - **现象**: 通过顶部按钮关闭 cursor 后再打开，所有文本框消失、vline 不同步
   - **根因**: 重构后的 `toggle_cursor(show=False)` 没有清除 pinned 状态，而原始实现会清除
   - **修复**: 恢复原始逻辑，关闭 cursor 时清除 `is_cursor_pinned`、`pinned_x_values` 等状态

4. **`_end_data_reload` 恢复逻辑不完整（v5.0 修复）**
   - **根因**: 
     - 调用 `apply_cursor_mode` 只设置了 CursorManager 的属性，没有设置 `view_box.is_cursor_pinned`
     - 没有恢复 vline/vline2 的可见性
   - **修复**: 
     - 直接设置 widget 属性（`is_cursor_pinned`、`pinned_x_values` 等）
     - 设置 `view_box.is_cursor_pinned = True`
     - 按模式恢复 vline/vline2 可见性

#### 验证结果

- [x] IDE 诊断零错误
- [x] 语法检查通过（Python ast.parse）
- [x] 用户功能测试：光标移动、标签定位、固定/释放、模式切换、对象池复用
- [x] 重载数据后 cursor 状态正确恢复（v5.0）
- [x] 通过顶部按钮关闭/打开 cursor 功能正常（v5.0）

#### 变更文件汇总（v5.0 修复）

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `plot_context.py` | 新增 | 添加 `_data_version` property |
| `cursor_manager.py` | 修改 | 恢复 `toggle_cursor` 原始逻辑（关闭时清除 pinned 状态） |
| `file_loader_manager.py` | 修改 | 完善 `_end_data_reload` 恢复逻辑（view_box.is_cursor_pinned、vline 可见性） |

---

## 八、代码一致性验证规范（v5.0 新增）

### 8.1 背景

第二阶段重构（CursorManager 激活）过程中，由于 **未严格验证当前代码与备份 `plot_widget.py` 的实现一致性**，引入了多个 bug：

1. **`toggle_cursor` 行为不一致**: 重构后的实现省略了 pinned 状态清除逻辑
2. **`_end_data_reload` 恢复逻辑不完整**: 没有覆盖所有需要恢复的状态（`view_box.is_cursor_pinned`、vline 可见性）
3. **`PlotContext` 属性遗漏**: `_data_version` 属性未委托到 `MainWindow`

这些问题导致用户测试时发现光标重载数据后功能异常，经过多轮排查才定位根因。

### 8.2 重构时必须执行的验证步骤

**在激活任何 Manager 方法之前，必须执行以下验证：**

#### 步骤 1：逐行对比当前实现与备份

```bash
# 对比当前 cursor_manager.py 与备份 plot_widget.py 中的对应方法
diff -u \
  .trae/backup/v2.0-pre-refactor/plot_widget.py \
  src/ui/widgets/cursor_manager.py \
  | grep -A5 -B5 "def method_name"
```

**检查清单：**
- [ ] 方法签名完全一致（参数名、默认值）
- [ ] 方法体逻辑完全一致（特别是状态清除、属性设置）
- [ ] 所有被调用的属性和方法在目标上下文中可用

#### 步骤 2：验证状态管理完整性

对于涉及状态保存/恢复的方法（如 `_begin_data_reload` / `_end_data_reload`），必须验证：

```python
# 伪代码示例
saved_state = {
    'is_cursor_pinned': widget.is_cursor_pinned,
    'pinned_x_values': widget.pinned_x_values,
    'view_box.is_cursor_pinned': widget.view_box.is_cursor_pinned,  # ← 容易遗漏
    'vline.isVisible()': widget.vline.isVisible(),                   # ← 容易遗漏
    'vline2.isVisible()': widget.vline2.isVisible(),                 # ← 容易遗漏
    # ...
}
```

**验证方法：**
1. 在保存状态后，打印所有保存的属性值
2. 在恢复状态后，打印所有恢复的属性值
3. 对比两者是否一致

#### 步骤 3：验证 PlotContext 属性委托

任何在 Manager 中通过 `self.pw.plot_context.xxx` 访问的属性，必须在 `PlotContext` 类中有对应的 property 定义：

```python
# 检查清单
required_properties = [
    '_data_version',
    '_is_loading_new_data',
    '_is_time_correction_active',
    'cursor_mode',
    'pinned_x_values',
    # ... 根据实际使用情况补充
]

for prop in required_properties:
    assert hasattr(PlotContext, prop), f"PlotContext missing property: {prop}"
```

#### 步骤 4：添加临时 debug 日志

在重构的关键路径上添加 `logger.info()` 级别的日志（不是 `debug`），便于排查：

```python
logger.info("[CursorManager] toggle_cursor: show=%s, mode=%s, pinned_x_values=%s", 
            show, mode, self.pinned_x_values)
```

**验证完成后移除这些日志。**

### 8.3 常见陷阱

| 陷阱 | 说明 | 预防措施 |
|------|------|----------|
| **省略状态清除** | 重构时认为"这个状态不重要"而省略 | 逐行对比，不省略任何逻辑 |
| **遗漏 view_box 状态** | `view_box.is_cursor_pinned` 与 `widget.is_cursor_pinned` 是两个独立属性 | 显式列出所有需要恢复的属性 |
| **遗漏 vline 可见性** | 恢复 cursor 状态时忘记设置 `vline.setVisible(True)` | 在恢复逻辑中显式调用 `_set_vline_visibility_for_mode` |
| **PlotContext 属性缺失** | Manager 通过 `plot_context.xxx` 访问的属性未在 PlotContext 中定义 | 重构前检查所有 `plot_context.xxx` 访问，确保属性存在 |
| **property 机制绕过** | 直接设置 `self.pw.show_values_only = True` 而不是 `self.show_values_only = True` | 优先使用 property 机制，保持封装性 |

### 8.4 后续阶段验证模板

在执行第三阶段（PlotDataManager + MultiCurveManager）时，使用以下模板：

```markdown
## 第 X 阶段验证记录

### 方法对比检查
- [ ] `method_name`: 当前实现 vs 备份实现 — 一致性 ✓/✗
- [ ] ...

### 状态管理检查
- [ ] 保存的状态属性列表：[...]
- [ ] 恢复的状态属性列表：[...]
- [ ] 保存/恢复一致性验证：✓/✗

### PlotContext 属性检查
- [ ] 新增的属性：[...]
- [ ] 属性委托验证：✓/✗

### Debug 日志
- [ ] 添加的临时日志：[...]
- [ ] 验证完成后已移除：✓/✗

### 用户测试场景
- [ ] 场景 1: 描述 — 结果 ✓/✗
- [ ] 场景 2: 描述 — 结果 ✓/✗
```

## 第九节：v6.0 执行记录（Step 6: PlotDataManager + MultiCurveManager）

**执行日期**: 2026-06-15

### MultiCurveManager 更新

更新了以下方法以匹配 widget 的现有行为：

| 方法 | 变更说明 |
|---|---|
| `update_multi_curve_mode()` | 改用 header-based legend（调用 `update_legend()` / `update_left_header()`），替代 pyqtgraph 内置 legend |
| `update_legend()` | 改用 HTML 文本显示在 header 中（与 widget 现有行为一致），替代 pyqtgraph `legend.addItem()` |
| `toggle_curve_visibility_by_name()` | 增加 `update_legend()`、`_update_axes_for_multi_curve()`、`update_cursor_label()` 调用 |
| `_recreate_curve()` | 改用 `pw.add_variable_to_plot()` 重建曲线，替代直接创建 `PlotDataItem` |
| `_collect_visible_curve_arrays()` | 改用 `np.asarray()` + 空数组检查，增加 `getattr` 守卫 |
| `_collect_visible_curve_pairs()` | 同上 |
| `get_curve_x_limits()` | 增加单曲线模式处理（`pw.curve` / `pw.original_index_x`） |
| `_update_axes_for_multi_curve()` | 完整重写，匹配 widget 的 y 轴范围计算逻辑（含 `_set_safe_y_range` 两步法） |
| `_on_legend_clicked()` | 改用 QTextDocument hitTest 精确定位（匹配 widget 实现） |

### PlotDataManager 更新

| 方法 | 变更说明 |
|---|---|
| `clear_plot_item()` | 增加 `_axis_manager._reset_plot_limits()` 调用 |
| `_clear_plot_data()` | 增加 `_recalc_max_point_density()` 和 `_sync_min_xrange()` 调用 |
| `handle_single_point_limits()` | 重写以匹配 widget 的 `_get_safe_x_range` 实现 |
| `plot_variable()` | `_sync_min_xrange` 调用改为 `pw.window()` 方式，增加 `_is_updating_data` 守卫 |

### widget 方法委托（共 23 个方法）

**MultiCurveManager** (9 个):
`update_multi_curve_mode`, `update_legend`, `toggle_curve_visibility_by_name`, `_recreate_curve`, `_collect_visible_curve_arrays`, `_collect_visible_curve_pairs`, `get_curve_x_limits`, `_update_axes_for_multi_curve`, `_on_legend_clicked`

**PlotDataManager** (14 个):
`plot_variable`, `clear_plot_item`, `get_value_from_name`, `handle_single_point_limits`, `update_time_correction`, `datetime_to_unix_seconds`, `clear_value_cache`, `_validate_plot_data`, `_prepare_plot_data`, `_compute_valid_min_max`, `_get_y_range_in_x_window`, `_safe_clear_plot_items`, `_clear_plot_data`, `reset_plot`

### 代码规模变化

- `plot_widget.py`: ~4100+ → 2061 行（减少 ~2000 行）
- `multi_curve_manager.py`: 210 → 361 行（增加实现）
- `plot_data_manager.py`: 709 → 728 行（增加 `_sync_min_xrange` / `_recalc_max_point_density` 调用）

### Manager 激活状态

| Manager | 状态 |
|---|---|
| AxisManager | ✅ 已激活 (v3.0) |
| CursorManager | ✅ 已激活 (v4.0) |
| PlotDataManager | ✅ 已激活 (v6.0) |
| MultiCurveManager | ✅ 已激活 (v6.0) |
| MarkRegionManager | ✅ 已激活 (v7.1) |
| EventHandler | ✅ 已激活 (v8.0) |
| PlotUIManager | ✅ 已激活 (v8.0) |

**全部 7/7 Manager 已激活！**

### 下一步

- 用户功能测试验证
- 清理 plot_widget.py 中不再需要的 import

## 第十节：v7.0 代码审查与修复记录

**执行日期**: 2026-06-15

### 对比审查

与备份文件 `.trae/backup/v2.0-pre-refactor/plot_widget.py` 逐方法对比：

| 模块 | 方法数 | 一致 | 不一致 |
|---|---|---|---|
| MultiCurveManager | 10 | 9 | 1 |
| PlotDataManager | 14 | 14 | 0 |

### 发现的 Bug：`_on_legend_clicked` 缺少 fallback

[`multi_curve_manager.py`](file:///Users/melon/csv_plot/src/ui/widgets/multi_curve_manager.py#L283) 的 QTextDocument hitTest 在点击分隔符 " | " 区域或边缘时，备份中有三段 fallback 逻辑确保总能选中最近的曲线：

1. `hit_pos < 0` → 根据像素位置判断左右两侧
2. `hit_pos >= 0` 但未精确匹配 → 计算 `item_ranges` 最近距离
3. `max(0, min(index, len-1))` → 安全钳制

v6.0 实现缺少此 fallback，导致边缘点击无响应。v7.0 已修复。

### 用户验证通过

- 单曲线/多曲线绘制正常
- 图例点击切换曲线可见性正常
- 清除绘图正常
- 时间修正正常
- 光标交互正常

## 第十一节：v7.1 执行记录（Step 7: MarkRegionManager）

**执行日期**: 2026-06-15
**Commit**: `efb16fa`

### 委托方法清单（4 个）

| widget 方法 | 委托目标 | 减少行数 |
|-------------|----------|----------|
| `add_mark_region` | `MarkRegionManager.add_mark_region` | -5 |
| `remove_mark_region` | `MarkRegionManager.remove_mark_region` | -3 |
| `update_mark_region` | `MarkRegionManager.update_mark_region` | -3 |
| `get_mark_stats` | `MarkRegionManager.get_mark_stats` | -142 |

**总计减少约 153 行实现代码**

### MarkRegionManager 重写要点

1. **使用 `MarkStatEntry` dataclass** (`src.core.data_types`) 替代原 `MarkStats` NamedTuple，与主窗口 `mark_stats.py` 保持一致
2. **`_evaluate_float32_safety`** 用于 dtype 转换判断，替代原有的静态 `float64` 转换
3. **signal 连接**: `add_mark_region` 中 `self.pw.window().sync_mark_regions` 无条件连接（去除 `if self.pw.plot_context:` 守卫）
4. **`np.mean()` 不加 `float()` 包裹**，与备份行为一致

### 备份对比验证

与 `.trae/backup/v2.0-pre-refactor/plot_widget.py` 逐方法对比：

| 方法 | 对比结果 | 说明 |
|------|----------|------|
| `add_mark_region` | ✅ 一致 | `self.` → `self.pw.`, `self.window()` → `self.pw.window()` |
| `remove_mark_region` | ✅ 一致 | `self.` → `self.pw.` |
| `update_mark_region` | ✅ 一致 | `self.` → `self.pw.` |
| `get_mark_stats` | ✅ 一致 | `self.` → `self.pw.`, `MarkStatEntry` / `_evaluate_float32_safety` |

### 语法验证

- `mark_region_manager.py` — `py_compile` ✅
- `plot_widget.py` — `py_compile` ✅

### Manager 激活状态

| Manager | 状态 |
|---|---|
| AxisManager | ✅ 已激活 (v3.0) |
| CursorManager | ✅ 已激活 (v4.0) |
| PlotDataManager | ✅ 已激活 (v6.0) |
| MultiCurveManager | ✅ 已激活 (v6.0) |
| MarkRegionManager | ✅ 已激活 (v7.1) |
| EventHandler | ⬜ 待激活 |
| PlotUIManager | ⬜ 待激活 |

**当前: 5/7 已激活**

### 代码规模变化

- `plot_widget.py`: 2061 → 1916 行（减少 145 行）
- `mark_region_manager.py`: 使用 `MarkStatEntry` + `_evaluate_float32_safety`，226 行

### 下一步

- Step 1-3: 修复初始化管线 + 激活 EventHandler + PlotUIManager（最后 2 个 Manager）

## 第十二节：v8.0 执行记录（Steps 1-3: EventHandler + PlotUIManager + 初始化管线）

**执行日期**: 2026-06-15
**Commit**: `876f1cc`

### 初始化管线修复

1. **`_init_manager_chain()` 移至 `__init__`**: 在 `super().__init__()` 之后、`self.setup_ui()` 之前调用，确保所有 Manager 在 UI 初始化前创建
2. **`setup_ui()` 委托到 PlotUIManager**: 替换 ~60 行实现体为单行委托调用

### 委托方法清单（27 个）

**PlotUIManager**（10 个）

| widget 方法 | 委托目标 |
|---|---|
| `setup_ui` | `PlotUIManager.setup_ui` |
| `setup_header` | `PlotUIManager._setup_header` |
| `setup_plot_area` | `PlotUIManager._setup_plot_area` |
| `setup_axes` | `PlotUIManager._setup_axes` |
| `setup_interaction` | `PlotUIManager._setup_interaction` |
| `_init_ui_refresh_coordinator` | `PlotUIManager._init_ui_refresh_coordinator` |
| `_queue_ui_refresh` | `PlotUIManager._queue_ui_refresh` |
| `_cancel_ui_refresh` | `PlotUIManager._cancel_ui_refresh` |
| `_run_style_refresh` | `PlotUIManager._run_style_refresh` |
| `_run_cursor_refresh` | `PlotUIManager._run_cursor_refresh` |
| `_run_stats_refresh` | `PlotUIManager._run_stats_refresh` |

**EventHandler**（17 个）

| widget 方法 | 委托目标 |
|---|---|
| `_on_range_changed` | `EventHandler._on_range_changed` |
| `_start_interaction` | `EventHandler._start_interaction` |
| `_end_interaction` | `EventHandler._end_interaction` |
| `_schedule_cursor_geometry_update` | `EventHandler._schedule_cursor_geometry_update` |
| `_refresh_cursor_geometry` | `EventHandler._refresh_cursor_geometry` |
| `_connect_viewbox_signals` | `EventHandler._connect_viewbox_signals` |
| `_on_vb_jump` | `EventHandler._on_vb_jump` |
| `_on_vb_clear` | `EventHandler._on_vb_clear` |
| `_on_vb_auto_y` | `EventHandler._on_vb_auto_y` |
| `_on_vb_set_cursor_mode` | `EventHandler._on_vb_set_cursor_mode` |
| `_on_vb_show_cursor` | `EventHandler._on_vb_show_cursor` |
| `_on_vb_hide_cursor` | `EventHandler._on_vb_hide_cursor` |
| `_on_vb_set_row_height` | `EventHandler._on_vb_set_row_height` |
| `_on_vb_set_all_row_height` | `EventHandler._on_vb_set_all_row_height` |
| `_on_vb_copy_name` | `EventHandler._on_vb_copy_name` |
| `_on_vb_var_editor` | `EventHandler._on_vb_var_editor` |

**总计减少约 378 行实现代码**

### EventHandler 关键修复

| 问题 | 修复 |
|------|------|
| `_on_range_changed` 使用硬编码 `100` ms | 改为 `UI_DEBOUNCE_DELAY_MS` |
| `_schedule_cursor_geometry_update` 使用硬编码 `100` ms | 改为 `max(15, UI_DEBOUNCE_DELAY_MS)` |
| `_start_interaction` 缺少注释和 `pass` | 恢复完整注释结构 |
| 异常处理使用 `logger.debug` | 改为 `print(f"错误: {e}")` 匹配备份 |

### PlotUIManager 关键修复

| 问题 | 修复 |
|------|------|
| `_setup_plot_area` timer 连接到 `pw._end_interaction` | 改为 `pw._event_handler._end_interaction` |
| `_setup_plot_area` rangeChanged 连接到 `pw._on_range_changed` | 改为 `pw._event_handler._on_range_changed` |
| `_setup_interaction` timer 连接到 `pw._refresh_cursor_geometry` | 改为 `pw._event_handler._refresh_cursor_geometry` |
| `_connect_viewbox_signals` 调用 `pw._connect_viewbox_signals()` | 改为 `pw._event_handler._connect_viewbox_signals()` |
| `_run_stats_refresh` 使用 `pw.plot_context` | 改为 `pw.window()` 匹配备份 |

### 备份对比验证

与 `.trae/backup/v2.0-pre-refactor/plot_widget.py` 逐方法对比：

| 模块 | 方法数 | 一致 | 说明 |
|------|--------|------|------|
| EventHandler | 17 | 17 ✅ | `self`→`self.pw` 变换，`_on_vb_*` 使用 `plot_context` 代理 |
| PlotUIManager | 11 | 11 ✅ | `self`→`self.pw` 变换 |

### 语法验证

- `event_handler.py` — `py_compile` ✅
- `plot_ui_manager.py` — `py_compile` ✅
- `plot_widget.py` — `py_compile` ✅

### 代码规模变化

- `plot_widget.py`: 1916 → 1538 行（减少 378 行）
- `event_handler.py`: 修正 4 处差异，247 行
- `plot_ui_manager.py`: 修正 5 处差异，383 行

### 重构总量汇总

| 阶段 | 版本 | Manager | 委托方法数 | widget 行数 |
|------|------|---------|-----------|-------------|
| 重构前 | — | — | — | ~4100+ |
| Step 4 (CursorManager) | v4.0 | CursorManager | 22 | ~3300 |
| Step 5 (AxisManager + PlotData) | v5.0 | AxisManager + PlotData | ~19 | ~2800 |
| Step 6 (PlotData + MultiCurve) | v6.0 | PlotData + MultiCurve | 23 | 2061 |
| Step 7 (MarkRegion) | v7.1 | MarkRegionManager | 4 | 1916 |
| Steps 1-3 (Event + UI + Init) | v8.0 | EventHandler + PlotUIManager | 27 | 1538 |

**总委托: ~95 个方法到 7 个 Manager，widget 从 ~4100+ → 1538 行**

