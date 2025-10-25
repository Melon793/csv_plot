# 加载新数据问题修复总结

## 问题描述

1. **加载新数据时plot立即被清除**: 点击加载新数据后，原有plot中的数据会被立马清除
2. **加载新数据后plot不重新绘制**: 数据加载完毕后也不会重新绘制出来
3. **30%通道比例逻辑失效**: 即使新数据的通道比例>30%，也不会自动推送到plot中

## 问题分析

### 根本原因
1. **过早清理plot**: `_load_file`方法在加载新数据时立即调用`_cleanup_old_data()`
2. **reset_plot不完整**: `reset_plots_after_loading`方法只调用`reset_plot`，没有清空plot内容
3. **时序问题**: plot被清空后，`replots_after_loading`无法正确检测到需要重新绘制的变量

### 具体问题点
1. **UI立即清空**: 加载新数据时立即清空plot，用户体验差
2. **plot内容残留**: `reset_plot`只重置坐标轴，不清空plot内容
3. **变量检测失败**: 清空plot后，`unique_y_names`为空，导致逻辑错误

## 修复方案

### 1. 修改加载新数据时的清理逻辑
```python
def _load_file(self, file_path: str, is_reload: bool = False):
    # 无论是重载还是加载新数据，都不立即清理plot，等新数据加载完成后再处理
    # 这样可以避免UI立即清空，提供更好的用户体验
```

**效果**: 加载新数据时保留当前plot状态，避免UI立即清空

### 2. 完善reset_plots_after_loading方法
```python
def reset_plots_after_loading(self, index_xMin, index_xMax):
    for container in self.plot_widgets:
        # 先清空plot内容，然后重置坐标轴
        container.plot_widget.clear_plot_item()
        container.plot_widget.reset_plot(index_xMin, index_xMax)
        container.plot_widget.clear_value_cache()
```

**效果**: 确保plot被完全清空和重置

### 3. 保持数据加载完成后的处理逻辑
```python
def _on_load_done(self, loader, file_path: str):
    self._progress.close()
    
    # 清理旧的loader数据（无论是重载还是加载新数据）
    if hasattr(self, 'loader') and self.loader is not None:
        if hasattr(self.loader, '_df'):
            del self.loader._df
        del self.loader
    
    self.loader = loader
    self._apply_loader()  # 这会调用replots_after_loading()
    self._post_load_actions(file_path)
```

**效果**: 确保新数据正确应用到UI

## 修复后的数据流

### 加载新数据流程
```
1. 用户选择新文件
2. _load_file(file_path, is_reload=False)  # 不立即清理plot
3. 后台加载新数据
4. _on_load_done()  # 清理旧loader数据
5. _apply_loader()  # 更新所有数据引用
6. replots_after_loading()  # 重新绘制plot
   - 如果通道比例>30%: 保留现有plot并更新数据
   - 如果通道比例≤30%: 清空所有plot并重置
```

### 重载流程
```
1. 用户点击重载
2. _load_file(file_path, is_reload=True)  # 不立即清理plot
3. 后台加载新数据
4. _on_load_done()  # 清理旧loader数据
5. _apply_loader()  # 更新所有数据引用
6. replots_after_loading()  # 重新绘制plot
   - 如果通道比例>30%: 保留现有plot并更新数据
   - 如果通道比例≤30%: 清空所有plot并重置
```

## 关键改进

### 1. 用户体验优化
- **无UI闪烁**: 加载新数据时不会立即清空plot
- **平滑过渡**: 数据加载完成后才更新UI
- **保持状态**: 重载时保持当前plot状态

### 2. 逻辑完善
- **完整清理**: `reset_plots_after_loading`现在会完全清空plot
- **正确检测**: 30%通道比例逻辑现在能正确工作
- **数据一致性**: 确保所有plot_widget都有正确的数据引用

### 3. 时序优化
- **延迟清理**: 在数据加载完成后才清理旧数据
- **正确更新**: 确保新数据正确应用到UI
- **内存管理**: 在适当的时机清理旧数据

## 30%通道比例逻辑

### 工作原理
```python
def replots_after_loading(self):
    # 收集所有现有的y_name
    all_y_names = [container.plot_widget.y_name for container in self.plot_widgets if container.plot_widget.y_name]
    
    # 找到在新数据中存在的有效y_name
    found = [y for y in all_y_names if y in self.loader.var_names and self.loader.df_validity.get(y, -1) == 1]
    ratio = len(found) / len(all_y_names)
    
    if ratio <= RATIO_RESET_PLOTS or len(found) < 1:
        # 通道比例≤30%或没有有效变量：清空所有plot
        self.reset_plots_after_loading(1, self.loader.datalength)
    else:
        # 通道比例>30%：保留现有plot并更新数据
        # 更新每个plot的数据和坐标轴
        for container in self.plot_widgets:
            widget = container.plot_widget
            y_name = widget.y_name
            if y_name in self.loader.df.columns and self.loader.df_validity.get(y_name, -1) >= 0:
                success = widget.plot_variable(y_name)
                if not success:
                    widget.clear_plot_item()
```

### 效果
- **通道比例>30%**: 保留现有plot，更新数据，保持用户的工作状态
- **通道比例≤30%**: 清空所有plot，重新开始，避免显示无关数据

## 测试验证

### 测试场景
1. **加载新数据（通道比例>30%）**:
   - 加载包含现有变量的新文件
   - 验证plot被保留并更新
   
2. **加载新数据（通道比例≤30%）**:
   - 加载不包含现有变量的新文件
   - 验证plot被清空并重置
   
3. **重载数据**:
   - 重载当前文件
   - 验证plot被保留并更新

### 预期结果
✅ **加载新数据时UI不立即清空**: 保持当前plot状态直到新数据加载完成
✅ **30%通道比例逻辑正常工作**: 根据通道比例决定是否保留plot
✅ **数据正确更新**: 新数据加载完成后，plot正确显示
✅ **用户体验优化**: 无UI闪烁，平滑过渡

## 总结

通过修改加载新数据时的清理时序和完善plot重置逻辑，解决了加载新数据时plot不重新绘制的问题。现在无论是重载还是加载新数据，都能提供一致的用户体验，并且30%通道比例逻辑能够正常工作。