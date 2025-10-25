# 重载问题修复总结

## 问题描述

1. **重载/加载新数据后plot不重新绘制**: 即使新数据的通道比例>30%，原有plot中的数据也不会重新绘制
2. **重载时UI立即清空**: 点击重载的一瞬间，所有plot都被清空，即使数据还没被完全读入

## 问题分析

### 根本原因
在重构过程中，重载时的数据清理和UI更新时序出现了问题：

1. `_load_file`方法在开始时立即调用`_cleanup_old_data()`
2. `_cleanup_old_data()`调用`clear_all_plots()`，立即清空所有plot
3. 新数据加载完成后，`replots_after_loading`被调用，但可能由于时序问题导致重绘失败

### 具体问题点
1. **时序问题**: 重载时过早清理plot状态
2. **数据引用问题**: plot_widget的data属性可能没有正确更新
3. **清理逻辑问题**: 重载时不应该立即清空UI

## 修复方案

### 1. 修改重载时的清理逻辑
```python
def _load_file(self, file_path: str, is_reload: bool = False):
    # 重载时不立即清理plot，等新数据加载完成后再处理
    if not is_reload and hasattr(self, 'loader') and self.loader is not None:
        self._cleanup_old_data()
```

**效果**: 重载时保留当前plot状态，避免UI立即清空

### 2. 在数据加载完成后清理旧数据
```python
def _on_load_done(self, loader, file_path: str):
    self._progress.close()
    
    # 如果是重载，先清理旧的loader数据
    if hasattr(self, 'loader') and self.loader is not None:
        if hasattr(self.loader, '_df'):
            del self.loader._df
        del self.loader
    
    self.loader = loader
    self._apply_loader()
    self._post_load_actions(file_path)
```

**效果**: 确保在应用新数据前正确清理旧数据

### 3. 确保数据引用正确
```python
def _apply_loader(self):
    """把 loader 的内容同步到 UI"""
    self.var_names = self.loader.var_names
    self.units = self.loader.units
    self.time_channels_infos = self.loader.time_channels_info
    self.data_validity = self.loader.df_validity
    self.data = self.loader.df  # 设置主数据
    self.list_widget.populate(self.var_names, self.units, self.data_validity)

    # 更新所有 plot_widgets 的数据
    for container in self.plot_widgets:
        widget = container.plot_widget
        widget.data = self.loader.df  # 确保每个plot_widget都有正确的data
        widget.units = self.loader.units
        widget.time_channels_info = self.loader.time_channels_info

    self.replots_after_loading()  # 重新绘制
```

**效果**: 确保所有plot_widget都有正确的数据引用

## 修复后的数据流

### 重载流程
```
1. 用户点击重载
2. _load_file(file_path, is_reload=True)  # 不立即清理plot
3. 后台加载新数据
4. _on_load_done()  # 清理旧loader数据
5. _apply_loader()  # 更新所有数据引用
6. replots_after_loading()  # 重新绘制plot
```

### 新数据加载流程
```
1. 用户加载新文件
2. _load_file(file_path, is_reload=False)  # 立即清理旧数据
3. 后台加载新数据
4. _on_load_done()  # 清理旧loader数据
5. _apply_loader()  # 更新所有数据引用
6. replots_after_loading()  # 重新绘制plot
```

## 关键改进

### 1. 时序优化
- 重载时保留plot状态直到新数据加载完成
- 避免UI闪烁和用户体验问题

### 2. 数据一致性
- 确保所有plot_widget都有正确的数据引用
- 保证`replots_after_loading`能正确工作

### 3. 内存管理
- 在适当的时机清理旧数据
- 避免内存泄漏

## 测试验证

### 测试文件
创建了`test_reload.py`脚本，生成测试数据：
- `test_data1.csv` (1000行)
- `test_data2.csv` (1500行)

### 测试步骤
1. 加载 `test_data1.csv`
2. 绘制一些变量
3. 重载数据
4. 检查plot是否正确重新绘制
5. 加载 `test_data2.csv`
6. 检查30%通道比例逻辑是否工作

## 预期效果

修复后应该实现：
✅ **重载时UI不立即清空**: 保持当前plot状态直到新数据加载完成
✅ **重载后正确重绘**: 新数据加载完成后，plot正确重新绘制
✅ **30%通道比例逻辑**: 当新数据中存在的变量比例>30%时，自动推送到plot
✅ **内存管理**: 正确清理旧数据，避免内存泄漏

## 总结

通过调整重载时的数据清理时序和确保数据引用正确，解决了重载时plot不重新绘制的问题。现在重载功能应该能够正常工作，提供更好的用户体验。