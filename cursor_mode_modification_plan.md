# Cursor Mode 菜单功能增强修改计划

## 需求概述

1. 在右键菜单的"Cursor Mode"中添加一个"off"选项（放在最后）
2. "Cursor Mode"菜单需要始终保持激活状态（即可以点击）
3. 与顶部的"显示光标"/"隐藏光标"按钮实现双向联动：
   - 当光标处于"off"状态时，Cursor Mode菜单选中"off"选项
   - 当光标处于"on"状态时，Cursor Mode菜单选中当前激活的模式选项（1 free cursor / 1 anchored cursor / 2 anchored cursor）

---

## 当前代码结构分析

### 关键文件

1. **csv_plot_pyqt6.py** - 主程序文件
2. **src/ui/widgets/custom_viewbox.py** - 自定义右键菜单实现
3. **src/app/plot_context.py** - 绘图上下文服务层

### 当前状态

1. **custom_viewbox.py** (第93-96行)：
   - `cursor_enabled` 变量控制 Cursor Mode 菜单是否可用
   - 使用 `is_cursor_enabled()` 检查 `cursor_btn.isChecked()` 状态

2. **custom_viewbox.py** (第101-113行)：
   - 只有三个选项：["1 free cursor", "1 anchored cursor", "2 anchored cursor"]

3. **custom_viewbox.py** (第235-238行)：
   - `_get_current_cursor_mode()` 默认返回 "1 free cursor"

4. **csv_plot_pyqt6.py** (第4598行)：
   - `self.cursor_mode` 默认值为 "1 free cursor"
   - **默认激活 "1 free cursor" 模式**

5. **csv_plot_pyqt6.py** (第5896-5939行)：
   - `set_cursor_mode()` 函数只接受三个模式
   - 第5899行检查：`if mode not in ("1 free cursor", "1 anchored cursor", "2 anchored cursor"):`

6. **csv_plot_pyqt6.py** (第5945-5965行)：
   - `toggle_cursor_all()` 函数在点击按钮时调用
   - 第5964-5965行：当开启时设置为 "1 free cursor"

---

## 修改方案

### 1. 修改 `custom_viewbox.py`

#### 修改点 1: 移除 `cursor_enabled` 对 Cursor Mode 菜单的禁用限制
- 主菜单始终可用
- 所有选项（包括三个模式和"off"）都始终可用（不禁用）
- 当光标为 off 时，"off"选项被选中；当光标为 on 时，对应模式被选中

#### 修改点 2: 添加 "off" 选项到 Cursor Mode 菜单
- 在现有三个选项后添加 "off" 选项
- 实现当前选中状态的判断逻辑

#### 修改点 3: 处理所有模式的点击事件
- "off"选项点击时，发送信号通知上层关闭光标
- 三个模式选项点击时，即使光标当前为 off，也会先自动开启光标，然后应用该模式

### 2. 修改 `plot_context.py`

#### 修改点: 添加 `set_cursor_enabled()` 方法
- 用于控制光标启用/禁用

### 3. 修改 `csv_plot_pyqt6.py`

#### 修改点 1: 更新 `set_cursor_mode()` 函数
- 添加对 "off" 模式的支持
- 当设置为 "off" 时，触发 `toggle_cursor_all(False)`

#### 修改点 2: 更新 `toggle_cursor_all()` 函数
- 当关闭光标时，设置 `cursor_mode = "off"`
- 当开启光标时，恢复到之前的模式（如果有的话），或者默认 "1 free cursor"

#### 修改点 3: 保存上一个有效的光标模式
- 添加变量 `last_valid_cursor_mode` 用于存储非 off 的最后模式

---

## 具体修改内容

### 文件 1: src/ui/widgets/custom_viewbox.py

#### 第93-96行：修改 Cursor Mode 菜单的启用逻辑
```python
# 原代码
cursor_enabled = self._get_cursor_enabled()
cursor_menu = QMenu("Cursor Mode", menu)
cursor_menu.setEnabled(cursor_enabled)

# 修改后
cursor_enabled = self._get_cursor_enabled()
cursor_menu = QMenu("Cursor Mode", menu)
# Cursor Mode 菜单始终可用
cursor_menu.setEnabled(True)
```

#### 第98-113行：修改菜单项创建逻辑
```python
# 原代码
cursor_group = QActionGroup(cursor_menu)
cursor_group.setExclusive(True)
current_mode = self._get_current_cursor_mode()
for mode_text in ["1 free cursor", "1 anchored cursor", "2 anchored cursor"]:
    mode_act = QAction(mode_text, cursor_menu)
    mode_act.setCheckable(True)
    mode_act.setChecked(mode_text == current_mode)
    mode_act.setEnabled(cursor_enabled)
    mode_act.triggered.connect(...)
    ...

# 修改后
cursor_group = QActionGroup(cursor_menu)
cursor_group.setExclusive(True)
current_mode = self._get_current_cursor_mode()
cursor_enabled = self._get_cursor_enabled()

# 添加三个正常模式选项
for mode_text in ["1 free cursor", "1 anchored cursor", "2 anchored cursor"]:
    mode_act = QAction(mode_text, cursor_menu)
    mode_act.setCheckable(True)
    # 选中逻辑：光标开启时检查是否匹配当前模式，光标关闭时不选中
    mode_act.setChecked(cursor_enabled and mode_text == current_mode)
    # 所有选项始终可用
    mode_act.setEnabled(True)
    mode_act.triggered.connect(
        lambda checked, m=mode_text: self.signals.request_set_cursor_mode.emit(
            m, self.plot_widget, self.context_x
        )
    )
    cursor_group.addAction(mode_act)
    cursor_menu.addAction(mode_act)

# 添加 "off" 选项
off_act = QAction("off", cursor_menu)
off_act.setCheckable(True)
off_act.setChecked(current_mode == "off" or not cursor_enabled)
# "off" 选项始终可用
off_act.setEnabled(True)
off_act.triggered.connect(
    lambda checked: self.signals.request_set_cursor_mode.emit(
        "off", self.plot_widget, self.context_x
    )
)
cursor_group.addAction(off_act)
cursor_menu.addAction(off_act)
```

### 文件 2: src/app/plot_context.py

#### 在 PlotContext 类中添加辅助方法
```python
# 在 is_cursor_enabled() 和 set_cursor_checked() 方法之间添加
def set_cursor_enabled(self, enabled: bool) -> None:
    self._services.toggle_cursor_all(enabled)
```

### 文件 3: csv_plot_pyqt6.py

#### 第4597-4599行：添加保存上一个有效模式的变量
```python
# 原代码
self.cursor_values_hidden = False  # 默认显示完整cursor（包括圆圈和y值）
self.cursor_mode = "1 free cursor"
self.pinned_x_values = []

# 修改后
self.cursor_values_hidden = False  # 默认显示完整cursor（包括圆圈和y值）
self.cursor_mode = "1 free cursor"
self.last_valid_cursor_mode = "1 free cursor"  # 保存上一个有效的非off模式
self.pinned_x_values = []
```

#### 第5896-5939行：修改 set_cursor_mode() 函数
```python
# 原代码
def set_cursor_mode(self, mode, *, source_plot=None, context_x=None):
    if not hasattr(self, "cursor_btn") or not self.cursor_btn.isChecked():
        return
    if mode not in ("1 free cursor", "1 anchored cursor", "2 anchored cursor"):
        return
    ...

# 修改后
def set_cursor_mode(self, mode, *, source_plot=None, context_x=None):
    # 处理 "off" 模式
    if mode == "off":
        if self.cursor_btn.isChecked():
            self.toggle_cursor_all(False)
        return
    
    # 检查有效模式
    if mode not in ("1 free cursor", "1 anchored cursor", "2 anchored cursor"):
        return
    
    # 确保光标处于开启状态
    if not hasattr(self, "cursor_btn") or not self.cursor_btn.isChecked():
        self.toggle_cursor_all(True)
    
    # 保存为上一个有效模式
    self.last_valid_cursor_mode = mode
    ...
    # 其余代码保持不变
```

#### 第5945-5965行：修改 toggle_cursor_all() 函数
```python
# 原代码
def toggle_cursor_all(self, checked):
    ...
    if checked:
        self.cursor_mode = "1 free cursor"

# 修改后
def toggle_cursor_all(self, checked):
    ...
    if checked:
        # 恢复到上一个有效模式，或者使用默认值
        self.cursor_mode = self.last_valid_cursor_mode
    else:
        # 保存当前模式到 last_valid_cursor_mode（如果当前不是 off）
        if self.cursor_mode != "off":
            self.last_valid_cursor_mode = self.cursor_mode
        self.cursor_mode = "off"
```

---

## 数据流分析

### 场景 1: 用户点击顶部"隐藏光标"按钮
1. `cursor_btn.clicked` → `toggle_cursor_all(False)`
2. 设置 `cursor_mode = "off"`，同时保存当前模式到 `last_valid_cursor_mode`
3. 所有 plot 的 cursor 被隐藏
4. 右键点击打开菜单时，`_get_current_cursor_mode()` 返回 "off"
5. "off" 选项被选中，其他选项禁用

### 场景 2: 用户在 Cursor Mode 菜单中点击 "off"
1. 菜单点击 → `request_set_cursor_mode("off")`
2. `set_cursor_mode("off")` 被调用
3. 触发 `toggle_cursor_all(False)`
4. 同场景1的后续流程

### 场景 3: 用户点击顶部"显示光标"按钮
1. `cursor_btn.clicked` → `toggle_cursor_all(True)`
2. 设置 `cursor_mode = last_valid_cursor_mode`（恢复之前的模式）
3. 所有 plot 的 cursor 显示并应用相应模式
4. 右键点击打开菜单时，对应的模式选项被选中

### 场景 4: 用户在 Cursor Mode 菜单中点击某个正常模式（光标当前为 off）
1. 菜单点击 → `request_set_cursor_mode("1 free cursor")`
2. `set_cursor_mode()` 检测到光标为 off，先调用 `toggle_cursor_all(True)`
3. 设置 `last_valid_cursor_mode = "1 free cursor"`
4. 应用该光标模式

---

## 测试计划

测试以下场景：
1. 点击"隐藏光标"，检查菜单中"off"是否被选中
2. 菜单中点击"off"，检查顶部按钮状态是否变为未选中
3. 点击"显示光标"，检查是否恢复到之前的模式
4. 在 off 状态下菜单中点击某个模式，检查是否开启并应用该模式
5. 在开启状态下切换不同模式，检查菜单选中状态是否正确
