# Windows 最大化时出现短暂黑框问题分析

## 问题现象

在 Windows 平台上，应用打开后最大化窗口时：

1. 外框立即铺满全屏（Windows 系统级操作）
2. 内部元素未同步扩展，四周出现大面积"黑框"
3. 黑框持续约 0.3~0.5 秒后消失，内容正常填充

**关键特征**：即使不加载数据（plot 区域 hidden），最大化时也会有黑框现象，说明与 pyqtgraph 渲染无关。

---

## 根因分析

### 事件链

```
① 用户点击最大化按钮
         ↓
② Windows DWM 立即将窗口外框从原尺寸拉伸到全屏
   （OS 级 framebuffer resize，瞬间完成，但内容是旧的）
         ↓
③ Qt 收到 WM_SIZE → QMainWindow.resizeEvent(event) 
   [csv_plot.py:4612](file:///Users/xiaolin/CSV_Plot_PySide/csv_plot.py#L4612)
         ↓
④ super().resizeEvent(event) → QMainWindow 内部 layout 开始重新计算
   → QSplitter / QGridLayout / QVBoxLayout 逐个重新分配子 widget 尺寸
         ↓
⑤ self.layout_manager._handle_resize(event) → QTimer.singleShot(0, ...)
   [layout_manager.py:L75-L83](file:///Users/xiaolin/CSV_Plot_PySide/src/ui/layout_manager.py#L75-L83)
   将 splitter 宽度修正推迟到下一个事件循环
         ↓
⑥ Windows DWM 在当前帧合成画面
   Qt 的 widget repaint 还没完成 → 新暴露区域显示 widget 默认背景色（深灰/黑）
```

### 核心矛盾

**Qt 的 layout recalculation + widget repaint 是异步的、多事件循环的，而 Windows DWM 的桌面合成是每帧同步的。**

从 step② 到 step⑤ 之间，Qt 内部至少需要：
- 1 个 `resizeEvent` 处理周期
- +1 个事件循环（`QTimer.singleShot(0)` 的回调）
- +N 个 widget 的 `paintEvent`

而 DWM 在 step② 时就已经拿到了新尺寸的 framebuffer。在 Qt 完成所有 repaint 之前，新暴露的区域显示的是未初始化/默认背景色。

### 为什么 Windows 特别明显

| 平台   | 合成机制                          | 表现           |
|--------|----------------------------------|----------------|
| macOS  | Core Animation + layer-backed 双缓冲，过渡动画平滑 | 几乎无感       |
| Linux  | Compositor（mutter/kwin）提供隐式缓冲          | 通常无感       |
| Windows| DWM 不为自定义渲染的 Qt widget 提供过渡动画      | **黑框明显**   |

### 关键代码位置

1. [central widget 创建](file:///Users/xiaolin/CSV_Plot_PySide/csv_plot.py#L4345) — 未设置背景色
   ```python
   central = QWidget()
   self.setCentralWidget(central)
   ```

2. [resizeEvent](file:///Users/xiaolin/CSV_Plot_PySide/csv_plot.py#L4612-L4614) — 入口
   ```python
   def resizeEvent(self, event):
       super().resizeEvent(event)
       self.layout_manager._handle_resize(event)
   ```

3. [_handle_resize](file:///Users/xiaolin/CSV_Plot_PySide/src/ui/layout_manager.py#L75-L83) — 将 splitter 修正延迟到下一事件循环
   ```python
   def _handle_resize(self, _event):
       if (not self.mw.var_table_user_adjusted
           and getattr(self.mw, "_splitter_ready", False)
           and hasattr(self.mw, "main_splitter")):
           if not getattr(self.mw, "_pending_splitter_adjustment", False):
               self.mw._pending_splitter_adjustment = True
               QTimer.singleShot(0, self._apply_fixed_splitter_width)
   ```

---

## 解决方案

### 方案 1（推荐）：resizeEvent 中强制同步刷新

在 MainWindow 的 `resizeEvent` 中，最大化时调用 `processEvents()` + `repaint()`，
强制在当前帧完成所有 layout 和绘制。

**实现** (`csv_plot.py:4612`)：

```python
def resizeEvent(self, event):
    super().resizeEvent(event)
    self.layout_manager._handle_resize(event)
    if sys.platform == "win32" and self.isMaximized():
        QApplication.processEvents()
        self.repaint()
```

**原理**：
- `processEvents()` 强制处理 `QTimer.singleShot(0)` 等 pending 事件，让 splitter 修正立刻生效
- `repaint()` 强制整个窗口立即调用 `paintEvent`，不等操作系统下一次 repaint 请求
- 仅在 Windows 且最大化时触发，不影响正常拖拽缩放

**风险**：见下方"风险评估"章节


### 方案 2（低风险辅助）：设置白色背景

将 central widget、QSplitter、QMainWindow 的背景色统一设为白色（或应用品牌色），
使短暂未绘制区域的颜色与最终内容一致，大幅降低视觉突兀感。

**实现** (`csv_plot.py:4345`)：

```python
central = QWidget()
central.setAutoFillBackground(True)
pal = central.palette()
pal.setColor(central.backgroundRole(), QColor(255, 255, 255))
central.setPalette(pal)
self.setCentralWidget(central)
```

或用 stylesheet（覆盖更彻底）：
```python
self.main_splitter.setStyleSheet("QSplitter { background-color: white; }")
```

**原理**：
- 即使底层渲染延迟依旧存在，"白框"比"黑框"在视觉上突兀程度低得多
- 无任何性能或事件循环副作用

**风险**：见下方"风险评估"章节


### 方案 3（低风险辅助）：WA_OpaquePaintEvent

告诉 Qt 该 widget 会自行填满整个区域，不需要父窗口透底。

```python
central.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
```

需配合方案 2 使用，单独使用无效。

**风险**：如果某些子 widget 未完全覆盖父 widget，会露出未初始化像素


### 方案 4（不推荐）：为 QGraphicsView 设置 FullViewportUpdate

```python
self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
```

在大数据量下每次缩放/平移都触发全量重绘，性能下降明显。且本问题与 pyqtgraph 无关，此方案治标不治本。

---

## 风险评估

### 方案 1（processEvents + repaint）的风险

| 风险 | 严重程度 | 触发场景 | 缓解措施 |
|------|---------|---------|---------|
| **事件重入（reentrancy）** | ⚠️ 中 | `processEvents()` 期间收到新的 resize/mouse/key 事件，导致 `resizeEvent` 被嵌套调用 | 加 `_in_sync_refresh` 守卫标记，避免递归 |
| **拖拽缩放时性能抖动** | ⚠️ 低 | 仅 `isMaximized()` 为 true 时触发，拖拽缩放不会进入此分支 | 已通过条件判断规避 |
| **Timer 回调被提前执行** | ⚠️ 中 | `processEvents()` 会执行所有 pending timer，包括 `_apply_fixed_splitter_width` 和其他业务 timer | reset_plots 等 timer 可能在数据未就绪时执行；**需要仔细排查所有 `QTimer.singleShot(0)` 的用途** |
| **信号槽被提前分发** | ⚠️ 低 | `processEvents()` 会分发所有排队的信号，如 splitterMoved、textChanged 等 | 已有 `blockSignals` 保护的场景不受影响 |
| **平台差异** | ⚠️ 低 | `processEvents()` 行为在不同平台有细微差异 | 已通过 `sys.platform == "win32"` 限制 |

#### 重入风险详细说明

```
resizeEvent (第1次)
  → processEvents()
    → _handle_resize → QTimer.singleShot(0) 回调 → setSizes()
      → QSplitter 触发 layout recalculation
        → 可能触发新的 resizeEvent (第2次 / 嵌套)
```

如果在 `processEvents()` 期间 `QSplitter.setSizes()` 触发了新的 resize 事件，就会形成嵌套调用。虽然 Qt 的 resize event 通常是压缩的（同一帧内多次 resize 只保留最新一次），但不能完全排除嵌套的可能性。

**建议添加守卫**：

```python
def resizeEvent(self, event):
    super().resizeEvent(event)
    self.layout_manager._handle_resize(event)
    if sys.platform == "win32" and self.isMaximized():
        if not getattr(self, '_in_sync_resize', False):
            self._in_sync_resize = True
            try:
                QApplication.processEvents()
                self.repaint()
            finally:
                self._in_sync_resize = False
```


### 方案 2（白色背景）的风险

| 风险 | 严重程度 | 说明 |
|------|---------|------|
| **Windows 深色主题冲突** | ⚠️ 极低 | 应用本身是浅色主题（白底），白色背景与整体风格一致 |
| **stylesheet 覆盖范围不可控** | ⚠️ 极低 | 使用 `setPalette` 方式而非 stylesheet，影响范围精确可控 |

方案 2 几乎无风险，仅改善视觉体验。

---

## 推荐实施策略

**分两步走，先低风险后高风险**：

### 第一步（零风险，先上）
实施 **方案 2（白色背景）**。改动仅涉及 `central()` widget 的背景色设置，不改变任何事件处理逻辑。

### 第二步（需验证，后上）
实施 **方案 1（processEvents + repaint）**，并加上重入守卫。

上线前需在 Windows 上验证以下场景：
1. 不加载数据时最大化 — 确认无黑框
2. 加载大数据文件后最大化 — 确认无黑框、无崩溃
3. 最大化状态下拖拽 splitter — 确认无异常
4. 连续快速最大化/还原 — 确认无嵌套 resize 崩溃
5. 有 mark region / cursor 状态下最大化 — 确认数据正确

---

## 为什么不建议的方案的替代思路

### ❌ 不推荐：使用 `showMaximized()` 代替系统标题栏的全屏按钮

这不能解决问题，因为内部 layout 延迟依旧存在。

### ❌ 不推荐：设置 `Qt.FramelessWindowHint` + 自定义标题栏

改动量太大，且会失去原生窗口管理器的所有便利（Aero Snap、任务栏预览等）。

### ❌ 不推荐：使用 `changeEvent` 拦截 `WindowStateChange`

`changeEvent` 触发时机与 `resizeEvent` 相同，无法提前拦截。

---

## 参考

- [QTBUG-45121](https://bugreports.qt.io/browse/QTBUG-45121) — QWidget resize flickering on Windows
- Qt 文档：`QWidget::repaint()` vs `QWidget::update()` — `repaint()` 是同步的，`update()` 是异步的
- Qt 文档：`QApplication::processEvents()` — 注意事项和风险
