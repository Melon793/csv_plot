# 渲染性能优化实施计划

> 版本：v2.0
> 日期：2026-08-11
> 分支基线：`dev/perf-logging`（Phase 0 已完成）
> pyqtgraph 版本：0.14.0 / PySide6
> 跨平台：Mac + Win 双平台适用

**v2.0 变更**：合并 §7 方案 A（滚轮合并）+ 方案 C（关 Y autoVisible）为双平台核心优化；
移除 Phase 2（批量 XLink，高风险）和 Phase 3（OpenGL，实验性）；重新规划测试步骤。

---

## 0. 全局概览

### 0.1 当前渲染管线瓶颈分析

```
wheelEvent → vb.scaleBy()
  → pyqtgraph 内部 XLink 级联（同步，N-1 个 slave plot）
    → 每个 plot 触发 sigRangeChanged
      → setAutoVisible(y=True) → Y 轴 auto-range 扫描全量数据
      → peak downsampling 重算
      → clipToView 裁剪
      → CPU rasterization 全量重绘
```

**核心问题**：单次 wheel 事件的实际代价 = `N_plot × M_curve × (Y-scan + peak + rasterize)`，
其中 Y autoVisible 是隐藏的放大器——每次 X 范围变化都触发全数据 Y 轴扫描。

### 0.2 关键文件索引

| 文件 | 关键行 | 角色 |
|------|--------|------|
| `src/core/config.py` | L1-235 | 全局常量，无 PERF 常量 |
| `src/ui/widgets/event_handler.py` | L140-147 (`_start_interaction`), L149-160 (`_end_interaction`), L82-122 (`_on_range_changed`) | 交互生命周期 |
| `src/ui/widgets/plot_ui_manager.py` | L192 (`setAutoVisible`), L195-196 (`clipToView`/`downsampling`) | 绘图区初始化 |
| `src/ui/widgets/plot_widget.py` | L395-435 (`paintEvent`), L437-457 (`wheelEvent`) | 绘制与缩放入口 |
| `src/ui/layout_manager.py` | L641 (`setXLink`), L101-218 (`_sync_linked_x_ranges`) | XLink 建立与健康检查 |
| `csv_plot.py` | L21 (`pg.setConfigOptions`) | 全局 pyqtgraph 配置 |

### 0.3 约束与禁区

- **Peak downsampling 不可协商**：必须保持 `mode="peak", auto=True`
- **Phase 2A "freeze downsampling during interaction" 已回退**：曾导致性能劣化，不再考虑
- **应用层防抖无法抑制 pyqtgraph 内部重算**：XLink 级联是 pyqtgraph 内部同步机制
- **macOS OpenGL 自 10.14 起已废弃**：OpenGL 方案不作为本次优化目标

---

## Phase 0：性能日志基础设施

### 1. 目标

建立统一的 `[PERF]` 日志体系，为后续所有优化提供量化度量基准。

### 2. 修改文件

#### 2.1 `src/core/config.py` — 新增 PERF 常量

**位置**：在 L98 (`UI_DEBOUNCE_DELAY_MS`) 之后、L99 (`PLOT_ROW_MAX_DEFAULT`) 之前插入。

```python
# === Performance Logging ===
PERF_LOG_ENABLED = True           # 总开关：False 时所有 [PERF] 日志降级为 DEBUG
PERF_PAINT_WARN_MS = 16.0         # paintEvent 超过此值输出 WARNING
PERF_WHEEL_WARN_MS = 8.0          # wheelEvent scaleBy 超过此值输出 WARNING
PERF_INTERACTION_WARN_MS = 50.0   # _end_interaction 总耗时超过此值输出 WARNING
PERF_RANGE_CB_WARN_MS = 5.0       # _on_range_changed 回调耗时超过此值输出 WARNING
PERF_LOG_INTERVAL_FRAMES = 30     # 每 N 次事件输出一次汇总（避免日志洪泛）
```

**理由**：集中在 config.py 管理，便于运行时通过环境变量或 UI 开关调整。

#### 2.2 `src/ui/widgets/plot_widget.py` — paintEvent 计时

**目标函数**：`paintEvent()` (L395-435)

**修改点**：在 `super().paintEvent(event)` 调用前后插入 `time.perf_counter()` 计时。

```python
# 在文件顶部 import 区追加：
import time
from src.core.config import PERF_LOG_ENABLED, PERF_PAINT_WARN_MS, PERF_LOG_INTERVAL_FRAMES

# paintEvent 方法内，替换 L430-435 的 try 块：
```

**精确修改**：L430-435 的 `try` 块替换为：

```python
        try:
            if PERF_LOG_ENABLED:
                _t0 = time.perf_counter()
            super().paintEvent(event)
            if PERF_LOG_ENABLED:
                _dt_ms = (time.perf_counter() - _t0) * 1000
                pw_name = getattr(self, '_perf_paint_count', 0)
                self._perf_paint_count = pw_name + 1
                if _dt_ms > PERF_PAINT_WARN_MS:
                    logger.warning(
                        "[PERF][PAINT] slow paint: %.2fms (plot=%s, curves=%d)",
                        _dt_ms,
                        list(getattr(self, 'curves', {}).keys())[:2],
                        len(getattr(self, 'curves', {})),
                    )
        except RuntimeError as e:
            logger.debug("paintEvent RuntimeError (C++对象可能已销毁): %s", e)
        except Exception:
            logger.warning("paintEvent 异常", exc_info=True)
```

**注意**：仅在 guard 条件全部通过（非 loading、非 destroying、scene 有效）后才进入计时分支，确保不影响安全守卫路径。

#### 2.3 `src/ui/widgets/plot_widget.py` — wheelEvent 计时

**目标函数**：`wheelEvent()` (L437-457)

**修改点**：在 `vb.scaleBy()` 调用前后计时。

```python
# L451 替换为：
                if PERF_LOG_ENABLED:
                    _t0 = time.perf_counter()
                vb.scaleBy((factor, 1), center=(mouse_x, mouse_y))
                if PERF_LOG_ENABLED:
                    _dt_ms = (time.perf_counter() - _t0) * 1000
                    if _dt_ms > PERF_WHEEL_WARN_MS:
                        from src.core.config import PERF_WHEEL_WARN_MS
                        logger.warning(
                            "[PERF][WHEEL] slow scaleBy: %.2fms (factor=%.3f, center=(%.2f, %.2f))",
                            _dt_ms, factor, mouse_x, mouse_y,
                        )
```

需要在文件顶部 import `PERF_WHEEL_WARN_MS`。

#### 2.4 `src/ui/widgets/event_handler.py` — _on_range_changed 计时

**目标函数**：`_on_range_changed()` (L82-122)

**修改点**：在函数入口和出口计时。

```python
# 在文件顶部追加 import：
import time
from src.core.config import (
    safe_callback, UI_DEBOUNCE_DELAY_MS,
    PERF_LOG_ENABLED, PERF_RANGE_CB_WARN_MS,
)

# _on_range_changed 方法内，L83 try 块入口追加：
        _t0 = time.perf_counter() if PERF_LOG_ENABLED else 0

        # ... 原有逻辑不变 ...

        # 在函数末尾（try 块最后、except 之前）追加：
        if PERF_LOG_ENABLED:
            _dt_ms = (time.perf_counter() - _t0) * 1000
            if _dt_ms > PERF_RANGE_CB_WARN_MS:
                logger.warning(
                    "[PERF][RANGE_CB] slow _on_range_changed: %.2fms (interacting=%s)",
                    _dt_ms, self._is_interacting,
                )
```

#### 2.5 `src/ui/widgets/event_handler.py` — _end_interaction 计时

**目标函数**：`_end_interaction()` (L149-160)

**修改点**：测量整个 end-interaction 流程耗时。

```python
# _end_interaction 方法内追加计时：
    def _end_interaction(self):
        """结束交互时的处理，并广播刷新到所有 XLink 兄弟子图"""
        _t0 = time.perf_counter() if PERF_LOG_ENABLED else 0
        try:
            self._is_interacting = False
            self._queue_ui_refresh(immediate=True)
            self._refresh_siblings_after_interaction()
            if getattr(self.pw, '_pending_cursor_geometry_update', False):
                self.pw._pending_cursor_geometry_update = False
                self._schedule_cursor_geometry_update()
        except Exception:
            logger.warning("结束交互出错", exc_info=True)
        finally:
            if PERF_LOG_ENABLED:
                _dt_ms = (time.perf_counter() - _t0) * 1000
                from src.core.config import PERF_INTERACTION_WARN_MS
                if _dt_ms > PERF_INTERACTION_WARN_MS:
                    logger.warning(
                        "[PERF][END_INTERACT] slow _end_interaction: %.2fms",
                        _dt_ms,
                    )
```

### 3. 性能测量点汇总

| 标签 | 测量对象 | 文件 | 函数 | 输出条件 |
|------|----------|------|------|----------|
| `[PERF][PAINT]` | `super().paintEvent()` 耗时 | plot_widget.py | `paintEvent` | > 16ms |
| `[PERF][WHEEL]` | `vb.scaleBy()` 耗时 | plot_widget.py | `wheelEvent` | > 8ms |
| `[PERF][RANGE_CB]` | `_on_range_changed` 回调耗时 | event_handler.py | `_on_range_changed` | > 5ms |
| `[PERF][END_INTERACT]` | `_end_interaction` 总耗时 | event_handler.py | `_end_interaction` | > 50ms |

### 4. 回滚策略

- 所有 PERF 代码受 `PERF_LOG_ENABLED` 总开关控制
- 设置 `PERF_LOG_ENABLED = False` 即可完全禁用，开销仅剩一个 `if False` 分支
- 删除所有 `[PERF]` 相关代码不影响功能逻辑

### 5. 预期产出

- 建立 baseline 数据：当前每次 wheel/paint/interaction 的耗时分布
- 为 Phase 1-3 提供 before/after 对比依据
- 识别真正的热点（是 paint 慢还是 range_changed 慢还是 scaleBy 慢）

---

## Phase 1：滚轮事件合并节流（文档 §7 方案 A）

### 1. 目标

在 `wheelEvent` 中引入 16ms 合并器，将连续滚轮事件累积后每帧只执行一次 `scaleBy`，
消灭过期事件的废功重算。跨平台受益（Windows 收益更大，因其无原生事件合并）。

### 2. 原理

**问题**：Windows 消息循环逐个派发 `WM_MOUSEWHEEL`，连续滚动时事件成串到达。
当单事件处理耗时（30-50ms）超过事件间隔时，事件在队列堆积，**每个过期事件仍触发
完整的 N×M 重算+重绘**——全是废功。macOS 有原生事件合并，废功少得多。

**方案**：用 `QTimer(PreciseTimer, interval=16ms)` 做合并器：
- `wheelEvent` 只累积 factor（连乘）+ 记录最后一次鼠标位置
- Timer 到期时一次性 `scaleBy` 累积 factor，center 用最后一次鼠标位置
- 采用**节流带尾沿**模式（窗口内只启动一次、到期 flush），而非 restart 式防抖
  （restart 式在高频滚动时 timer 被无限重启，缩放完全不跟随）

**为什么用 PreciseTimer**：CoarseTimer 在 Windows 上有 ±15.6ms 量化，合并间隔会
抖到 16~31ms；PreciseTimer 配合 timeBeginPeriod(1)（仅 Windows 需要）可达 ~1ms 精度。

### 3. 修改文件

#### 3.1 `src/core/config.py` — 新增滚轮合并常量

在现有 PERF 常量之后追加：

```python
PERF_WHEEL_COALESCE_INTERVAL_MS = 16  # 滚轮合并器间隔（~60Hz）
```

#### 3.2 `src/ui/widgets/plot_widget.py` — 重写 wheelEvent

**当前代码**（L482-510）：直接 `vb.scaleBy()`，无合并。

**替换为**：

```python
from PySide6.QtCore import QTimer, Qt

# 在 __init__ 中创建合并器（在 self.factor = 1.0 之后）：
        self._wheel_coalesce_timer = QTimer(self)
        self._wheel_coalesce_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._wheel_coalesce_timer.setInterval(16)  # ~60Hz
        self._wheel_coalesce_timer.setSingleShot(True)
        self._wheel_coalesce_timer.timeout.connect(self._flush_wheel_zoom)
        self._wheel_accumulated_factor = 1.0
        self._wheel_last_mouse_x = 0.0
        self._wheel_last_mouse_y = 0.0
        self._wheel_coalesced_count = 0

# wheelEvent 重写：
    def wheelEvent(self, ev):
        vb = self.plot_item.getViewBox()
        delta = ev.angleDelta().y()
        if ev.modifiers() == Qt.KeyboardModifier.NoModifier:
            if delta != 0:
                # 获取鼠标位置
                mouse_pos = ev.position().toPoint()
                scene_pos = self.mapToScene(mouse_pos)
                view_pos = vb.mapSceneToView(scene_pos)
                mouse_x = view_pos.x()
                mouse_y = view_pos.y()

                factor = max(0.000001, 1 - FACTOR_SCROLL_ZOOM) if delta > 0 else (1 + FACTOR_SCROLL_ZOOM)

                # 累积 factor（连乘）+ 记录最后一次鼠标位置
                self._wheel_accumulated_factor *= factor
                self._wheel_last_mouse_x = mouse_x
                self._wheel_last_mouse_y = mouse_y
                self._wheel_coalesced_count += 1

                # 启动合并器（只启动一次，到期 flush）
                if not self._wheel_coalesce_timer.isActive():
                    self._wheel_coalesce_timer.start()

                ev.accept()
            else:
                super().wheelEvent(ev)
        else:
            super().wheelEvent(ev)

    def _flush_wheel_zoom(self):
        """合并器到期：一次性执行累积的缩放"""
        vb = self.plot_item.getViewBox()
        factor = self._wheel_accumulated_factor
        mouse_x = self._wheel_last_mouse_x
        mouse_y = self._wheel_last_mouse_y
        count = self._wheel_coalesced_count

        # 复位累积状态
        self._wheel_accumulated_factor = 1.0
        self._wheel_coalesced_count = 0

        if PERF_LOG_ENABLED:
            _t0 = time.perf_counter()
        vb.scaleBy((factor, 1), center=(mouse_x, mouse_y))
        if PERF_LOG_ENABLED:
            _dt_ms = (time.perf_counter() - _t0) * 1000
            if _dt_ms > PERF_WHEEL_WARN_MS:
                logger.warning(
                    "[PERF][WHEEL] flush scaleBy: %.2fms (factor=%.4f, coalesced=%d, center=(%.2f, %.2f))",
                    _dt_ms, factor, count, mouse_x, mouse_y,
                )
```

**关键设计决策**：
- **factor 连乘**而非 delta 累加：多次缩放的 factor 相乘等价于连续缩放
- **center 用最后一次鼠标位置**：保证缩放锚点是用户最终的意图位置
- **flush 前重新 mapToScene/mapToView**：不直接复用 wheelEvent 缓存的 view 坐标，
  因为 plot 之间的 XLink 联动可能导致坐标失真（在 flush 时重新计算）

### 4. 性能测量点

- `[PERF][WHEEL] flush scaleBy` 中的 `coalesced=N` 字段：N>1 说明合并生效，消灭了 N-1 次废功
- `[PERF][FRAME]` fps 变化：预期从 ~4fps 提升到 ~10-15fps
- `[PERF][PAINT]` 频率降低：paint 次数减少，但单次耗时不变

### 5. 回滚策略

- 恢复 `wheelEvent` 为直接 `vb.scaleBy()` 版本
- 删除 `_flush_wheel_zoom` 方法和合并器相关属性
- 删除 `_wheel_coalesce_timer` 初始化代码

### 6. 风险与注意事项

- **缩放锚点精度**：flush 时 center 用最后一次鼠标位置，与逐事件缩放略有差异，
  但用户感知上更接近"缩放跟随鼠标"
- **PreciseTimer 在 macOS 上无额外收益**：macOS 定时器精度本身就是 ~1ms，
  但合并器本身（消灭废功）跨平台受益
- **与 `_interaction_timer` 的关系**：两者独立，`_interaction_timer`(50ms) 控制
  样式/光标刷新防抖，`_wheel_coalesce_timer`(16ms) 控制 scaleBy 合并

### 7. 预期改善

- 消除过期滚轮事件的废功重算（Windows 收益最大）
- 重算频率钳制在 ≤60Hz（16ms 间隔）
- 预期整体交互流畅度提升 **40-60%**（跨平台）

---

## Phase 2：交互期间禁用 Y autoVisible（文档 §7 方案 C）

### 1. 目标

在用户交互（拖拽/缩放）期间，临时禁用所有可见 plot 的 Y 轴 autoVisible，
避免每次 X 范围变化触发全数据 Y 轴扫描。交互结束后恢复并执行一次最终 Y auto-range。

### 2. 原理

`setAutoVisible(x=False, y=True)` 意味着每次 viewRange 变化时，pyqtgraph 自动
扫描所有数据的 Y 值来调整 Y 轴范围。在交互期间（快速连续的 wheel/drag 事件），
这导致每个事件都触发 N_plot × M_curve 的全数据 Y 扫描。

禁用后，Y 轴范围保持不变（用户最后一次 auto-range 的结果），直到交互结束再一次性更新。

### 3. 修改文件

#### 3.1 `src/ui/widgets/event_handler.py` — 实现 `_start_interaction()`

**目标函数**：`_start_interaction()` (L140-147)

**当前代码**：
```python
    def _start_interaction(self):
        """..."""
        pass
```

**替换为**：
```python
    def _start_interaction(self):
        """交互开始时禁用所有可见 plot 的 Y autoVisible 以减少重算开销。

        交互期间 Y 轴范围冻结（保持上一次 auto-range 的结果），
        避免每次 X 范围变化触发全数据 Y 扫描。
        交互结束后由 _end_interaction 恢复并执行一次最终 Y auto-range。
        """
        import time as _time
        from src.core.config import PERF_LOG_ENABLED

        _t0 = _time.perf_counter() if PERF_LOG_ENABLED else 0

        pw = self.pw
        main_window = pw.window() if hasattr(pw, 'window') else None
        if main_window is None or not hasattr(main_window, 'plot_widgets'):
            return

        count = 0
        for container in main_window.plot_widgets:
            sibling = container.plot_widget
            if not container.isVisible():
                continue
            if not hasattr(sibling, 'view_box'):
                continue
            # 记录原始 autoVisible 状态（用于 _end_interaction 精确恢复）
            if not hasattr(sibling, '_pre_interaction_auto_visible'):
                sibling._pre_interaction_auto_visible = None
            try:
                # 读取当前 autoVisible 状态（优先使用 state 字典，兼容性更好）
                try:
                    auto_state = sibling.view_box.autoRangeEnabled()
                except (AttributeError, TypeError):
                    # fallback: 从内部 state 字典读取
                    auto_state = sibling.view_box.state.get('autoRange', {})
                sibling._pre_interaction_auto_visible = auto_state
                # 禁用 Y autoVisible，保持 X autoVisible 不变（X 本来就是 False）
                sibling.view_box.setAutoVisible(x=False, y=False)
                count += 1
            except Exception:
                logger.debug(
                    "[INTERACT] setAutoVisible(y=False) failed for plot",
                    exc_info=True,
                )

        if PERF_LOG_ENABLED:
            _dt_ms = (_time.perf_counter() - _t0) * 1000
            logger.info(
                "[PERF][START_INTERACT] disabled Y autoVisible on %d plots in %.2fms",
                count, _dt_ms,
            )
```

**关键设计决策**：
- 遍历 `main_window.plot_widgets`（所有 plot），不仅限于交互源 plot
- 原因：XLink 级联会将 range 变化传播到所有 linked plot，每个都会触发 Y auto-range
- 保存 `_pre_interaction_auto_visible` 以支持精确恢复

#### 3.2 `src/ui/widgets/event_handler.py` — 修改 `_end_interaction()`

**目标函数**：`_end_interaction()` (L149-160)

**当前代码**：
```python
    def _end_interaction(self):
        """结束交互时的处理，并广播刷新到所有 XLink 兄弟子图"""
        try:
            self._is_interacting = False
            self._queue_ui_refresh(immediate=True)
            self._refresh_siblings_after_interaction()
            if getattr(self.pw, '_pending_cursor_geometry_update', False):
                self.pw._pending_cursor_geometry_update = False
                self._schedule_cursor_geometry_update()
        except Exception:
            logger.warning("结束交互出错", exc_info=True)
```

**替换为**：
```python
    def _end_interaction(self):
        """结束交互时的处理：恢复 Y autoVisible → 触发一次 Y auto-range → 广播刷新"""
        import time as _time
        from src.core.config import PERF_LOG_ENABLED, PERF_INTERACTION_WARN_MS

        _t0 = _time.perf_counter() if PERF_LOG_ENABLED else 0
        try:
            self._is_interacting = False

            # Step 1: 恢复所有 plot 的 Y autoVisible 并执行一次最终 Y auto-range
            self._restore_y_auto_visible_and_range()

            # Step 2: 标准交互结束刷新
            self._queue_ui_refresh(immediate=True)
            self._refresh_siblings_after_interaction()

            if getattr(self.pw, '_pending_cursor_geometry_update', False):
                self.pw._pending_cursor_geometry_update = False
                self._schedule_cursor_geometry_update()
        except Exception:
            logger.warning("结束交互出错", exc_info=True)
        finally:
            if PERF_LOG_ENABLED:
                _dt_ms = (_time.perf_counter() - _t0) * 1000
                if _dt_ms > PERF_INTERACTION_WARN_MS:
                    logger.warning(
                        "[PERF][END_INTERACT] slow _end_interaction: %.2fms",
                        _dt_ms,
                    )
```

#### 3.3 `src/ui/widgets/event_handler.py` — 新增 `_restore_y_auto_visible_and_range()`

**位置**：在 `_end_interaction()` 方法之后、`_refresh_siblings_after_interaction()` 之前插入。

```python
    def _restore_y_auto_visible_and_range(self):
        """恢复所有可见 plot 的 Y autoVisible 并执行一次 Y 轴 auto-range。

        与 _start_interaction 配对：交互结束后统一恢复 Y autoVisible(y=True)，
        并对每个 plot 触发一次 enableAutoRange(Y) 以适配新的 X 范围下的 Y 值。
        """
        pw = self.pw
        main_window = pw.window() if hasattr(pw, 'window') else None
        if main_window is None or not hasattr(main_window, 'plot_widgets'):
            return

        count = 0
        for container in main_window.plot_widgets:
            sibling = container.plot_widget
            if not container.isVisible():
                continue
            if not hasattr(sibling, 'view_box'):
                continue
            try:
                # 恢复 Y autoVisible
                sibling.view_box.setAutoVisible(x=False, y=True)
                # 触发一次 Y auto-range（仅 Y 轴，X 保持手动）
                sibling.view_box.enableAutoRange(axis=sibling.view_box.YAxis, enable=True)
                # 清理临时状态
                if hasattr(sibling, '_pre_interaction_auto_visible'):
                    del sibling._pre_interaction_auto_visible
                count += 1
            except Exception:
                logger.debug(
                    "[INTERACT] restore Y autoVisible failed for plot",
                    exc_info=True,
                )

        logger.debug(
            "[INTERACT] restored Y autoVisible on %d plots", count,
        )
```

### 4. 传播机制

```
用户 wheel/drag 事件
  → _on_range_changed 检测到 !self._is_interacting
    → self._is_interacting = True
    → _start_interaction()
      → 遍历 main_window.plot_widgets
        → 每个可见 plot: setAutoVisible(x=False, y=False)
  → 后续事件：Y autoVisible 已禁用，不触发 Y 扫描
  → 防抖定时器超时
    → _end_interaction()
      → _restore_y_auto_visible_and_range()
        → 每个可见 plot: setAutoVisible(x=False, y=True) + enableAutoRange(Y)
      → _queue_ui_refresh(immediate=True)  — 触发 style/cursor/stats 刷新
      → _refresh_siblings_after_interaction()  — 广播兄弟 plot 刷新
```

### 5. 性能测量点

在 `_start_interaction` 和 `_restore_y_auto_visible_and_range` 中已有 `[PERF]` 日志。
对比指标：
- `[PERF][WHEEL]` 在 Phase 1 前后的 p50/p95 变化
- `[PERF][PAINT]` 在 Phase 1 前后的 p50/p95 变化
- `[PERF][START_INTERACT]` 的开销（预期 < 1ms）

### 6. 回滚策略

- 将 `_start_interaction()` 恢复为 `pass`
- 将 `_end_interaction()` 恢复为原始版本（去掉 `_restore_y_auto_visible_and_range` 调用）
- 删除 `_restore_y_auto_visible_and_range()` 方法
- 所有改动集中在 `event_handler.py` 一个文件，回滚影响面极小

### 7. 风险与注意事项

- **Y 轴范围冻结**：交互期间 Y 轴不跟随数据变化，用户可能看到曲线超出 Y 轴可视范围。
  这是预期行为，交互结束后立即恢复。
- **`autoRangeEnabled()` API 兼容性**：pyqtgraph 0.14.0 的 ViewBox 可能不直接暴露 `autoRangeEnabled()` 方法。
  备选方案：使用 `vb.state['autoRange']` 读取内部状态（dict 格式 `{0: x_state, 1: y_state}`）。
  代码中应做 try/except 保护，fallback 到直接 `setAutoVisible(y=False)` 而不保存原始状态。
- **首次交互**：`_pre_interaction_auto_visible` 在首次交互前不存在，
  代码已用 `hasattr` 保护。

### 8. 预期改善

- 交互期间单次 wheel 事件代价降低约 **30-60%**（消除 Y 轴全数据扫描）
- 对 12 plots × 多曲线场景效果更显著
- `_end_interaction` 会有一次性 Y auto-range 开销，但仅发生一次

---

## Phase 3：自定义批量 XLink（高级/可选优化）

> **前置条件**：Phase 1 + Phase 2 已实施且验证通过。若前两项已满足性能目标，可跳过本阶段。

### 1. 目标

替换 pyqtgraph 原生的 `setXLink()` 级联机制，改用应用层批量同步，
将 N 个 plot 的同步从「每事件逐 plot 级联」改为「16ms 批次统一同步」。

### 2. 原理

pyqtgraph 原生 XLink 的工作方式：
- master viewRange 变化 → 触发 linkedViewChanged 信号
- 每个 slave 收到信号后独立重算 viewRange（考虑自身几何尺寸）
- slave 的 viewRange 变化又触发其自身的 sigRangeChanged
- 形成 N-1 次同步回调链

自定义批量 XLink：
- 移除所有 `setXLink()` 调用
- 仅监听 master plot 的 `sigRangeChanged`
- 使用 16ms PreciseTimer 将同步请求批处理
- Timer 触发时一次性遍历所有 slave plot 执行 `setXRange()`
- 同步期间使用 `_is_syncing_range` 标志抑制 slave 的回调

### 3. 修改文件

#### 3.1 新建 `src/ui/widgets/batched_xlink.py` — BatchedXLinkSync 管理器

```python
"""
BatchedXLinkSync - 批量 XLink 同步管理器

替代 pyqtgraph 原生 setXLink 级联机制，
使用 16ms PreciseTimer 将多个同步请求批处理为一次批量操作。
"""

from __future__ import annotations
from typing import Any

from PySide6.QtCore import QTimer

from src.core.config import PERF_LOG_ENABLED
from src.core.logger import get_logger

logger = get_logger(__name__)


class BatchedXLinkSync:
    """批量 XLink 同步管理器

    生命周期：
    - 在 layout_manager.create_subplots_matrix 中创建
    - 绑定 master plot 的 sigRangeChanged
    - 在布局变化、可见性变化时更新 slave 列表
    - 在窗口销毁时调用 dispose()
    """

    BATCH_INTERVAL_MS = 16  # ~60fps 同步频率

    def __init__(self):
        self._master_vb = None
        self._slave_pws: list[Any] = []  # slave plot_widget 列表
        self._timer = QTimer()
        self._timer.setTimerType(2)  # PreciseTimer
        self._timer.setInterval(self.BATCH_INTERVAL_MS)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._flush_sync)
        self._pending = False
        self._disposed = False

    def setup(self, master_vb, slave_pws: list[Any]):
        """设置 master ViewBox 和 slave plot_widget 列表"""
        self.dispose()
        self._master_vb = master_vb
        self._slave_pws = list(slave_pws)
        master_vb.sigRangeChanged.connect(self._on_master_range_changed)

    def update_slaves(self, slave_pws: list[Any]):
        """更新 slave 列表（布局变化后调用）"""
        self._slave_pws = list(slave_pws)

    def _on_master_range_changed(self, view_box, range, changed=None):
        """master range 变化时调度批量同步"""
        if self._disposed:
            return
        self._pending = True
        if not self._timer.isActive():
            self._timer.start()

    def _flush_sync(self):
        """Timer 触发：批量同步所有 slave plot 的 X 范围"""
        if not self._pending or self._disposed:
            return
        self._pending = False

        if self._master_vb is None:
            return

        import time as _time
        _t0 = _time.perf_counter() if PERF_LOG_ENABLED else 0

        try:
            x_range = self._master_vb.viewRange()[0]
        except Exception:
            return

        xmin, xmax = x_range
        if xmin is None or xmax is None or abs(xmax - xmin) < 1e-12:
            return

        synced = 0
        for pw in self._slave_pws:
            if not hasattr(pw, 'view_box') or not hasattr(pw, 'isVisible'):
                continue
            container = pw.parentWidget()
            if container and not container.isVisible():
                continue
            try:
                pw._is_syncing_range = True
                vb = pw.view_box
                linked = vb.linkedView(0)
                if linked is not None:
                    vb.setXLink(None)
                try:
                    vb.enableAutoRange(x=False)
                    vb.setXRange(xmin, xmax, padding=0)
                finally:
                    if linked is not None:
                        vb.setXLink(linked)
                synced += 1
            except Exception:
                logger.debug("[BATCHED_XLINK] sync failed for plot", exc_info=True)
            finally:
                pw._is_syncing_range = False

        if PERF_LOG_ENABLED:
            _dt_ms = (_time.perf_counter() - _t0) * 1000
            logger.info(
                "[PERF][BATCHED_XLINK] synced %d slaves in %.2fms",
                synced, _dt_ms,
            )

    def dispose(self):
        """释放资源，断开信号连接"""
        self._disposed = True
        self._timer.stop()
        if self._master_vb is not None:
            try:
                self._master_vb.sigRangeChanged.disconnect(self._on_master_range_changed)
            except Exception:
                pass
        self._master_vb = None
        self._slave_pws.clear()
```

#### 3.2 `src/ui/layout_manager.py` — 修改 `create_subplots_matrix()`

**目标位置**：L638-641 的 XLink 建立逻辑

```python
# 原始代码:
#     if c == 0 and r == 0:
#         first_viewbox = plot_widget.view_box
#     else:
#         plot_widget.view_box.setXLink(first_viewbox)

# 替换为：
                if c == 0 and r == 0:
                    first_viewbox = plot_widget.view_box
                # 不再调用 setXLink —— 由 BatchedXLinkSync 管理
```

在 `create_subplots_matrix()` 末尾追加：
```python
        self._setup_batched_xlink()
```

新增方法：
```python
    def _setup_batched_xlink(self):
        """初始化或重建 BatchedXLinkSync"""
        from src.ui.widgets.batched_xlink import BatchedXLinkSync

        if not self.mw.plot_widgets:
            return

        master_vb = self.mw.plot_widgets[0].plot_widget.view_box
        slave_pws = []
        for container in self.mw.plot_widgets[1:]:
            if hasattr(container, 'plot_widget'):
                slave_pws.append(container.plot_widget)

        if not hasattr(self.mw, '_batched_xlink'):
            self.mw._batched_xlink = BatchedXLinkSync()
        self.mw._batched_xlink.setup(master_vb, slave_pws)
```

#### 3.3 `src/ui/layout_manager.py` — 适配其他 XLink 相关方法

以下方法需要移除/替换 `setXLink` / `setXLink(None)` 调用：

| 方法 | 位置 | 修改内容 |
|------|------|----------|
| `_sync_linked_x_ranges()` | L101-218 | 移除 `vb.setXLink(master_vb)` 的 link 重建逻辑，改为调度 `_batched_xlink.update_slaves()` |
| `_sync_xlink_after_visibility_change()` | L651-667 | 将 `setXLink` / `setXLink(None)` 替换为 `_batched_xlink.update_slaves()` |
| `_adjust_stretch_and_range()` | L817-853 | 移除 `setXLink(None)` / `setXLink(linked)` 的临时断开/恢复逻辑 |

**注意**：`axis_manager.py` 中 `set_xrange_with_link_handling()` 的 unlink 模式**必须保留**，
它是独立的 setXRange 保护，与 XLink 级联无关。

### 4. 性能测量点

- `[PERF][BATCHED_XLINK]`：批量同步耗时（预期 < 5ms for 11 slaves）
- 对比 `[PERF][WHEEL]` 在 Phase 3 前后的变化
- 对比 `[PERF][RANGE_CB]` 在 Phase 3 前后的变化（slave 不再触发回调）

### 5. 回滚策略

- 移除 `batched_xlink.py` 文件
- 恢复 `create_subplots_matrix()` 中的 `setXLink(first_viewbox)` 调用
- 恢复 `_sync_linked_x_ranges()` 中的 link 重建逻辑
- 恢复 `_sync_xlink_after_visibility_change()` 中的 setXLink/setXLink(None)
- 恢复 `_adjust_stretch_and_range()` 中的 unlink/restore 逻辑

### 6. 风险与注意事项

- **高风险改动**：XLink 是核心联动机制，改动影响面广
- **几何差异**：不同宽度的 plot 在原生 XLink 下会按像素比例计算 range，
  批量同步使用 master 的 range 直接 setXRange，可能导致不同宽度 plot 的 X 范围不一致
  - 缓解：在 `_flush_sync` 中对每个 slave 使用 `padding=0` + `enableAutoRange(x=False)`
- **`_is_syncing_range` 标志**：已被 `_on_range_changed` 用于短路判断（L89-94），
  批量同步期间设置此标志可正确抑制 slave 的回调

### 7. 预期改善

- 单次 wheel 事件的同步开销从 O(N) 次独立回调降低为 1 次批量操作
- 消除 slave plot 的重复 sigRangeChanged 触发链
- 预期整体交互延迟降低 **40-70%**（结合 Phase 1 + Phase 2）

---

## 测试步骤（Phase 1 + Phase 2 + Phase 3 联合验证）

### 测试 1：Baseline 数据采集（当前 dev/perf-logging 分支）

**目的**：建立优化前的性能基线。

1. 启动应用，加载大 CSV（百万行级）
2. 配置 4×3 布局（12 plots），每个 plot 加载多条曲线
3. 执行滚轮缩放操作（连续滚动 5-10 秒）
4. 收集日志中的以下指标：

| 指标 | 日志标签 | 记录内容 |
|------|----------|----------|
| 单 plot paint 耗时 | `[PERF][PAINT]` | p50/p95 值 |
| scaleBy 耗时 | `[PERF][WHEEL]` | p50/p95 值 |
| 帧间隔/fps | `[PERF][FRAME]` | avg interval, fps |
| wheel-to-paint 延迟 | `[PERF][WHEEL_LATENCY]` | 最大值 |
| range 回调耗时 | `[PERF][RANGE_CB]` | p50/p95 值 |
| 交互结束耗时 | `[PERF][END_INTERACT]` | 最大值 |

5. 记录 baseline 数据到表格

### 测试 2：Phase 1 验证（滚轮合并）

**目的**：验证滚轮合并消灭废功的效果。

1. 在 Phase 1 分支上重复测试 1 的操作
2. 重点观察：
   - `[PERF][WHEEL] flush scaleBy` 中 `coalesced=N` 的 N 值（应 > 1）
   - `[PERF][FRAME]` 的 fps 是否提升
   - `[PERF][WHEEL_LATENCY]` 是否减少
3. 体感验证：连续滚轮缩放是否更流畅
4. 功能验证：
   - 缩放锚点是否跟随鼠标位置
   - 缩放方向是否正确（向上放大、向下缩小）
   - 停止滚动后是否立即稳定（无拖尾）

### 测试 3：Phase 2 验证（关 Y autoVisible）

**目的**：验证 Y autoVisible 禁用/恢复的正确性。

1. 在 Phase 1 + Phase 2 分支上重复测试 1 的操作
2. 重点观察：
   - 交互期间 Y 轴范围是否冻结（曲线可能暂时超出 Y 轴范围——预期行为）
   - 交互结束后 Y 轴是否自动适配到正确范围
   - `[PERF][WHEEL]` 和 `[PERF][PAINT]` 的 p50/p95 是否下降
3. 功能验证：
   - 12 plots 全部正确恢复 Y autoVisible
   - 光标位置在交互后正确更新
   - 非交互场景（auto_range、setXRange）不受影响
   - 多次连续交互无状态泄漏

### 测试 4：Phase 3 验证（批量 XLink，可选）

**目的**：验证批量 XLink 替换原生 setXLink 后的同步正确性。

1. 在 Phase 1 + Phase 2 + Phase 3 分支上重复测试 1 的操作
2. 重点观察：
   - `[PERF][BATCHED_XLINK]` 批量同步耗时（预期 < 5ms）
   - `[PERF][RANGE_CB]` slave plot 不再触发回调
   - `[PERF][FRAME]` fps 是否进一步提升
3. 功能验证：
   - 所有 plot 的 X 轴范围保持同步
   - 不同宽度 plot 的 X 范围一致性（关键！）
   - 布局切换（1×1 → 4×3 → 2×2）后同步正常
   - 可见性切换后同步正常
   - 数据重载后 XLink 正常重建
   - `_sync_linked_x_ranges` 健康检查仍有效

### 测试 5：Phase 1 + Phase 2 联合效果对比

**目的**：量化两项优化的叠加收益。

| 指标 | Baseline | Phase 1 | Phase 1+2 | 改善幅度 |
|------|----------|---------|-----------|----------|
| fps | ___ | ___ | ___ | ___% |
| paint p50 | ___ | ___ | ___ | ___% |
| paint p95 | ___ | ___ | ___ | ___% |
| wheel p50 | ___ | ___ | ___ | ___% |
| wheel p95 | ___ | ___ | ___ | ___% |
| WHEEL_LATENCY max | ___ | ___ | ___ | ___% |

### 测试 6：Phase 1 + Phase 2 + Phase 3 全量效果对比

**目的**：量化三项优化的叠加收益。

| 指标 | Baseline | Phase 1 | Phase 1+2 | Phase 1+2+3 | 改善幅度 |
|------|----------|---------|-----------|-------------|----------|
| fps | ___ | ___ | ___ | ___ | ___% |
| paint p50 | ___ | ___ | ___ | ___ | ___% |
| paint p95 | ___ | ___ | ___ | ___ | ___% |
| wheel p50 | ___ | ___ | ___ | ___ | ___% |
| wheel p95 | ___ | ___ | ___ | ___ | ___% |
| WHEEL_LATENCY max | ___ | ___ | ___ | ___ | ___% |
| BATCHED_XLINK | - | - | - | ___ms | - |

### 测试 7：跨平台验证

**Mac**：
- [ ] 滚轮合并正常工作（coalesced > 1）
- [ ] Y autoVisible 禁用/恢复正常
- [ ] 无视觉异常

**Windows**（如有条件）：
- [ ] 滚轮合并正常工作
- [ ] Y autoVisible 禁用/恢复正常
- [ ] 确认 `timeBeginPeriod(1)` 是否已启用（影响 PreciseTimer 精度）
- [ ] 对比 Mac 和 Windows 的 fps 差异是否缩小

### 测试 8：回归测试

- [ ] 布局切换（1×1 → 4×3 → 2×2）后滚轮缩放正常
- [ ] 可见性切换后滚轮缩放正常
- [ ] 数据重载后滚轮缩放正常
- [ ] 光标拖拽不受影响
- [ ] 框选缩放不受影响
- [ ] 单 plot 场景滚轮缩放正常

---

## 实施顺序与依赖关系

```
Phase 0 (日志基础设施)          ✅ 已完成
  ↓
Phase 1 (滚轮合并节流)           ← 跨平台高收益，消灭废功
  ↓ 验证效果
Phase 2 (禁用 Y autoVisible)     ← 跨平台中收益，砍掉 ~50% 重算成本
  ↓ 验证效果
Phase 3 (批量 XLink)             ← 高级/可选，高风险高收益
  ↓ 联合验证
测试步骤 1-8（Baseline → P1 → P2 → P3 → 联合对比 → 跨平台 → 回归）
```

### 建议实施时间线

| Phase | 预计工时 | 风险等级 | 依赖 |
|-------|----------|----------|------|
| Phase 0 | ✅ 已完成 | 极低 | 无 |
| Phase 1 | 2-3h | 低 | Phase 0 |
| Phase 2 | 2-3h | 低 | Phase 0, Phase 1 |
| Phase 3 | 4-6h | 高 | Phase 0, Phase 1, Phase 2 |
| 测试验证 | 3-4h | - | Phase 1 + Phase 2 + Phase 3 |

---

## 附录 A：pyqtgraph ViewBox 关键 API 参考

```python
# 读取 autoRange 状态
vb.autoRangeEnabled()  # -> (x_enabled, y_enabled) 或 None

# 设置 autoVisible
vb.setAutoVisible(x=False, y=False)

# 启用/禁用 autoRange
vb.enableAutoRange(axis=vb.YAxis, enable=True)

# XLink
vb.setXLink(other_vb)  # 建立 X 轴联动
vb.linkedView(0)       # 获取 X 轴 linked ViewBox，None 表示未链接

# 范围操作
vb.setXRange(xmin, xmax, padding=0)
vb.viewRange()  # -> [(xmin, xmax), (ymin, ymax)]
vb.scaleBy((fx, fy), center=(cx, cy))
```

## 附录 B：验证检查清单

### Phase 0 验证 ✅ 已完成
- [x] 启动应用，加载数据
- [x] 执行 wheel 缩放，检查日志中出现 `[PERF][WHEEL]` 输出
- [x] 执行拖拽平移，检查日志中出现 `[PERF][RANGE_CB]` 输出
- [x] 检查 `[PERF][PAINT]` 在重绘时输出
- [x] 设置 `PERF_LOG_ENABLED = False`，确认无 `[PERF]` 输出

### Phase 1 验证（滚轮合并）
- [ ] `[PERF][WHEEL] flush scaleBy` 中 `coalesced=N` 的 N > 1
- [ ] `[PERF][FRAME]` fps 提升（预期从 ~4fps → ~10-15fps）
- [ ] 缩放锚点跟随鼠标位置
- [ ] 缩放方向正确（向上放大、向下缩小）
- [ ] 停止滚动后立即稳定（无拖尾）
- [ ] 布局切换后滚轮缩放正常
- [ ] 单 plot 场景滚轮缩放正常

### Phase 2 验证（关 Y autoVisible）
- [ ] 交互期间 Y 轴范围冻结（曲线可能暂时超出 Y 轴范围——预期行为）
- [ ] 交互结束后 Y 轴自动适配到正确范围
- [ ] 12 plots 全部正确恢复 Y autoVisible
- [ ] 光标位置在交互后正确更新
- [ ] 非交互场景（auto_range、setXRange）不受影响
- [ ] 多次连续交互无状态泄漏
- [ ] 数据重载后 Y autoVisible 正常

### Phase 3 验证（批量 XLink，可选）
- [ ] 所有 plot 的 X 轴范围保持同步
- [ ] 不同宽度 plot 的 X 范围一致性（关键！）
- [ ] 布局切换（1×1 → 4×3 → 2×2）后同步正常
- [ ] 可见性切换后同步正常
- [ ] `_sync_linked_x_ranges` 健康检查仍有效
- [ ] 无 regression：reload 数据后 XLink 正常重建
- [ ] `[PERF][BATCHED_XLINK]` 同步耗时 < 5ms
