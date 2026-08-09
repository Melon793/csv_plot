# Windows 11 流畅性优化指南

> 适用范围：当前 PySide6 + pyqtgraph 应用（`csv_plot.py`）
> 背景现象：同一应用在 M4 macOS 上明显比 i9-12900K + Windows 11 流畅
> 本文档给出代码级调整 + 系统级设置 + 验证方法，按性价比排序
> 修订记录（2026-08-10）：修正 QSG_INFO、"软件渲染回退"等技术错误；新增应用层优化 §2.5–2.8（rateLimit 首档取 30）
> 修订记录（2026-08-10 增补）：新增 §7 多 plot + 大数据滚轮缩放卡顿专题分析（滚轮事件合并方案）
> 修订记录（2026-08-10 review 落地）：
> - §2.1：修正 timeBeginPeriod "全局生效" 措辞（Win11 22H2+ 起仅对前台进程生效，不再全局影响系统时钟）；新增 `cleanup_windows_performance()` 配对 `timeEndPeriod(1)`
> - §2.2：修正"必须在 PySide6 导入之前生效"的不严谨措辞；入口示例补 aboutToQuit 挂接清理
> - §2.4：paint 计时新增 33.3ms warning 级阈值，便于日志筛选严重掉帧
> - §2.5：新增"前置测量"硬性步骤与 rateLimit 推荐表；新增"两层节流说明"（外层 SignalProxy + 内层 `_cursor_update_throttle`），避免误判
> - §2.6：补充 pinned 状态缓存的三个失效点（cursor_mode 切换 / 单 plot pin 切换 / 数据重载流程），与 project_memory 重载硬约束联动
> - §2.7：线程优先级措辞从"提升优先级"改为"相对提升调度权重"，并说明对已是 HIGH/REALTIME 的后台线程无效
> - §4.1：修复验证脚本在同一进程内重复创建 QApplication 导致崩溃的问题，改为 subprocess 子进程对比
> - §5：新增两条对称坑（"只做 timeBeginPeriod 不提高 rateLimit"、"只调外层不检查内层 throttle"）
> - §7.4 方案 A：补充 QTimer.PreciseTimer + interval=16 的实施机制选型说明

---

## 1. 根因总览（为什么 Mac 更流畅）

| 优先级 | 根因 | 影响 | 修复位置 |
| --- | --- | --- | --- |
| P0 | 应用层：游标鼠标事件被 `SignalProxy(rateLimit=20)` 节流到 20Hz | 游标跟随上限 20fps，拖拽游标（核心交互）直接显得一顿一顿 | 代码：提高 `rateLimit`（见 2.5，须与 timeBeginPeriod 配合） |
| P0 | 应用层：滚轮事件无合并，pyqtgraph 每事件立即全量重算（不可防抖） | 多 plot + 大数据滚轮缩放时，每格滚轮触发 N×M 重算重绘，事件堆积时废功放大，Windows 明显更卡 | 代码：wheelEvent 合并节流（见 §7） |
| P0 | Windows 系统定时器分辨率 ~15.6ms 量化，macOS ~1ms | 50ms 防抖调度等周期性定时器被量化，刷新节奏不均匀，表现为卡顿/延迟感（注：`singleShot(0)` 属立即投递，几乎不受量化影响） | 代码：`timeBeginPeriod(1)` |
| P0 | 位图合成管线：M4 统一内存近零拷贝 vs Windows PCIe 上传 + DWM 合成 | 每次交互重绘额外拷贝与延迟；驱动过旧时 DWM 合成质量下降 | 系统层：驱动 / HAGS / 整数倍缩放 |
| P1 | 12900K 大小核调度：GUI 线程可能在 P 核/E 核间迁移 | 周期性抖动；macOS 的线程 QoS 无此问题 | 代码：电源节流豁免（见 2.1 第 2 项说明）；系统层：高性能电源计划 |
| P1 | 刷新率差异：M4 本多为 120Hz ProMotion，Win 桌面常见 60Hz | 感知流畅度直接差一倍 | 系统层：高刷显示器 / 帧率对齐 |
| P2 | 鼠标轮询率（普通鼠标 125Hz） | 拖拽游标（核心交互）时明显 | 系统层：1000Hz 鼠标 |
| P2 | DPI 分数缩放（125%） | DWM 额外缩放合成开销 | 系统层：100% / 200% 整数倍 |

**核心事实：Qt 6 的 QWidget 绘制在任何平台都是 CPU 光栅化。** 你的应用不使用 GL 路径（`GLViewWidget` 是 3D 专用），因此"开启 GL"无法直接加速 QPainter 画线。可优化的是：定时器精度、窗口合成路径、进程调度、系统合成器。

---

## 2. 代码调整（核心）

### 2.1 新增 Windows 性能初始化函数

**文件：`src/utils/platform_setup.py`**（在文件末尾追加）

```python
def setup_windows_performance() -> list[str]:
    """Windows 专属性能初始化。必须在创建 QApplication 之前调用。

    Returns:
        已应用的措施列表（失败项以 "-failed" 后缀标记，用于启动日志排查）
    """
    applied: list[str] = []
    if sys.platform != "win32":
        return applied

    # 1. 系统定时器分辨率提到 1ms（QTimer 精度从 ~15.6ms 量化提升到 1ms）。
    #    这是 Qt 事件分发器在 Windows 上的已知行为（等待超时按系统时钟
    #    分辨率取整）。
    #    作用域说明（重要）：
    #    - Win11 22H2（2022 起）后，timeBeginPeriod 仅对"调用进程在前台时"
    #      生效，不再全局抬高整个系统的时钟分辨率，也不再显著影响整机功耗；
    #    - Win10 / Win11 21H2 及更早版本仍为全局生效，会禁用部分空闲 C 状态。
    #    因此本调用在所有版本上都是安全的：在新版系统上副作用已被收敛，
    #    在旧版系统上属于"按需付功耗换精度"的合理取舍。
    #    进程退出时建议配对调用 timeEndPeriod(1)（见函数末尾说明）。
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
        applied.append("timeBeginPeriod(1)")
    except Exception:
        applied.append("timeBeginPeriod-failed")

    # 2. 进程电源节流豁免。作用范围说明：前台 GUI 线程默认不会被系统限流，
    #    该 API 主要影响带后台/Eco QoS 标记的线程，防止应用在窗口失焦、
    #    系统"最佳能效"模式等场景下被降速。收益有限，属可选项。
    try:
        import ctypes
        from ctypes import wintypes

        class _PowerThrottlingState(ctypes.Structure):
            _fields_ = [
                ("Version", wintypes.DWORD),
                ("ControlMask", wintypes.DWORD),
                ("StateMask", wintypes.DWORD),
            ]

        PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
        state = _PowerThrottlingState(
            1, PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0
        )
        ok = ctypes.windll.kernel32.SetProcessInformation(
            ctypes.windll.kernel32.GetCurrentProcess(),
            4,  # ProcessPowerThrottling
            ctypes.byref(state),
            ctypes.sizeof(state),
        )
        if ok:
            applied.append("power-throttling-exemption")
        else:
            applied.append("power-throttling-failed")
    except Exception:
        applied.append("power-throttling-failed")

    # 3. （排查用逃生舱）强制 Qt 加载桌面 OpenGL——仅在显式开启时生效。
    #    注意：本应用为纯 QWidget 光栅绘制，不走 GL 渲染路径，该开关对画线
    #    性能几乎没有实际效果，仅用于排除"Qt 加载了软件 GL (opengl32sw.dll)"
    #    这类环境问题；驱动不支持时会导致崩溃/黑屏，不要盲目设置。
    if os.environ.get("CSV_PLOT_FORCE_GL", "") in ("1", "desktop"):
        os.environ.setdefault("QT_OPENGL", "desktop")
        applied.append("QT_OPENGL=desktop")

    return applied


def cleanup_windows_performance() -> None:
    """与 setup_windows_performance 配对的退出清理。

    在 QApplication.aboutToQuit 信号里调用，配对 timeEndPeriod(1)。
    - 长驻 GUI 应用：进程生命周期内不调用也无副作用（进程退出时系统自动回收）；
    - 托盘最小化 / 后台驻留场景：Win11 22H2+ 把 timeBeginPeriod 作用域收到
      "前台进程"，前台态切换会让 1ms 分辨率失效再恢复，行为略复杂——配对
      调用 timeEndPeriod 可让状态切换更干净，作为良好实践推荐实施。
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.winmm.timeEndPeriod(1)
    except Exception:
        pass
```

### 2.2 入口调用 + 启动诊断

**文件：`csv_plot.py`**

第一步：模块级调用（`timeBeginPeriod` 只需在定时器实际触发前生效即可，
与 `PySide6` 模块导入时机本身无关；放模块级只是为了保证在事件循环启动前
完成，是稳妥做法，不是硬性依赖关系）：

```python
from src.utils.platform_setup import setup_platform, setup_windows_performance

ico_path = setup_platform()
_applied_win_measures = setup_windows_performance()
```

第二步：在 `main()` 中创建 `app` 之后输出诊断日志，并挂接退出清理：

```python
    app = QApplication(sys.argv)
    _log_startup_environment()
    # 配对清理 timeBeginPeriod（托盘驻留场景推荐，见 §2.1 cleanup 说明）
    from src.utils.platform_setup import cleanup_windows_performance
    app.aboutToQuit.connect(cleanup_windows_performance)
```

第三步：新增诊断函数（放在 `main()` 上方）：

```python
def _log_startup_environment() -> None:
    """输出启动环境诊断信息，用于定位 Windows 流畅性问题。"""
    from src.core.logger import get_logger
    from PySide6.QtGui import QGuiApplication

    logger = get_logger("startup")
    app = QGuiApplication.instance()
    if app is None:
        return
    screen = app.primaryScreen()
    logger.info(
        "platform=%s refreshRate=%.1fHz dpr=%.2f win_measures=%s",
        app.platformName(),
        screen.refreshRate() if screen else -1.0,
        screen.devicePixelRatio() if screen else -1.0,
        _applied_win_measures,
    )
```

预期日志（Windows 正常情况）：

```
platform=windows refreshRate=144.0Hz dpr=1.00 win_measures=['timeBeginPeriod(1)', 'power-throttling-exemption']
```

排查点：
- `refreshRate` 若为 60 且显示器支持高刷 → 系统层去改刷新率；
- `dpr` 若为 1.25 / 1.5 等小数 → 是分数缩放，改成 100%/200%；
- `win_measures` 为空 → 没进入 win32 分支；含 `-failed` 后缀条目 → 对应 API 调用失败，按条目名定位。

### 2.3 调度器说明（无需改代码，理解即可）

`src/core/scheduler.py` 的 `UnifiedUpdateScheduler` 默认防抖 50ms。在未做 `timeBeginPeriod(1)` 时，Windows 上该定时器实际以 15.6ms 的整数倍触发（46.8ms / 62.4ms），导致刷新节奏不均匀；调用后恢复为接近精确的 50ms。`timeBeginPeriod(1)` 是全局生效的，一处调用即可修复全部周期性定时器。需要澄清：`QTimer.singleShot(0)` 属于立即投递，基本不受定时器量化影响，不靠它改善。另注意：2.5 节的 `SignalProxy` 节流内部同样依赖 QTimer，因此 `rateLimit` 提升必须与本项配合才真正生效。

### 2.4 （可选）渲染耗时诊断

如需量化"每次重绘耗了多少毫秒"，可在 `src/ui/widgets/plot_widget.py` 中**临时**加计时（仅调试用，性能验证完删除）。

⚠️ `DraggableGraphicsLayoutWidget` 已有复杂的 `paintEvent` 重写（数据重载期间跳过绘制防 SIGSEGV、异常捕获），**不要新增同名方法覆盖它**，应在现有方法内对 `super().paintEvent(event)` 调用处包裹计时：

```python
        try:
            import time
            t0 = time.perf_counter()
            super().paintEvent(event)
            dt = (time.perf_counter() - t0) * 1000
            if dt > 33.3:  # 低于 30fps → 严重掉帧，warning 级别便于日志快速筛选
                logger.warning("paint very slow: %.2fms", dt)
            elif dt > 16.0:  # 超过一帧(60fps) → 轻微超帧，debug 级别
                logger.debug("paint slow: %.2fms", dt)
        except RuntimeError as e:
            ...  # 保留原有的异常处理分支
```

说明：此处测得的是 CPU 光栅耗时，**不含 DWM 合成与显示延迟**；做 2.5/2.8 的 A/B 对比时以该日志为准。

对比方法：在同一台 Windows 机器上，分别以「开启 timeBeginPeriod / 关闭」两种状态拖动游标，观察 `paint slow` 日志频率与阈值，即可定位改善幅度。

### 2.5 游标跟随频率：SignalProxy rateLimit（应用层收益最大项）

**现状**：`src/ui/widgets/plot_ui_manager.py` 中 `SignalProxy(..., rateLimit=20, ...)` 把游标跟随上限钳制在 20fps——即使显示器 144Hz、鼠标 1000Hz，拖拽游标（核心交互）也只会以 20Hz 更新，体感一顿一顿。

**前置测量（硬性步骤，非可选）**：先在 [plot_widget.py](file:///Users/xiaolin/CSV_Plot_PySide/src/ui/widgets/plot_widget.py) 的 `mouse_moved` 入口加 `time.perf_counter()` 计时（沿用 §2.4 思路），在目标 Windows 机器 + 目标数据规模下测出单次 `mouse_moved` 的 P95 耗时。根据测量结果决定目标 rateLimit：

| `mouse_moved` P95 单次耗时 | 推荐目标 rateLimit | 理由 |
| --- | --- | --- |
| < 8ms | 60（或更高） | 单次成本远小于 16ms 帧预算，可一步到位 |
| 8–16ms | 30–45 | 接近一帧预算，留余量 |
| > 16ms | 30 | 单次已超一帧，需先做 §2.6 清理再考虑提高 |

**改动**（保守一档示例）：

```python
        pw.proxy = _pg.SignalProxy(
            pw.scene().sigMouseMoved,
            rateLimit=30,  # 原 20：游标跟随上限 20fps → 30fps（先保守一档）
            slot=pw.mouse_moved,
        )
```

**两层节流说明（重要，避免误判）**：当前游标跟随实际有**两层**节流叠加，
修改时必须同时考虑：

1. **外层 `SignalProxy(rateLimit=N)`**：限制 `sigMouseMoved` → `mouse_moved`
   的派发频率（即本节要改的）；
2. **内层自节流**：`plot_widget.py` 中 `_cursor_update_throttle = 0.016`
   （约 60Hz），在 `mouse_moved` 内部对实际刷新做二次丢弃。

因此把外层 `rateLimit` 从 20 提到 30 后，**有效刷新率仍是
`min(30, 内层阈值≈60) = 30Hz`**，受外层约束。若想真正达到 60Hz，必须同时
检查/上调内层 `_cursor_update_throttle`。否则会出现"外层提到 60 但体感
没变化"的假象，误判 SignalProxy 改动无效。

**关键约束**：
- `SignalProxy` 节流内部用 QTimer 实现，Windows 上不做 `timeBeginPeriod(1)` 时按 15.6ms 步进取整，`rateLimit` 提到 64 以上基本无效——**本项必须与 2.1 第 1 项绑定实施**。
- `rateLimit` 提高后每次鼠标移动的处理成本同比放大，应先用 2.4 的计时手段观察 `sync_crosshair` 单次耗时，再决定是否上探 60。
- 若提高后出现事件堆积（处理慢于产生），可追加 `QApplication.setAttribute(Qt.AA_CompressHighFrequencyEvents)` 压缩鼠标移动事件，丢弃中间帧保持实时性。

### 2.6 sync_crosshair 热路径清理

`src/ui/cursor_sync_manager.py` 的 `sync_crosshair` 当前每次鼠标移动都：
1. 全量遍历 plot 检查 `has_pinned_plot` → 建议缓存 pinned 状态，仅在 pin 变化时失效；
2. 对每个可见 plot 无条件调用 `w.vline.setVisible(True)` → 加 `if not w.vline.isVisible()` 判断避免冗余调用。

这两项开销在 `rateLimit` 提高后会被放大，建议与 2.5 一并实施。进一步方向：**vline 位置全速跟随，cursor_label 文本更新（searchsorted + 格式化 + TextItem 重排）走现有 50ms 防抖调度器**，分离两条刷新路径。

**pinned 状态缓存的失效点（必须挂接，否则重载后缓存脏值导致游标错乱）**：
项目数据重载流程有复杂的安全定时器取消与游标恢复机制（见 `project_memory.md`
"Hard Constraints" 与 "Engineering Conventions"），缓存失效必须挂在以下
三处，缺一不可：

1. **`cursor_mode` 切换**（如从"1 free cursor"切到"2 anchored cursors"）：
   pinned 语义改变，缓存必须重建；
2. **单个 plot 的 `is_cursor_pinned` 切换**（pin/unpin 操作）：直接失效该 plot
   的标记位，并触发 `has_pinned_plot` 总标记重算；
3. **数据重载流程**：在 `_begin_data_reload` 取消安全定时器时、以及
   `_post_reload_cursor_refreshing` 防重入完成后，必须显式失效整个缓存——
   重载期间 plot 列表可能增减、cursor 状态被重置，旧缓存值已无意义。

建议实现为 `CursorSyncManager` 上的一个 `_pinned_cache_dirty: bool` 标志 +
`_invalidate_pinned_cache()` 方法，在上述三处调用；`sync_crosshair` 入口
检查 dirty 标志，仅脏时重算 `has_pinned_plot`。

### 2.7 GUI 线程优先级提升（Windows，低成本补充）

在 `setup_windows_performance()` 末尾追加：

```python
    # 4. 相对提升 GUI 线程的调度权重（ABOVE_NORMAL），减少拖拽期间被
    #    后台线程（如数据加载）抢占的概率。注意：本调用不会降低后台线程的
    #    绝对优先级，只是让 GUI 线程在二者都为普通优先级时拿到更多时间片；
    #    若后台线程已是 HIGH/REALTIME，本项无效。ABOVE_NORMAL 是安全值，
    #    切勿使用 REALTIME/TIME_CRITICAL。
    try:
        import ctypes
        handle = ctypes.windll.kernel32.GetCurrentThread()
        ctypes.windll.kernel32.SetThreadPriority(handle, 1)  # THREAD_PRIORITY_ABOVE_NORMAL
        applied.append("thread-priority-above-normal")
    except Exception:
        applied.append("thread-priority-failed")
```

### 2.8 视口更新模式 A/B 实验（收益不确定）

pyqtgraph 的 GraphicsView 默认 SmartViewportUpdate；在"高频拖动一条细线（vline）"的场景下，Windows 上的增量更新有时比整帧重绘更慢且易出残影。可在 `plot_ui_manager.py` 创建 plot 时试验：

```python
        from PySide6.QtWidgets import QGraphicsView
        pw.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
```

收益取决于 plot 数量与尺寸，用 2.4 的计时方法做 A/B：有效保留，无效回退。

---

## 3. 系统层设置（非代码，按顺序执行）

1. **更新显卡驱动**（Intel 官网 / NVIDIA GeForce Experience）
   本应用是纯 QWidget 光栅绘制，不依赖 OpenGL，不存在"Qt 回退软件渲染"问题；但过旧驱动会降低 DWM 合成质量与效率，是"明明配置很好却卡"的常见原因。
2. **硬件加速 GPU 计划（HAGS）**
   `设置 → 系统 → 显示 → 图形 → 更改默认图形设置 → 打开"硬件加速 GPU 计划"`，重启生效。
3. **图形性能首选项**
   同一页面 →「添加应用」→ 选择打包后的 exe（开发期选 `python.exe`）→ 选项 → **高性能**（若有独显）。
4. **高性能电源计划**
   `设置 → 系统 → 电源 → 电源模式 → 最佳性能`。可选工具：Process Lasso 将 GUI 线程固定到 P 核，消除 E 核迁移抖动。
5. **整数倍缩放**
   系统显示缩放设 100% 或 200%，避免 125% 分数缩放。
6. **显示器刷新率**
   `设置 → 系统 → 显示 → 高级显示` 检查是否为显示器最高刷新率（144Hz/165Hz 等）。
7. **1000Hz 鼠标**
   拖拽游标（核心交互）的体感提升最直接。
8. **（可选）针对窗口化游戏的优化**
   `设置 → 系统 → 显示 → 图形 → 更改默认图形设置` 打开，减少 DWM 合成延迟。

---

## 4. 验证方法

### 4.1 确认定时器精度生效

在 Windows 上临时运行（需事件循环，脚本内置前后对比开关）：

> ⚠️ Qt 不允许在同一进程内重复创建 `QApplication`，否则第二次调用会抛
> `RuntimeError` 或触发隐蔽的 paint device 错误。下面的脚本通过 **subprocess
> 子进程**做前后对比，每组在独立进程内只创建一次 `QApplication`。

```python
import subprocess, sys, json

# 单次测量脚本（在子进程内执行）：
MEASURE_SCRIPT = r"""
import sys, json
from PySide6.QtCore import QTimer, QElapsedTimer
from PySide6.QtWidgets import QApplication

high_res = (sys.argv[1] == "1")
if high_res:
    import ctypes
    ctypes.windll.winmm.timeBeginPeriod(1)

app = QApplication([])
intervals = []
et = QElapsedTimer()
remaining = [20]

def tick():
    intervals.append(et.restart())
    remaining[0] -= 1
    if remaining[0] <= 0:
        app.quit()

timer = QTimer()
timer.setInterval(1)
timer.timeout.connect(tick)
et.start()
timer.start()
app.exec()
print(json.dumps(intervals))
"""

def measure(high_res: bool) -> float:
    out = subprocess.check_output(
        [sys.executable, "-c", MEASURE_SCRIPT, "1" if high_res else "0"],
        text=True,
    )
    intervals = json.loads(out)
    return sum(intervals) / max(1, len(intervals))

print(f"开启前（默认分辨率）: {measure(False):.2f}ms")
print(f"开启后（timeBeginPeriod=1）: {measure(True):.2f}ms")
```

- 开启前：约 15.6ms（或 46.8ms 这类整倍数）
- 开启后：接近 1ms

### 4.2 确认 Qt 图形环境

先澄清：本应用是纯 QWidget 光栅应用，`QSG_INFO` / `QSG_RHI_BACKEND` 等 Qt Quick 场景图环境变量**不适用**（无任何输出）。

- 工具方式：运行 `pyside6-qtdiag`（PySide6 安装的可执行文件，位于 venv 的 `Scripts/` 目录下，而非 PySide6 包目录内），查看可用的图形 API 与系统渲染能力。
- 代码方式：启动诊断日志中的 `platform` / `refreshRate` / `dpr` 字段（见 2.2）。
- WARP / llvmpipe 等软件 GL 问题仅在强制 `QT_OPENGL=software` 时才相关，与本应用正常运行无关。

### 4.3 体感对比流程

1. 准备一份大 CSV（百万行级）；
2. 在 Windows 机器上执行「关闭全部优化 → 拖动游标」与「开启全部优化 → 拖动游标」两组操作；
3. 结合 2.4 的 paint 耗时日志与肉眼感受判断；
4. 与 M4 机器做同样操作对照。

---

## 5. 常见坑与不建议做的事

| 操作 | 为什么不建议 |
| --- | --- |
| 给 pyqtgraph 的 GraphicsView 设置 `QOpenGLWidget` 作为 viewport | pyqtgraph 官方不推荐：cosmetic pen（线宽 0 的逻辑画笔）在 GL 下不可靠，会出现画线缺失/模糊 |
| 盲目设置 `QT_OPENGL=desktop` | 本应用不走 GL 渲染路径，对性能几乎无效果；驱动不支持桌面 GL 时还会黑屏/崩溃（见 2.1 第 3 项） |
| 设置 `QSG_INFO` / `QSG_RHI_BACKEND` | 那是 Qt Quick（QML）的开关，纯 QWidget 应用无任何输出 |
| 只提高 `rateLimit` 不做 timeBeginPeriod | Windows 上节流定时器按 15.6ms 量化，提到 64Hz 以上不生效（见 2.5） |
| 只做 timeBeginPeriod 不提高 `rateLimit` | SignalProxy 仍按 20Hz 节流，定时器再准也只 20fps，会误判 timeBeginPeriod 无效（见 2.5，两项须绑定） |
| 只调外层 `rateLimit` 不检查内层 `_cursor_update_throttle` | 有效刷新率受两层节流的 min 值约束，外层提到 60 但内层仍是 0.016(≈60Hz) 时可能无改善，需同步检查（见 2.5 "两层节流说明"） |
| 关闭 DWM 或禁用合成 | Windows 11 不支持，且会引入撕裂 |
| 在热路径（游标拖拽）里做 IO / 分配对象 | 重绘线程阻塞是卡顿主因，应保持绘制路径纯计算 |

**关于 GL 的最终结论**：当前应用是纯 QWidget 光栅绘制，"开 GL" 无法加速画线。真正的收益来源按权重是：`rateLimit` 提升（代码，核心交互，见 2.5）≈ `timeBeginPeriod(1)`（代码）> 驱动/HAGS（系统）> 电源/调度（系统）> 刷新率/鼠标（硬件）。

---

## 6. 故障排查速查表

| 症状 | 可能原因 | 排查/修复 |
| --- | --- | --- |
| 全程卡顿、CPU 占用高 | 定时器量化 或 DWM 合成效率低 | `timeBeginPeriod(1)`；`pyside6-qtdiag` 查环境；更新驱动 |
| 拖游标明显掉帧/一顿一顿 | `rateLimit=20` 节流上限 | 见 2.5，配合 timeBeginPeriod 提高 rateLimit |
| 滚轮缩放卡顿（多 plot + 大数据） | 滚轮事件无合并 + pyqtgraph 每事件立即重算 + 收尾拖尾 | 见 §7：滚轮合并 + timeBeginPeriod |
| 滚轮停下后还要顿一下才稳住 | 交互防抖定时器被量化（收尾拖尾） | `timeBeginPeriod(1)`（见 2.1）；检查 `_end_interaction` 触发延迟（见 §7.3） |
| 周期性抖动（隔一段卡一下） | E 核迁移 / 后台进程限流 | 高性能电源 + 电源节流豁免；Process Lasso 绑 P 核 |
| 帧率够但跟手感差 | 鼠标 125Hz 轮询 | 换 1000Hz 鼠标 |
| 界面模糊、缩放后卡 | 125% 分数缩放 | 改 100%/200% |
| 窗口拖动/缩放卡 | DWM 合成延迟 | 开"针对窗口化游戏的优化"；检查驱动 |
| `win_measures` 为空或含 `-failed` | win32 分支未生效或 API 失败 | 确认 `sys.platform`；失败项以 `-failed` 后缀标记，按条目名定位 |

---

## 7. 专题：多 plot + 大数据滚轮缩放卡顿分析

> 场景：多个可见 plot、每个 plot 含大量数据点，用鼠标中键（滚轮）缩放 X 轴时，Windows 流畅性明显不如 macOS。

### 7.1 一次滚轮事件的完整处理链

```
wheelEvent (delta=±120, factor=1±FACTOR_SCROLL_ZOOM(0.3))
  └→ vb.scaleBy → 源 plot X 范围变化
       └→ pyqtgraph XLink 立即传播到所有已链接的兄弟 plot
            └→ 每个 plot 的每条曲线（PlotDataItem）立即触发 viewRangeChanged：
                 ① clipToView 重新裁剪可见段
                 ② auto 降采样重算 factor + peak 降采样（可见段 numpy max/min 分块）
                 ③ setAutoVisible(y=True) → Y 轴 auto-range 再扫一遍可见数据
                 ④ 全量光栅重绘该 plot
       └→ DWM 合成 N 个 plot 的新位图
  └→ sigRangeChanged → _on_range_changed（应用层防抖，仅抑制样式/光标刷新）
```

**关键点**：应用层防抖（`_interaction_timer` 50ms + XLink 级联抑制，见 `src/ui/widgets/event_handler.py`）只保护了样式/光标刷新；**pyqtgraph 内部的降采样重算 + 重绘是每个事件立即执行、无法防抖的**。单次滚轮事件成本 ≈

```
N(plot) × M(曲线) × (peak降采样 + Y向扫描) + N(plot) × 全幅光栅重绘
```

百万点数据下，peak 降采样每次对可见段（缩到最外层时接近全量）做 numpy 分块 max/min，单曲线就是毫秒级；多 plot × 多曲线相乘，单次事件轻松超过 30–50ms——这一步两个平台都要付。

### 7.2 为什么 Windows 明显更差（平台差异放大器）

**① 滚轮事件无合并，过期事件仍全额执行（最大差异）**
- Windows 消息循环把 `WM_MOUSEWHEEL` 逐个派发，连续滚动时事件成串到达；当单事件处理（30–50ms）超过事件间隔时，事件在队列堆积，**每个过期的堆积事件仍触发完整的 N×M 重算+重绘**——全是废功。
- macOS 对滚动类事件有原生合并（coalescing），过期增量被合并丢弃，废功少得多。
- 当前 `wheelEvent`（`src/ui/widgets/plot_widget.py`）直接 `vb.scaleBy`，**没有任何合并/节流**。

**② 定时器量化让"交互收尾"拖尾**
- 滚轮停止后的收尾全靠 `_interaction_timer` 50ms 防抖 → `_end_interaction` 广播刷新所有兄弟 plot；Windows 上不做 `timeBeginPeriod(1)` 时 50ms 被量化成 46.8/62.4ms，收尾时机抖动；`_schedule_xlink_sync` 的 50ms 健康检查同样被量化。
- 表现为"滚轮停下后还要顿一下才稳住"，Mac 上没有这个拖尾。

**③ 每事件全幅重绘的合成开销差异**
- 每个滚轮事件 → N 个 plot 整幅重绘 → backing store 上传 → DWM 合成；多 plot 布局下每事件 N 次全幅位图上传，macOS Core Animation 图层合成 + 统一内存明显更轻；若叠加 125% 分数缩放，DWM 还要额外做非整数缩放。

**④ Y 轴 autoVisible 是双倍放大器（平台无关）**
- `setAutoVisible(x=False, y=True)`（`src/ui/widgets/plot_ui_manager.py`）使每次 X 变化都对全部曲线再扫一遍可见数据算 Y 范围，并再次触发 sigRangeChanged——上述所有成本约 ×2。

**⚠️ 已知无效的方向**：项目曾尝试"交互期间冻结降采样"（Phase 2A），实测**反而恶化性能已回退**——不要再走这条路。

### 7.3 验证方法（先量化再动手）

在 `wheelEvent` 入口临时加计时（沿用 2.4 的思路），重点看两个指标：
1. **事件间隔 vs 单事件耗时**：若单事件耗时 > 事件间隔 → 证实"事件堆积、废功重算"是主因；
2. **停止滚动到 `_end_interaction` 触发的延迟**：Windows 上若明显 >50ms 且抖动 → 证实定时器量化拖尾。

### 7.4 优化方向对比

| 方案 | 作用 | 收益 | 风险 |
| --- | --- | --- | --- |
| A. wheelEvent 合并节流：累积 delta，每帧（~16ms）只执行一次 `scaleBy` | 把重算频率钳制在 ≤60Hz，消灭过期事件废功 | ★★★★★ 直击主因，跨平台受益 | 低；需保证缩放锚点用最后一次鼠标位置 |
| B. timeBeginPeriod(1)（见 2.1） | 消除收尾拖尾、xlink sync 抖动 | ★★★ 修复"停下后还顿" | 低 |
| C. 交互期间临时关闭 Y 轴 autoVisible | 砍掉约一半重算成本 | ★★★ | 中：交互结束需恢复并触发一次 Y 自适应，时序要小心 |
| D. `_end_interaction` 的 immediate 全量刷新改为下一帧合并 | 削平收尾 burst | ★★ | 低 |
| E. 交互期间提高降采样倍率（临时 ×2） | 降低单次重绘点数 | ★ | 中高：与已回退的 Phase 2A 同类，需谨慎 A/B |

**方案 A 的实施机制选型**：用 `QTimer`（`timerType=Qt.PreciseTimer`，
`interval=16`）做合并器，在 `wheelEvent` 里累积 delta + 记录最后一次鼠标
位置，启动/重启该 timer；timer 超时时一次性 flush（`scaleBy` 累积 factor，
center 用最后一次鼠标位置）。

- **为什么用 `PreciseTimer` 而非默认 `CoarseTimer`**：`CoarseTimer` 在
  Windows 上有 ±15.6ms 的量化（正好是我们要消除的问题），合并间隔会抖到
  16~31ms；`PreciseTimer` 配合 `timeBeginPeriod(1)` 可达 ~1ms 精度，合并
  间隔稳定在 16ms 附近，才能真正把重算频率钳制在 ≤60Hz。
- **为什么不用 `QElapsedTimer` + 手动 busy-wait**：会阻塞事件循环，违背
  合并器"不阻塞、丢弃过期事件"的初衷。
- **flush 时机约束**：timer 超时回调里执行 `scaleBy` 前，必须重新
  `mapToScene`/`mapToView` 计算最新鼠标位置——直接复用 wheelEvent 缓存的
  view 坐标会因为 plot 之间的 XLink 联动而失真。

### 7.5 推荐结论

主因是**"滚轮事件无合并 + pyqtgraph 每事件立即全量重算"在 Windows 上被事件堆积放大，叠加定时器量化拖尾与 DWM 合成开销**。

推荐实施顺序：**A（滚轮合并）+ B（timeBeginPeriod）绑定实施** → 用 7.3 指标观察改善 → 视情况做 C/D。E 不建议优先。
