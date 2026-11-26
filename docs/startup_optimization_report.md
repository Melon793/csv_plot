# CSV Plot 启动与 Splash 优化验证报告

日期：2026-11-26
验证分支：`dev/startup-splash-optimization`
基线提交：`e092293`（`chore(version): 0.6.2`）
基线源码：`tmp/startup-baseline`，未修改
优化源码：当前工作区，基于同一提交

## 1. 结论摘要

本次优化把“先完成重导入和主题/字体准备，再显示 splash；splash 显示后同步构造主窗口”的链路，调整为：

1. 尽快显示轻量 splash；
2. splash 可见后，在后台线程预导入 pyqtgraph 和主窗口模块；
3. 回到 UI 线程构造主窗口，但在 9 个构造阶段之间主动让出事件循环；
4. splash 每帧只绘制静态层、呼吸环和进度段；
5. pandas、表格对话框、MDF/BLF 数据加载器等改为真正使用时才导入，`src.core` 公共导出也改为惰性解析；
6. 保留 800ms 最短展示时间、60s 看门狗、启动错误提示和 `--no-splash` 路径；看门狗发现后台导入线程仍在运行时同步保存诊断并安全退出，不再强制终止 Python 线程；
7. splash 静态层在设备像素比变化时自动重建，覆盖窗口跨 DPI 显示器移动的场景。

Python 3.13 + PyInstaller 打包后的默认启动实测中位数：

| 指标 | 基线 | 优化后 | 变化 |
|---|---:|---:|---:|
| 进程创建 → splash 可见 | 624.45 ms | 298.32 ms | **-326.13 ms / -52.2%** |
| splash 可见 → 主窗口可见 | 885.39 ms | 844.10 ms | -41.29 ms / -4.7% |
| 进程创建 → 主窗口可见 | 1515.47 ms | 1140.66 ms | **-374.81 ms / -24.7%** |
| splash 最大像素变化间隔中位数 | 437.52 ms | 63.87 ms | **-373.65 ms / -85.4%** |

结论：优化有效，且默认启动已接近 800ms 最短 splash 展示时长的下限。splash 动画从约每秒 2～3 次像素变化恢复到约 60fps 的常规节奏；内部帧间隔中位约 16ms，但 UI 线程构造阶段仍会造成偶发尖峰，不能把外部采样结果等同于显示器实际刷新率。强制 3 秒场景的每轮最大像素变化间隔中位数为 94.11ms，高于上一轮测量的 63.90ms，说明外部采样和系统调度仍有波动；其内部 paint P95 仅 2.34ms，内部相邻 paint 起始间隔 P95 为 17.38ms。

## 2. 环境与测量方法

### 2.1 运行环境

- Windows 10.0.19045 SP0，x64
- Python 3.13.18
- PyInstaller 6.22.3
- PySide6 6.12.0
- pandas 3.1.1
- NumPy 2.5.5
- pyqtgraph 0.14.0
- 沿用现有 PyInstaller onedir spec，UPX 关闭，未启用 PyInstaller 原生 splash

基线包和优化包均约 188.8MB，包体没有明显缩小，避免把分发体积变化误判为启动收益。最终优化包入口为 `dist-optimized/csv_plot/csv_plot.exe`（目录整体约 188.8MB，EXE 本身约 13.4MB），构建完成时间为 2026-11-26 10:57:19；分发时必须保留整个 `csv_plot` 目录，不能只复制 EXE。`build-optimized/` 和 `dist-optimized/` 已加入 `.gitignore`，不会作为源码变更进入版本控制。

### 2.2 外部 packaged-app 测量

新增 `scripts/measure_startup.py`，不依赖第三方库：

- 通过进程 PID 定位 splash 和主窗口，兼容打包版部分顶层窗口标题为空的情况；
- 记录进程创建、splash 出现、主窗口出现三个时间点；
- 用 Windows GDI 截取 splash 区域，以 CRC32 判断像素是否变化；
- 发送 `WM_CLOSE` 检查正常退出；
- 每轮使用隔离的 `CSV_PLOT_CONFIG_DIR`，同一组内第 1 轮为空配置，后续轮复用配置；
- 优化版通过 `CSV_PLOT_STARTUP_METRICS` 输出内部阶段和 paint 耗时。

最终包重新构建后，又顺序执行了 4 组、每组 6 轮，合计 24 次有效启动，全部检测到主窗口且返回码为 0。两组默认启动的首轮均出现了明显的一次性系统抖动：基线进程→splash 最大 1244.29ms、进程→主窗口最大 2134.82ms，优化版对应 1396.22ms 和 2750.83ms；后续轮次显著回落。因此下文以中位数作为主要比较依据，同时保留完整范围，避免把偶发尖峰误当成稳定收益：

- 基线默认 800ms splash；
- 优化默认 800ms splash；
- 基线强制 3000ms splash；
- 优化强制 3000ms splash。

这里的“cold”仅指空应用配置目录，不代表清除了操作系统文件缓存；不应把结果表述为严格的磁盘冷启动。外部 GDI 采样循环受系统计时器影响，实际中位间隔约 31.2ms，不能直接分辨 16ms 动画帧，因此动画结论同时参考优化版内部 paint/frame 遥测。

## 3. 基线启动短板分析

### 3.1 重工作发生在 splash 之前

原入口在创建 splash 前就导入 pyqtgraph，并准备主题、字体和应用外观。用户双击后要先等待 bootloader、Qt 和这些 Python 模块，才能看到第一张启动图。

### 3.2 主窗口导入链路过重

使用 `python -X importtime -c "import src.ui.main_window"` 得到：

| 模块 | 基线累计导入 | 优化后累计导入 |
|---|---:|---:|
| `src.ui.main_window` | 1464.52 ms | 750.86 ms |
| `src.ui.table_dialog` | 713.98 ms | 未导入 |
| pandas | 511.79 ms | 未导入 |
| `src.data.loader` | 7.48 ms | 未导入 |
| pyqtgraph | 275.68 ms | 422.86 ms |

最终顺序采集的源码导入中，主窗口导入累计耗时下降 **713.66ms / 48.7%**。优化后的源码导入仍包含 pyqtgraph，因为绘图功能需要它；关键变化是它不再阻塞 splash 首帧，而是在 splash 可见后后台加载。单次 `importtime` 会受文件缓存、模块收集顺序和系统负载影响，pyqtgraph 的累计值也不直接代表独立加载成本，因此该表只用于解释依赖链变化，不替代 packaged-app 的多轮测量。

### 3.3 主窗口构造阻塞 UI 线程

原 `MainWindow` 构造函数连续执行基础配置、窗口上下文、数据状态、中央控件、左右面板、状态栏、管理器和命令行参数处理。构造完成后事件循环才能继续处理 splash 定时器。

### 3.4 splash 每帧重复重绘和枚举字体

原 splash 每帧重绘背景、网格、图标、文字和进度，并在 `draw_text()` 中调用系统字体枚举。Windows 字体枚举可耗时数十至上百毫秒，叠加主窗口同步构造，造成明显卡顿。

## 4. 已实施的优化

### 4.1 先显示 splash，再后台预导入

`csv_plot.py` 现在先创建并显示 splash、处理首帧事件，再启动 `QThread` 预导入。后台线程只导入和配置 Python 模块，不创建任何 `QWidget`；只有线程 `finished` 后才排队回 UI 线程构造主窗口。导入异常会记录 traceback 并结束启动；若看门狗超时时导入线程仍未返回，则保存遥测、同步写入崩溃日志并记录 critical 后 `os._exit(1)`，不调用 `QThread.terminate()`，避免破坏 Python 导入锁或留下半初始化模块。

### 4.2 分阶段构造主窗口并让帧

`MainWindow(startup_yield=...)` 在 9 个构造阶段后分别调用 `processEvents(ExcludeUserInputEvents)`。这既保持 Qt 控件只在 UI 线程创建，又让 splash 有机会刷新，同时排除半初始化窗口的用户输入重入。构造期触发的零延迟回调用 `_startup_constructing` 守卫延后，待管理器全部创建后再执行，避免让帧重入半初始化对象。

优化版 12 轮内部遥测的阶段增量中位数：

| 阶段 | 增量中位数 |
|---|---:|
| 后台导入 pyqtgraph | 320.31 ms |
| 主窗口模块剩余导入 | 56.97 ms |
| 应用外观 | 3.41 ms |
| 基础配置 | 2.35 ms |
| 窗口上下文 | 67.02 ms |
| 数据状态 | 0.19 ms |
| 中央控件 | 1.33 ms |
| 左侧面板 | 11.54 ms |
| 右侧面板 | 16.52 ms |
| 状态栏 | 27.02 ms |
| 管理器 | 41.77 ms |
| 命令行参数 | 1.16 ms |

窗口上下文和管理器是后续最值得继续拆分的 UI 线程阶段。

### 4.3 splash 静态层缓存

背景、网格、图标底座和文字在 `showEvent` 中按设备像素比预渲染到 `QPixmap`。每帧只绘制该静态层、呼吸环和进度段。`paintEvent` 会比较当前 DPR 与缓存 DPR，跨显示器导致缩放变化时先重建静态层再绘制。字体对象缓存，Qt 自行处理缺失字体回退，不再枚举系统字体。

定时器改为 `PreciseTimer`、16ms 间隔。内部 paint 遥测：

| 场景 | paint 中位数 | paint P95 | paint 最大值 |
|---|---:|---:|---:|
| 默认 splash | 0.64 ms | 1.95 ms | 10.69 ms |
| 强制 3 秒 splash | 1.23 ms | 2.34 ms | 10.11 ms |

内部相邻 paint 起始时间中位数分别为 16.05ms 和 15.98ms。强制 3 秒场景 P95 为 17.38ms；默认场景 P95 为 39.41ms，因为短暂启动中构造阶段占比更高。paint 最大值仍可能受系统调度影响，不能据此推断每帧都稳定低于该值。

### 4.4 惰性导入与兼容层

- `src.core.__init__` 使用 PEP 562 `__getattr__` 延迟导出 settings、模板和自动保存类；导入 `startup_metrics` 或 `crash_handler` 不再间接初始化 Qt、设置和自动保存模块，同时保留原有包级导入接口；
- `src.data.__init__` 使用 PEP 562 `__getattr__` 延迟导出 loader、MDF/BLF 和 metadata；
- `file_loader_manager` 延迟解析 loader，同时保留模块属性，兼容既有测试 monkeypatch；
- 表格对话框、日志窗口、pandas 类型依赖移至实际使用位置；
- 从重型 `table_dialog` 提取轻量 `DropOverlay`，主窗口拖拽遮罩不再触发 pandas 导入；
- `PlotVariableEditorDialog` 改为模块级 `__getattr__` 延迟导出，既保持启动时不导入，也保留既有 `monkeypatch.setattr(plot_widget, "PlotVariableEditorDialog", ...)` 兼容性；
- 避免在运行时类型注解中引用未加载的 `pd`，保证 `typing.get_type_hints()` 可正常求值。

### 4.5 字体缓存快速路径

字体缓存命中时直接返回缓存字体名，不再重复枚举系统字体。缓存版本升至 2，强制旧配置重新检测一次；后续失效通过缓存版本升级或删除配置处理，缺失字体由 Qt 自然回退。

### 4.6 可观测性

新增标准库实现的 `src/core/startup_metrics.py`。仅当设置 `CSV_PLOT_STARTUP_METRICS` 时记录数据；关闭时只保留可忽略的快速返回开销。遥测写入失败不会改变应用行为。

### 4.7 合并前评审修复

第一轮独立评审给出“有条件合并”，包含 1 个 High、2 个 Medium、2 个 Low；本报告对应的最终包已包含全部修复：

- **High：轻量 metrics 导入触发 core 急切初始化。** `src.core.__init__` 改为 PEP 562 惰性导出。评审复现中 `import src.core.startup_metrics` 约需 254.38ms，并连带加载 QtCore、settings 和 auto-save；修复后单次 `-X importtime` 为 **54.22ms**，且明确未加载 `src.core.settings`、`src.core.auto_save_manager` 和 `PySide6.QtCore`。包级 `from src.core import AppSettings, ConfigKey, PlotConfig` 等原有接口仍可用。
- **Medium：运行时类型注解无法求值。** `PlotDataManager` 不再在注解中引用未导入的 `AxisManager` 和 `pd.Series`，pandas 改为方法内按需导入；新增整类 `typing.get_type_hints()` 回归。
- **Medium：看门狗强制终止导入线程。** 删除 `QThread.terminate()`。后台导入仍在运行时，先保存遥测、同步写崩溃日志、记录 critical，再 `os._exit(1)`；新增 FakeQThread 测试，断言从未调用 `terminate()`。
- **Low：构建产物未忽略。** `.gitignore` 增加 `build-optimized/`、`dist-optimized/`。
- **Low：跨 DPI 静态层不重建。** `paintEvent` 比较当前 DPR 与缓存 DPR，变化时重建；新增 1.0→2.0 DPR 回归。

上述修复共新增 4 个回归用例；`git diff --check` 通过，完整串行回归结果见第 7 节。

## 5. 打包性能对比

### 5.1 默认 800ms splash

| 指标 | 基线中位数 | 优化中位数 | 基线范围 | 优化范围 |
|---|---:|---:|---:|---:|
| 进程 → splash | 624.45 ms | 298.32 ms | 577.54～1244.29 ms | 294.77～1396.22 ms |
| splash → 主窗口 | 885.39 ms | 844.10 ms | 876.58～919.83 ms | 832.08～1354.61 ms |
| 进程 → 主窗口 | 1515.47 ms | 1140.66 ms | 1464.14～2134.82 ms | 1134.42～2750.83 ms |
| 最大像素变化间隔 | 437.52 ms | 63.87 ms | 436.72～562.76 ms | 62.15～92.81 ms |

### 5.2 强制 3000ms splash

强制延长展示时间可以排除“优化后主窗口更早准备好、但被最短展示时长截断”的干扰，更适合验证加载和动画过程。

| 指标 | 基线中位数 | 优化中位数 | 基线范围 | 优化范围 |
|---|---:|---:|---:|---:|
| 进程 → splash | 583.83 ms | 296.53 ms | 574.88～596.37 ms | 292.62～413.69 ms |
| splash → 主窗口 | 3100.64 ms | 3048.78 ms | 3079.44～3110.20 ms | 3041.64～3075.11 ms |
| 进程 → 主窗口 | 3687.18 ms | 3345.63 ms | 3665.41～3694.58 ms | 3342.70～3455.34 ms |
| 最大像素变化间隔 | 437.55 ms | 94.11 ms | 404.97～468.55 ms | 63.33～94.87 ms |

强制场景总耗时下降 341.55ms / 9.3%，主要来自 splash 首帧提前和真实准备工作减少；其中约 287.30ms / 49.2% 是首帧提前，剩余约 54.25ms 是加载/构造节省。3 秒下限掩盖了大部分后续优化，这是预期行为。

### 5.3 动画连续性判断

- 基线默认/强制场景的“每轮最大像素变化间隔”中位数分别为 437.52ms 和 437.55ms，远超外部采样循环的约 31ms，不能仅用采样频率解释；
- 优化默认/强制场景对应中位数为 63.87ms 和 94.11ms，分别下降 85.4% 和 78.5%；强制场景中位数高于上一轮 63.90ms，单轮 6 次采样不足以判定稳定退步，报告保留这次波动范围；
- 优化版各轮变化间隔 P95：默认 32.11～62.10ms，强制 33.93～47.51ms；各轮最大值分别达到 92.81ms 和 94.87ms，说明存在偶发尖峰；
- 内部 paint 的 P95 均不高于 2.34ms，内部相邻 paint 起始间隔中位数为 16.05ms/15.98ms，P95 为 39.41ms/17.38ms；剩余尖峰主要来自 UI 线程上的窗口上下文、管理器等阶段及系统调度，而非 splash 绘制成本。

## 6. 其他潜在方案与取舍

| 方案 | 状态 | 预期收益 | 风险/建议 |
|---|---|---|---|
| 当前惰性导入 + 后台预导入 + 分阶段让帧 | 已实施并实测 | 首帧 -52.2%，总启动 -24.7% | 推荐作为当前默认方案 |
| 当前静态 splash + 字体缓存 | 已实施并实测 | 默认最大像素间隔 -85.4% | 推荐保留 |
| Shell-first：先显示主窗口，窗口上下文/状态栏延后到首帧后 | 未实施 | 可继续减少 UI 线程阻塞和尖峰 | **下一步首选**；需保证恢复布局、状态栏和管理器初始化顺序 |
| PyInstaller 原生 splash | 未实施 | 可覆盖优化版约 210ms 的 bootloader → 应用 splash 首帧空档 | 只覆盖 bootloader 阶段；需验证资源、DPI、退出和错误路径，不建议替代现有错误处理 |
| 将最短 splash 从 800ms 降至 400～600ms | 未实施 | 默认启动可能再减少数百毫秒 | 当前默认已接近 800ms 下限；应先确认低端机器就绪时间和品牌展示需求 |
| PyInstaller `optimize=2` / 预编译字节码 | 未实施 | 可能小幅减少导入成本 | 会去除 docstring/assert，需完整回归，收益未验证 |
| 改为 onefile | 不推荐 | 分发文件更少 | 每次启动需解压完整包，当前 onedir 更适合启动性能 |
| 多进程预导入 | 不推荐 | 理论上可隔离重模块 | Python 导入结果不能方便地传给 GUI 进程，通常得不偿失 |

## 7. 验证与限制

- 评审修复后的最终串行回归：**3566 passed、4 skipped、8 deselected**，耗时 512.85s；相比评审前增加 4 个用例，覆盖 core 惰性导出、整类类型注解求值、看门狗不终止导入线程和 DPR 变化重建。启动优化、真实构造重入、字体缓存、splitter、数据加载、失败恢复和 BLF monkeypatch 兼容性均包含在内；
- 评审前最终回归为 3562 passed、4 skipped、8 deselected；首轮完整回归为 3560 passed、4 skipped、10 failed。其中 2 项由本次惰性导入引入（对话框 monkeypatch 兼容、运行时类型注解），已修复；剩余 8 项在未修改的 main 基线上逐项复现，涉及 Windows 路径/GBK 编码、状态栏字体尺寸、表格滚动锚点和并发文件替换，均为基线既有失败，因此最终回归明确 deselect 这 8 项，而不是修改无关业务代码；
- 优化包 `--no-splash` 路径实测 1/1 可显示主窗口、正常关闭并返回 0，进程创建→主窗口可见 718.71ms；该值仅作功能冒烟，不与默认 splash 路径直接比较；
- 4 组 packaged-app 对比共 24/24 次成功；
- PyInstaller 构建成功；构建日志中的 OpenGL、mypyc 子模块和 jinja2 缺失警告属于既有依赖收集警告，未影响本次启动验证；
- 未在低端 CPU、机械硬盘、多显示器混合 DPI 或杀毒软件实时扫描场景下测试；
- GDI 像素变化不是显示器实际刷新率的直接测量，内部遥测仅在优化版可用，因此不能给出基线 paint 耗时的直接对比；
- 完整回归会创建 GB 级稀疏文件，需保留足够临时磁盘空间；本次测试临时目录已在验证后清理。

## 8. 最终结论

本次采用的“轻量 splash 首帧 → 后台预导入 → UI 线程分阶段构造并让帧 → 静态层动画 + 惰性重依赖”方案已达到预期目标：默认 packaged-app 主窗口可见时间中位数下降 **24.7%**，splash 首帧提前 **52.2%**，最大像素变化间隔中位数下降 **85.4%**。`--no-splash`、启动失败提示、看门狗和正常退出路径均保留；合并前评审发现的 1 High、2 Medium、2 Low 已全部修复并纳入回归。

建议把当前方案作为默认实现。下一步若要继续减少 UI 线程尖峰，优先评估 Shell-first（先显示主窗口，再延后窗口上下文/状态栏），而不是先缩短 800ms 品牌展示时间或改 onefile。PyInstaller 原生 splash 只能覆盖 bootloader 空档，作为后续可选项，不应替代现有 Python 侧错误处理。

## 9. 复现命令

```bash
PY="$PWD/tmp/2026-11-23/venv313/Scripts/python.exe"

# 构建基线（基线 worktree 内）
(
  cd tmp/startup-baseline
  "$PY" scripts/generate_build_info.py
  "$PY" -m PyInstaller --noconfirm --clean \
    --distpath dist-baseline --workpath build-baseline \
    scripts/csv_plot_pyinstaller.spec
)

# 构建优化版
$PY scripts/generate_build_info.py
$PY -m PyInstaller --noconfirm --clean \
  --distpath dist-optimized --workpath build-optimized \
  scripts/csv_plot_pyinstaller.spec

# 默认与强制 splash 对比
$PY scripts/measure_startup.py \
  tmp/startup-baseline/dist-baseline/csv_plot/csv_plot.exe \
  --runs 6 --output tmp/startup-baseline-default-review.json
$PY scripts/measure_startup.py \
  dist-optimized/csv_plot/csv_plot.exe \
  --runs 6 --output tmp/startup-optimized-default-review.json
$PY scripts/measure_startup.py \
  tmp/startup-baseline/dist-baseline/csv_plot/csv_plot.exe \
  --runs 6 --splash-delay 3 --output tmp/startup-baseline-forced-review.json
$PY scripts/measure_startup.py \
  dist-optimized/csv_plot/csv_plot.exe \
  --runs 6 --splash-delay 3 --output tmp/startup-optimized-forced-review.json

# 无 splash 路径功能冒烟
$PY scripts/measure_startup.py \
  dist-optimized/csv_plot/csv_plot.exe \
  --runs 1 --no-splash --output tmp/startup-optimized-no-splash-review.json

# 源码导入剖析与定向回归
(
  cd tmp/startup-baseline
  "$PY" -X importtime -c "import src.ui.main_window" 2> ../importtime_baseline_review.txt
)
$PY -X importtime -c "import src.ui.main_window" 2> tmp/importtime_optimized_review.txt
$PY -X importtime -c "import src.core.startup_metrics" 2> tmp/importtime_metrics_review.txt
$PY -m pytest -q -n 0 tests/component/test_startup_optimization.py \
  tests/component/test_splash_startup_guard.py \
  tests/component/test_plot_widget_type_hints.py \
  tests/unit/core/test_font_family_memo.py \
  tests/component/test_splitter_ready_polling.py

# 完整串行回归（8 个 deselect 均为未修改 main 上已复现的基线失败）
$PY -m pytest -q -n 0 -p no:cacheprovider --basetemp=tmp/pytest-review-fix \
  --deselect tests/component/test_table_dialog_tab_model.py::test_new_tab_aligns_anchor_on_creation \
  --deselect tests/component/test_variable_info_dialog_copy.py::TestFieldCopyButton::test_long_value_shows_key_only_and_never_widens_window \
  --deselect tests/component/test_x_lock_drawer.py::test_open_geometry_footer_and_height \
  --deselect tests/e2e/test_axis_drawer.py::test_axis_drawer_uses_its_own_narrower_width \
  --deselect tests/e2e/test_axis_drawer.py::test_preview_line_is_not_clipped \
  --deselect tests/unit/core/test_auto_save.py::TestSaveLoadRoundtrip::test_concurrent_saves_keep_primary_and_backup_readable \
  --deselect tests/unit/test_build_wrappers.py::test_windows_wrapper_uses_selected_python_module \
  --deselect tests/unit/test_fetch_blf_fixtures.py::test_explicit_lock_can_refresh_manifest
```

原始对比数据位于 `tmp/startup-{baseline,optimized}-{default,forced}-review.json`，无 splash 冒烟数据位于 `tmp/startup-optimized-no-splash-review.json`，导入剖析位于 `tmp/importtime_{baseline,optimized,metrics}_review.txt`；评审修复后的构建与完整回归日志分别位于 `tmp/build-review-fix.log` 和 `tmp/pytest-review-fix.log`。完整回归临时目录已在验证后清理。
