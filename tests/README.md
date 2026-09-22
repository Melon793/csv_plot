# 测试体系说明（唯一入口）

本文件是 CSV Plot 测试体系的完整文档：快速开始、分层设计、环境隔离机制、新用例编写指南、已知陷阱清单与拓展路线图均集中于此。

## 1. 概览与现状

技术栈：**pytest + pytest-qt + pytest-cov + pytest-xdist + pytest-timeout**，`QT_QPA_PLATFORM=offscreen` 无头运行，三层金字塔组织（unit / component / e2e）。

| 层 | 用例数 | 实测耗时（串行） | 状态 |
|---|---|---|---|
| unit（纯逻辑，无 Qt） | 588 | ~2.8s | 稳定运行 |
| component（offscreen 控件级） | 456 | ~11.5s | 稳定运行 |
| e2e（主窗口冒烟） | 82 | ~11.1s | 稳定运行 |
| **全量** | **1126** | **~25.7s**（并行 `-n 4 --dist loadfile` **9.3s**；带 `--cov` 约 30s） | `1126 passed` |

> 上表数字为 **2026-09-22 在 macOS arm64 / offscreen 实测**（`--durations=20` 口径）。
> 优化前的同日基线是 **1127 例 / 73.5s**：用例数 **+1 −2**（补 1 条 4×3 显式矩阵用例、
> 合并 4 条重复的抽屉门禁用例为 2 条参数化），耗时差的 48s 来自固化等待改造与并行化；
> component 层从 41.6s 降到 11.5s 而用例数不变（只改等待，不删断言），e2e 27.4s → 11.1s。
> 更早的 214 例 / ~4.2s 是 Phase 1 快照，早已过期——改数字时请重跑，别顺手上调。

覆盖率现状（`--cov=src`）：

- 总体 **71%**（15910 语句 / 4671 未覆盖）；
- 已建测试的模块：`main_window` 85%、`excel_loader` 87%、`mdf_lazy_loader` 80%、
  `mark_region_manager` 77%、`font_cache` 67%、`plot_widget` 64%、`layout_manager` 60%、
  `cursor_manager` 55%、`file_loader_manager` 49%、`curve_strategy` 47%、`variable_list` 44%；
- 满分：`template_models` / `plot_config` / `utils.paths` 均 100%；
- 仍是短板的三个：`splash_screen` **15%**（142 语句里 120 未中，启动画面无冒烟用例）、
  `time_correction` **21%**（39 语句里 31 未中，只有抽屉侧的间接覆盖）、
  `curve_strategy` **47%**（32 语句里 17 未中）。
  `cursor_sync_manager` 已从早期快照的 8% 升到 **48%**（460 语句里 240 未中）——旧数字
  是 `cursor_x_domain` 系列用例落地前的，别再照抄。


## 2. 快速开始

```bash
# 1. 安装 dev 依赖（pytest / pytest-qt / pytest-cov 均在 dev 组）
uv sync --group dev
```

命令速查（全部实测可用）：

| 命令 | 用途 | 实测结果 |
|---|---|---|
| `uv run pytest -m unit` | 日常开发高频回归（秒级） | 588 passed / ~2.8s |
| `uv run pytest -m "unit or component"` | 提交前自检 | 1044 passed / ~14.3s |
| `uv run pytest` | CI 全量 | 1126 passed / ~25.7s |
| `uv run pytest -m e2e` | e2e 冒烟（含真实主窗口） | 82 passed / ~11.1s |
| `uv run pytest --cov=src --cov-report=term-missing` | 覆盖率（含未覆盖行号） | 总体 71% |
| `uv run pytest -m component -k legend` | 精准过滤（marker + 关键字组合） | 46 passed / ~0.9s |
| `uv run pytest -m "unit or component" -n 4 --dist loadfile` | 提交前自检（并行） | 1044 passed / ~6.1s |
| `uv run pytest -n 4 --dist loadfile` | CI 全量（并行） | 1126 passed / ~9.3s |

> 注：本机 `uv` 不在 PATH 时使用绝对路径 `~/.local/bin/uv`，或先将其加入 shell 配置。

### 2.1 并行（xdist）与护栏

`pytest-xdist` 与 `pytest-timeout` 已在 dev 组：`uv sync --group dev` 即可。

**并行口径**（2026-09-22 实测：`unit or component` 串行 14.2s → `-n 4 --dist loadfile` 6.1s；
全量串行 25.7s → 9.3s）。选型结论：

- 用 `--dist loadfile`：同文件的用例落在同一 worker，避免"同文件内的用例互相踩环境"
  （`-m unit -n auto` 实测 **无收益**——unit 只有 3.2s，worker 启动与 import 开销吃掉全部节省，
  日常单跑 unit 直接用串行）；
- e2e 并行实测稳定（1126 passed × 多次），未发现 README 陷阱 #3 类的顺序依赖——因为
  `CSV_PLOT_CONFIG_DIR` 由 `tempfile.mkdtemp` 每进程一份，worker 之间本就隔离；
- 本机 10 核（4 性能核）：`-n 4` 与 `-n auto` 的差别在 1s 内，选 `-n 4` 留出余量给
  后台任务，也避免 10 个 worker 争用导致时序敏感用例抖动。

**两道护栏**（`pyproject.toml` 的 `addopts` + `tests/conftest.py`）：

| 护栏 | 行为 |
|---|---|
| `--timeout=120 --timeout-method=thread` | 任何用例超 120s 立即中断并打印全线程栈。历史上"永久挂起"（陷阱 #1/#4/#10）从此是**可诊断失败**而不是静默卡死；实测把 `--timeout=0.05` 加到一个 0.3s 用例上，能准确点名 `tests/component/test_table_dialog_close_destroys.py:54` |
| `--durations=20` + `>1s` 点名 | 每轮收尾都列最慢 20 条；另有 `SLOW_TEST_SECONDS = 1.0` 的钩子（`tests/conftest.py`）把超过 1s 的用例红字点名——当前最慢单例仅 0.30s，留 3× 余量，优化被吃回去时当场可见 |

> 护栏的"点名"钩子在 xdist 下由控制器汇总各 worker 的报告，只打印一次（已实测）。

## 3. 目录结构与分层设计

```
tests/
├── conftest.py                  # 全局环境：offscreen、路径隔离、marker 自动分层、>1s 耗时护栏
├── fixtures/
│   ├── data_factory.py          # 合成数据工厂（make_timeseries / write_csv / write_xlsx / write_mdf）
│   └── waits.py                 # 等待工具：wait_until / settle / flush_deferred_deletes / pump
├── unit/                        # unit 层：无 Qt 依赖的纯逻辑测试
│   ├── core/                    #   config / plot_config / settings / storage /
│   │                            #   template_* / auto_save
│   ├── data/                    #   loader / metadata / excel / var_info …
│   │   ├── conftest.py          #   var_info 共用的合成 loader 夹具（mdf4/mdf3/attr4/attr3/csv）
│   │   └── test_var_info_{mdf,stats}.py   # var_info 用例按 MDF 侧 / 统计侧分文件
│   ├── ui/                      #   status_drawer_reveal
│   └── utils/                   #   paths
├── component/                   # component 层：offscreen + pytest-qt 控件级测试（47 个用例文件）
│   ├── conftest.py              #   plot_factory + legend/表格/变量信息弹窗的共享夹具
│   ├── _vid_shared.py           #   变量信息弹窗的替身主窗口与等待 helper（非用例文件）
│   ├── _tab_mode_shared.py      #   数值表 tab 模式的 loader 替身（非用例文件）
│   └── test_*.py                #   legend / viewbox / 表格 / 游标 / 抽屉 / 生命周期
└── e2e/                         # e2e 层：真实 MainWindow 冒烟（9 个用例文件）
    ├── conftest.py              #   dialog_stubs / main_window / loaded_window 夹具 + 清理竞态过滤
    └── test_*.py                #   启动 / 加载绘图 / 状态栏 / 抽屉 / 窗口几何
```

四层职责与运行策略：

| 层 | 依赖 | 单用例目标耗时 | 运行时机 | marker |
|---|---|---|---|---|
| unit | 无 Qt | < 50ms | 每次保存 / 高频 | `unit` |
| component | offscreen + QApplication | < 1s | 提交前 | `component` |
| e2e | offscreen + 完整主窗口 | < 10s | CI / 发布前 | `e2e` |
| perf | offscreen + benchmark | 秒级 | 按需手动：`uv run pytest -m perf`（默认集已排除，见 pyproject addopts） | `perf` |

**marker 无需手写**：`tests/conftest.py` 的 `pytest_collection_modifyitems` 按用例所在目录（`/unit/`、`/component/`、`/e2e/`、`/perf/`）自动打 marker，新用例只需放入对应目录。

## 4. 环境隔离机制

由 `tests/conftest.py` 实现，要点如下：

1. **offscreen 先于一切 Qt import**：在模块顶层（import pytest 之前）设置
   `QT_QPA_PLATFORM=offscreen` 与 `QT_LOGGING_RULES=qt.qpa.fonts=false`，
   保证无头 CI 环境可运行且屏蔽字体告警。
2. **路径注入**：`CSV_PLOT_CONFIG_DIR` / `CSV_PLOT_LOG_DIR` 指向会话级临时目录
   （`tempfile.mkdtemp(prefix="csv_plot_test_")`），配合 `src/core/settings.py::_get_config_dir()`
   与 `src/core/logger.py::_get_log_dir()` 的环境变量候选链，测试不会读写真实用户配置与日志。
3. **单例隔离**：`app_settings` fixture 在每个用例前后调用 `AppSettings._reset_for_tests()`，
   防止 QSettings 单例状态跨用例泄漏。
4. **会话级 QApplication**：`qapp` fixture 复用全局唯一实例（`QApplication.instance()`），
   pytest-qt 的 `qtbot` 依赖它；不要在用例内另行创建 QApplication。

## 5. 编写新测试指南

### 5.1 unit 层模板

纯逻辑、零 Qt import，数据一律来自合成工厂 `tests/fixtures/data_factory.py`：

```python
# tests/unit/data/test_xxx.py
from tests.fixtures.data_factory import make_simple_rows, write_csv

def test_something(tmp_path):
    csv = write_csv(
        tmp_path / "demo.csv",
        header=["time", "speed", "rpm", "flag"],
        units=["s", "km/h", "rpm", "-"],
        rows=make_simple_rows(20),
    )
    ...
```

工厂能力：`make_timeseries`（固定种子正弦 + 可选窗口外极值注入，用于 Y 轴范围类断言）、
`write_csv`（可控表头/单位行/分隔符/编码/描述行）、`make_simple_rows`（含常量列与非常量列）。
**禁止**依赖 `data/` 下的大文件作为夹具。

### 5.2 component 层模板

使用 `tests/component/conftest.py` 的 `plot_factory` fixture：它会构造可独立绘图的
`DraggableGraphicsLayoutWidget`，并自动注入三层替身：

- `FakePlotContext`：最小 PlotContext 替身（value_cache / loader / _enum_text_maps /
  request_mark_stats_refresh），使绘图主路径脱离 MainWindow 可运行；
- `FakeLayoutManager`：layout_manager 替身（drop/拖拽路径所需方法均 no-op）；
- `FakeHost`：伪宿主窗口，使 `self.window().layout_manager` 可解析（真实应用中
  window() 为 MainWindow，不暴露该属性）。

```python
# tests/component/test_xxx.py
def test_drop_behavior(plot_factory, qapp, monkeypatch):
    dst = plot_factory()          # 默认注入 a/b/c 三列小 DataFrame
    ...
```

参考实现：`test_legend_drop.py`（QDropEvent 构造 + 拖拽注册表登记 + `silent_dialogs`
fixture 用 monkeypatch 替换 QMessageBox 静态方法）。

### 5.3 Qt 事件时序规范

- **禁止** `time.sleep` 硬等待，一律 `qtbot.waitSignal` / `qtbot.waitUntil`；
- **禁止新增固定 `pump(大值)`**：`pump(ms)` 只等钟表不等条件，是本次优化削掉的主要
  虚耗（单文件曾靠 4 处 `pump(600)` 白等 2.4s）。改造/新增用例按 `tests/fixtures/waits.py`
  的选型顺序来：`wait_until(条件)` > `settle()` / `flush_deferred_deletes()` > `pump()`；
  确需 `pump` 时留 3× 余量并在注释里写清"为什么没有可观测条件"；
- **`deleteLater` 的落地口径**：用例不跑 `app.exec()`，`processEvents()` 不消费
  `DeferredDelete`；要让窗口真正析构必须 `flush_deferred_deletes()`（内部
  `sendPostedEvents(None, DeferredDelete)`）。反过来，若后面还要读 C++ 对象，
  就只能 `settle()`——flush 会把对象删掉；
- 范围类断言（viewRange / autoRange）前先 `app.processEvents()` 或
  `qtbot.waitUntil(lambda: vb.targetRange() != old)`，避免读到挂起的 autoRange 状态
  （Y 抖动根因分析中 paint 阶段才消费挂起标志的教训）；
- 拖拽/鼠标用例基于控件自身坐标构造事件（参考 `_drop_legend` 辅助函数），不用绝对屏幕坐标。

### 5.4 隐私红线：不得写入真实标识

本仓库是**公开仓库**，测试与夹具里不得出现真实的内网主机/共享盘名、客户与 OEM/供应商名、
项目号、机型/软件代号，以及本机用户名路径（含 `file:///Users/…` 引用）。需要"现场实测串"
时只取结构、不取字面量——用 `fileserver` / `team-share` / `PRJ-0000-00` / `Demo` / `ENG01`
这类合成标识符，仅保留断言真正依赖的结构特征（UNC 两层前导、空格、`=`、`_`、反斜杠）。

合成夹具是默认路径：测试数据一律来自 `tests/fixtures/data_factory.py`，**禁止**从 `data/`
目录抄真实文件名当夹具（`data/` 本身已被 gitignore，泄漏通道是源码字面量）。

检查方式：提交前跑 `python3 .qoder/privacy/check_private_terms.py --staged`（字面量清单 +
结构型模式双路；清单缺失时会报错退出而非放行）；改写历史或推送远端前跑 `--all`。

> **盲区（会给出假的"干净"）**：检查器各模式底层都是 `git grep`，**未跟踪文件对它不可见**。
> 新增测试文件在 `git add` 之前跑 `--staged` 一定放行。正确姿势二选一：
> ① 先 `git add -N <新文件>` 再 `--tree HEAD`，扫完 `git reset -q -- <新文件>`；
> ② 先真实 `git add` 再 `--staged`。只要本轮有新建文件，就必须走这两条之一。


## 6. 已知陷阱清单

| # | 现象 | 根因 | 规避 |
|---|---|---|---|
| 1 | 独立 plot widget 测试在 offscreen 下**永久挂起** | 未注入 `plot_context` 时 `get_value_from_name` 返回 None → `QMessageBox.warning` 模态弹窗无显示器永不返回 | 必须使用 `plot_factory`（自动注入 FakePlotContext）；新路径若弹窗，用 monkeypatch 替身 |
| 2 | dropEvent 测试**段错误（SIGSEGV）** | PySide6 事件不持有 QMimeData 所有权，临时构造的 mimeData 被 GC 后事件访问悬垂指针 | 模块级列表保活（见 `test_legend_drop.py::_keep_alive`） |
| 3 | 默认值断言**偶发失败** | 会话内共享同一测试配置目录，ini 持久化使前面的用例改写了默认值 | 断言默认值前显式 `settings.set(...)` 置位；涉及持久的用例互相不要依赖顺序 |
| 4 | 用例卡死在**任何模态对话框** | QMessageBox / QFileDialog 在 offscreen 下阻塞等待用户输入 | 统一 monkeypatch 静态方法为记录器替身（参考 `silent_dialogs` fixture） |
| 5 | e2e 构造 MainWindow 时**把 pytest 参数当数据文件加载** | `MainWindow._handle_cli_args` 读取 `sys.argv[1:]`，pytest 的命令行参数被当作文件路径，弹模态框永久阻塞 | `main_window` 夹具构造前 `monkeypatch.setattr("sys.argv", [...])` 重置 |
| 6 | e2e 用例后**随机出现不相关用例失败/报错** | 加载链路调度的 `QTimer.singleShot` 延迟回调在窗口销毁后触发，槽函数异常经 `sys.excepthook` 进入 pytest-qt 异常池，被错误归因给其他用例 | `e2e/conftest.py` 的 `main_window` 夹具包装 `sys.excepthook` 过滤已知清理竞态（`_KNOWN_TEARDOWN_RACE_MARKERS`）；收尾先 `qtbot.wait` 排空定时器再 close。**新撞到的回调优先在源码侧收口**（延迟回调先问 `_widget_alive`，见 `layout_manager.py` 与 `test_xlink_sync_dead_containers.py`）：往 marker 里加名字实测无效——`monkeypatch` 自身还原钩子的 teardown 更晚，过滤器会提前失效 |
| 7 | pyqtgraph 范围/autoRange 断言与预期不符 | `vb.state` 中的值是 numpy float，`is True` 断言必败；auto-range 重算依赖宿主窗口 show（有效视图尺寸非零） | 断言用 `bool(...)`；需要真实布局计算的夹具必须 `widget.window().show()` |
| 8 | 临时诊断脚本中 monkeypatch 类方法**污染后续用例** | 直接改 `ClassName.method` 而不走 pytest monkeypatch，不会自动还原 | 一律用 `monkeypatch.setattr(Class, "method", ...)`；诊断脚本用后即删 |
| 9 | 合成 Enter 事件**段错误（SIGSEGV，exit 139）** | `QEvent(QEvent.Type.Enter)` 会被 `QWidget::event` 按 `QEnterEvent` 做 static_cast 读字段，裸事件没有那些字段；`HoverEnter` 同理（按 `QHoverEvent` 转）。实测 offscreen 投 Enter、cocoa 投 HoverEnter 都直接崩进程（exit 139） | 投 `QtGui.QEnterEvent(local, scene, global)`；`Leave` 不做转换，裸 `QEvent` 仍然安全 |
| 10 | 全量 e2e/component **偶发永久挂起或段错误**（`-q` 下撞到，`-v` 下像"跑完了"） | pytest-qt 的 `WaitSignal` 在 Python 侧开嵌套 `QEventLoop.exec()`（`pytestqt/wait_signal.py:26`），其间 Shiboken 的 `mainThreadDeletionHandler` 等一把别的线程持有的锁 → 死锁。**崩与挂是同一条根因的两个落点**：自动分代回收在 worker 线程执行时，会把主线程创建的 `QObject` 包装器拿到 worker 上析构（分代堆是进程级的，**不需要**worker 持有 Qt 引用）；锁序成环就是整窗冻结，抢锁没成环就在主线程定时器链表留下悬垂节点、由下一个事件循环踩中 → 段错误固定停在 `QTimerInfoList::activateTimers` | **根治已落地**：`src/core/gc_guard.no_autogc()` 包住 worker 窗口（`DataLoadThread.run()` 整段、`VarInfoWorker` 按单条任务），窗口内禁自动回收、退出时若 gen-0 越阈值用 `freeze()+unfreeze()` 清零且不做任何析构；护栏 `test_worker_no_autogc.py`（`gc.callbacks` 是自动收集唯一可观测点，打桩 `gc.collect` 拦不到）+ `test_load_worker_no_full_gc.py`（钉显式 collect）。定位手法：挂起时 `sample <pid>`，看 worker 线程栈是否出现 `SbkDeallocWrapperCommon → ~QObject` 且在等 GIL、主线程是否 `mainThreadDeletionHandler → QBasicMutex::lockInternal` 在等 Qt 锁。**判断某条线程要不要包窗口的依据是"它是否让被跟踪容器净增长"**：`logging.QueueListener` 线程实测 5 万条日志、145 次自动收集，0 次落在它身上（短命 dict 自己抵消），故不包 |
| 11 | 抽屉里 `QLineEdit.selectAll()` 后**一 pump 事件选区就没了**（`selectionStart/End` 变 `-1`，`selectedText()` 返回空串） | offscreen 弹窗拿不到键盘（stderr 直说 `This plugin does not support grabbing the keyboard`），`processEvents()` 期间补发一次 FocusOut，`QLineEdit` 随之清空选区；cocoa 上实测不会清 | 选区断言在 `selectAll()`/拖选**当拍**做，中间不要 `processEvents()`；要验像素表现就去 cocoa 截图（`tmp/_p02_hover_shot.py`） |
| 12 | 新用例让**全量耗时悄悄涨回去** | 固定 `pump(600)` 这类"等钟表不等条件"的写法最好抄（历史基线 73.5s 里约 3/4 是这类虚耗） | 按 §5.3 的选型顺序写等待；收尾的 `>1s` 点名与 `--durations=20` 会当场暴露新增慢例 |

## 7. tmp/ 脚本转正流程（五步法）

历史经验：`tmp/` 下的 offscreen 验证脚本是最有价值的测试资产来源，但**不能直接搬运**。标准转正流程：

1. **依赖审计**：grep 脚本对分支特定模块（如 `viewbox_state_compat`、`batched_xlink`）的引用，仅在当前分支存在的才可转正；
2. **API 核对**：对照当前分支现行语义（如 legend 拖拽"默认移动 / Ctrl 复制 / Shift 忽略"），基于旧设计的脚本须按现行语义重写；
3. **试跑识别挂起**：先在 pytest 环境跑一遍，快速暴露 QMessageBox 阻塞类挂起（用 faulthandler 定位）；
4. **重构 + 加断言**：`check()` 打印改为 `assert`；模态框替换身；按第 5 节模板组织；
5. **全量验证**：`uv run pytest -q` 全绿 + `ruff check tests/` 无告警。

### 待转正高价值脚本清单

| 脚本 | 建议目标 | 断言要点 | 状态 |
|---|---|---|---|
| `tmp/verify_freeze_restore_e2e.py` | `component/test_y_freeze.py` | setXRange 后 `autoRange[1]` 保持 True；冻结期 Y 不重算；恢复后 Y 重算正确（7 项断言） | ⚠️ 依赖 `viewbox_state_compat`（perf 分支），已按现行语义改写为 `test_y_autorange.py`；冻结/恢复细节待 perf 合流后补 |
| `tmp/verify_yrange_jitter_claims.py` | `component/test_y_freeze.py` | 可见段 Y 范围数值断言（约 38.71~61.29） | 待转正（依赖可见段重算语义） |
| `tmp/verify_ownership_inversion.py` | `component/test_xlink_sync.py` | XLink 级联下交互所有权归属 | 部分覆盖：`test_viewbox_zoom.py` 已有 XLink 同步用例，级联所有权待补 |
| `tmp/verify_reload_y_visibility.py` | `component/test_plot_widget.py` | reload 后 Y 轴可见性 | 待转正 |
| `tmp/verify_incremental_marks.py` | `component/test_mark_region.py` | 增量区间标记 | 待转正 |
| `tmp/verify_box_zoom_prod_aligned.py` | `component/test_plot_widget.py` | 框选缩放与生产行为一致 | ✅ 已按现行语义改写为 `test_viewbox_zoom.py` 框选用例（原脚本依赖 `batched_xlink`） |
| `tmp/test_template_conflict_choice.py` | `e2e/test_template_roundtrip.py` | 模板冲突选择路径 | 待转正 |

`diagnose_*` / `trace_*` / `watch_*` / `bisect_*` 类一次性诊断脚本不转正，留在 tmp/ 作历史参考。

## 8. 拓展路线图

### 覆盖率短板（e2e 已破冰，剩余为深化项）

| 模块 | 语句数 | 覆盖率 | 破冰/深化途径 |
|---|---|---|---|
| `src/ui/splash_screen.py` | 142 | **15%** | `src/ui` 最深的洞，且从无冒烟用例：`test_splash_screen.py` 起停 + 进度回调 |
| `src/ui/dialogs/time_correction.py` | 39 | **21%** | 只有抽屉/游标用例的间接覆盖：补 `test_time_correction.py`（偏移解析与越界分支） |
| `src/ui/cursor_sync_manager.py` | 460 | 48% | ✅ 已破冰（`cursor_x_domain` 系列）；深化：链接/解链与恢复路径 |
| `src/ui/variable_list.py` | 328 | 44% | ✅ 已破冰；深化：搜索/多选/拖拽出口 |
| `src/ui/file_loader_manager.py` | 869 | 49% | ✅ 已破冰；深化：reload 全流程、错误路径 |
| `src/ui/layout_manager.py` | 633 | 60% | ✅ 已破冰；深化：多子图布局/拖拽重排 |
| `src/ui/main_window.py` | 780 | 85% | ✅ 已破冰；深化：模板菜单/快捷键全集 |

### Phase 3：e2e 从 0 到 1（已完成 ✅ / 剩余项）

- ✅ `test_app_boot.py`：启动 → 主窗口构建 → 初始状态 → 空重载防护；
- ✅ `test_load_and_plot.py`：加载 CSV → 变量列表 → 拖拽绘图 → Ctrl+R/Ctrl+Y 快捷键；
- `test_template_roundtrip.py`：保存模板 → 清空 → 应用恢复（含冲突路径）；
- reload 全流程用例（历史 bug 密集区：游标链接、Y 轴可见性、变量搜索栏数据源）；
- `test_splash_screen.py`：启动画面冒烟（15%，`src/ui` 里最深的洞）。

### unit / component 补强

- `time_correction`（21%）与 `curve_strategy`（47%）、`font_cache`（67%）：src 层遗留空白，
  纯逻辑单元测试即可覆盖；
- `mdf_lazy_loader`（80%）：剩余集中在 asammdf 通道枚举与文件尾解析分支，用合成小型 MDF 补；
- `excel_loader`（87%）：calamine 快路径已覆盖，缺的是 openpyxl 回退与多 sheet 选择分支；
- component：Y 冻结转正（第 7 节清单）、游标管理（`cursor_manager` 904 语句 55%）、
  区间标记（`mark_region_manager` 105 语句 77%）。

### Phase 4：护栏与 CI

- ✅ 挂起护栏：`--timeout=120 --timeout-method=thread`（见 2.1，把静默卡死变成可诊断失败）；
- ✅ 耗时护栏：`--durations=20` + `tests/conftest.py` 的 `>1s` 点名钩子；
- ✅ 并行：`pytest-xdist` 全量 `-n 4 --dist loadfile`（见 2.1；`-m unit -n auto` 实测无收益，已否）；
- GitHub Actions 三平台矩阵（macOS / Windows / Linux，offscreen 模式无需 xvfb）；
- 覆盖率基线护栏：`--cov-fail-under=65` 防回退（2026-09-22 实测总体 71%，留 6 点余量；
  原先计划的 30 早已低于真实值，起不到防回退作用）；
- `pytest-benchmark` 加载/降采样基线，守护滚轮合并节流等历史优化成果；
- 可选：`hypothesis` 属性测试覆盖 loader 边界输入。
