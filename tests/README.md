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
>
> **上表是 macOS 快照，不是当前值**：当前（Windows + 本 venv）全量已到 **1401** 例
> （unit + component 1291、e2e 110），见 §1.1 的口径核对。本表只在"哪天在 macOS 上重跑过"
> 时才该更新整体口径，平时只动 §1.1。

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

### 1.1 Windows / 缺内嵌字体环境的环境性基线（非缺陷）

上表是 macOS arm64 口径。在 **Windows + 当前 venv** 下，有一批用例因平台差异
恒定失败，属于**环境基线**而非回归，改动前请先按这份清单核对，别误判成自己改坏的：

| 用例 | 现象 | 根因 |
|---|---|---|
| `test_variable_info_dialog_copy.py`（3 例：`TestFilePathCopy` 2 + `TestFieldCopyButton` 1） | 期望 `//server/share`，实得 `\\server\share` | 用例按 POSIX 写死了正斜杠；Windows 上项目会把 UNC 归一成反斜杠。同类断言已用 `仅 POSIX` 跳过（见 `tests/unit/utils/test_paths.py`） |
| `test_table_dialog_tab_model.py`（3 例） | `_row_header_width(999, font)` 期望 48，实得 58 | 该宽度由 `QFontMetrics` 算出；本 venv 缺 Qt 自带字体目录（stderr 报 `QFontDatabase: Cannot find font directory`），回退字体更宽 |
| `test_axis_drawer.py`（2 例） | 抽屉/预览宽度超出设计值 | 同上，布局尺寸来自字体度量 |

口径核对法（本改动实测，Windows + 当前 venv）：`-m "unit or component"` 选中 **1443** =
文档基线 **1044** + 新增 **399**；其中 6 失败 + 1 跳过全部落在上表，新增的 399 例全绿。
`-m e2e` 选中 **122** = 基线 **82** + 新增 **40**，2 项失败同样落在上表。
分层实测：unit **735**（= 588 + 147）、component **708**（= 456 + 252）、e2e **122**（= 82 + 40），
全量 **1565**。**先看"选中数 - 新增数"是否等于文档基线**，就能把"环境失败"与"回归"分开。
（勘误：上一版把 component 写成 686 / 自校 230，是把 241 算成 230 的手误——
标注 component 文件 23+78+11+28+9+28+28+36 = **241**，697 + 11 = 708 才与实测吻合。）

本轮（曲线标注功能：P1 + P0 收尾 + P3 折线 + P3 复制粘贴/撤销重做/标靶 + 双击/箭头/无边框
+ 时间戳 X 轴）累计新增 **439** 例：unit 147、component 252、e2e 40，分布在

| 文件 | 例 | 职责 |
|---|---|---|
| `tests/unit/core/test_annotation_models.py` | 118 | 数据模型：非法输入收拢、序列化往返幂等、折点上限与 `minimum_points` 下限、**线宽 0 = 文字无边框**（零 Qt 依赖） |
| `tests/unit/core/test_plot_config_annotations.py` | 15 | 模板字段**向后兼容**：老 YAML 没有 `annotations` 键也要读得出来 |
| `tests/component/test_annotation_no_interference.py` | 23 | **零干扰对照**：有无标注时 ViewBox 手势/自动范围逐位一致 + 门控不变量（含折线线段、标靶标签） |
| `tests/component/test_annotation_manager.py` | 78 | 管理器公共接口、资源回收、销毁期安全、**双击命中两段式**（BSP 快路径 + 容差兜底）与双击入口分流 |
| `tests/component/test_annotation_arrow.py` | 11 | **箭头头部方向**：8 方向视觉指向点积为 1、三角基边垂直于箭身、缩放后重定向、零长度退化 |
| `tests/component/test_annotation_polyline.py` | 28 | **折线专有边界**：拓扑/折点上限、父坐标几何往返、子图元 z 跟随、模板往返、**浏览态线段不抢平移** |
| `tests/component/test_annotation_placement.py` | 9 | **落点核对**：逐类比对"模型坐标 → 屏幕位置"，守住"构造期漏了写坐标"这类**静默分裂** |
| `tests/component/test_annotation_target.py` | 28 | **标靶专有**：标签文案（含花括号陷阱）、标签门控（重建后要重上）、样式跟随、屏幕尺寸恒定、**从标签起手拖**仍归 ViewBox |
| `tests/component/test_annotation_clipboard.py` | 28 | **复制/粘贴**：容器级剪贴板跨子图共享、原地更新、4% 错开、换新 id、坏条目跳过、数量上限 |
| `tests/component/test_annotation_undo.py` | 36 | **撤销/重做**：手势合并成一条、各类变更各占一步、快照去重、容量上限、重入守卫 |
| `tests/e2e/test_annotation_roundtrip.py` | 24 | 真实 MainWindow：三条入口状态一致、换布局保留（**七类**图元各走一遍）、模板往返、清除入口 |
| `tests/e2e/test_annotation_interaction_state.py` | 5 | **P0 出口条件**：真实窗口 + 2×1 XLink 下，10 条标注前后 `sigRangeChanged` 次数与 `_is_interacting` 时序逐项一致 |
| `tests/unit/core/test_time_axis_mapping.py` | 14 | **时间戳 X 轴映射**：行号→时间插值、字符串列按格式解析（生产路径 dtype）、越界/坏行回退、刻度格式三态判定（零 Qt 依赖） |
| `tests/component/test_time_axis_item.py` | 11 | **竖排时间轴图元**：tickStrings 映射、越界留空、mapper=None 回退数字、轴高先于首绘撑起、渲染有墨迹、offset/factor 反解、**刻度密度加倍**（spacing 减半的物理判定） |
| `tests/e2e/test_x_axis_timestamp.py` | 11 | **时间戳模式全链路**：真实 MainWindow 切换/恢复原轴、**网格/笔样式继承默认轴**、刻度与游标读数同口径、布局切换继承、重载无时间列静默回落、右键菜单构建与触发派发 |

> 上一轮（P1 + P0 收尾 + 折线）是 **275** 例（unit 130、component 117、e2e 28）；
> 复制/粘贴 + 撤销/重做 + 标靶一轮 **104** 例（unit +2、component +101、e2e +1）；
> 双击属性入口 + 箭头方向修复 + 文字无边框一轮 **24** 例（unit +1、component +23）；
> 时间戳 X 轴本轮 **36** 例（unit +14、component +11、e2e +11）。
> 差值不整齐的原因各自写在各处：P1→P3 的 31≠32 是因为 P1 的
> `test_polyline_is_accepted_by_model_but_not_rendered`（当时断言"折线不渲染"）被
> `test_polyline_is_rendered` **原地替换**，不算新增；复制粘贴轮 unit 只 +2，是因为
> `test_annotation_models.py` 里那几条按 `ANNOTATION_KINDS` 参数化的用例多收了一个 kind。
>
> **本表可以自校**：component 的标注文件 23+78+11+28+9+28+28+36 = **241**，加时间轴 11 = **252**；
> unit 的 118+15+14 = **147**；e2e 的 24+5+11 = **40**；三层合计正好是上面那三组新增数。


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
├── component/                   # component 层：offscreen + pytest-qt 控件级测试（50 个用例文件）
│   ├── conftest.py              #   plot_factory + legend/表格/变量信息弹窗的共享夹具
│   ├── _vid_shared.py           #   变量信息弹窗的替身主窗口与等待 helper（非用例文件）
│   ├── _tab_mode_shared.py      #   数值表 tab 模式的 loader 替身（非用例文件）
│   ├── _viewbox_events.py       #   合成鼠标/滚轮事件构造器（非用例文件）
│   └── test_*.py                #   legend / viewbox / 表格 / 游标 / 抽屉 / 标注 / 生命周期
└── e2e/                         # e2e 层：真实 MainWindow 冒烟（11 个用例文件）
    ├── conftest.py              #   dialog_stubs / main_window / loaded_window 夹具 + 清理竞态过滤
    └── test_*.py                #   启动 / 加载绘图 / 状态栏 / 抽屉 / 窗口几何 / 标注
```

四层职责与运行策略：

| 层 | 依赖 | 单用例目标耗时 | 运行时机 | marker |
|---|---|---|---|---|
| unit | 无 Qt | < 50ms | 每次保存 / 高频 | `unit` |
| component | offscreen + QApplication | < 1s | 提交前 | `component` |
| e2e | offscreen + 完整主窗口 | < 10s | CI / 发布前 | `e2e` |
| perf | offscreen + benchmark | 秒级 | 按需手动 | `perf` |

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
  **唯一例外是"节奏"而不是"等待条件"**：`test_annotation_polyline.py::_rate_limit_gap`
  用 `time.sleep(0.02)` 给 `GraphicsScene.mouseRateLimit`（默认 5ms）留出间隔——
  它没有可观测条件可等，等的是"两次 move 别贴在一起"这个节流窗口（见陷阱 #19）；
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
| 13 | 合成鼠标拖拽的**平移量偏小**，且"有图元那侧反而对" | `ViewBox.mouseDragEvent` 用 `childGroup.transform()` 这个**缓存**矩阵把屏幕像素换算成数据单位，而该矩阵要等下一次 repaint 才刷新。改完范围立刻拖拽，用的还是旧范围的**比例尺**（实测表现为 X 偏小、Y 正常——因为 Y 没变所以缓存不脏）。哪一侧先碰到重绘哪一侧就对，于是对照测试看起来像"标注影响了平移" | 合成的每个手势之间**必须 `qapp.processEvents()`**（真实事件循环里鼠标移动本身就在持续泵事件）。对照见 `test_annotation_no_interference.py::_run_gesture_sequence` |
| 14 | `autoRange()` / `childrenBounds()` 的断言**在两次调用之间就不相等**，跨 widget 更不可比 | `ViewBox.autoRange` 不是纯函数：它读当前 viewRect/targetRect 再写回，Y 又开着 `autoVisibleOnly`，X↔Y 互相影响，要几轮才到不动点；`childrenBounds()` 同样跟着当前视图范围漂，两个实例历史不同、收敛值就不同 | 比较的必须是**同一 widget 的前后对照**（推稳 → 记录 → 只改被测变量 → 再推稳 → 再记录），不能比两个实例。实测 `childrenBounds` 同一 widget 前后 4/4 一致、换跨实例 4 轮里 3 轮不一致，**且与标注多少无关**（只建一条矩形照样漂）——`test_annotation_no_interference.py` 的 `_settle_autorange` 与 `test_children_bounds_unchanged_by_annotations` 都按这个口径写 |
| 15 | 裸 plot widget 上**「清图 → 立刻重新绘图」挂死或原生崩溃**（无 Python 回溯，`-q` 下像"跑完了"） | 项目既有脆弱点，**与标注无关**：最小复现 `python -u tmp/repro_clear_replot.py without` 完全不碰标注代码，同样停在 step 4。崩/挂是同一条根因的两个落点（同 §10 的 Shiboken 包装器析构），区别只在哪个线程先撞上 | 用例不要写"清空后立刻重绘"这条链路：把生命周期断言放在重绘**之前**，`clear_current_plot` 另用一只全新 widget 验。参照 `test_annotation_manager.py` 与 `tmp/annotation_smoke.py` 的分段顺序（e2e 真实主窗口上走这条链路目前是通的，见 `test_annotation_roundtrip.py` 的模板往返） |
| 16 | 监听范围变化的槽**静默记错东西**：记下来的"子图索引"变成 `[True, False]`，而时间线看着还挺像样 | 本项目的 `ViewBox.sigRangeChanged` 是**三参**信号（`(viewBox, viewRange, changed)`，`changed` 是 `[x变了没, y变了没]`）。写成 `lambda vb, rng, i=index: ...` 这种"固定形参 + 默认值兜底"时，第三个实参**顶掉**默认值，`i` 就变成了 `changed` | 槽一律收 `*args` 或 `(vb, rng, changed=None)`；用默认值兜可变参数个数是坑。见 `test_annotation_interaction_state.py::_range_slot` |
| 17 | XLink 下**交互态持有者不是被操作的那个子图**（改主图范围，反倒是从图进了交互态；直接改从图，反倒是主图进） | `ViewBox.updateViewRange` 先 `link.linkedViewChanged()` 通知从图、**最后**才 emit 自己的 `sigRangeChanged`（ViewBox.py 该函数末尾），从图的 `_on_range_changed` 先跑就抢到了交互态；反向那一半来自 `setRange` 里 `link.linkedViewChanged(self, ax)` —— 写从图会**反向传播**到主图 | 既有行为，不要断言"持有者是某一个固定索引"。要守的是「同时只有一个持有者 + 进入退出配对」。另：想制造"从图掉队"让 `_sync_linked_x_ranges` 真的写范围，靠"把从图范围挪开"没用（反向传播后两边又一致了），得走 `setXLink(None)` → 健康检查分支接回 |
| 18 | 前后对照的两次运行**起点其实不同**：拖拽 Y 平移量差 0.1 | 序列里若有 `mw.resize()`（本用例为验 XLink 同步而故意 resize），跑完不收回，第二次运行就从更大的窗口出发；绘图区尺寸变了，像素→数据换算跟着变 | 对照实验的"重置"必须**连窗口几何一起钉**（`reset_view()` 里 `mw.resize(_WINDOW_W, _WINDOW_H)`），只钉视图范围不够。见 `test_annotation_interaction_state.py::reset_view` |
| 19 | 「某个图元抢走了 ViewBox 的拖拽」这条路径**用 press+move 合成事件测不到**，断言恒真 | pyqtgraph 的 `GraphicsScene` 在 **hover** 阶段收集 `dragItems`（`HoverEvent.acceptDrags`，先登记先得），一旦有图元登记了左键，之后 `sendDragEvent` 走 init 分支就**直接用登记到的图元当 `dragItem`，不再回退去问 ViewBox** —— 视图平不动。只发 press+move 时 `lastHoverEvent.dragItems()` 是空的，走的是"遍历候选、问 `mouseDragEvent`"的兜底分支，永远轮不到那条抢法 | 想测"抢不抢拖拽"，合成序列必须是 **hover → press → move**，并直接断言 `scene.dragItem is pw.view_box`。另：`GraphicsScene.mouseRateLimit` 默认 200/s（5ms），两次 move 挨得太近**后一次会被整个丢掉**，所以在 hover 与拖拽 move 之间要留出间隔（`_rate_limit_gap()`，真实事件循环里鼠标移动天然有间隔）。见 `test_annotation_polyline.py::_hover_then_drag` 与 `TestPanParity` |
| 20 | 折线在浏览态**悄悄把视图锁死**（拖不动），删掉标注又好了 | `PolyLineROI` 的线段 `_PolyLineSegment` 是**独立子图元**，容器关不住它。它抢不抢拖拽的真正开关是**父容器的 `translatable`**（`_PolyLineSegment.hoverEvent` 里那句 `acceptDrags` 的条件就是它），而不是线段自己的 `acceptedMouseButtons` —— 后者挡的是"点段插折点"那条路。另外 `PolyLineROI.addSegment` 会给两端手柄 `setDeletable(True)`，把手柄按键 **OR** 上 `RightButton`（右键菜单能删折点） | 门控**一律赋值，不用 `\|=`**（订阅式 OR 会越关越多）。折线的三层要一起管：容器 `translatable/resizable`、手柄（可见性 + 按键）、线段按键；`write_geometry` 里 `setPoints` 之后必须重上门控（它会把手柄与线段全拆重建）。见 `src/ui/widgets/annotation_items.py` 约束 2/5 与 `test_annotation_polyline.py::TestPanParity` |
| 21 | 存模板再套回来，图元**跳回原位**（屏幕上明明拖过） | ROI 的**自由手柄**（`addFreeHandle`）的 `pos()` 是 ROI 的**局部**坐标，ROI 整体平移只改自己的位置、手柄一个都不动；而 `RectROI` 这类"pos + size"型 ROI 的 `getState()['pos']` 本身就是数据坐标。把两者混着当"数据坐标"用，平移量就静默丢了（P1 的直线/箭头踩过） | 几何读写统一走 `mapToParent` / `mapFromParent`（`_free_handle_points` / `_to_local_points`）；断言手柄坐标前先换算。见 `src/ui/widgets/annotation_items.py` 约束 4 |
| 22 | **"某类图元拖不动"是假故障**：把合成拖拽的 move 间隔去掉，target 就一动不动，别的类却照动 | `GraphicsScene.mouseRateLimit` 默认 200/s（5ms），`_moveEventIsAllowed()` 为假时 `mouseMoveEvent` **整段早退**——move 一条都不发。哪一类"看起来能动"取决于它接受的第一个 move 落在哪个时间片，于是排查时像"只有 target 坏了" | 合成拖拽的每一步之间都要留间隔（`_rate_limit_gap()` / `_gap()`，`time.sleep(0.02~0.03)`）。**判据是"历史 +几步、模型跟没跟"，不是"看起来动了"**：`test_annotation_undo.py::TestGestureMerge` 先断言几何真的变了，再做合并断言 |
| 23 | 某个关键字参数**被同名的循环变量覆盖**，"关掉"的那条分支永远不生效 | `clear(*, record=True)` 里原来是 `for record in records:` —— 循环结束后 `record` 是最后一个图元（真值），`if record:` 恒真。表现为 `load(replace=True)` 记**两步**（先"空图"再内容），撤销一次只得到空图 | 循环变量**不要与关键字参数同名**（改名 `entry`）。这类坑不会报错、只会多出一步状态，必须靠**"计步"断言**抓：`assert _history_len() == base + 1`，而不是只断言最终内容对。见 `test_annotation_undo.py::TestTemplateAndReplay` |
| 24 | 带**附属子图元**的图元（标靶 `TargetLabel`）在浏览态**照吃鼠标**，从标签上起手拖拽把视图锁死 | 标签是独立的 `TextItem` 子图元，容器的 `acceptedMouseButtons` 管不到它；而且 `setLabel()` 是**整只换新**，新建出来的标签带默认按键 —— 光在门控里拔一次不够，**每次重建都要重上** | 门控必须覆盖"容器 + 每一个附属子图元"，且在**任何重建子图元的路径末尾**再调一次 `_apply_gating()`（`set_label_text` 就是这条）。零干扰对照只测"拖拽归属"，不测门控**状态**，所以另写 `test_annotation_target.py::TestLabelGating` 直接钉 `acceptedMouseButtons()` |
| 25 | 撤销 / 重做**写进模板或回放之后 Ctrl+Z 把刚回放出来的标注整份抹掉** | 换布局回放（`layout_manager.replay_annotations` → `load(record=False)`）与"套用模板"（`load()` 默认 `record=True`）语义不同：前者的结果是**新的历史起点**，后者的结果是**可撤销的一步**。把两者混成一条 `load` 路径，要么回放留了一颗地雷，要么模板套用撤不了 | 快照式撤销的"栈顶 = 当前状态"意味着 `reset_history()` 与 `_record_history()` 必须严格区分；`_restoring` 守卫保证撤销**自己不入栈**。见 `test_annotation_undo.py::TestTemplateAndReplay` 与 `TestReentrancy` |
| 26 | **箭头头部沿水平轴镜像**：带垂直分量的箭头三角反折（朝上的箭头三角朝下），8 个方向只有水平两支是对的 | `QTransform.rotate` 在屏幕坐标系（y 向下）里正角为顺时针；`update_arrow_head` 的角度公式若按 data 空间的手感写 `atan2(unit_y, -unit_x)`，`unit_y` 的符号就错了。而 `pg.ArrowItem` 的 `pxMode` 自带 `ItemIgnoresTransformations`，方向完全由 `setStyle(angle=...)` 决定，错了也没有缩放"掩护" | 屏幕坐标里的方向计算，`unit_y` 必须取反：`atan2(-unit_y, -unit_x)`。判定口径：decor 的 `angle` 反算视觉指向 `(-cos θ, -sin θ)`，与 tail→head 的屏幕方向点积为 1。见 `test_annotation_arrow.py` |
| 27 | **双击矩形底边/角点命中失败**（落到"打开变量编辑器"），边线中点也漏 | Qt 的几何命中是**半开区间**（`QRectF.contains` 不含右/下边界），恰好压在数学边界上的点 `scene.items(点)` 与 shape 精筛都会漏；编辑态手柄中心恰好压在矩形角点上，是同一缺口的必然触发点 | 自建命中测试要两段式：BSP 快路径 + **容差兜底**（膨胀 4px 的探针矩形与 `mapToScene(shape())` 相交判定，只扫本子图标注）。见 `annotation_manager.hit_test` 与 `test_annotation_manager.py::TestHitTest` |
| 28 | **旋转刻度文字整帧画不出来**（轴高对了、文字没了） | `AxisItem` 按"刻度文字的**高**"预留轴高；旋转后占位的是文字的**宽**。在 `tickStrings`（正处于 `generateDrawSpecs` 的 paint 途中）里调 `setMaximumHeight` 触发重新布局，**本帧作废**，画面停回旧高度 | 高度预算必须在**首绘前**就位：挂 mapper 时用 `sample_text` 预量文字宽就地撑高（事件循环中安全）；绘制中只记录更宽值，经 `QTimer.singleShot(0)` 推迟到事件循环应用。见 `time_axis_item.py` 的时序约束注释与 `test_time_axis_item.py::TestRotatedGeometry` |
| 29 | 生产路径 CSV 的**时间列是字符串 dtype 不是 datetime64**，按 dtype 找时间列永远找空 | `FastDataLoader.do_parse_date` 默认 `False` 且 UI 路径没人传 True——`parse_dates` 形同虚设，时间列保持字符串，只有 `loader.date_formats` 记录了"这列可按某格式解析" | 可作时间轴的列判定以 **`loader.date_formats` 的键**为准（加载器已做"列名含 time 关键字 + 格式可解析"双重识别），datetime64 列只作补充；映射器自己 `pd.to_datetime(series, format=fmt)` 解析。见 `MainWindow.available_time_columns` 与 `test_x_axis_timestamp.py::TestDetection` |
| 30 | **换轴后网格变纯黑**：切时间戳 X 轴后整片网格线从淡灰变实黑 | 项目的淡灰网格来自 `showGrid(alpha=0.1)`，它走 PlotItem ctrl 滑条把**已存在**的轴 `setGrid(25)`；事后新建的 `TimestampAxisItem` 硬编码 `setGrid(255)` 就成了黑网格。pyqtgraph 网格线用的是轴笔色 × grid alpha，没有独立"网格色" | 换轴新建轴时**样式必须抄默认轴**：`setPen / setTextPen / setGrid(default_axis.grid)`，别硬编码。见 `MainWindow.apply_x_axis_to_widget` 与 `test_x_axis_timestamp.py::TestToggleAndRestore` |
| 31 | 时间轴**刻度标签比默认轴稀疏/稠密**，调 `tickStrings` 没用 | 标签密度由 `AxisItem._tickDensity` 决定：`spacing = dif / (2.25 × density × √(size/ref))`，**density 越大间距越小、标签越密**（方向容易想反） | 用 `setTickDensity()`，别自己改 `tickValues`。时间轴取 `TICK_DENSITY = 2.0`（密度加倍；竖排文字横向只占字高，加密后不会被文本密度判定跳过）。见 `time_axis_item.py` 与 `test_time_axis_item.py::TestTickDensity` |

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
