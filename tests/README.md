# 测试体系说明（唯一入口）

本文件是 CSV Plot 测试体系的完整文档：快速开始、分层设计、环境隔离机制、新用例编写指南、已知陷阱清单与拓展路线图均集中于此。

## 1. 概览与现状

技术栈：**pytest + pytest-qt + pytest-cov**，`QT_QPA_PLATFORM=offscreen` 无头运行，三层金字塔组织（unit / component / e2e）。

| 层 | 用例数 | 实测耗时 | 状态 |
|---|---|---|---|
| unit（纯逻辑，无 Qt） | 142 | ~1.2s | 稳定运行 |
| component（offscreen 控件级） | 62 | ~1.2s | 稳定运行 |
| e2e（主窗口冒烟） | 10 | ~2.1s | 稳定运行 |
| **全量** | **214** | **~4.2s** | `214 passed` |

覆盖率现状（`--cov=src`）：

- 总体 **45%**（e2e 破冰前 31%）；
- `src/core` 层 0%~100%（已建测试的模块 73%~100%；短板：`curve_strategy` 0%、`font_cache` 22%；
  `template_models` / `plot_config` 为 100%，`src/utils/paths` 亦为 100%）；
- `src/data` 层 21%~82%（短板：`mdf_lazy_loader` 21%、`excel_loader` 53%）；
- `src/ui` 层已破冰：`main_window` 69%、`plot_widget` 61%、`layout_manager` 41%、
  `variable_list` 40%、`file_loader_manager` 38%、`cursor_sync_manager` 28%；
  仍为 0% 的仅剩 `splash_screen`（启动画面，待单独冒烟）。

## 2. 快速开始

```bash
# 1. 安装 dev 依赖（pytest / pytest-qt / pytest-cov 均在 dev 组）
uv sync --group dev
```

命令速查（全部实测可用）：

| 命令 | 用途 | 实测结果 |
|---|---|---|
| `uv run pytest -m unit` | 日常开发高频回归（秒级） | 142 passed |
| `uv run pytest -m "unit or component"` | 提交前自检 | 204 passed |
| `uv run pytest` | CI 全量 | 214 passed |
| `uv run pytest -m e2e` | e2e 冒烟（含真实主窗口） | 10 passed |
| `uv run pytest --cov=src --cov-report=term-missing` | 覆盖率（含未覆盖行号） | - |
| `uv run pytest -m component -k legend` | 精准过滤（marker + 关键字组合） | 28 passed |

> 注：本机 `uv` 不在 PATH 时使用绝对路径 `~/.local/bin/uv`，或先将其加入 shell 配置。

## 3. 目录结构与分层设计

```
tests/
├── conftest.py                  # 全局环境：offscreen、路径隔离、marker 自动分层
├── fixtures/
│   └── data_factory.py          # 合成数据工厂（make_timeseries / write_csv / make_simple_rows）
├── unit/                        # unit 层：无 Qt 依赖的纯逻辑测试
│   ├── core/                    #   config / plot_config / settings / storage /
│   │                            #   template_* / auto_save
│   ├── data/                    #   fast_loader / base_loader / metadata / excel_loader
│   └── utils/                   #   paths
└── component/                   # component 层：offscreen + pytest-qt 控件级测试
    ├── conftest.py              #   FakePlotContext / FakeLayoutManager / FakeHost 替身
    ├── test_legend_drag.py      #   legend 拖拽单元（19 用例）
    ├── test_legend_drop.py      #   legend drop 路径（9 用例，含 QMessageBox 替身）
    ├── test_variable_search_bar.py  # 变量搜索栏增量标记（10 用例）
    ├── test_viewbox_zoom.py     #   滚轮缩放/框选缩放/XLink 同步/双击清除（8 用例）
    ├── test_y_autorange.py      #   Y 轴范围语义：首绘全量/缩放冻结/Ctrl+Y 重算（5 用例）
    └── test_drop_add_replace.py #   变量拖拽添加/替换/悬停文案（11 用例）
└── e2e/                         # e2e 层：真实 MainWindow 冒烟（offscreen）
    ├── conftest.py              #   dialog_stubs / main_window / loaded_window 夹具 + 清理竞态过滤
    ├── test_app_boot.py         #   启动冒烟：构建/初始状态/快捷键静默/空重载防护（4 用例）
    └── test_load_and_plot.py    #   加载→变量列表→拖拽绘图→快捷键（6 用例）
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
- 范围类断言（viewRange / autoRange）前先 `app.processEvents()` 或
  `qtbot.waitUntil(lambda: vb.targetRange() != old)`，避免读到挂起的 autoRange 状态
  （Y 抖动根因分析中 paint 阶段才消费挂起标志的教训）；
- 拖拽/鼠标用例基于控件自身坐标构造事件（参考 `_drop_legend` 辅助函数），不用绝对屏幕坐标。

## 6. 已知陷阱清单

| # | 现象 | 根因 | 规避 |
|---|---|---|---|
| 1 | 独立 plot widget 测试在 offscreen 下**永久挂起** | 未注入 `plot_context` 时 `get_value_from_name` 返回 None → `QMessageBox.warning` 模态弹窗无显示器永不返回 | 必须使用 `plot_factory`（自动注入 FakePlotContext）；新路径若弹窗，用 monkeypatch 替身 |
| 2 | dropEvent 测试**段错误（SIGSEGV）** | PySide6 事件不持有 QMimeData 所有权，临时构造的 mimeData 被 GC 后事件访问悬垂指针 | 模块级列表保活（见 `test_legend_drop.py::_keep_alive`） |
| 3 | 默认值断言**偶发失败** | 会话内共享同一测试配置目录，ini 持久化使前面的用例改写了默认值 | 断言默认值前显式 `settings.set(...)` 置位；涉及持久的用例互相不要依赖顺序 |
| 4 | 用例卡死在**任何模态对话框** | QMessageBox / QFileDialog 在 offscreen 下阻塞等待用户输入 | 统一 monkeypatch 静态方法为记录器替身（参考 `silent_dialogs` fixture） |
| 5 | e2e 构造 MainWindow 时**把 pytest 参数当数据文件加载** | `MainWindow._handle_cli_args` 读取 `sys.argv[1:]`，pytest 的命令行参数被当作文件路径，弹模态框永久阻塞 | `main_window` 夹具构造前 `monkeypatch.setattr("sys.argv", [...])` 重置 |
| 6 | e2e 用例后**随机出现不相关用例失败/报错** | 加载链路调度的 `QTimer.singleShot` 延迟回调在窗口销毁后触发，槽函数异常经 `sys.excepthook` 进入 pytest-qt 异常池，被错误归因给其他用例 | `e2e/conftest.py` 的 `main_window` 夹具包装 `sys.excepthook` 过滤已知清理竞态（`_KNOWN_TEARDOWN_RACE_MARKERS`）；收尾先 `qtbot.wait` 排空定时器再 close |
| 7 | pyqtgraph 范围/autoRange 断言与预期不符 | `vb.state` 中的值是 numpy float，`is True` 断言必败；auto-range 重算依赖宿主窗口 show（有效视图尺寸非零） | 断言用 `bool(...)`；需要真实布局计算的夹具必须 `widget.window().show()` |
| 8 | 临时诊断脚本中 monkeypatch 类方法**污染后续用例** | 直接改 `ClassName.method` 而不走 pytest monkeypatch，不会自动还原 | 一律用 `monkeypatch.setattr(Class, "method", ...)`；诊断脚本用后即删 |

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
| `src/ui/file_loader_manager.py` | 812 | 38% | ✅ e2e 已破冰；深化：reload 全流程、错误路径 |
| `src/ui/main_window.py` | 497 | 69% | ✅ e2e 已破冰；深化：模板菜单/快捷键全集 |
| `src/ui/layout_manager.py` | 588 | 41% | ✅ e2e 已破冰；深化：多子图布局/拖拽重排 |
| `src/ui/variable_list.py` | 354 | 40% | ✅ e2e 已破冰；深化：搜索/多选 |
| `src/ui/cursor_sync_manager.py` | 467 | 28% | component：游标用例深化 |
| `src/ui/splash_screen.py` | 141 | 0% | 唯一仍 0%：启动画面单独冒烟 |

### Phase 3：e2e 从 0 到 1（已完成 ✅ / 剩余项）

- ✅ `test_app_boot.py`：启动 → 主窗口构建 → 初始状态 → 空重载防护；
- ✅ `test_load_and_plot.py`：加载 CSV → 变量列表 → 拖拽绘图 → Ctrl+R/Ctrl+Y 快捷键；
- `test_template_roundtrip.py`：保存模板 → 清空 → 应用恢复（含冲突路径）；
- reload 全流程用例（历史 bug 密集区：游标链接、Y 轴可见性、变量搜索栏数据源）；
- `test_splash_screen.py`：启动画面冒烟（最后 0% 模块）。

### unit / component 补强

- `curve_strategy`（0%）与 `font_cache`（22%）：src/core 层遗留空白，纯逻辑单元测试即可覆盖；
- `mdf_lazy_loader`（21%）：用 asammdf 合成小型 MDF 文件，测懒加载接口与通道枚举；
- `excel_loader`（53%）：calamine 路径、多 sheet 选择；
- component：Y 冻结转正（第 7 节清单）、游标管理（`cursor_manager` 909 语句仅 31%）、
  区间标记（`mark_region_manager` 23%）。

### Phase 4：护栏与 CI

- GitHub Actions 三平台矩阵（macOS / Windows / Linux，offscreen 模式无需 xvfb）；
- 覆盖率基线护栏：`--cov-fail-under=30` 防回退；
- `pytest-benchmark` 加载/降采样基线，守护滚轮合并节流等历史优化成果；
- `pytest-xdist` 仅对 unit 层启用并行（`-m unit -n auto`），Qt 用例保持串行；
- 可选：`hypothesis` 属性测试覆盖 loader 边界输入。
