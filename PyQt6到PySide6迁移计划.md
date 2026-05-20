# PyQt6 → PySide6 迁移计划 v2.1（csv-plot 项目）

> **项目名称**：csv-plot（CSV/MDF 数据可视化工具）
> **文档版本**：v2.1（适用于全新仓库）
> **更新日期**：2026-05-20
> **基于**：`.reference/` 原始 PyQt6 代码 + v1.0 迁移分析报告

---

## 目录

- [一、项目现状分析](#一项目现状分析)
- [二、API 差异对照表](#二api-差异对照表)
- [三、迁移可行性评估](#三迁移可行性评估)
- [四、迁移影响范围清单](#四迁移影响范围清单)
- [五、Debug Log 系统移除方案](#五debug-log-系统移除方案)
- [六、分阶段迁移计划](#六分阶段迁移计划)
- [七、pyqtgraph 兼容性专项分析](#七pyqtgraph-兼容性专项分析)
- [八、自动化迁移脚本](#八自动化迁移脚本)
- [九、常见问题与解决方案](#九常见问题与解决方案)
- [十、验证测试计划](#十验证测试计划)
- [十一、最佳实践建议](#十一最佳实践建议)

---

## 一、项目现状分析

### 1.1 项目概况

| 维度 | 详情 |
|---|---|
| 项目类型 | CSV/MDF 数据可视化桌面应用 |
| Python 版本 | >= 3.12 |
| 原始主入口 | `csv_plot_pyqt6.py`（约 4976 行） |
| 原始 src/ 模块数 | 34 个 Python 文件 |
| 原始 Qt 绑定 | **仅 PyQt6**（无 PySide/PyQt5/QtPy） |
| 目标 Qt 绑定 | **PySide6 >= 6.11.1** |
| UI 构建方式 | **纯代码动态创建**（无 .ui 文件） |

### 1.2 依赖清单

```
pyqt6>=6.9.1,<6.10          ← 需替换为 PySide6>=6.11.1
pyqt6-qt6>=6.9.1,<6.10      ← 需移除（PySide6 自带 Qt 库）
pyqtgraph>=0.14.0            ← 保留，确认绑定选择
numpy>=2.0.2                 ← 无影响
pandas>=2.3.2                ← 无影响
asammdf>=7.4.0               ← 无影响
ujson>=5.11.0                ← 无影响
charset-normalizer>=3.0.0    ← 无影响
pathlib>=1.0.1               ← 无影响
```

### 1.3 PyQt6 精确使用统计（已核验）

| 检查项 | 精确数量 | 分布文件数 | 迁移影响 |
|---|---|---|---|
| `from PyQt6.QtCore import ...` | 26 处 | 19 个文件 | 需全局替换为 `PySide6` |
| `from PyQt6.QtGui import ...` | 9 处 | 9 个文件 | 需全局替换为 `PySide6` |
| `from PyQt6.QtWidgets import ...` | 15 处 | 14 个文件 | 需全局替换为 `PySide6` |
| 函数内延迟 `from PyQt6.*` | 14 处 | 6 个文件 | 自动化脚本可覆盖 |
| `pyqtSignal` 类属性定义 | **13 个**（2 个类） | 2 个文件 | `pyqtSignal` → `Signal` |
| `pyqtSignal` 在 import 中 | 2 处 | 2 个文件 | import 行也需替换 |
| `pyqtSlot` / `pyqtProperty` | **0 个** | — | 无需处理 ✅ |
| `.ui` 文件 / `uic.loadUi` | **0 个** | — | 无需处理 ✅ |
| `QVariant` | **0 个** | — | 无需处理 ✅ |
| `WebEngine` | **0 个** | — | 无需处理 ✅ |
| Qt 枚举使用 | ~84 处 | 多个文件 | 已用全限定语法，兼容 ✅ |
| 信号 `.connect()` 调用 | ~90 处 | 多个文件 | 语法兼容，无需修改 ✅ |
| 信号 `.emit()` 调用 | ~50+ 处 | 多个文件 | 语法兼容，无需修改 ✅ |
| pyqtgraph 导入 | **16 处** | 10 个文件 | 需确保绑定选择正确 |
| `QSignalBlocker` 使用 | 7 处 | 5 个文件 | 语法兼容，注意 PySide6 差异 |

### 1.4 核心架构理解

项目采用「主文件 + 模块化 src/」架构，主入口 `csv_plot_pyqt6.py` 包含：
- `DraggableGraphicsLayoutWidget` 类（主绘图控件，约 4000 行）
- `MainWindow` 类（主窗口，约 800 行）
- 全局启动代码

src/ 模块按功能分层：
- `src/core/` — 配置常量、数据类型定义、调度器
- `src/data/` — CSV/MDF 数据加载、元数据管理
- `src/ui/` — UI 组件（变量列表、对话框、数据表格）
- `src/ui/widgets/` — 绘图核心管理器（光标、标记、多曲线、事件处理等）

### 1.5 关键结论

> **本项目迁移条件非常理想。** 项目未使用 `pyqtSlot`、`pyqtProperty`、`.ui` 文件、`QVariant`、`WebEngine` 等容易出问题的特性。枚举已全部使用 PyQt6 全限定语法（如 `Qt.AlignmentFlag.AlignCenter`），与 PySide6 高度兼容。迁移工作量主要集中在 **PyQt6 → PySide6 import 替换**、**pyqtSignal → Signal 替换**、**Debug Log 系统移除** 三项操作上。

---

## 二、API 差异对照表

### 2.1 导入模块对照

| PyQt6 写法 | PySide6 写法 | 说明 |
|---|---|---|
| `from PyQt6.QtCore import Qt` | `from PySide6.QtCore import Qt` | 模块路径替换 PyQt6 → PySide6 |
| `from PyQt6.QtGui import QColor` | `from PySide6.QtGui import QColor` | 同上 |
| `from PyQt6.QtWidgets import QApplication` | `from PySide6.QtWidgets import QApplication` | 同上 |
| `from PyQt6 import QtCore` | `from PySide6 import QtCore` | 包级导入也需替换 |

### 2.2 信号/槽/属性对照

| PyQt6 写法 | PySide6 写法 | 优先级 |
|---|---|---|
| `from PyQt6.QtCore import pyqtSignal` | `from PySide6.QtCore import Signal` | P0 |
| `progress = pyqtSignal(int)` | `progress = Signal(int)` | P0 |
| `from PyQt6.QtCore import pyqtSlot` | `from PySide6.QtCore import Slot` | 本项目未使用 |
| `from PyQt6.QtCore import pyqtProperty` | `from PySide6.QtCore import Property` | 本项目未使用 |
| `signal.connect(callback)` | `signal.connect(callback)` | 语法兼容 ✅ |
| `signal.emit(*args)` | `signal.emit(*args)` | 语法兼容 ✅ |
| `signal.disconnect(callback)` | `signal.disconnect(callback)` | 语法兼容 ✅ |

### 2.3 枚举使用对照

| 场景 | PyQt6 | PySide6（推荐） | 兼容性 |
|---|---|---|---|
| 全限定写法 | `Qt.AlignmentFlag.AlignCenter` | `Qt.AlignmentFlag.AlignCenter` | ✅ 完全兼容 |
| 短命名写法 | `Qt.AlignCenter`（PyQt6 也支持） | `Qt.AlignCenter`（PySide6 也支持） | ✅ 官方推荐全限定 |
| 自定义枚举 | `Qt.KeyboardModifier.ShiftModifier` | `Qt.KeyboardModifier.ShiftModifier` | ✅ 完全兼容 |
| `QtMsgType` | `QtMsgType.QtDebugMsg` | `QtMsgType.QtDebugMsg` | ✅ 都来自 QtCore |

> **本项目优势**：已全部使用全限定枚举语法（如 `Qt.AlignmentFlag.AlignCenter`），与 PySide6 的推荐写法完全一致，无需任何枚举修改。

> **PySide6 宽容模式**：PySide6 对 PyQt6 风格的枚举取值具有向后兼容，即使存在写法差异，大多数情况下不会报错。

### 2.4 关键 API 行为差异

| API | PyQt6 行为 | PySide6 行为 | 影响 |
|---|---|---|---|
| `QApplication.instance()` | 返回 QApplication 实例 | 返回 QApplication 实例 | ✅ 无差异 |
| `QSignalBlocker(obj)` | 上下文管理器，阻塞对象信号 | 上下文管理器，阻塞对象信号 | ✅ 无差异 |
| `qInstallMessageHandler()` | 安装 Qt 消息处理器 | 安装 Qt 消息处理器 | ✅ 兼容（本项目移除后不涉及） |
| `QThread` 生命周期 | 线程完成后自动清理 | 同 PyQt6 | ✅ 无差异 |
| `QTimer.singleShot()` | 静态方法 | 静态方法 | ✅ 无差异 |
| `QMimeData` | 拖放数据容器 | 拖放数据容器 | ✅ 无差异 |
| `QCursor.pos()` | 全局鼠标位置 | 全局鼠标位置 | ✅ 无差异 |
| 对象删除检测 | `RuntimeError: wrapped C/C++ object has been deleted` | 同 PyQt6 | ✅ 无差异 |

### 2.5 打包参数对照

| 打包工具 | PyQt6 参数 | PySide6 参数 | 说明 |
|---|---|---|---|
| PyInstaller | `--hidden-import PyQt6` | `--hidden-import PySide6` | hidden import 替换 |
| Nuitka | `--enable-plugin=pyqt6` | `--enable-plugin=pyside6` | 插件名替换 |
| Nuitka | `--include-package=pyqt6` | `--include-package=PySide6` | package 名替换 |

---

## 三、迁移可行性评估

### 3.1 可行性评分

| 评估维度 | 评分 | 说明 |
|---|---|---|
| API 兼容性 | ⭐⭐⭐⭐⭐ | 项目已用全限定枚举语法，几乎无枚举风险 |
| 迁移工作量 | ⭐⭐⭐⭐☆ | import + signal + debug log 移除，约 3-5 小时 |
| pyqtgraph 兼容性 | ⭐⭐⭐⭐⭐ | 官方支持，自动检测绑定 |
| 第三方库风险 | ⭐⭐⭐⭐⭐ | 无 WebEngine/UIC/QVariant 等高风险点 |
| 打包部署 | ⭐⭐⭐⭐☆ | 需更新打包参数 |
| 测试覆盖 | ⭐⭐⭐☆☆ | 建议补充自动化回归测试 |

### 3.2 预估工时

| 阶段 | 预估时间 | 说明 |
|---|---|---|
| 代码修改 | 2 ~ 4 小时 | import 替换 + signal 替换 + debug log 移除 |
| 依赖更新 | 15 分钟 | 修改 pyproject.toml 和 uv.lock |
| 编译验证 | 15 分钟 | 确保无语法错误 |
| 功能测试 | 2 ~ 4 小时 | 核心功能回归测试 |
| 打包验证 | 1 ~ 2 小时 | PyInstaller/Nuitka 打包测试 |
| **合计** | **5.5 ~ 10.5 小时** | 约 1-1.5 个工作日 |

---

## 四、迁移影响范围清单

### 4.1 需修改的文件总览

共涉及 **22 个 Python 源文件** + **2 个配置文件**：

### 4.2 🔴 高优先级文件

| 文件 | PyQt6 使用 | 修改内容 |
|---|---|---|
| `csv_plot_pyqt6.py` | 3 行顶层 import + pyqtSignal 导入 + 2 处延迟 import | import 替换、pyqtSignal 移除、debug log 移除 |
| `src/core/config.py` | 2 行 import + QtMsgType 使用 | import 替换、**debug log 系统全部移除** |
| `src/ui/widgets/custom_viewbox.py` | 3 行 import + 13 个 pyqtSignal 定义 | import 替换、pyqtSignal → Signal |
| `src/data/loader.py` | 1 行 import + 3 个 pyqtSignal 定义 | import 替换、pyqtSignal → Signal、debug log 移除 |

### 4.3 🟡 中等优先级文件

| 文件 | PyQt6 import 行 | 额外修改 |
|---|---|---|
| `src/core/scheduler.py` | 1 行 | 无额外修改 |
| `src/ui/drag_drop.py` | 3 行 | 无额外修改 |
| `src/ui/mark_stats.py` | 2 行 | 无额外修改 |
| `src/ui/variable_list.py` | 3 行顶层 + 3 处延迟 | 无额外修改 |
| `src/ui/table_dialog.py` | 3 行 | 无额外修改 |
| `src/ui/plot_variable_editor.py` | 3 行 | debug log 移除 |
| `src/ui/file_loader_manager.py` | 2 行 | debug log 移除 |
| `src/ui/layout_manager.py` | 2 行 | debug log 移除 |
| `src/ui/cursor_sync_manager.py` | 2 行 | debug log 移除 |

### 4.4 🟢 低优先级文件（仅 import 替换）

| 文件 | PyQt6 import 行 |
|---|---|
| `src/ui/dialogs/axis.py` | 2 行 |
| `src/ui/dialogs/help.py` | 1 行 |
| `src/ui/dialogs/layout_input.py` | 2 行 |
| `src/ui/dialogs/time_correction.py` | 2 行 |
| `src/ui/widgets/plot_container.py` | 2 行 |
| `src/ui/widgets/plot_ui_manager.py` | 3 行顶层 + 1 处延迟 + debug log |
| `src/ui/widgets/cursor_manager.py` | 5 处延迟 import + debug log |
| `src/ui/widgets/plot_data_manager.py` | 1 行顶层 + 3 处延迟 import |
| `src/ui/widgets/mark_region_manager.py` | 1 处延迟 import |
| `src/ui/widgets/event_handler.py` | 1 处延迟 import + debug log |

### 4.5 配置文件修改

| 文件 | 修改内容 |
|---|---|
| `pyproject.toml` | 替换 `pyqt6>=6.9.1,<6.10` + `pyqt6-qt6>=6.9.1,<6.10` → `PySide6>=6.11.1` |
| `uv.lock` | 删除后重新 `uv sync` 生成 |

### 4.6 无需修改的部分 ✅

- **所有信号连接**（`.connect()`）：语法完全兼容
- **所有信号发射**（`.emit()`）：语法完全兼容
- **所有枚举使用**：已使用全限定语法，与 PySide6 完全兼容
- **所有 pyqtgraph API 调用**：`pg.GraphicsLayoutWidget`、`pg.ViewBox`、`pg.InfiniteLine` 等均兼容
- **所有布局和样式代码**：纯 Python 代码，无 .ui 文件
- **数据加载逻辑**：`QThread`、`QTimer` API 完全兼容
- **所有 pandas/numpy/asammdf 代码**：与 Qt 无关

---

## 五、Debug Log 系统移除方案

### 5.1 现状分析

原始 PyQt6 代码在 `src/core/config.py` 中内置了一套临时调试日志系统，由 `DEBUG_LOG_ENABLED = False` 控制开关。该系统的设计意图是临时排查问题，不应保留在正式版本中。

### 5.2 Debug Log 组件清单

#### 5.2.1 config.py 中的核心组件

| 组件 | 位置 | 说明 |
|---|---|---|
| `DEBUG_LOG_ENABLED = False` | 第 16 行 | 总开关，始终为 False |
| `_DEBUG_LOGGER` | 第 19 行 | logging.Logger 实例 |
| `debug_log()` | 第 39-46 行 | 日志封装函数 |
| `safe_callback()` | 第 49-101 行 | 安全回调装饰器（依赖 debug_log） |
| `_install_faulthandler()` | 第 104-123 行 | 崩溃追踪安装器 |
| `_log_uncaught_exception()` | 第 126-130 行 | 未捕获异常日志 |
| `_threading_exception_logger()` | 第 133-140 行 | 线程异常日志 |
| `_qt_message_handler()` | 第 143-156 行 | Qt 消息日志（依赖 `QtMsgType`） |
| `install_global_debug_hooks()` | 第 159-178 行 | 钩子安装入口（依赖 `QApplication`） |
| `_FAULTHANDLER_FILE` | 第 33 行 | faulthandler 日志文件句柄 |
| `_ORIGINAL_EXCEPTHOOK` | 第 34 行 | 原始异常钩子引用 |
| `_ORIGINAL_THREADING_EXCEPTHOOK` | 第 35 行 | 原始线程异常钩子引用 |
| `_QT_MESSAGE_HANDLER_INSTALLED` | 第 36 行 | Qt 消息处理器安装标记 |

#### 5.2.2 散布在其他文件中的 debug_log 调用

| 文件 | debug_log 调用次数 | DEBUG_LOG_ENABLED 引用次数 |
|---|---|---|
| `csv_plot_pyqt6.py` | ~25 处 | ~15 处 |
| `src/data/loader.py` | ~8 处 | ~1 处 |
| `src/ui/file_loader_manager.py` | ~18 处 | 0 处 |
| `src/ui/layout_manager.py` | ~4 处 | 0 处 |
| `src/ui/cursor_sync_manager.py` | ~7 处 | 0 处 |
| `src/ui/plot_variable_editor.py` | ~4 处 | 0 处 |
| `src/ui/widgets/cursor_manager.py` | ~10 处 | ~1 处 |
| `src/ui/widgets/plot_ui_manager.py` | ~10 处 | ~7 处 |
| `src/ui/widgets/event_handler.py` | ~3 处 | 0 处 |

### 5.3 移除策略

**总体策略：分三步走，确保安全移除。**

#### 步骤一：config.py 清理

1. **删除的文件顶部的 import**：
   ```python
   # 删除以下行
   import faulthandler        # 仅 debug_log 使用
   import signal              # 仅 _install_faulthandler 使用
   import threading           # 仅 _threading_exception_logger 使用
   import traceback           # 仅 debug hooks 使用
   from PyQt6.QtCore import QtMsgType, qInstallMessageHandler  # 仅 _qt_message_handler 使用
   from PyQt6.QtWidgets import QApplication  # 仅 install_global_debug_hooks 使用
   ```

2. **删除的函数和变量**（第 16-178 行中与 debug log 相关的全部删除）：
   - `DEBUG_LOG_ENABLED`
   - `_DEBUG_LOGGER` 及相关日志配置
   - `debug_log()` 函数
   - `_FAULTHANDLER_FILE`
   - `_ORIGINAL_EXCEPTHOOK`
   - `_ORIGINAL_THREADING_EXCEPTHOOK`
   - `_QT_MESSAGE_HANDLER_INSTALLED`
   - `_install_faulthandler()`
   - `_log_uncaught_exception()`
   - `_threading_exception_logger()`
   - `_qt_message_handler()`
   - `install_global_debug_hooks()`

3. **改造 `safe_callback()` 装饰器**：移除所有 `debug_log()` 调用，替换为简洁的异常处理：
   ```python
   def safe_callback(func):
       from functools import wraps

       @wraps(func)
       def wrapper(*args, **kwargs):
           try:
               return func(*args, **kwargs)
           except RuntimeError as e:
               err_msg = str(e).lower()
               if "deleted" in err_msg or "wrapped" in err_msg or "c++ object" in err_msg:
                   return None
               raise
           except Exception:
               return None

       return wrapper
   ```

4. **保留的常量**（这些与 debug log 无关，继续保留）：
   - `DEFAULT_SHOW_X_AXIS_LABEL`
   - `DEFAULT_PADDING_VAL_X`、`DEFAULT_PADDING_VAL_Y`
   - `FILE_SIZE_LIMIT_BACKGROUND_LOADING`
   - `RATIO_RESET_PLOTS`
   - `FROZEN_VIEW_WIDTH_DEFAULT`
   - `XRANGE_THRESHOLD_FOR_SYMBOLS`
   - `BLINK_PULSE`、`FACTOR_SCROLL_ZOOM`
   - `MIN_INDEX_LENGTH`
   - `DEFAULT_LINE_WIDTH`、`THICK_LINE_WIDTH`、`THIN_LINE_WIDTH`
   - `UI_DEBOUNCE_DELAY_MS`
   - `PLOT_ROW_MAX_DEFAULT` 等布局常量
   - `FLOAT32_SAFE_MAX`
   - `UNIT_KEYWORD_RATIO_THRESHOLD`、`VALID_NUMERIC_RATIO_THRESHOLD`
   - `_UNIT_KEYWORDS`
   - `_evaluate_float32_safety()`

#### 步骤二：散布文件中的 debug_log 调用清理

对以下文件中所有 `debug_log(...)` 调用执行**删除**（因为 `DEBUG_LOG_ENABLED` 始终为 `False`，这些调用实际上是死代码）：

| 文件 | 处理方式 |
|---|---|
| `csv_plot_pyqt6.py` | 删除所有 `debug_log(...)` 调用；删除所有 `if DEBUG_LOG_ENABLED:` 条件块及其内部代码；删除 `install_global_debug_hooks(app)` 调用 |
| `src/data/loader.py` | 删除所有 `debug_log(...)` 调用；删除 `if DEBUG_LOG_ENABLED:` 条件块 |
| `src/ui/file_loader_manager.py` | 删除所有 `debug_log(...)` 调用 |
| `src/ui/layout_manager.py` | 删除所有 `debug_log(...)` 调用 |
| `src/ui/cursor_sync_manager.py` | 删除所有 `debug_log(...)` 调用 |
| `src/ui/plot_variable_editor.py` | 删除所有 `debug_log(...)` 调用 |
| `src/ui/widgets/cursor_manager.py` | 删除所有 `debug_log(...)` 调用；删除 `if DEBUG_LOG_ENABLED:` 条件块 |
| `src/ui/widgets/plot_ui_manager.py` | 删除所有 `debug_log(...)` 调用；删除 `if DEBUG_LOG_ENABLED:` 条件块 |
| `src/ui/widgets/event_handler.py` | 删除所有 `debug_log(...)` 调用 |

#### 步骤三：import 语句清理

从所有文件的 import 中移除 `debug_log`、`DEBUG_LOG_ENABLED` 等 debug 相关导入：

```python
# 删除前
from src.core.config import (
    DEBUG_LOG_ENABLED,
    debug_log,
    safe_callback,
    DEFAULT_LINE_WIDTH,
    ...
)

# 删除后
from src.core.config import (
    safe_callback,
    DEFAULT_LINE_WIDTH,
    ...
)
```

### 5.4 移除验证清单

- [ ] `grep -r "debug_log" src/ csv_plot_pyqt6.py` 返回空（除 `safe_callback` 自身定义外）
- [ ] `grep -r "DEBUG_LOG_ENABLED" src/ csv_plot_pyqt6.py` 返回空
- [ ] `grep -r "install_global_debug_hooks" src/ csv_plot_pyqt6.py` 返回空
- [ ] `grep -r "_install_faulthandler" src/ csv_plot_pyqt6.py` 返回空
- [ ] `grep -r "qt_message_handler" src/ csv_plot_pyqt6.py` 返回空
- [ ] `grep -r "QtMsgType" src/ csv_plot_pyqt6.py` 返回空
- [ ] `grep -r "qInstallMessageHandler" src/ csv_plot_pyqt6.py` 返回空
- [ ] `safe_callback` 仍可正常导入使用

---

## 六、分阶段迁移计划

### 阶段零：确认项目结构（2 分钟）

```bash
# 确认当前在正确的项目目录
cd /path/to/csv_plot_pyside

# 查看项目根目录文件
ls -la

# 确认 .reference/ 目录存在（原始 PyQt6 代码备份）
ls -la .reference/
```

---

### 阶段一：环境准备与依赖更新（15 分钟）

**目标**：搭建 PySide6 开发环境，更新依赖声明。

```bash
# 1. 确认 PySide6 已安装（当前项目 pyproject.toml 中已有）
pip show PySide6

# 2. 如未安装，执行
pip install "PySide6>=6.11.1"

# 3. 验证 PySide6 安装
python -c "import PySide6; print(PySide6.__version__); from PySide6.QtCore import Qt; print('OK')"
```

**pyproject.toml 修改**（如尚未修改）：

```diff
 dependencies = [
     "asammdf>=7.4.0",
     "charset-normalizer>=3.0.0",
     "numpy>=2.0.2",
     "pandas>=2.3.2",
     "pathlib>=1.0.1",
-    "pyqt6>=6.9.1,<6.10",
-    "pyqt6-qt6>=6.9.1,<6.10",
+    "PySide6>=6.11.1",
     "pyqtgraph>=0.14.0",
     "ujson>=5.11.0",
 ]
```

**检查点**：`pip install -e .` 成功安装所有依赖，`import PySide6` 无错误。

---

### 阶段二：自动化代码迁移（30 分钟）

**目标**：使用增强版自动化脚本完成所有机械替换。

详见 [第八节：自动化迁移脚本](#八自动化迁移脚本)。

运行方式：

```bash
# 预览模式（不修改文件，仅显示将要做的更改）
python scripts/migrate_pyqt6_to_pyside6.py --dry-run

# 正式执行
python scripts/migrate_pyqt6_to_pyside6.py

# 验证
python scripts/migrate_pyqt6_to_pyside6.py --verify
```

**脚本自动完成的替换**：

| 替换类型 | 规则 | 影响数量 |
|---|---|---|
| 顶层 import | `from PyQt6.` → `from PySide6.` | 50+ 处 |
| 延迟 import | `from PyQt6.` → `from PySide6.`（函数内） | 14 处 |
| 信号定义 | `pyqtSignal` → `Signal` | 13 处 |
| 信号导入 | import 行中的 `pyqtSignal` → `Signal` | 2 处 |

**检查点**：
- `grep -r "PyQt6" src/ csv_plot_pyqt6.py` 返回空
- `grep -r "pyqtSignal" src/ csv_plot_pyqt6.py` 返回空
- `grep -r "pyqtSlot" src/ csv_plot_pyqt6.py` 返回空

---

### 阶段三：Debug Log 系统移除（1-2 小时）

**目标**：按 [第五节方案](#五debug-log-系统移除方案) 彻底移除 debug log。

> **注意**：此阶段需要手动执行，因为它涉及结构性代码删除而非简单替换。

#### 3.1 清理 config.py

遵循 [5.3 节步骤一](#步骤一configpy-清理) 的详细指导，删除所有 debug 相关代码。

#### 3.2 清理散布的 debug_log 调用

遵循 [5.3 节步骤二](#步骤二散布文件中的-debug_log-调用清理)，逐文件删除 `debug_log()` 调用和 `DEBUG_LOG_ENABLED` 条件块。

**操作建议**：使用编辑器的全局搜索功能，依次搜索以下模式：
1. `debug_log(` → 删掉整行
2. `if DEBUG_LOG_ENABLED` → 删掉整个条件块
3. `, debug_log` → 从 import 中删除
4. `DEBUG_LOG_ENABLED,` → 从 import 中删除
5. `install_global_debug_hooks(` → 删掉整行

#### 3.3 清理 import 语句

从所有文件的 `from src.core.config import (...)` 中移除 debug 相关导入。

#### 3.4 验证

```bash
grep -rn "debug_log" src/ csv_plot_pyqt6.py
grep -rn "DEBUG_LOG_ENABLED" src/ csv_plot_pyqt6.py
grep -rn "install_global_debug_hooks" src/ csv_plot_pyqt6.py
grep -rn "QtMsgType" src/ csv_plot_pyqt6.py
grep -rn "qInstallMessageHandler" src/ csv_plot_pyqt6.py
grep -rn "faulthandler" src/ csv_plot_pyqt6.py
```

以上所有 grep 命令应返回空结果。

**检查点**：所有 debug 引用已清除，`safe_callback` 仍可用。

---

### 阶段四：pyqtgraph 绑定确认（10 分钟）

**目标**：确保 pyqtgraph 正确识别 PySide6。

#### 4.1 环境变量方式（如果自动检测失败）

```python
# 在 csv_plot_pyqt6.py 的 if __name__ == "__main__" 块开头添加
import os
os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
```

#### 4.2 启动验证

```python
# 在应用启动后验证
import pyqtgraph as pg
print(f"Qt 绑定: {pg.Qt.QT_LIB}")        # 应输出: PySide6
print(f"Qt 版本: {pg.Qt.VERSION_INFO}") # 应输出: PySide6 x.x.x Qt x.x.x
```

**检查点**：应用启动后 `pg.Qt.QT_LIB` 输出 `PySide6`。

---

### 阶段五：编译验证（15 分钟）

**目标**：确保代码无语法错误。

```bash
# 编译检查所有 Python 文件
python -m py_compile csv_plot_pyqt6.py

# 更全面的检查
python -c "
import py_compile, sys
files = ['csv_plot_pyqt6.py']
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f'OK: {f}')
    except py_compile.PyCompileError as e:
        print(f'ERROR: {e}')
        sys.exit(1)
"

# 尝试导入主模块（不执行 GUI）
python -c "
import sys
sys.path.insert(0, '.')
# 在无头模式下导入模块检查
import importlib
try:
    # 测试各核心模块是否能被解析
    modules_to_check = [
        'src.core.config',
        'src.core.types',
        'src.core.scheduler',
        'src.data.metadata',
    ]
    for m in modules_to_check:
        importlib.import_module(m)
        print(f'OK: {m}')
except Exception as e:
    print(f'ERROR: {e}')
"
```

**检查点**：所有模块可成功解析，无 import 错误。

---

### 阶段六：功能验证测试（2-4 小时）

**目标**：全面验证迁移后功能正常。

详见 [第十节：验证测试计划](#十验证测试计划)。

---

### 阶段七：打包验证（1-2 小时）

**目标**：验证打包后的应用正常运行。

```bash
# PyInstaller 打包测试
pyinstaller csv_plot_pyqt6.py \
    --onefile \
    --name csv_plot \
    --noconsole \
    --clean \
    --noconfirm \
    --hidden-import PySide6 \
    --hidden-import pyqtgraph

# Nuitka 打包测试（推荐）
nuitka --onefile --standalone \
    --output-filename=csv_plot \
    --windows-console-mode=disable \
    --enable-plugin=pyside6 \
    csv_plot_pyqt6.py
```

**检查点**：打包后的可执行文件正常运行，所有功能正常。

---

## 七、pyqtgraph 兼容性专项分析

### 7.1 本项目 pyqtgraph 使用清单

| 组件 | 使用位置 | PySide6 兼容性 |
|---|---|---|
| `pg.GraphicsLayoutWidget` | 主绘图布局（csv_plot_pyqt6.py） | ✅ 完全兼容 |
| `pg.ViewBox` | 自定义视图框（custom_viewbox.py） | ✅ 完全兼容 |
| `pg.PlotWidget` | 散点图对话框（table_dialog.py） | ✅ 完全兼容 |
| `pg.PlotDataItem` | 曲线数据项（types.py） | ✅ 完全兼容 |
| `pg.InfiniteLine` | 光标竖线（csv_plot_pyqt6.py, plot_ui_manager.py） | ✅ 完全兼容 |
| `pg.LinearRegionItem` | 标记区域（csv_plot_pyqt6.py, mark_region_manager.py） | ✅ 完全兼容 |
| `pg.ScatterPlotItem` | 散点图（csv_plot_pyqt6.py, table_dialog.py, cursor_manager.py） | ✅ 完全兼容 |
| `pg.TextItem` | 文本标注（csv_plot_pyqt6.py） | ✅ 完全兼容 |
| `pg.GraphicsWidget` | 头部区域（csv_plot_pyqt6.py） | ✅ 完全兼容 |
| `pg.SignalProxy` | 鼠标信号代理（csv_plot_pyqt6.py） | ✅ 完全兼容 |
| `pg.mkPen()` | 创建画笔（20+ 处） | ✅ 完全兼容 |
| `pg.mkBrush()` | 创建画刷（csv_plot_pyqt6.py） | ✅ 完全兼容 |
| `pg.setConfigOptions()` | 全局配置（csv_plot_pyqt6.py） | ✅ 完全兼容 |

### 7.2 继承关系兼容性

本项目有两个类继承了 pyqtgraph 的类：

```python
# csv_plot_pyqt6.py
class DraggableGraphicsLayoutWidget(pg.GraphicsLayoutWidget): ...

# src/ui/widgets/custom_viewbox.py
class CustomViewBox(pg.ViewBox): ...
```

**风险评估**：pyqtgraph 的 `GraphicsLayoutWidget` 和 `ViewBox` 内部不依赖特定的 Qt 绑定实现细节。它们通过 `pyqtgraph.Qt` 抽象层访问 Qt API，该抽象层在检测到 PySide6 后自动映射。继承关系在 PySide6 下完全兼容。

### 7.3 信号交互兼容性

本项目通过 `CustomViewBoxSignals(QObject)` 定义了 13 个自定义信号。迁移后这些信号使用 PySide6 的 `Signal`，与 pyqtgraph 的事件回调链完全兼容。

```python
# 信号定义（custom_viewbox.py）
class CustomViewBoxSignals(QObject):
    request_jump_to_data = Signal(object, object)  # 迁移后
    ...

# 信号连接（event_handler.py）
vb.signals.request_jump_to_data.connect(self._on_jump_to_data)

# 信号发射（custom_viewbox.py）
self.signals.request_jump_to_data.emit(self.plot_widget, context_x)
```

**风险评估**：✅ 无风险。PySide6 的 `Signal` 与 PyQt6 的 `pyqtSignal` 在连接和发射行为上完全一致。

### 7.4 绑定选择策略

**推荐使用环境变量方式**（最可靠）：

```python
# 在 csv_plot_pyqt6.py 的 if __name__ == "__main__" 块中，所有 import 之前
import os
os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
```

这样无论 pyqtgraph 在哪个子模块中延迟导入，都能正确绑定 PySide6。

---

## 八、自动化迁移脚本

### 8.1 增强版脚本

以下脚本专为 csv-plot 项目定制，覆盖所有迁移场景：

```python
#!/usr/bin/env python3
"""
csv-plot 项目：PyQt6 → PySide6 自动化迁移脚本 v2.0

功能：
  1. 全局替换 PyQt6 → PySide6 import
  2. 替换 pyqtSignal → Signal
  3. 同步更新 import 行中的 pyqtSignal → Signal
  4. 更新打包命令注释
  5. 支持 dry-run 预览模式

使用方法：
    python migrate_pyqt6_to_pyside6.py [--dry-run] [--path <项目根目录>] [--verify]

参数：
    --dry-run   仅显示将要修改的内容，不实际修改文件
    --path      项目根目录路径（默认为当前目录）
    --verify    仅验证，不修改文件（检查是否有残留 PyQt6 引用）
"""

import re
import sys
import argparse
from pathlib import Path


REPLACEMENTS = [
    (r'from PyQt6\.(\S+)', r'from PySide6.\1'),
    (r'import PyQt6\.(\S+)', r'import PySide6.\1'),
]

# 需要特殊处理的信号文件
SIGNAL_FILES = {
    'src/data/loader.py',
    'src/ui/widgets/custom_viewbox.py',
}

# 打包命令注释替换规则
PACKAGING_REPLACEMENTS = [
    (r'--enable-plugin=pyqt6', '--enable-plugin=pyside6'),
    (r'csv_plot_pyqt6', 'csv_plot'),
]


def migrate_file_content(content: str, filepath: Path) -> tuple[str, list[str]]:
    changes = []

    for pattern, replacement in REPLACEMENTS:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            matches = re.findall(pattern, content)
            for m in matches:
                changes.append(f"  import: PyQt6.{m} → PySide6.{m}")
            content = new_content

    # 替换 pyqtSignal → Signal（类属性和 import 行）
    if 'pyqtSignal' in content:
        # 处理 as pyqtSignal 的导入别名
        content = re.sub(r'\bpyqtSignal\b', 'Signal', content)
        changes.append("  pyqtSignal → Signal")

    # 替换打包命令注释
    for pattern, replacement in PACKAGING_REPLACEMENTS:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changes.append(f"  packaging: {pattern} → {replacement}")

    return content, changes


def migrate_file(filepath: Path, dry_run: bool = False) -> list[str]:
    all_changes = []

    try:
        original = filepath.read_text(encoding='utf-8')
    except Exception as e:
        return [f"❌ 无法读取 {filepath}: {e}"]

    new_content, changes = migrate_file_content(original, filepath)

    if new_content != original:
        if not dry_run:
            filepath.write_text(new_content, encoding='utf-8')
        all_changes.append(f"\n📄 {filepath}")
        all_changes.extend(changes)

    return all_changes


def verify_project(root: Path) -> tuple[list[str], bool]:
    issues = []
    clean = True

    py_files = list(root.rglob('*.py'))
    py_files = [f for f in py_files
                if not any(p in f.parts for p in ('__pycache__', '.git', '.venv', 'venv', '.reference'))]

    for py_file in py_files:
        try:
            content = py_file.read_text(encoding='utf-8')
        except Exception:
            continue

        for i, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            if 'PyQt6' in stripped:
                issues.append(f"  ⚠️  {py_file}:{i}: {stripped.strip()}")
                clean = False
            if 'pyqtSignal' in stripped:
                issues.append(f"  ⚠️  {py_file}:{i}: {stripped.strip()}")
                clean = False
            if 'pyqtSlot' in stripped:
                issues.append(f"  ⚠️  {py_file}:{i}: {stripped.strip()}")
                clean = False

    return issues, clean


def main():
    parser = argparse.ArgumentParser(description='PyQt6 → PySide6 迁移脚本 v2.0')
    parser.add_argument('--dry-run', action='store_true', help='仅预览，不修改文件')
    parser.add_argument('--path', default='.', help='项目根目录')
    parser.add_argument('--verify', action='store_true', help='仅验证，不修改')
    args = parser.parse_args()

    root = Path(args.path).resolve()
    all_changes = []

    # 收集所有 Python 文件
    py_files = list(root.rglob('*.py'))
    py_files = [f for f in py_files
                if not any(p in f.parts for p in ('__pycache__', '.git', '.venv', 'venv', '.reference'))]

    if args.verify:
        issues, clean = verify_project(root)
        if clean:
            print("\n✅ 验证通过：未发现残留 PyQt6 引用")
        else:
            print(f"\n⚠️  发现 {len(issues)} 处残留 PyQt6 引用：")
            for issue in issues:
                print(issue)
        return 0 if clean else 1

    # 执行迁移
    for py_file in sorted(py_files):
        changes = migrate_file(py_file, dry_run=args.dry_run)
        all_changes.extend(changes)

    if all_changes:
        print(f"\n{'='*60}")
        mode = '[DRY-RUN] ' if args.dry_run else ''
        print(f"{mode}迁移完成，共修改 {sum(1 for c in all_changes if c.startswith('📄'))} 个文件")
        print(f"{'='*60}")
        for change in all_changes:
            print(change)
    else:
        print("未发现需要迁移的内容。")

    # 迁移后验证
    if not args.dry_run:
        issues, clean = verify_project(root)
        if clean:
            print("\n✅ 验证通过：所有 PyQt6 引用已成功替换")
        else:
            print(f"\n⚠️  仍有 {len(issues)} 处残留引用需要手动处理：")
            for issue in issues:
                print(issue)


if __name__ == '__main__':
    sys.exit(main())
```

---

## 九、常见问题与解决方案

### 9.1 导入错误

#### Q1: `ModuleNotFoundError: No module named 'PySide6'`

**原因**：PySide6 未安装或安装不完整。

**解决**：
```bash
pip uninstall PySide6 PySide6-Addons PySide6-Essentials -y
pip install "PySide6>=6.11.1"
```

#### Q2: `ImportError: cannot import name 'pyqtSignal' from 'PySide6.QtCore'`

**原因**：import 语句中的 `pyqtSignal` 未被脚本替换。

**解决**：手动将 `from PySide6.QtCore import pyqtSignal` 改为 `from PySide6.QtCore import Signal`。

#### Q3: 双重导入冲突

**原因**：同时安装了 PyQt6 和 PySide6。

**解决**：
```bash
pip uninstall pyqt6 pyqt6-qt6 pyqt6-sip -y
pip list | grep -i pyqt  # 确认已全部卸载
```

### 9.2 枚举相关

#### Q4: `AttributeError: type object 'Qt' has no attribute 'XXX'`

**原因**：使用了 PySide6 不支持的枚举写法。

**解决**：确认使用全限定语法。本项目已全部使用全限定语法，不会遇到此问题。

#### Q5: 枚举比较警告

**原因**：PySide6 中某些枚举比较可能出现类型不匹配。

**解决**：使用 `==` 比较枚举值即可，PySide6 6.x 已改善此问题。

### 9.3 信号相关

#### Q6: `TypeError: Signal has no attribute 'connect'`

**原因**：可能在类外部定义了 Signal，PySide6 要求 Signal 必须是 QObject 子类的类属性。

**解决**：确保所有 `Signal(...)` 定义都在 `QObject` 子类内部。本项目符合此要求。

#### Q7: 信号连接后无响应

**原因**：信号签名不匹配。

**解决**：检查 `Signal(type1, type2)` 的类型声明与 `.emit(v1, v2)` 传递的参数类型是否一致。

### 9.4 pyqtgraph 相关

#### Q8: pyqtgraph 仍使用 PyQt6

**现象**：`pg.Qt.QT_LIB` 输出 `PyQt6` 而不是 `PySide6`。

**解决**：
```python
import os
os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
# 确保在所有 import pyqtgraph 之前设置
```

#### Q9: `RuntimeError: wrapped C/C++ object has been deleted`

**原因**：访问了已被 Qt 销毁的 C++ 对象。

**解决**：`safe_callback` 装饰器已处理此场景。如果在新代码中遇到，使用 `sip.isdeleted()`（PyQt6）或 `shiboken6.isValid()`（PySide6）检查对象有效性。

### 9.5 打包相关

#### Q10: PyInstaller 打包后报错 `No module named 'PySide6'`

**解决**：
```bash
pyinstaller csv_plot_pyqt6.py \
    --hidden-import PySide6 \
    --hidden-import PySide6.QtCore \
    --hidden-import PySide6.QtGui \
    --hidden-import PySide6.QtWidgets \
    --hidden-import shiboken6
```

#### Q11: Nuitka 打包 `--enable-plugin=pyside6` 无效

**解决**：确保 Nuitka 版本 >= 1.5，检查插件是否可用：
```bash
nuitka --plugin-list | grep pyside6
```

### 9.6 运行时错误

#### Q12: 应用闪退无报错

**解决**：
1. 在终端中启动应用，查看 stderr 输出
2. 使用 `python -X dev csv_plot_pyqt6.py` 获取更多调试信息
3. 检查是否有 segfault（可能与 Qt 插件缺失有关）

#### Q13: macOS 上字体/图标显示异常

**解决**：
```python
# 在应用启动前添加
from PySide6.QtGui import QFontDatabase
QFontDatabase.addApplicationFont("/System/Library/Fonts/Helvetica.ttc")
```

---

## 十、验证测试计划

### 10.1 冒烟测试

```python
# 在 csv_plot_pyqt6.py 的 if __name__ == "__main__" 块中添加
import pyqtgraph as pg
print(f"Qt 绑定: {pg.Qt.QT_LIB}")
print(f"Qt 版本: {pg.Qt.VERSION_INFO}")
assert pg.Qt.QT_LIB == "PySide6", f"预期 PySide6，实际 {pg.Qt.QT_LIB}"
print("✅ Qt 绑定验证通过")
```

**验证项**：
- [ ] 应用正常启动，无导入错误
- [ ] `pg.Qt.QT_LIB` 输出 `PySide6`
- [ ] 主窗口正常显示，无渲染异常

### 10.2 完整功能测试矩阵

#### P0（阻塞发布 — 必须通过）

| 编号 | 功能模块 | 测试项 | 测试方法 | 预期结果 |
|---|---|---|---|---|
| T01 | 文件加载 | CSV 文件加载和解析 | 点击加载按钮，选择测试 CSV | 变量列表正确显示 |
| T02 | 文件加载 | MDF 文件懒加载 | 加载 .mf4 文件 | 变量列表正确显示，枚举通道有文本标签 |
| T03 | 文件加载 | 大文件后台加载 | 加载 >10MB CSV 文件 | 进度条显示，完成后变量列表更新 |
| T04 | 文件加载 | 拖拽加载 | 拖拽 CSV 文件到窗口 | 文件成功加载 |
| T05 | 绘图显示 | 单曲线绘制 | 拖拽变量到绘图区 | 曲线正确显示 |
| T06 | 绘图显示 | 多曲线绘制 | 拖拽多个变量到同一绘图区 | 多曲线正确显示，颜色不同 |
| T07 | 绘图显示 | 缩放和平移 | 滚轮缩放、左键拖拽 | 缩放/平移流畅，X 轴同步 |
| T08 | 光标功能 | 光标显示 | 启用光标模式 | 竖线光标出现，坐标值正确 |
| T09 | 光标功能 | 光标同步 | 多个子图对比 | 所有子图光标 X 轴同步 |
| T10 | 光标功能 | 跳转到数据 | 右键菜单 → Jump to Data | 数据表对话框打开并定位到正确行 |
| T11 | 标记功能 | 区域标记 | 在图表上拖拽创建标记 | 标记区域正确显示 |
| T12 | 标记功能 | 标记统计 | 创建标记后查看统计窗口 | 统计值正确（最小/最大/平均值等） |
| T13 | 布局管理 | 网格布局调整 | 修改行列数 | 子图重新排列正确 |
| T14 | 性能 | 大数据量渲染 | 加载 >50000 行数据并绘图 | 渲染流畅，无卡顿 |

#### P1（重要 — 应通过）

| 编号 | 功能模块 | 测试项 | 测试方法 |
|---|---|---|---|
| T15 | 文件加载 | 编码自动检测 | 加载 GBK 编码的 CSV |
| T16 | 文件加载 | 分隔符自动检测 | 加载 tab 分隔的 TSV |
| T17 | 绘图显示 | 自适应 Y 轴 | 右键 → Autoscale in x-Range |
| T18 | 交互 | 右键上下文菜单 | 右键绘图区域 |
| T19 | 交互 | 变量编辑器 | 双击绘图区 |
| T20 | 交互 | 清除图表 | 双击中键 |
| T21 | 光标功能 | 光标值显示/隐藏 | 右键 → Show/Hide Cursor Value |
| T22 | 光标功能 | 多曲线光标 | 多曲线模式下移动光标 |
| T23 | 光标功能 | MDF 枚举光标 | MDF 枚举通道的光标显示文本标签 |
| T24 | 标记功能 | 清除标记 | 删除已有标记 |
| T25 | 布局管理 | 行高调整 | 右键 → Adjust Height |
| T26 | 数据表 | 数据显示和排序 | 双击变量名打开表格 |
| T27 | 数据表 | XY 散点图 | 在表格中选中两列 → 散点图 |
| T28 | 性能 | 多曲线性能 | 同时显示 >20 条曲线 |

#### P2（一般 — 最好通过）

| 编号 | 功能模块 | 测试项 | 测试方法 |
|---|---|---|---|
| T29 | 对话框 | 轴设置对话框 | 双击坐标轴标签 |
| T30 | 对话框 | 帮助对话框 | 打开帮助 |
| T31 | 对话框 | 布局输入对话框 | 修改行列数对话框 |
| T32 | 对话框 | 时间校正对话框 | 打开时间校正 |
| T33 | 交互 | 多窗口 | 同时打开多个数据表 |
| T34 | 交互 | 键盘修饰符 | Ctrl/Shift + 滚轮缩放特定轴 |

### 10.3 回归测试重点

以下区域因涉及信号传递和 Qt 特定行为，需重点回归：

#### 信号传递链路

**`CustomViewBox` → `EventHandler` → `MainWindow`**：

1. **`request_jump_to_data`** — 右键"Jump to Data"
2. **`request_clear_plot`** — 右键"Clear Plot"
3. **`request_auto_y`** — 右键"Autoscale in x-Range"
4. **`request_set_cursor_mode`** — 右键光标模式切换
5. **`request_show_cursor_value`** / **`request_hide_cursor_value`** — 光标值显隐
6. **`request_set_row_height`** — 行高调整
7. **`request_set_all_row_height`** — 全部行高重置
8. **`request_copy_name`** — 复制变量名
9. **`request_variable_editor`** — 变量编辑器

**测试方法**：逐个右键菜单项测试，确认信号正确发射和接收。

#### QThread 数据加载

**`DataLoadThread` 的 3 个信号链**：
- `progress` → 进度条更新
- `finished` → 数据加载完成回调
- `error` → 错误处理和提示

**测试方法**：加载不同大小、不同编码、不同格式的文件，验证：
- 进度条正确更新
- 加载完成后变量列表正确填充
- 加载失败时错误信息正确显示

#### 调度器

**`UnifiedUpdateScheduler` 的 QTimer 定时刷新**：
- 验证定时器正常触发
- 验证批量更新逻辑（UI 防抖）
- 验证交互期间的暂停/恢复

**测试方法**：快速缩放/拖拽，确认 UI 正确响应且无抖动。

#### pyqtgraph 交互

- 鼠标滚轮缩放 → `sigRangeChanged` 信号
- 拖拽平移 → ViewBox 范围更新
- 标记区域拖拽 → `sigRegionChanged` 信号
- `SignalProxy` 鼠标信号代理

**测试方法**：在包含多个子图的布局中进行各种鼠标交互，确认所有子图正确响应。

### 10.4 自动化测试建议

```python
# test_basic_import.py — 基础导入测试
import sys
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # 无头模式

def test_pyside6_import():
    from PySide6.QtCore import Qt, Signal, QTimer, QThread
    from PySide6.QtGui import QColor, QPen
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    assert app is not None

def test_pyqtgraph_binding():
    import pyqtgraph as pg
    assert "PySide6" in pg.Qt.QT_LIB, f"期望 PySide6，实际 {pg.Qt.QT_LIB}"

def test_core_imports():
    from src.core.config import safe_callback, DEFAULT_LINE_WIDTH
    from src.core.types import CurveInfo, FormatInfo
    from src.core.scheduler import UnifiedUpdateScheduler
    assert safe_callback is not None
```

---

## 十一、最佳实践建议

### 11.1 迁移策略建议

1. **一次迁移，分批修改**：将迁移分为多个小步骤，每个步骤只做一类修改（如 import 替换、pyqtSignal → Signal、debug log 移除），方便问题定位。

2. **先 dry-run，后执行**：始终先用 `--dry-run` 参数运行自动化脚本，确认修改范围后再正式执行。

3. **保持 PyQt6 备份**：在 `.reference/` 目录中保留原始 PyQt6 代码作为对照，不要删除。如果迁移后遇到行为差异，可以快速对比原始代码。

### 11.2 代码质量建议

1. **统一 import 风格**：推荐使用 `from PySide6.QtCore import Qt` 而非 `from PySide6 import QtCore`，保持与原始代码风格一致。

2. **枚举使用全限定语法**：始终使用 `Qt.AlignmentFlag.AlignCenter` 而非 `Qt.AlignCenter`，确保跨绑定兼容性。

3. **Signal 签名声明**：在定义 `Signal` 时明确参数类型，如 `Signal(int, str)` 而非 `Signal(object)`，便于 IDE 类型检查和调试。

### 11.3 调试建议

1. **Qt 日志级别**：开发期间设置 `QT_LOGGING_RULES="*=true"` 以获取详细的 Qt 内部日志：
   ```python
   os.environ["QT_LOGGING_RULES"] = "*.debug=true"
   ```

2. **pyqtgraph 绑定诊断**：启动时打印绑定信息：
   ```python
   import pyqtgraph as pg
   print(f"Qt 绑定: {pg.Qt.QT_LIB}")
print(f"Qt 版本: {pg.Qt.VERSION_INFO}")
print(f"pyqtgraph 版本: {pg.__version__}")
   ```

3. **信号调试**：使用 `Signal.connect` 的回调中打印日志，确认信号链路正常工作。

### 11.4 性能建议

1. **PySide6 与 PyQt6 性能无差异**：两者底层使用相同的 Qt6 C++ 库，Python 绑定层的性能开销几乎一致。

2. **启动时间**：PySide6 的 import 时间与 PyQt6 在同一量级（均为 ~200-400ms）。

3. **内存占用**：PySide6 默认不使用 `sip` 模块（PyQt6 的绑定引擎），可能略微减少内存占用。

### 11.5 长期维护建议

1. **避免绑定特定代码**：后续开发中避免使用 PySide6 专属 API（如 `shiboken6`），保持代码在两种绑定下的可移植性。如需检测当前绑定：
   ```python
   import sys
   if "PySide6" in sys.modules:
       pass  # PySide6 specific
   ```

2. **考虑引入 QtPy**（可选）：如果未来需要支持多种 Qt 绑定，可考虑引入 `qtpy` 抽象层。但对于本项目当前需求，直接使用 PySide6 更为简单和高效。

3. **版本锁定**：在 `pyproject.toml` 中锁定 `PySide6` 的版本范围，避免大版本升级引入破坏性变更。

### 11.6 文件命名建议

迁移完成后，建议将主入口文件重命名，去除绑定名称后缀：

```bash
# 重命名主入口文件
mv csv_plot_pyqt6.py csv_plot.py
```

同步更新：
- 所有引用该文件名的打包脚本
- 文档中的安装/运行命令
- IDE 的运行配置

---

## 附录 A：文件修改快速参考

### A.1 一键执行全流程

```bash
# 1. 确认项目结构
ls -la

# 2. 更新依赖
# 编辑 pyproject.toml，替换 PyQt6 → PySide6
uv sync

# 3. 自动化 import 和 signal 替换
python scripts/migrate_pyqt6_to_pyside6.py --dry-run  # 先预览
python scripts/migrate_pyqt6_to_pyside6.py             # 正式执行

# 4. 验证 import 替换
python scripts/migrate_pyqt6_to_pyside6.py --verify

# 5. Debug log 移除（手动操作 — 参考第五节）

# 6. 编译验证
python -m py_compile csv_plot_pyqt6.py

# 7. 冒烟测试
python csv_plot_pyqt6.py

# 8. 确认 pyqtgraph 绑定
# 输出应包含: Qt 绑定: PySide6
```

### A.2 关键命令速查

| 操作 | 命令 |
|---|---|
| 检查是否还有 PyQt6 引用 | `grep -rn "PyQt6" src/ csv_plot_pyqt6.py` |
| 检查是否还有 pyqtSignal | `grep -rn "pyqtSignal" src/ csv_plot_pyqt6.py` |
| 检查是否还有 debug_log | `grep -rn "debug_log" src/ csv_plot_pyqt6.py` |
| 安装 PySide6 | `pip install "PySide6>=6.11.1"` |
| 卸载 PyQt6 | `pip uninstall pyqt6 pyqt6-qt6 pyqt6-sip -y` |
| PyInstaller 打包 | `pyinstaller csv_plot_pyqt6.py --onefile --hidden-import PySide6` |
| Nuitka 打包 | `nuitka --onefile --standalone --enable-plugin=pyside6 csv_plot_pyqt6.py` |

### A.3 手动回滚参考（如需要）

如果迁移后出现问题，可以从 `.reference/` 目录恢复原始文件：

```bash
# 恢复某个特定文件
cp .reference/csv_plot_pyqt6.py ./csv_plot_pyqt6.py

# 恢复整个 src/ 目录
cp -r .reference/src/ ./src/
```

---

## 附录 B：版本更新记录

| 版本 | 日期 | 更新内容 |
|---|---|---|
| v1.0 | 2026-05-20 | 初始版本，基于迁移分析报告 |
| v2.0 | 2026-05-20 | 新增：API 差异对照表、Debug Log 移除方案、常见问题解决方案、最佳实践建议、增强版自动化脚本、完整测试矩阵 |
| v2.1 | 2026-05-20 | 更新：适用于全新仓库，移除 Git 分支相关内容，简化回滚说明 |

---

> **文档维护者**：请在每次重大变更后更新本文档的版本号和更新日期。
