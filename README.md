# CSV Plot (PySide6)

一个基于 PySide6 + pyqtgraph 的高性能交互式数据可视化工具，支持 CSV、Excel 和 MDF（ASAM MDF4/DAT）格式文件，专为快速加载、浏览和分析时序数据而设计。支持百万级数据点的流畅绘制和多曲线模式。

![软件截图](docs/snapshot1.png)

## 主要特性

### 数据加载与显示
- **多格式支持**: CSV / Excel（.xlsx / .xlsm）、MDF（.mf4 / .mdf / .dat）及兼容格式
- **MDF 专属**: 枚举类型通道自动映射文本标签、多 Channel Group 聚合、跨 Group 同名变量智能后缀
- **拖拽加载**: 直接将文件拖拽到窗口中即可加载
- **智能解析**: CSV 自动检测编码、分隔符、标题行/单位行；MDF 自动识别枚举通道
- **大数据支持**: 后台线程加载，进度显示，支持百万级数据点
- **大 CSV 惰性加载（polars + parquet）**:
  - 后台加载时 ≥ 50 MB 的 CSV 自动转为临时列式文件（parquet），逐列按需读取 + LRU 缓存，常驻内存不随行数线性增长
  - 临时目录落在系统缓存区（`<CacheLocation>/CSVPlot/lazy_tmp`），退出/启动时按进程存活与 72h 陈旧度清扫
  - 转换失败自动回退内存加载并在状态栏提示，不弹窗、不中断分析
- **数据质量指示 (mdf不支持) **:
  - 🟢 绿色：变化的有效数值
  - 🟡 黄色：无变化的有效数值（常数）
  - 🔴 红色：非有效数值
- **快速浏览**: 双击变量名打开「变量数值表」查看完整数值

### 强大的图表功能
- **拖拽式绘图**: 从变量列表拖拽变量到绘图区域即可绘图；默认为“添加”操作，按住 Shift + 拖拽执行“替换”操作（先清空当前曲线再绘制）
- **绘图变量编辑器**: 可视化编辑曲线顺序、颜色、可见性
- **灵活布局**: 支持多子图网格排列（行列数可配置），可自由调整布局
- **同步缩放**: 所有图表 X 轴自动同步（XLink），支持鼠标交互缩放
- **高性能渲染**:
  - 智能降采样（peak 模式保留峰值特征）
  - 交互期间自适应降低样式更新频率
  - 支持百万级数据点流畅绘制
  - 曲线样式自动切换：缩放至细节时自动显示符号点
- **精确控制**:
  - 鼠标滚轮缩放 X 轴（以鼠标位置为中心）
  - 鼠标左键拖拽移动视图
  - 双击中键清除单个图表
  - Ctrl/Shift + 滚轮缩放特定轴
- **绘图区右键菜单**: 跳转至变量数值表、按 X 范围调节 Y 轴、游标模式（单自由 / 单固定 / 双固定 / 关闭）、显示或隐藏游标数值、复制变量名、绘图变量编辑器、调整高度、清除绘图；
- **坐标轴设置**: 双击坐标轴标签可设置范围/刻度

### 游标与交互功能
- **十字游标**: 实时显示当前鼠标位置的 X/Y 坐标值
- **多游标模式**: 单自由游标 / 单固定游标 / 双固定游标
- **多曲线游标**: 在多曲线模式下同时显示所有曲线在当前 X 位置的值
- **MDF 枚举游标**: 枚举通道自动显示文本标签而非原始整数值
- **同步游标**: 多个图表之间的游标位置自动同步
- **交互优化**: 缩放/拖动期间自动禁用游标更新

### 标记与统计分析
- **区间标记**: 在图表上标记感兴趣的数据区间
- **自动统计**: 实时计算标记区域的统计值（最小/最大/平均值/斜率等）
- **标记统计窗口**: 汇总显示所有图表的标记统计信息

### 变量数值表
- **快速访问**: 双击变量名或点击绘图区右键菜单的「跳转至变量数值表」打开
- **冻结列功能**: 支持列冻结，方便对比查看
- **智能高亮**: 选中单元格高亮 + 所在行/列浅蓝高亮
- **XY 散点图**: 选中两列数据可快速绘制散点图
- **同步滚动**: 冻结列与非冻结列保持同步

## 项目结构

```
csv_plot/
├── csv_plot.py                  # 主入口
├── pyproject.toml               # 项目配置与依赖管理
├── uv.lock                      # uv 依赖锁文件
├── README.md
├── .gitignore
├── assets/                      # 图标资源
│   ├── icon.png
│   ├── icon.ico
│   └── icon.icns
├── docs/                        # 文档
│   ├── help.md                  # 帮助文档（打包进产物，应用内 HelpDialog 渲染）
│   ├── snapshot1.png            # 截图
│   └── template.yaml            # YAML 模板示例
├── tests/                       # 测试体系（offscreen 无头，1315 例）
│   ├── README.md                # 测试体系唯一入口文档
│   ├── conftest.py              #   环境隔离 + marker 自动分层 + >1s 耗时护栏
│   ├── fixtures/                #   data_factory（合成数据工厂）/ waits（等待工具）
│   ├── unit/                    #   纯逻辑，无 Qt（755 例，含 parquet 等价性用例）
│   ├── component/               #   控件级（474 例）
│   ├── e2e/                     #   真实 MainWindow 冒烟（86 例）
│   └── perf/                    #   惰性链量级护栏（4 例，默认 deselected）
├── scripts/                     # 打包脚本
│   ├── build_exe_nuitka         # macOS/Linux Nuitka 编译脚本
│   ├── build_exe_pyinstaller    # macOS/Linux PyInstaller 打包脚本
│   ├── build_exe_pyinstaller.bat # Windows PyInstaller 打包批处理
│   ├── build_win.py             # Windows Nuitka 编译脚本
│   ├── generate_build_info.py   # 注入版本号与编译时间（src/_build_info.py）
│   └── csv_plot_pyinstaller.spec # PyInstaller spec 配置
└── src/                         # 模块化源码
    ├── __init__.py
    ├── app/
    │   └── plot_context.py      # PlotContext 服务层（依赖注入）
    ├── core/
    │   ├── config.py            # 全局常量、float32 安全检查、惰性转换阈值
    │   ├── data_types.py        # FormatInfo / CurveInfo / MarkStatEntry 数据类型
    │   ├── curve_strategy.py    # 曲线策略（单/多曲线模式切换）
    │   ├── scheduler.py         # UnifiedUpdateScheduler 防抖调度器
    │   ├── font_cache.py        # 字体缓存（基于 AppSettings）
    │   ├── logger.py            # Logger 日志管理器
    │   ├── crash_handler.py     # excepthook + faulthandler 崩溃兜底
    │   ├── gc_guard.py          # no_autogc()：worker 线程内屏蔽自动分代回收
    │   ├── settings.py          # AppSettings 统一配置管理器 + ConfigKey 枚举
    │   ├── plot_config.py       # PlotSessionConfig / PlotConfig 配置模型
    │   ├── template_models.py   # PlotTemplate / TemplateMetadata 模板数据模型
    │   ├── storage.py           # TemplateStorage 模板持久化存储
    │   ├── template_manager.py  # TemplateManager 模板 CRUD 管理
    │   └── auto_save_manager.py # AutoSaveManager 自动保存与恢复
    ├── data/
    │   ├── loader_factory.py    # create_loader() 单一分派点（含惰性转换开关）
    │   ├── loader.py            # FastDataLoader CSV 加载 + probe_csv_schema + DataLoadThread
    │   ├── base_loader.py       # BaseDataLoader 基础加载器父类（LOADER_TYPE / IS_LAZY）
    │   ├── loader_caps.py       # is_lazy_loader() 能力谓词（与格式谓词分离）
    │   ├── excel_loader.py      # ExcelDataLoader Excel 加载（calamine / openpyxl）
    │   ├── parquet_converter.py # CSV/Excel → 临时 parquet 转换（polars 显式 schema）
    │   ├── parquet_lazy_loader.py # ParquetLazyLoader 逐列惰性读取（meta.json 元数据）
    │   ├── temp_cache_dir.py    # TempCacheDir 临时目录创建 / 清扫 / atexit 兜底
    │   ├── _column_cache.py     # 列级 LRU 缓存（条数 + 字节双预算）
    │   ├── mdf_lazy_loader.py   # MDFLazyLoader MDF4/DAT 按需加载 + LRU 缓存
    │   ├── mdf_attribution.py   # MDF 变量归属信息推断（纯函数，零 Qt）
    │   ├── var_info.py          # 变量信息快照（元数据零 I/O，统计交后台线程）
    │   ├── file_info.py         # 状态栏「文件信息抽屉」数据源
    │   └── metadata.py          # VarMetadata 数据类 + 有效性分类工具
    ├── utils/
    │   ├── paths.py             # resource_path / display_path 路径解析
    │   └── platform_setup.py    # 平台初始化（字体/DPI）
    └── ui/
        ├── main_window.py       # MainWindow 主窗口
        ├── drag_drop.py         # 拖放解析
        ├── table_dialog.py      # DataTableDialog + PandasTableModel + XYScatterPlotDialog
        ├── variable_list.py     # MyTableWidget 变量列表面板
        ├── variable_actions.py  # 变量操作共享实现（列表右键 / 变量信息窗口共用）
        ├── mark_stats.py        # MarkStatsWindow 标记统计窗口
        ├── theme.py             # 界面色板与文字层级
        ├── plot_config_manager.py  # PlotConfigManager 配置协调
        ├── plot_variable_editor.py  # PlotVariableEditorDialog 变量编辑器
        ├── main_window_base_manager.py  # MainWindow 基础管理器
        ├── file_loader_manager.py  # 文件加载管理器（同步/异步分流、回退播报）
        ├── cursor_sync_manager.py  # 游标同步管理器
        ├── layout_manager.py    # 布局管理器
        ├── splash_screen.py     # SplashScreen 启动画面
        ├── dialogs/
        │   ├── help.py          # HelpDialog 帮助文档
        │   ├── layout_grid_selector.py  # LayoutGridSelector 网格布局选择（类 Word 插入表格）
        │   ├── axis.py          # AxisDialog 坐标轴设置
        │   ├── time_correction.py  # TimeCorrectionDialog 时间修正
        │   ├── sheet_selector.py  # SheetSelectorDialog Excel 工作表选择
        │   ├── log_window.py    # LogWindow 日志窗口
        │   ├── variable_info_dialog.py # VariableInfoDialog 变量信息窗口（标签页累积）
        │   ├── template_editor_dialog.py  # TemplateEditorDialog 模板编辑器
        │   └── template_manager_dialog.py # TemplateManagerDialog 模板管理器
        └── widgets/
            ├── __init__.py
            ├── base_manager.py  # 管理器基类
            ├── custom_viewbox.py # 信号化 CustomViewBox
            ├── plot_widget.py   # PlotWidget 主绘图组件
            ├── plot_container.py # PlotContainerWidget 绘图容器
            ├── plot_ui_manager.py # PlotUIManager 绘图 UI 管理器
            ├── plot_data_manager.py # PlotDataManager 绘图数据管理器
            ├── axis_manager.py  # AxisManager 坐标轴管理器
            ├── cursor_manager.py # CursorManager 游标管理器
            ├── multi_curve_manager.py # MultiCurveManager 多曲线管理器
            ├── mark_region_manager.py # MarkRegionManager 标记区域管理器
            ├── event_handler.py # EventHandler 事件处理器
            ├── log_viewer.py    # LogViewer 日志查看器
            ├── status_drawer.py # StatusDrawer 状态栏上翻抽屉（文件信息 / 时间基准）
            └── variable_search_bar.py # VariableSearchBar 绘图变量编辑器内嵌搜索栏
```

## 快速开始

### 系统要求
- Python 3.12 或更高版本
- 支持的操作系统：Windows、macOS、Linux
- pyqtgraph 建议 0.14.0 以上

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/Melon793/csv_plot.git
   cd csv_plot
   ```

2. **安装依赖**（使用 uv）
   ```bash
   uv sync
   ```

   或使用 pip：
   ```bash
   pip install pyside6 pyqtgraph pandas polars numpy asammdf charset-normalizer ujson openpyxl python-calamine pyyaml
   ```

3. **运行程序**
   ```bash
   uv run csv_plot.py
   ```

### 打包为独立应用

项目提供了 `scripts/` 目录下的打包脚本，可直接运行。

首先安装 dev 依赖组（含 pytest / nuitka / pyinstaller）：

```bash
uv sync --group dev
```

然后运行对应的打包脚本：

```bash
# PyInstaller（单目录模式，启动快）
bash scripts/build_exe_pyinstaller

# Nuitka（编译为原生可执行文件，性能更高）
bash scripts/build_exe_nuitka

# 打包为独立应用（Windows）
uv run scripts/build_win.py
```

## 测试

项目使用 pytest + pytest-qt + pytest-cov 构建 offscreen 无头测试体系，三层组织（unit 755 / component 474 / e2e 86，共 1315 例，全量串行 ~37s、`-n 4` 并行 ~13s），测试环境自动隔离，不会污染真实用户配置。parquet 惰性链另有逐位等价性用例与按需手跑的 perf 层。命令速查、分层设计、新用例编写指南与已知陷阱清单详见 [tests/README.md](tests/README.md)。

## 使用指南

见 [docs/help.md](docs/help.md)

### MDF 文件特别说明

| MDF 特性 | 支持方式 |
|----------|---------|
| 多 Channel Group | 自动聚合，同名变量加 `_G{index}` 后缀区分 |
| 枚举类型通道 | 自动构建 `{int→text_label}` 映射，绘图和游标显示文本标签 |
| 时间序列 | 自动提取时间通道作为 X 轴（`t` / `time` / `timestamp`） |
| asammdf 兼容 | 支持 7.x 和 8.x 两种 API |


## 技术栈

**界面与绘图**
- **PySide6** — 现代化图形界面框架（LGPL 许可）
- **pyqtgraph** — 高性能科学绘图库

**数据层（双引擎分工）**
- **pandas** — 数据分析库
- **polars** — 大 CSV 惰性加载链（Rust 引擎）
- **numpy** — 数值计算基础库，绘图与统计的实际数据载体
**格式解析与工具**
- **asammdf** — ASAM MDF 文件解析（7.x / 8.x 双 API）
- **openpyxl** — Excel 回退读取与工作表元数据（名称 / 行列数）
- **python-calamine** — Rust 引擎 Excel 解析（比 openpyxl 快 10-20 倍）
- **charset-normalizer** — 字符编码检测
- **ujson** — 高性能 JSON 处理
- **pyyaml** — YAML 配置解析

## 支持与反馈

如果您在使用过程中遇到问题或有改进建议，欢迎：

- 提交 Issue: [GitHub Issues](https://github.com/Melon793/csv_plot/issues)
- 参与讨论: [GitHub Discussions](https://github.com/Melon793/csv_plot/discussions)
- 给项目点赞: [GitHub Star](https://github.com/Melon793/csv_plot)

## 许可证

本项目采用开源许可证，供学习与研究使用。欢迎自由修改和扩展。

---

**如果这个项目对您有帮助，请给我们一个 Star！**
